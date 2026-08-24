"""Run the V2.1 U / Subject / Background regional style pilot."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from diagnostics.injection_schedule import build_processor_map, build_variant_schedule, processor_block_weights  # noqa: E402
from diagnostics.ip_adapter_instrumentation import (  # noqa: E402
    ResidualLogger,
    instrument_ip_adapter_processors,
    set_spatial_gate,
)
from metrics.mask_utils import load_binary_mask  # noqa: E402
from regions.v2_1_masks import load_region_mask_set  # noqa: E402
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_injection_variants import _scalar_timestep, set_ip_adapter_schedule_step  # noqa: E402
from run_ip_adapter_plus_variants import fit_square_crop, load_pipe  # noqa: E402
from run_v2_0_rigid_only import aligned_content_path, frozen_control_path, read_cases, select_case  # noqa: E402


CONTENT_NAMES = {
    "v1_5_demuth_church": "photo_church.png",
    "v1_5_kulhanek_snow_winter": "photo_snow_winter.png",
    "v1_5_demuth_wave": "photo_wave.png",
}


def save_mask(array: np.ndarray, path: Path) -> None:
    Image.fromarray(array.astype(np.uint8) * 255, mode="L").save(path)


def save_region_snapshots(gate: torch.Tensor, out_dir: Path, label: str) -> dict[str, object]:
    snapshot_dir = out_dir / "effective_region_gates" / label
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    base = gate.detach().float().cpu()[None, None]
    summary: dict[str, object] = {"label": label, "source_size": [512, 512], "scales": {}}
    for size in (512, 64, 32, 16, 8):
        pooled = F.adaptive_max_pool2d(base, (size, size))[0, 0].numpy()
        Image.fromarray(np.clip(np.rint(pooled * 255), 0, 255).astype(np.uint8), mode="L").save(
            snapshot_dir / f"gate_{size}x{size}.png"
        )
        summary["scales"][str(size)] = {
            "active_tokens": int(np.count_nonzero(pooled > 0.0)),
            "active_fraction": float((pooled > 0.0).mean()),
        }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def region_paths(project_root: Path, case_id: str, config: dict) -> tuple[Path, Path, Path, Path, Path]:
    name = CONTENT_NAMES[case_id]
    stem = Path(name).stem
    root = project_root / "data/derived/v2_0_geometry_risk"
    source_root = root / "annotations/soft_stylization"
    return (
        source_root / f"{stem}_S.png",
        source_root / f"{stem}_B.png",
        root / "annotations/rigid_structure" / name,
        root / "valid_masks/valid_content" / name,
        root / "valid_masks/valid_eval" / name,
    )


def run_case(case: dict[str, str], project_root: Path, args: argparse.Namespace, config: dict) -> None:
    out_dir = project_root / args.run_root / case["canonical_case_id"] / f"seed{case['seed']}"
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    content = fit_square(aligned_content_path(project_root, case), args.size)
    style = fit_square_crop(project_root / case["style_path"], args.size)
    control_path = frozen_control_path(project_root, case)
    control = Image.open(control_path).convert("RGB") if control_path else make_canny(content)
    content.save(out_dir / "content.png")
    style.save(out_dir / "style.png")
    control.save(out_dir / "canny.png")

    subject_path, background_path, rigid_path, valid_content_path, valid_eval_path = region_paths(
        project_root, case["canonical_case_id"], config
    )
    masks = load_region_mask_set(
        subject_path,
        background_path,
        rigid_path,
        valid_content_path,
        valid_eval_path,
        threshold=int(config["region_masks"]["threshold"]),
    )
    mask_dir = out_dir / "masks"
    mask_dir.mkdir()
    for name, value in {
        "subject": masks.subject,
        "background": masks.background,
        "neutral": masks.neutral,
        "rigid_excluded": masks.rigid & masks.valid_eval,
        "valid_eval": masks.valid_eval,
    }.items():
        save_mask(value, mask_dir / f"{name}.png")
    (mask_dir / "mask_report.json").write_text(json.dumps(masks.report(), indent=2), encoding="utf-8")

    pipe = load_pipe(project_root, args)
    residual_logger = ResidualLogger()
    uniform_gate = torch.ones((512, 512), dtype=torch.float32)
    instrumented = instrument_ip_adapter_processors(
        pipe,
        residual_logger,
        spatial_gate=uniform_gate,
        enable_logging=True,
        spatial_gate_pooling="maximum",
        audit_region_masks={
            "subject": torch.from_numpy(masks.subject.astype(np.float32)),
            "background": torch.from_numpy(masks.background.astype(np.float32)),
        },
    )
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=style,
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    processor_map = [info.to_dict() for info in build_processor_map(pipe.unet.attn_processors)]
    block_weights = processor_block_weights(build_processor_map(pipe.unet.attn_processors))
    (out_dir / "processor_map.json").write_text(json.dumps(processor_map, indent=2), encoding="utf-8")
    (out_dir / "instrumented_processors.json").write_text(json.dumps(instrumented, indent=2), encoding="utf-8")

    variants = [
        ("U", uniform_gate),
        ("S_subject", torch.from_numpy(masks.subject.astype(np.float32))),
        ("S_background", torch.from_numpy(masks.background.astype(np.float32))),
    ]
    if args.include_s_match:
        if args.subject_gain <= 0:
            raise ValueError("--subject-gain must be positive when --include-s-match is used.")
        variants.append(
            ("S_match", torch.from_numpy(masks.subject.astype(np.float32)) * float(args.subject_gain))
        )
    index = []
    for label, gate in variants:
        set_spatial_gate(pipe, gate)
        effective_snapshot = save_region_snapshots(gate, out_dir, label)
        residual_logger.clear()
        schedule = build_variant_schedule(
            "A2_highres_only", args.ip_adapter_scale, args.num_inference_steps, block_weights
        )
        pipe.scheduler.set_timesteps(args.num_inference_steps, device="cuda")
        scheduler_timesteps = list(pipe.scheduler.timesteps)
        set_ip_adapter_schedule_step(pipe, schedule, 0)
        residual_logger.set_step(0, _scalar_timestep(scheduler_timesteps[0]))

        def callback(pipe_ref, step_index, timestep, callback_kwargs):
            next_step = min(step_index + 1, args.num_inference_steps - 1)
            set_ip_adapter_schedule_step(pipe_ref, schedule, next_step)
            residual_logger.set_step(next_step, _scalar_timestep(scheduler_timesteps[next_step]))
            return callback_kwargs

        start = time.time()
        image = pipe(
            prompt=case["prompt"],
            negative_prompt=args.negative_prompt,
            image=content,
            control_image=control,
            ip_adapter_image_embeds=embeds,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(int(case["seed"])),
            callback_on_step_end=callback,
        ).images[0]
        if not args.audit_only:
            image.save(out_dir / f"{label}.png")
        with (out_dir / f"{label}_residuals.jsonl").open("w", encoding="utf-8") as handle:
            for record in residual_logger.to_dicts():
                handle.write(json.dumps(record) + "\n")
        index.append(
            {
                "label": label,
                "mask": "uniform" if label == "U" else label,
                "mask_report": masks.report(),
                "effective_gate": effective_snapshot,
                "schedule": schedule.to_dict(),
                "audit_only": bool(args.audit_only),
                "elapsed_sec": round(time.time() - start, 4),
            }
        )
        print(f"[OK] {case['canonical_case_id']} seed{case['seed']} {label}")

    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "case": case,
                "experiment": "v2_1_regional_style_pilot",
                "variants": [label for label, _ in variants],
                "subject_gain": float(args.subject_gain) if args.include_s_match else None,
                "mask_pooling": "adaptive_max_pool2d",
                "rigid_exclusion": True,
                "valid_eval_exclusion": True,
                "index": index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_1_regional_pilot.yaml")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--include-sanity", action="store_true")
    parser.add_argument("--run-root")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--controlnet-scale", type=float)
    parser.add_argument("--ip-adapter-scale", type=float)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter_plus")
    parser.add_argument("--weight-name", default="ip-adapter-plus_sd15.safetensors")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--include-s-match", action="store_true")
    parser.add_argument("--subject-gain", type=float, default=1.0)
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    fixed = config["fixed_parameters"]
    args.run_root = args.run_root or config["experiment"]["output_root"]
    args.seed = args.seed or int(config["pilot_seed"])
    args.num_inference_steps = args.num_inference_steps or int(fixed["num_inference_steps"])
    args.strength = args.strength or float(fixed["strength"])
    args.guidance_scale = args.guidance_scale or float(fixed["guidance_scale"])
    args.controlnet_scale = args.controlnet_scale or float(fixed["controlnet_scale"])
    args.ip_adapter_scale = args.ip_adapter_scale or float(fixed["ip_adapter_scale"])
    rows = read_cases(ROOT / config["experiment"]["manifest"])
    case_ids = args.case_id or list(config["formal_cases"])
    if args.include_sanity:
        case_ids += list(config["sanity_cases"])
    for case_id in case_ids:
        run_case(select_case(rows, case_id, args.seed), ROOT, args, config)


if __name__ == "__main__":
    main()
