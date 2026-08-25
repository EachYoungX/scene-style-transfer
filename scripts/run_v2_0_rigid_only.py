"""Run the V2.0 Rigid-only high-resolution IP-Adapter experiment."""

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
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))
sys.path.append(str(ROOT / "scripts"))

from diagnostics.injection_schedule import (  # noqa: E402
    build_processor_map,
    build_variant_schedule,
    processor_block_weights,
)
from diagnostics.ip_adapter_instrumentation import (  # noqa: E402
    ResidualLogger,
    instrument_ip_adapter_processors,
    set_spatial_gate,
)
from metrics.mask_utils import load_binary_mask  # noqa: E402
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_variants import fit_square_crop, load_pipe  # noqa: E402
from run_ip_adapter_plus_injection_variants import (  # noqa: E402
    _scalar_timestep,
    set_ip_adapter_schedule_step,
)


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_case(rows: list[dict[str, str]], case_id: str, seed: int) -> dict[str, str]:
    matches = [row for row in rows if row["canonical_case_id"] == case_id and int(row["seed"]) == seed]
    if len(matches) != 1:
        raise ValueError(f"Expected one frozen row for {case_id}/seed{seed}, found {len(matches)}")
    return matches[0]


def load_gate(project_root: Path, case: dict[str, str], retain_ratio: float, args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, int | float]]:
    content_name = {
        "v1_5_demuth_church": "photo_church.png",
        "v1_5_demuth_wave": "photo_wave.png",
        "v1_5_kulhanek_snow_winter": "photo_snow_winter.png",
    }[case["canonical_case_id"]]
    rigid_path = project_root / args.rigid_masks / content_name
    valid_eval_path = project_root / args.valid_eval_masks / content_name
    rigid = load_binary_mask(rigid_path)
    valid_eval = load_binary_mask(valid_eval_path)
    # Outside the valid evaluation area the A2 residual is left unchanged. This
    # keeps the empty-rigid Wave sanity check numerically equivalent to Uniform.
    gate = np.ones((512, 512), dtype=np.float32)
    protected = valid_eval & rigid
    gate[protected] = float(retain_ratio)
    return torch.from_numpy(gate), {
        "retain_ratio": float(retain_ratio),
        "rigid_pixels": int(rigid.sum()),
        "valid_eval_pixels": int(valid_eval.sum()),
        "protected_pixels": int(protected.sum()),
    }


def load_effective_rigid_mask(project_root: Path, case: dict[str, str], args: argparse.Namespace) -> torch.Tensor:
    content_name = {
        "v1_5_demuth_church": "photo_church.png",
        "v1_5_demuth_wave": "photo_wave.png",
        "v1_5_kulhanek_snow_winter": "photo_snow_winter.png",
    }[case["canonical_case_id"]]
    rigid = load_binary_mask(project_root / args.rigid_masks / content_name)
    valid_eval = load_binary_mask(project_root / args.valid_eval_masks / content_name)
    return torch.from_numpy((rigid & valid_eval).astype(np.float32))


def parse_roi(value: str | None) -> tuple[int, int, int, int] | None:
    if value is None:
        return None
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--audit-roi must be x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0 <= x0 < x1 <= 512 and 0 <= y0 < y1 <= 512):
        raise ValueError("--audit-roi must be within 512x512 and use x0<x1, y0<y1")
    return x0, y0, x1, y1


def aligned_content_path(project_root: Path, case: dict[str, str]) -> Path:
    """Use the frozen 512px source when the original private photo is absent."""
    names = {
        "v1_5_demuth_church": "photo_church.png",
        "v1_5_demuth_wave": "photo_wave.png",
        "v1_5_kulhanek_snow_winter": "photo_snow_winter.png",
    }
    aligned_name = names.get(case["canonical_case_id"])
    if aligned_name is None:
        return project_root / case["content_path"]
    aligned = project_root / "data/derived/v2_0_geometry_risk/annotation_sources/content" / aligned_name
    raw = project_root / case["content_path"]
    return raw if raw.exists() else aligned


def frozen_control_path(project_root: Path, case: dict[str, str]) -> Path | None:
    output_path = case.get("output_path", "").strip()
    if not output_path:
        return None
    path = project_root / Path(output_path).parent / "canny.png"
    return path if path.exists() else None


def save_gate_snapshots(gate: torch.Tensor, out_dir: Path, label: str) -> dict[str, object]:
    """Save diagnostic runtime gates without changing the final annotation masks."""
    snapshot_dir = out_dir / "gate_snapshots" / label
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    base = gate.detach().float().cpu()[None, None]
    summary: dict[str, object] = {"label": label, "source_size": [512, 512], "scales": {}}
    for size in (512, 64, 32, 16, 8):
        pooled = -torch.nn.functional.adaptive_max_pool2d(-base, (size, size))[0, 0].numpy()
        Image.fromarray(np.clip(np.rint(pooled * 255), 0, 255).astype(np.uint8), mode="L").save(
            snapshot_dir / f"gate_{size}x{size}.png"
        )
        summary["scales"][str(size)] = {
            "suppressed_tokens": int(np.count_nonzero(pooled < 0.999)),
            "mean_gate": float(pooled.mean()),
        }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def run_case(
    case: dict[str, str],
    project_root: Path,
    args: argparse.Namespace,
    ratios: list[tuple[str, float]],
) -> None:
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

    pipe = load_pipe(project_root, args)
    residual_logger = ResidualLogger()
    gate, gate_stats = load_gate(project_root, case, 1.0, args)
    effective_rigid_mask = load_effective_rigid_mask(project_root, case, args) if args.audit_roi else None
    instrumented = instrument_ip_adapter_processors(
        pipe,
        residual_logger,
        spatial_gate=gate,
        enable_logging=args.log_residuals,
        audit_rigid_mask=effective_rigid_mask,
        audit_roi=args.audit_roi,
        audit_outer_ring_px=args.audit_outer_ring_px,
        spatial_gate_only_resolution=(args.only_resolution, args.only_resolution)
        if args.only_resolution
        else None,
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
    if args.audit_roi:
        (out_dir / "audit_config.json").write_text(
            json.dumps(
                {
                    "roi_name": "center_building",
                    "roi": list(args.audit_roi),
                    "outer_ring_px": args.audit_outer_ring_px,
                    "effective_rigid_mask": str(
                        (project_root / args.rigid_masks).relative_to(project_root)
                    ),
                    "valid_eval_mask": str(
                        (project_root / args.valid_eval_masks).relative_to(project_root)
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    index: list[dict[str, object]] = []
    for label, ratio in ratios:
        gate, gate_stats = load_gate(project_root, case, ratio, args)
        set_spatial_gate(pipe, gate)
        gate_stats["snapshots"] = save_gate_snapshots(gate, out_dir, label)
        if args.log_residuals:
            residual_logger.clear()
        schedule = build_variant_schedule(
            "A2_highres_only",
            args.ip_adapter_scale,
            args.num_inference_steps,
            block_weights,
        )
        pipe.scheduler.set_timesteps(args.num_inference_steps, device="cuda")
        scheduler_timesteps = list(pipe.scheduler.timesteps)
        set_ip_adapter_schedule_step(pipe, schedule, 0)
        if args.log_residuals:
            residual_logger.set_step(0, _scalar_timestep(scheduler_timesteps[0]))

        def callback(pipe_ref, step_index, timestep, callback_kwargs):
            next_step = min(step_index + 1, args.num_inference_steps - 1)
            set_ip_adapter_schedule_step(pipe_ref, schedule, next_step)
            if args.log_residuals:
                residual_logger.set_step(next_step, _scalar_timestep(scheduler_timesteps[next_step]))
            return callback_kwargs

        start = time.time()
        image = pipe(
            prompt=args.prompt_override or case["prompt"],
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
        image.save(out_dir / f"{label}.png")
        if args.log_residuals:
            with (out_dir / f"{label}_residuals.jsonl").open("w", encoding="utf-8") as handle:
                for record in residual_logger.to_dicts():
                    handle.write(json.dumps(record) + "\n")
        index.append(
            {
                "label": label,
                "retain_ratio": ratio,
                "schedule": schedule.to_dict(),
                "gate": gate_stats,
                "elapsed_sec": round(time.time() - start, 4),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            }
        )
        print(f"[OK] {case['canonical_case_id']} seed{case['seed']} {label}")

    metadata = {
        "case": case,
        "experiment": "v2_0_rigid_only",
        "spatial_interpolation": "minimum_pool",
        "spatial_gate_downsampling": "minimum_pool_preserve_thin_rigid",
        "control_source": str(control_path.relative_to(project_root)) if control_path else "generated_from_aligned_content",
        "gate_scope": "IP-Adapter image residual only",
        "spatial_gate_only_resolution": args.only_resolution,
        "index": index,
    }
    (out_dir / "index.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_rigid_only.yaml")
    parser.add_argument("--mode", choices=("sweep", "multiseed", "sanity"), default="sweep")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--seed", action="append", type=int)
    parser.add_argument("--retain-ratio", type=float)
    parser.add_argument("--run-root")
    parser.add_argument("--num-inference-steps", type=int)
    parser.add_argument("--strength", type=float)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--controlnet-scale", type=float)
    parser.add_argument("--ip-adapter-scale", type=float)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--prompt-override")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter_plus")
    parser.add_argument("--weight-name", default="ip-adapter-plus_sd15.safetensors")
    parser.add_argument("--rigid-masks", default="data/derived/v2_0_geometry_risk/annotations/rigid_structure")
    parser.add_argument("--valid-eval-masks", default="data/derived/v2_0_geometry_risk/valid_masks/valid_eval")
    parser.add_argument("--log-residuals", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--audit-roi")
    parser.add_argument("--audit-outer-ring-px", type=int, default=12)
    parser.add_argument("--only-resolution", type=int, choices=(8, 16, 32, 64, 128, 256, 512))
    args = parser.parse_args()

    args.audit_roi = parse_roi(args.audit_roi)

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    fixed = config["fixed_parameters"]
    args.num_inference_steps = args.num_inference_steps or int(fixed["num_inference_steps"])
    args.strength = args.strength or float(fixed["strength"])
    args.guidance_scale = args.guidance_scale or float(fixed["guidance_scale"])
    args.controlnet_scale = args.controlnet_scale or float(fixed["controlnet_scale"])
    args.ip_adapter_scale = args.ip_adapter_scale or float(fixed["ip_adapter_scale"])
    args.run_root = args.run_root or config["experiment"]["output_root"] + f"_{args.mode}"

    rows = read_cases(ROOT / config["experiment"]["manifest"])
    if args.mode == "sweep":
        case_ids = args.case_id or config["formal_cases"]
        seeds = args.seed or config["seeds"]["sweep"]
        ratios = [("uniform", 1.0)] + [
            (f"rigid_suppress_{value:.2f}".replace(".", "p"), float(value))
            for value in config["suppression_sweep"]
        ]
    elif args.mode == "multiseed":
        case_ids = args.case_id or config["formal_cases"]
        seeds = args.seed or config["seeds"]["multiseed"]
        if args.retain_ratio is None:
            raise ValueError("--retain-ratio is required for multiseed mode")
        ratios = [(f"rigid_suppress_{args.retain_ratio:.2f}".replace(".", "p"), args.retain_ratio)]
    else:
        case_ids = args.case_id or [config["sanity_case"]]
        seeds = args.seed or [42]
        ratios = [("uniform", 1.0), ("rigid_suppress_0p00", 0.0)]

    for case_id in case_ids:
        for seed in seeds:
            run_case(select_case(rows, case_id, seed), ROOT, args, ratios)


if __name__ == "__main__":
    main()
