"""Run only Snow seed42 purity-aware V2.1 probes."""

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
sys.path.extend([str(ROOT / "src"), str(ROOT / "scripts")])

from diagnostics.injection_schedule import build_processor_map, build_variant_schedule, processor_block_weights  # noqa: E402
from diagnostics.ip_adapter_instrumentation import ResidualLogger, instrument_ip_adapter_processors, set_spatial_gate  # noqa: E402
from regions.v2_1_purity import build_purity_routes, route_gate, route_masks, save_purity_overlay  # noqa: E402
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_injection_variants import _scalar_timestep, set_ip_adapter_schedule_step  # noqa: E402
from run_ip_adapter_plus_variants import fit_square_crop, load_pipe  # noqa: E402
from run_v2_0_rigid_only import aligned_content_path, frozen_control_path, read_cases, select_case  # noqa: E402
from run_v2_1_regional_pilot import CONTENT_NAMES, region_paths, save_mask  # noqa: E402
from regions.v2_1_masks import load_region_mask_set  # noqa: E402


PROBE_VARIANTS = ("S_sep_neutral", "S_sep_conservative")
RESOLUTIONS = (64, 32, 16, 8)


def save_route_snapshots(route: dict[tuple[int, int], torch.Tensor], out_dir: Path, label: str) -> None:
    snapshot_dir = out_dir / "effective_region_gates" / label
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    summary = {"label": label, "scales": {}}
    for resolution in RESOLUTIONS:
        gate = route[(resolution, resolution)].detach().cpu().numpy()
        Image.fromarray(np.clip(np.rint(gate * 255), 0, 255).astype(np.uint8), mode="L").save(
            snapshot_dir / f"gate_{resolution}x{resolution}.png"
        )
        summary["scales"][str(resolution)] = {
            "active_or_nonzero_tokens": int(np.count_nonzero(gate > 0.0)),
            "mean_gain": float(gate.mean()),
            "max_gain": float(gate.max()),
        }
    (snapshot_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def write_purity_outputs(content: Image.Image, routes: dict[int, object], out_dir: Path) -> None:
    overlay_dir = out_dir / "purity_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    report = {}
    for resolution, purity in routes.items():
        report[str(resolution)] = purity.report()
        save_purity_overlay(content, purity, overlay_dir / f"purity_{resolution}x{resolution}.png")
        for name, mask in route_masks(purity).items():
            save_mask(mask, overlay_dir / f"{name}_{resolution}x{resolution}.png")
    (out_dir / "purity_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


def run_probe(project_root: Path, config: dict, args: argparse.Namespace) -> None:
    rows = read_cases(project_root / config["experiment"]["manifest"])
    case = select_case(rows, "v1_5_kulhanek_snow_winter", 42)
    out_dir = project_root / args.output_root / case["canonical_case_id"] / "seed42"
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

    routes = build_purity_routes(
        masks.subject,
        masks.background,
        masks.valid_eval,
        resolutions=RESOLUTIONS,
        purity_threshold=args.purity_threshold,
        subject_gain=1.0,
        background_gain=0.0,
    )
    write_purity_outputs(content, routes, out_dir)

    route_gates: dict[str, dict[tuple[int, int], torch.Tensor]] = {}
    for variant in PROBE_VARIANTS:
        route_gates[variant] = {
            (resolution, resolution): torch.from_numpy(route_gate(routes[resolution], variant))
            for resolution in RESOLUTIONS
        }

    audit_regions = {}
    for name in ("pure_subject", "mixed", "pure_background", "valid"):
        audit_regions[name] = {
            (resolution, resolution): torch.from_numpy(route_masks(routes[resolution])[name])
            for resolution in RESOLUTIONS
        }

    pipe = load_pipe(project_root, args)
    residual_logger = ResidualLogger()
    instrumented = instrument_ip_adapter_processors(
        pipe,
        residual_logger,
        spatial_gate=route_gates[PROBE_VARIANTS[0]],
        enable_logging=True,
        spatial_gate_pooling="maximum",
        audit_region_masks=audit_regions,
    )
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=style,
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    processor_infos = build_processor_map(pipe.unet.attn_processors)
    (out_dir / "processor_map.json").write_text(
        json.dumps([info.to_dict() for info in processor_infos], indent=2), encoding="utf-8"
    )
    (out_dir / "instrumented_processors.json").write_text(json.dumps(instrumented, indent=2), encoding="utf-8")
    block_weights = processor_block_weights(processor_infos)

    index = []
    for label in PROBE_VARIANTS:
        gate = route_gates[label]
        set_spatial_gate(pipe, gate)
        save_route_snapshots(gate, out_dir, label)
        residual_logger.clear()
        schedule = build_variant_schedule(
            "A2_highres_only", args.ip_adapter_scale, args.num_inference_steps, block_weights
        )
        pipe.scheduler.set_timesteps(args.num_inference_steps, device="cuda")
        timesteps = list(pipe.scheduler.timesteps)
        set_ip_adapter_schedule_step(pipe, schedule, 0)
        residual_logger.set_step(0, _scalar_timestep(timesteps[0]))

        def callback(pipe_ref, step_index, timestep, callback_kwargs):
            next_step = min(step_index + 1, args.num_inference_steps - 1)
            set_ip_adapter_schedule_step(pipe_ref, schedule, next_step)
            residual_logger.set_step(next_step, _scalar_timestep(timesteps[next_step]))
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
            generator=torch.Generator(device="cuda").manual_seed(42),
            callback_on_step_end=callback,
        ).images[0]
        image.save(out_dir / f"{label}.png")
        with (out_dir / f"{label}_residuals.jsonl").open("w", encoding="utf-8") as handle:
            for record in residual_logger.to_dicts():
                handle.write(json.dumps(record) + "\n")
        index.append(
            {
                "label": label,
                "strategy": label,
                "purity_threshold": args.purity_threshold,
                "subject_gain": 1.0,
                "background_gain": 0.0,
                "mixed_gain": 1.0 if label == "S_sep_neutral" else 0.0,
                "schedule": schedule.to_dict(),
                "elapsed_sec": round(time.time() - start, 4),
            }
        )
        print(f"[OK] {label}")

    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "case": case,
                "experiment": "v2_1_purity_probe",
                "variants": list(PROBE_VARIANTS),
                "purity_threshold": args.purity_threshold,
                "pooling": "adaptive_avg_pool2d_occupancy_then_direct_token_route",
                "index": index,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_1_regional_pilot.yaml")
    parser.add_argument("--output-root", default="runs/ip_adapter_plus_injection/v2_1_purity_probe")
    parser.add_argument("--purity-threshold", type=float, default=0.8)
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
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    fixed = config["fixed_parameters"]
    args.num_inference_steps = args.num_inference_steps or int(fixed["num_inference_steps"])
    args.strength = args.strength or float(fixed["strength"])
    args.guidance_scale = args.guidance_scale or float(fixed["guidance_scale"])
    args.controlnet_scale = args.controlnet_scale or float(fixed["controlnet_scale"])
    args.ip_adapter_scale = args.ip_adapter_scale or float(fixed["ip_adapter_scale"])
    run_probe(ROOT, config, args)


if __name__ == "__main__":
    main()
