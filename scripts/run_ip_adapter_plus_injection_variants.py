"""Run layer/time IP-Adapter Plus injection diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from diagnostics.injection_schedule import (  # noqa: E402
    build_processor_map,
    build_variant_schedule,
    legacy_base_layer_scales,
    legacy_layer_group,
    legacy_time_multiplier,
    parse_processor_name,
    processor_block_weights,
    SUPPORTED_VARIANTS,
)
from diagnostics.ip_adapter_instrumentation import (  # noqa: E402
    ResidualLogger,
    instrument_ip_adapter_processors,
)
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_variants import fit_square_crop, load_pipe  # noqa: E402


VARIANTS = SUPPORTED_VARIANTS


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def layer_group(processor_name: str) -> str:
    return legacy_layer_group(processor_name)


def base_layer_scales(variant: str, base_scale: float) -> dict[str, float]:
    return legacy_base_layer_scales(variant, base_scale)


def time_multiplier(variant: str, step_index: int, num_steps: int) -> float:
    return legacy_time_multiplier(variant, step_index, num_steps)


def set_ip_adapter_scales(pipe, layer_scales: dict[str, float], multiplier: float = 1.0) -> dict[str, float]:
    applied: dict[str, float] = {}
    for name, processor in pipe.unet.attn_processors.items():
        if not hasattr(processor, "scale"):
            continue
        group = layer_group(name)
        scale = layer_scales.get(group, 0.0) * multiplier
        processor.scale = [scale for _ in processor.scale] if isinstance(processor.scale, list) else scale
        applied[name] = scale
    return applied


def set_ip_adapter_schedule_step(pipe, schedule, step_index: int) -> dict[str, float]:
    applied: dict[str, float] = {}
    for name, processor in pipe.unet.attn_processors.items():
        if not hasattr(processor, "scale"):
            continue
        info = parse_processor_name(name, processor)
        scale = schedule.scale_for(info.block, step_index) if info.block != "other" else 0.0
        processor.scale = [scale for _ in processor.scale] if isinstance(processor.scale, list) else scale
        applied[name] = scale
    return applied


def label_panel(image: Image.Image, label: str, height: int = 384) -> Image.Image:
    width = int(image.width * height / image.height)
    canvas = image.resize((width, height)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 6
    draw.rectangle((0, 0, bbox[2] - bbox[0] + pad * 2, bbox[3] - bbox[1] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(out_dir: Path, variants: list[str]) -> None:
    panels = [
        ("content", Image.open(out_dir / "content.png").convert("RGB")),
        ("style", Image.open(out_dir / "style.png").convert("RGB")),
    ]
    for variant in variants:
        panels.append((variant, Image.open(out_dir / f"{variant}.png").convert("RGB")))

    images = [(label, label_panel(image, label)) for label, image in panels]
    gutter = 8
    height = images[0][1].height
    grid = Image.new("RGB", (sum(image.width for _, image in images) + gutter * (len(images) - 1), height), (32, 32, 32))
    x = 0
    for _, image in images:
        grid.paste(image, (x, 0))
        x += image.width + gutter
    grid.save(out_dir / "injection_grid.png")


def run_case(case: dict[str, str], project_root: Path, args: argparse.Namespace) -> None:
    out_dir = project_root / "runs" / "ip_adapter_plus_injection" / args.run_name / case["case_id"]
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pipe = load_pipe(project_root, args)
    residual_logger = ResidualLogger() if args.log_residuals else None
    instrumented_processors = []
    if residual_logger is not None:
        instrumented_processors = instrument_ip_adapter_processors(pipe, residual_logger)
    content = fit_square(project_root / case["content"], args.size)
    style = fit_square_crop(project_root / case["style"], args.size)
    control = make_canny(content)
    content.save(out_dir / "content.png")
    style.save(out_dir / "style.png")
    control.save(out_dir / "canny.png")

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
    (out_dir / "block_weights.json").write_text(json.dumps(block_weights, indent=2), encoding="utf-8")
    if instrumented_processors:
        (out_dir / "instrumented_processors.json").write_text(
            json.dumps(instrumented_processors, indent=2),
            encoding="utf-8",
        )

    index = []
    for variant in args.variants:
        if residual_logger is not None:
            residual_logger.clear()
        schedule = build_variant_schedule(
            variant,
            args.ip_adapter_scale,
            args.num_inference_steps,
            block_weights,
            residual_energy_scale_factor=args.residual_energy_scale_factor,
        )
        # Diffusers 0.35.2 exposes step-end callbacks only. Set the scheduler
        # timetable up front so the first residual and every following step
        # receive the timestep used by the processor that produced them.
        pipe.scheduler.set_timesteps(args.num_inference_steps, device="cuda")
        scheduler_timesteps = list(pipe.scheduler.timesteps)
        set_ip_adapter_schedule_step(pipe, schedule, 0)
        if residual_logger is not None:
            residual_logger.set_step(0, _scalar_timestep(scheduler_timesteps[0]))

        def callback(pipe_ref, step_index, timestep, callback_kwargs):
            next_step = min(step_index + 1, args.num_inference_steps - 1)
            next_timestep = scheduler_timesteps[next_step]
            if residual_logger is not None:
                residual_logger.set_step(next_step, _scalar_timestep(next_timestep))
            set_ip_adapter_schedule_step(pipe_ref, schedule, next_step)
            return callback_kwargs

        torch.cuda.reset_peak_memory_stats()
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
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
            callback_on_step_end=callback,
        ).images[0]
        image.save(out_dir / f"{variant}.png")
        if residual_logger is not None:
            residual_path = out_dir / f"{variant}_residuals.jsonl"
            with residual_path.open("w", encoding="utf-8") as f:
                for record in residual_logger.to_dicts():
                    f.write(json.dumps(record) + "\n")
        index.append(
            {
                "variant": variant,
                "layer_scales": _legacy_layer_scales_or_empty(variant, args.ip_adapter_scale),
                "schedule": schedule.to_dict(),
                "scale_area": schedule.scale_area(block_weights),
                "residual_records": len(residual_logger.records) if residual_logger is not None else 0,
                "elapsed_sec": round(time.time() - start, 4),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
            }
        )
        print(f"[OK] {case['case_id']} / {variant}")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    make_grid(out_dir, list(args.variants))
    print(out_dir / "injection_grid.png")


def _scalar_timestep(timestep):
    if hasattr(timestep, "item"):
        return timestep.item()
    return timestep


def _legacy_layer_scales_or_empty(variant: str, base_scale: float) -> dict[str, float]:
    try:
        return base_layer_scales(variant, base_scale)
    except ValueError:
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/clean_artvee_baseline_pairs.csv")
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--run-name", default="v1_layer_time_injection_12step")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--strength", type=float, default=0.76)
    parser.add_argument("--guidance-scale", type=float, default=5.8)
    parser.add_argument("--controlnet-scale", type=float, default=0.72)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.9)
    parser.add_argument("--residual-energy-scale-factor", type=float)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter_plus")
    parser.add_argument("--weight-name", default="ip-adapter-plus_sd15.safetensors")
    parser.add_argument("--log-residuals", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    selected = set(args.case_id)
    for case in read_cases(project_root / args.manifest):
        if case["case_id"] in selected:
            run_case(case, project_root, args)


if __name__ == "__main__":
    main()
