"""Run V2.2a global A2 reference-strength frontier for seed42."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.extend([str(ROOT / "src"), str(ROOT / "scripts")])

from diagnostics.injection_schedule import build_processor_map, build_variant_schedule, processor_block_weights  # noqa: E402
from diagnostics.ip_adapter_instrumentation import ResidualLogger, instrument_ip_adapter_processors, set_spatial_gate  # noqa: E402
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_injection_variants import _scalar_timestep, set_ip_adapter_schedule_step  # noqa: E402
from run_ip_adapter_plus_variants import fit_square_crop, load_pipe  # noqa: E402
from run_v2_0_rigid_only import aligned_content_path, frozen_control_path, read_cases, select_case  # noqa: E402


CASE_IDS = (
    "v1_5_demuth_church",
    "v1_5_kulhanek_snow_winter",
    "v1_5_demuth_wave",
)
LAMBDAS = (0.2, 0.4, 0.6, 0.8)


def lambda_label(value: float) -> str:
    return f"lambda_{value:.1f}".replace(".", "p")


def run_one(
    pipe,
    case: dict[str, str],
    content: Image.Image,
    style: Image.Image,
    control: Image.Image,
    embeds,
    residual_logger: ResidualLogger,
    block_weights: dict[str, float],
    args: argparse.Namespace,
    multiplier: float,
    seed: int,
) -> tuple[Image.Image, list[dict], dict, float]:
    set_spatial_gate(pipe, torch.ones((512, 512), dtype=torch.float32) * float(multiplier))
    residual_logger.clear()
    schedule = build_variant_schedule("A2_highres_only", args.ip_adapter_scale, args.num_inference_steps, block_weights)
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
        generator=torch.Generator(device="cuda").manual_seed(seed),
        callback_on_step_end=callback,
    ).images[0]
    return image, residual_logger.to_dicts(), schedule.to_dict(), time.time() - start


def run_case(project_root: Path, case: dict[str, str], config: dict, args: argparse.Namespace, multipliers: tuple[float, ...]) -> None:
    case_root = project_root / args.output_root / case["canonical_case_id"] / f"seed{args.seed}"
    if case_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{case_root} exists. Use --overwrite.")
        shutil.rmtree(case_root)
    case_root.mkdir(parents=True)

    content = fit_square(aligned_content_path(project_root, case), args.size)
    style = fit_square_crop(project_root / case["style_path"], args.size)
    control_path = frozen_control_path(project_root, case)
    control = Image.open(control_path).convert("RGB") if control_path else make_canny(content)
    content.save(case_root / "content.png")
    style.save(case_root / "style.png")
    control.save(case_root / "canny.png")

    pipe = load_pipe(project_root, args)
    residual_logger = ResidualLogger()
    instrumented = instrument_ip_adapter_processors(
        pipe,
        residual_logger,
        spatial_gate=torch.ones((512, 512), dtype=torch.float32),
        enable_logging=True,
        spatial_gate_pooling="maximum",
    )
    embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=style,
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    processor_infos = build_processor_map(pipe.unet.attn_processors)
    block_weights = processor_block_weights(processor_infos)
    (case_root / "processor_map.json").write_text(
        json.dumps([info.to_dict() for info in processor_infos], indent=2), encoding="utf-8"
    )
    (case_root / "instrumented_processors.json").write_text(json.dumps(instrumented, indent=2), encoding="utf-8")

    index = []
    for multiplier in multipliers:
        label = lambda_label(multiplier)
        out_dir = case_root / label
        out_dir.mkdir()
        image, records, schedule, elapsed = run_one(
            pipe,
            case,
            content,
            style,
            control,
            embeds,
            residual_logger,
            block_weights,
            args,
            multiplier,
            args.seed,
        )
        image.save(out_dir / "output.png")
        with (out_dir / "residuals.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        (out_dir / "index.json").write_text(
            json.dumps(
                {
                    "case": case,
                    "seed": seed,
                    "lambda": multiplier,
                    "schedule": schedule,
                    "elapsed_sec": round(elapsed, 4),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        index.append({"lambda": multiplier, "label": label, "elapsed_sec": round(elapsed, 4)})
        print(f"[OK] {case['canonical_case_id']} seed{args.seed} lambda={multiplier:.1f}")
    (case_root / "index.json").write_text(json.dumps({"case": case, "variants": index}, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_1_regional_pilot.yaml")
    parser.add_argument("--manifest")
    parser.add_argument("--output-root", default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier")
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
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda", dest="lambda_values", type=float, action="append")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    fixed = config["fixed_parameters"]
    args.num_inference_steps = args.num_inference_steps or int(fixed["num_inference_steps"])
    args.strength = args.strength or float(fixed["strength"])
    args.guidance_scale = args.guidance_scale or float(fixed["guidance_scale"])
    args.controlnet_scale = args.controlnet_scale or float(fixed["controlnet_scale"])
    args.ip_adapter_scale = args.ip_adapter_scale or float(fixed["ip_adapter_scale"])
    cases = read_cases(ROOT / (args.manifest or config["experiment"]["manifest"]))
    case_ids = args.case_id or list(CASE_IDS)
    multipliers = tuple(args.lambda_values or LAMBDAS)
    if not multipliers:
        raise ValueError("At least one --lambda is required.")
    for case_id in case_ids:
        run_case(ROOT, select_case(cases, case_id, args.seed), config, args, multipliers)


if __name__ == "__main__":
    main()
