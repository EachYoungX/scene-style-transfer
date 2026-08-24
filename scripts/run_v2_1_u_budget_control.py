"""Run seed42 U_budget controls matched to the existing S_raw residual budget."""

from __future__ import annotations

import argparse
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
from metrics.mask_utils import load_binary_mask  # noqa: E402
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_injection_variants import _scalar_timestep, set_ip_adapter_schedule_step  # noqa: E402
from run_ip_adapter_plus_variants import fit_square_crop, load_pipe  # noqa: E402
from run_v2_0_rigid_only import aligned_content_path, frozen_control_path, read_cases, select_case  # noqa: E402
from run_v2_1_regional_pilot import CONTENT_NAMES, region_paths, save_mask  # noqa: E402
from regions.v2_1_masks import load_region_mask_set  # noqa: E402


INITIAL_LAMBDAS = {
    "v1_5_demuth_church": 0.266,
    "v1_5_kulhanek_snow_winter": 0.143,
    "v1_5_demuth_wave": 0.393,
}
CASE_IDS = tuple(INITIAL_LAMBDAS)


def residual_energy(records: list[dict]) -> float:
    return float(sum(float(record["gated_ip_residual_l2"]) ** 2 for record in records))


def existing_ratio(case_id: str) -> float:
    root = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
    u_records = jsonl(root / case_id / "seed42/U_residuals.jsonl")
    s_records = jsonl(root / case_id / "seed42/S_subject_residuals.jsonl")
    return (residual_energy(s_records) / max(residual_energy(u_records), 1e-24)) ** 0.5


def jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run_once(
    pipe,
    case: dict[str, str],
    content: Image.Image,
    style: Image.Image,
    control: Image.Image,
    residual_logger: ResidualLogger,
    block_weights: dict[str, float],
    args: argparse.Namespace,
    gate_value: float,
) -> tuple[Image.Image, list[dict], float]:
    gate = torch.ones((512, 512), dtype=torch.float32) * float(gate_value)
    set_spatial_gate(pipe, gate)
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
        ip_adapter_image_embeds=args.embeds,
        strength=args.strength,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.controlnet_scale,
        num_inference_steps=args.num_inference_steps,
        generator=torch.Generator(device="cuda").manual_seed(42),
        callback_on_step_end=callback,
    ).images[0]
    return image, residual_logger.to_dicts(), time.time() - start


def run_case(project_root: Path, case: dict[str, str], config: dict, args: argparse.Namespace) -> None:
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
        "valid_eval": masks.valid_eval,
    }.items():
        save_mask(value, mask_dir / f"{name}.png")
    (mask_dir / "mask_report.json").write_text(json.dumps(masks.report(), indent=2), encoding="utf-8")

    pipe = load_pipe(project_root, args)
    residual_logger = ResidualLogger()
    instrumented = instrument_ip_adapter_processors(
        pipe,
        residual_logger,
        spatial_gate=torch.ones((512, 512), dtype=torch.float32),
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
    args.embeds = embeds
    processor_infos = build_processor_map(pipe.unet.attn_processors)
    block_weights = processor_block_weights(processor_infos)
    (out_dir / "processor_map.json").write_text(
        json.dumps([info.to_dict() for info in processor_infos], indent=2), encoding="utf-8"
    )
    (out_dir / "instrumented_processors.json").write_text(json.dumps(instrumented, indent=2), encoding="utf-8")

    target_ratio = existing_ratio(case["canonical_case_id"])
    initial_lambda = INITIAL_LAMBDAS[case["canonical_case_id"]]
    initial_image, initial_records, initial_elapsed = run_once(
        pipe, case, content, style, control, residual_logger, block_weights, args, initial_lambda
    )
    u_energy = residual_energy(jsonl(ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot" / case["canonical_case_id"] / "seed42/U_residuals.jsonl"))
    initial_ratio = (residual_energy(initial_records) / max(u_energy, 1e-24)) ** 0.5
    final_lambda = initial_lambda
    final_image = initial_image
    final_records = initial_records
    final_elapsed = initial_elapsed
    calibrated = False
    if abs(initial_ratio / max(target_ratio, 1e-12) - 1.0) > args.tolerance:
        final_lambda = initial_lambda * target_ratio / max(initial_ratio, 1e-12)
        final_image, final_records, final_elapsed = run_once(
            pipe, case, content, style, control, residual_logger, block_weights, args, final_lambda
        )
        calibrated = True
    final_ratio = (residual_energy(final_records) / max(u_energy, 1e-24)) ** 0.5
    final_image.save(out_dir / "U_budget.png")
    with (out_dir / "U_budget_residuals.jsonl").open("w", encoding="utf-8") as handle:
        for record in final_records:
            handle.write(json.dumps(record) + "\n")
    (out_dir / "calibration.json").write_text(
        json.dumps(
            {
                "target_s_raw_ratio": target_ratio,
                "initial_lambda": initial_lambda,
                "initial_actual_ratio": initial_ratio,
                "final_lambda": final_lambda,
                "final_actual_ratio": final_ratio,
                "relative_error_to_s_raw": final_ratio / max(target_ratio, 1e-12) - 1.0,
                "calibrated_once": calibrated,
                "tolerance": args.tolerance,
                "initial_image_saved": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "case": case,
                "experiment": "v2_1_u_budget_control",
                "variant": "U_budget",
                "seed": 42,
                "schedule": "A2_highres_only",
                "elapsed_sec": final_elapsed,
                "calibration": json.loads((out_dir / "calibration.json").read_text(encoding="utf-8")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[OK] {case['canonical_case_id']} lambda={final_lambda:.6f} ratio={final_ratio:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_1_regional_pilot.yaml")
    parser.add_argument("--output-root", default="runs/ip_adapter_plus_injection/v2_1_u_budget_control")
    parser.add_argument("--tolerance", type=float, default=0.05)
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
    cases = read_cases(ROOT / config["experiment"]["manifest"])
    for case_id in CASE_IDS:
        run_case(ROOT, select_case(cases, case_id, 42), config, args)


if __name__ == "__main__":
    main()
