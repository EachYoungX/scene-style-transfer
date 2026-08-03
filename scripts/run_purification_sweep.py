"""Run A2-path reference-token masking and SVD purification strength sweeps."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline import fit_square, make_canny  # noqa: E402
from run_ip_adapter_plus_injection_variants import (  # noqa: E402
    fit_square_crop,
    load_pipe,
    set_ip_adapter_schedule_step,
)
from diagnostics.injection_schedule import (  # noqa: E402
    build_processor_map,
    build_variant_schedule,
    processor_block_weights,
)
from diagnostics.ip_adapter_instrumentation import (  # noqa: E402
    ResidualLogger,
    instrument_ip_adapter_processors,
)


METHODS = (
    "mask_highnorm",
    "mask_global_deviation",
    "svd_fixed",
    "svd_time_aware",
)


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _patches(embed: torch.Tensor) -> torch.Tensor:
    if embed.ndim != 4 or embed.shape[2] < 2:
        raise ValueError(f"Expected [batch, images, global+patches, dim], got {tuple(embed.shape)}")
    return embed[:, :, 1:, :]


def purify_embeds(
    embeds: list[torch.Tensor],
    method: str,
    drop_ratio: float = 0.5,
    retain_ratio: float = 0.7,
    tail_beta: float = 0.25,
) -> list[torch.Tensor]:
    """Apply a deterministic, batch-safe purification to Plus patch tokens."""
    output: list[torch.Tensor] = []
    for embed in embeds:
        transformed = embed.clone()
        patches = _patches(transformed)
        if method.startswith("mask_"):
            count = patches.shape[2]
            n_drop = max(1, min(count - 1, round(count * drop_ratio)))
            if method == "mask_highnorm":
                score = torch.linalg.vector_norm(patches.float(), dim=-1)
            elif method == "mask_global_deviation":
                global_direction = patches.float().mean(dim=2, keepdim=True)
                score = 1.0 - torch.nn.functional.cosine_similarity(
                    patches.float(), global_direction, dim=-1, eps=1e-8
                )
            else:
                raise ValueError(f"Unknown masking method: {method}")
            drop_indices = torch.topk(score, k=n_drop, dim=2).indices
            keep = torch.ones_like(score, dtype=torch.bool)
            keep.scatter_(2, drop_indices, False)
            patches.mul_(keep.unsqueeze(-1).to(dtype=patches.dtype))
        elif method.startswith("svd_"):
            if not 0.0 < retain_ratio <= 1.0:
                raise ValueError("retain_ratio must be in (0, 1]")
            for batch_index in range(patches.shape[0]):
                for image_index in range(patches.shape[1]):
                    matrix = patches[batch_index, image_index].float()
                    mean = matrix.mean(dim=0, keepdim=True)
                    centered = matrix - mean
                    u, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
                    rank = max(1, min(singular_values.shape[0], round(singular_values.shape[0] * retain_ratio)))
                    filtered = (u[:, :rank] * singular_values[:rank]) @ vh[:rank]
                    filtered = filtered + mean
                    if method == "svd_fixed":
                        filtered = mean + (filtered - mean) + tail_beta * (matrix - filtered)
                    patches[batch_index, image_index] = filtered.to(dtype=patches.dtype)
        else:
            raise ValueError(f"Unknown purification method: {method}")
        output.append(transformed)
    return output


def label_panel(image: Image.Image, label: str, height: int = 256) -> Image.Image:
    width = int(image.width * height / image.height)
    canvas = image.resize((width, height)).convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    pad = 5
    draw.rectangle((0, 0, bbox[2] + pad * 2, bbox[3] + pad * 2), fill=(0, 0, 0))
    draw.text((pad, pad), label, fill=(255, 255, 255), font=font)
    return canvas


def make_grid(out_dir: Path, labels: list[str]) -> None:
    panels = [("content", Image.open(out_dir / "content.png").convert("RGB"))]
    for label in labels:
        panels.append((label, Image.open(out_dir / f"{label}.png").convert("RGB")))
    rendered = [(label, label_panel(image, label)) for label, image in panels]
    cols = 4
    rows = (len(rendered) + cols - 1) // cols
    gutter = 8
    cell_w = max(image.width for _, image in rendered)
    cell_h = max(image.height for _, image in rendered)
    grid = Image.new("RGB", (cols * cell_w + (cols - 1) * gutter, rows * cell_h + (rows - 1) * gutter), (32, 32, 32))
    for index, (_, image) in enumerate(rendered):
        x = (index % cols) * (cell_w + gutter)
        y = (index // cols) * (cell_h + gutter)
        grid.paste(image, (x, y))
    grid.save(out_dir / "purification_sweep_grid.png")


def run_case(case: dict[str, str], root: Path, args: argparse.Namespace) -> None:
    out_dir = root / "runs" / "ip_adapter_plus_injection" / args.run_name / case["case_id"]
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists; use --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pipe = load_pipe(root, args)
    logger = ResidualLogger()
    instrument_ip_adapter_processors(pipe, logger)
    content = fit_square(root / case["content"], args.size)
    style = fit_square_crop(root / case["style"], args.size)
    control = make_canny(content)
    content.save(out_dir / "content.png")
    style.save(out_dir / "style.png")
    control.save(out_dir / "canny.png")
    base_embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=style,
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )

    block_weights = processor_block_weights(build_processor_map(pipe.unet.attn_processors))
    labels: list[str] = []
    manifest: list[dict[str, object]] = []
    if getattr(args, "frozen_pairing", False):
        method_strengths = zip(args.methods, args.strength_multipliers)
    else:
        method_strengths = ((method, strength_multiplier) for method in args.methods for strength_multiplier in args.strength_multipliers)
    for method, strength_multiplier in method_strengths:
            label = f"{method}_x{strength_multiplier:g}"
            labels.append(label)
            logger.clear()
            embeds_holder = [purify_embeds(base_embeds, method, args.drop_ratio, args.retain_ratio, args.tail_beta)]
            base_scale = args.ip_adapter_scale * strength_multiplier
            schedule = build_variant_schedule(
                "A2_highres_only",
                base_scale,
                args.num_inference_steps,
                block_weights,
            )
            pipe.scheduler.set_timesteps(args.num_inference_steps, device="cuda")
            timesteps = list(pipe.scheduler.timesteps)
            set_ip_adapter_schedule_step(pipe, schedule, 0)
            logger.set_step(0, timesteps[0].item() if hasattr(timesteps[0], "item") else timesteps[0])

            def callback(pipe_ref, step_index, timestep, callback_kwargs):
                next_step = min(step_index + 1, args.num_inference_steps - 1)
                if method == "svd_time_aware":
                    progress = (next_step + 1) / max(args.num_inference_steps, 1)
                    beta = 0.0 if progress <= 0.4 else (0.25 if progress <= 0.7 else 0.5)
                    embeds_holder[0][:] = purify_embeds(
                        base_embeds, "svd_fixed", args.drop_ratio, args.retain_ratio, beta
                    )
                logger.set_step(next_step, timesteps[next_step].item() if hasattr(timesteps[next_step], "item") else timesteps[next_step])
                set_ip_adapter_schedule_step(pipe_ref, schedule, next_step)
                return callback_kwargs

            start = time.time()
            image = pipe(
                prompt=case["prompt"],
                negative_prompt=args.negative_prompt,
                image=content,
                control_image=control,
                ip_adapter_image_embeds=embeds_holder[0],
                strength=args.strength,
                guidance_scale=args.guidance_scale,
                controlnet_conditioning_scale=args.controlnet_scale,
                num_inference_steps=args.num_inference_steps,
                generator=torch.Generator(device="cuda").manual_seed(args.seed),
                callback_on_step_end=callback,
            ).images[0]
            image.save(out_dir / f"{label}.png")
            residual_path = out_dir / f"{label}_residuals.jsonl"
            with residual_path.open("w", encoding="utf-8") as f:
                for record in logger.to_dicts():
                    f.write(json.dumps(record) + "\n")
            manifest.append({
                "label": label,
                "method": method,
                "strength_multiplier": strength_multiplier,
                "a2_base_scale": base_scale,
                "elapsed_sec": round(time.time() - start, 4),
                "residual_records": len(logger.records),
            })
            print(f"[OK] {case['case_id']} / {label}")
    (out_dir / "index.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    make_grid(out_dir, labels)
    print(out_dir / "purification_sweep_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config")
    parser.add_argument("--manifest", default="configs/experiment/v1_5_cases.csv")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--run-name", default="purification_sweep_seed42")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--strength-multipliers", nargs="+", type=float, default=[0.7, 1.0, 1.3])
    parser.add_argument("--drop-ratio", type=float, default=0.5)
    parser.add_argument("--retain-ratio", type=float, default=0.7)
    parser.add_argument("--tail-beta", type=float, default=0.25)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--strength", type=float, default=0.76)
    parser.add_argument("--guidance-scale", type=float, default=5.8)
    parser.add_argument("--controlnet-scale", type=float, default=0.72)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter_plus")
    parser.add_argument("--weight-name", default="ip-adapter-plus_sd15.safetensors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.config:
        config_path = root / args.config
        frozen = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        args.manifest = frozen.get("manifest", args.manifest)
        experiment = frozen["experiment"]
        args.num_inference_steps = int(experiment["num_inference_steps"])
        args.run_name = f"{experiment['run_name']}_seed{args.seed}"
        args.case_id = list(frozen["pairs"])
        args.methods = [entry["method"] for entry in frozen["baselines"]]
        args.strength_multipliers = [float(entry["strength_multiplier"]) for entry in frozen["baselines"]]
        args.frozen_pairing = True
        if len(set(args.strength_multipliers)) != 1:
            raise ValueError("Frozen purification baselines must use one global strength multiplier")
    if not args.case_id:
        raise ValueError("Provide --case-id or --config")
    selected = set(args.case_id)
    cases = [case for case in read_cases(root / args.manifest) if case["case_id"] in selected]
    if len(cases) != len(selected):
        raise ValueError(f"Unknown case id(s): {sorted(selected - {case['case_id'] for case in cases})}")
    for case in cases:
        run_case(case, root, args)


if __name__ == "__main__":
    main()
