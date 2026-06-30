"""Run IP-Adapter Plus token-variant diagnostics for V1-A."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline import fit_square, make_canny  # noqa: E402

from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline


VARIANTS = ("raw", "pooled", "layout_suppressed", "shuffled", "texture_bank", "global_plus_texture")


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fit_square_crop(image_path: Path, size: int) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    side = min(image.width, image.height)
    left = (image.width - side) // 2
    top = (image.height - side) // 2
    image = image.crop((left, top, left + side, top + side))
    return image.resize((size, size), Image.Resampling.LANCZOS)


def transform_embeds(
    embeds: list[torch.Tensor],
    variant: str,
    seed: int,
    texture_embeds: list[torch.Tensor] | None = None,
    texture_residual_scale: float = 0.35,
) -> list[torch.Tensor]:
    if variant == "raw":
        return [embed.clone() for embed in embeds]
    if variant == "texture_bank":
        if texture_embeds is None:
            raise ValueError("texture_bank variant requires texture_embeds.")
        return [embed.clone() for embed in texture_embeds]

    output = []
    generator = torch.Generator(device=embeds[0].device).manual_seed(seed)
    if variant == "global_plus_texture" and texture_embeds is None:
        raise ValueError("global_plus_texture variant requires texture_embeds.")
    for index, embed in enumerate(embeds):
        transformed = embed.clone()
        if embed.ndim != 4:
            raise ValueError(f"Expected IP-Adapter Plus 4D embeds, got shape {tuple(embed.shape)}")
        # Shape: [batch_with_cfg, num_images, tokens, channels]. Token 0 is kept
        # as a global anchor; patch tokens are manipulated to reduce layout.
        patches = transformed[:, :, 1:, :]
        if variant == "pooled":
            pooled = patches.mean(dim=2, keepdim=True)
            transformed[:, :, 1:, :] = pooled.expand_as(patches)
        elif variant == "layout_suppressed":
            pooled = patches.mean(dim=2, keepdim=True)
            transformed[:, :, 1:, :] = 0.65 * patches + 0.35 * pooled.expand_as(patches)
        elif variant == "shuffled":
            perm = torch.randperm(patches.shape[2], generator=generator, device=patches.device)
            transformed[:, :, 1:, :] = patches[:, :, perm, :]
        elif variant == "global_plus_texture":
            texture = texture_embeds[index]
            if texture.shape != embed.shape:
                raise ValueError(f"Texture embed shape {tuple(texture.shape)} does not match {tuple(embed.shape)}")
            texture_patches = texture[:, :, 1:, :]
            source_mean = patches.mean(dim=2, keepdim=True)
            texture_mean = texture_patches.mean(dim=2, keepdim=True)
            transformed[:, :, 0:1, :] = embed[:, :, 0:1, :]
            transformed[:, :, 1:, :] = source_mean + texture_residual_scale * (texture_patches - texture_mean)
        else:
            raise ValueError(f"Unknown variant: {variant}")
        output.append(transformed)
    return output


def make_texture_bank_image(style: Image.Image, size: int, patch_sizes: list[int], seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    source = style.resize((size, size), Image.Resampling.LANCZOS).convert("RGB")
    source_np = np.asarray(source).astype(np.float32)
    canvas = np.zeros((size, size, 3), dtype=np.float32)
    tile = max(16, min(patch_sizes))

    for y in range(0, size, tile):
        for x in range(0, size, tile):
            patch_size = int(rng.choice(patch_sizes))
            sx = int(rng.integers(0, max(1, size - patch_size + 1)))
            sy = int(rng.integers(0, max(1, size - patch_size + 1)))
            crop = source_np[sy : sy + patch_size, sx : sx + patch_size]
            crop = Image.fromarray(crop.astype(np.uint8), mode="RGB").resize((tile, tile), Image.Resampling.BICUBIC)
            tile_np = np.asarray(crop).astype(np.float32)
            y1 = min(y + tile, size)
            x1 = min(x + tile, size)
            canvas[y:y1, x:x1] = tile_np[: y1 - y, : x1 - x]

    image = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), mode="RGB")
    bank = np.asarray(image.filter(ImageFilter.GaussianBlur(radius=0.35)))
    return Image.fromarray(np.clip(bank, 0, 255).astype(np.uint8), mode="RGB")


def load_pipe(project_root: Path, args: argparse.Namespace):
    controlnet = ControlNetModel.from_pretrained(
        project_root / args.controlnet_dir,
        torch_dtype=torch.float16,
        local_files_only=True,
        variant="fp16",
    )
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        project_root / args.model_dir,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
        variant="fp16",
    )
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter(
        str(project_root / args.ip_adapter_dir),
        subfolder="models",
        weight_name=args.weight_name,
        local_files_only=True,
    )
    pipe.set_ip_adapter_scale(args.ip_adapter_scale)
    return pipe.to("cuda")


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
    panels = []
    for label, path in [("content", out_dir / "content.png"), ("style", out_dir / "style.png")]:
        panels.append((label, Image.open(path).convert("RGB")))
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
    grid.save(out_dir / "variant_grid.png")


def run_case(case: dict[str, str], project_root: Path, args: argparse.Namespace) -> None:
    out_dir = project_root / "runs" / "ip_adapter_plus_variants" / args.run_name / case["case_id"]
    if out_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{out_dir} exists. Use --overwrite.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    pipe = load_pipe(project_root, args)
    content = fit_square(project_root / case["content"], args.size)
    style = fit_square_crop(project_root / case["style"], args.size)
    control = make_canny(content)
    content.save(out_dir / "content.png")
    style.save(out_dir / "style.png")
    control.save(out_dir / "canny.png")
    texture_style = None
    texture_embeds = None
    if any(variant in {"texture_bank", "global_plus_texture"} for variant in args.variants):
        texture_style = make_texture_bank_image(style, args.size, args.texture_patch_sizes, args.seed)
        texture_style.save(out_dir / "texture_bank_style.png")

    base_embeds = pipe.prepare_ip_adapter_image_embeds(
        ip_adapter_image=style,
        ip_adapter_image_embeds=None,
        device="cuda",
        num_images_per_prompt=1,
        do_classifier_free_guidance=True,
    )
    if texture_style is not None:
        texture_embeds = pipe.prepare_ip_adapter_image_embeds(
            ip_adapter_image=texture_style,
            ip_adapter_image_embeds=None,
            device="cuda",
            num_images_per_prompt=1,
            do_classifier_free_guidance=True,
        )

    index = []
    for variant in args.variants:
        embeds = transform_embeds(
            base_embeds,
            variant,
            args.seed,
            texture_embeds=texture_embeds,
            texture_residual_scale=args.texture_residual_scale,
        )
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        image = pipe(
            prompt=f"{case['prompt']}, strong reference style, preserve content layout",
            negative_prompt=args.negative_prompt,
            image=content,
            control_image=control,
            ip_adapter_image_embeds=embeds,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            controlnet_conditioning_scale=args.controlnet_scale,
            num_inference_steps=args.num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
        image.save(out_dir / f"{variant}.png")
        index.append(
            {
                "variant": variant,
                "elapsed_sec": round(time.time() - start, 4),
                "peak_allocated_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 4),
                "embed_shapes": [tuple(embed.shape) for embed in embeds],
            }
        )
        print(f"[OK] {case['case_id']} / {variant}")

    (out_dir / "index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    make_grid(out_dir, list(args.variants))
    print(out_dir / "variant_grid.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/compatibility_diagnostic_pairs.csv")
    parser.add_argument("--case-id", action="append", required=True)
    parser.add_argument("--run-name", default="v1a_plus_token_variants_12step")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--strength", type=float, default=0.76)
    parser.add_argument("--guidance-scale", type=float, default=5.8)
    parser.add_argument("--controlnet-scale", type=float, default=0.72)
    parser.add_argument("--ip-adapter-scale", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--texture-patch-sizes", nargs="+", type=int, default=[32, 64, 96])
    parser.add_argument("--texture-residual-scale", type=float, default=0.35)
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted, text, watermark, copied objects")
    parser.add_argument("--model-dir", default="models/sd15")
    parser.add_argument("--controlnet-dir", default="models/controlnet_canny")
    parser.add_argument("--ip-adapter-dir", default="models/ip_adapter_plus")
    parser.add_argument("--weight-name", default="ip-adapter-plus_sd15.safetensors")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    cases = read_cases(project_root / args.manifest)
    selected = set(args.case_id)
    for case in cases:
        if case["case_id"] in selected:
            run_case(case, project_root, args)


if __name__ == "__main__":
    main()
