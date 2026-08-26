"""Run one local InstantStyle SD1.5 smoke test."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from ip_adapter import IPAdapter


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/sd15"
IMAGE_ENCODER = ROOT / "models/external/clip_vit_h"
IP_CKPT = ROOT / "models/ip_adapter/models/ip-adapter_sd15.safetensors"
CONTENT = ROOT / "data/raw/_photo_ref/photo_architecture_basilica.jpg"
STYLE = ROOT / "data/raw/monet/monet_garden_giverny_1900.jpg"
OUT = ROOT / "external_benchmark/runs/smoke/instantstyle/extv1_architecture_basilica_monet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    content = Image.open(CONTENT).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    style = Image.open(STYLE).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    content.save(OUT / "content.png")
    style.save(OUT / "style.png")

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        variant="fp16",
        local_files_only=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_vae_tiling()
    ip_model = IPAdapter(pipe, str(IMAGE_ENCODER), str(IP_CKPT), "cuda", target_blocks=["block"])

    tokenizer = CLIPTokenizer.from_pretrained(IMAGE_ENCODER, local_files_only=True)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        IMAGE_ENCODER, local_files_only=True
    ).to("cuda", dtype=pipe.dtype)
    tokens = tokenizer(["person, animal, plant, or object in the foreground"], return_tensors="pt")
    with torch.inference_mode():
        neg_content_emb = text_encoder(**{key: value.to("cuda") for key, value in tokens.items()}).text_embeds
        neg_content_emb *= 0.8

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = ip_model.generate(
        pil_image=style,
        prompt="an architectural basilica scene with preserved facade layout and spatial composition",
        negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, blurry",
        scale=1.0,
        guidance_scale=5.0,
        num_samples=1,
        num_inference_steps=8,
        seed=42,
        neg_content_emb=neg_content_emb,
    )[0]
    runtime = time.perf_counter() - start
    output.save(OUT / "output.png")
    (OUT / "runtime.json").write_text(
        json.dumps(
            {
                "status": "success",
                "runtime_sec": round(runtime, 4),
                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
                "device": torch.cuda.get_device_name(0),
                "steps": 8,
                "seed": 42,
                "route": "InstantStyle SD1.5 global feature subtraction",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT / "output.png")


if __name__ == "__main__":
    main()
