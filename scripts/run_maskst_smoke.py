"""Run one local MaskST image-driven smoke test using the official masking route."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, UNet2DConditionModel
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

from annotator.hed import SOFT_HEDdetector
from ip_adapter_styleshot import StyleContentStableDiffusionControlNetPipeline, StyleShot


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/sd15"
STYLESHOT = ROOT / "models/external/StyleShot"
CLIP = ROOT / "models/external/clip_vit_h"
CONTENT = ROOT / "data/raw/_photo_ref/photo_architecture_basilica.jpg"
STYLE = ROOT / "data/raw/monet/monet_garden_giverny_1900.jpg"
OUT = ROOT / "external_benchmark/runs/smoke/maskst/extv1_architecture_basilica_monet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    content = Image.open(CONTENT).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    style = Image.open(STYLE).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
    content.save(OUT / "content.png")
    style.save(OUT / "style.png")

    detector = SOFT_HEDdetector()
    contour = detector(np.asarray(content))
    contour_image = Image.fromarray(contour).convert("RGB")
    contour_image.save(OUT / "contour.png")

    unet = UNet2DConditionModel.from_pretrained(
        BASE, subfolder="unet", torch_dtype=torch.float16, variant="fp16", local_files_only=True
    )
    controlnet = ControlNetModel.from_unet(unet)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        BASE,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
        requires_safety_checker=False,
        variant="fp16",
        local_files_only=True,
    )
    styleshot = StyleShot(
        "cuda",
        pipe,
        str(STYLESHOT / "pretrained_weight/ip.bin"),
        str(STYLESHOT / "pretrained_weight/style_aware_encoder.bin"),
        str(CLIP),
    )

    tokenizer = CLIPTokenizer.from_pretrained(BASE / "tokenizer", local_files_only=True)
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        CLIP, local_files_only=True
    ).to("cuda", dtype=pipe.dtype)
    ids = tokenizer(
        "person, animal, plant, or object in the foreground",
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids
    with torch.inference_mode():
        neg_content_embd = text_encoder(ids.to("cuda")).text_embeds

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = styleshot.generate(
        style_image=style,
        prompt=[["an architectural basilica scene with preserved facade layout and spatial composition"]],
        content_image=contour_image,
        neg_content_embd=neg_content_embd,
        less_condition=True,
        seed=42,
        num_samples=1,
        num_inference_steps=8,
        guidance_scale=7.5,
    )[0][0]
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
                "route": "MaskST official less_condition masking",
                "preprocessor": "Contour",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT / "output.png")


if __name__ == "__main__":
    main()
