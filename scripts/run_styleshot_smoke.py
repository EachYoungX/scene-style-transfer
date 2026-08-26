"""Run one local StyleShot Contour image-driven smoke test."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, UNet2DConditionModel
from huggingface_hub import snapshot_download
from PIL import Image

from annotator.hed import SOFT_HEDdetector
from ip_adapter_styleshot import StyleContentStableDiffusionControlNetPipeline, StyleShot


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "models/sd15"
STYLESHOT = ROOT / "models/external/StyleShot"
CLIP = ROOT / "models/external/clip_vit_h"
CONTENT = ROOT / "data/raw/_photo_ref/photo_architecture_basilica.jpg"
STYLE = ROOT / "data/raw/monet/monet_garden_giverny_1900.jpg"
OUT = ROOT / "external_benchmark/runs/smoke/styleshot/extv1_architecture_basilica_monet"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    content = Image.open(CONTENT).convert("RGB")
    style = Image.open(STYLE).convert("RGB")
    content_512 = content.resize((512, 512), Image.Resampling.LANCZOS)
    style_512 = style.resize((512, 512), Image.Resampling.LANCZOS)
    content_512.save(OUT / "content.png")
    style_512.save(OUT / "style.png")

    detector = SOFT_HEDdetector()
    content_np = cv2.cvtColor(np.asarray(content_512), cv2.COLOR_RGB2BGR)
    contour = detector(cv2.cvtColor(content_np, cv2.COLOR_BGR2RGB))
    contour_image = Image.fromarray(contour).convert("RGB")
    contour_image.save(OUT / "contour.png")

    unet = UNet2DConditionModel.from_pretrained(
        BASE,
        subfolder="unet",
        torch_dtype=torch.float16,
        variant="fp16",
        local_files_only=True,
    )
    content_fusion_encoder = ControlNetModel.from_unet(unet)
    pipe = StyleContentStableDiffusionControlNetPipeline.from_pretrained(
        BASE,
        controlnet=content_fusion_encoder,
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

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = styleshot.generate(
        style_image=style_512,
        prompt=[["an architectural basilica scene with preserved facade layout and spatial composition"]],
        content_image=contour_image,
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
                "preprocessor": "Contour",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(OUT / "output.png")


if __name__ == "__main__":
    main()
