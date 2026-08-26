"""Run the frozen StyleShot Contour image-driven Track A benchmark."""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path

os.environ["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib" + (
    os.pathsep + os.environ["LD_LIBRARY_PATH"] if os.environ.get("LD_LIBRARY_PATH") else ""
)

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, UNet2DConditionModel
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external_benchmark/vendor/maskst"))

from annotator.hed import SOFT_HEDdetector  # noqa: E402
from ip_adapter_styleshot import StyleContentStableDiffusionControlNetPipeline, StyleShot  # noqa: E402


PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
RUN_MANIFEST = ROOT / "external_benchmark/manifests/runs.csv"
OUTPUT_ROOT = ROOT / "external_benchmark/outputs/styleshot/track_A"
BASE = ROOT / "models/sd15"
STYLESHOT = ROOT / "models/external/StyleShot"
CLIP = ROOT / "models/external/clip_vit_h"
SEEDS = (42, 123, 777)
STEPS = 50
GUIDANCE = 7.5
NEGATIVE_PROMPT = "monochrome, lowres, bad anatomy, worst quality, low quality"
COMMIT = "f56e566517d222ae48fdcb82bbc75af72ae86f97"
CHECKPOINT = "models/external/StyleShot+models/external/clip_vit_h+models/sd15"


def contour_image(detector: SOFT_HEDdetector, image: Image.Image) -> Image.Image:
    image_np = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    contour = detector(cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB))
    return Image.fromarray(contour).convert("RGB")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        fieldnames = next(csv.reader(handle))

    detector = SOFT_HEDdetector()
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

    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for pair in pairs:
            pair_root = OUTPUT_ROOT / pair["pair_id"]
            pair_root.mkdir(parents=True, exist_ok=True)
            content = Image.open(ROOT / pair["content_path"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            style = Image.open(ROOT / pair["reference_path"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            contour = contour_image(detector, content)
            content.save(pair_root / "content.png")
            style.save(pair_root / "style.png")
            contour.save(pair_root / "contour.png")
            for seed in SEEDS:
                output_dir = pair_root / f"seed_{seed}"
                output_path = output_dir / "output.png"
                if output_path.exists():
                    print(f"[SKIP] {pair['pair_id']} seed={seed}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                status = "success"
                runtime = 0.0
                peak_vram_mb = ""
                try:
                    torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    output = styleshot.generate(
                        style_image=style,
                        prompt=[[pair["prompt"]]],
                        negative_prompt=[NEGATIVE_PROMPT],
                        content_image=contour,
                        seed=seed,
                        num_samples=1,
                        guidance_scale=GUIDANCE,
                        num_inference_steps=STEPS,
                    )[0][0]
                    runtime = time.perf_counter() - started
                    output.save(output_path)
                    peak_vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2, 2)
                except Exception as exc:  # record an experimental failure and continue
                    status = f"failed_{type(exc).__name__}"
                    print(f"[FAIL] {pair['pair_id']} seed={seed}: {exc}", flush=True)
                writer.writerow(
                    {
                        "track_id": "A_native_quality",
                        "method": "styleshot",
                        "pair_id": pair["pair_id"],
                        "seed": seed,
                        "status": status,
                        "output_path": str(output_path.relative_to(ROOT)) if status == "success" else "",
                        "backbone": "official SD1.5",
                        "checkpoint": CHECKPOINT,
                        "commit_hash": COMMIT,
                        "resolution": 512,
                        "prompt": pair["prompt"],
                        "negative_prompt": NEGATIVE_PROMPT,
                        "steps": STEPS,
                        "cfg": GUIDANCE,
                        "scheduler": pipe.scheduler.__class__.__name__,
                        "style_strength": 1.0,
                        "content_control": "Contour",
                        "preprocessing_time_sec": 0,
                        "runtime_sec": round(runtime, 4),
                        "total_time_sec": round(runtime, 4),
                        "peak_vram_mb": peak_vram_mb,
                        "inversion_required": False,
                        "test_time_optimization_required": False,
                        "method_specific_training_required": False,
                        "dedicated_training_corpus_required": False,
                        "auxiliary_learned_models": "StyleShot; CLIP ViT-H; Contour ControlNet",
                        "reference_count": 1,
                        "per_pair_tuning": False,
                        "notes": "Frozen StyleShot Contour image-driven Track A route; seed policy 42/123/777.",
                    }
                )
                handle.flush()
                print(
                    f"[{ 'OK' if status == 'success' else 'FAIL' }] {pair['pair_id']} seed={seed} runtime={runtime:.2f}s vram={peak_vram_mb}",
                    flush=True,
                )
                torch.cuda.empty_cache()
    print("StyleShot Track A batch complete", flush=True)


if __name__ == "__main__":
    main()
