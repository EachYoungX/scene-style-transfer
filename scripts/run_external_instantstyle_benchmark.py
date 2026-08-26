"""Run the frozen InstantStyle native-quality external track."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler
from PIL import Image
from transformers import CLIPTextModelWithProjection, CLIPTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ip_adapter import IPAdapter  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
RUN_MANIFEST = ROOT / "external_benchmark/manifests/runs.csv"
OUTPUT_ROOT = ROOT / "external_benchmark/outputs/instantstyle/track_A"
BASE = ROOT / "models/sd15"
IMAGE_ENCODER = ROOT / "models/external/clip_vit_h"
IP_CKPT = ROOT / "models/ip_adapter/models/ip-adapter_sd15.safetensors"
SEEDS = (42, 123, 777)
STEPS = 30
NEGATIVE_PROMPT = "text, watermark, lowres, low quality, worst quality, deformed, blurry"
CHECKPOINT = "models/ip_adapter/models/ip-adapter_sd15.safetensors+models/external/clip_vit_h+models/sd15"
COMMIT = "6b40588e263c958653353ec24eb7eb990cfa3da7"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        fieldnames = next(csv.reader(handle))

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

    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for pair in pairs:
            pair_root = OUTPUT_ROOT / pair["pair_id"]
            pair_root.mkdir(parents=True, exist_ok=True)
            content = Image.open(ROOT / pair["content_path"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            style = Image.open(ROOT / pair["reference_path"]).convert("RGB").resize((512, 512), Image.Resampling.LANCZOS)
            content.save(pair_root / "content.png")
            style.save(pair_root / "style.png")
            for seed in SEEDS:
                output_dir = pair_root / f"seed_{seed}"
                output_path = output_dir / "output.png"
                if output_path.exists():
                    print(f"[SKIP] {pair['pair_id']} seed={seed}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                output = ip_model.generate(
                    pil_image=style,
                    prompt=pair["prompt"],
                    negative_prompt=NEGATIVE_PROMPT,
                    scale=1.0,
                    guidance_scale=5.0,
                    num_samples=1,
                    num_inference_steps=STEPS,
                    seed=seed,
                    neg_content_emb=neg_content_emb,
                )[0]
                runtime = time.perf_counter() - start
                output.save(output_path)
                peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
                writer.writerow(
                    {
                        "track_id": "A_native_quality",
                        "method": "instantstyle",
                        "pair_id": pair["pair_id"],
                        "seed": seed,
                        "status": "success",
                        "output_path": str(output_path.relative_to(ROOT)),
                        "backbone": "SD1.5 official experimental route",
                        "checkpoint": CHECKPOINT,
                        "commit_hash": COMMIT,
                        "resolution": 512,
                        "prompt": pair["prompt"],
                        "negative_prompt": NEGATIVE_PROMPT,
                        "steps": STEPS,
                        "cfg": 5.0,
                        "scheduler": pipe.scheduler.__class__.__name__,
                        "style_strength": 1.0,
                        "content_control": "official text prompt; no external control image",
                        "preprocessing_time_sec": 0,
                        "runtime_sec": round(runtime, 4),
                        "total_time_sec": round(runtime, 4),
                        "peak_vram_mb": round(peak_vram_mb, 2),
                        "inversion_required": False,
                        "test_time_optimization_required": False,
                        "method_specific_training_required": False,
                        "dedicated_training_corpus_required": False,
                        "auxiliary_learned_models": "LAION CLIP ViT-H; IP-Adapter",
                        "reference_count": 1,
                        "per_pair_tuning": False,
                        "notes": "Frozen Track A native-quality InstantStyle route; seed policy 42/123/777.",
                    }
                )
                handle.flush()
                print(
                    f"[OK] {pair['pair_id']} seed={seed} runtime={runtime:.2f}s vram={peak_vram_mb:.0f}MB",
                    flush=True,
                )
                torch.cuda.empty_cache()
    print("InstantStyle Track A batch complete", flush=True)


if __name__ == "__main__":
    main()
