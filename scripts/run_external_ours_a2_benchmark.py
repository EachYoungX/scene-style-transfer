"""Run the frozen Ours A2 fixed external track."""

from __future__ import annotations

import csv
import time
from dataclasses import replace
from pathlib import Path

from PIL import Image

from scene_style_transfer.pipeline import PipelineConfig, StyleTransferPipeline, fit_square_crop


ROOT = Path(__file__).resolve().parents[1]
PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
RUN_MANIFEST = ROOT / "external_benchmark/manifests/runs.csv"
OUTPUT_ROOT = ROOT / "external_benchmark/outputs/ours_a2_fixed/track_B"
SEEDS = (42, 123, 777)
STEPS = 30
NEGATIVE_PROMPT = "low quality, blurry, distorted, text, watermark, copied objects"
CHECKPOINT = "models/ip_adapter_plus/models/ip-adapter-plus_sd15.safetensors+models/sd15+models/controlnet_canny"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        fieldnames = next(csv.reader(handle))

    pipe = StyleTransferPipeline(
        ROOT,
        PipelineConfig(
            num_inference_steps=STEPS,
            prompt="",
            negative_prompt=NEGATIVE_PROMPT,
            ip_adapter_scale=0.9,
        ),
    )

    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for pair in pairs:
            pair_root = OUTPUT_ROOT / pair["pair_id"]
            pair_root.mkdir(parents=True, exist_ok=True)
            content = Image.open(ROOT / pair["content_path"]).convert("RGB")
            style = Image.open(ROOT / pair["reference_path"]).convert("RGB")
            fit_square_crop(content, 512).save(pair_root / "content.png")
            fit_square_crop(style, 512).save(pair_root / "style.png")
            for seed in SEEDS:
                output_dir = pair_root / f"seed_{seed}"
                output_path = output_dir / "output.png"
                if output_path.exists():
                    print(f"[SKIP] {pair['pair_id']} seed={seed}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                pipe.config = replace(pipe.config, prompt=pair["prompt"])
                start = time.perf_counter()
                result = pipe.generate(content, style, seed=seed, reference_strength=0.6)
                runtime = time.perf_counter() - start
                result.image.save(output_path)
                peak_vram_mb = (result.peak_allocated_gb or 0.0) * 1024
                writer.writerow(
                    {
                        "track_id": "B_controlled_sd15",
                        "method": "ours_a2_fixed",
                        "pair_id": pair["pair_id"],
                        "seed": seed,
                        "status": "success",
                        "output_path": str(output_path.relative_to(ROOT)),
                        "backbone": "SD1.5",
                        "checkpoint": CHECKPOINT,
                        "commit_hash": "local_frozen_a2_path",
                        "resolution": 512,
                        "prompt": pair["prompt"],
                        "negative_prompt": NEGATIVE_PROMPT,
                        "steps": STEPS,
                        "cfg": 5.8,
                        "scheduler": "DDIMScheduler",
                        "style_strength": 0.6,
                        "content_control": "Canny ControlNet",
                        "preprocessing_time_sec": 0,
                        "runtime_sec": round(runtime, 4),
                        "total_time_sec": round(runtime, 4),
                        "peak_vram_mb": round(peak_vram_mb, 2),
                        "inversion_required": False,
                        "test_time_optimization_required": False,
                        "method_specific_training_required": False,
                        "dedicated_training_corpus_required": False,
                        "auxiliary_learned_models": "IP-Adapter Plus; Canny ControlNet; frozen A2 schedule",
                        "reference_count": 1,
                        "per_pair_tuning": False,
                        "notes": "Frozen A2_highres_only path; global reference strength lambda=0.6; seed policy 42/123/777.",
                    }
                )
                handle.flush()
                print(
                    f"[OK] {pair['pair_id']} seed={seed} runtime={runtime:.2f}s vram={peak_vram_mb:.0f}MB",
                    flush=True,
                )
    print("Ours A2 Track B batch complete", flush=True)


if __name__ == "__main__":
    main()
