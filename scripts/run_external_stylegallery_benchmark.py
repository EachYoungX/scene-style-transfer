"""Run the frozen StyleGallery official accelerated Track A benchmark."""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "external_benchmark/vendor/stylegallery"
PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
RUN_MANIFEST = ROOT / "external_benchmark/manifests/runs.csv"
OUTPUT_ROOT = ROOT / "external_benchmark/outputs/stylegallery/track_A"
PYTHON = ROOT / "external_benchmark/.venvs/stylegallery_uv/bin/python"
SEEDS = (42, 123, 777)
COMMIT = "95f1b125158d9c5ef4f6ba5e9f17bcb11820d6dc"
CHECKPOINT = (
    "external_benchmark/vendor/stylegallery/pretrained_models/"
    "runwayml_stable-diffusion-v1-5+facebook/dinov2-base+"
    "dpv2/ckpts/depth_anything_v2_vitl.pth+lcm-lora-sd1.5"
)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    with RUN_MANIFEST.open(newline="", encoding="utf-8") as handle:
        fieldnames = next(csv.reader(handle))

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = "/usr/lib/wsl/lib" + (
        os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""
    )

    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for pair in pairs:
            for seed in SEEDS:
                output_dir = OUTPUT_ROOT / pair["pair_id"] / f"seed_{seed}"
                output_path = output_dir / "result_lcm.png"
                if output_path.exists():
                    print(f"[SKIP] {pair['pair_id']} seed={seed}", flush=True)
                    continue
                output_dir.mkdir(parents=True, exist_ok=True)
                content = str(ROOT / pair["content_path"])
                style = str(ROOT / pair["reference_path"])
                command = [
                    str(PYTHON),
                    "demo_accelerate.py",
                    "--model_name",
                    "pretrained_models/runwayml_stable-diffusion-v1-5",
                    "--mode",
                    "lcm",
                    "--content_image",
                    content,
                    "--style_images",
                    style,
                    "--output_folder",
                    str(output_dir),
                    "--seed",
                    str(seed),
                    "--use_depth",
                ]
                print(f"[RUN] {pair['pair_id']} seed={seed}", flush=True)
                started = time.perf_counter()
                result = subprocess.run(command, cwd=VENDOR, env=env)
                runtime = time.perf_counter() - started
                success = result.returncode == 0 and output_path.exists()
                writer.writerow(
                    {
                        "track_id": "A_native_quality",
                        "method": "stylegallery",
                        "pair_id": pair["pair_id"],
                        "seed": seed,
                        "status": "success" if success else f"failed_rc_{result.returncode}",
                        "output_path": str(output_path.relative_to(ROOT)) if success else "",
                        "backbone": "SD1.5 official accelerated LCM route",
                        "checkpoint": CHECKPOINT,
                        "commit_hash": COMMIT,
                        "resolution": 512,
                        "prompt": pair["prompt"],
                        "negative_prompt": "",
                        "steps": 28,
                        "cfg": "",
                        "scheduler": "LCMScheduler",
                        "style_strength": "",
                        "content_control": "semantic regional matching + Depth Anything V2",
                        "preprocessing_time_sec": 0,
                        "runtime_sec": round(runtime, 4),
                        "total_time_sec": round(runtime, 4),
                        "peak_vram_mb": "",
                        "inversion_required": True,
                        "test_time_optimization_required": True,
                        "method_specific_training_required": False,
                        "dedicated_training_corpus_required": False,
                        "auxiliary_learned_models": "DINOv2-base; Depth Anything V2 ViT-L; LCM LoRA",
                        "reference_count": 1,
                        "per_pair_tuning": False,
                        "notes": "Frozen optional StyleGallery Track A route; official demo_accelerate defaults; seed policy 42/123/777.",
                    }
                )
                handle.flush()
                print(
                    f"[{ 'OK' if success else 'FAIL' }] {pair['pair_id']} seed={seed} runtime={runtime:.2f}s",
                    flush=True,
                )
    print("StyleGallery Track A batch complete", flush=True)


if __name__ == "__main__":
    main()
