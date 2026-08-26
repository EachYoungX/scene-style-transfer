"""Run the frozen Vanilla IP-Adapter + same-Canny external track."""

from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import torch

from scene_style_transfer.pipeline import fit_square_crop, make_canny

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baseline import load_controlnet_pipeline  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
PAIR_MANIFEST = ROOT / "external_benchmark/configs/external_eval_pairs_v1.csv"
RUN_MANIFEST = ROOT / "external_benchmark/manifests/runs.csv"
OUTPUT_ROOT = ROOT / "external_benchmark/outputs/ipadapter_vanilla/track_B"
SEEDS = (42, 123, 777)
STEPS = 20
NEGATIVE_PROMPT = "low quality, blurry, distorted, text, watermark, copied objects"
CHECKPOINT = "models/ip_adapter/models/ip-adapter_sd15.safetensors+models/sd15+models/controlnet_canny"
COMMIT = "6b40588e263c958653353ec24eb7eb990cfa3da7"


def _run_row(pair: dict[str, str], seed: int, pipe, writer: csv.DictWriter) -> None:
    content = fit_square_crop(ROOT / pair["content_path"], 512)
    style = fit_square_crop(ROOT / pair["reference_path"], 512)
    control = make_canny(content)
    pair_root = OUTPUT_ROOT / pair["pair_id"]
    pair_root.mkdir(parents=True, exist_ok=True)
    content.save(pair_root / "content.png")
    style.save(pair_root / "style.png")
    control.save(pair_root / "canny.png")

    output_dir = pair_root / f"seed_{seed}"
    output_path = output_dir / "output.png"
    if output_path.exists():
        print(f"[SKIP] {pair['pair_id']} seed={seed}", flush=True)
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = pipe(
        prompt=pair["prompt"],
        negative_prompt=NEGATIVE_PROMPT,
        image=content,
        control_image=control,
        ip_adapter_image=style,
        strength=0.5,
        guidance_scale=6.5,
        controlnet_conditioning_scale=0.8,
        num_inference_steps=STEPS,
        generator=torch.Generator(device="cuda").manual_seed(seed),
    ).images[0]
    result.save(output_path)
    runtime = time.perf_counter() - start
    peak_vram_mb = torch.cuda.max_memory_allocated() / 1024**2
    writer.writerow(
        {
            "track_id": "B_controlled_sd15",
            "method": "ipadapter_vanilla",
            "pair_id": pair["pair_id"],
            "seed": seed,
            "status": "success",
            "output_path": str(output_path.relative_to(ROOT)),
            "backbone": "SD1.5",
            "checkpoint": CHECKPOINT,
            "commit_hash": COMMIT,
            "resolution": 512,
            "prompt": pair["prompt"],
            "negative_prompt": NEGATIVE_PROMPT,
            "steps": STEPS,
            "cfg": 6.5,
            "scheduler": pipe.scheduler.__class__.__name__,
            "style_strength": 0.45,
            "content_control": "Canny ControlNet",
            "preprocessing_time_sec": 0,
            "runtime_sec": round(runtime, 4),
            "total_time_sec": round(runtime, 4),
            "peak_vram_mb": round(peak_vram_mb, 2),
            "inversion_required": False,
            "test_time_optimization_required": False,
            "method_specific_training_required": False,
            "dedicated_training_corpus_required": False,
            "auxiliary_learned_models": "SD1.5; Canny ControlNet; IP-Adapter image encoder",
            "reference_count": 1,
            "per_pair_tuning": False,
            "notes": "Frozen B1 same-backbone baseline; seed policy 42/123/777.",
        }
    )
    print(
        f"[OK] {pair['pair_id']} seed={seed} runtime={runtime:.2f}s vram={peak_vram_mb:.0f}MB",
        flush=True,
    )
    torch.cuda.empty_cache()


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    with PAIR_MANIFEST.open(newline="", encoding="utf-8") as handle:
        pairs = list(csv.DictReader(handle))
    with RUN_MANIFEST.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_MANIFEST.read_text(encoding="utf-8").splitlines()[0].split(","))
        for pair in pairs:
            for seed in SEEDS:
                if (OUTPUT_ROOT / pair["pair_id"] / f"seed_{seed}" / "output.png").exists():
                    print(f"[SKIP] {pair['pair_id']} seed={seed}", flush=True)
                    continue
                pipe = getattr(main, "pipe", None)
                if pipe is None:
                    main.pipe = load_controlnet_pipeline(
                        ROOT / "models/sd15",
                        ROOT / "models/controlnet_canny",
                        ROOT / "models/ip_adapter",
                        0.45,
                    )
                    pipe = main.pipe
                _run_row(pair, seed, pipe, writer)
                handle.flush()
    print("B1 batch complete", flush=True)


if __name__ == "__main__":
    main()
