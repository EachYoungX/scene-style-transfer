#!/usr/bin/env python3
"""Create V1.5 residual heatmaps and a structured human-review sheet."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


BLOCKS = [
    "down_blocks.0",
    "down_blocks.1",
    "down_blocks.2",
    "mid_block",
    "up_blocks.0",
    "up_blocks.1",
    "up_blocks.2",
    "up_blocks.3",
]


def block_name(processor_name: str) -> str:
    for block in BLOCKS:
        if processor_name.startswith(block):
            return block
    return "other"


def load_records(run_root: Path):
    values = defaultdict(lambda: defaultdict(list))
    for path in run_root.glob("*/**/*_residuals.jsonl"):
        variant = path.name.removesuffix("_residuals.jsonl")
        for line in path.read_text().splitlines():
            row = json.loads(line)
            values[variant][(block_name(row["processor_name"]), int(row["step"]))].append(
                float(row["ip_residual_rms"])
            )
    return values


def write_summary(values, output: Path):
    rows = []
    for variant, cells in sorted(values.items()):
        for (block, step), samples in sorted(cells.items()):
            rows.append({
                "variant": variant,
                "block": block,
                "step": step,
                "mean_ip_residual_rms": sum(samples) / len(samples),
                "processor_count": len(samples),
            })
    with (output / "residual_heatmap_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def write_review_sheet(run_roots: list[Path], output: Path):
    cases = sorted({p.name for root in run_roots for p in root.iterdir() if p.is_dir()})
    fields = [
        "case_id", "seed", "variant", "content_identity_0_4",
        "structure_preservation_0_4", "style_strength_0_4",
        "reference_leakage_0_4", "local_geometry_risk_0_4", "notes",
    ]
    with (output / "human_review_sheet.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for root in run_roots:
            seed = root.name.split("seed")[-1].split("_")[0]
            for case in cases:
                for variant in ("A0_raw_all", "A0_all_residual_energy_matched", "A2_highres_only"):
                    writer.writerow({"case_id": case, "seed": seed, "variant": variant})


def make_heatmaps(values, output: Path):
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise SystemExit("matplotlib and numpy are required for heatmap generation") from exc

    for variant, cells in values.items():
        steps = sorted({step for _, step in cells})
        matrix = np.full((len(BLOCKS), len(steps)), np.nan)
        for (block, step), samples in cells.items():
            if block in BLOCKS:
                matrix[BLOCKS.index(block), steps.index(step)] = sum(samples) / len(samples)
        fig, ax = plt.subplots(figsize=(11, 4.8))
        image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
        ax.set_title(f"{variant}: mean IP residual RMS")
        ax.set_xlabel("Denoising step")
        ax.set_ylabel("UNet block")
        ax.set_yticks(range(len(BLOCKS)), BLOCKS)
        fig.colorbar(image, ax=ax, label="residual RMS")
        fig.tight_layout()
        fig.savefig(output / f"{variant}_residual_heatmap.png", dpi=160)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    combined = defaultdict(lambda: defaultdict(list))
    for root in args.run_root:
        values = load_records(root)
        for variant, cells in values.items():
            for key, samples in cells.items():
                combined[variant][key].extend(samples)
    write_summary(combined, args.output)
    write_review_sheet(args.run_root, args.output)
    make_heatmaps(combined, args.output)


if __name__ == "__main__":
    main()
