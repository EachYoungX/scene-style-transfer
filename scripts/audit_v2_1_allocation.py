"""Audit V2.1 residual budgets and token coverage from existing run artifacts.

This script does not load the diffusion model and does not generate images. It
aggregates the residual JSONL files already written by the V2.1 pilot and the
saved effective gate snapshots.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_ROOT = ROOT / "runs/ip_adapter_plus_injection/v2_1_regional_pilot"
DEFAULT_OUTPUT = DEFAULT_RUN_ROOT / "audits/v2_1_allocation"
CASE_DIRS = {
    "church": "v1_5_demuth_church",
    "snow_winter": "v1_5_kulhanek_snow_winter",
    "wave": "v1_5_demuth_wave",
}
VARIANTS = ("U", "S_subject", "S_background")
RESOLUTIONS = (64, 32, 16)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def aggregate_resolution(records: list[dict], resolution: int) -> dict[str, float | int | None]:
    selected = [record for record in records if record.get("spatial_gate_height") == resolution]
    if not selected:
        return {
            "records": 0,
            "raw_ip_rms": None,
            "gated_ip_rms": None,
            "rms_ratio": None,
            "raw_l2_energy": None,
            "gated_l2_energy": None,
        }

    raw_energy = sum(float(record["raw_ip_residual_l2"]) ** 2 for record in selected)
    gated_energy = sum(float(record["gated_ip_residual_l2"]) ** 2 for record in selected)
    raw_count = sum(
        float(record["raw_ip_residual_l2"]) ** 2 / max(float(record["raw_ip_residual_rms"]) ** 2, 1e-24)
        for record in selected
    )
    gated_count = sum(
        float(record["gated_ip_residual_l2"]) ** 2 / max(float(record["gated_ip_residual_rms"]) ** 2, 1e-24)
        for record in selected
    )
    raw_rms = math.sqrt(raw_energy / max(raw_count, 1e-24))
    gated_rms = math.sqrt(gated_energy / max(gated_count, 1e-24))
    return {
        "records": len(selected),
        "raw_ip_rms": raw_rms,
        "gated_ip_rms": gated_rms,
        "rms_ratio": gated_rms / max(raw_rms, 1e-12),
        "raw_l2_energy": raw_energy,
        "gated_l2_energy": gated_energy,
    }


def aggregate_region(records: list[dict], resolution: int, region: str) -> dict[str, float | int | None]:
    """Aggregate region RMS when logs were produced with region auditing enabled."""
    raw_key = f"raw_rms_{region}"
    gated_key = f"gated_rms_{region}"
    count_key = f"{region}_token_count"
    selected = [
        record
        for record in records
        if record.get("spatial_gate_height") == resolution
        and record.get(raw_key) is not None
        and record.get(gated_key) is not None
        and record.get(count_key) is not None
    ]
    if not selected:
        return {
            "token_count": None,
            "raw_ip_rms": None,
            "gated_ip_rms": None,
            "rms_ratio": None,
        }
    raw_energy = 0.0
    gated_energy = 0.0
    raw_count = 0.0
    gated_count = 0.0
    token_count = 0
    for record in selected:
        full_tokens = int(record["spatial_gate_height"]) * int(record["spatial_gate_width"])
        raw_elements = float(record["raw_ip_residual_l2"]) ** 2 / max(float(record["raw_ip_residual_rms"]) ** 2, 1e-24)
        gated_elements = float(record["gated_ip_residual_l2"]) ** 2 / max(float(record["gated_ip_residual_rms"]) ** 2, 1e-24)
        raw_hidden = raw_elements / max(full_tokens, 1)
        gated_hidden = gated_elements / max(full_tokens, 1)
        count = int(record[count_key])
        raw_rms = float(record[raw_key])
        gated_rms = float(record[gated_key])
        raw_energy += raw_rms**2 * count * raw_hidden
        gated_energy += gated_rms**2 * count * gated_hidden
        raw_count += count * raw_hidden
        gated_count += count * gated_hidden
        token_count = max(token_count, count)
    raw_rms = math.sqrt(raw_energy / max(raw_count, 1e-24))
    gated_rms = math.sqrt(gated_energy / max(gated_count, 1e-24))
    return {
        "token_count": token_count,
        "raw_ip_rms": raw_rms,
        "gated_ip_rms": gated_rms,
        "rms_ratio": gated_rms / max(raw_rms, 1e-12),
    }


def coverage(run_dir: Path, variant: str, resolution: int) -> dict[str, float | int]:
    summary_path = run_dir / "effective_region_gates" / variant / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    values = summary["scales"][str(resolution)]
    gate = np.asarray(
        Image.open(run_dir / "effective_region_gates" / variant / f"gate_{resolution}x{resolution}.png")
    )
    active = gate > 0
    return {
        "active_tokens": int(values["active_tokens"]),
        "token_fraction": float(values["active_fraction"]),
        "nonzero_pixels": int(active.sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 777])
    parser.add_argument("--include-s-match", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    variants = (*VARIANTS, "S_match") if args.include_s_match else VARIANTS

    residual_rows: list[dict[str, object]] = []
    total_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    region_coverage_rows: list[dict[str, object]] = []

    for case, case_dir_name in CASE_DIRS.items():
        for seed in args.seeds:
            seed_dir = args.run_root / case_dir_name / f"seed{seed}"
            aggregate_by_variant_resolution: dict[str, dict[int, dict]] = {}
            for variant in variants:
                records = load_jsonl(seed_dir / f"{variant}_residuals.jsonl")
                aggregate_by_variant_resolution[variant] = {}
                for resolution in RESOLUTIONS:
                    values = aggregate_resolution(records, resolution)
                    subject_values = aggregate_region(records, resolution, "subject")
                    background_values = aggregate_region(records, resolution, "background")
                    aggregate_by_variant_resolution[variant][resolution] = values
                    residual_rows.append(
                        {
                            "case": case,
                            "seed": seed,
                            "variant": variant,
                            "resolution": resolution,
                            **values,
                            "subject_token_count": subject_values["token_count"],
                            "subject_raw_ip_rms": subject_values["raw_ip_rms"],
                            "subject_gated_ip_rms": subject_values["gated_ip_rms"],
                            "subject_rms_ratio": subject_values["rms_ratio"],
                            "background_token_count": background_values["token_count"],
                            "background_raw_ip_rms": background_values["raw_ip_rms"],
                            "background_gated_ip_rms": background_values["gated_ip_rms"],
                            "background_rms_ratio": background_values["rms_ratio"],
                        }
                    )
                    gate_values = coverage(seed_dir, variant, resolution)
                    coverage_rows.append(
                        {
                            "case": case,
                            "seed": seed,
                            "variant": variant,
                            "resolution": resolution,
                            **gate_values,
                        }
                    )
            for resolution in RESOLUTIONS:
                subject_gate = np.asarray(
                    Image.open(seed_dir / "effective_region_gates" / "S_subject" / f"gate_{resolution}x{resolution}.png")
                ) > 0
                background_gate = np.asarray(
                    Image.open(seed_dir / "effective_region_gates" / "S_background" / f"gate_{resolution}x{resolution}.png")
                ) > 0
                overlap = subject_gate & background_gate
                subject_count = int(subject_gate.sum())
                background_count = int(background_gate.sum())
                overlap_count = int(overlap.sum())
                region_coverage_rows.append(
                    {
                        "case": case,
                        "seed": seed,
                        "resolution": resolution,
                        "subject_active_tokens": subject_count,
                        "background_active_tokens": background_count,
                        "overlap_tokens": overlap_count,
                        "union_tokens": int((subject_gate | background_gate).sum()),
                        "overlap_of_subject": overlap_count / max(subject_count, 1),
                        "overlap_of_background": overlap_count / max(background_count, 1),
                    }
                )

            for resolution in RESOLUTIONS:
                u_energy = aggregate_by_variant_resolution["U"][resolution]["gated_l2_energy"]
                for variant in variants:
                    values = aggregate_by_variant_resolution[variant][resolution]
                    total_rows.append(
                        {
                            "case": case,
                            "seed": seed,
                            "resolution": resolution,
                            "variant": variant,
                            "gated_total_residual_ratio_vs_U": (
                                math.sqrt(values["gated_l2_energy"] / max(u_energy, 1e-24))
                                if u_energy is not None and values["gated_l2_energy"] is not None
                                else None
                            ),
                            "gated_total_l2_energy": values["gated_l2_energy"],
                        }
                    )

    def write_csv(name: str, rows: list[dict[str, object]]) -> None:
        path = args.output_dir / name
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("residual_by_resolution.csv", residual_rows)
    write_csv("total_residual_ratio_vs_u.csv", total_rows)
    write_csv("token_gate_coverage.csv", coverage_rows)
    write_csv("token_region_coverage.csv", region_coverage_rows)
    manifest = {
        "description": "V2.1 allocation audit from existing residual logs and effective gate snapshots",
        "model_inference": False,
        "new_images": False,
        "regions": "Exact regional raw/gated RMS are not present in current logs; token coverage and global RMS are direct.",
        "residual_aggregation": "energy-weighted RMS over residual records at each feature resolution",
        "total_ratio": "sqrt(sum(gated_l2^2_variant) / sum(gated_l2^2_U))",
        "resolutions": list(RESOLUTIONS),
        "variants": list(variants),
        "seeds": args.seeds,
        "outputs": [
            "residual_by_resolution.csv",
            "total_residual_ratio_vs_u.csv",
            "token_gate_coverage.csv",
            "token_region_coverage.csv",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"output_dir={args.output_dir}")
    print("Direct global residual audit at 16x16 (mean across requested seeds):")
    for case in CASE_DIRS:
        print(case)
        for variant in variants:
            rows = [
                row
                for row in residual_rows
                if row["case"] == case and row["variant"] == variant and row["resolution"] == 16
            ]
            rms_ratio = np.asarray([float(row["rms_ratio"]) for row in rows])
            total_ratio = np.asarray(
                [
                    float(row["gated_total_residual_ratio_vs_U"])
                    for row in total_rows
                    if row["case"] == case and row["variant"] == variant and row["resolution"] == 16
                ]
            )
            fractions = np.asarray(
                [
                    float(row["token_fraction"])
                    for row in coverage_rows
                    if row["case"] == case and row["variant"] == variant and row["resolution"] == 16
                ]
            )
            print(
                f"  {variant:12} gate={fractions.mean():.4f} "
                f"raw_to_gated={rms_ratio.mean():.4f} "
                f"total_vs_U={total_ratio.mean():.4f}"
            )


if __name__ == "__main__":
    main()
