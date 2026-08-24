"""Evaluate V2.3 seed42 pair-response profiles."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from evaluate_v2_2a_safe_strength_frontier import aggregate, edge_metrics, reference_style_proxy, rgb_metrics  # noqa: E402

LAMBDAS = (0.2, 0.4, 0.6, 0.8, 1.0)
INVALID_STYLE_CASES = {"compat_G4_city_mismatch"}


def read_cases(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def image(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)


def valid_eval_mask(raw_path: Path, size: int = 512, erosion_radius: int = 2) -> np.ndarray:
    source = Image.open(raw_path).convert("RGB")
    source.thumbnail((size, size), Image.Resampling.LANCZOS)
    mask = np.zeros((size, size), dtype=np.uint8)
    left = (size - source.width) // 2
    top = (size - source.height) // 2
    mask[top : top + source.height, left : left + source.width] = 1
    if erosion_radius:
        kernel = np.ones((2 * erosion_radius + 1, 2 * erosion_radius + 1), dtype=np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
    return mask.astype(bool)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def slope_auc(xs: list[float], ys: list[float]) -> tuple[float, float]:
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / max(sum((x - xm) ** 2 for x in xs), 1e-12)
    auc = sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)) / (xs[-1] - xs[0])
    return slope, auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/v2_3_pair_response_profiles.csv")
    parser.add_argument("--output-root", default="runs/ip_adapter_plus_injection/v2_3_pair_response_profiles")
    parser.add_argument("--audit-subdir", default="audits")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--seed", type=int, action="append")
    args = parser.parse_args()
    args.seed = args.seed or [42]
    cases = read_cases(ROOT / args.manifest)
    if args.case_id:
        cases = [case for case in cases if case["canonical_case_id"] in set(args.case_id)]
    output_root = ROOT / args.output_root
    audits = output_root / args.audit_subdir
    audits.mkdir(parents=True, exist_ok=True)
    residual_rows: list[dict] = []
    review_rows: list[dict] = []
    for case in cases:
        case_id = case["canonical_case_id"]
        for seed in args.seed:
            case_root = output_root / case_id / f"seed{seed}"
            content = image(case_root / "content.png")
            reference = image(case_root / "style.png")
            valid = valid_eval_mask(ROOT / case["content_path"])
            regions = {"global": valid}
            for multiplier in LAMBDAS:
                label = f"lambda_{multiplier:.1f}".replace(".", "p")
                run_dir = case_root / label
                records = [json.loads(line) for line in (run_dir / "residuals.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
                output = image(run_dir / "output.png")
                for resolution in (64, 32, 16):
                    residual_rows.append({"case": case_id, "seed": seed, "lambda": multiplier, "resolution": resolution, **aggregate(records, resolution)})
                review_rows.append(
                    {
                        "case": case_id,
                        "seed": seed,
                        "lambda": multiplier,
                        **rgb_metrics(output, content, regions),
                        **edge_metrics(output, content, valid),
                        **reference_style_proxy(output, content, reference),
                        "human_style_score_0_4": "",
                        "baseline_takeover_0_3": "" if multiplier == 0.2 else "NA",
                        "incremental_takeover_0_3": "NA" if multiplier == 0.2 else "",
                        "style_valid": "false" if case_id in INVALID_STYLE_CASES else "true",
                        "reference": "",
                        "review_note": "",
                    }
                )
    for case in cases:
        case_id = case["canonical_case_id"]
        for seed in args.seed:
            rows = [row for row in review_rows if row["case"] == case_id and row["seed"] == seed]
            for metric in ("rgb_mae_output_content_global", "edge_f1_vs_content", "edge_chamfer_content_to_output"):
                slope, auc = slope_auc(list(LAMBDAS), [float(row[metric]) for row in rows])
                for row in rows:
                    row[f"{metric}_slope"] = slope
                    row[f"{metric}_auc"] = auc
    write_csv(audits / "response_metrics.csv", review_rows)
    write_csv(audits / "residual_metrics.csv", residual_rows)
    slope_rows = []
    for case_id in [case["canonical_case_id"] for case in cases]:
        for seed in args.seed:
            rows = [row for row in review_rows if row["case"] == case_id and row["seed"] == seed]
            slope_rows.append(
                {
                    "case": case_id,
                    "seed": seed,
                    "seed_count": 1,
                    "rgb_response_slope": rows[0]["rgb_mae_output_content_global_slope"],
                    "rgb_response_auc": rows[0]["rgb_mae_output_content_global_auc"],
                    "edge_f1_slope": rows[0]["edge_f1_vs_content_slope"],
                    "edge_f1_auc": rows[0]["edge_f1_vs_content_auc"],
                    "edge_chamfer_slope": rows[0]["edge_chamfer_content_to_output_slope"],
                    "edge_chamfer_auc": rows[0]["edge_chamfer_content_to_output_auc"],
                }
            )
    write_csv(audits / "slope_auc.csv", slope_rows)
    (audits / "manifest.json").write_text(json.dumps({"pairs": [case["canonical_case_id"] for case in cases], "seeds": args.seed, "lambdas": list(LAMBDAS), "valid_eval_erosion_radius": 2, "human_scores_pending": True}, indent=2), encoding="utf-8")
    print(audits)


if __name__ == "__main__":
    main()
