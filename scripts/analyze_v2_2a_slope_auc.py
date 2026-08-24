"""Compute objective and optional human V2.2a response slopes/AUCs."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def slope_auc(xs: list[float], ys: list[float]) -> tuple[float, float]:
    x_mean, y_mean = mean(xs), mean(ys)
    denominator = sum((x - x_mean) ** 2 for x in xs)
    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / max(denominator, 1e-12)
    auc = sum((xs[index + 1] - xs[index]) * (ys[index] + ys[index + 1]) / 2.0 for index in range(len(xs) - 1)) / (xs[-1] - xs[0])
    return slope, auc


def spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda index: values[index])
        output = [0.0] * len(values)
        position = 0
        while position < len(order):
            end = position
            while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
                end += 1
            rank = (position + end) / 2.0 + 1.0
            for index in range(position, end + 1):
                output[order[index]] = rank
            position = end + 1
        return output

    xr, yr = ranks(x), ranks(y)
    xm, ym = mean(xr), mean(yr)
    numerator = sum((a - xm) * (b - ym) for a, b in zip(xr, yr))
    denominator = math.sqrt(sum((a - xm) ** 2 for a in xr) * sum((b - ym) ** 2 for b in yr))
    return numerator / denominator if denominator else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed")
    args = parser.parse_args()
    input_dir = ROOT / args.input_dir
    review_rows = read_csv(input_dir / "targeted_review_metrics_and_labels.csv")
    residual_rows = read_csv(input_dir / "targeted_residual_metrics.csv")
    annotations_path = input_dir / "human_sensitivity_annotations.csv"
    annotations = read_csv(annotations_path) if annotations_path.exists() else []

    grouped: dict[tuple[str, int], dict[float, dict]] = defaultdict(dict)
    for row in review_rows:
        grouped[(row["case"], int(row["seed"]))][float(row["lambda"])] = row
    residual_energy: dict[tuple[str, int], dict[float, float]] = defaultdict(lambda: defaultdict(float))
    for row in residual_rows:
        residual_energy[(row["case"], int(row["seed"]))][float(row["lambda"])] += float(row["gated_l2_energy"])

    objective_specs = {
        "rgb_response": lambda row, group: float(row["rgb_mae_output_content_global"]),
        "edge_loss": lambda row, group: 1.0 - float(row["edge_f1_vs_content"]),
        "edge_chamfer": lambda row, group: float(row["edge_chamfer_content_to_output"]),
        "reference_color_proxy": lambda row, group: float(row["reference_color_similarity_vs_content"]),
        "reference_contrast_proxy": lambda row, group: float(row["reference_contrast_similarity_vs_content"]),
        "ip_residual_magnitude": lambda row, group: math.sqrt(residual_energy[group][float(row["lambda"])]),
    }
    per_seed_rows: list[dict] = []
    for group, values in sorted(grouped.items()):
        case, seed = group
        row = {"case": case, "seed": seed}
        xs = list(LAMBDAS)
        for name, getter in objective_specs.items():
            slope, auc = slope_auc(xs, [getter(values[lam], group) for lam in LAMBDAS])
            row[f"{name}_slope"] = slope
            row[f"{name}_auc"] = auc
        if annotations:
            annotation_by_lambda = {
                float(item["lambda"]): item
                for item in annotations
                if item["case"] == case and int(item["seed"]) == seed
            }
            for name, field in (("human_style", "human_style_score_0_4"), ("human_takeover", "human_takeover_score_0_3")):
                scores = [annotation_by_lambda[lam][field] for lam in LAMBDAS]
                if all(score.strip() for score in scores):
                    slope, auc = slope_auc(xs, [float(score) for score in scores])
                    row[f"{name}_slope"] = slope
                    row[f"{name}_auc"] = auc
                else:
                    row[f"{name}_slope"] = ""
                    row[f"{name}_auc"] = ""
        else:
            row.update({"human_style_slope": "", "human_style_auc": "", "human_takeover_slope": "", "human_takeover_auc": ""})
        per_seed_rows.append(row)

    objective_fields = [key for key in per_seed_rows[0] if key not in {"case", "seed"} and not key.startswith("human_")]
    summary_rows: list[dict] = []
    for case in sorted({row["case"] for row in per_seed_rows}):
        rows = [row for row in per_seed_rows if row["case"] == case]
        output = {"case": case, "seed_count": len(rows)}
        for field in objective_fields + ["human_style_slope", "human_style_auc", "human_takeover_slope", "human_takeover_auc"]:
            values = [float(row[field]) for row in rows if str(row.get(field, "")).strip()]
            output[f"{field}_mean"] = mean(values) if values else ""
            output[f"{field}_std"] = std(values) if values else ""
        summary_rows.append(output)

    correlation_rows: list[dict] = []
    for objective in objective_specs:
        for target in ("human_takeover_slope", "human_takeover_auc", "human_style_slope", "human_style_auc"):
            pairs = [(float(row[f"{objective}_slope"]), float(row[target])) for row in per_seed_rows if str(row.get(target, "")).strip()]
            x, y = [pair[0] for pair in pairs], [pair[1] for pair in pairs]
            correlation_rows.append({"objective": objective, "target": target, "n": len(pairs), "spearman_rho": spearman(x, y) if len(pairs) >= 3 else "", "status": "ready" if len(pairs) >= 3 else "pending_human_scores"})

    write_csv(input_dir / "slope_auc_by_seed.csv", per_seed_rows)
    write_csv(input_dir / "slope_auc_summary.csv", summary_rows)
    write_csv(input_dir / "objective_human_spearman.csv", correlation_rows)
    print(input_dir)


if __name__ == "__main__":
    main()
