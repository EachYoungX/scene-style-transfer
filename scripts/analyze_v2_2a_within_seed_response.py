"""Analyze V2.2a as within-seed response curves, not fixed thresholds."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAMBDA_ORDER = (0.2, 0.4, 0.6, 0.8, 1.0)
TRANSITIONS = tuple(zip(LAMBDA_ORDER[:-1], LAMBDA_ORDER[1:]))


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(row: dict, key: str) -> float:
    return float(row[key])


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((value - average) ** 2 for value in values) / (len(values) - 1))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed",
    )
    args = parser.parse_args()
    input_dir = ROOT / args.input_dir
    review_rows = read_csv(input_dir / "targeted_review_metrics_and_labels.csv")
    residual_rows = read_csv(input_dir / "targeted_residual_metrics.csv")

    review_by_group: dict[tuple[str, int], dict[float, dict]] = defaultdict(dict)
    for row in review_rows:
        review_by_group[(row["case"], int(row["seed"]))][float(row["lambda"])] = row

    residual_by_group: dict[tuple[str, int], dict[float, float]] = defaultdict(dict)
    energy_by_group: dict[tuple[str, int], dict[float, float]] = defaultdict(dict)
    for row in residual_rows:
        group = (row["case"], int(row["seed"]))
        lam = float(row["lambda"])
        energy_by_group[group].setdefault(lam, 0.0)
        energy_by_group[group][lam] += f(row, "gated_l2_energy")
    for group, values in energy_by_group.items():
        residual_by_group[group] = {lam: math.sqrt(energy) for lam, energy in values.items()}

    adjacent_rows: list[dict] = []
    sensitivity_rows: list[dict] = []
    for (case, seed), values in sorted(review_by_group.items()):
        for low, high in TRANSITIONS:
            before, after = values[low], values[high]
            adjacent_rows.append(
                {
                    "case": case,
                    "seed": seed,
                    "transition": f"{low:.1f}->{high:.1f}",
                    "delta_reference_color_similarity": f(after, "reference_color_similarity_vs_content") - f(before, "reference_color_similarity_vs_content"),
                    "delta_reference_contrast_similarity": f(after, "reference_contrast_similarity_vs_content") - f(before, "reference_contrast_similarity_vs_content"),
                    "delta_rgb_change_from_content": f(after, "rgb_mae_output_content_global") - f(before, "rgb_mae_output_content_global"),
                    "delta_edge_f1": f(after, "edge_f1_vs_content") - f(before, "edge_f1_vs_content"),
                    "delta_edge_chamfer_content_to_output": f(after, "edge_chamfer_content_to_output") - f(before, "edge_chamfer_content_to_output"),
                    "delta_residual_magnitude": residual_by_group[(case, seed)][high] - residual_by_group[(case, seed)][low],
                }
            )
        low, high = 0.2, 1.0
        before, after = values[low], values[high]
        style_color = f(after, "reference_color_similarity_vs_content") - f(before, "reference_color_similarity_vs_content")
        style_contrast = f(after, "reference_contrast_similarity_vs_content") - f(before, "reference_contrast_similarity_vs_content")
        edge_loss = f(before, "edge_f1_vs_content") - f(after, "edge_f1_vs_content")
        chamfer_increase = f(after, "edge_chamfer_content_to_output") - f(before, "edge_chamfer_content_to_output")
        normalized_chamfer = chamfer_increase / max(f(before, "edge_chamfer_content_to_output"), 1e-6)
        structure_risk_proxy = max(0.0, edge_loss) + max(0.0, normalized_chamfer)
        sensitivity_rows.append(
            {
                "case": case,
                "seed": seed,
                "style_color_sensitivity": style_color / 0.8,
                "style_contrast_sensitivity": style_contrast / 0.8,
                "content_change_sensitivity": (f(after, "rgb_mae_output_content_global") - f(before, "rgb_mae_output_content_global")) / 0.8,
                "structure_edge_loss_sensitivity": edge_loss / 0.8,
                "structure_chamfer_increase_sensitivity": chamfer_increase / 0.8,
                "structure_risk_proxy": structure_risk_proxy,
                "residual_sensitivity": (residual_by_group[(case, seed)][high] - residual_by_group[(case, seed)][low]) / 0.8,
                "manual_takeover_label": "",
                "manual_leakage_label": "",
            }
        )

    aggregate_rows: list[dict] = []
    adjacent_summary_rows: list[dict] = []
    by_case: dict[str, list[dict]] = defaultdict(list)
    for row in sensitivity_rows:
        by_case[row["case"]].append(row)
    numeric_fields = [
        "style_color_sensitivity",
        "style_contrast_sensitivity",
        "content_change_sensitivity",
        "structure_edge_loss_sensitivity",
        "structure_chamfer_increase_sensitivity",
        "structure_risk_proxy",
        "residual_sensitivity",
    ]
    for case, rows in sorted(by_case.items()):
        output = {"case": case, "seed_count": len(rows)}
        for field in numeric_fields:
            values = [float(row[field]) for row in rows]
            output[f"{field}_mean"] = mean(values)
            output[f"{field}_std"] = std(values)
        aggregate_rows.append(output)

    adjacent_numeric_fields = [
        "delta_reference_color_similarity",
        "delta_reference_contrast_similarity",
        "delta_rgb_change_from_content",
        "delta_edge_f1",
        "delta_edge_chamfer_content_to_output",
        "delta_residual_magnitude",
    ]
    adjacent_by_group: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in adjacent_rows:
        adjacent_by_group[(row["case"], row["transition"])].append(row)
    for (case, transition), rows in sorted(adjacent_by_group.items()):
        output = {"case": case, "transition": transition, "seed_count": len(rows)}
        for field in adjacent_numeric_fields:
            values = [float(row[field]) for row in rows]
            output[f"{field}_mean"] = mean(values)
            output[f"{field}_std"] = std(values)
        adjacent_summary_rows.append(output)

    write_csv(input_dir / "within_seed_adjacent_delta_metrics.csv", adjacent_rows)
    write_csv(input_dir / "within_seed_adjacent_delta_summary.csv", adjacent_summary_rows)
    write_csv(input_dir / "within_seed_sensitivity_by_seed.csv", sensitivity_rows)
    write_csv(input_dir / "within_seed_sensitivity_summary.csv", aggregate_rows)
    print(input_dir)


if __name__ == "__main__":
    main()
