"""Run small family-holdout feasibility probes on the 23-pair V2.4c table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

try:
    from build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, write_csv
except ModuleNotFoundError:
    from scripts.build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, write_csv


ANALYSIS = ROOT / "analysis/v2_4c_common_seed_profiles.csv"
TARGET_CONFIG = ROOT / "configs/experiment/v2_4c_feature_targets.csv"


def sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -40, 40)))


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 2500, learning_rate: float = 0.08, l2: float = 0.1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    standardized = (x - mean) / scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    weights = np.zeros(design.shape[1], dtype=float)
    for _ in range(steps):
        probability = sigmoid(design @ weights)
        gradient = (design.T @ (probability - y)) / len(y)
        gradient[1:] += l2 * weights[1:]
        weights -= learning_rate * gradient
    return weights, mean, scale


def predict_logistic(weights: np.ndarray, mean: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
    standardized = (x - mean) / scale
    return sigmoid(np.column_stack([np.ones(len(standardized)), standardized]) @ weights)


def balanced_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float | str:
    values = []
    for label in (0, 1):
        mask = actual == label
        if mask.any():
            values.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(values)) if len(values) == 2 else ""


def labels_for(target: str, rows: list[dict[str, str]]) -> list[int | None]:
    labels: list[int | None] = []
    for row in rows:
        if target == "style_viability":
            labels.append(int(float(row["style_valid"])))
            continue
        if target == "initial_susceptibility":
            value = numeric(row.get("baseline_takeover_02"))
            labels.append(0 if value == 0 else 1 if value is not None and value >= 2 else None)
            continue
        if target == "style_responsiveness":
            value = numeric(row.get("style_gain_if_valid"))
            labels.append(0 if value is not None and value <= 2 else 1 if value is not None and value >= 3 else None)
            continue
        if target == "pressure_escalation":
            value = numeric(row.get("late_escalation"))
            labels.append(0 if value is not None and value < 0.5 else 1 if value is not None else None)
            continue
        raise ValueError(target)
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis/v2_4d_feasibility_probe.csv")
    parser.add_argument("--summary", default="analysis/v2_4d_feasibility_summary.csv")
    args = parser.parse_args()

    rows = read_csv(ANALYSIS)
    config = read_csv(TARGET_CONFIG)
    features_by_target: dict[str, list[str]] = {}
    for row in config:
        features_by_target.setdefault(row["target"], []).append(row["feature"])

    split_groups = {
        "leave_one_content_family_out": [row["content_family"] for row in rows],
        "leave_one_reference_family_out": [row["reference_family"] for row in rows],
    }
    output_rows = []
    for target, selected_features in features_by_target.items():
        labels = labels_for(target, rows)
        for feature_count in range(1, min(3, len(selected_features)) + 1):
            chosen = selected_features[:feature_count]
            usable = [
                index
                for index, label in enumerate(labels)
                if label is not None and all(numeric(rows[index].get(feature)) is not None for feature in chosen)
            ]
            if len({labels[index] for index in usable}) < 2:
                continue
            for split_name, groups in split_groups.items():
                for held_out in sorted(set(groups)):
                    test_indices = [index for index in usable if groups[index] == held_out]
                    train_indices = [index for index in usable if groups[index] != held_out]
                    if not test_indices or len({labels[index] for index in train_indices}) < 2:
                        continue
                    x_train = np.asarray([[float(numeric(rows[index].get(feature))) for feature in chosen] for index in train_indices])
                    y_train = np.asarray([labels[index] for index in train_indices], dtype=float)
                    x_test = np.asarray([[float(numeric(rows[index].get(feature))) for feature in chosen] for index in test_indices])
                    y_test = np.asarray([labels[index] for index in test_indices], dtype=int)
                    weights, mean, scale = fit_logistic(x_train, y_train)
                    probabilities = predict_logistic(weights, mean, scale, x_test)
                    predictions = (probabilities >= 0.5).astype(int)
                    output_rows.append(
                        {
                            "target": target,
                            "feature_count": feature_count,
                            "features": "+".join(chosen),
                            "split": split_name,
                            "held_out_group": held_out,
                            "n_usable": len(usable),
                            "n_train": len(train_indices),
                            "n_test": len(test_indices),
                            "test_positive_rate": float(np.mean(y_test)),
                            "accuracy": float(np.mean(predictions == y_test)),
                            "balanced_accuracy": balanced_accuracy(y_test, predictions),
                            "mean_positive_probability": float(np.mean(probabilities)),
                        }
                    )

    write_csv(ROOT / args.output, output_rows)
    summary_rows = []
    for target in sorted({row["target"] for row in output_rows}):
        for split in sorted({row["split"] for row in output_rows if row["target"] == target}):
            values = [row for row in output_rows if row["target"] == target and row["split"] == split and row["balanced_accuracy"] != ""]
            if not values:
                continue
            best = max(values, key=lambda row: float(row["balanced_accuracy"]))
            summary_rows.append(
                {
                    "target": target,
                    "split": split,
                    "n_evaluated": len(values),
                    "best_features": best["features"],
                    "best_balanced_accuracy": best["balanced_accuracy"],
                    "median_balanced_accuracy": float(np.median([float(row["balanced_accuracy"]) for row in values])),
                    "warning": "exploratory only; family holdout groups are tiny",
                }
            )
    write_csv(ROOT / args.summary, summary_rows)
    print(ROOT / args.output)
    print(ROOT / args.summary)


if __name__ == "__main__":
    main()
