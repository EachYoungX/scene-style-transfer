"""Run a deliberately small leave-group-out feasibility probe for V2.4b."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

try:
    from build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, write_csv
except ModuleNotFoundError:
    from scripts.build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, write_csv


ANALYSIS = ROOT / "analysis/v2_4_pair_feature_analysis.csv"
SHORTLIST = ROOT / "configs/experiment/v2_4b_feature_shortlist.csv"


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
        if not mask.any():
            continue
        values.append(float(np.mean(predicted[mask] == label)))
    return float(np.mean(values)) if len(values) == 2 else ""


def group_name(path: str, root: str) -> str:
    relative = Path(path).as_posix()
    if relative.startswith(root):
        relative = relative[len(root) :]
    return relative


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis/v2_4b_feasibility_probe.csv")
    args = parser.parse_args()
    rows = read_csv(ANALYSIS)
    shortlist = read_csv(SHORTLIST)
    features_by_target: dict[str, list[str]] = {}
    for row in shortlist:
        features_by_target.setdefault(row["target"], []).append(row["feature"])

    target_values: dict[str, list[float | None]] = {
        "initial_susceptibility": [numeric(row.get("baseline_takeover_median")) for row in rows],
        "style_responsiveness": [numeric(row.get("style_gain_median")) for row in rows],
        "pressure_escalation": [numeric(row.get("late_escalation_frequency")) for row in rows],
    }
    labels: dict[str, list[int | None]] = {
        "initial_susceptibility": [0 if value == 0 else 1 if value is not None and value >= 2 else None for value in target_values["initial_susceptibility"]],
        "style_responsiveness": [0 if value is not None and value <= 2 else 1 if value is not None and value >= 3 else None for value in target_values["style_responsiveness"]],
        "pressure_escalation": [0 if value is not None and value < 0.5 else 1 if value is not None else None for value in target_values["pressure_escalation"]],
    }
    groups = {
        "leave_one_content_family_out": [group_name(row["content_path"], "data/raw/_photo_ref/") for row in rows],
        "leave_one_reference_family_out": [Path(row["reference_path"]).parent.name for row in rows],
    }
    output_rows = []
    for target, selected_features in features_by_target.items():
        for feature_count in range(1, len(selected_features) + 1):
            chosen = selected_features[:feature_count]
            usable = [index for index, label in enumerate(labels[target]) if label is not None and all(numeric(rows[index].get(feature)) is not None for feature in chosen)]
            if len({labels[target][index] for index in usable}) < 2:
                continue
            for split_name, split_groups in groups.items():
                for held_out in sorted(set(split_groups)):
                    test_indices = [index for index in usable if split_groups[index] == held_out]
                    train_indices = [index for index in usable if split_groups[index] != held_out]
                    if not test_indices or len({labels[target][index] for index in train_indices}) < 2:
                        continue
                    x_train = np.asarray([[float(numeric(rows[index].get(feature))) for feature in chosen] for index in train_indices])
                    y_train = np.asarray([labels[target][index] for index in train_indices], dtype=float)
                    x_test = np.asarray([[float(numeric(rows[index].get(feature))) for feature in chosen] for index in test_indices])
                    y_test = np.asarray([labels[target][index] for index in test_indices], dtype=int)
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
                            "n_train": len(train_indices),
                            "n_test": len(test_indices),
                            "test_positive_rate": float(np.mean(y_test)),
                            "accuracy": float(np.mean(predictions == y_test)),
                            "balanced_accuracy": balanced_accuracy(y_test, predictions),
                            "mean_positive_probability": float(np.mean(probabilities)),
                        }
                    )
    write_csv(ROOT / args.output, output_rows)
    print(ROOT / args.output)


if __name__ == "__main__":
    main()
