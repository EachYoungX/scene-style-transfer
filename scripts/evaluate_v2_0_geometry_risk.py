"""Evaluate frozen R0 risk maps against completed V2.0 annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics.geometry_risk_metrics import (  # noqa: E402
    binary_risk_metrics,
    continuous_risk_metrics,
    threshold_risk,
    top_fraction_risk,
)
from metrics.mask_utils import (  # noqa: E402
    binary_mask,
    load_continuous_risk,
    load_mask,
    valid_mask,
    validate_alignment,
)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return sorted(csv.DictReader(handle), key=lambda row: row["sample_id"])


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    args = parser.parse_args()
    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    risk_config = config["risk_map"]
    annotation_config = config["annotations"]
    output_root = ROOT / experiment["output_root"]
    rows = read_manifest(ROOT / experiment["annotation_manifest"])
    result_root = output_root / "evaluation"
    result_root.mkdir(parents=True, exist_ok=True)

    threshold_rows: list[dict[str, object]] = []
    continuous_rows: list[dict[str, object]] = []
    missing: list[str] = []
    incomplete: list[str] = []
    for row in rows:
        status_fields = (
            "rigid_status",
            "soft_status",
            "geometry_failure_status",
            "uncertainty_status",
        )
        if any(row.get(field) != "complete" for field in status_fields):
            incomplete.append(row["sample_id"])
            continue
        required = [
            ROOT / row["risk_path"],
            ROOT / row["rigid_structure_mask"],
            ROOT / row["geometry_failure_mask"],
            ROOT / row["soft_stylization_mask"],
        ]
        if not all(path.exists() for path in required):
            missing.append(row["sample_id"])
            continue
        risk = load_continuous_risk(required[0])
        rigid_raw = load_mask(required[1])
        failure_raw = load_mask(required[2])
        soft_raw = load_mask(required[3])
        uncertainty_path = ROOT / row["uncertainty_mask"]
        uncertainty = load_mask(uncertainty_path) if uncertainty_path.exists() else None
        validate_alignment(
            row["sample_id"], risk=risk, rigid=rigid_raw, failure=failure_raw, soft=soft_raw, uncertainty=uncertainty
        )
        valid = valid_mask(risk.shape, uncertainty, float(annotation_config["uncertainty_threshold"]))
        mask_threshold = float(annotation_config["mask_threshold"])
        rigid = binary_mask(rigid_raw, mask_threshold)
        failure = binary_mask(failure_raw, mask_threshold)
        soft = binary_mask(soft_raw, mask_threshold)
        continuous = continuous_risk_metrics(risk, failure, valid)
        continuous_rows.append(
            {
                "sample_id": row["sample_id"],
                "case_id": row["canonical_case_id"],
                "seed": row["seed"],
                "scene_type": row["scene_type"],
                "content_rigidity": row["content_rigidity"],
                **continuous.to_dict(),
            }
        )

        selections: list[tuple[str, np.ndarray, float]] = []
        for threshold in risk_config["fixed_thresholds"]:
            selections.append((f"fixed_{float(threshold):.2f}", threshold_risk(risk, float(threshold)), float(threshold)))
        for fraction in risk_config["top_fractions"]:
            selected, cutoff = top_fraction_risk(risk, float(fraction), valid)
            selections.append((f"top_{round(float(fraction) * 100):02d}pct", selected, cutoff))
        for label, prediction, cutoff in selections:
            metrics = binary_risk_metrics(prediction, failure, soft, rigid, valid)
            threshold_rows.append(
                {
                    "sample_id": row["sample_id"],
                    "case_id": row["canonical_case_id"],
                    "seed": row["seed"],
                    "threshold_label": label,
                    "effective_cutoff": cutoff,
                    **metrics.to_dict(),
                }
            )

    if incomplete:
        raise RuntimeError(
            "Annotations are not marked complete for: "
            + ", ".join(incomplete)
            + ". Mark all four status columns complete only after reviewing the masks."
        )
    if missing:
        raise FileNotFoundError(
            "Annotations are incomplete for: " + ", ".join(missing) + ". Complete all frozen samples before evaluation."
        )
    write_csv(result_root / "per_sample_continuous.csv", continuous_rows)
    write_csv(result_root / "per_sample_threshold.csv", threshold_rows)

    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in threshold_rows:
        grouped[(str(row["case_id"]), str(row["threshold_label"]))].append(row)
    pair_rows: list[dict[str, object]] = []
    metric_names = ("failure_coverage", "risk_precision", "failure_iou", "soft_fpr", "rigid_recall")
    for (case_id, label), group in sorted(grouped.items()):
        pair_rows.append(
            {
                "case_id": case_id,
                "threshold_label": label,
                "sample_count": len(group),
                **{name: finite_mean([float(item[name]) for item in group]) for name in metric_names},
            }
        )
    write_csv(result_root / "pair_threshold_summary.csv", pair_rows)
    summary = {
        "sample_count": len(continuous_rows),
        "mean_risk_separation": finite_mean([float(row["mean_difference"]) for row in continuous_rows]),
        "mean_auroc": finite_mean([float(row["auroc"]) for row in continuous_rows]),
        "mean_auprc": finite_mean([float(row["auprc"]) for row in continuous_rows]),
        "mean_positive_prevalence": finite_mean([float(row["positive_prevalence"]) for row in continuous_rows]),
    }
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
