"""Confirm V2.4c feature directions on the existing multiseed subset only."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

try:
    from analyze_v2_4c_common_seed_features import normalize_v22, pair_profile
    from build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, spearman, write_csv
except ModuleNotFoundError:
    from scripts.analyze_v2_4c_common_seed_features import normalize_v22, pair_profile
    from scripts.build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, spearman, write_csv


COMMON = ROOT / "analysis/v2_4c_common_seed_profiles.csv"
V22_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed/human_sensitivity_annotations.csv"
V23_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/representative_multiseed/human_sensitivity_annotations.csv"

TARGETS = (
    "style_valid",
    "baseline_takeover_02",
    "style_gain_if_valid",
    "incremental_takeover_max",
    "incremental_nonzero_count",
    "late_escalation",
)


def target_value(row: dict[str, object], target: str) -> float | None:
    if target == "style_valid":
        return float(row["style_valid"])
    return numeric(row.get(target))


def correlations(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    excluded = {
        "case",
        "content_path",
        "reference_path",
        "content_family",
        "reference_family",
        "label_source",
        "seed_scope",
        "feature_source",
        "generation",
        "profile_label",
        "label_status",
    }
    feature_fields = [
        field
        for field in rows[0]
        if field not in excluded
        and field not in TARGETS
        and field not in {"seed_count", "style_valid_rate"}
        and not field.endswith("_median")
        and not field.endswith("_max")
        and not field.endswith("_frequency")
        and not field.endswith("_count")
        and not field.endswith("_02")
        and not field.endswith("_if_valid")
        and not field.endswith("_status")
        and not field.endswith("_takeover")
        and not field.endswith("_gain")
    ]
    output = []
    for target in TARGETS:
        subset = rows if target != "style_gain_if_valid" else [row for row in rows if row["style_valid"] == 1]
        for feature in feature_fields:
            pairs = [(target_value(row, target), numeric(row.get(feature))) for row in subset]
            pairs = [(target_, feature_) for target_, feature_ in pairs if target_ is not None and feature_ is not None]
            output.append(
                {
                    "target": target,
                    "subset": "multiseed_9_pairs" if target != "style_gain_if_valid" else "multiseed_style_valid_true_only",
                    "feature": feature,
                    "n": len(pairs),
                    "spearman_rho": spearman([pair[1] for pair in pairs], [pair[0] for pair in pairs]),
                }
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis/v2_4c_multiseed_profiles.csv")
    parser.add_argument("--correlations", default="analysis/v2_4c_multiseed_correlations.csv")
    parser.add_argument("--comparison", default="analysis/v2_4c_direction_comparison.csv")
    args = parser.parse_args()

    common_rows = {row["case"]: row for row in read_csv(COMMON)}
    labels = {}
    for row in normalize_v22(read_csv(V22_LABELS)):
        labels.setdefault(row["case"], []).append(row)
    for row in read_csv(V23_LABELS):
        labels.setdefault(row["case"], []).append(row)
    if not labels or not all(len(values) >= 15 for values in labels.values()):
        raise ValueError("Multiseed confirmation requires at least 3 seeds × 5 lambdas per pair")

    profiles = []
    for case in sorted(labels):
        if case not in common_rows:
            raise ValueError(f"Multiseed case missing from common feature table: {case}")
        source = common_rows[case]
        profile = pair_profile(labels[case])
        profiles.append({**source, **profile, "seed_scope": "multiseed_confirmation"})
    write_csv(ROOT / args.output, profiles)

    multi_corr = correlations(profiles)
    write_csv(ROOT / args.correlations, multi_corr)
    common_corr = read_csv(ROOT / "analysis/v2_4c_common_seed_correlations.csv")
    comparison = []
    for left in common_corr:
        matches = [
            row
            for row in multi_corr
            if row["target"] == left["target"] and row["feature"] == left["feature"]
        ]
        right = matches[0] if matches else {}
        common_rho = numeric(left.get("spearman_rho"))
        multi_rho = numeric(right.get("spearman_rho"))
        comparison.append(
            {
                "target": left["target"],
                "feature": left["feature"],
                "common_n": left["n"],
                "common_rho": common_rho if common_rho is not None else "",
                "multiseed_n": right.get("n", ""),
                "multiseed_rho": multi_rho if multi_rho is not None else "",
                "same_nonzero_direction": int(common_rho * multi_rho > 0) if common_rho is not None and multi_rho is not None else "",
            }
        )
    write_csv(ROOT / args.comparison, comparison)
    print(ROOT / args.output)
    print(ROOT / args.correlations)
    print(ROOT / args.comparison)


if __name__ == "__main__":
    main()
