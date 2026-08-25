"""Summarize the 23-pair seed42 operating-point sweep and freeze global lambda."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    from analyze_v2_4c_common_seed_features import normalize_v22, seed42_rows
    from build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, write_csv
except ModuleNotFoundError:
    from scripts.analyze_v2_4c_common_seed_features import normalize_v22, seed42_rows
    from scripts.build_v2_4_pair_preflight_analysis import ROOT, numeric, read_csv, write_csv


V22_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed/human_sensitivity_annotations.csv"
V23_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_3_pair_response_profiles/audits/human_sensitivity_annotations.csv"
V24B_LABELS = ROOT / "runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/audits/human_sensitivity_annotations.csv"
LAMBDAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def load_labels() -> list[dict[str, str]]:
    rows = normalize_v22(seed42_rows(V22_LABELS))
    rows.extend(seed42_rows(V23_LABELS))
    rows.extend(seed42_rows(V24B_LABELS))
    if len(rows) != 23 * 5:
        raise ValueError(f"Expected 115 common-seed rows, got {len(rows)}")
    keys = {(row["case"], float(row["lambda"])) for row in rows}
    if len(keys) != 23 * 5:
        raise ValueError("Common-seed lambda table does not contain exactly 23 × 5 unique rows")
    return rows


def values(rows: list[dict[str, str]], field: str, lam: float, *, valid_only: bool = False) -> list[float]:
    output = []
    for row in rows:
        if float(row["lambda"]) != lam:
            continue
        if valid_only and row.get("style_valid", "").lower() != "true":
            continue
        value = numeric(row.get(field))
        if value is not None:
            output.append(value)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="analysis/v2_4e_global_lambda_summary.csv")
    parser.add_argument("--decision", default="analysis/v2_4e_global_lambda_decision.json")
    parser.add_argument("--active-risk-threshold", type=float, default=0.25)
    parser.add_argument("--severe-threshold", type=float, default=0.20)
    args = parser.parse_args()

    rows = load_labels()
    baseline = values(rows, "baseline_takeover_0_3", 0.2)
    baseline_risk_rate = float(np.mean(np.asarray(baseline) > 0))
    summary = []
    for lam in LAMBDAS:
        lam_rows = [row for row in rows if float(row["lambda"]) == lam]
        valid_rows = [row for row in lam_rows if row.get("style_valid", "").lower() == "true"]
        style_scores = values(rows, "human_style_score_0_4", lam, valid_only=True)
        style_ge2 = int(sum(score >= 2 for score in style_scores))
        active_field = "baseline_takeover_0_3" if lam == 0.2 else "incremental_takeover_0_3"
        active = values(rows, active_field, lam)
        risk_count = int(sum(score > 0 for score in active))
        severe_count = int(sum(score >= 2 for score in active))
        summary.append(
            {
                "lambda": f"{lam:.1f}",
                "n_pairs": len(lam_rows),
                "n_style_valid": len(valid_rows),
                "valid_style_rate": len(valid_rows) / len(lam_rows),
                "median_style_score_valid": float(np.median(style_scores)) if style_scores else "",
                "style_ge2_fraction_valid": style_ge2 / len(style_scores) if style_scores else "",
                "effective_style_coverage": style_ge2 / len(lam_rows),
                "baseline_takeover_rate": baseline_risk_rate,
                "active_risk_field": active_field,
                "active_takeover_rate": risk_count / len(lam_rows),
                "severe_takeover_rate": severe_count / len(lam_rows),
                "n_active_risk": risk_count,
                "n_severe_takeover": severe_count,
                "selection_eligible": (
                    risk_count / len(lam_rows) <= args.active_risk_threshold
                    and severe_count / len(lam_rows) <= args.severe_threshold
                ),
            }
        )

    eligible = [row for row in summary if row["selection_eligible"]]
    if not eligible:
        raise ValueError(f"No lambda satisfies severe takeover threshold {args.severe_threshold}")
    max_coverage = max(float(row["effective_style_coverage"]) for row in eligible)
    selected = min(
        (row for row in eligible if float(row["effective_style_coverage"]) == max_coverage),
        key=lambda row: float(row["lambda"]),
    )
    decision = {
        "method": "A2_fixed",
        "selected_lambda": float(selected["lambda"]),
        "selection_rule": "Among lambdas with active_takeover_rate <= active-risk threshold and severe_takeover_rate <= severe threshold, maximize effective_style_coverage (style_valid AND style_score>=2), then choose the smallest lambda.",
        "active_risk_threshold": args.active_risk_threshold,
        "severe_takeover_threshold": args.severe_threshold,
        "selected_row": selected,
        "candidate_rows": summary,
        "pair_count": 23,
        "seed_scope": "seed42_common_screening",
        "style_invalid_excluded_from_style_score": True,
    }
    write_csv(ROOT / args.output, summary)
    (ROOT / args.decision).write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(ROOT / args.output)
    print(ROOT / args.decision)
    print(f"selected_lambda={selected['lambda']}")


if __name__ == "__main__":
    main()
