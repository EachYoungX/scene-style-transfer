import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.build_v2_4_pair_preflight_analysis import (
    aggregate_human,
    average_ranks,
    normalize_legacy_v22_rows,
    profile_label,
)


def test_average_ranks_uses_average_ties():
    assert average_ranks([10, 10, 20]).tolist() == [0.5, 0.5, 2.0]


def test_aggregate_human_keeps_baseline_separate_from_incremental_scores():
    rows = []
    for seed in ("42", "123", "777"):
        rows.extend(
            [
                {
                    "case": "pair",
                    "seed": seed,
                    "lambda": "0.2",
                    "human_style_score_0_4": "1",
                    "baseline_takeover_0_3": "0",
                    "incremental_takeover_0_3": "NA",
                    "style_valid": "TRUE",
                },
                {
                    "case": "pair",
                    "seed": seed,
                    "lambda": "0.4",
                    "human_style_score_0_4": "2",
                    "baseline_takeover_0_3": "NA",
                    "incremental_takeover_0_3": "1",
                    "style_valid": "TRUE",
                },
                {
                    "case": "pair",
                    "seed": seed,
                    "lambda": "1.0",
                    "human_style_score_0_4": "4",
                    "baseline_takeover_0_3": "NA",
                    "incremental_takeover_0_3": "0",
                    "style_valid": "TRUE",
                },
            ]
        )

    result = aggregate_human(rows)

    assert result["seed_count"] == 3
    assert result["baseline_takeover_median"] == 0.0
    assert result["style_gain_median"] == 3.0
    assert result["incremental_takeover_max_median"] == 1.0
    assert result["incremental_nonzero_interval_count"] == 1.0
    assert result["label_status"] == "complete"
    assert profile_label(result) == "P1_low_risk_high_response"


def test_normalize_legacy_v22_rows_splits_baseline_and_incremental_takeover():
    rows = [
        {
            "case": "pair",
            "seed": "42",
            "lambda": "0.2",
            "human_style_score_0_4": "1",
            "human_takeover_score_0_3": "2",
            "human_reference_leakage_note": "",
            "human_review_note": "",
        },
        {
            "case": "pair",
            "seed": "42",
            "lambda": "0.4",
            "human_style_score_0_4": "2",
            "human_takeover_score_0_3": "1",
            "human_reference_leakage_note": "",
            "human_review_note": "",
        },
    ]

    normalized = normalize_legacy_v22_rows(rows)

    assert normalized[0]["baseline_takeover_0_3"] == "2"
    assert normalized[0]["incremental_takeover_0_3"] == "NA"
    assert normalized[1]["baseline_takeover_0_3"] == "NA"
    assert normalized[1]["incremental_takeover_0_3"] == "1"
