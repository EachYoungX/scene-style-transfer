"""Prepare the blank V2.4b seed42 human annotation table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAMBDAS = (0.2, 0.4, 0.6, 0.8, 1.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/v2_4b_targeted_profile_candidates.csv")
    parser.add_argument(
        "--output",
        default="runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/audits/human_sensitivity_annotations.csv",
    )
    args = parser.parse_args()

    with (ROOT / args.manifest).open(newline="", encoding="utf-8") as handle:
        cases = list(csv.DictReader(handle))
    rows = []
    for case in cases:
        for value in LAMBDAS:
            rows.append(
                {
                    "case": case["case_id"],
                    "content_family": case["content_family"],
                    "reference_family": case["reference_family"],
                    "seed": case["seed"],
                    "lambda": f"{value:.1f}",
                    "human_style_score_0_4": "",
                    "baseline_takeover_0_3": "",
                    "incremental_takeover_0_3": "",
                    "style_valid": "",
                    "reference_leakage_note": "",
                    "review_note": "",
                }
            )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
