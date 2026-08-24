"""Prepare the unified V2.3 human review table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INVALID_STYLE_CASES = {"compat_G4_city_mismatch"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed/targeted_review_metrics_and_labels.csv",
    )
    parser.add_argument(
        "--output",
        default="runs/ip_adapter_plus_injection/v2_2a_safe_strength_frontier/audits/targeted_multiseed/human_sensitivity_annotations.csv",
    )
    args = parser.parse_args()
    rows = []
    with (ROOT / args.input).open(newline="", encoding="utf-8") as handle:
        for source in csv.DictReader(handle):
            lam = float(source["lambda"])
            rows.append(
                {
                    "case": source["case"],
                    "seed": source["seed"],
                    "lambda": source["lambda"],
                    "human_style_score_0_4": "",
                    "baseline_takeover_0_3": "",
                    "incremental_takeover_0_3": "",
                    "style_valid": "false" if source["case"] in INVALID_STYLE_CASES else "true",
                    "reference": source.get("reference", ""),
                    "review_note": source.get("note", ""),
                }
            )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(output)


if __name__ == "__main__":
    main()
