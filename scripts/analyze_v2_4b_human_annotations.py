"""Aggregate completed V2.4b seed42 human annotations into pair profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from build_v2_4_pair_preflight_analysis import aggregate_human, profile_label


ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path) -> list[dict[str, str]]:
    for encoding in ("utf-8-sig", "gb18030", "cp936"):
        try:
            with path.open(newline="", encoding=encoding) as handle:
                return list(csv.DictReader(handle))
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Unable to decode CSV: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotations",
        default="runs/ip_adapter_plus_injection/v2_4b_targeted_profile_candidates/audits/human_sensitivity_annotations.csv",
    )
    parser.add_argument(
        "--manifest",
        default="configs/experiment/v2_4b_targeted_profile_candidates.csv",
    )
    parser.add_argument("--output", default="analysis/v2_4b_human_profiles.csv")
    args = parser.parse_args()

    annotations = read_csv(ROOT / args.annotations)
    manifest = {row["case_id"]: row for row in read_csv(ROOT / args.manifest)}
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in annotations:
        by_case.setdefault(row["case"], []).append(row)

    fields = [
        "case",
        "content_family",
        "reference_family",
        "design_role",
        "hypothesis",
        "seed_count",
        "baseline_takeover_median",
        "baseline_takeover_max",
        "style_at_02_median",
        "style_at_10_median",
        "style_gain_median",
        "incremental_takeover_max_median",
        "incremental_nonzero_interval_count",
        "late_escalation_frequency",
        "style_valid_rate",
        "label_status",
        "profile_label",
    ]
    output_rows = []
    for case in sorted(by_case):
        profile = aggregate_human(by_case[case])
        source = manifest[case]
        output_rows.append(
            {
                "case": case,
                "content_family": source["content_family"],
                "reference_family": source["reference_family"],
                "design_role": source["design_role"],
                "hypothesis": source["hypothesis"],
                **{field: profile[field] for field in fields[5:-1]},
                "profile_label": profile_label(profile),
            }
        )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)
    print(output)


if __name__ == "__main__":
    main()
