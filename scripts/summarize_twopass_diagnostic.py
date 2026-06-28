"""Summarize two-pass diagnostic labels by rigidity group."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="configs/experiment/twopass_diagnostic_labels.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / args.labels).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_group = defaultdict(list)
    conclusions = Counter()
    for row in rows:
        by_group[row["scene_group"]].append(row)
        conclusions[row["main_conclusion"]] += 1

    print("Two-pass diagnostic by rigidity group:")
    for group, group_rows in sorted(by_group.items()):
        print(f"  Group {group}: {len(group_rows)} case(s)")
        for row in group_rows:
            print(f"    {row['case_id']}: {row['main_conclusion']}")

    print("\nConclusion counts:")
    for conclusion, count in conclusions.items():
        print(f"  {count}x {conclusion}")


if __name__ == "__main__":
    main()
