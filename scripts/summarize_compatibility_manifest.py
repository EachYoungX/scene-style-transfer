"""Summarize compatibility diagnostic cases by group."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/compatibility_diagnostic_pairs.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / args.manifest).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for row in rows:
        groups[row["compat_group"]].append(row)

    for group, group_rows in sorted(groups.items()):
        print(f"{group}: {len(group_rows)} case(s)")
        for row in group_rows:
            print(
                "  "
                f"{row['case_id']}: semantic={row['semantic_match']} "
                f"layout={row['layout_match']} direction={row['directional_match']} "
                f"rigidity={row['rigidity_level']} expected={row['expected_failure']}"
            )


if __name__ == "__main__":
    main()
