"""Summarize subjective style/structure labels from sweep experiments."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default="configs/experiment/style_sweep_labels.csv")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    rows: list[dict[str, str]] = []
    with (root / args.labels).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    by_sweep: dict[str, list[dict[str, str]]] = defaultdict(list)
    failures = Counter()
    for row in rows:
        by_sweep[row["sweep_id"]].append(row)
        failures.update(row["main_failures"].split("|"))

    print("Style/structure averages by sweep:")
    for sweep_id, sweep_rows in by_sweep.items():
        style = sum(float(row["style_similarity_subjective"]) for row in sweep_rows) / len(sweep_rows)
        structure = sum(float(row["structure_similarity_subjective"]) for row in sweep_rows) / len(sweep_rows)
        tradeoff = style - (100 - structure)
        print(f"  {sweep_id:24s} style={style:5.1f} structure={structure:5.1f} tradeoff={tradeoff:6.1f}")

    print("\nFailure counts:")
    for failure, count in sorted(failures.items()):
        print(f"  {failure}: {count}")


if __name__ == "__main__":
    main()
