"""List style-sweep grids for a compatibility diagnostic run."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/compatibility_diagnostic_pairs.csv")
    parser.add_argument("--run-name", default="compatibility_diag_12step")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    with (root / args.manifest).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        grid = root / "runs" / "style_sweep" / args.run_name / row["case_id"] / "style_sweep_grid.png"
        print(
            f"{row['compat_group']} {row['case_id']} "
            f"semantic={row['semantic_match']} layout={row['layout_match']} "
            f"direction={row['directional_match']} rigidity={row['rigidity_level']}"
        )
        print(f"  {grid}")


if __name__ == "__main__":
    main()
