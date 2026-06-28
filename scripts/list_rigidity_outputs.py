"""List output grids for a rigidity diagnostic run."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", default="rigidity_diag_12step")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    style_root = root / "runs" / "style_sweep" / args.run_name
    diagnostic_root = root / "runs" / "twopass_diagnostic" / args.run_name

    for case_dir in sorted(diagnostic_root.iterdir()):
        if not case_dir.is_dir():
            continue
        style_grid = style_root / case_dir.name / "style_sweep_grid.png"
        diagnostic_grid = case_dir / "diagnostic_grid.png"
        alpha_grid = case_dir / "alpha_grid.png"
        print(case_dir.name)
        print(f"  style:      {style_grid}")
        print(f"  diagnostic: {diagnostic_grid}")
        print(f"  alpha:      {alpha_grid}")


if __name__ == "__main__":
    main()
