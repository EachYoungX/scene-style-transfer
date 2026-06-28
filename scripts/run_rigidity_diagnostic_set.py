"""Run style sweep, routing analysis, and T0-T4 diagnostics for rigidity groups."""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path


def read_cases(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/rigidity_diagnostic_pairs.csv")
    parser.add_argument("--run-name", default="rigidity_diag_12step")
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-sweep", action="store_true")
    parser.add_argument("--skip-routing", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    python = sys.executable
    overwrite = ["--overwrite"] if args.overwrite else []

    if not args.skip_sweep:
        run_cmd(
            [
                python,
                "scripts/run_style_sweep.py",
                "--manifest",
                args.manifest,
                "--run-name",
                args.run_name,
                "--num-inference-steps",
                str(args.num_inference_steps),
                *overwrite,
            ],
            root,
        )

    routing_run = f"{args.run_name}_routing"
    if not args.skip_routing:
        run_cmd(
            [
                python,
                "scripts/analyze_routing_v0.py",
                "--manifest",
                args.manifest,
                "--run-name",
                routing_run,
                *overwrite,
            ],
            root,
        )

    for case in read_cases(root / args.manifest):
        run_cmd(
            [
                python,
                "scripts/run_twopass_diagnostic.py",
                "--case-id",
                case["case_id"],
                "--style-sweep-run",
                args.run_name,
                "--routing-run",
                routing_run,
                "--run-name",
                args.run_name,
            ],
            root,
        )


if __name__ == "__main__":
    main()
