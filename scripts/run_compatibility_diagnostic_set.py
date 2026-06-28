"""Run style-strength sweeps for content-reference compatibility groups."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd: list[str], cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="configs/experiment/compatibility_diagnostic_pairs.csv")
    parser.add_argument("--run-name", default="compatibility_diag_12step")
    parser.add_argument("--num-inference-steps", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    overwrite = ["--overwrite"] if args.overwrite else []
    run_cmd(
        [
            sys.executable,
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


if __name__ == "__main__":
    main()
