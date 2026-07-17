"""Estimate the A0 scale factor needed to match a target residual energy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from diagnostics.normalization import estimate_scale_factor_from_jsonl  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-jsonl", required=True, type=Path)
    parser.add_argument("--target-jsonl", required=True, type=Path)
    parser.add_argument("--field", default="ip_residual_rms")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result = estimate_scale_factor_from_jsonl(args.source_jsonl, args.target_jsonl, args.field)
    text = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()

