"""Convert reviewed rigid centerlines into fixed-width final structural bands."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from annotations.geometry_protocol import dilate_centerline, require_binary_uint8  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    parser.add_argument(
        "--confirm-centerlines-reviewed",
        action="store_true",
        help="Confirm that candidate centerlines were manually cleaned and supplemented.",
    )
    args = parser.parse_args()
    if not args.confirm_centerlines_reviewed:
        parser.error("--confirm-centerlines-reviewed is required")

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    radius = int(config["annotations"]["rigid_centerline"]["dilation_radius_px"])
    manifest_path = ROOT / experiment["annotation_manifest"]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    processed: set[tuple[Path, Path]] = set()
    for row in rows:
        centerline_path = ROOT / row["rigid_structure_centerline"]
        final_path = ROOT / row["rigid_structure_mask"]
        pair = (centerline_path, final_path)
        if pair not in processed:
            image = Image.open(centerline_path)
            if image.mode != "L":
                raise ValueError(f"Rigid centerline must be 8-bit grayscale: {centerline_path}")
            centerline = require_binary_uint8(np.asarray(image), str(centerline_path))
            Image.fromarray(dilate_centerline(centerline, radius)).save(final_path)
            processed.add(pair)
        row["rigid_centerline_status"] = "reviewed"
        row["rigid_status"] = "in_progress"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Materialized {len(processed)} rigid masks with radius={radius} px")


if __name__ == "__main__":
    main()
