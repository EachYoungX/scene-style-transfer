"""Validate reviewed V2.0 masks and mark the frozen annotation set complete."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
STATUS_FIELDS = (
    "rigid_status",
    "soft_status",
    "geometry_failure_status",
    "uncertainty_status",
)
MASK_FIELDS = (
    "rigid_structure_mask",
    "soft_stylization_mask",
    "geometry_failure_mask",
    "uncertainty_mask",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiment/v2_0_geometry_risk_eval.yaml")
    parser.add_argument(
        "--confirm-reviewed",
        action="store_true",
        help="Confirm that every mask, including intentionally empty masks, has been manually reviewed.",
    )
    args = parser.parse_args()
    if not args.confirm_reviewed:
        parser.error("--confirm-reviewed is required; pending black masks may be intentional or untouched")

    config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    experiment = config["experiment"]
    size = int(experiment["image_size"])
    manifest_path = ROOT / experiment["annotation_manifest"]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Annotation manifest is empty")

    checked: set[Path] = set()
    for row in rows:
        if row.get("source_status") != "ready":
            raise RuntimeError(f"A2 source is not ready: {row['sample_id']}")
        for field in MASK_FIELDS:
            path = ROOT / row[field]
            if path in checked:
                continue
            if not path.exists():
                raise FileNotFoundError(path)
            image = Image.open(path)
            if image.mode != "L" or image.size != (size, size):
                raise ValueError(f"Mask must be {size}x{size} 8-bit grayscale: {path}")
            checked.add(path)
        for field in STATUS_FIELDS:
            row[field] = "complete"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Marked {len(rows)} samples complete after validating {len(checked)} unique masks")


if __name__ == "__main__":
    main()
