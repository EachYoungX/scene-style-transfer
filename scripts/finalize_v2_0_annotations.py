"""Validate reviewed V2.0 masks and mark the frozen annotation set complete."""

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

from annotations.geometry_protocol import apply_rigid_priority  # noqa: E402
from metrics.mask_utils import load_binary_mask  # noqa: E402
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
    guard_radius = int(config["annotations"]["soft_stylization"]["rigid_guard_radius_px"])
    manifest_path = ROOT / experiment["annotation_manifest"]
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Annotation manifest is empty")

    checked: set[Path] = set()
    normalized: set[tuple[Path, Path, Path]] = set()
    for row in rows:
        if row.get("source_status") != "ready":
            raise RuntimeError(f"A2 source is not ready: {row['sample_id']}")
        for field in MASK_FIELDS:
            path = ROOT / row[field]
            if path in checked:
                continue
            if not path.exists():
                raise FileNotFoundError(path)
            load_binary_mask(path, (size, size))
            checked.add(path)
        valid_content_path = ROOT / row["valid_content_mask"]
        valid_content = load_binary_mask(valid_content_path, (size, size))
        rigid_path = ROOT / row["rigid_structure_mask"]
        soft_path = ROOT / row["soft_stylization_mask"]
        content_masks = (rigid_path, soft_path, valid_content_path)
        if content_masks not in normalized:
            rigid, soft = apply_rigid_priority(
                load_binary_mask(rigid_path, (size, size)).astype(np.uint8) * 255,
                load_binary_mask(soft_path, (size, size)).astype(np.uint8) * 255,
                valid_content.astype(np.uint8) * 255,
                guard_radius,
            )
            Image.fromarray(rigid).save(rigid_path)
            Image.fromarray(soft).save(soft_path)
            if ((rigid == 255) & (soft == 255)).any():
                raise AssertionError(f"Rigid/soft overlap remains after normalization: {rigid_path}")
            normalized.add(content_masks)
        for field in ("geometry_failure_mask", "uncertainty_mask"):
            path = ROOT / row[field]
            clipped = load_binary_mask(path, (size, size)) & valid_content
            Image.fromarray(clipped.astype(np.uint8) * 255).save(path)
        for field in STATUS_FIELDS:
            row[field] = "complete"

    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Marked {len(rows)} samples complete after validating {len(checked)} unique masks")


if __name__ == "__main__":
    main()
