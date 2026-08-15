"""Convert manually painted V2.0 rigid masks to strict 8-bit binary PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from metrics.mask_utils import load_binary_mask  # noqa: E402


def normalize_mask(path: Path, valid_content_path: Path, threshold: int) -> dict[str, int | str]:
    image = Image.open(path)
    if image.size != (512, 512):
        raise ValueError(f"Rigid mask must be 512x512: {path} is {image.size}")

    raw = np.asarray(image)
    if raw.ndim == 3:
        if raw.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported channel count for {path}: {raw.shape}")
        rgb = raw[:, :, :3]
        if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
            raise ValueError(f"RGB rigid mask has unequal channels: {path}")
        gray = rgb[:, :, 0]
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported mask array shape for {path}: {raw.shape}")

    valid = np.asarray(Image.open(valid_content_path).convert("L"), dtype=np.uint8)
    if valid.shape != gray.shape:
        raise ValueError(f"Rigid mask and valid_content shape differ: {path} / {valid_content_path}")

    binary = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    binary[valid == 0] = 0
    Image.fromarray(binary, mode="L").save(path)
    load_binary_mask(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "source_mode": image.mode,
        "source_gray_values": int(np.unique(gray).size),
        "gray_pixels_thresholded": int(np.count_nonzero((gray >= threshold) & (gray < 255))),
        "final_white_pixels": int(np.count_nonzero(binary)),
        "outside_valid_removed": int(np.count_nonzero((gray >= threshold) & (valid == 0))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--name", action="append", help="Normalize one filename; repeatable.")
    args = parser.parse_args()
    if not 1 <= args.threshold <= 254:
        raise ValueError("threshold must be between 1 and 254")

    root = ROOT / "data/derived/v2_0_geometry_risk"
    rigid_root = root / "annotations/rigid_structure"
    valid_root = root / "valid_masks/valid_content"
    names = args.name or sorted(path.name for path in rigid_root.glob("*.png"))
    for name in names:
        path = rigid_root / name
        valid_path = valid_root / name
        if not path.exists() or not valid_path.exists():
            raise FileNotFoundError(f"Missing rigid or valid_content mask: {path} / {valid_path}")
        print(normalize_mask(path, valid_path, args.threshold))


if __name__ == "__main__":
    main()
