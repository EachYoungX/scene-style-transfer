"""Load and validate V2.1 Subject/Background masks.

The annotation editor saves RGB PNGs, so V2.1 accepts RGB or grayscale files,
requires equal RGB channels when RGB is used, thresholds once at 128, then
applies valid-eval and rigid exclusion to the effective injection regions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
TARGET_SHAPE = (512, 512)


def load_annotation_binary(path: Path, threshold: int = 128) -> np.ndarray:
    """Load an editor-exported mask and convert it to a boolean 512px mask."""
    if not 1 <= threshold <= 254:
        raise ValueError("threshold must be between 1 and 254")
    image = Image.open(path)
    if image.size != (TARGET_SHAPE[1], TARGET_SHAPE[0]):
        raise ValueError(f"V2.1 region mask must be 512x512: {path} is {image.size}")
    raw = np.asarray(image)
    if raw.ndim == 3:
        if raw.shape[2] not in (3, 4):
            raise ValueError(f"Unsupported channel count for {path}: {raw.shape}")
        rgb = raw[:, :, :3]
        if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 0], rgb[:, :, 2])):
            raise ValueError(f"RGB V2.1 region mask has unequal channels: {path}")
        gray = rgb[:, :, 0]
    elif raw.ndim == 2:
        gray = raw
    else:
        raise ValueError(f"Unsupported mask array shape for {path}: {raw.shape}")
    return gray >= threshold


def load_strict_mask(path: Path) -> np.ndarray:
    """Load an existing V2.0 strict black/white mask."""
    from metrics.mask_utils import load_binary_mask

    return load_binary_mask(path, TARGET_SHAPE)


@dataclass(frozen=True)
class RegionMaskSet:
    subject_raw: np.ndarray
    background_raw: np.ndarray
    rigid: np.ndarray
    valid_content: np.ndarray
    valid_eval: np.ndarray
    subject: np.ndarray
    background: np.ndarray
    neutral: np.ndarray
    raw_overlap: int
    raw_neutral_inside_valid_eval: int
    effective_overlap: int
    rigid_excluded_subject: int
    rigid_excluded_background: int
    invalid_excluded_subject: int
    invalid_excluded_background: int

    def report(self) -> dict[str, int]:
        values = asdict(self)
        return {key: int(value.sum()) if isinstance(value, np.ndarray) else int(value) for key, value in values.items()}


def load_region_mask_set(
    subject_path: Path,
    background_path: Path,
    rigid_path: Path,
    valid_content_path: Path,
    valid_eval_path: Path,
    threshold: int = 128,
) -> RegionMaskSet:
    subject_raw = load_annotation_binary(subject_path, threshold)
    background_raw = load_annotation_binary(background_path, threshold)
    rigid = load_strict_mask(rigid_path)
    valid_content = load_strict_mask(valid_content_path)
    valid_eval = load_strict_mask(valid_eval_path)
    shape = subject_raw.shape
    if any(array.shape != shape for array in (background_raw, rigid, valid_content, valid_eval)):
        raise ValueError("V2.1 Subject, Background, rigid, and valid masks must share one shape")

    valid_eval &= valid_content
    # The intended protocol has no hand-drawn Neutral. If antialiased RGB
    # export creates a threshold overlap, use a fixed, reproducible policy:
    # Subject wins, then rigid exclusion is applied to both regions.
    subject = subject_raw & valid_eval & ~rigid
    background = background_raw & ~subject_raw & valid_eval & ~rigid
    neutral = valid_eval & ~(subject | background | rigid)
    return RegionMaskSet(
        subject_raw=subject_raw,
        background_raw=background_raw,
        rigid=rigid,
        valid_content=valid_content,
        valid_eval=valid_eval,
        subject=subject,
        background=background,
        neutral=neutral,
        raw_overlap=int((subject_raw & background_raw & valid_eval).sum()),
        raw_neutral_inside_valid_eval=int((valid_eval & ~(subject_raw | background_raw)).sum()),
        effective_overlap=int((subject & background).sum()),
        rigid_excluded_subject=int((subject_raw & valid_eval & rigid).sum()),
        rigid_excluded_background=int((background_raw & valid_eval & rigid).sum()),
        invalid_excluded_subject=int((subject_raw & ~valid_eval).sum()),
        invalid_excluded_background=int((background_raw & ~valid_eval).sum()),
    )
