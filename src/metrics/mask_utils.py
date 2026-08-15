"""Image and mask loading helpers for V2.0 geometry-risk validation."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image


TARGET_SHAPE = (512, 512)


def _resize(array: np.ndarray, shape: tuple[int, int], interpolation: int) -> np.ndarray:
    height, width = shape
    if array.shape[:2] == shape:
        return array
    return cv2.resize(array, (width, height), interpolation=interpolation)


def load_rgb(path: Path, shape: tuple[int, int] = TARGET_SHAPE) -> np.ndarray:
    """Load an RGB image and resize it with high-quality interpolation."""
    image = np.asarray(Image.open(path).convert("RGB"))
    return _resize(image, shape, cv2.INTER_LANCZOS4)


def load_continuous_risk(path: Path, shape: tuple[int, int] = TARGET_SHAPE) -> np.ndarray:
    """Load a float risk map; .npy is lossless and image formats use [0, 255]."""
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    else:
        array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    if array.ndim != 2:
        raise ValueError(f"Continuous risk map must be 2-D, got {array.shape} from {path}")
    array = _resize(array.astype(np.float32), shape, cv2.INTER_LINEAR)
    if not np.isfinite(array).all():
        raise ValueError(f"Continuous risk map contains non-finite values: {path}")
    return np.clip(array, 0.0, 1.0)


def load_mask(path: Path, shape: tuple[int, int] = TARGET_SHAPE) -> np.ndarray:
    """Load a grayscale annotation mask with nearest-neighbour resizing."""
    array = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    array = _resize(array, shape, cv2.INTER_NEAREST)
    return np.clip(array, 0.0, 1.0)


def load_binary_mask(path: Path, shape: tuple[int, int] = TARGET_SHAPE) -> np.ndarray:
    """Load a final annotation without resizing or thresholding it.

    Final V2.0 annotations are definitionally 8-bit grayscale PNGs containing
    only absolute black and white. Rejecting all other inputs removes an
    otherwise hidden annotation-confidence threshold from evaluation.
    """
    image = Image.open(path)
    if image.mode != "L":
        raise ValueError(f"Binary mask must be 8-bit grayscale (mode L): {path}")
    array = np.asarray(image, dtype=np.uint8)
    if array.shape != shape:
        raise ValueError(f"Binary mask must have shape {shape}, got {array.shape}: {path}")
    values = set(np.unique(array).tolist())
    if not values <= {0, 255}:
        raise ValueError(f"Binary mask must contain only 0 and 255, got {sorted(values)}: {path}")
    return array == 255


def binary_mask(mask: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"Mask must be 2-D, got {mask.shape}")
    return np.asarray(mask >= threshold, dtype=bool)


def valid_mask(shape: tuple[int, int], uncertainty: np.ndarray | None = None, threshold: float = 0.01) -> np.ndarray:
    valid = np.ones(shape, dtype=bool)
    if uncertainty is not None:
        if uncertainty.shape != shape:
            raise ValueError(f"Uncertainty shape {uncertainty.shape} does not match {shape}")
        valid &= uncertainty < threshold
    return valid


def validate_alignment(sample_id: str, **arrays: np.ndarray) -> tuple[int, int]:
    """Require all named arrays to share one spatial shape."""
    shapes = {name: value.shape[:2] for name, value in arrays.items() if value is not None}
    unique = set(shapes.values())
    if len(unique) != 1:
        raise ValueError(f"{sample_id}: spatial shape mismatch: {shapes}")
    if not unique:
        raise ValueError(f"{sample_id}: no arrays supplied for alignment validation")
    return next(iter(unique))
