"""Deterministic helpers for the V2.0 binary annotation protocol.

These helpers are intentionally independent of ``structure_risk.py``. They
prepare annotation candidates; they are never used as geometry-failure ground
truth and they do not consume a frozen risk map.
"""

from __future__ import annotations

import cv2
import numpy as np


def require_binary_uint8(array: np.ndarray, label: str = "mask") -> np.ndarray:
    values = np.unique(array)
    if array.dtype != np.uint8 or not set(values.tolist()) <= {0, 255}:
        raise ValueError(f"{label} must be uint8 with values only 0 and 255; got {values.tolist()}")
    return array


def canny_centerlines(rgb: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Canny(gray, low_threshold, high_threshold)


def lsd_centerlines(rgb: np.ndarray, min_length: float = 24.0) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    detected = detector.detect(gray)[0]
    canvas = np.zeros(gray.shape, dtype=np.uint8)
    if detected is None:
        return canvas
    for line in detected[:, 0, :]:
        x1, y1, x2, y2 = (float(value) for value in line)
        if np.hypot(x2 - x1, y2 - y1) < min_length:
            continue
        cv2.line(canvas, (round(x1), round(y1)), (round(x2), round(y2)), 255, thickness=1)
    return canvas


def rigid_centerline_candidate(canny: np.ndarray, lsd: np.ndarray) -> np.ndarray:
    require_binary_uint8(canny, "canny")
    require_binary_uint8(lsd, "lsd")
    if canny.shape != lsd.shape:
        raise ValueError(f"Canny/LSD shape mismatch: {canny.shape} vs {lsd.shape}")
    return np.where((canny > 0) | (lsd > 0), 255, 0).astype(np.uint8)


def dilate_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    require_binary_uint8(mask, "binary mask")
    if radius < 0:
        raise ValueError("Dilation radius must be non-negative")
    if radius == 0:
        return mask.copy()
    diameter = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.dilate(mask, kernel, iterations=1)


def erode_binary(mask: np.ndarray, radius: int) -> np.ndarray:
    require_binary_uint8(mask, "binary mask")
    if radius < 0:
        raise ValueError("Erosion radius must be non-negative")
    if radius == 0:
        return mask.copy()
    diameter = radius * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    return cv2.erode(mask, kernel, iterations=1, borderType=cv2.BORDER_CONSTANT, borderValue=0)


def apply_rigid_priority(
    rigid: np.ndarray, soft: np.ndarray, valid_content: np.ndarray, guard_radius: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    require_binary_uint8(rigid, "rigid")
    require_binary_uint8(soft, "soft")
    require_binary_uint8(valid_content, "valid_content")
    if rigid.shape != soft.shape or rigid.shape != valid_content.shape:
        raise ValueError("Rigid, soft, and valid-content masks must share one shape")
    valid = valid_content == 255
    rigid_final = np.where((rigid == 255) & valid, 255, 0).astype(np.uint8)
    guard = dilate_binary(rigid_final, guard_radius) == 255
    soft_final = np.where((soft == 255) & valid & ~guard, 255, 0).astype(np.uint8)
    return rigid_final, soft_final


def edge_difference_helper(
    content_rgb: np.ndarray, output_rgb: np.ndarray, valid_eval: np.ndarray | None = None
) -> np.ndarray:
    if content_rgb.shape != output_rgb.shape:
        raise ValueError(f"Content/output shape mismatch: {content_rgb.shape} vs {output_rgb.shape}")
    content_edges = canny_centerlines(content_rgb)
    output_edges = canny_centerlines(output_rgb)
    difference = cv2.bitwise_xor(content_edges, output_edges)
    if valid_eval is not None:
        require_binary_uint8(valid_eval, "valid_eval")
        if valid_eval.shape != difference.shape:
            raise ValueError("valid_eval must match edge-difference shape")
        difference = cv2.bitwise_and(difference, valid_eval)
    return difference
