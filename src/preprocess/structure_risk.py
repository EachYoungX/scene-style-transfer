"""Heuristic structure-risk maps for scene stylization.

The V0 risk map is intentionally simple and interpretable. It does not try to
segment objects; it highlights regions where strong local style pressure is
likely to change scene layout or copy reference semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RiskStats:
    mean: float
    p90: float
    edge_density: float
    line_density: float
    central_path_risk: float


def load_rgb(path: Path, size: int = 512) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), (0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def normalize01(array: np.ndarray) -> np.ndarray:
    array = array.astype(np.float32)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.float32)
    return (array - min_value) / (max_value - min_value)


def edge_map(rgb: Image.Image, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    gray = cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, low_threshold, high_threshold)
    return edges.astype(np.float32) / 255.0


def line_map(edges: np.ndarray) -> np.ndarray:
    h, w = edges.shape
    lines = cv2.HoughLinesP(
        (edges * 255).astype(np.uint8),
        rho=1,
        theta=np.pi / 180,
        threshold=45,
        minLineLength=max(24, w // 12),
        maxLineGap=max(6, w // 96),
    )
    canvas = np.zeros_like(edges, dtype=np.float32)
    if lines is None:
        return canvas
    for line in lines[:, 0]:
        x1, y1, x2, y2 = [int(v) for v in line]
        cv2.line(canvas, (x1, y1), (x2, y2), 1.0, thickness=3)
    return cv2.GaussianBlur(canvas, (0, 0), sigmaX=3.0)


def central_path_prior(shape: tuple[int, int], scene_type: str) -> np.ndarray:
    """Approximate path/road risk in the lower-center image region."""
    h, w = shape
    if scene_type not in {"vegetation", "city_architecture", "water_coast", "natural_landscape"}:
        return np.zeros(shape, dtype=np.float32)

    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, w, dtype=np.float32)[None, :]
    lower_weight = np.clip((y - 0.30) / 0.70, 0, 1)
    center_weight = np.exp(-(x / 0.34) ** 2)
    prior = lower_weight * center_weight

    if scene_type == "vegetation":
        return np.clip(1.45 * prior, 0, 1).astype(np.float32)
    if scene_type == "city_architecture":
        return (0.45 * prior).astype(np.float32)
    return (0.25 * prior).astype(np.float32)


def scene_prior(shape: tuple[int, int], scene_type: str) -> np.ndarray:
    h, _ = shape
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if scene_type == "city_architecture":
        # Building facades and roofs often occupy the middle/upper image.
        return np.broadcast_to(np.clip(1.0 - np.abs(y - 0.48) / 0.55, 0, 1), shape).astype(np.float32)
    if scene_type == "vegetation":
        # Preserve the path/trunk layout more than small high-contrast canopy gaps.
        lower_half = np.clip((y - 0.25) / 0.75, 0, 1)
        return np.broadcast_to(0.25 + 0.45 * lower_half, shape).astype(np.float32)
    if scene_type in {"water_coast", "natural_landscape"}:
        horizon_band = np.exp(-((y - 0.48) / 0.08) ** 2)
        return np.broadcast_to(0.45 * horizon_band, shape).astype(np.float32)
    return np.zeros(shape, dtype=np.float32)


def scene_edge_weight(shape: tuple[int, int], scene_type: str) -> np.ndarray:
    h, _ = shape
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    if scene_type == "vegetation":
        # Avoid treating small bright sky slits at the top as the main structure.
        weight = 0.25 + 0.85 * np.clip((y - 0.18) / 0.82, 0, 1)
        return np.broadcast_to(weight, shape).astype(np.float32)
    if scene_type == "natural_landscape":
        # Prefer horizon/shoreline layout over dense foreground foliage edges.
        horizon = 0.55 + 0.45 * np.exp(-((y - 0.48) / 0.18) ** 2)
        foreground_penalty = 1.0 - 0.30 * np.clip((y - 0.72) / 0.28, 0, 1)
        return np.broadcast_to(horizon * foreground_penalty, shape).astype(np.float32)
    return np.ones(shape, dtype=np.float32)


def compute_structure_risk(rgb: Image.Image, scene_type: str) -> tuple[np.ndarray, RiskStats]:
    edges = edge_map(rgb)
    lines = line_map(edges)
    edge_weight = scene_edge_weight(edges.shape, scene_type)
    edge_blur = cv2.GaussianBlur(edges * edge_weight, (0, 0), sigmaX=2.0)
    lines = lines * edge_weight
    path = central_path_prior(edges.shape, scene_type)
    prior = scene_prior(edges.shape, scene_type)

    if scene_type == "vegetation":
        risk = 0.30 * normalize01(edge_blur) + 0.20 * normalize01(lines) + 0.35 * path + 0.15 * prior
    elif scene_type == "natural_landscape":
        risk = 0.35 * normalize01(edge_blur) + 0.20 * normalize01(lines) + 0.10 * path + 0.35 * prior
    else:
        risk = 0.45 * normalize01(edge_blur) + 0.30 * normalize01(lines) + 0.15 * path + 0.10 * prior
    risk = np.clip(risk, 0.0, 1.0)

    stats = RiskStats(
        mean=round(float(risk.mean()), 4),
        p90=round(float(np.percentile(risk, 90)), 4),
        edge_density=round(float(edges.mean()), 4),
        line_density=round(float((lines > 0.05).mean()), 4),
        central_path_risk=round(float(path.mean()), 4),
    )
    return risk, stats


def save_risk_map(risk: np.ndarray, path: Path) -> None:
    heat = (np.clip(risk, 0, 1) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat, cv2.COLORMAP_INFERNO)
    rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(path)
