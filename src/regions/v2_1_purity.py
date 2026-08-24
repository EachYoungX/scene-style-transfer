"""Average-pool Subject/Background occupancy and purity-aware token routes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


@dataclass(frozen=True)
class TokenPurity:
    resolution: int
    subject_fraction: np.ndarray
    background_fraction: np.ndarray
    valid_fraction: np.ndarray
    pure_subject: np.ndarray
    mixed: np.ndarray
    pure_background: np.ndarray
    invalid: np.ndarray

    def report(self) -> dict[str, int | float]:
        valid = self.pure_subject | self.mixed | self.pure_background
        return {
            "resolution": self.resolution,
            "valid_tokens": int(valid.sum()),
            "pure_subject": int(self.pure_subject.sum()),
            "mixed": int(self.mixed.sum()),
            "pure_background": int(self.pure_background.sum()),
            "invalid_tokens": int(self.invalid.sum()),
            "subject_fraction_mean_valid": float(self.subject_fraction[valid].mean()) if valid.any() else 0.0,
            "background_fraction_mean_valid": float(self.background_fraction[valid].mean()) if valid.any() else 0.0,
        }


def _pool(mask: np.ndarray, resolution: int) -> np.ndarray:
    tensor = torch.from_numpy(mask.astype(np.float32))[None, None]
    return F.adaptive_avg_pool2d(tensor, (resolution, resolution))[0, 0].numpy()


def project_token_purity(
    subject_mask: np.ndarray,
    background_mask: np.ndarray,
    valid_eval: np.ndarray,
    resolution: int,
    purity_threshold: float = 0.8,
) -> TokenPurity:
    """Project pixel masks by occupancy, then classify valid tokens by purity."""
    if subject_mask.shape != background_mask.shape or subject_mask.shape != valid_eval.shape:
        raise ValueError("Subject, Background, and valid_eval masks must share a shape")
    if not 0.5 < purity_threshold <= 1.0:
        raise ValueError("purity_threshold must be in (0.5, 1.0]")
    subject_fraction = _pool(subject_mask, resolution)
    background_fraction = _pool(background_mask, resolution)
    valid_fraction = _pool(valid_eval, resolution)
    valid = valid_fraction > 0.0
    pure_subject = valid & (subject_fraction >= purity_threshold)
    pure_background = valid & (background_fraction >= purity_threshold)
    mixed = valid & ~(pure_subject | pure_background)
    invalid = ~valid
    if np.any(pure_subject & pure_background):
        raise AssertionError("A token cannot be both Pure Subject and Pure Background")
    if not np.array_equal((pure_subject | mixed | pure_background), valid):
        raise AssertionError("Purity classes must partition valid tokens")
    return TokenPurity(
        resolution=resolution,
        subject_fraction=subject_fraction,
        background_fraction=background_fraction,
        valid_fraction=valid_fraction,
        pure_subject=pure_subject,
        mixed=mixed,
        pure_background=pure_background,
        invalid=invalid,
    )


def build_purity_routes(
    subject_mask: np.ndarray,
    background_mask: np.ndarray,
    valid_eval: np.ndarray,
    resolutions: tuple[int, ...] = (64, 32, 16),
    purity_threshold: float = 0.8,
    subject_gain: float = 1.0,
    background_gain: float = 0.0,
) -> dict[int, TokenPurity]:
    if subject_gain < 0 or background_gain < 0:
        raise ValueError("Region gains must be non-negative")
    return {
        resolution: project_token_purity(
            subject_mask,
            background_mask,
            valid_eval,
            resolution,
            purity_threshold,
        )
        for resolution in resolutions
    }


def route_gate(
    purity: TokenPurity,
    strategy: str,
    subject_gain: float = 1.0,
    background_gain: float = 0.0,
) -> np.ndarray:
    """Return the exact per-token gain for one purity-aware strategy."""
    if strategy not in {"S_sep_neutral", "S_sep_conservative"}:
        raise ValueError(f"Unknown purity route: {strategy}")
    mixed_gain = 1.0 if strategy == "S_sep_neutral" else min(subject_gain, background_gain)
    gate = np.zeros((purity.resolution, purity.resolution), dtype=np.float32)
    gate[purity.pure_subject] = subject_gain
    gate[purity.pure_background] = background_gain
    gate[purity.mixed] = mixed_gain
    gate[purity.invalid] = 0.0
    return gate


def route_masks(purity: TokenPurity) -> dict[str, np.ndarray]:
    return {
        "pure_subject": purity.pure_subject,
        "mixed": purity.mixed,
        "pure_background": purity.pure_background,
        "valid": purity.pure_subject | purity.mixed | purity.pure_background,
    }


def save_purity_overlay(content: Image.Image, purity: TokenPurity, path) -> None:
    """Save a nearest-neighbor 512px class overlay for quick coordinate review."""
    classes = np.zeros((purity.resolution, purity.resolution, 3), dtype=np.uint8)
    classes[purity.pure_subject] = (230, 60, 60)
    classes[purity.mixed] = (245, 180, 40)
    classes[purity.pure_background] = (60, 110, 235)
    overlay = Image.fromarray(classes, mode="RGB").resize(content.size, Image.Resampling.NEAREST)
    Image.blend(content.convert("RGB"), overlay, 0.42).save(path)
