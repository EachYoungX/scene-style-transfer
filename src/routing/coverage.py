"""Manifest-based reference coverage estimates for V0 routing."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


SCENE_REQUIRED_TAGS = {
    "city_architecture": {"architecture"},
    "vegetation": {"vegetation"},
    "natural_landscape": {"vegetation", "sky", "water"},
    "water_coast": {"water", "sky"},
}

SCENE_RISK_TAGS = {
    "city_architecture": {"architecture", "facade", "cathedral", "city"},
    "vegetation": {"forest", "trees", "garden", "vegetation"},
    "natural_landscape": {"mountains", "landscape", "water", "sky", "hills"},
    "water_coast": {"ocean", "wave", "water", "beach", "sky"},
}


@dataclass(frozen=True)
class CoverageResult:
    image_id: str
    scene_type: str
    style_group: str
    score: float
    matched_tags: list[str]
    missing_required: list[str]
    risky_reference_tags: list[str]


def image_id_from_path(path: str) -> str:
    return Path(path).stem


def split_tags(value: str) -> set[str]:
    return {part.strip() for part in value.split("|") if part.strip()}


def load_style_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["image_id"]: row for row in rows}


def estimate_coverage(style_path: str, scene_type: str, manifest: dict[str, dict[str, str]]) -> CoverageResult:
    image_id = image_id_from_path(style_path)
    row = manifest.get(image_id)
    if row is None:
        raise KeyError(f"Style image {image_id!r} not found in manifest")

    tags = split_tags(row.get("scene_tags", "")) | split_tags(row.get("dominant_elements", ""))
    for flag, tag in [
        ("has_architecture", "architecture"),
        ("has_water", "water"),
        ("has_vegetation", "vegetation"),
        ("has_sky", "sky"),
    ]:
        if row.get(flag) == "1":
            tags.add(tag)

    required = SCENE_REQUIRED_TAGS.get(scene_type, set())
    matched = sorted(required & tags)
    missing = sorted(required - tags)
    risky_tags = sorted((SCENE_RISK_TAGS.get(scene_type, set()) ^ tags) & tags)

    required_score = len(matched) / max(1, len(required))
    soft_overlap = len(SCENE_RISK_TAGS.get(scene_type, set()) & tags) / max(1, len(SCENE_RISK_TAGS.get(scene_type, set())))
    score = round(float(0.75 * required_score + 0.25 * soft_overlap), 4)

    return CoverageResult(
        image_id=image_id,
        scene_type=scene_type,
        style_group=row.get("style_group", ""),
        score=score,
        matched_tags=matched,
        missing_required=missing,
        risky_reference_tags=risky_tags[:10],
    )
