from pathlib import Path

from PIL import Image

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from preprocess.structure_risk import compute_structure_risk
from routing.coverage import estimate_coverage, load_style_manifest
from routing.rule_router import build_routing_plan


ROOT = Path(__file__).resolve().parents[1]


def test_coverage_matches_city_architecture_reference():
    manifest = load_style_manifest(ROOT / "data/manifests/style_refs.csv")

    coverage = estimate_coverage(
        "data/raw/monet/monet_rouen_cathedral_1894.jpg",
        "city_architecture",
        manifest,
    )

    assert coverage.score >= 0.75
    assert "architecture" in coverage.matched_tags


def test_coverage_matches_forest_reference():
    manifest = load_style_manifest(ROOT / "data/manifests/style_refs.csv")

    coverage = estimate_coverage(
        "data/raw/klimt/klimt_birch_forest_1903.jpg",
        "vegetation",
        manifest,
    )

    assert coverage.score >= 0.75
    assert "vegetation" in coverage.matched_tags


def test_structure_risk_returns_normalized_map():
    image = Image.new("RGB", (128, 128), (0, 0, 0))
    for x in range(20, 110, 15):
        for y in range(128):
            image.putpixel((x, y), (255, 255, 255))

    risk, stats = compute_structure_risk(image, "vegetation")

    assert risk.shape == (128, 128)
    assert 0.0 <= float(risk.min()) <= float(risk.max()) <= 1.0
    assert stats.p90 > 0


def test_router_reduces_local_style_when_risk_is_high():
    manifest = load_style_manifest(ROOT / "data/manifests/style_refs.csv")
    coverage = estimate_coverage(
        "data/raw/monet/monet_rouen_cathedral_1894.jpg",
        "city_architecture",
        manifest,
    )

    class Risk:
        mean = 0.35
        p90 = 0.8

    plan = build_routing_plan("case", coverage, Risk())

    assert plan.structure_lock_weight > plan.local_style_weight
    assert "high_structure_risk" in plan.rationale
