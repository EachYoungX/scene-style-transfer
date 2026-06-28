"""Rule-based V0 routing plans for strong-style scene transfer."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from preprocess.structure_risk import RiskStats
from routing.coverage import CoverageResult


SCENE_SENSITIVITY = {
    "city_architecture": 0.18,
    "vegetation": 0.15,
    "water_coast": 0.08,
    "natural_landscape": 0.08,
}

REGION_ATTRIBUTE_POLICY = {
    "key_geometry": {
        "risk": "high",
        "local_style": "restricted",
        "global_appearance": "allowed",
    },
    "flat_material": {
        "risk": "medium",
        "local_style": "moderate",
        "global_appearance": "allowed",
    },
    "shadow_low_texture": {
        "risk": "medium_low",
        "local_style": "moderate",
        "global_appearance": "allowed",
    },
    "texture_surface": {
        "risk": "low_to_medium",
        "local_style": "allowed",
        "global_appearance": "allowed",
    },
}


@dataclass(frozen=True)
class RoutingPlan:
    case_id: str
    local_style_weight: float
    global_appearance_weight: float
    structure_lock_weight: float
    recommended_regime: str
    rationale: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def build_routing_plan(case_id: str, coverage: CoverageResult, risk: RiskStats) -> RoutingPlan:
    sensitivity = SCENE_SENSITIVITY.get(coverage.scene_type, 0.05)
    risk_level = min(1.0, 0.80 * risk.p90 + 0.20 * risk.mean + sensitivity)
    coverage_level = coverage.score

    local_style = coverage_level * (1.0 - risk_level)
    global_style = 0.55 + 0.35 * coverage_level
    structure_lock = 0.45 + 0.50 * risk_level

    rationale: list[str] = []
    if sensitivity >= 0.15:
        rationale.append("scene_structurally_sensitive")
    if coverage_level < 0.55:
        rationale.append("low_reference_coverage")
        global_style += 0.10
        local_style *= 0.55
    if risk_level > 0.32:
        rationale.append("high_structure_risk")
        local_style *= 0.75
        structure_lock += 0.10
    if coverage.missing_required:
        rationale.append("missing_required_style_semantics")
    if not rationale:
        rationale.append("safe_for_moderate_local_style")

    if local_style < 0.20:
        regime = "global_only_or_very_weak_local"
    elif local_style < 0.55:
        regime = "moderate_local_with_structure_lock"
    else:
        regime = "stronger_local_allowed"

    return RoutingPlan(
        case_id=case_id,
        local_style_weight=round(float(min(local_style, 1.0)), 4),
        global_appearance_weight=round(float(min(global_style, 1.0)), 4),
        structure_lock_weight=round(float(min(structure_lock, 1.0)), 4),
        recommended_regime=regime,
        rationale=rationale,
    )
