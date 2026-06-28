from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from routing.rule_router import REGION_ATTRIBUTE_POLICY  # noqa: E402


def test_region_policy_distinguishes_geometry_from_material():
    assert REGION_ATTRIBUTE_POLICY["key_geometry"]["local_style"] == "restricted"
    assert REGION_ATTRIBUTE_POLICY["flat_material"]["local_style"] == "moderate"
    assert REGION_ATTRIBUTE_POLICY["shadow_low_texture"]["risk"] == "medium_low"
