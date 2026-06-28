from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_routed_v0 import map_plan_to_params  # noqa: E402


def test_routed_params_stay_below_style_only_pressure():
    params = map_plan_to_params(
        {
            "local_style_weight": 0.28,
            "global_appearance_weight": 0.9,
            "structure_lock_weight": 0.86,
        }
    )

    assert 0.52 <= params["strength"] <= 0.76
    assert params["controlnet_scale"] >= 0.75
    assert params["ip_adapter_scale"] < 1.05


def test_push_variant_increases_style_pressure():
    plan = {
        "local_style_weight": 0.28,
        "global_appearance_weight": 0.9,
        "structure_lock_weight": 0.86,
    }
    default = map_plan_to_params(plan)
    pushed = map_plan_to_params(plan, "push")

    assert pushed["strength"] > default["strength"]
    assert pushed["ip_adapter_scale"] > default["ip_adapter_scale"]
    assert pushed["controlnet_scale"] < default["controlnet_scale"]


def test_low_lock_allows_more_local_style_pressure():
    low_lock = map_plan_to_params(
        {
            "local_style_weight": 0.67,
            "global_appearance_weight": 0.88,
            "structure_lock_weight": 0.59,
        }
    )
    high_lock = map_plan_to_params(
        {
            "local_style_weight": 0.28,
            "global_appearance_weight": 0.9,
            "structure_lock_weight": 0.86,
        }
    )

    assert low_lock["ip_adapter_scale"] > high_lock["ip_adapter_scale"]
    assert low_lock["controlnet_scale"] < high_lock["controlnet_scale"]
