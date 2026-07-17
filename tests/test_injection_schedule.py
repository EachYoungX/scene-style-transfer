from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics.injection_schedule import (  # noqa: E402
    build_variant_schedule,
    build_processor_map,
    five_bin_time_curve,
    legacy_variant_schedule,
    parse_processor_name,
    processor_block_weights,
)


class DummyIpProcessor:
    scale = 1.0


def test_parse_processor_name_extracts_block_and_attention():
    info = parse_processor_name("up_blocks.2.attentions.1.transformer_blocks.0.attn2.processor")

    assert info.block_group == "up"
    assert info.block_index == 2
    assert info.block == "up_blocks.2"
    assert info.attention_index == 1


def test_build_processor_weights_count_ip_processors_only():
    processors = {
        "down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor": DummyIpProcessor(),
        "down_blocks.0.attentions.1.transformer_blocks.0.attn2.processor": DummyIpProcessor(),
        "up_blocks.3.attentions.0.transformer_blocks.0.attn2.processor": DummyIpProcessor(),
    }
    weights = processor_block_weights(build_processor_map(processors))

    assert weights == {"down_blocks.0": 2.0, "up_blocks.3": 1.0}


def test_a0_sets_all_known_blocks_to_same_scale():
    schedule = legacy_variant_schedule("A0_raw_all", 0.9, 30)

    assert set(schedule.block_scales.values()) == {0.9}
    assert set(schedule.time_curve) == {1.0}


def test_a2_only_enables_up_blocks():
    schedule = legacy_variant_schedule("A2_highres_only", 0.9, 30)

    assert schedule.block_scales["down_blocks.0"] == 0.0
    assert schedule.block_scales["mid_block"] == 0.0
    assert schedule.block_scales["up_blocks.0"] == 0.9
    assert schedule.block_scales["up_blocks.3"] == 0.9


def test_t3_late_highres_step_boundaries_for_30_steps():
    schedule = legacy_variant_schedule("T3_late_highres", 0.9, 30)

    assert schedule.time_curve[11] == 0.0
    assert schedule.time_curve[12] == 0.45
    assert schedule.time_curve[20] == 0.45
    assert schedule.time_curve[21] == 1.0


def test_five_bin_time_curve_has_expected_30_step_boundaries():
    curve = five_bin_time_curve(30, (0.0, 0.2, 0.4, 0.8, 1.0))

    assert curve[:6] == [0.0] * 6
    assert curve[6:12] == [0.2] * 6
    assert curve[12:18] == [0.4] * 6
    assert curve[18:24] == [0.8] * 6
    assert curve[24:] == [1.0] * 6


def test_scale_area_normalization_matches_target():
    source = legacy_variant_schedule("A0_raw_all", 0.9, 30)
    target = legacy_variant_schedule("A2_highres_only", 0.9, 30)
    weights = {
        "down_blocks.0": 2.0,
        "down_blocks.1": 2.0,
        "down_blocks.2": 2.0,
        "mid_block": 1.0,
        "up_blocks.0": 3.0,
        "up_blocks.1": 3.0,
        "up_blocks.2": 3.0,
        "up_blocks.3": 3.0,
    }

    matched = source.normalized_to_scale_area(target.scale_area(weights), weights)

    assert abs(matched.scale_area(weights) - target.scale_area(weights)) < 1e-9
    assert matched.normalization == "scale_area"


def test_named_a0_scale_area_matched_variant_targets_a2_area():
    weights = {
        "down_blocks.0": 2.0,
        "down_blocks.1": 2.0,
        "down_blocks.2": 2.0,
        "mid_block": 1.0,
        "up_blocks.0": 3.0,
        "up_blocks.1": 3.0,
        "up_blocks.2": 3.0,
        "up_blocks.3": 3.0,
    }
    schedule = build_variant_schedule("A0_all_scale_area_matched_to_A2", 0.9, 30, weights)
    target = build_variant_schedule("A2_highres_only", 0.9, 30, weights)

    assert schedule.name == "A0_all_scale_area_matched_to_A2"
    assert schedule.normalization == "scale_area"
    assert abs(schedule.scale_area(weights) - target.scale_area(weights)) < 1e-9
    assert schedule.block_scales["down_blocks.0"] > 0.0
    assert schedule.block_scales["up_blocks.3"] > 0.0


def test_named_a0_residual_energy_matched_variant_requires_factor():
    try:
        build_variant_schedule("A0_all_residual_energy_matched", 0.9, 30)
    except ValueError as exc:
        assert "residual_energy_scale_factor" in str(exc)
    else:
        raise AssertionError("Expected missing residual factor to fail.")


def test_named_a0_residual_energy_matched_variant_applies_factor():
    schedule = build_variant_schedule("A0_all_residual_energy_matched", 0.9, 30, residual_energy_scale_factor=0.25)

    assert schedule.normalization == "residual_energy"
    assert schedule.block_scales["down_blocks.0"] == 0.225
    assert schedule.block_scales["mid_block"] == 0.225
    assert schedule.block_scales["up_blocks.3"] == 0.225
