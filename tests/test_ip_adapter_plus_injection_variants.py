from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_ip_adapter_plus_injection_variants import base_layer_scales, layer_group, time_multiplier  # noqa: E402


def test_layer_group_from_processor_name():
    assert layer_group("down_blocks.0.attentions.0.transformer_blocks.0.attn2.processor") == "down"
    assert layer_group("mid_block.attentions.0.transformer_blocks.0.attn2.processor") == "mid"
    assert layer_group("up_blocks.1.attentions.0.transformer_blocks.0.attn2.processor") == "up"


def test_highres_only_closes_down_and_mid():
    scales = base_layer_scales("A2_highres_only", 0.9)

    assert scales["down"] == 0.0
    assert scales["mid"] == 0.0
    assert scales["up"] == 0.9


def test_late_style_time_multiplier():
    assert time_multiplier("T1_late_style", 0, 10) == 0.0
    assert time_multiplier("T1_late_style", 7, 10) == 1.0


def test_gradual_style_time_multiplier():
    assert time_multiplier("T2_gradual_style", 0, 10) == 0.15
    assert time_multiplier("T2_gradual_style", 5, 10) == 0.45
    assert time_multiplier("T2_gradual_style", 8, 10) == 1.0
