from pathlib import Path

import numpy as np
from PIL import Image
import pytest

import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))
from run_baseline import RunConfig, fit_square, make_canny, run  # noqa: E402


def test_fit_square_pads_to_requested_size(tmp_path):
    source = tmp_path / "wide.png"
    Image.new("RGB", (80, 40), (255, 0, 0)).save(source)

    result = fit_square(source, 64)

    assert result.size == (64, 64)


def test_make_canny_returns_rgb_edge_image():
    image = Image.fromarray(np.full((64, 64, 3), 255, dtype=np.uint8))

    result = make_canny(image)

    assert result.mode == "RGB"
    assert result.size == (64, 64)


def test_ip_adapter_requires_style(tmp_path, monkeypatch):
    content = tmp_path / "content.png"
    Image.new("RGB", (16, 16), (255, 255, 255)).save(content)
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    config = RunConfig(
        method="ip_adapter_canny",
        content="content.png",
        style=None,
        prompt="x",
        negative_prompt="",
        seed=1,
        num_inference_steps=1,
        guidance_scale=1.0,
        strength=0.5,
        controlnet_scale=0.8,
        ip_adapter_scale=0.45,
        size=16,
        model_dir="models/sd15",
        controlnet_dir="models/controlnet_canny",
        ip_adapter_dir="models/ip_adapter",
    )

    with pytest.raises(ValueError, match="--style is required"):
        run(config, tmp_path)
