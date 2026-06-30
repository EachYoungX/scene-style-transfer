from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_ip_adapter_plus_variants import fit_square_crop, make_texture_bank_image, transform_embeds  # noqa: E402


def test_pooled_variant_preserves_shape_and_equalizes_patches():
    embed = torch.arange(2 * 1 * 5 * 3, dtype=torch.float32).reshape(2, 1, 5, 3)

    pooled = transform_embeds([embed], "pooled", seed=1)[0]

    assert pooled.shape == embed.shape
    assert torch.allclose(pooled[:, :, 1, :], pooled[:, :, 2, :])


def test_shuffled_variant_preserves_cls_token():
    embed = torch.randn(2, 1, 5, 3)

    shuffled = transform_embeds([embed], "shuffled", seed=1)[0]

    assert torch.allclose(shuffled[:, :, 0, :], embed[:, :, 0, :])


def test_texture_bank_variant_uses_texture_embeds():
    embed = torch.zeros(2, 1, 5, 3)
    texture = torch.ones(2, 1, 5, 3)

    output = transform_embeds([embed], "texture_bank", seed=1, texture_embeds=[texture])[0]

    assert torch.allclose(output, texture)


def test_global_plus_texture_preserves_source_global_token():
    embed = torch.zeros(2, 1, 5, 3)
    embed[:, :, 0, :] = 10
    texture = torch.arange(2 * 1 * 5 * 3, dtype=torch.float32).reshape(2, 1, 5, 3)

    output = transform_embeds([embed], "global_plus_texture", seed=1, texture_embeds=[texture])[0]

    assert torch.allclose(output[:, :, 0, :], embed[:, :, 0, :])
    assert not torch.allclose(output[:, :, 1:, :], embed[:, :, 1:, :])


def test_global_plus_texture_zero_scale_uses_source_patch_mean():
    embed = torch.arange(2 * 1 * 5 * 3, dtype=torch.float32).reshape(2, 1, 5, 3)
    texture = torch.randn(2, 1, 5, 3)

    output = transform_embeds(
        [embed],
        "global_plus_texture",
        seed=1,
        texture_embeds=[texture],
        texture_residual_scale=0.0,
    )[0]

    expected = embed[:, :, 1:, :].mean(dim=2, keepdim=True).expand_as(output[:, :, 1:, :])
    assert torch.allclose(output[:, :, 1:, :], expected)


def test_make_texture_bank_image_returns_requested_size():
    from PIL import Image

    image = Image.new("RGB", (80, 64), (128, 64, 32))

    bank = make_texture_bank_image(image, size=64, patch_sizes=[16, 24], seed=1)

    assert bank.size == (64, 64)


def test_fit_square_crop_removes_non_square_padding_source(tmp_path):
    from PIL import Image

    image_path = tmp_path / "wide.png"
    Image.new("RGB", (80, 40), (255, 255, 255)).save(image_path)

    cropped = fit_square_crop(image_path, size=32)

    assert cropped.size == (32, 32)
