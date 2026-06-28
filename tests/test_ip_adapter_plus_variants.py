from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "scripts"))

from run_ip_adapter_plus_variants import transform_embeds  # noqa: E402


def test_pooled_variant_preserves_shape_and_equalizes_patches():
    embed = torch.arange(2 * 1 * 5 * 3, dtype=torch.float32).reshape(2, 1, 5, 3)

    pooled = transform_embeds([embed], "pooled", seed=1)[0]

    assert pooled.shape == embed.shape
    assert torch.allclose(pooled[:, :, 1, :], pooled[:, :, 2, :])


def test_shuffled_variant_preserves_cls_token():
    embed = torch.randn(2, 1, 5, 3)

    shuffled = transform_embeds([embed], "shuffled", seed=1)[0]

    assert torch.allclose(shuffled[:, :, 0, :], embed[:, :, 0, :])
