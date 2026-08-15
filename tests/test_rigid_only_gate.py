from __future__ import annotations

from pathlib import Path
import sys

import torch

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics.ip_adapter_instrumentation import InstrumentedIPAdapterAttnProcessor2_0


def _processor(gate: torch.Tensor) -> InstrumentedIPAdapterAttnProcessor2_0:
    processor = InstrumentedIPAdapterAttnProcessor2_0(hidden_size=4, cross_attention_dim=4)
    processor.spatial_gate = gate
    return processor


def test_rigid_only_gate_uses_nearest_neighbor_on_flattened_grid():
    gate = torch.ones(4, 4)
    gate[:, :2] = 0.0
    ip_delta = torch.ones(1, 16, 3)

    gated = _processor(gate)._apply_spatial_gate(ip_delta, input_ndim=3)

    expected = gate.reshape(1, 16, 1).expand(1, 16, 3)
    assert torch.equal(gated, expected)


def test_empty_rigid_gate_is_identity():
    ip_delta = torch.randn(1, 64, 5)
    gated = _processor(torch.ones(8, 8))._apply_spatial_gate(ip_delta, input_ndim=3)

    assert torch.equal(gated, ip_delta)


def test_thin_rigid_pixel_is_preserved_when_grid_is_coarsened():
    gate = torch.ones(8, 8)
    gate[3, 3] = 0.0
    ip_delta = torch.ones(1, 4, 1)

    gated = _processor(gate)._apply_spatial_gate(ip_delta, input_ndim=3)

    assert gated[0, 0, 0] == 0.0
    assert gated[0, 1:, 0].eq(1.0).all()


def test_non_square_flattened_grid_fails_closed():
    processor = _processor(torch.ones(4, 4))
    ip_delta = torch.ones(1, 15, 2)

    try:
        processor._apply_spatial_gate(ip_delta, input_ndim=3)
    except ValueError as exc:
        assert "square token grids" in str(exc)
    else:
        raise AssertionError("Non-square token grids must not be spatially gated silently")
