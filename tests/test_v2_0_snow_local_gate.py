from pathlib import Path
import sys

import torch
from diffusers.models.attention_processor import Attention

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics.ip_adapter_instrumentation import (  # noqa: E402
    InstrumentedIPAdapterAttnProcessor2_0,
    ResidualLogger,
)


def _attention(hidden_size: int = 16, cross_attention_dim: int = 24) -> Attention:
    return Attention(
        query_dim=hidden_size,
        cross_attention_dim=cross_attention_dim,
        heads=2,
        dim_head=hidden_size // 2,
        bias=False,
        out_bias=False,
    )


def test_snow_local_rho_zero_logs_raw_to_gated_rigid_suppression():
    torch.manual_seed(23)
    gate = torch.ones(4, 4)
    gate[1, 1] = 0.0
    rigid = torch.zeros(4, 4)
    rigid[1, 1] = 1.0
    logger = ResidualLogger()
    processor = InstrumentedIPAdapterAttnProcessor2_0(
        hidden_size=16,
        cross_attention_dim=24,
        scale=0.5,
        processor_name="up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
        residual_logger=logger,
        spatial_gate=gate,
        audit_rigid_mask=rigid,
        audit_roi=(4, 4, 12, 12),
    )
    logger.set_step(0, 999)

    processor(_attention(), torch.randn(1, 16, 16), (torch.randn(1, 5, 24), [torch.randn(1, 4, 24)]))

    assert len(logger.records) == 1
    record = logger.records[0]
    assert record.raw_ip_residual_rms is not None
    assert record.gated_ip_residual_rms is not None
    assert record.rigid_token_count == 1
    assert record.rigid_related_missed_tokens == 0
    assert record.rigid_rms_ratio is not None and record.rigid_rms_ratio < 1e-4
    assert record.nonrigid_rms_ratio is not None and abs(record.nonrigid_rms_ratio - 1.0) < 1e-6


def test_rho_one_is_identity_in_same_forward_audit():
    torch.manual_seed(29)
    gate = torch.ones(4, 4)
    gate[1, 1] = 1.0
    rigid = torch.zeros(4, 4)
    rigid[1, 1] = 1.0
    logger = ResidualLogger()
    processor = InstrumentedIPAdapterAttnProcessor2_0(
        hidden_size=16,
        cross_attention_dim=24,
        scale=0.5,
        residual_logger=logger,
        spatial_gate=gate,
        audit_rigid_mask=rigid,
    )

    processor(_attention(), torch.randn(1, 16, 16), (torch.randn(1, 5, 24), [torch.randn(1, 4, 24)]))

    record = logger.records[0]
    assert record.global_rms_ratio is not None and abs(record.global_rms_ratio - 1.0) < 1e-6
    assert record.rigid_rms_ratio is not None and abs(record.rigid_rms_ratio - 1.0) < 1e-6


def test_region_pooling_preserves_any_active_source_pixel():
    gate = torch.zeros(4, 4)
    gate[1, 1] = 1.0
    processor = InstrumentedIPAdapterAttnProcessor2_0(
        hidden_size=4,
        cross_attention_dim=4,
        spatial_gate=gate,
        spatial_gate_pooling="maximum",
    )
    ip_delta = torch.ones(1, 1, 1)
    gated = processor._apply_spatial_gate(ip_delta, input_ndim=3)
    assert gated.item() == 1.0
