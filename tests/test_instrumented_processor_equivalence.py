from pathlib import Path
import sys

import torch
from diffusers.models.attention_processor import Attention, IPAdapterAttnProcessor2_0

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics.ip_adapter_instrumentation import (  # noqa: E402
    InstrumentedIPAdapterAttnProcessor2_0,
    ResidualLogger,
)


def test_instrumented_processor_matches_diffusers_processor_without_logging():
    torch.manual_seed(7)
    hidden_size = 32
    cross_attention_dim = 48
    heads = 4
    num_tokens = 4

    attn = Attention(
        query_dim=hidden_size,
        cross_attention_dim=cross_attention_dim,
        heads=heads,
        dim_head=hidden_size // heads,
        bias=False,
        out_bias=False,
    )
    source = IPAdapterAttnProcessor2_0(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        num_tokens=(num_tokens,),
        scale=0.7,
    )
    instrumented = InstrumentedIPAdapterAttnProcessor2_0.from_processor(
        source,
        processor_name="up_blocks.0.attentions.0.transformer_blocks.0.attn2.processor",
        residual_logger=ResidualLogger(),
        enable_logging=False,
    )

    hidden_states = torch.randn(2, 8, hidden_size)
    encoder_hidden_states = torch.randn(2, 6, cross_attention_dim)
    ip_hidden_states = [torch.randn(2, num_tokens, cross_attention_dim)]

    expected = source(attn, hidden_states, (encoder_hidden_states, ip_hidden_states))
    actual = instrumented(attn, hidden_states, (encoder_hidden_states, ip_hidden_states))

    assert torch.max(torch.abs(expected - actual)).item() < 1e-6
    assert torch.mean(torch.abs(expected - actual)).item() < 1e-7


def test_instrumented_processor_logs_scalar_residuals():
    torch.manual_seed(11)
    hidden_size = 16
    cross_attention_dim = 24
    num_tokens = 4
    logger = ResidualLogger()
    logger.set_step(3, 812)

    attn = Attention(
        query_dim=hidden_size,
        cross_attention_dim=cross_attention_dim,
        heads=2,
        dim_head=8,
        bias=False,
        out_bias=False,
    )
    processor = InstrumentedIPAdapterAttnProcessor2_0(
        hidden_size=hidden_size,
        cross_attention_dim=cross_attention_dim,
        num_tokens=(num_tokens,),
        scale=0.5,
        processor_name="mid_block.attentions.0.transformer_blocks.0.attn2.processor",
        residual_logger=logger,
    )

    hidden_states = torch.randn(1, 5, hidden_size)
    encoder_hidden_states = torch.randn(1, 4, cross_attention_dim)
    ip_hidden_states = [torch.randn(1, num_tokens, cross_attention_dim)]

    processor(attn, hidden_states, (encoder_hidden_states, ip_hidden_states))

    assert len(logger.records) == 1
    record = logger.records[0]
    assert record.processor_name.startswith("mid_block")
    assert record.step == 3
    assert record.timestep == 812
    assert record.scale == 0.5
    assert record.ip_residual_l2 > 0.0
    assert record.ip_residual_rms > 0.0
    assert record.base_hidden_rms > 0.0

