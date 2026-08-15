"""Instrumented IP-Adapter attention processors for residual-energy logging."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, List, Optional

import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import (
    Attention,
    IPAdapterAttnProcessor2_0,
    IPAdapterMaskProcessor,
    deprecate,
)


@dataclass(frozen=True)
class ResidualRecord:
    processor_name: str
    step: int | None
    timestep: int | float | None
    scale: float
    ip_residual_l2: float
    ip_residual_rms: float
    base_hidden_rms: float
    relative_ip_energy: float
    spatial_gate_mean: float | None = None
    spatial_gate_suppressed_fraction: float | None = None
    spatial_gate_height: int | None = None
    spatial_gate_width: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResidualLogger:
    def __init__(self) -> None:
        self.step: int | None = None
        self.timestep: int | float | None = None
        self.records: list[ResidualRecord] = []

    def set_step(self, step: int | None, timestep: int | float | None) -> None:
        self.step = step
        self.timestep = timestep

    def log(self, record: ResidualRecord) -> None:
        self.records.append(record)

    def clear(self) -> None:
        self.records.clear()

    def to_dicts(self) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self.records]


class InstrumentedIPAdapterAttnProcessor2_0(IPAdapterAttnProcessor2_0):
    """Diffusers 0.35.2 IP-Adapter processor with scalar IP residual logging."""

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int | None = None,
        num_tokens: int | tuple[int, ...] | list[int] = (4,),
        scale: float | list[float] = 1.0,
        processor_name: str = "",
        residual_logger: ResidualLogger | None = None,
        enable_logging: bool = True,
        spatial_gate: torch.Tensor | None = None,
    ) -> None:
        super().__init__(
            hidden_size=hidden_size,
            cross_attention_dim=cross_attention_dim,
            num_tokens=num_tokens,
            scale=scale,
        )
        self.processor_name = processor_name
        self.residual_logger = residual_logger
        self.enable_logging = enable_logging
        self.spatial_gate = spatial_gate
        self._last_gate_mean: float | None = None
        self._last_gate_suppressed_fraction: float | None = None
        self._last_gate_height: int | None = None
        self._last_gate_width: int | None = None

    @classmethod
    def from_processor(
        cls,
        processor: IPAdapterAttnProcessor2_0,
        processor_name: str = "",
        residual_logger: ResidualLogger | None = None,
        enable_logging: bool = True,
        spatial_gate: torch.Tensor | None = None,
    ) -> "InstrumentedIPAdapterAttnProcessor2_0":
        instrumented = cls(
            hidden_size=processor.hidden_size,
            cross_attention_dim=processor.cross_attention_dim,
            num_tokens=list(processor.num_tokens),
            scale=list(processor.scale),
            processor_name=processor_name,
            residual_logger=residual_logger,
            enable_logging=enable_logging,
            spatial_gate=spatial_gate,
        )
        instrumented.load_state_dict(processor.state_dict())
        reference_param = next(processor.parameters(), None)
        if reference_param is not None:
            instrumented.to(device=reference_param.device, dtype=reference_param.dtype)
        return instrumented

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        scale: float = 1.0,
        ip_adapter_masks: Optional[torch.Tensor] = None,
    ):
        residual = hidden_states

        if encoder_hidden_states is not None:
            if isinstance(encoder_hidden_states, tuple):
                encoder_hidden_states, ip_hidden_states = encoder_hidden_states
            else:
                deprecation_message = (
                    "You have passed a tensor as `encoder_hidden_states`. This is deprecated and will be removed in a future release."
                    " Please make sure to update your script to pass `encoder_hidden_states` as a tuple to suppress this warning."
                )
                deprecate("encoder_hidden_states not a tuple", "1.0.0", deprecation_message, standard_warn=False)
                end_pos = encoder_hidden_states.shape[1] - self.num_tokens[0]
                encoder_hidden_states, ip_hidden_states = (
                    encoder_hidden_states[:, :end_pos, :],
                    [encoder_hidden_states[:, end_pos:, :]],
                )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        height = width = None

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)
        base_hidden_states = hidden_states

        if ip_adapter_masks is not None:
            if not isinstance(ip_adapter_masks, List):
                ip_adapter_masks = list(ip_adapter_masks.unsqueeze(1))
            if not (len(ip_adapter_masks) == len(self.scale) == len(ip_hidden_states)):
                raise ValueError(
                    f"Length of ip_adapter_masks array ({len(ip_adapter_masks)}) must match "
                    f"length of self.scale array ({len(self.scale)}) and number of ip_hidden_states "
                    f"({len(ip_hidden_states)})"
                )
            else:
                for index, (mask, scale, ip_state) in enumerate(zip(ip_adapter_masks, self.scale, ip_hidden_states)):
                    if mask is None:
                        continue
                    if not isinstance(mask, torch.Tensor) or mask.ndim != 4:
                        raise ValueError(
                            "Each element of the ip_adapter_masks array should be a tensor with shape "
                            "[1, num_images_for_ip_adapter, height, width]."
                            " Please use `IPAdapterMaskProcessor` to preprocess your mask"
                        )
                    if mask.shape[1] != ip_state.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of ip images ({ip_state.shape[1]}) at index {index}"
                        )
                    if isinstance(scale, list) and not len(scale) == mask.shape[1]:
                        raise ValueError(
                            f"Number of masks ({mask.shape[1]}) does not match "
                            f"number of scales ({len(scale)}) at index {index}"
                        )
        else:
            ip_adapter_masks = [None] * len(self.scale)

        for current_ip_hidden_states, scale, to_k_ip, to_v_ip, mask in zip(
            ip_hidden_states, self.scale, self.to_k_ip, self.to_v_ip, ip_adapter_masks
        ):
            skip = False
            if isinstance(scale, list):
                if all(s == 0 for s in scale):
                    skip = True
            elif scale == 0:
                skip = True
            if not skip:
                if mask is not None:
                    if not isinstance(scale, list):
                        scale = [scale] * mask.shape[1]

                    current_num_images = mask.shape[1]
                    for i in range(current_num_images):
                        ip_key = to_k_ip(current_ip_hidden_states[:, i, :, :])
                        ip_value = to_v_ip(current_ip_hidden_states[:, i, :, :])

                        ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                        ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                        _current_ip_hidden_states = F.scaled_dot_product_attention(
                            query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                        )

                        _current_ip_hidden_states = _current_ip_hidden_states.transpose(1, 2).reshape(
                            batch_size, -1, attn.heads * head_dim
                        )
                        _current_ip_hidden_states = _current_ip_hidden_states.to(query.dtype)

                        mask_downsample = IPAdapterMaskProcessor.downsample(
                            mask[:, i, :, :],
                            batch_size,
                            _current_ip_hidden_states.shape[1],
                            _current_ip_hidden_states.shape[2],
                        )

                        mask_downsample = mask_downsample.to(dtype=query.dtype, device=query.device)
                        ip_delta = scale[i] * (_current_ip_hidden_states * mask_downsample)
                        ip_delta = self._apply_spatial_gate(ip_delta, input_ndim, height, width)
                        self._log_ip_delta(ip_delta, base_hidden_states, float(scale[i]))
                        hidden_states = hidden_states + ip_delta
                else:
                    ip_key = to_k_ip(current_ip_hidden_states)
                    ip_value = to_v_ip(current_ip_hidden_states)

                    ip_key = ip_key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
                    ip_value = ip_value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

                    current_ip_hidden_states = F.scaled_dot_product_attention(
                        query, ip_key, ip_value, attn_mask=None, dropout_p=0.0, is_causal=False
                    )

                    current_ip_hidden_states = current_ip_hidden_states.transpose(1, 2).reshape(
                        batch_size, -1, attn.heads * head_dim
                    )
                    current_ip_hidden_states = current_ip_hidden_states.to(query.dtype)

                    ip_delta = scale * current_ip_hidden_states
                    ip_delta = self._apply_spatial_gate(ip_delta, input_ndim, height, width)
                    self._log_ip_delta(ip_delta, base_hidden_states, float(scale))
                    hidden_states = hidden_states + ip_delta

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states

    def _apply_spatial_gate(
        self,
        ip_delta: torch.Tensor,
        input_ndim: int,
        height: int | None = None,
        width: int | None = None,
    ) -> torch.Tensor:
        """Apply a nearest-neighbor spatial gate to the image-branch residual only."""
        if self.spatial_gate is None:
            return ip_delta
        if input_ndim == 3:
            side = int(math.isqrt(ip_delta.shape[1]))
            if side * side != ip_delta.shape[1]:
                raise ValueError(
                    "Rigid-only spatial gating requires square token grids when attention hidden states are 3-D."
                )
            height = width = side
        elif input_ndim != 4 or height is None or width is None:
            raise ValueError("Rigid-only spatial gating requires image-shaped attention hidden states.")

        gate = self.spatial_gate
        if gate.ndim == 2:
            gate = gate.unsqueeze(0).unsqueeze(0)
        elif gate.ndim == 3:
            gate = gate.unsqueeze(0)
        if gate.ndim != 4 or gate.shape[0] != 1 or gate.shape[1] != 1:
            raise ValueError("spatial_gate must have shape [H, W], [1, H, W], or [1, 1, H, W].")
        # Rigid GT is intentionally a 1–2 px line at 512px. Ordinary nearest
        # sampling can miss such a line completely on 64/32/16/8 token grids.
        # Minimum pooling preserves a lower retain ratio if any source pixel in
        # the corresponding token cell is rigid, without changing the saved GT.
        gate = -F.adaptive_max_pool2d(
            -gate.to(device=ip_delta.device, dtype=ip_delta.dtype),
            output_size=(height, width),
        ).reshape(1, height * width, 1)
        with torch.no_grad():
            self._last_gate_mean = float(gate.float().mean().item())
            self._last_gate_suppressed_fraction = float((gate.float() < 0.999).float().mean().item())
            self._last_gate_height = height
            self._last_gate_width = width
        if gate.shape[1] != ip_delta.shape[1]:
            raise ValueError(
                f"Spatial gate token count {gate.shape[1]} does not match IP residual token count "
                f"{ip_delta.shape[1]} ({height}x{width})."
            )
        return ip_delta * gate

    def _log_ip_delta(self, ip_delta: torch.Tensor, base_hidden_states: torch.Tensor, scale: float) -> None:
        if not self.enable_logging or self.residual_logger is None:
            return
        with torch.no_grad():
            ip_float = ip_delta.detach().float()
            base_float = base_hidden_states.detach().float()
            ip_residual_l2 = torch.linalg.vector_norm(ip_float).item()
            ip_residual_rms = torch.sqrt(torch.mean(ip_float * ip_float)).item()
            base_hidden_rms = torch.sqrt(torch.mean(base_float * base_float)).item()
            relative_ip_energy = ip_residual_rms / max(base_hidden_rms, 1e-12)
        self.residual_logger.log(
            ResidualRecord(
                processor_name=self.processor_name,
                step=self.residual_logger.step,
                timestep=self.residual_logger.timestep,
                scale=scale,
                ip_residual_l2=ip_residual_l2,
                ip_residual_rms=ip_residual_rms,
                base_hidden_rms=base_hidden_rms,
                relative_ip_energy=relative_ip_energy,
                spatial_gate_mean=self._last_gate_mean,
                spatial_gate_suppressed_fraction=self._last_gate_suppressed_fraction,
                spatial_gate_height=self._last_gate_height,
                spatial_gate_width=self._last_gate_width,
            )
        )


def instrument_ip_adapter_processors(
    pipe: Any,
    residual_logger: ResidualLogger | None = None,
    *,
    spatial_gate: torch.Tensor | None = None,
    enable_logging: bool = True,
) -> list[str]:
    replaced: list[str] = []
    processors = dict(pipe.unet.attn_processors)
    for name, processor in processors.items():
        if not isinstance(processor, IPAdapterAttnProcessor2_0):
            continue
        processors[name] = InstrumentedIPAdapterAttnProcessor2_0.from_processor(
            processor,
            processor_name=name,
            residual_logger=residual_logger,
            enable_logging=enable_logging,
            spatial_gate=spatial_gate,
        )
        replaced.append(name)
    pipe.unet.set_attn_processor(processors)
    return replaced


def set_spatial_gate(pipe: Any, spatial_gate: torch.Tensor) -> None:
    """Update the gate for already instrumented IP-Adapter processors."""
    for processor in pipe.unet.attn_processors.values():
        if isinstance(processor, InstrumentedIPAdapterAttnProcessor2_0):
            processor.spatial_gate = spatial_gate
