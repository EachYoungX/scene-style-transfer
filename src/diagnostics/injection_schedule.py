"""Configurable IP-Adapter layer and timestep injection schedules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Iterable, Mapping


LEGACY_VARIANTS = (
    "A0_raw_all",
    "A1_lowres_only",
    "A2_highres_only",
    "A3_highres_plus_weak_mid",
    "T1_late_style",
    "T2_gradual_style",
    "T3_late_highres",
)

NORMALIZED_VARIANTS = (
    "A0_all_scale_area_matched_to_A2",
    "A0_all_residual_energy_matched",
)

SUPPORTED_VARIANTS = LEGACY_VARIANTS + NORMALIZED_VARIANTS

BLOCK_ORDER = (
    "down_blocks.0",
    "down_blocks.1",
    "down_blocks.2",
    "mid_block",
    "up_blocks.0",
    "up_blocks.1",
    "up_blocks.2",
    "up_blocks.3",
)

_BLOCK_RE = re.compile(r"^(down_blocks|up_blocks)\.(\d+)\.")


@dataclass(frozen=True)
class ProcessorInfo:
    processor_name: str
    block_group: str
    block_index: int | None
    block: str
    attention_index: int | None
    is_ip_adapter: bool
    feature_resolution: int | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class InjectionSchedule:
    name: str
    block_scales: dict[str, float]
    time_curve: list[float]
    normalization: str = "none"

    def scale_for(self, block: str, step_index: int) -> float:
        if step_index < 0 or step_index >= len(self.time_curve):
            raise IndexError(f"step_index out of range: {step_index}")
        return float(self.block_scales.get(block, 0.0)) * float(self.time_curve[step_index])

    def matrix(self, blocks: Iterable[str] = BLOCK_ORDER) -> dict[str, list[float]]:
        return {block: [self.scale_for(block, i) for i in range(len(self.time_curve))] for block in blocks}

    def scale_area(self, block_weights: Mapping[str, float] | None = None) -> float:
        weights = block_weights or {block: 1.0 for block in self.block_scales}
        return sum(
            float(weights.get(block, 0.0)) * float(block_scale) * float(time_scale)
            for block, block_scale in self.block_scales.items()
            for time_scale in self.time_curve
        )

    def normalized_to_scale_area(self, target_area: float, block_weights: Mapping[str, float] | None = None) -> "InjectionSchedule":
        current_area = self.scale_area(block_weights)
        if current_area == 0:
            raise ValueError("Cannot normalize a zero-area injection schedule.")
        factor = float(target_area) / current_area
        return InjectionSchedule(
            name=f"{self.name}_scale_area_matched",
            block_scales={block: scale * factor for block, scale in self.block_scales.items()},
            time_curve=list(self.time_curve),
            normalization="scale_area",
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def parse_processor_name(processor_name: str, processor: object | None = None) -> ProcessorInfo:
    match = _BLOCK_RE.match(processor_name)
    attention_index = _parse_attention_index(processor_name)
    is_ip_adapter = processor is None or hasattr(processor, "scale")

    if match:
        group = match.group(1)
        index = int(match.group(2))
        return ProcessorInfo(
            processor_name=processor_name,
            block_group="down" if group == "down_blocks" else "up",
            block_index=index,
            block=f"{group}.{index}",
            attention_index=attention_index,
            is_ip_adapter=is_ip_adapter,
        )

    if processor_name.startswith("mid_block"):
        return ProcessorInfo(
            processor_name=processor_name,
            block_group="mid",
            block_index=None,
            block="mid_block",
            attention_index=attention_index,
            is_ip_adapter=is_ip_adapter,
        )

    return ProcessorInfo(
        processor_name=processor_name,
        block_group="other",
        block_index=None,
        block="other",
        attention_index=attention_index,
        is_ip_adapter=is_ip_adapter,
    )


def build_processor_map(attn_processors: Mapping[str, object]) -> list[ProcessorInfo]:
    return [parse_processor_name(name, processor) for name, processor in attn_processors.items()]


def processor_block_weights(processors: Iterable[ProcessorInfo]) -> dict[str, float]:
    weights: dict[str, float] = {}
    for info in processors:
        if not info.is_ip_adapter or info.block == "other":
            continue
        weights[info.block] = weights.get(info.block, 0.0) + 1.0
    return weights


def legacy_layer_group(processor_name: str) -> str:
    return parse_processor_name(processor_name).block_group


def legacy_base_layer_scales(variant: str, base_scale: float) -> dict[str, float]:
    if variant in {"A0_raw_all", "T1_late_style", "T2_gradual_style"}:
        return {"down": base_scale, "mid": base_scale, "up": base_scale}
    if variant == "A1_lowres_only":
        return {"down": base_scale, "mid": base_scale, "up": 0.0}
    if variant == "A2_highres_only":
        return {"down": 0.0, "mid": 0.0, "up": base_scale}
    if variant in {"A3_highres_plus_weak_mid", "T3_late_highres"}:
        return {"down": 0.0, "mid": base_scale * 0.25, "up": base_scale}
    raise ValueError(f"Unknown variant: {variant}")


def legacy_time_multiplier(variant: str, step_index: int, num_steps: int) -> float:
    progress = (step_index + 1) / max(num_steps, 1)
    if variant == "T1_late_style":
        return 0.0 if progress <= 0.4 else 1.0
    if variant == "T2_gradual_style":
        if progress <= 0.4:
            return 0.15
        if progress <= 0.7:
            return 0.45
        return 1.0
    if variant == "T3_late_highres":
        if progress <= 0.4:
            return 0.0
        if progress <= 0.7:
            return 0.45
        return 1.0
    return 1.0


def legacy_variant_schedule(variant: str, base_scale: float, num_steps: int) -> InjectionSchedule:
    group_scales = legacy_base_layer_scales(variant, base_scale)
    block_scales: dict[str, float] = {}
    for block in BLOCK_ORDER:
        group = "mid" if block == "mid_block" else block.split("_blocks", 1)[0]
        block_scales[block] = group_scales[group]

    return InjectionSchedule(
        name=variant,
        block_scales=block_scales,
        time_curve=[legacy_time_multiplier(variant, step, num_steps) for step in range(num_steps)],
    )


def build_variant_schedule(
    variant: str,
    base_scale: float,
    num_steps: int,
    block_weights: Mapping[str, float] | None = None,
    residual_energy_scale_factor: float | None = None,
) -> InjectionSchedule:
    if variant in LEGACY_VARIANTS:
        return legacy_variant_schedule(variant, base_scale, num_steps)

    if variant == "A0_all_scale_area_matched_to_A2":
        source = legacy_variant_schedule("A0_raw_all", base_scale, num_steps)
        target = legacy_variant_schedule("A2_highres_only", base_scale, num_steps)
        matched = source.normalized_to_scale_area(target.scale_area(block_weights), block_weights)
        return InjectionSchedule(
            name=variant,
            block_scales=matched.block_scales,
            time_curve=matched.time_curve,
            normalization=matched.normalization,
        )

    if variant == "A0_all_residual_energy_matched":
        if residual_energy_scale_factor is None:
            raise ValueError("A0_all_residual_energy_matched requires residual_energy_scale_factor.")
        source = legacy_variant_schedule("A0_raw_all", base_scale * residual_energy_scale_factor, num_steps)
        return InjectionSchedule(
            name=variant,
            block_scales=source.block_scales,
            time_curve=source.time_curve,
            normalization="residual_energy",
        )

    raise ValueError(f"Unknown variant: {variant}")


def five_bin_time_curve(num_steps: int, values: tuple[float, float, float, float, float]) -> list[float]:
    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if len(values) != 5:
        raise ValueError("values must contain exactly five entries.")
    curve: list[float] = []
    for step in range(num_steps):
        progress = (step + 1) / num_steps
        bin_index = min(4, int(progress * 5 - 1e-12))
        curve.append(float(values[bin_index]))
    return curve


def _parse_attention_index(processor_name: str) -> int | None:
    marker = ".attentions."
    if marker not in processor_name:
        return None
    tail = processor_name.split(marker, 1)[1]
    raw = tail.split(".", 1)[0]
    return int(raw) if raw.isdigit() else None
