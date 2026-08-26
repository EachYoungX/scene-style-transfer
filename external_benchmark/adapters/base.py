"""Common adapter contract for external benchmark methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GenerationRequest:
    method_id: str
    pair_id: str
    content_path: Path
    reference_path: Path
    output_path: Path
    prompt: str = ""
    negative_prompt: str = ""
    seed: int | None = None
    config: dict[str, Any] = field(default_factory=dict)


@dataclass
class GenerationResult:
    method_id: str
    pair_id: str
    seed: int | None
    output_path: Path
    status: str
    runtime_sec: float | None = None
    peak_vram_mb: float | None = None
    notes: str = ""


class BenchmarkAdapter(ABC):
    """A thin wrapper around one official method inference recipe."""

    method_id: str
    stochastic: bool = False

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate one normalized 512px output and return run metadata."""

    @staticmethod
    def validate_output(path: Path, expected_size: tuple[int, int] = (512, 512)) -> None:
        from PIL import Image

        if not path.exists():
            raise FileNotFoundError(path)
        with Image.open(path) as image:
            if image.size != expected_size:
                raise ValueError(f"Expected {expected_size}, got {image.size}: {path}")
            if image.mode not in {"RGB", "RGBA"}:
                raise ValueError(f"Expected RGB/RGBA output, got {image.mode}: {path}")
