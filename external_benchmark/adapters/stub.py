"""Explicit placeholder used until an official method adapter is implemented."""

from __future__ import annotations

from .base import BenchmarkAdapter, GenerationRequest, GenerationResult


class UnimplementedAdapter(BenchmarkAdapter):
    def __init__(self, method_id: str):
        self.method_id = method_id

    def generate(self, request: GenerationRequest) -> GenerationResult:
        raise NotImplementedError(
            f"{self.method_id} adapter is not implemented; complete smoke-test setup before batch generation."
        )
