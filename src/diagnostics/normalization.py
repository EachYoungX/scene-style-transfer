"""Normalization helpers for IP-Adapter causal diagnostics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


def cumulative_residual_energy(records: Iterable[dict], field: str = "ip_residual_rms") -> float:
    total = 0.0
    for record in records:
        total += float(record[field])
    return total


def estimate_residual_scale_factor(
    source_energy: float,
    target_energy: float,
    min_source_energy: float = 1e-12,
) -> float:
    if source_energy <= min_source_energy:
        raise ValueError("source_energy is too small for residual-energy matching.")
    return float(target_energy) / float(source_energy)


def read_jsonl_records(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def estimate_scale_factor_from_jsonl(
    source_path: Path,
    target_path: Path,
    field: str = "ip_residual_rms",
) -> dict[str, float]:
    source_energy = cumulative_residual_energy(read_jsonl_records(source_path), field)
    target_energy = cumulative_residual_energy(read_jsonl_records(target_path), field)
    scale_factor = estimate_residual_scale_factor(source_energy, target_energy)
    return {
        "source_energy": source_energy,
        "target_energy": target_energy,
        "scale_factor": scale_factor,
    }

