from pathlib import Path
import sys

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from diagnostics.normalization import (  # noqa: E402
    cumulative_residual_energy,
    estimate_residual_scale_factor,
)


def test_cumulative_residual_energy_sums_selected_field():
    records = [
        {"ip_residual_rms": 1.5, "ip_residual_l2": 10.0},
        {"ip_residual_rms": 2.5, "ip_residual_l2": 20.0},
    ]

    assert cumulative_residual_energy(records) == 4.0
    assert cumulative_residual_energy(records, "ip_residual_l2") == 30.0


def test_estimate_residual_scale_factor_matches_target_over_source():
    assert estimate_residual_scale_factor(source_energy=8.0, target_energy=2.0) == 0.25


def test_estimate_residual_scale_factor_rejects_zero_source():
    with pytest.raises(ValueError):
        estimate_residual_scale_factor(source_energy=0.0, target_energy=2.0)

