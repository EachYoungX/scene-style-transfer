from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from regions.v2_1_purity import build_purity_routes, project_token_purity, route_gate  # noqa: E402


def test_average_pool_purity_partitions_valid_tokens():
    subject = np.zeros((512, 512), dtype=bool)
    background = np.zeros((512, 512), dtype=bool)
    subject[:, :256] = True
    background[:, 256:] = True
    valid = np.ones((512, 512), dtype=bool)

    purity = project_token_purity(subject, background, valid, 16, purity_threshold=0.8)

    assert purity.pure_subject.sum() == 128
    assert purity.pure_background.sum() == 128
    assert purity.mixed.sum() == 0
    assert purity.invalid.sum() == 0
    assert np.allclose(purity.subject_fraction[:, :8], 1.0)
    assert np.allclose(purity.background_fraction[:, 8:], 1.0)


def test_mixed_tokens_are_explicit_and_routes_use_configured_gain():
    subject = np.zeros((512, 512), dtype=bool)
    background = np.zeros((512, 512), dtype=bool)
    subject[:, :240] = True
    background[:, 240:] = True
    valid = np.ones((512, 512), dtype=bool)
    routes = build_purity_routes(subject, background, valid, resolutions=(16,), purity_threshold=0.8)
    purity = routes[16]
    neutral = route_gate(purity, "S_sep_neutral", subject_gain=1.0, background_gain=0.0)
    conservative = route_gate(purity, "S_sep_conservative", subject_gain=1.0, background_gain=0.0)

    assert purity.mixed.sum() > 0
    assert np.all(neutral[purity.mixed] == 1.0)
    assert np.all(conservative[purity.mixed] == 0.0)
    assert np.all(neutral[purity.pure_background] == 0.0)
    assert np.all(neutral[purity.pure_subject] == 1.0)


def test_invalid_tokens_are_zero_gain():
    subject = np.ones((512, 512), dtype=bool)
    background = np.zeros((512, 512), dtype=bool)
    valid = np.ones((512, 512), dtype=bool)
    valid[:32, :32] = False
    purity = project_token_purity(subject, background, valid, 16)
    gate = route_gate(purity, "S_sep_neutral")
    assert np.all(gate[purity.invalid] == 0.0)
