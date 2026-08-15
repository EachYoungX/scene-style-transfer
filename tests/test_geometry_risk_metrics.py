from pathlib import Path

import numpy as np
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from metrics.geometry_risk_metrics import (  # noqa: E402
    binary_risk_metrics,
    continuous_risk_metrics,
    threshold_risk,
    top_fraction_risk,
)


def masks(shape=(4, 4)):
    empty = np.zeros(shape, dtype=bool)
    return empty.copy(), empty.copy(), empty.copy()


def test_perfect_overlap_has_unit_coverage_precision_and_iou():
    failure, soft, rigid = masks()
    failure[1:3, 1:3] = True
    rigid[:] = failure
    result = binary_risk_metrics(failure, failure, soft, rigid)
    assert result.failure_coverage == 1.0
    assert result.risk_precision == 1.0
    assert result.failure_iou == 1.0
    assert result.rigid_recall == 1.0


def test_disjoint_masks_have_zero_overlap_metrics():
    failure, soft, rigid = masks()
    failure[:2, :2] = True
    predicted = np.zeros_like(failure)
    predicted[2:, 2:] = True
    result = binary_risk_metrics(predicted, failure, soft, rigid)
    assert result.failure_coverage == 0.0
    assert result.risk_precision == 0.0
    assert result.failure_iou == 0.0


def test_both_empty_masks_have_unit_iou_and_undefined_ratios():
    failure, soft, rigid = masks()
    result = binary_risk_metrics(failure, failure, soft, rigid)
    assert result.failure_iou == 1.0
    assert np.isnan(result.failure_coverage)
    assert np.isnan(result.risk_precision)


def test_continuous_metrics_separate_failures_and_beat_prevalence():
    risk = np.array([[0.1, 0.2], [0.8, 0.9]], dtype=np.float32)
    failure = np.array([[False, False], [True, True]])
    result = continuous_risk_metrics(risk, failure)
    assert result.mean_difference > 0
    assert result.auroc == 1.0
    assert result.auprc == 1.0
    assert result.auprc > result.positive_prevalence


def test_auprc_no_positive_pixels_does_not_crash():
    risk = np.linspace(0, 1, 16, dtype=np.float32).reshape(4, 4)
    result = continuous_risk_metrics(risk, np.zeros((4, 4), dtype=bool))
    assert np.isnan(result.auprc)
    assert np.isnan(result.auroc)


def test_threshold_and_top_fraction_ordering():
    risk = np.arange(100, dtype=np.float32).reshape(10, 10) / 99.0
    assert threshold_risk(risk, 0.7).sum() <= threshold_risk(risk, 0.5).sum()
    top20, cutoff20 = top_fraction_risk(risk, 0.20)
    top50, cutoff50 = top_fraction_risk(risk, 0.50)
    assert top20.sum() <= top50.sum()
    assert cutoff20 >= cutoff50


def test_spatial_tolerance_is_applied_in_metrics_not_ground_truth():
    failure, soft, rigid = masks((9, 9))
    rigid[4, 4] = True
    failure[4, 4] = True
    predicted = np.zeros_like(failure)
    predicted[4, 6] = True
    result = binary_risk_metrics(predicted, failure, soft, rigid, tolerance_radius=2)
    assert result.failure_coverage == 0.0
    assert result.failure_coverage_tolerant == 1.0
    assert result.rigid_recall == 0.0
    assert result.rigid_recall_tolerant == 1.0
