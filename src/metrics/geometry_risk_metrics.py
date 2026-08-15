"""Pixel-level metrics for geometry-risk map validation."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def threshold_risk(risk: np.ndarray, threshold: float) -> np.ndarray:
    return np.asarray(risk >= threshold, dtype=bool)


def top_fraction_risk(risk: np.ndarray, top_fraction: float, valid: np.ndarray | None = None) -> tuple[np.ndarray, float]:
    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    valid = np.ones(risk.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    values = risk[valid]
    if values.size == 0:
        raise ValueError("No valid pixels available for quantile threshold")
    cutoff = float(np.quantile(values, 1.0 - top_fraction))
    selected = (risk >= cutoff) & valid
    return selected, cutoff


@dataclass(frozen=True)
class BinaryRiskMetrics:
    failure_coverage: float
    risk_precision: float
    failure_iou: float
    soft_fpr: float
    rigid_recall: float
    risk_fraction: float
    failure_fraction: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def binary_risk_metrics(
    predicted_risk: np.ndarray,
    failure: np.ndarray,
    soft: np.ndarray,
    rigid: np.ndarray,
    valid: np.ndarray | None = None,
) -> BinaryRiskMetrics:
    shape = predicted_risk.shape
    if any(array.shape != shape for array in (failure, soft, rigid)):
        raise ValueError("All metric masks must share one shape")
    valid = np.ones(shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    if valid.shape != shape:
        raise ValueError("Valid mask must share the metric shape")

    risk = np.asarray(predicted_risk, dtype=bool) & valid
    failure = np.asarray(failure, dtype=bool) & valid
    soft = np.asarray(soft, dtype=bool) & valid
    rigid = np.asarray(rigid, dtype=bool) & valid
    intersection = int((risk & failure).sum())
    union = int((risk | failure).sum())
    valid_count = int(valid.sum())
    iou = 1.0 if union == 0 else float(intersection / union)
    return BinaryRiskMetrics(
        failure_coverage=_safe_ratio(intersection, int(failure.sum())),
        risk_precision=_safe_ratio(intersection, int(risk.sum())),
        failure_iou=iou,
        soft_fpr=_safe_ratio(int((risk & soft).sum()), int(soft.sum())),
        rigid_recall=_safe_ratio(int((risk & rigid).sum()), int(rigid.sum())),
        risk_fraction=_safe_ratio(int(risk.sum()), valid_count),
        failure_fraction=_safe_ratio(int(failure.sum()), valid_count),
    )


def _auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=np.float64)
    start = 0
    while start < scores.size:
        end = start + 1
        while end < scores.size and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    positive_rank_sum = float(ranks[labels].sum())
    return (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    true_positives = np.cumsum(ordered_labels, dtype=np.float64)
    ranks = np.arange(1, labels.size + 1, dtype=np.float64)
    return float((true_positives[ordered_labels] / ranks[ordered_labels]).sum() / positives)


@dataclass(frozen=True)
class ContinuousRiskMetrics:
    mean_failure: float
    mean_non_failure: float
    median_failure: float
    median_non_failure: float
    mean_difference: float
    median_difference: float
    cohens_d: float
    auroc: float
    auprc: float
    positive_prevalence: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def continuous_risk_metrics(
    risk: np.ndarray,
    failure: np.ndarray,
    valid: np.ndarray | None = None,
) -> ContinuousRiskMetrics:
    if risk.shape != failure.shape:
        raise ValueError("Risk and failure arrays must share one shape")
    valid = np.ones(risk.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    scores = np.asarray(risk, dtype=np.float64)[valid]
    labels = np.asarray(failure, dtype=bool)[valid]
    if scores.size == 0:
        raise ValueError("No valid pixels available for continuous metrics")
    positive_scores = scores[labels]
    negative_scores = scores[~labels]

    def mean(values: np.ndarray) -> float:
        return float(values.mean()) if values.size else float("nan")

    def median(values: np.ndarray) -> float:
        return float(np.median(values)) if values.size else float("nan")

    mean_positive = mean(positive_scores)
    mean_negative = mean(negative_scores)
    median_positive = median(positive_scores)
    median_negative = median(negative_scores)
    if positive_scores.size > 1 and negative_scores.size > 1:
        pooled_variance = (
            (positive_scores.size - 1) * positive_scores.var(ddof=1)
            + (negative_scores.size - 1) * negative_scores.var(ddof=1)
        ) / (positive_scores.size + negative_scores.size - 2)
        cohens_d = (mean_positive - mean_negative) / np.sqrt(pooled_variance) if pooled_variance > 0 else float("nan")
    else:
        cohens_d = float("nan")
    return ContinuousRiskMetrics(
        mean_failure=mean_positive,
        mean_non_failure=mean_negative,
        median_failure=median_positive,
        median_non_failure=median_negative,
        mean_difference=mean_positive - mean_negative,
        median_difference=median_positive - median_negative,
        cohens_d=float(cohens_d),
        auroc=float(_auroc(scores, labels)),
        auprc=float(_average_precision(scores, labels)),
        positive_prevalence=float(labels.mean()),
    )
