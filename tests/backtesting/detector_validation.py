"""Validate pattern detector firings against the expected activation matrix.

Used by both the pilot (pre-activation scan) and backtest (full validation)
scripts to classify detector results as TP/TN/FP/FN and compute aggregate rates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pretia.projection.patterns import DetectedPattern
from visualization.colors import EXPECTED_DETECTORS, classify_detector_result

logger = logging.getLogger(__name__)

ALL_DETECTORS = [
    "context_growth",
    "loop_count_variance",
    "high_token_variance",
    "step_count_variance",
    "bimodality",
]

_RAW_STAT_EXTRACTORS: dict[str, tuple[str, float]] = {
    # detector -> (attribute_or_evidence_key, threshold)
    "context_growth": ("pearson_r_squared", 0.7),
    "loop_count_variance": ("evidence.cv", 0.5),
    "high_token_variance": ("evidence.p95_p50_ratio", 3.0),
    "step_count_variance": ("step_count_cv", 0.3),
    "bimodality": ("bimodal_bic_delta", 6.0),
}


@dataclass(frozen=True, slots=True)
class DetectorResult:
    """Classification of a single detector firing for one workflow."""

    workflow: str
    detector: str
    fired: bool
    expected: bool
    classification: str  # "TP", "TN", "FP", "FN"
    raw_statistic: float | None
    threshold: float | None


def _strip_workflow_id(workflow_id: str) -> str:
    """Extract the bare workflow ID (e.g. 'W1-support-simple' -> 'W1')."""
    return workflow_id.split("-")[0].upper()


def _extract_raw_statistic(
    detector: str,
    pattern: DetectedPattern | None,
) -> tuple[float | None, float]:
    """Extract the raw test statistic and threshold for a detector.

    Return (raw_statistic, threshold). raw_statistic is None when the
    detector did not fire (no pattern found).
    """
    thresholds = {
        "context_growth": 0.7,
        "loop_count_variance": 0.5,
        "high_token_variance": 3.0,
        "step_count_variance": 0.3,
        "bimodality": 6.0,
    }
    threshold = thresholds[detector]

    if pattern is None:
        return None, threshold

    if detector == "context_growth":
        return pattern.pearson_r_squared, threshold

    if detector == "loop_count_variance":
        return pattern.evidence.get("cv"), threshold

    if detector == "high_token_variance":
        # Prefer cost ratio, fall back to token ratio
        ratio = pattern.evidence.get("p95_p50_ratio_cost")
        if ratio is None:
            ratio = pattern.evidence.get("p95_p50_ratio_tokens")
        return ratio, threshold

    if detector == "step_count_variance":
        return pattern.step_count_cv, threshold

    if detector == "bimodality":
        return pattern.bimodal_bic_delta, threshold

    return None, threshold


def validate_detectors(
    workflow_id: str,
    patterns: list[DetectedPattern],
) -> list[DetectorResult]:
    """Validate all 5 detectors for a workflow against expected activations.

    For each detector, check whether it fired (appears in the patterns list)
    and classify the result using the expected activation matrix.
    """
    wf_key = _strip_workflow_id(workflow_id)

    if wf_key not in EXPECTED_DETECTORS:
        logger.warning(
            "Workflow %s (key=%s) not in EXPECTED_DETECTORS; defaulting all expectations to False",
            workflow_id,
            wf_key,
        )

    expected_map = EXPECTED_DETECTORS.get(wf_key, {})

    # Index patterns by pattern_type for fast lookup.
    # If multiple patterns share a type (e.g. context_growth on two steps),
    # keep the first — we only need to know whether the detector fired at all.
    pattern_by_type: dict[str, DetectedPattern] = {}
    for p in patterns:
        if p.pattern_type not in pattern_by_type:
            pattern_by_type[p.pattern_type] = p

    results: list[DetectorResult] = []
    for detector in ALL_DETECTORS:
        fired = detector in pattern_by_type
        expected = expected_map.get(detector, False)
        classification = classify_detector_result(wf_key, detector, fired)
        matched_pattern = pattern_by_type.get(detector)
        raw_stat, threshold = _extract_raw_statistic(detector, matched_pattern)

        results.append(
            DetectorResult(
                workflow=wf_key,
                detector=detector,
                fired=fired,
                expected=expected,
                classification=classification,
                raw_statistic=raw_stat,
                threshold=threshold,
            )
        )

    return results


def compute_detector_rates(
    all_results: list[DetectorResult],
) -> dict[str, float]:
    """Compute aggregate TP/FN/FP/TN rates across all detector results.

    Returns a dict with keys: tp_rate, fn_rate, fp_rate, tn_rate.
    Rates are fractions in [0, 1]. Division by zero yields 0.0.
    """
    tp = sum(1 for r in all_results if r.classification == "TP")
    fn = sum(1 for r in all_results if r.classification == "FN")
    fp = sum(1 for r in all_results if r.classification == "FP")
    tn = sum(1 for r in all_results if r.classification == "TN")

    total_expected_true = tp + fn
    total_expected_false = fp + tn

    return {
        "tp_rate": tp / total_expected_true if total_expected_true > 0 else 0.0,
        "fn_rate": fn / total_expected_true if total_expected_true > 0 else 0.0,
        "fp_rate": fp / total_expected_false if total_expected_false > 0 else 0.0,
        "tn_rate": tn / total_expected_false if total_expected_false > 0 else 0.0,
    }


def has_false_negatives(results: list[DetectorResult]) -> bool:
    """Return True if any result is a false negative.

    False negatives block a workflow from passing because they mean a
    real cost pattern went undetected.
    """
    return any(r.classification == "FN" for r in results)


def results_to_dict(results: list[DetectorResult]) -> dict[str, dict[str, Any]]:
    """Convert a list of DetectorResults to a dict keyed by detector name.

    Suitable for JSON serialization in backtesting datasets.
    """
    out: dict[str, dict[str, Any]] = {}
    for r in results:
        out[r.detector] = {
            "fired": r.fired,
            "expected": r.expected,
            "classification": r.classification,
            "raw_statistic": r.raw_statistic,
            "threshold": r.threshold,
        }
    return out
