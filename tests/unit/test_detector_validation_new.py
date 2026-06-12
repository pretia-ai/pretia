"""Test detector validation logic from tests/backtesting/detector_validation.py."""

from __future__ import annotations

from typing import Any

import pytest

from agentcost.projection.patterns import DetectedPattern
from tests.backtesting.detector_validation import (
    DetectorResult,
    _strip_workflow_id,
    compute_detector_rates,
    has_false_negatives,
    results_to_dict,
    validate_detectors,
)


def _make_pattern(
    pattern_type: str,
    step_name: str = "test_step",
    severity: str = "warning",
    **kwargs: Any,
) -> DetectedPattern:
    """Create a DetectedPattern with minimal required fields."""
    return DetectedPattern(
        pattern_type=pattern_type,
        step_name=step_name,
        severity=severity,
        evidence=kwargs.pop("evidence", {}),
        description=kwargs.pop("description", f"Test {pattern_type} pattern"),
        **kwargs,
    )


def _make_result(
    workflow: str,
    detector: str,
    fired: bool,
    expected: bool,
    classification: str,
    raw_statistic: float | None = None,
    threshold: float | None = None,
) -> DetectorResult:
    """Create a DetectorResult directly."""
    return DetectorResult(
        workflow=workflow,
        detector=detector,
        fired=fired,
        expected=expected,
        classification=classification,
        raw_statistic=raw_statistic,
        threshold=threshold,
    )


# ---------------------------------------------------------------------------
# validate_detectors
# ---------------------------------------------------------------------------


class TestValidateDetectors:
    """Verify detector validation against expected activation matrix."""

    def test_validate_detectors_all_silent(self) -> None:
        """No patterns detected for W1 (all expected False) should yield all TN."""
        results = validate_detectors("W1-support-simple", [])

        assert len(results) == 5
        assert all(r.classification == "TN" for r in results)
        assert all(r.fired is False for r in results)
        assert all(r.workflow == "W1" for r in results)

    def test_validate_detectors_true_positive(self) -> None:
        """Context growth pattern for W2 (expected True) should yield TP."""
        pattern = _make_pattern(
            pattern_type="context_growth",
            pearson_r_squared=0.85,
            evidence={"r_squared": 0.85},
        )
        results = validate_detectors("W2-loop-workflow", [pattern])

        context_result = next(r for r in results if r.detector == "context_growth")
        assert context_result.classification == "TP"
        assert context_result.fired is True
        assert context_result.expected is True
        assert context_result.raw_statistic == 0.85

    def test_validate_detectors_false_negative(self) -> None:
        """W2 expected context_growth but no patterns detected should yield FN."""
        # W2 expects context_growth=True, but we pass no patterns
        results = validate_detectors("W2-loop-workflow", [])

        context_result = next(r for r in results if r.detector == "context_growth")
        assert context_result.classification == "FN"
        assert context_result.fired is False
        assert context_result.expected is True

    def test_validate_detectors_false_positive(self) -> None:
        """Pattern fired for a detector not expected should yield FP."""
        # W1 expects all detectors False
        pattern = _make_pattern(
            pattern_type="context_growth",
            pearson_r_squared=0.9,
            evidence={"r_squared": 0.9},
        )
        results = validate_detectors("W1-support-simple", [pattern])

        context_result = next(r for r in results if r.detector == "context_growth")
        assert context_result.classification == "FP"
        assert context_result.fired is True
        assert context_result.expected is False

    def test_validate_detectors_covers_all_five(self) -> None:
        """Validation should cover all 5 detectors."""
        results = validate_detectors("W1-support-simple", [])
        detectors = {r.detector for r in results}
        expected_detectors = {
            "context_growth",
            "loop_count_variance",
            "high_token_variance",
            "step_count_variance",
            "bimodality",
        }
        assert detectors == expected_detectors

    def test_validate_detectors_multiple_patterns(self) -> None:
        """Multiple fired patterns for W2 should be correctly classified."""
        # W2 expects: context_growth=True, loop_count_variance=True,
        #             step_count_variance=True
        patterns = [
            _make_pattern(
                pattern_type="context_growth",
                pearson_r_squared=0.85,
                evidence={"r_squared": 0.85},
            ),
            _make_pattern(
                pattern_type="loop_count_variance",
                evidence={"cv": 0.8},
            ),
            _make_pattern(
                pattern_type="step_count_variance",
                step_count_cv=0.5,
                evidence={"cv": 0.5},
            ),
        ]
        results = validate_detectors("W2-loop-workflow", patterns)

        classifications = {r.detector: r.classification for r in results}
        assert classifications["context_growth"] == "TP"
        assert classifications["loop_count_variance"] == "TP"
        assert classifications["step_count_variance"] == "TP"
        # high_token_variance and bimodality are expected False for W2 and not fired
        assert classifications["high_token_variance"] == "TN"
        assert classifications["bimodality"] == "TN"


# ---------------------------------------------------------------------------
# compute_detector_rates
# ---------------------------------------------------------------------------


class TestComputeDetectorRates:
    """Verify aggregate rate computation."""

    def test_compute_detector_rates_all_tn(self) -> None:
        """All TN results should yield tp_rate=0 and fn_rate=0."""
        results = [
            _make_result("W1", "context_growth", False, False, "TN"),
            _make_result("W1", "loop_count_variance", False, False, "TN"),
            _make_result("W1", "high_token_variance", False, False, "TN"),
            _make_result("W1", "step_count_variance", False, False, "TN"),
            _make_result("W1", "bimodality", False, False, "TN"),
        ]
        rates = compute_detector_rates(results)

        assert rates["tp_rate"] == 0.0
        assert rates["fn_rate"] == 0.0
        assert rates["fp_rate"] == 0.0
        assert rates["tn_rate"] == 1.0

    def test_compute_detector_rates_mixed(self) -> None:
        """Mix of TP/FN/TN should produce correct rates."""
        results = [
            # W2: context_growth expected=True, fired=True -> TP
            _make_result("W2", "context_growth", True, True, "TP"),
            # W2: loop_count_variance expected=True, fired=False -> FN
            _make_result("W2", "loop_count_variance", False, True, "FN"),
            # W2: high_token_variance expected=False, fired=False -> TN
            _make_result("W2", "high_token_variance", False, False, "TN"),
            # W2: step_count_variance expected=True, fired=True -> TP
            _make_result("W2", "step_count_variance", True, True, "TP"),
            # W2: bimodality expected=False, fired=False -> TN
            _make_result("W2", "bimodality", False, False, "TN"),
        ]
        rates = compute_detector_rates(results)

        # TP=2, FN=1 -> tp_rate = 2/3, fn_rate = 1/3
        assert rates["tp_rate"] == pytest.approx(2 / 3)
        assert rates["fn_rate"] == pytest.approx(1 / 3)
        # FP=0, TN=2 -> fp_rate = 0, tn_rate = 1.0
        assert rates["fp_rate"] == 0.0
        assert rates["tn_rate"] == 1.0

    def test_compute_detector_rates_no_expected_true(self) -> None:
        """When no expected=True, tp_rate and fn_rate should be 0."""
        results = [
            _make_result("W1", "context_growth", False, False, "TN"),
        ]
        rates = compute_detector_rates(results)

        assert rates["tp_rate"] == 0.0
        assert rates["fn_rate"] == 0.0


# ---------------------------------------------------------------------------
# has_false_negatives
# ---------------------------------------------------------------------------


class TestHasFalseNegatives:
    """Verify false negative detection."""

    def test_has_false_negatives_true(self) -> None:
        """FN present should return True."""
        results = [
            _make_result("W2", "context_growth", False, True, "FN"),
            _make_result("W2", "loop_count_variance", True, True, "TP"),
        ]
        assert has_false_negatives(results) is True

    def test_has_false_negatives_false(self) -> None:
        """No FN should return False."""
        results = [
            _make_result("W2", "context_growth", True, True, "TP"),
            _make_result("W1", "loop_count_variance", False, False, "TN"),
        ]
        assert has_false_negatives(results) is False

    def test_has_false_negatives_empty(self) -> None:
        """Empty results should return False."""
        assert has_false_negatives([]) is False


# ---------------------------------------------------------------------------
# results_to_dict
# ---------------------------------------------------------------------------


class TestResultsToDict:
    """Verify dict serialization of detector results."""

    def test_results_to_dict_structure(self) -> None:
        """Dict should have correct keys and types."""
        results = [
            _make_result("W1", "context_growth", False, False, "TN", None, 0.7),
            _make_result("W1", "loop_count_variance", True, False, "FP", 0.65, 0.5),
        ]
        d = results_to_dict(results)

        assert isinstance(d, dict)
        assert set(d.keys()) == {"context_growth", "loop_count_variance"}

        cg = d["context_growth"]
        assert cg["fired"] is False
        assert cg["expected"] is False
        assert cg["classification"] == "TN"
        assert cg["raw_statistic"] is None
        assert cg["threshold"] == 0.7

        lcv = d["loop_count_variance"]
        assert lcv["fired"] is True
        assert lcv["expected"] is False
        assert lcv["classification"] == "FP"
        assert lcv["raw_statistic"] == 0.65
        assert lcv["threshold"] == 0.5

    def test_results_to_dict_all_five_detectors(self) -> None:
        """All 5 detectors should appear as keys when all are present."""
        results = [
            _make_result("W1", det, False, False, "TN")
            for det in [
                "context_growth",
                "loop_count_variance",
                "high_token_variance",
                "step_count_variance",
                "bimodality",
            ]
        ]
        d = results_to_dict(results)
        assert len(d) == 5


# ---------------------------------------------------------------------------
# _strip_workflow_id
# ---------------------------------------------------------------------------


class TestStripWorkflowId:
    """Verify workflow ID extraction."""

    def test_strip_workflow_id_with_suffix(self) -> None:
        """'W1-support-simple' should strip to 'W1'."""
        assert _strip_workflow_id("W1-support-simple") == "W1"

    def test_strip_workflow_id_with_long_suffix(self) -> None:
        """'W13-routing-agent' should strip to 'W13'."""
        assert _strip_workflow_id("W13-routing-agent") == "W13"

    def test_strip_workflow_id_bare(self) -> None:
        """'W2' alone should remain 'W2'."""
        assert _strip_workflow_id("W2") == "W2"

    def test_strip_workflow_id_lowercase(self) -> None:
        """'w19-multi-turn' should normalize to 'W19'."""
        assert _strip_workflow_id("w19-multi-turn") == "W19"
