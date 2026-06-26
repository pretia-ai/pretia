"""Projection quality validation and backtesting."""

from __future__ import annotations

from pretia.validation.confidence import (
    ConfidenceResult,
    compute_confidence,
    compute_conformal_interval,
)
from pretia.validation.scoring import (
    CalibrationScore,
    ComparisonScore,
    format_calibration_report,
    score_comparison,
    score_projection,
)
from pretia.validation.suite import (
    BacktestConfig,
    BacktestResult,
    BacktestSuiteResult,
    ComparisonResult,
    FailureAttribution,
    attribute_failure,
    format_suite_report,
    run_backtesting_suite,
)
from pretia.validation.validate_cmd import (
    ValidateResult,
    format_validate_report,
    run_validation,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestSuiteResult",
    "CalibrationScore",
    "ComparisonResult",
    "ComparisonScore",
    "ConfidenceResult",
    "FailureAttribution",
    "ValidateResult",
    "attribute_failure",
    "compute_confidence",
    "compute_conformal_interval",
    "format_calibration_report",
    "format_suite_report",
    "format_validate_report",
    "run_backtesting_suite",
    "run_validation",
    "score_comparison",
    "score_projection",
]
