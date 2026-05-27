"""Projection quality validation and backtesting."""

from __future__ import annotations

from agentcost.validation.confidence import ConfidenceResult, compute_confidence
from agentcost.validation.scoring import (
    CalibrationScore,
    format_calibration_report,
    score_projection,
)
from agentcost.validation.suite import (
    BacktestConfig,
    BacktestResult,
    BacktestSuiteResult,
    format_suite_report,
    run_backtesting_suite,
)
from agentcost.validation.validate_cmd import (
    ValidateResult,
    format_validate_report,
    run_validation,
)

__all__ = [
    "BacktestConfig",
    "BacktestResult",
    "BacktestSuiteResult",
    "CalibrationScore",
    "ConfidenceResult",
    "ValidateResult",
    "compute_confidence",
    "format_calibration_report",
    "format_suite_report",
    "format_validate_report",
    "run_backtesting_suite",
    "run_validation",
    "score_projection",
]
