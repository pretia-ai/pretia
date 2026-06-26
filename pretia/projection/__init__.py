"""Cost projection and pattern detection."""

from __future__ import annotations

from pretia.projection.montecarlo import MonteCarloResult, PercentileProjection, simulate
from pretia.projection.patterns import DetectedPattern, detect_patterns
from pretia.projection.projector import (
    ProjectionResult,
    TrafficProjection,
    project,
)
from pretia.projection.stats import (
    PercentileStats,
    ProfilingStats,
    RunStats,
    StepStats,
    compute_percentile_stats,
    compute_stats,
)

__all__ = [
    "DetectedPattern",
    "MonteCarloResult",
    "PercentileProjection",
    "PercentileStats",
    "ProfilingStats",
    "ProjectionResult",
    "RunStats",
    "StepStats",
    "TrafficProjection",
    "compute_percentile_stats",
    "compute_stats",
    "detect_patterns",
    "project",
    "simulate",
]
