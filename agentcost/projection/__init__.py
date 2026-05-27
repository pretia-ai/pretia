"""Cost projection and pattern detection."""

from __future__ import annotations

from agentcost.projection.patterns import DetectedPattern, detect_patterns
from agentcost.projection.stats import (
    PercentileStats,
    ProfilingStats,
    RunStats,
    StepStats,
    compute_percentile_stats,
    compute_stats,
)

__all__ = [
    "DetectedPattern",
    "PercentileStats",
    "ProfilingStats",
    "RunStats",
    "StepStats",
    "compute_percentile_stats",
    "compute_stats",
    "detect_patterns",
]
