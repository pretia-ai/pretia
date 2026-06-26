"""Unified projection entry point: linear or Monte Carlo, with confidence tiers."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.projection.montecarlo import (
    MonteCarloResult,
    PercentileProjection,
    simulate,
)
from pretia.projection.patterns import DetectedPattern
from pretia.projection.stats import ProfilingStats
from pretia.validation.confidence import ConfidenceResult, compute_confidence

logger = logging.getLogger(__name__)

_DEFAULT_TRAFFIC = [100, 1000, 10000]


@dataclass(frozen=True, slots=True)
class TrafficProjection:
    """Projection for one traffic volume level."""

    daily_volume: int
    monthly_cost: PercentileProjection
    daily_cost: PercentileProjection
    cost_per_run: PercentileProjection

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "daily_volume": self.daily_volume,
            "monthly_cost": self.monthly_cost.to_dict(),
            "daily_cost": self.daily_cost.to_dict(),
            "cost_per_run": self.cost_per_run.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """Full projection output with confidence and method metadata."""

    method: str
    traffic_volumes: list[int]
    projections: dict[int, TrafficProjection]
    confidence: ConfidenceResult
    warnings: list[str] = field(default_factory=list)
    patterns_detected: list[DetectedPattern] = field(default_factory=list)
    montecarlo_result: MonteCarloResult | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "method": self.method,
            "traffic_volumes": list(self.traffic_volumes),
            "projections": {k: v.to_dict() for k, v in self.projections.items()},
            "confidence": self.confidence.to_dict(),
            "warnings": list(self.warnings),
            "patterns_detected": [p.to_dict() for p in self.patterns_detected],
            "montecarlo_result": (
                self.montecarlo_result.to_dict() if self.montecarlo_result else None
            ),
        }


def _linear_project(
    stats: ProfilingStats,
    traffic: list[int],
) -> dict[int, TrafficProjection]:
    """Scale observed percentiles linearly to each traffic volume."""
    cpr = stats.cost_per_run
    if cpr is None:
        zero = PercentileProjection(p50=0, p75=0, p90=0, p95=0, p99=0, mean=0)
        return {
            v: TrafficProjection(
                daily_volume=v,
                monthly_cost=zero,
                daily_cost=zero,
                cost_per_run=zero,
            )
            for v in traffic
        }

    per_run = PercentileProjection(
        p50=cpr.p50,
        p75=cpr.p75,
        p90=cpr.p90,
        p95=cpr.p95,
        p99=cpr.p99,
        mean=cpr.mean,
    )

    projections: dict[int, TrafficProjection] = {}
    for v in traffic:
        daily = PercentileProjection(
            p50=cpr.p50 * v,
            p75=cpr.p75 * v,
            p90=cpr.p90 * v,
            p95=cpr.p95 * v,
            p99=cpr.p99 * v,
            mean=cpr.mean * v,
        )
        monthly = PercentileProjection(
            p50=cpr.p50 * v * 30,
            p75=cpr.p75 * v * 30,
            p90=cpr.p90 * v * 30,
            p95=cpr.p95 * v * 30,
            p99=cpr.p99 * v * 30,
            mean=cpr.mean * v * 30,
        )
        projections[v] = TrafficProjection(
            daily_volume=v,
            monthly_cost=monthly,
            daily_cost=daily,
            cost_per_run=per_run,
        )

    return projections


def _montecarlo_project(
    stats: ProfilingStats,
    patterns: list[DetectedPattern],
    traffic: list[int],
    runs: list[list[StepRecord]],
) -> tuple[dict[int, TrafficProjection], dict[int, MonteCarloResult]]:
    """Run Monte Carlo simulation for each traffic volume."""
    projections: dict[int, TrafficProjection] = {}
    mc_results: dict[int, MonteCarloResult] = {}

    for v in traffic:
        mc = simulate(stats, patterns, daily_volume=v, runs=runs)
        mc_results[v] = mc
        projections[v] = TrafficProjection(
            daily_volume=v,
            monthly_cost=mc.monthly_projection,
            daily_cost=mc.daily_projection,
            cost_per_run=mc.per_run_projection,
        )

    return projections, mc_results


def project(
    stats: ProfilingStats,
    patterns: list[DetectedPattern],
    traffic: list[int] | None = None,
    runs: list[list[StepRecord]] | None = None,
    input_source: str = "auto-generate",
) -> ProjectionResult:
    """Produce cost projections using the best available method."""
    if traffic is None:
        traffic = list(_DEFAULT_TRAFFIC)

    run_costs = [rs.total_cost for rs in stats.run_stats] if stats.run_stats else None
    default_traffic = traffic[0] if traffic else 1000
    confidence = compute_confidence(
        stats.total_runs,
        stats.step_stats,
        patterns,
        input_source,
        run_costs=run_costs,
        traffic=default_traffic,
    )

    use_montecarlo = len(patterns) > 0
    warnings: list[str] = []
    mc_result: MonteCarloResult | None = None

    if use_montecarlo:
        if runs is None:
            warnings.append(
                "Monte Carlo requested but raw run data not available. "
                "Falling back to linear projection."
            )
            method = "linear"
            projections = _linear_project(stats, traffic)
        else:
            method = "montecarlo"
            for p in patterns:
                if p.severity == "danger":
                    warnings.append(f"Monte Carlo triggered by: {p.description}")
            projections, mc_results = _montecarlo_project(
                stats,
                patterns,
                traffic,
                runs,
            )
            first_volume = traffic[0] if traffic else None
            if first_volume is not None and first_volume in mc_results:
                mc_result = mc_results[first_volume]
                if not mc_result.convergence_check:
                    warnings.append(
                        "Monte Carlo may not have converged. Consider increasing sample count."
                    )
    else:
        method = "linear"
        warnings.append("Linear projection used. No significant non-linear patterns detected.")
        projections = _linear_project(stats, traffic)

    return ProjectionResult(
        method=method,
        traffic_volumes=traffic,
        projections=projections,
        confidence=confidence,
        warnings=warnings,
        patterns_detected=list(patterns),
        montecarlo_result=mc_result,
    )
