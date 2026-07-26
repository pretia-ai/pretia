"""Unified projection entry point: linear or Monte Carlo, with confidence tiers."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import MODEL_CACHE_HIT_PRICING, MODEL_PRICING, resolve_model
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


def _bootstrap_daily(
    run_costs: list[float],
    daily_volume: int,
    n_sims: int = 5000,
    seed: int = 42,
) -> dict[str, float]:
    """Project daily cost distribution by resampling observed per-run costs."""
    rng = random.Random(seed)  # noqa: S311
    daily_totals = sorted(sum(rng.choices(run_costs, k=daily_volume)) for _ in range(n_sims))
    n = len(daily_totals)
    return {
        "p50": daily_totals[n // 2],
        "p75": daily_totals[int(n * 0.75)],
        "p90": daily_totals[int(n * 0.90)],
        "p95": daily_totals[int(n * 0.95)],
        "p99": daily_totals[int(n * 0.99)],
        "mean": sum(daily_totals) / n,
    }


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
    montecarlo_results: dict[int, MonteCarloResult] = field(default_factory=dict)
    warm_discount: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        first_mc = next(iter(self.montecarlo_results.values()), None)
        d: dict[str, Any] = {
            "method": self.method,
            "traffic_volumes": list(self.traffic_volumes),
            "projections": {k: v.to_dict() for k, v in self.projections.items()},
            "confidence": self.confidence.to_dict(),
            "warnings": list(self.warnings),
            "patterns_detected": [p.to_dict() for p in self.patterns_detected],
            "montecarlo_result": first_mc.to_dict() if first_mc else None,
            "montecarlo_results": {k: v.to_dict() for k, v in self.montecarlo_results.items()},
        }
        if self.warm_discount is not None:
            d["warm_discount"] = self.warm_discount
        return d


def _scenario_project(
    stats: ProfilingStats,
    traffic: list[int],
) -> dict[int, TrafficProjection]:
    """Project monthly costs using spread-adaptive interpolation.

    Each monthly scenario blends the per-run median with higher percentiles,
    dampened by the risk factor (derived from the per-run p95/p50 spread).
    Volatile workflows get wide projections; stable ones stay tight.
    """
    cpr = stats.cost_per_run
    zero = PercentileProjection(p50=0, p75=0, p90=0, p95=0, p99=0, mean=0)
    if cpr is None:
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

    spread = cpr.p95 / cpr.p50 if cpr.p50 > 0 else 1.0
    risk_factor = min(math.log(max(spread, 1.01)) / math.log(5), 1.0)

    def _blend(px: float) -> float:
        return cpr.p50 + (px - cpr.p50) * risk_factor

    projections: dict[int, TrafficProjection] = {}
    for v in traffic:
        n = v * 30
        monthly = PercentileProjection(
            p50=cpr.p50 * n,
            p75=_blend(cpr.p75) * n,
            p90=_blend(cpr.p90) * n,
            p95=_blend(cpr.p95) * n,
            p99=_blend(cpr.p99) * n,
            mean=cpr.mean * n,
        )
        daily = PercentileProjection(
            p50=cpr.p50 * v,
            p75=_blend(cpr.p75) * v,
            p90=_blend(cpr.p90) * v,
            p95=_blend(cpr.p95) * v,
            p99=_blend(cpr.p99) * v,
            mean=cpr.mean * v,
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
    n_simulations: int = 10000,
) -> tuple[dict[int, TrafficProjection], dict[int, MonteCarloResult]]:
    """Run MC simulation for per-run cost modeling, then apply scenario projections.

    MC captures pattern-specific per-run costs (context growth, loops). The
    scenario formula then projects those per-run costs to monthly using
    spread-adaptive interpolation instead of CLT aggregation.
    """
    mc_results: dict[int, MonteCarloResult] = {}

    for v in traffic:
        mc = simulate(stats, patterns, daily_volume=v, runs=runs, n_simulations=n_simulations)
        mc_results[v] = mc

    first_mc = mc_results[traffic[0]] if traffic else None
    mc_per_run = first_mc.per_run_projection if first_mc else None

    if mc_per_run and mc_per_run.p50 > 0:
        spread = mc_per_run.p95 / mc_per_run.p50
        risk_factor = min(math.log(max(spread, 1.01)) / math.log(5), 1.0)

        def _blend(px: float) -> float:
            return mc_per_run.p50 + (px - mc_per_run.p50) * risk_factor

        projections: dict[int, TrafficProjection] = {}
        for v in traffic:
            n = v * 30
            monthly = PercentileProjection(
                p50=mc_per_run.p50 * n,
                p75=_blend(mc_per_run.p75) * n,
                p90=_blend(mc_per_run.p90) * n,
                p95=_blend(mc_per_run.p95) * n,
                p99=_blend(mc_per_run.p99) * n,
                mean=mc_per_run.mean * n,
            )
            daily = PercentileProjection(
                p50=mc_per_run.p50 * v,
                p75=_blend(mc_per_run.p75) * v,
                p90=_blend(mc_per_run.p90) * v,
                p95=_blend(mc_per_run.p95) * v,
                p99=_blend(mc_per_run.p99) * v,
                mean=mc_per_run.mean * v,
            )
            projections[v] = TrafficProjection(
                daily_volume=v,
                monthly_cost=monthly,
                daily_cost=daily,
                cost_per_run=mc_per_run,
            )
    else:
        projections = _scenario_project(stats, traffic)

    return projections, mc_results


def _estimate_warm_discount(
    runs: list[list[StepRecord]] | None,
    stats: ProfilingStats,
) -> float | None:
    """Estimate the cost reduction from prompt caching in production.

    Returns a multiplier (e.g. 0.65 means warm costs are 65% of cold) or None
    if no cacheable models are used.
    """
    if not runs or not stats.step_stats:
        return None

    total_cold_cost = 0.0
    total_warm_cost = 0.0

    for _step_name, ss in stats.step_stats.items():
        if ss.step_type != "llm" or not ss.model:
            continue
        try:
            canonical = resolve_model(ss.model)
        except (ValueError, KeyError):
            continue

        if canonical not in MODEL_PRICING:
            continue

        input_price_per_m, output_price_per_m = MODEL_PRICING[canonical]
        cache_rate = MODEL_CACHE_HIT_PRICING.get(canonical)

        avg_input = ss.input_tokens.mean
        avg_output = ss.output_tokens.mean

        cold_input_cost = avg_input * input_price_per_m / 1_000_000
        cold_output_cost = avg_output * output_price_per_m / 1_000_000
        cold_step_cost = cold_input_cost + cold_output_cost

        if cache_rate is not None and avg_input > 0:
            # Estimate system prompt as ~60% of input tokens (typical for agents)
            cacheable_tokens = avg_input * 0.6
            non_cacheable_tokens = avg_input * 0.4
            warm_input_cost = (
                non_cacheable_tokens * input_price_per_m / 1_000_000
                + cacheable_tokens * cache_rate / 1_000_000
            )
            warm_step_cost = warm_input_cost + cold_output_cost
        else:
            warm_step_cost = cold_step_cost

        call_count = ss.call_count or 1
        total_cold_cost += cold_step_cost * call_count
        total_warm_cost += warm_step_cost * call_count

    if total_cold_cost <= 0:
        return None

    discount = total_warm_cost / total_cold_cost
    if discount >= 0.99:
        return None
    return round(discount, 3)


def project(
    stats: ProfilingStats,
    patterns: list[DetectedPattern],
    traffic: list[int] | None = None,
    runs: list[list[StepRecord]] | None = None,
    input_source: str = "auto-generate",
    n_simulations: int = 10000,
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
    all_mc_results: dict[int, MonteCarloResult] = {}

    if use_montecarlo:
        if runs is None:
            warnings.append(
                "Monte Carlo requested but raw run data not available. "
                "Falling back to linear projection."
            )
            method = "linear"
            projections = _scenario_project(stats, traffic)
        else:
            method = "montecarlo"
            for p in patterns:
                if p.severity == "danger":
                    warnings.append(f"Monte Carlo triggered by: {p.description}")
            projections, all_mc_results = _montecarlo_project(
                stats,
                patterns,
                traffic,
                runs,
                n_simulations=n_simulations,
            )
            first_volume = traffic[0] if traffic else None
            if first_volume is not None and first_volume in all_mc_results:
                if not all_mc_results[first_volume].convergence_check:
                    warnings.append(
                        "Monte Carlo may not have converged. Consider increasing sample count."
                    )
    else:
        method = "linear"
        warnings.append("Linear projection used. No significant non-linear patterns detected.")
        projections = _scenario_project(stats, traffic)

    warm_discount = _estimate_warm_discount(runs, stats)
    if warm_discount is not None:
        warnings.append(
            f"With prompt caching enabled, costs may be "
            f"~{round((1 - warm_discount) * 100)}% lower."
        )

    return ProjectionResult(
        method=method,
        traffic_volumes=traffic,
        projections=projections,
        confidence=confidence,
        warnings=warnings,
        patterns_detected=list(patterns),
        montecarlo_results=all_mc_results,
        warm_discount=warm_discount,
    )
