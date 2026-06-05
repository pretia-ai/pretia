"""Tests for the conformal confidence tier system."""

from __future__ import annotations

import json
import math

from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.stats import PercentileStats, StepStats
from agentcost.validation.confidence import (
    compute_confidence,
    compute_conformal_interval,
    compute_effective_sample_size,
)


def _make_step_stats(
    name: str = "step_a",
    cost_mean: float = 0.01,
    cost_std: float = 0.002,
) -> StepStats:
    ps = PercentileStats(
        min=cost_mean * 0.5,
        max=cost_mean * 2.0,
        mean=cost_mean,
        std=cost_std,
        p50=cost_mean,
        p75=cost_mean * 1.3,
        p90=cost_mean * 1.6,
        p95=cost_mean * 1.8,
        p99=cost_mean * 2.0,
    )
    tok_ps = PercentileStats(
        min=100,
        max=500,
        mean=300,
        std=80,
        p50=280,
        p75=350,
        p90=400,
        p95=450,
        p99=490,
    )
    iter_ps = PercentileStats(
        min=1,
        max=1,
        mean=1,
        std=0,
        p50=1,
        p75=1,
        p90=1,
        p95=1,
        p99=1,
    )
    return StepStats(
        step_name=name,
        step_type="llm",
        model="gpt-4o-mini",
        call_count=20,
        runs_present=20,
        input_tokens=tok_ps,
        output_tokens=tok_ps,
        total_tokens=tok_ps,
        cost=ps,
        duration_ms=tok_ps,
        context_size=tok_ps,
        iterations_per_run=iter_ps,
        mean_iterations=1.0,
    )


def _make_pattern(
    pattern_type: str = "context_growth",
    step_name: str = "review",
    severity: str = "danger",
) -> DetectedPattern:
    return DetectedPattern(
        pattern_type=pattern_type,
        step_name=step_name,
        severity=severity,
        evidence={},
        description=f"Test: {pattern_type} on {step_name}",
    )


# ---------------------------------------------------------------------------
# Conformal interval
# ---------------------------------------------------------------------------


class TestConformalIntervalCoverage:
    def test_coverage_over_synthetic_workflows(self):
        import random

        rng = random.Random(42)
        true_mean = 0.05
        sigma = 0.5
        n = 50
        covered = 0
        trials = 100
        for _ in range(trials):
            costs = [math.exp(math.log(true_mean) + sigma * rng.gauss(0, 1)) for _ in range(n)]
            mu, lo, hi = compute_conformal_interval(costs, alpha=0.10)
            if lo <= true_mean <= hi:
                covered += 1
        assert covered / trials >= 0.75


class TestConformalIntervalWidthDecreases:
    def test_width_decreases_with_n(self):
        import random

        avg_widths = []
        for n in [10, 50, 200]:
            trial_widths = []
            for seed in range(20):
                rng = random.Random(seed + n * 1000)
                costs = [0.05 + rng.gauss(0, 0.02) for _ in range(n)]
                _, lo, hi = compute_conformal_interval(costs, alpha=0.10)
                trial_widths.append(hi - lo)
            avg_widths.append(sum(trial_widths) / len(trial_widths))
        assert avg_widths[0] > avg_widths[1] > avg_widths[2]


class TestConformalMonthlyPropagation:
    def test_sqrt_n_scaling(self):
        import random

        rng = random.Random(42)
        costs = [0.03 + rng.gauss(0, 0.005) for _ in range(50)]
        mu, lo, hi = compute_conformal_interval(costs, alpha=0.10)
        per_run_se = hi - mu

        n_monthly = 30000
        monthly_width = 2 * 1.645 * math.sqrt(n_monthly) * per_run_se
        naive_width = 2 * n_monthly * per_run_se

        # sqrt(N) scaling produces much smaller width than N scaling
        assert monthly_width < naive_width * 0.1


# ---------------------------------------------------------------------------
# Confidence tier from interval width
# ---------------------------------------------------------------------------


class TestConfidenceTierFromWidth:
    def test_tiers(self):
        from agentcost.validation.confidence import _tier_from_relative_width

        assert _tier_from_relative_width(1.5) == "HIGH"
        assert _tier_from_relative_width(3.0) == "MODERATE"
        assert _tier_from_relative_width(7.0) == "LOW"
        assert _tier_from_relative_width(15.0) == "VERY_LOW"


# ---------------------------------------------------------------------------
# compute_confidence integration
# ---------------------------------------------------------------------------


class TestConfidenceHigh:
    def test_high_with_tight_data(self):
        steps = {"step_a": _make_step_stats()}
        costs = [0.03 + i * 0.0001 for i in range(200)]
        result = compute_confidence(200, steps, [], run_costs=costs)
        assert result.tier in ("HIGH", "MODERATE")
        assert result.monthly_lower > 0
        assert result.monthly_upper > result.monthly_lower


class TestConfidenceWithPatterns:
    def test_patterns_listed(self):
        steps = {"step_a": _make_step_stats()}
        patterns = [_make_pattern("context_growth")]
        costs = [0.03 + i * 0.001 for i in range(50)]
        result = compute_confidence(50, steps, patterns, run_costs=costs)
        assert "context_growth" in result.patterns_detected


class TestConfidenceToDict:
    def test_serializes(self):
        steps = {"step_a": _make_step_stats()}
        costs = [0.03] * 50
        result = compute_confidence(50, steps, [], run_costs=costs)
        d = result.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["tier"] in ("HIGH", "MODERATE", "LOW", "VERY_LOW")
        assert "monthly_lower" in parsed
        assert "monthly_upper" in parsed


class TestConfidenceNoRunCosts:
    def test_works_without_run_costs(self):
        steps = {"step_a": _make_step_stats()}
        result = compute_confidence(50, steps, [])
        assert result.tier in ("HIGH", "MODERATE", "LOW", "VERY_LOW")


# ---------------------------------------------------------------------------
# effective sample size (diagnostic, not tier-driving)
# ---------------------------------------------------------------------------


class TestEffectiveSampleSize:
    def test_uniform_high_neff(self):
        costs = [0.01 * (i + 1) for i in range(100)]
        n_eff = compute_effective_sample_size(costs)
        assert n_eff > 50

    def test_clustered_low_neff(self):
        costs = [0.03] * 45 + [0.10] * 5
        n_eff = compute_effective_sample_size(costs)
        assert n_eff < 50


class TestTailInflationRemoved:
    def test_no_tail_inflation_in_mc(self):
        from datetime import UTC, datetime

        from agentcost.collectors.base import StepRecord
        from agentcost.projection.montecarlo import simulate
        from agentcost.projection.stats import compute_stats

        runs = [
            [
                StepRecord(
                    step_name="s",
                    step_type="llm",
                    model="gpt-4o-mini",
                    input_tokens=100,
                    output_tokens=50,
                    context_size=100,
                    tool_definitions_tokens=0,
                    system_prompt_hash="a",
                    system_prompt_tokens=50,
                    output_format="text",
                    is_retry=False,
                    iteration=1,
                    parent_step=None,
                    duration_ms=100,
                    timestamp=datetime(2026, 1, 1, tzinfo=UTC),
                )
            ]
            for _ in range(20)
        ]
        stats = compute_stats(runs)
        result = simulate(stats, [], daily_volume=100, runs=runs, n_simulations=100)
        assert result.tail_inflation_factor is None
