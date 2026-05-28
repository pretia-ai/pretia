"""Tests for Monte Carlo cost simulation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.projection.montecarlo import _sample_step_cost, simulate
from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.stats import compute_stats


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o-mini",
    input_tokens: int = 100,
    output_tokens: int = 50,
    iteration: int = 1,
    context_size: int | None = None,
) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        step_type="llm",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=context_size if context_size is not None else input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash="abc123",
        system_prompt_tokens=50,
        output_format="text",
        is_retry=False,
        iteration=iteration,
        parent_step=None,
        duration_ms=100,
        timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    )


def _stable_runs(n: int = 20) -> list[list[StepRecord]]:
    """Create n runs where every run costs approximately the same."""
    import random

    rng = random.Random(123)
    runs = []
    for _ in range(n):
        base = 100 + rng.randint(-5, 5)
        runs.append([_make_record("classify", "gpt-4o-mini", base, 50)])
    return runs


def _high_variance_runs(n: int = 20) -> list[list[StepRecord]]:
    """Create n runs: 85% normal, 15% expensive outliers."""
    import random

    rng = random.Random(456)
    runs = []
    for _i in range(n):
        if rng.random() < 0.15:
            runs.append([_make_record("classify", "gpt-4o-mini", 1000, 500)])
        else:
            runs.append([_make_record("classify", "gpt-4o-mini", 100, 50)])
    return runs


def _context_growth_runs(n: int = 20) -> list[list[StepRecord]]:
    """Create runs with a 'review' step that has clear context growth."""
    import random

    rng = random.Random(789)
    runs = []
    for _ in range(n):
        n_iters = rng.choice([3, 4, 5, 6, 7])
        run_records = []
        for k in range(1, n_iters + 1):
            ctx = 1000 + 800 * k
            run_records.append(
                _make_record(
                    "review",
                    "gpt-4o",
                    ctx,
                    200,
                    iteration=k,
                    context_size=ctx,
                ),
            )
        runs.append(run_records)
    return runs


class TestSimulateStableData:
    def test_mean_within_tolerance(self):
        runs = _stable_runs(20)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=1000,
            runs=runs,
            n_simulations=1000,
        )
        expected = stats.cost_per_run.mean * 1000 * 30
        assert result.monthly_projection.mean == pytest.approx(expected, rel=0.05)
        assert result.convergence_check is True


class TestSimulateHighVariance:
    def test_p95_exceeds_mean(self):
        runs = _high_variance_runs(20)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=1000,
            runs=runs,
            n_simulations=1000,
        )
        assert result.monthly_projection.p95 > result.monthly_projection.mean * 1.2


class TestSimulateContextGrowth:
    def test_captures_growth_tail(self):
        runs = _context_growth_runs(20)
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="context_growth",
            step_name="review",
            severity="danger",
            evidence={"r_squared": 0.95, "slope": 800},
            description="Context grows in 'review'",
        )
        mc = simulate(
            stats,
            [pattern],
            daily_volume=1000,
            runs=runs,
            n_simulations=1000,
        )
        linear_mean = stats.cost_per_run.mean * 1000 * 30
        assert mc.monthly_projection.p95 > linear_mean


class TestSimulateLinearVsLogGrowth:
    def test_growth_models_differ(self):
        runs = _context_growth_runs(20)
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="context_growth",
            step_name="review",
            severity="danger",
            evidence={"r_squared": 0.95, "slope": 800},
            description="Context grows in 'review'",
        )
        mc = simulate(
            stats,
            [pattern],
            daily_volume=1000,
            runs=runs,
            n_simulations=1000,
        )
        assert mc.growth_model_delta > 0
        assert mc.linear_monthly.p95 != mc.log_monthly.p95


class TestSimulateReproducible:
    def test_same_seed_same_result(self):
        runs = _stable_runs(20)
        stats = compute_stats(runs)
        r1 = simulate(stats, [], daily_volume=100, runs=runs, seed=42)
        r2 = simulate(stats, [], daily_volume=100, runs=runs, seed=42)
        assert r1.monthly_projection.mean == r2.monthly_projection.mean
        assert r1.monthly_projection.p95 == r2.monthly_projection.p95


class TestSimulateDifferentSeeds:
    def test_different_seeds_differ(self):
        runs = _high_variance_runs(20)
        stats = compute_stats(runs)
        r1 = simulate(stats, [], daily_volume=100, runs=runs, seed=42)
        r2 = simulate(stats, [], daily_volume=100, runs=runs, seed=99)
        assert r1.monthly_projection.mean != r2.monthly_projection.mean


class TestSimulateConvergenceCheck:
    def test_converges_on_stable_data(self):
        runs = _stable_runs(20)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=1000,
            runs=runs,
            n_simulations=10000,
        )
        assert result.convergence_check is True


class TestSimulateSmallN:
    def test_no_crash_with_10_sims(self):
        runs = _stable_runs(5)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=100,
            runs=runs,
            n_simulations=10,
        )
        assert result.n_simulations == 10
        assert result.monthly_projection.mean > 0


class TestSampleStepCostNoPattern:
    def test_returns_observed_cost(self):
        import random

        rng = random.Random(42)
        step_run_costs = {"classify": [0.01, 0.02, 0.015]}
        result, _, _ = _sample_step_cost(
            "classify",
            rng,
            step_run_costs=step_run_costs,
            step_iterations={},
            step_occurrence_costs={},
            step_growth={},
            growth_steps=set(),
            loop_variance_steps=set(),
        )
        assert result in step_run_costs["classify"]


class TestSampleStepCostWithContextGrowth:
    def test_accounts_for_growth(self):
        import random

        rng = random.Random(42)
        step_growth = {
            "review": {
                "slope": 800.0,
                "base_context": 1000.0,
                "model": "gpt-4o",
                "mean_output_tokens": 200.0,
            },
        }
        avg, linear, log = _sample_step_cost(
            "review",
            rng,
            step_run_costs={"review": [0.05]},
            step_iterations={"review": [3, 4, 5]},
            step_occurrence_costs={"review": [0.01]},
            step_growth=step_growth,
            growth_steps={"review"},
            loop_variance_steps=set(),
        )
        assert avg > 0
        assert linear > 0
        assert log > 0
