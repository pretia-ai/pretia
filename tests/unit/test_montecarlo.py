"""Tests for Monte Carlo cost simulation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pretia.collectors.base import StepRecord
from pretia.projection.montecarlo import _safe_cost, _sample_step_cost, simulate
from pretia.projection.patterns import DetectedPattern
from pretia.projection.stats import compute_stats


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
        assert result.monthly_projection.p95 > result.monthly_projection.mean


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
            daily_volume=10,
            runs=runs,
            n_simulations=500,
        )
        linear_mean = stats.cost_per_run.mean * 10 * 30
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
            daily_volume=10,
            runs=runs,
            n_simulations=500,
        )
        assert mc.growth_model_delta > 0
        assert mc.linear_monthly.p95 != mc.log_monthly.p95


class TestSimulateReproducible:
    def test_same_seed_same_result(self):
        runs = _stable_runs(20)
        stats = compute_stats(runs)
        r1 = simulate(stats, [], daily_volume=10, runs=runs, seed=42, n_simulations=1000)
        r2 = simulate(stats, [], daily_volume=10, runs=runs, seed=42, n_simulations=1000)
        assert r1.monthly_projection.mean == r2.monthly_projection.mean
        assert r1.monthly_projection.p95 == r2.monthly_projection.p95


class TestSimulateDifferentSeeds:
    def test_different_seeds_differ(self):
        runs = _high_variance_runs(20)
        stats = compute_stats(runs)
        r1 = simulate(stats, [], daily_volume=10, runs=runs, seed=42, n_simulations=1000)
        r2 = simulate(stats, [], daily_volume=10, runs=runs, seed=99, n_simulations=1000)
        assert r1.monthly_projection.mean != r2.monthly_projection.mean


class TestSimulateConvergenceCheck:
    def test_converges_on_stable_data(self):
        runs = _stable_runs(20)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=10,
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
        result, _, _, capped = _sample_step_cost(
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
        assert capped is False


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
        avg, linear, log, _capped = _sample_step_cost(
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


# ---------------------------------------------------------------------------
# Fix 1 (CLT) tests
# ---------------------------------------------------------------------------


def _uniform_cost_runs(n: int = 50) -> list[list[StepRecord]]:
    """Create n runs with costs uniformly distributed between $5 and $15.

    Uses gpt-4o with output_tokens=0 so cost = input_tokens * 2.5e-6.
    """
    runs = []
    for i in range(n):
        cost_target = 5.0 + i * (10.0 / (n - 1))
        input_tokens = int(cost_target / 2.5e-6)
        runs.append([_make_record("compute", "gpt-4o", input_tokens, 0)])
    return runs


class TestCltVarianceScaling:
    def test_clt_variance_scaling(self):
        runs = _uniform_cost_runs(50)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=10000,
            runs=runs,
            n_simulations=3000,
        )
        n_monthly = 10000 * 30
        expected_mean = n_monthly * stats.cost_per_run.mean
        assert result.monthly_projection.p50 == pytest.approx(expected_mean, rel=0.05)
        gap = result.monthly_projection.p95 - result.monthly_projection.p50
        # CLT reduces the gap from ~1.4M (N² variance bug) to ~46K (N²/K + N variance).
        assert gap < 100000


class TestCltLowVolume:
    def test_clt_low_volume(self):
        runs = _uniform_cost_runs(50)
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=100,
            runs=runs,
            n_simulations=3000,
            seed=7,
        )
        n_monthly = 100 * 30
        expected_mean = n_monthly * stats.cost_per_run.mean
        assert result.monthly_projection.p50 == pytest.approx(expected_mean, rel=0.10)

        run_costs = [_safe_cost("gpt-4o", r[0].input_tokens, r[0].output_tokens) for r in runs]
        import math

        std_run = math.sqrt(sum((x - stats.cost_per_run.mean) ** 2 for x in run_costs) / 49)
        max_gap = 5 * std_run * math.sqrt(n_monthly)
        gap = result.monthly_projection.p95 - result.monthly_projection.p50
        assert gap < max_gap


class TestCltSingleValue:
    def test_clt_single_value(self):
        runs = [[_make_record("compute", "gpt-4o", 4000000, 0)] for _ in range(20)]
        stats = compute_stats(runs)
        result = simulate(
            stats,
            [],
            daily_volume=10000,
            runs=runs,
            n_simulations=1000,
            seed=1,
        )
        n_monthly = 10000 * 30
        expected = n_monthly * 10.0
        assert result.monthly_projection.p50 == pytest.approx(expected, rel=0.001)


# ---------------------------------------------------------------------------
# Tail inflation removed (conformal intervals replace it)
# ---------------------------------------------------------------------------


class TestTailInflationRemoved:
    def test_no_inflation_at_small_n(self):
        runs = _stable_runs(20)
        stats = compute_stats(runs)
        result = simulate(stats, [], daily_volume=10, runs=runs, n_simulations=1000, seed=42)
        assert result.tail_inflation_factor is None

    def test_no_inflation_at_large_n(self):
        runs = _stable_runs(50)
        stats = compute_stats(runs)
        result = simulate(stats, [], daily_volume=10, runs=runs, n_simulations=1000, seed=42)
        assert result.tail_inflation_factor is None


class TestCvarInMcResult:
    def test_cvar_populated(self):
        runs = _stable_runs(50)
        stats = compute_stats(runs)
        result = simulate(stats, [], daily_volume=100, runs=runs, n_simulations=1000)
        assert result.cvar_95 > 0
        assert result.cvar_95 >= result.monthly_projection.p95


# ---------------------------------------------------------------------------
# Fix 3 (independence / whole-run sampling) tests
# ---------------------------------------------------------------------------


class TestWholeRunSampling:
    def test_whole_run_sampling(self):
        runs: list[list[StepRecord]] = []
        for _ in range(5):
            runs.append(
                [
                    _make_record("step_a", "gpt-4o-mini", 100, 50),
                    _make_record("step_b", "gpt-4o-mini", 100, 50),
                    _make_record("step_c", "gpt-4o-mini", 100, 50),
                ]
            )
        for _ in range(5):
            runs.append(
                [
                    _make_record("step_a", "gpt-4o-mini", 1000, 500),
                    _make_record("step_b", "gpt-4o-mini", 1000, 500),
                    _make_record("step_c", "gpt-4o-mini", 1000, 500),
                ]
            )

        stats = compute_stats(runs)
        debug_costs: list[float] = []
        simulate(
            stats,
            [],
            daily_volume=100,
            runs=runs,
            n_simulations=1000,
            _debug_run_costs=debug_costs,
        )

        cheap_total = 3 * _safe_cost("gpt-4o-mini", 100, 50)
        expensive_total = 3 * _safe_cost("gpt-4o-mini", 1000, 500)

        for cost in debug_costs:
            is_cheap = abs(cost - cheap_total) / cheap_total < 0.01
            is_expensive = abs(cost - expensive_total) / expensive_total < 0.01
            assert is_cheap or is_expensive, (
                f"Got intermediate cost {cost}, expected {cheap_total} or {expensive_total}"
            )


# ---------------------------------------------------------------------------
# Fix 4 (extrapolation cap) tests
# ---------------------------------------------------------------------------


class TestContextGrowthCapped:
    def test_context_growth_capped(self):
        import random

        rng = random.Random(42)
        _, _, _, capped = _sample_step_cost(
            "grow",
            rng,
            step_run_costs={"grow": [0.1]},
            step_iterations={"grow": [5, 5, 10, 10, 10]},
            step_occurrence_costs={"grow": [0.02]},
            step_growth={
                "grow": {
                    "slope": 1000.0,
                    "base_context": 1000.0,
                    "model": "gpt-4o",
                    "mean_output_tokens": 200.0,
                },
            },
            growth_steps={"grow"},
            loop_variance_steps=set(),
            max_observed_iters={"grow": 5},
        )

        # With iterations [5,5,10,10,10], some samples will pick 10 > max_obs=5
        # Run enough samples to trigger capping
        cap_count = 0
        for _ in range(100):
            _, _, _, c = _sample_step_cost(
                "grow",
                rng,
                step_run_costs={"grow": [0.1]},
                step_iterations={"grow": [5, 5, 10, 10, 10]},
                step_occurrence_costs={"grow": [0.02]},
                step_growth={
                    "grow": {
                        "slope": 1000.0,
                        "base_context": 1000.0,
                        "model": "gpt-4o",
                        "mean_output_tokens": 200.0,
                    },
                },
                growth_steps={"grow"},
                loop_variance_steps=set(),
                max_observed_iters={"grow": 5},
            )
            if c:
                cap_count += 1
        assert cap_count > 0

    def test_extrapolation_cap_warnings_in_result(self):
        runs: list[list[StepRecord]] = []
        import random as random_mod

        rng = random_mod.Random(42)
        for _ in range(20):
            n_iters = rng.choice([3, 4, 5, 8, 10])
            run_records = []
            for k in range(1, n_iters + 1):
                ctx = 1000 + 1000 * k
                run_records.append(
                    _make_record("grow", "gpt-4o", ctx, 200, iteration=k, context_size=ctx)
                )
            runs.append(run_records)

        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="context_growth",
            step_name="grow",
            severity="danger",
            evidence={"r_squared": 0.99, "slope": 1000},
            description="Context grows in 'grow'",
        )
        mc = simulate(
            stats,
            [pattern],
            daily_volume=10,
            runs=runs,
            n_simulations=200,
        )
        assert mc.extrapolation_cap_warnings >= 0


# ---------------------------------------------------------------------------
# Fix 5 (seed stability) test
# ---------------------------------------------------------------------------


class TestSeedStabilityHeavyTail:
    def test_seed_stability_heavy_tail(self):
        import math

        heavy_tail_costs = [
            0.37,
            0.42,
            0.56,
            0.61,
            0.74,
            0.82,
            0.91,
            1.03,
            1.12,
            1.25,
            1.41,
            1.58,
            1.82,
            2.14,
            2.53,
            3.01,
            3.72,
            4.81,
            6.92,
            12.18,
        ]

        runs: list[list[StepRecord]] = []
        for cost_val in heavy_tail_costs:
            input_tokens = int(cost_val / 2.5e-6)
            runs.append([_make_record("heavy", "gpt-4o", input_tokens, 0)])

        stats = compute_stats(runs)

        p50_values = []
        p95_values = []
        for seed in [1, 2, 3, 4, 5]:
            result = simulate(
                stats,
                [],
                daily_volume=5,
                runs=runs,
                n_simulations=10000,
                seed=seed,
            )
            p50_values.append(result.monthly_projection.p50)
            p95_values.append(result.monthly_projection.p95)

        def cv(vals: list[float]) -> float:
            mean = sum(vals) / len(vals)
            if mean == 0:
                return 0.0
            var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
            return math.sqrt(var) / mean

        cv_p50 = cv(p50_values)
        cv_p95 = cv(p95_values)

        if cv_p50 >= 0.05 or cv_p95 >= 0.05:
            print(
                f"DIAGNOSTIC: Seed stability check failed. "
                f"CV(p50)={cv_p50:.4f}, CV(p95)={cv_p95:.4f}. "
                f"Consider increasing from 10,000 to 50,000 simulations."
            )
        assert cv_p50 < 0.05, f"p50 CV={cv_p50:.4f} exceeds 5%"
        assert cv_p95 < 0.05, f"p95 CV={cv_p95:.4f} exceeds 5%"


# ---------------------------------------------------------------------------
# Change 2: Power-law Monte Carlo model tests
# ---------------------------------------------------------------------------


def _sublinear_growth_runs(n: int = 20) -> list[list[StepRecord]]:
    """Create runs with sub-linear context growth (sqrt-like)."""
    import random as random_mod

    rng = random_mod.Random(111)
    runs: list[list[StepRecord]] = []
    for _ in range(n):
        n_iters = rng.choice([5, 6, 7, 8])
        run_records = []
        for k in range(1, n_iters + 1):
            ctx = int(1000 * (k**0.5))
            run_records.append(
                _make_record("slow_grow", "gpt-4o", ctx, 200, iteration=k, context_size=ctx)
            )
        runs.append(run_records)
    return runs


def _superlinear_growth_runs(n: int = 20) -> list[list[StepRecord]]:
    """Create runs with super-linear context growth (quadratic)."""
    import random as random_mod

    rng = random_mod.Random(222)
    runs: list[list[StepRecord]] = []
    for _ in range(n):
        n_iters = rng.choice([4, 5, 6])
        run_records = []
        for k in range(1, n_iters + 1):
            ctx = int(500 * (k**2))
            run_records.append(
                _make_record("fast_grow", "gpt-4o", ctx, 200, iteration=k, context_size=ctx)
            )
        runs.append(run_records)
    return runs


class TestPowerLawModelSublinear:
    def test_power_law_model_sublinear(self):
        runs = _sublinear_growth_runs(20)
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="context_growth",
            step_name="slow_grow",
            severity="danger",
            evidence={"r_squared": 0.9, "slope": 300},
            description="Sub-linear growth in 'slow_grow'",
            growth_type="nonlinear",
            power_law_alpha=0.5,
            power_law_base=1000.0,
            growth_classification="sub_linear",
        )
        mc = simulate(stats, [pattern], daily_volume=10, runs=runs, n_simulations=200)
        assert mc.monthly_projection.mean > 0
        assert mc.per_run_projection.mean > 0


class TestPowerLawModelSuperlinear:
    def test_power_law_model_superlinear(self):
        runs = _superlinear_growth_runs(20)
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="context_growth",
            step_name="fast_grow",
            severity="danger",
            evidence={"r_squared": 0.85, "slope": 2000},
            description="Super-linear growth in 'fast_grow'",
            growth_type="nonlinear",
            power_law_alpha=2.0,
            power_law_base=500.0,
            growth_classification="super_linear",
        )
        mc = simulate(stats, [pattern], daily_volume=10, runs=runs, n_simulations=200)
        assert mc.monthly_projection.mean > 0
        assert mc.per_run_projection.mean > 0


class TestLinearGrowthModelUnchanged:
    def test_linear_growth_model_unchanged(self):
        runs = _context_growth_runs(20)
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="context_growth",
            step_name="review",
            severity="danger",
            evidence={"r_squared": 0.95, "slope": 800},
            description="Linear growth in 'review'",
            growth_type="linear",
        )
        mc_linear = simulate(
            stats,
            [pattern],
            daily_volume=10,
            runs=runs,
            n_simulations=200,
            seed=42,
        )

        pattern_none = DetectedPattern(
            pattern_type="context_growth",
            step_name="review",
            severity="danger",
            evidence={"r_squared": 0.95, "slope": 800},
            description="Growth in 'review'",
        )
        mc_none = simulate(
            stats,
            [pattern_none],
            daily_volume=10,
            runs=runs,
            n_simulations=200,
            seed=42,
        )

        assert mc_linear.monthly_projection.mean == pytest.approx(
            mc_none.monthly_projection.mean, rel=0.01
        )


# ---------------------------------------------------------------------------
# Mapping verification: new detectors use whole-run resampling, not custom cost models
# ---------------------------------------------------------------------------


class TestStepCountVarianceNoMcCostModel:
    def test_step_count_no_custom_model(self):
        runs: list[list[StepRecord]] = []
        for i in range(20):
            if i < 10:
                runs.append(
                    [
                        _make_record("step_a", "gpt-4o-mini", 200, 100),
                        _make_record("step_b", "gpt-4o-mini", 200, 100),
                        _make_record("step_c", "gpt-4o-mini", 200, 100),
                    ]
                )
            else:
                runs.append(
                    [
                        _make_record("step_a", "gpt-4o-mini", 200, 100),
                        _make_record("step_b", "gpt-4o-mini", 0, 0),
                        _make_record("step_c", "gpt-4o-mini", 0, 0),
                    ]
                )
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="step_count_variance",
            step_name="_workflow_",
            severity="danger",
            evidence={"cv": 0.65},
            description="Step count varies",
            step_count_cv=0.65,
            step_count_min=1,
            step_count_max=3,
            step_count_mean=2.0,
        )
        debug_costs: list[float] = []
        simulate(
            stats,
            [pattern],
            daily_volume=100,
            runs=runs,
            n_simulations=500,
            _debug_run_costs=debug_costs,
        )
        cheap = 1 * _safe_cost("gpt-4o-mini", 200, 100)
        expensive = 3 * _safe_cost("gpt-4o-mini", 200, 100)
        for cost in debug_costs:
            is_cheap = abs(cost - cheap) / cheap < 0.01
            is_expensive = abs(cost - expensive) / expensive < 0.01
            assert is_cheap or is_expensive, (
                f"Got cost {cost}, expected {cheap} or {expensive} (whole-run sampling)"
            )


class TestBimodalityNoMcCostModel:
    def test_bimodality_no_custom_model(self):
        runs: list[list[StepRecord]] = []
        for _ in range(15):
            runs.append([_make_record("work", "gpt-4o-mini", 100, 50)])
        for _ in range(5):
            runs.append([_make_record("work", "gpt-4o-mini", 5000, 2000)])
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="bimodality",
            step_name="_workflow_",
            severity="warning",
            evidence={"bic_delta": 20},
            description="Bimodal costs",
            bimodal_bic_delta=20.0,
        )
        mc = simulate(
            stats,
            [pattern],
            daily_volume=100,
            runs=runs,
            n_simulations=500,
        )
        assert mc.monthly_projection.mean > 0
        assert mc.per_run_projection.mean > 0


class TestGmmMcSamplingBimodal:
    def test_bimodal_gmm_produces_both_modes(self):
        import math
        import random as _rng

        r = _rng.Random(42)
        runs = []
        for _ in range(20):
            cost = 100 + r.randint(-10, 10)
            runs.append([_make_record("work", "gpt-4o-mini", cost, 50)])
        for _ in range(10):
            cost = 2000 + r.randint(-200, 200)
            runs.append([_make_record("work", "gpt-4o-mini", cost, 1000)])
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="bimodality",
            step_name="_workflow_",
            severity="warning",
            evidence={"n_runs": 30},
            description="Bimodal costs",
            bimodal_bic_delta=20.0,
            gmm_means=[math.log(0.02), math.log(0.40)],
            gmm_stds=[0.15, 0.20],
            gmm_weights=[0.67, 0.33],
        )
        per_run_costs: list[float] = []
        simulate(
            stats,
            [pattern],
            daily_volume=100,
            runs=runs,
            n_simulations=1000,
            _debug_run_costs=per_run_costs,
        )
        cheap = sum(1 for c in per_run_costs if c < 0.10)
        expensive = sum(1 for c in per_run_costs if c > 0.20)
        total = len(per_run_costs)
        assert cheap / total > 0.10
        assert expensive / total > 0.10


class TestGmmMcFallsBackWithoutParams:
    def test_no_gmm_params_uses_whole_run_resampling(self):
        import random as _rng

        r = _rng.Random(42)
        runs = []
        for _ in range(20):
            cost = 100 + r.randint(-10, 10)
            runs.append([_make_record("work", "gpt-4o-mini", cost, 50)])
        stats = compute_stats(runs)
        pattern = DetectedPattern(
            pattern_type="bimodality",
            step_name="_workflow_",
            severity="warning",
            evidence={"n_runs": 20},
            description="Bimodal costs — no GMM params",
            bimodal_bic_delta=20.0,
        )
        mc = simulate(
            stats,
            [pattern],
            daily_volume=100,
            runs=runs,
            n_simulations=100,
        )
        assert mc.monthly_projection.mean > 0
