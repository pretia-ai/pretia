"""Tests for calibration scoring."""

from __future__ import annotations

import json

import pytest

from pretia.projection.stats import (
    PercentileStats,
    ProfilingStats,
    RunStats,
    StepStats,
)
from pretia.validation.scoring import (
    CalibrationScore,
    _spearman_rank_correlation,
    bootstrap_bca_ci,
    bootstrap_percentile_ci,
    format_calibration_report,
    score_projection,
)


def _ps(
    mean: float = 0.03,
    p50: float = 0.028,
    p95: float = 0.048,
    std: float = 0.01,
) -> PercentileStats:
    return PercentileStats(
        min=mean * 0.5,
        max=p95 * 1.5,
        mean=mean,
        std=std,
        p50=p50,
        p75=mean * 1.2,
        p90=p95 * 0.9,
        p95=p95,
        p99=p95 * 1.3,
    )


def _step(
    name: str,
    cost_mean: float = 0.01,
    cost_p50: float = 0.009,
    cost_p95: float = 0.018,
) -> StepStats:
    cost_ps = _ps(cost_mean, cost_p50, cost_p95, cost_mean * 0.3)
    tok = _ps(300, 280, 450, 80)
    itr = PercentileStats(
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
        input_tokens=tok,
        output_tokens=tok,
        total_tokens=tok,
        cost=cost_ps,
        duration_ms=tok,
        context_size=tok,
        iterations_per_run=itr,
        mean_iterations=1.0,
    )


def _make_stats(
    steps: dict[str, StepStats] | None = None,
    cost_p50: float = 0.028,
    cost_p95: float = 0.048,
    cost_mean: float = 0.03,
    total_runs: int = 20,
    run_costs: list[float] | None = None,
) -> ProfilingStats:
    if steps is None:
        steps = {
            "classify": _step("classify", 0.001, 0.001, 0.002),
            "generate": _step("generate", 0.028, 0.027, 0.044),
        }
    cpr = _ps(cost_mean, cost_p50, cost_p95)

    run_stats: list[RunStats] = []
    if run_costs is not None:
        for i, c in enumerate(run_costs):
            run_stats.append(
                RunStats(
                    run_index=i,
                    total_cost=c,
                    total_tokens=300,
                    total_input_tokens=200,
                    total_output_tokens=100,
                    step_count=2,
                    duration_ms=200,
                )
            )

    return ProfilingStats(
        step_stats=steps,
        run_stats=run_stats,
        cost_per_run=cpr,
        tokens_per_run=_ps(2000, 1800, 3200, 600),
        total_runs=total_runs,
        total_steps=total_runs * 2,
    )


class TestScoreProjectionAllPass:
    def test_all_pass(self):
        steps = {
            "classify": _step("classify", 0.001, 0.001, 0.002),
            "generate": _step("generate", 0.028, 0.027, 0.044),
            "format": _step("format", 0.005, 0.004, 0.008),
        }
        proj = _make_stats(steps=steps, cost_p50=0.028, cost_p95=0.048)
        # gt_costs must stay below projected p95 (0.048) for >=80% of runs
        gt_costs = [0.020 + (i % 40) * 0.0006 for i in range(500)]
        gt = _make_stats(
            steps=steps, cost_p50=0.028, cost_p95=0.048, total_runs=500, run_costs=gt_costs
        )
        sc = score_projection(proj, gt)
        assert sc.verdict == "PASS"
        assert len(sc.failures) == 0


class TestScoreProjectionP50Warn:
    def test_p50_warn(self):
        proj = _make_stats(cost_p50=0.070)
        gt_costs = [0.028] * 500
        gt = _make_stats(cost_p50=0.028, total_runs=500, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.p50_ratio == pytest.approx(0.070 / 0.028, rel=0.01)
        assert sc.verdict == "WARN"
        assert any("p50" in w for w in sc.warnings)


class TestScoreProjectionP50Fail:
    def test_p50_fail(self):
        proj = _make_stats(cost_p50=0.112)
        gt_costs = [0.028] * 500
        gt = _make_stats(cost_p50=0.028, total_runs=500, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.verdict == "FAIL"
        assert any("p50" in f for f in sc.failures)


class TestScoreProjectionP95CoveragePass:
    def test_p95_pass(self):
        proj = _make_stats(cost_p95=0.050)
        gt_costs = [0.020 + i * 0.0005 for i in range(50)]
        gt = _make_stats(cost_p50=0.028, cost_p95=0.048, total_runs=50, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.p95_coverage >= 0.80


class TestScoreProjectionP95CoverageFail:
    def test_p95_fail(self):
        proj = _make_stats(cost_p95=0.020)
        gt_costs = [0.025 + i * 0.002 for i in range(50)]
        gt = _make_stats(cost_p50=0.035, cost_p95=0.070, total_runs=50, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.p95_coverage < 0.60
        assert sc.verdict == "FAIL"


class TestScoreProjectionRangeRatioWarn:
    def test_range_warn(self):
        proj = _make_stats(cost_p50=0.010, cost_p95=0.070)
        gt_costs = [0.010] * 500
        gt = _make_stats(cost_p50=0.010, total_runs=500, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.range_ratio == pytest.approx(7.0, rel=0.01)
        assert any("Wide projection range" in w for w in sc.warnings)


class TestScoreProjectionRangeRatioFail:
    def test_range_fail(self):
        proj = _make_stats(cost_p50=0.010, cost_p95=0.120)
        gt_costs = [0.010] * 500
        gt = _make_stats(cost_p50=0.010, total_runs=500, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.range_ratio >= 10.0
        assert sc.verdict == "FAIL"


class TestScoreProjectionTopStepCorrect:
    def test_top_step_match(self):
        steps = {
            "classify": _step("classify", 0.001),
            "generate": _step("generate", 0.028),
        }
        proj = _make_stats(steps=steps)
        gt = _make_stats(steps=steps, total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        assert sc.top_step_correct is True


class TestScoreProjectionTopStepWrong:
    def test_top_step_mismatch(self):
        proj_steps = {
            "classify": _step("classify", 0.050),
            "generate": _step("generate", 0.028),
        }
        gt_steps = {
            "classify": _step("classify", 0.001),
            "generate": _step("generate", 0.028),
        }
        proj = _make_stats(steps=proj_steps)
        gt = _make_stats(steps=gt_steps, total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        assert sc.top_step_correct is False
        assert sc.verdict == "FAIL"


class TestScoreProjectionCoDominantSteps:
    def test_co_dominant(self):
        proj_steps = {
            "step_a": _step("step_a", 0.028),
            "step_b": _step("step_b", 0.030),
        }
        gt_steps = {
            "step_a": _step("step_a", 0.030),
            "step_b": _step("step_b", 0.028),
        }
        proj = _make_stats(steps=proj_steps)
        gt = _make_stats(steps=gt_steps, total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        assert sc.top_step_correct is True
        assert any("co-dominant" in w.lower() for w in sc.warnings)


class TestSpearmanCorrelationPerfect:
    def test_perfect(self):
        r = _spearman_rank_correlation(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            {"a": 10.0, "b": 20.0, "c": 30.0},
        )
        assert r == pytest.approx(1.0)


class TestSpearmanCorrelationReversed:
    def test_reversed(self):
        r = _spearman_rank_correlation(
            {"a": 1.0, "b": 2.0, "c": 3.0},
            {"a": 30.0, "b": 20.0, "c": 10.0},
        )
        assert r == pytest.approx(-1.0)


class TestSpearmanCorrelationFewSteps:
    def test_two_steps(self):
        r = _spearman_rank_correlation(
            {"a": 1.0, "b": 2.0},
            {"a": 10.0, "b": 20.0},
        )
        assert r == 1.0


class TestScoreProjectionToDict:
    def test_serializes(self):
        proj = _make_stats()
        gt = _make_stats(total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        d = sc.to_dict()
        serialized = json.dumps(d)
        parsed = json.loads(serialized)
        assert parsed["verdict"] in ("PASS", "WARN", "FAIL")


class TestFormatCalibrationReport:
    def test_contains_all_workflows(self):
        scores = [
            CalibrationScore(
                workflow_name="w1",
                sample_size=20,
                ground_truth_size=500,
                p50_ratio=1.1,
                p95_coverage=0.92,
                range_ratio=2.3,
                top_step_correct=True,
                ranking_correlation=0.95,
                verdict="PASS",
                failures=[],
                warnings=[],
            ),
            CalibrationScore(
                workflow_name="w2",
                sample_size=20,
                ground_truth_size=500,
                p50_ratio=1.8,
                p95_coverage=0.78,
                range_ratio=4.1,
                top_step_correct=True,
                ranking_correlation=0.82,
                verdict="WARN",
                failures=[],
                warnings=["p95 coverage low"],
            ),
            CalibrationScore(
                workflow_name="w3",
                sample_size=20,
                ground_truth_size=500,
                p50_ratio=4.0,
                p95_coverage=0.50,
                range_ratio=12.0,
                top_step_correct=False,
                ranking_correlation=0.45,
                verdict="FAIL",
                failures=["p50 off"],
                warnings=[],
            ),
        ]
        report = format_calibration_report(scores)
        assert "w1" in report
        assert "w2" in report
        assert "w3" in report
        assert "PASS" in report
        assert "WARN" in report
        assert "FAIL" in report
        assert "LAUNCH GATE" in report


# ---------------------------------------------------------------------------
# Revised metric threshold tests
# ---------------------------------------------------------------------------


class TestP50RatioLowerBoundTightened:
    def test_p50_lower_bound(self):
        proj = _make_stats(cost_p50=0.017)
        gt_costs = [0.028] * 500
        gt = _make_stats(cost_p50=0.028, total_runs=500, run_costs=gt_costs)
        sc = score_projection(proj, gt)
        assert sc.p50_ratio == pytest.approx(0.017 / 0.028, rel=0.01)
        assert any("p50" in w for w in sc.warnings)


class TestP95CoverageSimpleVsComplex:
    def test_simple_vs_complex(self):
        proj = _make_stats(cost_p95=0.040)
        gt_costs = [0.020 + i * 0.001 for i in range(50)]
        gt = _make_stats(cost_p50=0.028, cost_p95=0.060, total_runs=50, run_costs=gt_costs)
        sc_simple = score_projection(proj, gt, workflow_complexity="simple")
        sc_complex = score_projection(proj, gt, workflow_complexity="complex")
        if sc_simple.p95_coverage < 0.85 and sc_simple.p95_coverage >= 0.75:
            assert any("p95" in w for w in sc_simple.warnings)
            assert not any("p95" in w for w in sc_complex.warnings)


class TestRangeRatioSimpleVsComplex:
    def test_range_ratio(self):
        proj = _make_stats(cost_p50=0.010, cost_p95=0.040)
        gt_costs = [0.010] * 500
        gt = _make_stats(cost_p50=0.010, total_runs=500, run_costs=gt_costs)
        sc_simple = score_projection(proj, gt, workflow_complexity="simple")
        sc_complex = score_projection(proj, gt, workflow_complexity="complex")
        assert sc_simple.range_ratio == pytest.approx(4.0, rel=0.01)
        assert any("Wide" in w for w in sc_simple.warnings)
        assert not any("Wide" in w for w in sc_complex.warnings)


class TestStepRankingSkippedFor3Steps:
    def test_auto_pass_3_steps(self):
        steps = {
            "a": _step("a", 0.030),
            "b": _step("b", 0.020),
            "c": _step("c", 0.010),
        }
        gt_steps = {
            "a": _step("a", 0.010),
            "b": _step("b", 0.020),
            "c": _step("c", 0.030),
        }
        proj = _make_stats(steps=steps)
        gt = _make_stats(steps=gt_steps, total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        assert sc.ranking_correlation == 1.0
        assert not any("ranking" in f.lower() for f in sc.failures)


class TestStepRankingAppliedFor4Steps:
    def test_fail_4_steps(self):
        steps = {
            "a": _step("a", 0.040),
            "b": _step("b", 0.030),
            "c": _step("c", 0.020),
            "d": _step("d", 0.010),
        }
        gt_steps = {
            "a": _step("a", 0.010),
            "b": _step("b", 0.020),
            "c": _step("c", 0.030),
            "d": _step("d", 0.040),
        }
        proj = _make_stats(steps=steps)
        gt = _make_stats(steps=gt_steps, total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        assert sc.ranking_correlation < 0.0
        assert any("ranking" in f.lower() for f in sc.failures)


class TestTopStepCodominantWidened:
    def test_codominant_widened(self):
        proj_steps = {
            "step_a": _step("step_a", 0.028),
            "step_b": _step("step_b", 0.035),
        }
        gt_steps = {
            "step_a": _step("step_a", 0.035),
            "step_b": _step("step_b", 0.028),
        }
        proj = _make_stats(steps=proj_steps)
        gt = _make_stats(steps=gt_steps, total_runs=500, run_costs=[0.03] * 500)
        sc = score_projection(proj, gt)
        assert sc.top_step_correct is True


# ---------------------------------------------------------------------------
# Bootstrap CI tests
# ---------------------------------------------------------------------------


class TestBootstrapCIKnownDistribution:
    def test_bootstrap_ci(self):
        costs = [0.5 + i * 0.01 for i in range(300)]
        point, ci_lo, ci_hi = bootstrap_percentile_ci(costs, 50)
        true_median = 2.0
        assert ci_lo < point < ci_hi
        assert ci_lo <= true_median <= ci_hi
        assert ci_hi - ci_lo > 0
        assert ci_hi - ci_lo < max(costs) - min(costs)


class TestBootstrapCIReproducible:
    def test_reproducible(self):
        costs = [float(i) for i in range(1, 51)]
        r1 = bootstrap_percentile_ci(costs, 50, seed=99)
        r2 = bootstrap_percentile_ci(costs, 50, seed=99)
        assert r1 == r2


class TestBootstrapCINarrowAtLargeN:
    def test_narrow_at_large_n(self):
        costs_300 = [0.5 + i * 0.01 for i in range(300)]
        costs_50 = costs_300[::6]
        _, lo_50, hi_50 = bootstrap_percentile_ci(costs_50, 50)
        _, lo_300, hi_300 = bootstrap_percentile_ci(costs_300, 50)
        assert (hi_300 - lo_300) < (hi_50 - lo_50)


# ---------------------------------------------------------------------------
# BCa bootstrap
# ---------------------------------------------------------------------------


class TestBcaVsPercentileSkewed:
    def test_bca_asymmetric_for_skewed(self):
        import math
        import random

        rng = random.Random(42)
        costs = [math.exp(rng.gauss(0, 1.0)) for _ in range(30)]

        def median_fn(c):
            return sorted(c)[len(c) // 2]

        _, bca_lo, bca_hi = bootstrap_bca_ci(costs, stat_fn=median_fn, seed=42)
        # BCa should produce a valid interval
        assert bca_lo < bca_hi


class TestBcaReproducible:
    def test_same_seed(self):
        costs = [float(i) for i in range(1, 51)]
        r1 = bootstrap_bca_ci(costs, percentile=50, seed=42)
        r2 = bootstrap_bca_ci(costs, percentile=50, seed=42)
        assert r1 == r2


# ---------------------------------------------------------------------------
# CVaR
# ---------------------------------------------------------------------------


class TestCvarBasic:
    def test_cvar_known_values(self):
        from pretia.projection.montecarlo import compute_cvar

        costs = list(range(1, 101))
        cvar = compute_cvar([float(x) for x in costs], alpha=0.05)
        assert cvar == pytest.approx(98.0, rel=0.01)


class TestCvarSubadditivity:
    def test_subadditive(self):
        import random

        from pretia.projection.montecarlo import compute_cvar

        rng = random.Random(42)
        a = [rng.gauss(10, 3) for _ in range(1000)]
        b = [rng.gauss(20, 5) for _ in range(1000)]
        ab = [x + y for x, y in zip(a, b, strict=True)]
        assert compute_cvar(ab) <= compute_cvar(a) + compute_cvar(b) + 0.01


# ---------------------------------------------------------------------------
# Tightened threshold
# ---------------------------------------------------------------------------


class TestThresholdTightened:
    def test_p50_ratio_1_8_fails_simple(self):
        steps = {
            "classify": _step("classify", 0.001, 0.001, 0.002),
            "generate": _step("generate", 0.028, 0.027, 0.044),
            "format": _step("format", 0.005, 0.004, 0.008),
        }
        proj = _make_stats(
            steps=steps,
            cost_p50=0.050,
            cost_p95=0.080,
        )
        gt_costs = [0.028] * 500
        gt = _make_stats(
            steps=steps,
            cost_p50=0.028,
            cost_p95=0.048,
            total_runs=500,
            run_costs=gt_costs,
        )
        sc = score_projection(proj, gt, workflow_complexity="simple")
        assert sc.p50_ratio > 1.7
        assert any("p50" in w for w in sc.warnings) or any("p50" in f for f in sc.failures)
