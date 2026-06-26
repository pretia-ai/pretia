"""Tests for the 3-comparison scoring protocol and failure attribution."""

from __future__ import annotations

from pretia.projection.stats import (
    PercentileStats,
    ProfilingStats,
    RunStats,
    StepStats,
)
from pretia.validation.scoring import (
    _COMPARISON_TARGETS,
    ComparisonScore,
    score_comparison,
)
from pretia.validation.suite import (
    FailureAttribution,
    _compute_recovery,
    attribute_failure,
)


def _ps(
    mean: float = 0.03,
    p50: float = 0.028,
    p75: float = 0.035,
    p95: float = 0.048,
    std: float = 0.01,
) -> PercentileStats:
    return PercentileStats(
        min=mean * 0.5,
        max=p95 * 1.5,
        mean=mean,
        std=std,
        p50=p50,
        p75=p75,
        p90=p95 * 0.9,
        p95=p95,
        p99=p95 * 1.3,
    )


def _step(name: str, cost_mean: float = 0.01) -> StepStats:
    cost_ps = _ps(cost_mean, cost_mean * 0.95, cost_mean * 1.15, cost_mean * 1.7, cost_mean * 0.3)
    tok = _ps(300, 280, 340, 450, 80)
    itr = PercentileStats(min=1, max=1, mean=1, std=0, p50=1, p75=1, p90=1, p95=1, p99=1)
    return StepStats(
        step_name=name,
        step_type="llm",
        model="gpt-4o-mini",
        call_count=50,
        runs_present=50,
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
    cost_mean: float = 0.03,
    cost_p50: float = 0.028,
    cost_p75: float = 0.035,
    cost_p95: float = 0.048,
    total_runs: int = 50,
    run_costs: list[float] | None = None,
) -> ProfilingStats:
    steps = {
        "classify": _step("classify", 0.001),
        "generate": _step("generate", 0.028),
    }
    cpr = _ps(cost_mean, cost_p50, cost_p75, cost_p95)
    if run_costs is None:
        run_costs = [
            cost_mean * (0.8 + 0.4 * i / max(total_runs - 1, 1)) for i in range(total_runs)
        ]
    run_stats = [
        RunStats(
            run_index=i,
            total_cost=c,
            total_tokens=600,
            total_input_tokens=340,
            total_output_tokens=260,
            step_count=2,
            duration_ms=400,
        )
        for i, c in enumerate(run_costs)
    ]
    return ProfilingStats(
        step_stats=steps,
        run_stats=run_stats,
        cost_per_run=cpr,
        tokens_per_run=_ps(600, 580, 650, 800, 100),
        total_runs=total_runs,
        total_steps=2,
    )


class TestScoreComparisonNoDriftPass:
    def test_passes_when_accurate(self):
        proj = _make_stats(cost_mean=0.030, cost_p50=0.028, cost_p75=0.035, cost_p95=0.048)
        gt = _make_stats(
            cost_mean=0.031, cost_p50=0.029, cost_p75=0.036, cost_p95=0.050, total_runs=200
        )
        score = score_comparison(proj, gt, "W1", "A")
        assert score.passes
        assert score.comparison == "A"
        assert score.mean_error_pct < 10.0


class TestScoreComparisonNoDriftFail:
    def test_fails_when_mean_error_exceeds_target(self):
        proj = _make_stats(cost_mean=0.045)
        gt = _make_stats(cost_mean=0.030, total_runs=200)
        score = score_comparison(proj, gt, "W1", "A")
        assert not score.passes
        assert score.mean_error_pct > 10.0
        assert any("Mean error" in f for f in score.failures)


class TestScoreComparisonDriftedRelaxedTargets:
    def test_15pct_mean_error_passes_drifted_fails_no_drift(self):
        proj = _make_stats(cost_mean=0.0345)
        gt = _make_stats(cost_mean=0.030, total_runs=200)
        score_drifted = score_comparison(proj, gt, "W1", "B")
        score_no_drift = score_comparison(proj, gt, "W1", "A")
        assert score_drifted.mean_error_pct < 20.0
        assert score_no_drift.mean_error_pct > 10.0


class TestComparisonScoreSerialization:
    def test_roundtrip(self):
        score = ComparisonScore(
            workflow_name="W1",
            comparison="A",
            mean_error_pct=5.0,
            p75_error_pct=8.0,
            ci_coverage_pct=90.0,
            monthly_error_pct=5.0,
            cvar95_error_pct=15.0,
            passes=True,
            failures=[],
        )
        d = score.to_dict()
        restored = ComparisonScore.from_dict(d)
        assert restored.workflow_name == score.workflow_name
        assert restored.comparison == score.comparison
        assert restored.passes == score.passes
        assert restored.mean_error_pct == score.mean_error_pct
        assert restored.cvar95_error_pct == score.cvar95_error_pct


class TestFailureAttributionBucket1:
    def test_a_fails_returns_bucket_1(self):
        score_a = ComparisonScore(
            workflow_name="W1",
            comparison="A",
            mean_error_pct=25.0,
            p75_error_pct=30.0,
            ci_coverage_pct=60.0,
            monthly_error_pct=25.0,
            cvar95_error_pct=50.0,
            passes=False,
            failures=["Mean error 25.0% exceeds 10% target"],
        )
        result = attribute_failure("W1", score_a, None, None)
        assert result is not None
        assert result.bucket == 1
        assert result.bucket_label == "engine_problem"

    def test_a_none_returns_bucket_1(self):
        result = attribute_failure("W1", None, None, None)
        assert result is not None
        assert result.bucket == 1


class TestFailureAttributionBucket2:
    def test_a_pass_b_fail_c_recovers(self):
        score_a = ComparisonScore(
            workflow_name="W13",
            comparison="A",
            mean_error_pct=7.0,
            p75_error_pct=11.0,
            ci_coverage_pct=89.0,
            monthly_error_pct=6.5,
            cvar95_error_pct=18.0,
            passes=True,
        )
        score_b = ComparisonScore(
            workflow_name="W13",
            comparison="B",
            mean_error_pct=22.0,
            p75_error_pct=28.0,
            ci_coverage_pct=72.0,
            monthly_error_pct=21.0,
            cvar95_error_pct=45.0,
            passes=False,
            failures=["Mean error 22.0% exceeds 20% target"],
        )
        score_c = ComparisonScore(
            workflow_name="W13",
            comparison="C",
            mean_error_pct=9.0,
            p75_error_pct=14.0,
            ci_coverage_pct=86.0,
            monthly_error_pct=8.5,
            cvar95_error_pct=22.0,
            passes=True,
        )
        result = attribute_failure("W13", score_a, score_b, score_c)
        assert result is not None
        assert result.bucket == 2
        assert result.bucket_label == "drift_sensitivity"
        assert "--traffic-mix" in result.recommended_action


class TestFailureAttributionBucket3:
    def test_a_pass_b_fail_c_no_recovery(self):
        score_a = ComparisonScore(
            workflow_name="W19",
            comparison="A",
            mean_error_pct=8.0,
            p75_error_pct=12.0,
            ci_coverage_pct=87.0,
            monthly_error_pct=7.5,
            cvar95_error_pct=20.0,
            passes=True,
        )
        score_b = ComparisonScore(
            workflow_name="W19",
            comparison="B",
            mean_error_pct=35.0,
            p75_error_pct=40.0,
            ci_coverage_pct=55.0,
            monthly_error_pct=34.0,
            cvar95_error_pct=60.0,
            passes=False,
            failures=["Mean error 35.0% exceeds 20% target"],
        )
        score_c = ComparisonScore(
            workflow_name="W19",
            comparison="C",
            mean_error_pct=30.0,
            p75_error_pct=36.0,
            ci_coverage_pct=60.0,
            monthly_error_pct=29.0,
            cvar95_error_pct=55.0,
            passes=False,
            failures=["Mean error 30.0% exceeds 20% target"],
        )
        result = attribute_failure("W19", score_a, score_b, score_c)
        assert result is not None
        assert result.bucket == 3
        assert result.bucket_label == "structural_drift"
        assert "re-profile" in result.recommended_action.lower()


class TestFailureAttributionAllPass:
    def test_a_and_b_pass_returns_none(self):
        score_a = ComparisonScore(
            workflow_name="W1",
            comparison="A",
            mean_error_pct=5.0,
            p75_error_pct=8.0,
            ci_coverage_pct=92.0,
            monthly_error_pct=5.0,
            cvar95_error_pct=15.0,
            passes=True,
        )
        score_b = ComparisonScore(
            workflow_name="W1",
            comparison="B",
            mean_error_pct=12.0,
            p75_error_pct=18.0,
            ci_coverage_pct=80.0,
            monthly_error_pct=11.0,
            cvar95_error_pct=30.0,
            passes=True,
        )
        result = attribute_failure("W1", score_a, score_b, None)
        assert result is None


class TestComputeRecovery:
    def test_full_recovery(self):
        score_a = ComparisonScore(
            workflow_name="W1",
            comparison="A",
            mean_error_pct=5.0,
            p75_error_pct=8.0,
            ci_coverage_pct=90.0,
            monthly_error_pct=5.0,
            cvar95_error_pct=15.0,
            passes=True,
        )
        score_b = ComparisonScore(
            workflow_name="W1",
            comparison="B",
            mean_error_pct=25.0,
            p75_error_pct=30.0,
            ci_coverage_pct=65.0,
            monthly_error_pct=24.0,
            cvar95_error_pct=50.0,
            passes=False,
        )
        score_c = ComparisonScore(
            workflow_name="W1",
            comparison="C",
            mean_error_pct=5.0,
            p75_error_pct=8.0,
            ci_coverage_pct=90.0,
            monthly_error_pct=5.0,
            cvar95_error_pct=15.0,
            passes=True,
        )
        recovery = _compute_recovery(score_a, score_b, score_c)
        assert recovery == 100.0

    def test_no_recovery(self):
        score_a = ComparisonScore(
            workflow_name="W1",
            comparison="A",
            mean_error_pct=5.0,
            p75_error_pct=8.0,
            ci_coverage_pct=90.0,
            monthly_error_pct=5.0,
            cvar95_error_pct=15.0,
            passes=True,
        )
        score_b = ComparisonScore(
            workflow_name="W1",
            comparison="B",
            mean_error_pct=25.0,
            p75_error_pct=30.0,
            ci_coverage_pct=65.0,
            monthly_error_pct=24.0,
            cvar95_error_pct=50.0,
            passes=False,
        )
        score_c = ComparisonScore(
            workflow_name="W1",
            comparison="C",
            mean_error_pct=26.0,
            p75_error_pct=31.0,
            ci_coverage_pct=64.0,
            monthly_error_pct=25.0,
            cvar95_error_pct=51.0,
            passes=False,
        )
        recovery = _compute_recovery(score_a, score_b, score_c)
        assert recovery == 0.0


class TestFailureAttributionSerialization:
    def test_to_dict(self):
        attr = FailureAttribution(
            workflow_name="W13",
            bucket=2,
            bucket_label="drift_sensitivity",
            explanation="Test explanation",
            recommended_action="Use --traffic-mix",
        )
        d = attr.to_dict()
        assert d["bucket"] == 2
        assert d["bucket_label"] == "drift_sensitivity"
        assert d["workflow_name"] == "W13"


class TestComparisonTargets:
    def test_no_drift_targets_stricter(self):
        nd = _COMPARISON_TARGETS["no_drift"]
        dr = _COMPARISON_TARGETS["drifted"]
        assert nd["mean_error"] < dr["mean_error"]
        assert nd["ci_coverage"] > dr["ci_coverage"]
        assert nd["cvar95_error"] < dr["cvar95_error"]
