"""Tests for calibration scoring."""

from __future__ import annotations

import json

import pytest

from agentcost.projection.stats import (
    PercentileStats,
    ProfilingStats,
    RunStats,
    StepStats,
)
from agentcost.validation.scoring import (
    CalibrationScore,
    _spearman_rank_correlation,
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
