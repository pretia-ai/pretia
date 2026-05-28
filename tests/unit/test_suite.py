"""Tests for the backtesting suite runner."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agentcost.projection.stats import PercentileStats, StepStats
from agentcost.store import ProfilingSession
from agentcost.validation.suite import (
    BacktestConfig,
    run_backtesting_suite,
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


def _step(name: str, cost_mean: float = 0.01) -> StepStats:
    cost_ps = _ps(cost_mean, cost_mean * 0.95, cost_mean * 1.7, cost_mean * 0.3)
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


def _make_session(
    cost_p50: float = 0.028,
    cost_p95: float = 0.048,
    cost_mean: float = 0.03,
    total_runs: int = 20,
    run_costs: list[float] | None = None,
) -> ProfilingSession:
    steps = {
        "classify": _step("classify", 0.001),
        "generate": _step("generate", 0.028),
        "format": _step("format", 0.005),
    }
    cpr = _ps(cost_mean, cost_p50, cost_p95)
    run_stats: list[dict] = []
    if run_costs:
        for i, c in enumerate(run_costs):
            run_stats.append(
                {
                    "run_index": i,
                    "total_cost": c,
                    "total_tokens": 300,
                    "total_input_tokens": 200,
                    "total_output_tokens": 100,
                    "step_count": 2,
                    "duration_ms": 200,
                }
            )

    stats_dict = {
        "total_runs": total_runs,
        "total_steps": total_runs * 2,
        "cost_per_run": cpr.to_dict(),
        "tokens_per_run": _ps(2000, 1800, 3200, 600).to_dict(),
        "step_stats": {n: s.to_dict() for n, s in steps.items()},
        "run_stats": run_stats,
    }

    return ProfilingSession(
        workflow_name="test.py",
        workflow_hash="abc",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=total_runs,
        input_mode="auto-generate",
        runs=[],
        metadata={"stats": stats_dict, "patterns": []},
    )


def _make_config(name: str) -> BacktestConfig:
    return BacktestConfig(
        name=name,
        archetype="support-agent",
        complexity="simple",
        workflow_path=f"{name}.py",
        description=f"Test workflow {name}",
        expected_models=["gpt-4o-mini"],
        has_loops=False,
        expected_cost_range=(0.01, 0.10),
    )


class TestRunSuiteAllPass:
    def test_all_pass(self):
        configs = [_make_config("w1"), _make_config("w2"), _make_config("w3")]
        gt_costs = [0.020 + (i % 40) * 0.0006 for i in range(500)]
        profiles = {}
        for cfg in configs:
            profiles[cfg.name] = {
                "synth20": _make_session(total_runs=20),
                "synth100": _make_session(total_runs=100),
                "real500": _make_session(
                    total_runs=500,
                    run_costs=gt_costs,
                ),
            }
        result = run_backtesting_suite(profiles, configs)
        assert result.launch_gate is True
        assert result.pass_count == 3
        assert result.fail_count == 0


class TestRunSuiteOneFail:
    def test_one_fail(self):
        configs = [_make_config("w1"), _make_config("w2")]

        gt_costs = [0.028] * 500
        profiles = {
            "w1": {
                "synth20": _make_session(total_runs=20),
                "synth100": _make_session(total_runs=100),
                "real500": _make_session(total_runs=500, run_costs=gt_costs),
            },
            "w2": {
                "synth20": _make_session(cost_p50=0.200, total_runs=20),
                "synth100": _make_session(total_runs=100),
                "real500": _make_session(total_runs=500, run_costs=gt_costs),
            },
        }
        result = run_backtesting_suite(profiles, configs)
        assert result.launch_gate is False
        assert result.fail_count >= 1


class TestRunSuiteMissingProfiles:
    def test_missing_skipped(self):
        configs = [_make_config("w1"), _make_config("w2"), _make_config("w3")]
        gt_costs = [0.028] * 500
        profiles = {
            "w1": {
                "synth20": _make_session(total_runs=20),
                "synth100": _make_session(total_runs=100),
                "real500": _make_session(total_runs=500, run_costs=gt_costs),
            },
            "w2": {
                "synth20": _make_session(total_runs=20),
                "synth100": _make_session(total_runs=100),
                "real500": _make_session(total_runs=500, run_costs=gt_costs),
            },
        }
        result = run_backtesting_suite(profiles, configs)
        assert len(result.results) == 2


class TestConvergenceCalculation:
    def test_convergence_pct(self):
        configs = [_make_config("w1")]
        gt_costs = [0.030] * 500
        profiles = {
            "w1": {
                "synth20": _make_session(cost_p50=0.035, total_runs=20),
                "synth100": _make_session(cost_p50=0.030, total_runs=100),
                "real500": _make_session(total_runs=500, run_costs=gt_costs),
            },
        }
        result = run_backtesting_suite(profiles, configs)
        conv = result.results[0].convergence_20_to_100
        assert conv is not None
        assert abs(conv - 16.7) < 1.0


class TestSuiteResultToDict:
    def test_serializes(self):
        configs = [_make_config("w1")]
        gt_costs = [0.028] * 500
        profiles = {
            "w1": {
                "synth20": _make_session(total_runs=20),
                "synth100": _make_session(total_runs=100),
                "real500": _make_session(total_runs=500, run_costs=gt_costs),
            },
        }
        result = run_backtesting_suite(profiles, configs)
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0
