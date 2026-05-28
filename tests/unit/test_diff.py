"""Tests for the diff engine: comparing baselines to new profiles."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agentcost.ci.baseline import Baseline, BaselineStep
from agentcost.ci.diff import diff_baseline, format_diff_report
from agentcost.store import ProfilingSession


def _make_baseline_step(
    model: str = "gpt-4o-mini",
    cost_mean: float = 0.001,
    cost_p50: float = 0.001,
    cost_p95: float = 0.002,
) -> BaselineStep:
    return BaselineStep(
        model=model,
        tokens_input={"p50": 230.0, "p95": 320.0},
        tokens_output={"p50": 65.0, "p95": 95.0},
        cost_per_run={"p50": cost_p50, "p95": cost_p95, "mean": cost_mean},
        iterations={"mean": 1.0, "max": 1},
        system_prompt_hash="abc",
        system_prompt_tokens=50,
        output_format="text",
        flags=[],
        task_complexity_tier=None,
    )


def _make_baseline(
    steps: dict[str, BaselineStep] | None = None,
    total_monthly: dict[str, float] | None = None,
    patterns: list[str] | None = None,
) -> Baseline:
    if steps is None:
        steps = {
            "classify": _make_baseline_step("gpt-4o-mini", 0.001, 0.001, 0.002),
            "generate": _make_baseline_step("gpt-4o", 0.028, 0.027, 0.044),
        }
    if total_monthly is None:
        total_monthly = {"p50": 840.0, "p75": 1050.0, "p90": 1260.0, "p95": 1440.0}
    if patterns is None:
        patterns = []
    return Baseline(
        version="1.0",
        workflow="test_agent.py",
        profiled_at="2026-05-25T14:00:00+00:00",
        sample_size=20,
        traffic_assumption="1000/day",
        input_source="auto-generate",
        collector_type="auto",
        confidence_tier="MODERATE",
        steps=steps,
        total_monthly=total_monthly,
        patterns=patterns,
        assumptions=["Test baseline."],
    )


def _make_session_with_stats(
    step_stats: dict | None = None,
    cost_per_run: dict | None = None,
    patterns: list | None = None,
) -> ProfilingSession:
    if step_stats is None:
        step_stats = {
            "classify": {
                "step_name": "classify", "step_type": "llm",
                "model": "gpt-4o-mini", "call_count": 20, "runs_present": 20,
                "input_tokens": {"mean": 250.0, "p50": 230.0, "p75": 270.0,
                                 "p90": 300.0, "p95": 320.0, "p99": 350.0,
                                 "min": 150.0, "max": 400.0, "std": 50.0},
                "output_tokens": {"mean": 70.0, "p50": 65.0, "p75": 80.0,
                                  "p90": 90.0, "p95": 95.0, "p99": 100.0,
                                  "min": 40.0, "max": 120.0, "std": 15.0},
                "total_tokens": {"mean": 320.0, "p50": 295.0, "p75": 350.0,
                                 "p90": 390.0, "p95": 415.0, "p99": 450.0,
                                 "min": 190.0, "max": 520.0, "std": 65.0},
                "cost": {"mean": 0.001, "p50": 0.001, "p75": 0.0014,
                         "p90": 0.0016, "p95": 0.0018, "p99": 0.002,
                         "min": 0.0005, "max": 0.003, "std": 0.0004},
                "duration_ms": {"mean": 200.0, "p50": 180.0, "p75": 220.0,
                                "p90": 260.0, "p95": 290.0, "p99": 320.0,
                                "min": 100.0, "max": 400.0, "std": 50.0},
                "context_size": {"mean": 250.0, "p50": 230.0, "p75": 270.0,
                                 "p90": 300.0, "p95": 320.0, "p99": 350.0,
                                 "min": 150.0, "max": 400.0, "std": 50.0},
                "iterations_per_run": {"mean": 1.0, "p50": 1.0, "p75": 1.0,
                                       "p90": 1.0, "p95": 1.0, "p99": 1.0,
                                       "min": 1.0, "max": 1.0, "std": 0.0},
                "mean_iterations": 1.0,
            },
            "generate": {
                "step_name": "generate", "step_type": "llm",
                "model": "gpt-4o", "call_count": 20, "runs_present": 20,
                "input_tokens": {"mean": 1500.0, "p50": 1400.0, "p75": 1700.0,
                                 "p90": 2000.0, "p95": 2200.0, "p99": 2500.0,
                                 "min": 800.0, "max": 3000.0, "std": 400.0},
                "output_tokens": {"mean": 400.0, "p50": 380.0, "p75": 450.0,
                                  "p90": 520.0, "p95": 580.0, "p99": 650.0,
                                  "min": 200.0, "max": 800.0, "std": 100.0},
                "total_tokens": {"mean": 1900.0, "p50": 1780.0, "p75": 2150.0,
                                 "p90": 2520.0, "p95": 2780.0, "p99": 3150.0,
                                 "min": 1000.0, "max": 3800.0, "std": 500.0},
                "cost": {"mean": 0.028, "p50": 0.027, "p75": 0.033,
                         "p90": 0.039, "p95": 0.044, "p99": 0.052,
                         "min": 0.014, "max": 0.065, "std": 0.01},
                "duration_ms": {"mean": 500.0, "p50": 450.0, "p75": 600.0,
                                "p90": 700.0, "p95": 800.0, "p99": 1000.0,
                                "min": 200.0, "max": 1200.0, "std": 200.0},
                "context_size": {"mean": 1500.0, "p50": 1400.0, "p75": 1700.0,
                                 "p90": 2000.0, "p95": 2200.0, "p99": 2500.0,
                                 "min": 800.0, "max": 3000.0, "std": 400.0},
                "iterations_per_run": {"mean": 1.0, "p50": 1.0, "p75": 1.0,
                                       "p90": 1.0, "p95": 1.0, "p99": 1.0,
                                       "min": 1.0, "max": 1.0, "std": 0.0},
                "mean_iterations": 1.0,
            },
        }
    if cost_per_run is None:
        cost_per_run = {
            "mean": 0.03, "p50": 0.028, "p75": 0.035,
            "p90": 0.042, "p95": 0.048, "p99": 0.06,
            "min": 0.015, "max": 0.08, "std": 0.012,
        }
    if patterns is None:
        patterns = []

    return ProfilingSession(
        workflow_name="test_agent.py",
        workflow_hash="def456",
        profiled_at=datetime(2026, 5, 26, 14, 0, 0, tzinfo=UTC),
        sample_size=20,
        input_mode="auto-generate",
        runs=[],
        metadata={
            "stats": {
                "total_runs": 20,
                "total_steps": 40,
                "cost_per_run": cost_per_run,
                "tokens_per_run": {
                    "mean": 2000.0, "p50": 1800.0, "p75": 2400.0,
                    "p90": 2800.0, "p95": 3200.0, "p99": 3800.0,
                    "min": 1000.0, "max": 4500.0, "std": 600.0,
                },
                "step_stats": step_stats,
                "run_stats": [],
            },
            "patterns": patterns,
            "projection": {
                "method": "linear",
                "traffic_volumes": [100, 1000, 10000],
                "projections": {
                    "1000": {
                        "daily_volume": 1000,
                        "monthly_cost": {
                            "p50": 840.0, "p75": 1050.0, "p90": 1260.0,
                            "p95": 1440.0, "p99": 1800.0, "mean": 900.0,
                        },
                        "daily_cost": {"p50": 28.0, "mean": 30.0,
                                       "p75": 35.0, "p90": 42.0,
                                       "p95": 48.0, "p99": 60.0},
                        "cost_per_run": {"p50": 0.028, "p75": 0.035,
                                         "p90": 0.042, "p95": 0.048,
                                         "p99": 0.06, "mean": 0.03},
                    },
                },
                "confidence": {"score": 72, "tier": "MODERATE",
                               "display_range": "p50 – p95",
                               "language": "estimated",
                               "deductions": [], "bonuses": []},
                "warnings": [],
                "patterns_detected": [],
            },
            "confidence": {"score": 72, "tier": "MODERATE",
                           "display_range": "p50 – p95",
                           "language": "estimated",
                           "deductions": [], "bonuses": []},
        },
    )


class TestDiffNoChange:
    def test_unchanged(self):
        bl = _make_baseline()
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)

        assert abs(result.total_monthly_pct_change.get("p50", 0)) < 5
        assert "unchanged" in result.summary.lower()
        assert len(result.model_changes) == 0


class TestDiffCostIncrease:
    def test_detects_increase(self):
        bl = _make_baseline(
            steps={
                "classify": _make_baseline_step("gpt-4o-mini", 0.001),
                "review": _make_baseline_step("gpt-4o", 0.007, 0.006, 0.012),
            },
            total_monthly={"p50": 240.0, "p75": 300.0, "p90": 360.0, "p95": 420.0},
        )
        # New session: review cost jumped to 0.092
        step_stats = dict(_make_session_with_stats().metadata["stats"]["step_stats"])
        step_stats["review"] = {
            "step_name": "review", "step_type": "llm",
            "model": "gpt-4o", "call_count": 20, "runs_present": 20,
            "input_tokens": {"mean": 5000.0, "p50": 4500.0, "p75": 6000.0,
                             "p90": 7000.0, "p95": 7500.0, "p99": 8000.0,
                             "min": 2000.0, "max": 10000.0, "std": 1500.0},
            "output_tokens": {"mean": 400.0, "p50": 380.0, "p75": 450.0,
                              "p90": 520.0, "p95": 580.0, "p99": 650.0,
                              "min": 200.0, "max": 800.0, "std": 100.0},
            "total_tokens": {"mean": 5400.0, "p50": 4880.0, "p75": 6450.0,
                             "p90": 7520.0, "p95": 8080.0, "p99": 8650.0,
                             "min": 2200.0, "max": 10800.0, "std": 1600.0},
            "cost": {"mean": 0.092, "p50": 0.085, "p75": 0.11,
                     "p90": 0.13, "p95": 0.14, "p99": 0.16,
                     "min": 0.04, "max": 0.2, "std": 0.03},
            "duration_ms": {"mean": 800.0, "p50": 700.0, "p75": 900.0,
                            "p90": 1000.0, "p95": 1100.0, "p99": 1200.0,
                            "min": 400.0, "max": 1500.0, "std": 200.0},
            "context_size": {"mean": 5000.0, "p50": 4500.0, "p75": 6000.0,
                             "p90": 7000.0, "p95": 7500.0, "p99": 8000.0,
                             "min": 2000.0, "max": 10000.0, "std": 1500.0},
            "iterations_per_run": {"mean": 1.0, "p50": 1.0, "p75": 1.0,
                                   "p90": 1.0, "p95": 1.0, "p99": 1.0,
                                   "min": 1.0, "max": 1.0, "std": 0.0},
            "mean_iterations": 1.0,
        }
        session = _make_session_with_stats(step_stats=step_stats)
        result = diff_baseline(bl, session)

        assert result.total_monthly_pct_change["p50"] > 0
        assert "review" in result.step_diffs
        assert result.step_diffs["review"].cost_change_pct > 100
        assert "increased" in result.summary.lower()


class TestDiffCostDecrease:
    def test_detects_decrease(self):
        bl = _make_baseline(
            total_monthly={"p50": 2000.0, "p75": 2500.0, "p90": 3000.0, "p95": 3500.0},
        )
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)
        assert result.total_monthly_pct_change["p50"] < 0
        assert "decreased" in result.summary.lower()


class TestDiffNewStep:
    def test_detects_new_step(self):
        bl = _make_baseline(
            steps={"classify": _make_baseline_step()},
        )
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)
        assert "generate" in result.new_steps


class TestDiffRemovedStep:
    def test_detects_removed_step(self):
        bl = _make_baseline(
            steps={
                "classify": _make_baseline_step(),
                "generate": _make_baseline_step("gpt-4o", 0.028),
                "old_step": _make_baseline_step("gpt-4o", 0.005),
            },
        )
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)
        assert "old_step" in result.removed_steps


class TestDiffModelChange:
    def test_detects_model_change(self):
        bl = _make_baseline(
            steps={
                "classify": _make_baseline_step("gpt-4o-mini"),
                "generate": _make_baseline_step("claude-opus-4", 0.028),
            },
        )
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)
        assert len(result.model_changes) == 1
        mc = result.model_changes[0]
        assert mc.step_name == "generate"
        assert mc.old_model == "claude-opus-4"
        assert mc.new_model == "gpt-4o"


class TestDiffPatternChanges:
    def test_detects_resolved_pattern(self):
        bl = _make_baseline(patterns=["context_growth"])
        session = _make_session_with_stats(patterns=[])
        result = diff_baseline(bl, session)
        assert "context_growth" in result.pattern_changes.resolved_patterns
        assert len(result.pattern_changes.new_patterns) == 0


class TestFormatDiffReport:
    def test_produces_readable_output(self):
        bl = _make_baseline(
            steps={
                "classify": _make_baseline_step("gpt-4o-mini"),
                "generate": _make_baseline_step("claude-opus-4", 0.028),
            },
        )
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)
        report = format_diff_report(result)

        assert "AgentCost Diff Report" in report
        assert "Step Comparison" in report
        assert "Monthly Projection Change" in report
        assert "Model changes" in report


class TestDiffToDict:
    def test_serializes(self):
        bl = _make_baseline()
        session = _make_session_with_stats()
        result = diff_baseline(bl, session)
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0
