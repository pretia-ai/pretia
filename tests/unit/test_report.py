"""Tests for CLI report formatting."""

from __future__ import annotations

from datetime import UTC, datetime

from rich.panel import Panel
from rich.table import Table

from agentcost.ci.report import _fmt_cost, format_cli_report
from agentcost.store import ProfilingSession


def _make_session() -> ProfilingSession:
    return ProfilingSession(
        workflow_name="test_agent.py",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 14, 0, 0, tzinfo=UTC),
        sample_size=10,
        input_mode="auto-generate",
        runs=[],
        metadata={},
    )


def _make_cost_summary() -> dict:
    return {
        "per_step": {
            "expensive_step": {
                "count": 10,
                "cost_mean": 0.05,
                "cost_min": 0.02,
                "cost_max": 0.10,
                "cost_p50": 0.045,
                "cost_p95": 0.09,
                "input_tokens_mean": 2000,
                "output_tokens_mean": 500,
                "duration_ms_mean": 300,
                "max_iteration": 1,
                "model": "gpt-4o",
                "step_type": "llm",
                "tier": "mid",
            },
            "cheap_step": {
                "count": 10,
                "cost_mean": 0.002,
                "cost_min": 0.001,
                "cost_max": 0.004,
                "cost_p50": 0.002,
                "cost_p95": 0.003,
                "input_tokens_mean": 200,
                "output_tokens_mean": 50,
                "duration_ms_mean": 80,
                "max_iteration": 1,
                "model": "gpt-4o-mini",
                "step_type": "llm",
                "tier": "fast",
            },
            "tool_step": {
                "count": 10,
                "cost_mean": 0.0,
                "cost_min": 0.0,
                "cost_max": 0.0,
                "cost_p50": 0.0,
                "cost_p95": 0.0,
                "input_tokens_mean": 0,
                "output_tokens_mean": 0,
                "duration_ms_mean": 50,
                "max_iteration": 1,
                "model": "",
                "step_type": "tool",
                "tier": "tool",
            },
        },
        "run_totals": [0.05, 0.06],
        "mean_cost_per_run": 0.05,
        "min_cost_per_run": 0.04,
        "max_cost_per_run": 0.07,
        "p95_cost_per_run": 0.065,
        "total_session_cost": 0.55,
        "projection_100_day": 150.0,
        "projection_1000_day": 1500.0,
        "projection_10000_day": 15000.0,
    }


class TestFormatReport:
    def test_returns_non_empty(self):
        result = format_cli_report(
            _make_session(), _make_cost_summary(),
        )
        assert len(result) > 0

    def test_contains_tables_and_panels(self):
        result = format_cli_report(
            _make_session(), _make_cost_summary(),
        )
        types = {type(r) for r in result}
        assert Table in types
        assert Panel in types

    def test_steps_ordered_by_cost_desc(self):
        summary = _make_cost_summary()
        result = format_cli_report(_make_session(), summary)

        step_table = next(r for r in result if isinstance(r, Table))
        step_col = step_table.columns[0]
        assert step_col._cells[0] == "expensive_step"


class TestCostFormatting:
    def test_large_amount(self):
        assert _fmt_cost(12.34) == "$12.34"

    def test_small_amount(self):
        assert _fmt_cost(0.0023) == "$0.0023"

    def test_zero(self):
        assert _fmt_cost(0) == "$0.00"


class TestProjectionMath:
    def test_1000_per_day(self):
        summary = _make_cost_summary()
        summary["mean_cost_per_run"] = 0.05
        summary["projection_1000_day"] = 0.05 * 1000 * 30

        result = format_cli_report(_make_session(), summary)
        proj_panel = [
            r for r in result
            if isinstance(r, Panel) and "Projection" in str(r.title)
        ]
        assert len(proj_panel) == 1


class TestFlags:
    def test_high_iteration_flag(self):
        summary = _make_cost_summary()
        summary["per_step"]["loopy"] = {
            "count": 10,
            "cost_mean": 0.01,
            "cost_min": 0.005,
            "cost_max": 0.02,
            "cost_p50": 0.01,
            "cost_p95": 0.015,
            "input_tokens_mean": 100,
            "output_tokens_mean": 50,
            "duration_ms_mean": 50,
            "max_iteration": 8,
            "model": "gpt-4o-mini",
            "step_type": "llm",
            "tier": "fast",
        }

        result = format_cli_report(_make_session(), summary)
        flag_panels = [
            r for r in result
            if isinstance(r, Panel) and "Flags" in str(r.title)
        ]
        assert len(flag_panels) == 1
