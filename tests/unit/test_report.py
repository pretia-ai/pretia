"""Tests for CLI report formatting."""

from __future__ import annotations

from datetime import UTC, datetime

from rich.panel import Panel
from rich.table import Table

from agentcost.ci.report import format_cli_report, format_cost, format_tokens
from agentcost.store import ProfilingSession


def _make_session(**overrides) -> ProfilingSession:
    defaults = {
        "workflow_name": "test_agent.py",
        "workflow_hash": "abc123",
        "profiled_at": datetime(2026, 5, 25, 14, 0, 0, tzinfo=UTC),
        "sample_size": 10,
        "input_mode": "auto-generate",
        "runs": [],
        "metadata": {},
    }
    defaults.update(overrides)
    return ProfilingSession(**defaults)


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


def _make_stats_metadata():
    return {
        "stats": {
            "total_runs": 10,
            "total_steps": 30,
            "cost_per_run": {
                "mean": 0.0342,
                "p50": 0.0298,
                "p75": 0.04,
                "p90": 0.048,
                "p95": 0.0512,
                "p99": 0.0687,
                "min": 0.0201,
                "max": 0.0823,
                "std": 0.0145,
            },
            "tokens_per_run": {
                "mean": 2340.0,
                "p50": 2100.0,
                "p75": 2800.0,
                "p90": 3200.0,
                "p95": 3890.0,
                "p99": 4500.0,
                "min": 1200.0,
                "max": 5200.0,
                "std": 800.0,
            },
            "step_stats": {
                "generate": {
                    "step_name": "generate",
                    "step_type": "llm",
                    "model": "gpt-4o",
                    "call_count": 30,
                    "runs_present": 10,
                    "cost": {
                        "mean": 0.0189, "p50": 0.015, "p75": 0.02,
                        "p90": 0.025, "p95": 0.0312, "p99": 0.04,
                        "min": 0.008, "max": 0.045, "std": 0.01,
                    },
                    "total_tokens": {
                        "mean": 2340.0, "p50": 2100.0, "p75": 2800.0,
                        "p90": 3200.0, "p95": 3890.0, "p99": 4500.0,
                        "min": 1200.0, "max": 5200.0, "std": 800.0,
                    },
                    "input_tokens": {
                        "mean": 1800.0, "p50": 1600.0, "p75": 2200.0,
                        "p90": 2500.0, "p95": 3000.0, "p99": 3500.0,
                        "min": 900.0, "max": 4000.0, "std": 600.0,
                    },
                    "output_tokens": {
                        "mean": 540.0, "p50": 500.0, "p75": 600.0,
                        "p90": 700.0, "p95": 890.0, "p99": 1000.0,
                        "min": 300.0, "max": 1200.0, "std": 200.0,
                    },
                    "duration_ms": {
                        "mean": 500.0, "p50": 450.0, "p75": 600.0,
                        "p90": 700.0, "p95": 800.0, "p99": 1000.0,
                        "min": 200.0, "max": 1200.0, "std": 200.0,
                    },
                    "context_size": {
                        "mean": 1800.0, "p50": 1600.0, "p75": 2200.0,
                        "p90": 2500.0, "p95": 3000.0, "p99": 3500.0,
                        "min": 900.0, "max": 4000.0, "std": 600.0,
                    },
                    "iterations_per_run": {
                        "mean": 1.0, "p50": 1.0, "p75": 1.0,
                        "p90": 1.0, "p95": 1.0, "p99": 1.0,
                        "min": 1.0, "max": 1.0, "std": 0.0,
                    },
                    "mean_iterations": 1.0,
                },
                "classify": {
                    "step_name": "classify",
                    "step_type": "llm",
                    "model": "gpt-4o-mini",
                    "call_count": 10,
                    "runs_present": 10,
                    "cost": {
                        "mean": 0.0012, "p50": 0.001, "p75": 0.0013,
                        "p90": 0.0015, "p95": 0.0015, "p99": 0.002,
                        "min": 0.0005, "max": 0.002, "std": 0.0004,
                    },
                    "total_tokens": {
                        "mean": 320.0, "p50": 300.0, "p75": 350.0,
                        "p90": 380.0, "p95": 410.0, "p99": 450.0,
                        "min": 200.0, "max": 500.0, "std": 60.0,
                    },
                    "input_tokens": {
                        "mean": 250.0, "p50": 230.0, "p75": 270.0,
                        "p90": 300.0, "p95": 320.0, "p99": 350.0,
                        "min": 150.0, "max": 400.0, "std": 50.0,
                    },
                    "output_tokens": {
                        "mean": 70.0, "p50": 70.0, "p75": 80.0,
                        "p90": 80.0, "p95": 90.0, "p99": 100.0,
                        "min": 50.0, "max": 100.0, "std": 10.0,
                    },
                    "duration_ms": {
                        "mean": 200.0, "p50": 180.0, "p75": 220.0,
                        "p90": 250.0, "p95": 280.0, "p99": 300.0,
                        "min": 100.0, "max": 350.0, "std": 50.0,
                    },
                    "context_size": {
                        "mean": 250.0, "p50": 230.0, "p75": 270.0,
                        "p90": 300.0, "p95": 320.0, "p99": 350.0,
                        "min": 150.0, "max": 400.0, "std": 50.0,
                    },
                    "iterations_per_run": {
                        "mean": 1.0, "p50": 1.0, "p75": 1.0,
                        "p90": 1.0, "p95": 1.0, "p99": 1.0,
                        "min": 1.0, "max": 1.0, "std": 0.0,
                    },
                    "mean_iterations": 1.0,
                },
            },
            "run_stats": [],
        },
        "patterns": [],
        "cost_summary": _make_cost_summary(),
    }


def _render_to_string(renderables):
    from io import StringIO

    from rich.console import Console as TestConsole
    buf = StringIO()
    c = TestConsole(file=buf, width=120, force_terminal=True)
    for r in renderables:
        c.print(r)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# format_cli_report — new stats format
# ---------------------------------------------------------------------------

class TestFormatReportWithStats:
    def test_returns_non_empty(self):
        session = _make_session(metadata=_make_stats_metadata())
        result = format_cli_report(session)
        assert len(result) > 0

    def test_contains_cost_per_run_section(self):
        session = _make_session(metadata=_make_stats_metadata())
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "Cost Per Run" in output

    def test_contains_step_names(self):
        session = _make_session(metadata=_make_stats_metadata())
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "generate" in output
        assert "classify" in output

    def test_contains_dollar_amounts(self):
        session = _make_session(metadata=_make_stats_metadata())
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "$" in output

    def test_contains_tables_and_panels(self):
        session = _make_session(metadata=_make_stats_metadata())
        result = format_cli_report(session)
        types = {type(r) for r in result}
        assert Table in types
        assert Panel in types


# ---------------------------------------------------------------------------
# format_cli_report — backward compatibility (old cost_summary format)
# ---------------------------------------------------------------------------

class TestFormatReportBackwardCompat:
    def test_old_format_no_crash(self):
        session = _make_session(metadata={"cost_summary": _make_cost_summary()})
        result = format_cli_report(session, cost_summary=_make_cost_summary())
        assert len(result) > 0

    def test_old_format_contains_cost(self):
        session = _make_session(metadata={"cost_summary": _make_cost_summary()})
        result = format_cli_report(session, cost_summary=_make_cost_summary())
        output = _render_to_string(result)
        assert "$" in output

    def test_old_format_step_names(self):
        session = _make_session(metadata={"cost_summary": _make_cost_summary()})
        result = format_cli_report(session, cost_summary=_make_cost_summary())
        output = _render_to_string(result)
        assert "expensive_step" in output


# ---------------------------------------------------------------------------
# Patterns display
# ---------------------------------------------------------------------------

class TestPatternsDisplay:
    def test_no_patterns_message(self):
        meta = _make_stats_metadata()
        meta["patterns"] = []
        session = _make_session(metadata=meta)
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "No non-linear cost patterns detected" in output

    def test_with_patterns(self):
        meta = _make_stats_metadata()
        meta["patterns"] = [
            {
                "pattern_type": "context_growth",
                "step_name": "review",
                "severity": "danger",
                "evidence": {"r_squared": 0.94, "slope": 1200},
                "description": "Context grows by ~1200 tokens per iteration.",
            },
            {
                "pattern_type": "loop_count_variance",
                "step_name": "review",
                "severity": "warning",
                "evidence": {"cv": 0.67},
                "description": "Loop count varies from 2 to 12 iterations.",
            },
        ]
        session = _make_session(metadata=meta)
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "PATTERNS DETECTED" in output
        assert "Context grows by ~1200" in output
        assert "Loop count varies" in output


# ---------------------------------------------------------------------------
# Iteration section
# ---------------------------------------------------------------------------

class TestIterationSection:
    def test_with_iterations(self):
        meta = _make_stats_metadata()
        meta["stats"]["step_stats"]["review"] = {
            "step_name": "review",
            "step_type": "llm",
            "model": "gpt-4o",
            "call_count": 45,
            "runs_present": 10,
            "cost": {
                "mean": 0.0098, "p50": 0.008, "p75": 0.01, "p90": 0.013,
                "p95": 0.0201, "p99": 0.03, "min": 0.004, "max": 0.035, "std": 0.007,
            },
            "total_tokens": {
                "mean": 1560.0, "p50": 1400.0, "p75": 1800.0, "p90": 2000.0,
                "p95": 2100.0, "p99": 2400.0, "min": 800.0, "max": 2800.0, "std": 400.0,
            },
            "input_tokens": {
                "mean": 1200.0, "p50": 1000.0, "p75": 1400.0, "p90": 1600.0,
                "p95": 1700.0, "p99": 1900.0, "min": 600.0, "max": 2200.0, "std": 300.0,
            },
            "output_tokens": {
                "mean": 360.0, "p50": 400.0, "p75": 400.0, "p90": 400.0,
                "p95": 400.0, "p99": 500.0, "min": 200.0, "max": 600.0, "std": 100.0,
            },
            "duration_ms": {
                "mean": 400.0, "p50": 350.0, "p75": 450.0, "p90": 500.0,
                "p95": 600.0, "p99": 800.0, "min": 150.0, "max": 900.0, "std": 150.0,
            },
            "context_size": {
                "mean": 1200.0, "p50": 1000.0, "p75": 1400.0, "p90": 1600.0,
                "p95": 1700.0, "p99": 1900.0, "min": 600.0, "max": 2200.0, "std": 300.0,
            },
            "iterations_per_run": {
                "mean": 4.4, "p50": 4.0, "p75": 5.0, "p90": 7.0,
                "p95": 8.0, "p99": 10.0, "min": 2.0, "max": 12.0, "std": 2.5,
            },
            "mean_iterations": 4.4,
        }
        session = _make_session(metadata=meta)
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "Iteration Counts Per Run" in output
        assert "review" in output

    def test_no_iterations(self):
        meta = _make_stats_metadata()
        session = _make_session(metadata=meta)
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "Iteration Counts Per Run" not in output


# ---------------------------------------------------------------------------
# Projection table with confidence badge
# ---------------------------------------------------------------------------

class TestProjectionTable:
    def test_renders_with_projection_metadata(self):
        meta = _make_stats_metadata()
        meta["projection"] = {
            "method": "montecarlo",
            "traffic_volumes": [100, 1000, 10000],
            "projections": {
                "100": {
                    "daily_volume": 100,
                    "monthly_cost": {
                        "p50": 89.0, "p75": 134.0, "p90": 198.0,
                        "p95": 256.0, "p99": 350.0, "mean": 103.0,
                    },
                    "daily_cost": {
                        "p50": 2.97, "p75": 4.47, "p90": 6.6,
                        "p95": 8.53, "p99": 11.67, "mean": 3.43,
                    },
                    "cost_per_run": {
                        "p50": 0.03, "p75": 0.045, "p90": 0.066,
                        "p95": 0.085, "p99": 0.12, "mean": 0.034,
                    },
                },
                "1000": {
                    "daily_volume": 1000,
                    "monthly_cost": {
                        "p50": 890.0, "p75": 1340.0, "p90": 1980.0,
                        "p95": 2560.0, "p99": 3500.0, "mean": 1030.0,
                    },
                    "daily_cost": {
                        "p50": 29.7, "p75": 44.7, "p90": 66.0,
                        "p95": 85.3, "p99": 116.7, "mean": 34.3,
                    },
                    "cost_per_run": {
                        "p50": 0.03, "p75": 0.045, "p90": 0.066,
                        "p95": 0.085, "p99": 0.12, "mean": 0.034,
                    },
                },
                "10000": {
                    "daily_volume": 10000,
                    "monthly_cost": {
                        "p50": 8900.0, "p75": 13400.0, "p90": 19800.0,
                        "p95": 25600.0, "p99": 35000.0, "mean": 10300.0,
                    },
                    "daily_cost": {
                        "p50": 297.0, "p75": 447.0, "p90": 660.0,
                        "p95": 853.0, "p99": 1167.0, "mean": 343.0,
                    },
                    "cost_per_run": {
                        "p50": 0.03, "p75": 0.045, "p90": 0.066,
                        "p95": 0.085, "p99": 0.12, "mean": 0.034,
                    },
                },
            },
            "confidence": {
                "score": 65,
                "tier": "MODERATE",
                "display_range": "p50 – p95",
                "language": "estimated",
                "deductions": ["Small sample size (20 runs)."],
                "bonuses": [],
            },
            "warnings": [
                "Monte Carlo triggered by: Context grows in 'review'",
            ],
            "patterns_detected": [],
            "montecarlo_result": {
                "n_simulations": 10000,
                "growth_model_delta": 12.5,
                "convergence_check": True,
                "monthly_projection": {},
                "daily_projection": {},
                "per_run_projection": {},
                "linear_monthly": {},
                "log_monthly": {},
            },
        }
        session = _make_session(metadata=meta)
        result = format_cli_report(session)
        output = _render_to_string(result)
        assert "Monte Carlo" in output
        assert "MODERATE" in output
        assert "$" in output


# ---------------------------------------------------------------------------
# format_cost
# ---------------------------------------------------------------------------

class TestFormatCost:
    def test_tiny_amount(self):
        assert format_cost(0.0034) == "$0.0034"

    def test_small_amount(self):
        assert format_cost(0.34) == "$0.34"

    def test_medium_amount(self):
        assert format_cost(34.2) == "$34.20"

    def test_large_amount(self):
        result = format_cost(1234.5)
        assert "$1,234" in result or "$1,235" in result
        assert "." not in result

    def test_zero(self):
        assert format_cost(0) == "$0.00"


# ---------------------------------------------------------------------------
# format_tokens
# ---------------------------------------------------------------------------

class TestFormatTokens:
    def test_small(self):
        assert format_tokens(150) == "150"

    def test_thousands(self):
        assert format_tokens(2340) == "2,340"

    def test_millions(self):
        assert format_tokens(1500000) == "1,500,000"
