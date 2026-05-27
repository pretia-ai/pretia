"""Tests for the stats module: percentile computation and profiling statistics."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.projection.stats import (
    ProfilingStats,
    compute_percentile_stats,
    compute_stats,
)


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    context_size: int = 100,
    iteration: int = 1,
    duration_ms: int = 500,
    step_type: str = "llm",
) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        step_type=step_type,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=context_size,
        tool_definitions_tokens=0,
        system_prompt_hash="abc123",
        system_prompt_tokens=50,
        output_format="text",
        is_retry=False,
        iteration=iteration,
        parent_step=None,
        duration_ms=duration_ms,
        timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    )


def _simple_cost_fn(model: str, inp: int, out: int) -> float:
    return (inp + out) * 0.001


# ---------------------------------------------------------------------------
# PercentileStats
# ---------------------------------------------------------------------------

class TestPercentileStats:
    def test_percentile_stats_basic(self):
        ps = compute_percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0])
        assert ps.mean == 3.0
        assert ps.min == 1.0
        assert ps.max == 5.0
        assert ps.p50 == 3.0
        assert 4.5 <= ps.p95 <= 5.0
        assert ps.std > 0

    def test_percentile_stats_single_value(self):
        ps = compute_percentile_stats([42.0])
        assert ps.min == 42.0
        assert ps.max == 42.0
        assert ps.mean == 42.0
        assert ps.std == 0.0
        assert ps.p50 == 42.0
        assert ps.p75 == 42.0
        assert ps.p90 == 42.0
        assert ps.p95 == 42.0
        assert ps.p99 == 42.0

    def test_percentile_stats_empty(self):
        with pytest.raises(ValueError, match="Cannot compute stats on empty data"):
            compute_percentile_stats([])

    def test_percentile_stats_two_values(self):
        ps = compute_percentile_stats([10.0, 20.0])
        assert ps.min == 10.0
        assert ps.max == 20.0
        assert ps.mean == 15.0
        assert ps.p50 == 15.0

    def test_percentile_stats_to_dict(self):
        ps = compute_percentile_stats([1.0, 2.0, 3.0])
        d = ps.to_dict()
        assert set(d.keys()) == {"min", "max", "mean", "std", "p50", "p75", "p90", "p95", "p99"}
        json.dumps(d)


# ---------------------------------------------------------------------------
# compute_stats basic
# ---------------------------------------------------------------------------

class TestComputeStats:
    def test_compute_stats_basic(self):
        runs = [
            [
                _make_record("classify", "gpt-4o-mini", 100, 50),
                _make_record("generate", "gpt-4o", 200, 100),
            ],
            [
                _make_record("classify", "gpt-4o-mini", 120, 60),
                _make_record("generate", "gpt-4o", 250, 130),
            ],
            [
                _make_record("classify", "gpt-4o-mini", 80, 40),
                _make_record("generate", "gpt-4o", 180, 90),
            ],
        ]
        stats = compute_stats(runs, cost_fn=_simple_cost_fn)

        assert stats.total_runs == 3
        assert "classify" in stats.step_stats
        assert "generate" in stats.step_stats
        assert len(stats.step_stats) == 2
        assert stats.step_stats["classify"].call_count == 3
        assert stats.step_stats["classify"].model == "gpt-4o-mini"
        assert stats.step_stats["generate"].model == "gpt-4o"
        assert stats.cost_per_run is not None
        assert stats.cost_per_run.mean > 0
        assert len(stats.run_stats) == 3

    def test_compute_stats_empty_runs(self):
        stats = compute_stats([])
        assert stats.total_runs == 0
        assert stats.step_stats == {}
        assert stats.run_stats == []

    def test_compute_stats_single_run(self):
        runs = [[_make_record("step_a", "gpt-4o", 100, 50)]]
        stats = compute_stats(runs, cost_fn=_simple_cost_fn)
        assert stats.total_runs == 1
        assert stats.step_stats["step_a"].call_count == 1
        assert stats.cost_per_run is not None

    def test_compute_stats_with_iterations(self):
        run1 = [
            _make_record("review", iteration=1),
            _make_record("review", iteration=2),
            _make_record("review", iteration=3),
        ]
        run2 = [
            _make_record("review", iteration=1),
            _make_record("review", iteration=2),
            _make_record("review", iteration=3),
            _make_record("review", iteration=4),
            _make_record("review", iteration=5),
        ]
        stats = compute_stats([run1, run2], cost_fn=_simple_cost_fn)

        review = stats.step_stats["review"]
        assert review.call_count == 8
        assert review.iterations_per_run.mean == pytest.approx(4.0)
        assert review.iterations_per_run.min == 3.0
        assert review.iterations_per_run.max == 5.0
        assert review.mean_iterations == pytest.approx(4.0)

    def test_compute_stats_step_missing_in_some_runs(self):
        runs = [
            [_make_record("classify"), _make_record("generate")],
            [
                _make_record("classify"),
                _make_record("generate"),
                _make_record("fallback"),
            ],
            [_make_record("classify"), _make_record("generate")],
        ]
        stats = compute_stats(runs, cost_fn=_simple_cost_fn)

        assert stats.step_stats["fallback"].runs_present == 1
        assert stats.step_stats["fallback"].call_count == 1
        assert stats.step_stats["classify"].runs_present == 3

    def test_compute_stats_unknown_model(self):
        runs = [[_make_record("step_x", model="nonexistent-model-v99")]]
        stats = compute_stats(runs, cost_fn=None)
        assert stats.step_stats["step_x"].cost.mean == 0.0

    def test_run_stats_totals(self):
        runs = [
            [
                _make_record("a", input_tokens=100, output_tokens=50, duration_ms=200),
                _make_record("b", input_tokens=200, output_tokens=100, duration_ms=300),
                _make_record("c", input_tokens=50, output_tokens=25, duration_ms=100),
            ],
        ]
        stats = compute_stats(runs, cost_fn=_simple_cost_fn)
        rs = stats.run_stats[0]

        assert rs.total_input_tokens == 350
        assert rs.total_output_tokens == 175
        assert rs.total_tokens == 525
        assert rs.duration_ms == 600
        assert rs.step_count == 3
        assert rs.total_cost == pytest.approx(0.525)

    def test_total_steps_count(self):
        runs = [
            [_make_record("a"), _make_record("b")],
            [_make_record("a"), _make_record("b"), _make_record("c")],
        ]
        stats = compute_stats(runs, cost_fn=_simple_cost_fn)
        assert stats.total_steps == 5


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    def test_profiling_stats_to_dict(self):
        runs = [
            [_make_record("classify", "gpt-4o-mini", 100, 50)],
            [_make_record("classify", "gpt-4o-mini", 120, 60)],
        ]
        stats = compute_stats(runs, cost_fn=_simple_cost_fn)
        d = stats.to_dict()

        assert "step_stats" in d
        assert "run_stats" in d
        assert "cost_per_run" in d
        assert "total_runs" in d

        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_empty_stats_to_dict(self):
        stats = compute_stats([])
        d = stats.to_dict()
        json.dumps(d)
        assert d["total_runs"] == 0
