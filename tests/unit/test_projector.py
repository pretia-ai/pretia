"""Tests for the unified projection entry point."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.projector import project
from agentcost.projection.stats import (
    PercentileStats,
    ProfilingStats,
    StepStats,
    compute_stats,
)
from agentcost.validation.confidence import ConfidenceResult


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


def _make_stats(mean_cost: float = 0.03, p95_cost: float = 0.05) -> ProfilingStats:
    ps = PercentileStats(
        min=mean_cost * 0.5,
        max=p95_cost * 1.5,
        mean=mean_cost,
        std=mean_cost * 0.3,
        p50=mean_cost,
        p75=mean_cost * 1.3,
        p90=p95_cost * 0.9,
        p95=p95_cost,
        p99=p95_cost * 1.3,
    )
    tok_ps = PercentileStats(
        min=100,
        max=500,
        mean=300,
        std=80,
        p50=280,
        p75=350,
        p90=400,
        p95=450,
        p99=490,
    )
    iter_ps = PercentileStats(
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
    step = StepStats(
        step_name="classify",
        step_type="llm",
        model="gpt-4o-mini",
        call_count=20,
        runs_present=20,
        input_tokens=tok_ps,
        output_tokens=tok_ps,
        total_tokens=tok_ps,
        cost=ps,
        duration_ms=tok_ps,
        context_size=tok_ps,
        iterations_per_run=iter_ps,
        mean_iterations=1.0,
    )
    return ProfilingStats(
        step_stats={"classify": step},
        run_stats=[],
        cost_per_run=ps,
        tokens_per_run=tok_ps,
        total_runs=20,
        total_steps=20,
    )


def _make_danger_pattern(step_name: str = "review") -> DetectedPattern:
    return DetectedPattern(
        pattern_type="context_growth",
        step_name=step_name,
        severity="danger",
        evidence={"r_squared": 0.92, "slope": 800},
        description=f"Context grows in '{step_name}'",
    )


def _make_warning_pattern(step_name: str = "review") -> DetectedPattern:
    return DetectedPattern(
        pattern_type="loop_count_variance",
        step_name=step_name,
        severity="warning",
        evidence={"cv": 0.55},
        description=f"Loop count varies in '{step_name}'",
    )


class TestLinearProjectionBasic:
    def test_mean_monthly(self):
        stats = _make_stats(mean_cost=0.03, p95_cost=0.05)
        result = project(stats, [], traffic=[1000])
        assert result.method == "linear"
        monthly_mean = result.projections[1000].monthly_cost.mean
        assert monthly_mean == pytest.approx(0.03 * 1000 * 30, rel=0.01)


class TestLinearProjectionMultipleVolumes:
    def test_three_volumes(self):
        stats = _make_stats(mean_cost=0.03)
        result = project(stats, [], traffic=[100, 1000, 10000])
        assert len(result.projections) == 3
        m100 = result.projections[100].monthly_cost.mean
        m1000 = result.projections[1000].monthly_cost.mean
        m10000 = result.projections[10000].monthly_cost.mean
        assert m1000 == pytest.approx(m100 * 10, rel=0.01)
        assert m10000 == pytest.approx(m100 * 100, rel=0.01)


class TestMonteCarloTriggered:
    def test_danger_triggers_montecarlo(self):
        runs = [
            [
                _make_record(
                    "review",
                    "gpt-4o",
                    1000 + i * 800,
                    200,
                    iteration=i + 1,
                    context_size=1000 + i * 800,
                )
                for i in range(5)
            ]
            for _ in range(20)
        ]
        stats = compute_stats(runs)
        patterns = [_make_danger_pattern("review")]
        result = project(stats, patterns, traffic=[1000], runs=runs)
        assert result.method == "montecarlo"
        assert result.projections[1000].monthly_cost.mean > 0


class TestWarningOnlyUsesLinear:
    def test_warning_stays_linear(self):
        stats = _make_stats()
        result = project(stats, [_make_warning_pattern()])
        assert result.method == "linear"


class TestMonteCarloFallbackWithoutRuns:
    def test_falls_back_to_linear(self):
        stats = _make_stats()
        patterns = [_make_danger_pattern()]
        result = project(stats, patterns, runs=None)
        assert result.method == "linear"
        assert any("Falling back" in w for w in result.warnings)


class TestProjectionIncludesConfidence:
    def test_has_confidence(self):
        stats = _make_stats()
        result = project(stats, [])
        assert isinstance(result.confidence, ConfidenceResult)
        assert result.confidence.tier in ("HIGH", "MODERATE", "LOW", "VERY_LOW")


class TestProjectionToDict:
    def test_serializes(self):
        stats = _make_stats()
        result = project(stats, [], traffic=[100])
        d = result.to_dict()
        serialized = json.dumps(d)
        assert len(serialized) > 0
        deserialized = json.loads(serialized)
        assert deserialized["method"] == "linear"
        assert "100" in deserialized["projections"] or 100 in deserialized["projections"]
