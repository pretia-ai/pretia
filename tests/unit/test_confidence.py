"""Tests for the confidence tier system."""

from __future__ import annotations

import json

from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.stats import PercentileStats, StepStats
from agentcost.validation.confidence import compute_confidence


def _make_step_stats(
    name: str = "step_a",
    cost_mean: float = 0.01,
    cost_std: float = 0.002,
) -> StepStats:
    ps = PercentileStats(
        min=cost_mean * 0.5,
        max=cost_mean * 2.0,
        mean=cost_mean,
        std=cost_std,
        p50=cost_mean,
        p75=cost_mean * 1.3,
        p90=cost_mean * 1.6,
        p95=cost_mean * 1.8,
        p99=cost_mean * 2.0,
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
    return StepStats(
        step_name=name,
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


def _make_pattern(
    pattern_type: str = "context_growth",
    step_name: str = "review",
    severity: str = "danger",
) -> DetectedPattern:
    return DetectedPattern(
        pattern_type=pattern_type,
        step_name=step_name,
        severity=severity,
        evidence={},
        description=f"Test pattern: {pattern_type} on {step_name}",
    )


class TestConfidenceHigh:
    def test_high_confidence(self):
        steps = {"step_a": _make_step_stats("step_a", 0.01, 0.002)}
        result = compute_confidence(200, steps, [], input_source="langfuse")
        assert result.tier == "HIGH"
        assert result.score >= 80
        assert result.language == "projected"


class TestConfidenceModerate:
    def test_moderate_confidence(self):
        steps = {"step_a": _make_step_stats("step_a", 0.01, 0.006)}
        result = compute_confidence(25, steps, [])
        assert result.tier == "MODERATE"
        assert 60 <= result.score < 80


class TestConfidenceLow:
    def test_low_confidence(self):
        steps = {"step_a": _make_step_stats("step_a")}
        patterns = [_make_pattern("context_growth", "review", "danger")]
        result = compute_confidence(8, steps, patterns)
        assert result.tier in ("LOW", "VERY_LOW")
        assert result.score < 60


class TestConfidenceVeryLow:
    def test_very_low_confidence(self):
        steps = {
            "s1": _make_step_stats("s1", 0.01, 0.015),
            "s2": _make_step_stats("s2", 0.02, 0.025),
        }
        patterns = [
            _make_pattern("context_growth", "s1", "danger"),
            _make_pattern("loop_count_variance", "s2", "danger"),
        ]
        result = compute_confidence(3, steps, patterns)
        assert result.tier == "VERY_LOW"
        assert result.score < 40


class TestConfidenceDeductionsCapped:
    def test_step_variance_cap(self):
        steps = {f"step_{i}": _make_step_stats(f"step_{i}", 0.01, 0.006) for i in range(10)}
        result = compute_confidence(200, steps, [], input_source="langfuse")
        total_step_deduction = sum(
            8 for s in steps.values() if (s.cost.std / s.cost.mean if s.cost.mean > 0 else 0) > 0.5
        )
        assert total_step_deduction > 30
        assert result.score >= 100 + 10 + 15 - 30


class TestConfidenceLangfuseBonus:
    def test_langfuse_bonus_adds_15(self):
        steps = {"step_a": _make_step_stats("step_a")}
        patterns = [_make_pattern("context_growth", "review", "danger")]
        auto = compute_confidence(50, steps, patterns, input_source="auto-generate")
        langfuse = compute_confidence(50, steps, patterns, input_source="langfuse")
        assert langfuse.score - auto.score == 15


class TestConfidenceToDict:
    def test_round_trip(self):
        steps = {"step_a": _make_step_stats("step_a")}
        result = compute_confidence(100, steps, [])
        d = result.to_dict()
        serialized = json.dumps(d)
        deserialized = json.loads(serialized)
        assert deserialized["tier"] == result.tier
        assert deserialized["score"] == result.score
        assert isinstance(deserialized["deductions"], list)
        assert isinstance(deserialized["bonuses"], list)
