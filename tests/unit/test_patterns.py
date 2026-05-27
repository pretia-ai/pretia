"""Tests for pattern detection: context growth, loop variance, high token variance."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.projection.patterns import DetectedPattern, detect_patterns
from agentcost.projection.stats import compute_stats


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    context_size: int = 100,
    iteration: int = 1,
    duration_ms: int = 500,
) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        step_type="llm",
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
# Context growth
# ---------------------------------------------------------------------------

class TestContextGrowth:
    def test_context_growth_detected(self):
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=2200, iteration=2),
                _make_record("review", context_size=3400, iteration=3),
                _make_record("review", context_size=4600, iteration=4),
                _make_record("review", context_size=5800, iteration=5),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].step_name == "review"
        assert cg[0].evidence["r_squared"] > 0.9
        assert 1100 <= cg[0].evidence["slope"] <= 1300

    def test_context_growth_not_detected_flat(self):
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=1000, iteration=2),
                _make_record("review", context_size=1000, iteration=3),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0

    def test_context_growth_not_detected_few_points(self):
        runs = [
            [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=2000, iteration=2),
            ],
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0

    def test_context_growth_severity_danger(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=2000, iteration=2),
                _make_record("review", context_size=3000, iteration=3),
                _make_record("review", context_size=4000, iteration=4),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].severity == "danger"
        assert cg[0].evidence["r_squared"] > 0.85


# ---------------------------------------------------------------------------
# Loop count variance
# ---------------------------------------------------------------------------

class TestLoopCountVariance:
    def test_loop_count_variance_detected(self):
        runs = []
        for n_iters in [2, 3, 2, 12, 3]:
            run = [
                _make_record("review", iteration=i + 1)
                for i in range(n_iters)
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 1
        assert lcv[0].evidence["max_iterations"] == 12
        assert lcv[0].evidence["cv"] > 0.5

    def test_loop_count_variance_not_detected_stable(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record("review", iteration=1),
                _make_record("review", iteration=2),
                _make_record("review", iteration=3),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 0

    def test_loop_count_variance_skip_single_iteration(self):
        runs = [
            [_make_record("classify", iteration=1)],
            [_make_record("classify", iteration=1)],
            [_make_record("classify", iteration=1)],
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 0

    def test_loop_count_variance_danger_severity(self):
        runs = []
        for n_iters in [2, 2, 2, 2, 20]:
            run = [_make_record("review", iteration=i + 1) for i in range(n_iters)]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 1
        assert lcv[0].severity == "danger"


# ---------------------------------------------------------------------------
# High token variance
# ---------------------------------------------------------------------------

class TestHighTokenVariance:
    def test_high_token_variance_detected(self):
        runs = []
        for i in range(20):
            if i < 18:
                run = [_make_record("generate", input_tokens=300, output_tokens=200)]
            else:
                run = [_make_record("generate", input_tokens=3000, output_tokens=2000)]
            runs.append(run)

        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 1
        assert htv[0].evidence["p95_p50_ratio_tokens"] > 3.0

    def test_high_token_variance_not_detected_uniform(self):
        runs = []
        for i in range(10):
            tok = 490 + i * 3
            run = [_make_record("generate", input_tokens=tok, output_tokens=tok // 2)]
            runs.append(run)

        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 0

    def test_high_token_variance_skips_zero_p50(self):
        runs = [
            [_make_record("tool_step", input_tokens=0, output_tokens=0, context_size=0)],
            [_make_record("tool_step", input_tokens=0, output_tokens=0, context_size=0)],
        ]
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 0


# ---------------------------------------------------------------------------
# Combined / edge cases
# ---------------------------------------------------------------------------

class TestCombined:
    def test_detect_patterns_combined(self):
        context_growth_run = [
            _make_record("grower", context_size=1000, iteration=1),
            _make_record("grower", context_size=3000, iteration=2),
            _make_record("grower", context_size=5000, iteration=3),
            _make_record("grower", context_size=7000, iteration=4),
        ]
        runs = [context_growth_run] * 3

        variable_loop_runs = []
        for n in [2, 3, 2, 12, 3]:
            variable_loop_runs.append(
                [_make_record("looper", iteration=i + 1) for i in range(n)]
            )

        heavy_tail_runs = []
        for i in range(20):
            if i < 17:
                heavy_tail_runs.append(
                    [_make_record("heavy", input_tokens=200, output_tokens=100)]
                )
            else:
                heavy_tail_runs.append(
                    [_make_record("heavy", input_tokens=3000, output_tokens=2000)]
                )

        all_runs = []
        for i in range(max(len(runs), len(variable_loop_runs), len(heavy_tail_runs))):
            combined_run = []
            if i < len(runs):
                combined_run.extend(runs[i])
            if i < len(variable_loop_runs):
                combined_run.extend(variable_loop_runs[i])
            if i < len(heavy_tail_runs):
                combined_run.extend(heavy_tail_runs[i])
            if combined_run:
                all_runs.append(combined_run)

        stats = compute_stats(all_runs, _simple_cost_fn)
        patterns = detect_patterns(all_runs, stats)

        types = {p.pattern_type for p in patterns}
        assert "context_growth" in types
        assert "loop_count_variance" in types
        assert "high_token_variance" in types

        severities = [p.severity for p in patterns]
        danger_idx = [i for i, s in enumerate(severities) if s == "danger"]
        warning_idx = [i for i, s in enumerate(severities) if s == "warning"]
        if danger_idx and warning_idx:
            assert max(danger_idx) < min(warning_idx)

    def test_detect_patterns_empty_runs(self):
        patterns = detect_patterns([])
        assert patterns == []

    def test_detect_patterns_single_step_no_patterns(self):
        runs = [[_make_record("simple")]] * 5
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        assert patterns == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

class TestPatternSerialization:
    def test_detected_pattern_to_dict(self):
        p = DetectedPattern(
            pattern_type="context_growth",
            step_name="review",
            severity="warning",
            evidence={"r_squared": 0.82, "slope": 1200.0},
            description="Context grows.",
        )
        d = p.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["pattern_type"] == "context_growth"
        assert d["step_name"] == "review"
