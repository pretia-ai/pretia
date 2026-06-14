"""Tests for agentcost.recommend.architecture — architecture generators."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.pricing.tables import MODEL_CACHE_HIT_PRICING, get_model_pricing
from agentcost.recommend.architecture import (
    CacheContextGenerator,
    PromptCachingGenerator,
    ToolFilterGenerator,
)
from agentcost.recommend.base import _DEFAULT_DAILY_VOLUME
from agentcost.store import ProfilingSession

_PER_MILLION = 1_000_000


def _make_record(
    step_name: str = "classify",
    model: str = "claude-sonnet-4-6",
    input_tokens: int = 1000,
    output_tokens: int = 200,
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": 1200,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 200,
        "output_format": "text",
        "is_retry": False,
        "iteration": 1,
        "parent_step": None,
        "duration_ms": 500,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


def _make_session(
    runs: list[list[StepRecord]],
    patterns: list[dict] | None = None,
) -> ProfilingSession:
    return ProfilingSession(
        workflow_name="test_workflow",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=len(runs),
        input_mode="auto",
        runs=runs,
        metadata={"patterns": patterns or []},
    )


# ──────────────────────────────────────────────────────────────────────────────
# PromptCachingGenerator
# ──────────────────────────────────────────────────────────────────────────────


def _cache_pattern(
    step_name: str = "research_loop",
    model: str = "claude-sonnet-4-6",
    cache_hit_ratio: float = 0.02,
    total_cache_miss_tokens: int = 50_000,
) -> dict:
    return {
        "pattern_type": "cache_utilization_opportunity",
        "step_name": step_name,
        "severity": "warning",
        "description": f"Low cache utilization in {step_name}",
        "evidence": {
            "cache_hit_ratio": cache_hit_ratio,
            "total_cache_miss_tokens": total_cache_miss_tokens,
            "model": model,
        },
    }


class TestPromptCachingGenerator:
    def test_fires_with_cache_pattern(self) -> None:
        pattern = _cache_pattern(model="claude-sonnet-4-6", total_cache_miss_tokens=50_000)
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[pattern])

        gen = PromptCachingGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.type == "architecture"
        assert rec.id == "prompt-caching-research_loop"
        assert rec.confidence == "HIGH"

    def test_savings_calculation(self) -> None:
        miss_tokens = 50_000
        model = "claude-sonnet-4-6"
        pattern = _cache_pattern(model=model, total_cache_miss_tokens=miss_tokens)
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[pattern])

        gen = PromptCachingGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1

        standard_rate = get_model_pricing(model)[0]
        cache_rate = MODEL_CACHE_HIT_PRICING[model] / _PER_MILLION
        expected = round(
            miss_tokens * (standard_rate - cache_rate) * _DEFAULT_DAILY_VOLUME * 30, 2
        )
        assert recs[0].monthly_savings == pytest.approx(expected, rel=1e-4)

    def test_no_recommendation_without_pattern(self) -> None:
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[])
        gen = PromptCachingGenerator()
        assert gen.generate(session) == []

    def test_no_recommendation_below_50_threshold(self) -> None:
        pattern = _cache_pattern(total_cache_miss_tokens=1)
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[pattern])
        gen = PromptCachingGenerator()
        assert gen.generate(session) == []

    def test_no_recommendation_unsupported_model(self) -> None:
        pattern = _cache_pattern(model="gpt-4o")
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[pattern])
        gen = PromptCachingGenerator()
        assert gen.generate(session) == []

    def test_no_recommendation_unknown_model(self) -> None:
        pattern = _cache_pattern(model="totally-unknown-xyz")
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[pattern])
        gen = PromptCachingGenerator()
        assert gen.generate(session) == []

    def test_evidence_fields(self) -> None:
        pattern = _cache_pattern()
        runs = [[_make_record()]]
        session = _make_session(runs, patterns=[pattern])
        gen = PromptCachingGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        ev = recs[0].evidence
        assert "model" in ev
        assert "cache_hit_ratio" in ev
        assert "cache_miss_tokens" in ev
        assert "pct_reduction" in ev
        assert ev["daily_volume"] == _DEFAULT_DAILY_VOLUME


# ──────────────────────────────────────────────────────────────────────────────
# ToolFilterGenerator
# ──────────────────────────────────────────────────────────────────────────────


def _tool_step_runs(
    step_name: str = "agent_step",
    model: str = "gpt-4o",
    input_tokens: int = 3000,
    tool_definitions_tokens: int = 2000,
    tool_names: list[str | None] | None = None,
    n_runs: int = 5,
) -> list[list[StepRecord]]:
    if tool_names is None:
        tool_names = ["search", None, "search"]
    runs: list[list[StepRecord]] = []
    for i in range(n_runs):
        tn = tool_names[i % len(tool_names)]
        runs.append([
            _make_record(
                step_name=step_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=100,
                tool_definitions_tokens=tool_definitions_tokens,
                tool_name=tn,
            )
        ])
    return runs


class TestToolFilterGenerator:
    def test_fires_when_share_above_30pct(self) -> None:
        runs = _tool_step_runs(
            input_tokens=3000,
            tool_definitions_tokens=2000,
            tool_names=["search", "search", "search"],
        )
        session = _make_session(runs)
        gen = ToolFilterGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.type == "architecture"
        assert rec.id == "tool-filter-agent_step"
        assert rec.confidence == "MODERATE"

    def test_no_recommendation_share_below_30pct(self) -> None:
        runs = _tool_step_runs(
            input_tokens=3000,
            tool_definitions_tokens=500,
            tool_names=["search"],
        )
        session = _make_session(runs)
        gen = ToolFilterGenerator()
        assert gen.generate(session) == []

    def test_no_recommendation_without_tool_name_data(self) -> None:
        runs = _tool_step_runs(
            input_tokens=3000,
            tool_definitions_tokens=2000,
            tool_names=[None, None, None],
        )
        session = _make_session(runs)
        gen = ToolFilterGenerator()
        assert gen.generate(session) == []

    def test_identifies_used_tools(self) -> None:
        runs = _tool_step_runs(
            input_tokens=3000,
            tool_definitions_tokens=2000,
            tool_names=["search", "calculator", "search"],
        )
        session = _make_session(runs)
        gen = ToolFilterGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert set(recs[0].evidence["used_tools"]) == {"calculator", "search"}

    def test_empty_runs(self) -> None:
        session = _make_session([])
        gen = ToolFilterGenerator()
        assert gen.generate(session) == []

    def test_evidence_fields(self) -> None:
        runs = _tool_step_runs(
            input_tokens=3000,
            tool_definitions_tokens=2000,
            tool_names=["search"],
        )
        session = _make_session(runs)
        gen = ToolFilterGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        ev = recs[0].evidence
        assert "tool_definition_share" in ev
        assert "used_tools" in ev
        assert "n_used_tools" in ev
        assert "savings_tokens" in ev
        assert ev["daily_volume"] == _DEFAULT_DAILY_VOLUME


# ──────────────────────────────────────────────────────────────────────────────
# CacheContextGenerator
# ──────────────────────────────────────────────────────────────────────────────


class TestCacheContextGenerator:
    def test_fires_with_consecutive_same_hash(self) -> None:
        runs = [
            [
                _make_record(
                    step_name="step_a",
                    model="gpt-4o",
                    system_prompt_hash="same_hash",
                    system_prompt_tokens=500,
                    input_tokens=1000,
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="step_b",
                    model="gpt-4o",
                    system_prompt_hash="same_hash",
                    system_prompt_tokens=500,
                    input_tokens=1000,
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.type == "architecture"
        assert rec.confidence == "HIGH"
        assert "step_a" in rec.affected_steps
        assert "step_b" in rec.affected_steps

    def test_savings_calculation(self) -> None:
        prompt_tokens = 500
        model = "gpt-4o"
        runs = [
            [
                _make_record(
                    step_name="step_a",
                    model=model,
                    system_prompt_hash="same_hash",
                    system_prompt_tokens=prompt_tokens,
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="step_b",
                    model=model,
                    system_prompt_hash="same_hash",
                    system_prompt_tokens=prompt_tokens,
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1

        input_price = get_model_pricing(model)[0]
        redundant_cost = prompt_tokens * input_price
        expected = round(redundant_cost * _DEFAULT_DAILY_VOLUME * 30, 2)
        assert recs[0].monthly_savings == pytest.approx(expected, rel=1e-4)

    def test_no_recommendation_different_hashes(self) -> None:
        runs = [
            [
                _make_record(
                    step_name="step_a",
                    system_prompt_hash="hash_a",
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="step_b",
                    system_prompt_hash="hash_b",
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        assert gen.generate(session) == []

    def test_no_recommendation_same_step_name(self) -> None:
        """Same step repeating (loop) should not trigger this."""
        runs = [
            [
                _make_record(
                    step_name="loop_step",
                    system_prompt_hash="same",
                    iteration=1,
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="loop_step",
                    system_prompt_hash="same",
                    iteration=2,
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        assert gen.generate(session) == []

    def test_empty_runs(self) -> None:
        session = _make_session([])
        gen = CacheContextGenerator()
        assert gen.generate(session) == []

    def test_evidence_fields(self) -> None:
        runs = [
            [
                _make_record(
                    step_name="step_a",
                    model="gpt-4o",
                    system_prompt_hash="same",
                    system_prompt_tokens=500,
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="step_b",
                    model="gpt-4o",
                    system_prompt_hash="same",
                    system_prompt_tokens=500,
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        ev = recs[0].evidence
        assert "system_prompt_tokens" in ev
        assert "occurrences" in ev
        assert ev["occurrences"] == 5
        assert "daily_volume" in ev

    def test_id_uses_sorted_step_names(self) -> None:
        runs = [
            [
                _make_record(
                    step_name="z_step",
                    model="gpt-4o",
                    system_prompt_hash="same",
                    system_prompt_tokens=500,
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="a_step",
                    model="gpt-4o",
                    system_prompt_hash="same",
                    system_prompt_tokens=500,
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].id == "cache-context-a_step-z_step"
