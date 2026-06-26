"""Tests for DeepSeek cache token extraction, differential pricing, and cache-busting."""

from __future__ import annotations

import types
from datetime import UTC, datetime

import pytest

from pretia.collectors.base import StepRecord
from pretia.collectors.cache_bust import cache_bust_prompt, needs_cache_busting
from pretia.collectors.generic import GenericCollector, _try_extract
from pretia.pricing.tables import calculate_cost


def _make_record(**kwargs: object) -> StepRecord:
    defaults = {
        "step_name": "test",
        "step_type": "llm",
        "model": "deepseek-v4-flash",
        "input_tokens": 2000,
        "output_tokens": 500,
        "context_size": 2000,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc",
        "system_prompt_tokens": 50,
        "output_format": "text",
        "is_retry": False,
        "iteration": 1,
        "parent_step": None,
        "duration_ms": 100,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


# ---------------------------------------------------------------------------
# Change 1: Cache token extraction
# ---------------------------------------------------------------------------


class TestDeepseekCacheTokensExtracted:
    def test_cache_fields_extracted(self):
        collector = GenericCollector()
        tracker = collector.step("test")
        tracker._iteration = 1
        tracker._start_ns = 0
        mock = types.SimpleNamespace(
            model="deepseek-chat",
            usage=types.SimpleNamespace(
                prompt_tokens=2000,
                completion_tokens=500,
                prompt_cache_hit_tokens=1500,
                prompt_cache_miss_tokens=500,
            ),
        )
        _try_extract(tracker, mock)
        assert tracker._recorded is not None
        assert tracker._recorded["cache_hit_tokens"] == 1500
        assert tracker._recorded["cache_miss_tokens"] == 500


class TestNonDeepseekCacheTokensNone:
    def test_no_cache_fields(self):
        collector = GenericCollector()
        tracker = collector.step("test")
        tracker._iteration = 1
        tracker._start_ns = 0
        mock = types.SimpleNamespace(
            model="gpt-4o",
            usage=types.SimpleNamespace(
                prompt_tokens=1500,
                completion_tokens=350,
            ),
        )
        _try_extract(tracker, mock)
        assert tracker._recorded is not None
        assert tracker._recorded["cache_hit_tokens"] is None
        assert tracker._recorded["cache_miss_tokens"] is None


class TestStepRecordBackwardCompatible:
    def test_without_cache_fields(self):
        rec = _make_record()
        assert rec.cache_hit_tokens is None
        assert rec.cache_miss_tokens is None

    def test_with_cache_fields(self):
        rec = _make_record(cache_hit_tokens=1500, cache_miss_tokens=500)
        assert rec.cache_hit_tokens == 1500
        assert rec.cache_miss_tokens == 500


# ---------------------------------------------------------------------------
# Change 2: Differential pricing
# ---------------------------------------------------------------------------


class TestDeepseekCacheDifferentialPricing:
    def test_differential_pricing(self):
        cost = calculate_cost(
            "deepseek-v4-flash",
            input_tokens=2000,
            output_tokens=500,
            cache_hit_tokens=1500,
            cache_miss_tokens=500,
        )
        input_cost = (500 / 1_000_000 * 0.14) + (1500 / 1_000_000 * 0.0028)
        output_cost = 500 / 1_000_000 * 0.28
        expected = input_cost + output_cost
        assert cost == pytest.approx(expected, rel=0.01)


class TestDeepseekNoCacheUsesStandardPrice:
    def test_standard_price(self):
        cost = calculate_cost("deepseek-v4-flash", input_tokens=2000, output_tokens=500)
        expected = (2000 / 1_000_000 * 0.14) + (500 / 1_000_000 * 0.28)
        assert cost == pytest.approx(expected, rel=0.01)


class TestNoCachePricingIgnoresCacheTokens:
    def test_ignores_cache(self):
        cost_with = calculate_cost(
            "gpt-4o",
            input_tokens=2000,
            output_tokens=500,
            cache_hit_tokens=1500,
            cache_miss_tokens=500,
        )
        cost_without = calculate_cost("gpt-4o", input_tokens=2000, output_tokens=500)
        assert cost_with == cost_without


class TestCachePricingV4Pro:
    def test_v4_pro_cache(self):
        cost = calculate_cost(
            "deepseek-v4-pro",
            input_tokens=2000,
            output_tokens=500,
            cache_hit_tokens=1500,
            cache_miss_tokens=500,
        )
        input_cost = (500 / 1_000_000 * 0.435) + (1500 / 1_000_000 * 0.012)
        output_cost = 500 / 1_000_000 * 0.87
        expected = input_cost + output_cost
        assert cost == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Change 3: Cache-busting
# ---------------------------------------------------------------------------


class TestCacheBustPromptUnique:
    def test_unique(self):
        r1 = cache_bust_prompt("Hello", "run-1")
        r2 = cache_bust_prompt("Hello", "run-2")
        assert r1 != r2
        assert "Hello" in r1
        assert "Hello" in r2
        assert len(r1) > len("Hello")
        assert len(r2) > len("Hello")


class TestCacheBustPromptSmallOverhead:
    def test_small_overhead(self):
        result = cache_bust_prompt("Hello", "run-1")
        suffix = result[len("Hello") :]
        assert len(suffix) < 100


class TestNeedsCacheBusting:
    def test_deepseek_models(self):
        assert needs_cache_busting("deepseek-v4-flash") is True
        assert needs_cache_busting("DeepSeek-V4-Pro") is True

    def test_anthropic_models(self):
        assert needs_cache_busting("claude-sonnet-4-6") is True
        assert needs_cache_busting("claude-haiku-4-5") is True
        assert needs_cache_busting("claude-opus-4-7") is True

    def test_non_cached_models(self):
        assert needs_cache_busting("gpt-4o") is False
        assert needs_cache_busting("gemini-2.5-flash") is False


class TestCacheModeColdDefault:
    def test_default_cold(self):
        from pretia.runner import ProfileRunner

        runner = ProfileRunner(workflow_path="dummy.py")
        assert runner.cache_mode == "cold"
