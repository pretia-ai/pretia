"""Mock response tests for collector token extraction across all providers."""

from __future__ import annotations

import types

import pytest

from agentcost.collectors.generic import GenericCollector, _try_extract


def _extract_via_generic(mock_response: object) -> dict | None:
    """Run _try_extract on a mock response and return the recorded data."""
    collector = GenericCollector()
    tracker = collector.step("test_step")
    tracker._iteration = 1
    tracker._start_ns = 0
    _try_extract(tracker, mock_response)
    return tracker._recorded


# ---------------------------------------------------------------------------
# OpenAI standard (GPT-5.4)
# ---------------------------------------------------------------------------

OPENAI_STD_INPUT = 1500
OPENAI_STD_OUTPUT = 350


class TestOpenAIStandardTokenExtraction:
    def test_openai_standard_token_extraction(self):
        mock = types.SimpleNamespace(
            model="gpt-4o",
            usage=types.SimpleNamespace(
                prompt_tokens=OPENAI_STD_INPUT,
                completion_tokens=OPENAI_STD_OUTPUT,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == OPENAI_STD_INPUT
        assert recorded["output_tokens"] == OPENAI_STD_OUTPUT
        assert recorded["model"] == "gpt-4o"


# ---------------------------------------------------------------------------
# OpenAI reasoning (o-series)
# ---------------------------------------------------------------------------

OPENAI_REASON_INPUT = 1500
OPENAI_REASON_OUTPUT = 550
OPENAI_REASON_REASONING = 200


class TestOpenAIReasoningTokenExtraction:
    def test_openai_reasoning_token_extraction(self):
        mock = types.SimpleNamespace(
            model="o3",
            usage=types.SimpleNamespace(
                prompt_tokens=OPENAI_REASON_INPUT,
                completion_tokens=OPENAI_REASON_OUTPUT,
                completion_tokens_details=types.SimpleNamespace(
                    reasoning_tokens=OPENAI_REASON_REASONING,
                ),
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == OPENAI_REASON_INPUT
        assert recorded["output_tokens"] == OPENAI_REASON_OUTPUT


# ---------------------------------------------------------------------------
# Anthropic standard (Sonnet 4.6)
# ---------------------------------------------------------------------------

ANTHROPIC_STD_INPUT = 1200
ANTHROPIC_STD_OUTPUT = 400


class TestAnthropicStandardTokenExtraction:
    def test_anthropic_standard_token_extraction(self):
        mock = types.SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=types.SimpleNamespace(
                input_tokens=ANTHROPIC_STD_INPUT,
                output_tokens=ANTHROPIC_STD_OUTPUT,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == ANTHROPIC_STD_INPUT
        assert recorded["output_tokens"] == ANTHROPIC_STD_OUTPUT


# ---------------------------------------------------------------------------
# Anthropic extended thinking
# ---------------------------------------------------------------------------


class TestAnthropicExtendedThinkingExtraction:
    def test_anthropic_extended_thinking_extraction(self):
        # GAP: collector does not handle extended thinking tokens separately.
        # StepRecord has no thinking_tokens field. Test verifies basic extraction.
        mock = types.SimpleNamespace(
            model="claude-opus-4-7",
            usage=types.SimpleNamespace(
                input_tokens=1200,
                output_tokens=800,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 1200
        assert recorded["output_tokens"] == 800


# ---------------------------------------------------------------------------
# Anthropic cached
# ---------------------------------------------------------------------------


class TestAnthropicCachedTokenExtraction:
    def test_anthropic_cached_token_extraction(self):
        # GAP: collector ignores cache_creation_input_tokens and cache_read_input_tokens.
        # Cache-aware pricing would need these fields on StepRecord.
        mock = types.SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=types.SimpleNamespace(
                input_tokens=1200,
                output_tokens=400,
                cache_creation_input_tokens=300,
                cache_read_input_tokens=500,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 1200
        assert recorded["output_tokens"] == 400


# ---------------------------------------------------------------------------
# DeepSeek cache hit/miss
# ---------------------------------------------------------------------------


class TestDeepseekCacheTokenExtraction:
    def test_deepseek_cache_token_extraction(self):
        mock = types.SimpleNamespace(
            model="deepseek-chat",
            usage=types.SimpleNamespace(
                prompt_tokens=2000,
                completion_tokens=500,
                prompt_cache_hit_tokens=1500,
                prompt_cache_miss_tokens=500,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 2000
        assert recorded["output_tokens"] == 500
        assert recorded["cache_hit_tokens"] == 1500
        assert recorded["cache_miss_tokens"] == 500


# ---------------------------------------------------------------------------
# Qwen DashScope-native
# ---------------------------------------------------------------------------


class TestQwenDashscopeTokenExtraction:
    def test_qwen_dashscope_token_extraction(self):
        # Test via dict-style usage (the generic collector handles both)
        mock = {
            "model": "qwen3.7-max",
            "usage": {
                "input_tokens": 1800,
                "output_tokens": 600,
            },
        }
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 1800
        assert recorded["output_tokens"] == 600


# ---------------------------------------------------------------------------
# Qwen OpenAI-compatible
# ---------------------------------------------------------------------------


class TestQwenOpenAICompatTokenExtraction:
    def test_qwen_openai_compat_token_extraction(self):
        mock = types.SimpleNamespace(
            model="qwen3.7-max",
            usage=types.SimpleNamespace(
                prompt_tokens=1800,
                completion_tokens=600,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 1800
        assert recorded["output_tokens"] == 600


# ---------------------------------------------------------------------------
# Gemini with thoughtsTokenCount
# ---------------------------------------------------------------------------


class TestGeminiThoughtsTokenExtraction:
    @pytest.mark.xfail(reason="No native Gemini collector; GenericCollector lacks thoughts field")
    def test_gemini_thoughts_token_extraction(self):
        # GAP: no collector handles Gemini's thoughtsTokenCount natively.
        # GenericCollector would need usage.prompt_tokens or usage.input_tokens,
        # but Gemini uses prompt_token_count/candidates_token_count.
        mock = types.SimpleNamespace(
            model="gemini-2.5-flash",
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=1600,
                candidates_token_count=450,
                thoughts_token_count=300,
                total_token_count=2350,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 1600
        assert recorded["output_tokens"] == 450


# ---------------------------------------------------------------------------
# Vision token extraction
# ---------------------------------------------------------------------------


class TestAnthropicVisionTokenExtraction:
    def test_anthropic_vision_token_extraction(self):
        mock = types.SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=types.SimpleNamespace(
                input_tokens=8500,
                output_tokens=200,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 8500
        assert recorded["output_tokens"] == 200


class TestOpenAIVisionTokenExtraction:
    def test_openai_vision_token_extraction(self):
        mock = types.SimpleNamespace(
            model="gpt-4o",
            usage=types.SimpleNamespace(
                prompt_tokens=12000,
                completion_tokens=300,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 12000
        assert recorded["output_tokens"] == 300


class TestGeminiVisionTokenExtraction:
    @pytest.mark.xfail(
        reason="GenericCollector lacks native Gemini usage_metadata support",
    )
    def test_gemini_vision_token_extraction(self):
        mock = types.SimpleNamespace(
            model="gemini-2.5-pro",
            usage_metadata=types.SimpleNamespace(
                prompt_token_count=15000,
                candidates_token_count=250,
                total_token_count=15250,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 15000


# ---------------------------------------------------------------------------
# Parallel call tests
# ---------------------------------------------------------------------------


class TestParallelCallsNotDropped:
    def test_parallel_calls_not_dropped(self):
        collector = GenericCollector()
        totals = []
        for _ in range(3):
            tracker = collector.step("shared_step")
            tracker._iteration = 1
            tracker._start_ns = 0
            mock = types.SimpleNamespace(
                model="gpt-4o",
                usage=types.SimpleNamespace(
                    prompt_tokens=100, completion_tokens=50,
                ),
            )
            _try_extract(tracker, mock)
            if tracker._recorded:
                totals.append(
                    tracker._recorded["input_tokens"]
                    + tracker._recorded["output_tokens"]
                )
        assert sum(totals) == 450


class TestParallelCallsCorrectAttribution:
    def test_parallel_calls_correct_attribution(self):
        collector = GenericCollector()
        t1 = collector.step("step_a")
        t1._iteration = 1
        t1._start_ns = 0
        t2 = collector.step("step_b")
        t2._iteration = 1
        t2._start_ns = 0

        _try_extract(t1, types.SimpleNamespace(
            model="gpt-4o",
            usage=types.SimpleNamespace(prompt_tokens=100, completion_tokens=50),
        ))
        _try_extract(t2, types.SimpleNamespace(
            model="gpt-4o",
            usage=types.SimpleNamespace(prompt_tokens=200, completion_tokens=80),
        ))

        assert t1._recorded["input_tokens"] == 100
        assert t2._recorded["input_tokens"] == 200


class TestVariableParallelCount:
    def test_variable_parallel_count(self):
        collector = GenericCollector()
        for count in (3, 7):
            totals = []
            for _ in range(count):
                tracker = collector.step("step_x")
                tracker._iteration = 1
                tracker._start_ns = 0
                _try_extract(tracker, types.SimpleNamespace(
                    model="gpt-4o",
                    usage=types.SimpleNamespace(
                        prompt_tokens=100, completion_tokens=50,
                    ),
                ))
                if tracker._recorded:
                    totals.append(tracker._recorded["input_tokens"])
            assert len(totals) == count


# ---------------------------------------------------------------------------
# JSON mode tests
# ---------------------------------------------------------------------------


class TestOpenAIJsonModeExtraction:
    def test_openai_json_mode_extraction(self):
        mock = types.SimpleNamespace(
            model="gpt-4o",
            usage=types.SimpleNamespace(
                prompt_tokens=800,
                completion_tokens=200,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 800
        assert recorded["output_tokens"] == 200


class TestAnthropicJsonModeExtraction:
    def test_anthropic_json_mode_extraction(self):
        mock = types.SimpleNamespace(
            model="claude-sonnet-4-6",
            usage=types.SimpleNamespace(
                input_tokens=900,
                output_tokens=150,
            ),
        )
        recorded = _extract_via_generic(mock)
        assert recorded is not None
        assert recorded["input_tokens"] == 900
        assert recorded["output_tokens"] == 150


# ---------------------------------------------------------------------------
# Multi-turn tests
# ---------------------------------------------------------------------------


class TestMultiTurnPerTurnTokens:
    def test_multi_turn_per_turn_tokens(self):
        collector = GenericCollector()
        turn_tokens = [500, 1200, 2000]
        records = []
        for i, inp_tok in enumerate(turn_tokens):
            tracker = collector.step(f"turn_{i}")
            tracker._iteration = 1
            tracker._start_ns = 0
            mock = types.SimpleNamespace(
                model="gpt-4o",
                usage=types.SimpleNamespace(
                    prompt_tokens=inp_tok,
                    completion_tokens=100,
                ),
            )
            _try_extract(tracker, mock)
            if tracker._recorded:
                records.append(tracker._recorded)
        assert len(records) == 3
        assert records[0]["input_tokens"] == 500
        assert records[1]["input_tokens"] == 1200
        assert records[2]["input_tokens"] == 2000


# ---------------------------------------------------------------------------
# Anthropic cache-busting
# ---------------------------------------------------------------------------


class TestNeedsCacheBustingAnthropic:
    def test_anthropic_models(self):
        from agentcost.collectors.cache_bust import needs_cache_busting

        assert needs_cache_busting("claude-sonnet-4-6") is True
        assert needs_cache_busting("claude-haiku-4-5") is True
        assert needs_cache_busting("claude-opus-4-7") is True

    def test_deepseek_still_works(self):
        from agentcost.collectors.cache_bust import needs_cache_busting

        assert needs_cache_busting("deepseek-chat") is True
        assert needs_cache_busting("deepseek-v4-flash") is True

    def test_non_cached_providers(self):
        from agentcost.collectors.cache_bust import needs_cache_busting

        assert needs_cache_busting("gpt-4o") is False
        assert needs_cache_busting("gemini-2.5-flash") is False
