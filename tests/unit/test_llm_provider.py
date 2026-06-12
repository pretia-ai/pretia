"""Unit tests for agents.providers.llm — LLM call wrapper utilities."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace

import pytest

from bt_agents.providers.llm import (
    LLMResponse,
    _CACHE_BUST_PLACEHOLDER,
    _extract_cache_tokens,
    _extract_tool_calls,
    _substitute_cache_bust,
    _to_litellm_model,
    call_model,
)


# ---------------------------------------------------------------------------
# _substitute_cache_bust
# ---------------------------------------------------------------------------


class TestSubstituteCacheBust:
    """Tests for _substitute_cache_bust."""

    def test_placeholder_replaced_and_prefix_prepended(self) -> None:
        """Prompt gets a unique prefix prepended and placeholder substituted."""
        prompt = f"You are a helpful assistant. {_CACHE_BUST_PLACEHOLDER}"
        result = _substitute_cache_bust(prompt)
        assert _CACHE_BUST_PLACEHOLDER not in result
        assert result.startswith("<!-- req:")
        assert "You are a helpful assistant." in result

    def test_prompt_without_placeholder_gets_prefix(self) -> None:
        """Prompt with no placeholder still gets the unique prefix prepended."""
        prompt = "Plain system prompt with no special markers."
        result = _substitute_cache_bust(prompt)
        assert result.startswith("<!-- req:")
        assert result.endswith(prompt)

    def test_two_calls_produce_different_uuids(self) -> None:
        """Consecutive calls yield distinct substitution values (cache uniqueness)."""
        prompt = f"prefix-{_CACHE_BUST_PLACEHOLDER}-suffix"
        result_a = _substitute_cache_bust(prompt)
        result_b = _substitute_cache_bust(prompt)
        assert result_a != result_b

    def test_multiple_placeholders_all_replaced(self) -> None:
        """Every occurrence of the placeholder in a single prompt is replaced."""
        prompt = f"A {_CACHE_BUST_PLACEHOLDER} middle {_CACHE_BUST_PLACEHOLDER} end"
        result = _substitute_cache_bust(prompt)
        assert _CACHE_BUST_PLACEHOLDER not in result
        assert "A " in result
        assert " middle " in result
        assert result.endswith(" end")


# ---------------------------------------------------------------------------
# _to_litellm_model
# ---------------------------------------------------------------------------


class TestToLitellmModel:
    """Tests for _to_litellm_model."""

    def test_claude_haiku(self) -> None:
        assert _to_litellm_model("claude-haiku-4-5") == "anthropic/claude-haiku-4-5"

    def test_deepseek_v4_pro(self) -> None:
        assert _to_litellm_model("deepseek-v4-pro") == "deepseek/deepseek-v4-pro"

    def test_gpt_5_4_nano(self) -> None:
        assert _to_litellm_model("gpt-5.4-nano") == "openai/gpt-5.4-nano"

    def test_unknown_model_falls_through_with_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Model in MODEL_PRICING but absent from LITELLM_MODEL_MAP logs a
        warning and returns the canonical name directly."""
        # "gpt-4o" exists in MODEL_PRICING but has no LITELLM_MODEL_MAP entry
        with caplog.at_level(logging.WARNING, logger="bt_agents.providers.llm"):
            result = _to_litellm_model("gpt-4o")
        assert result == "gpt-4o"
        assert "No LiteLLM mapping" in caplog.text


# ---------------------------------------------------------------------------
# _extract_cache_tokens
# ---------------------------------------------------------------------------


class TestExtractCacheTokens:
    """Tests for _extract_cache_tokens."""

    def test_deepseek_style_prompt_cache_hit(self) -> None:
        """DeepSeek-style usage with prompt_cache_hit_tokens."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=500,
                prompt_cache_hit_tokens=100,
                completion_tokens=50,
            )
        )
        hit, miss = _extract_cache_tokens(response)
        assert hit == 100
        assert miss == 400  # prompt_tokens - hit

    def test_anthropic_style_cache_read(self) -> None:
        """Anthropic-style usage with cache_read_input_tokens."""
        response = SimpleNamespace(
            usage=SimpleNamespace(
                prompt_tokens=600,
                prompt_cache_hit_tokens=0,
                cache_read_input_tokens=200,
                cache_creation_input_tokens=0,
            )
        )
        hit, miss = _extract_cache_tokens(response)
        assert hit == 200
        assert miss == 400  # prompt_tokens - hit

    def test_no_usage_attribute(self) -> None:
        """Object with no usage attribute returns (0, 0)."""
        response = SimpleNamespace()
        hit, miss = _extract_cache_tokens(response)
        assert (hit, miss) == (0, 0)


# ---------------------------------------------------------------------------
# _extract_tool_calls
# ---------------------------------------------------------------------------


class TestExtractToolCalls:
    """Tests for _extract_tool_calls."""

    def test_response_with_tool_calls(self) -> None:
        """Extract tool call dicts with id, type, function.name, function.arguments."""
        tool_call = SimpleNamespace(
            id="call_abc123",
            type="function",
            function=SimpleNamespace(
                name="search_web",
                arguments='{"query": "weather"}',
            ),
        )
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(tool_calls=[tool_call])
                )
            ]
        )
        result = _extract_tool_calls(response)
        assert len(result) == 1
        assert result[0]["id"] == "call_abc123"
        assert result[0]["type"] == "function"
        assert result[0]["function"]["name"] == "search_web"
        assert result[0]["function"]["arguments"] == '{"query": "weather"}'

    def test_response_without_tool_calls(self) -> None:
        """Response with no tool_calls returns an empty list."""
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(tool_calls=None)
                )
            ]
        )
        assert _extract_tool_calls(response) == []


# ---------------------------------------------------------------------------
# LLMResponse defaults
# ---------------------------------------------------------------------------


class TestLLMResponse:
    """Tests for LLMResponse dataclass defaults."""

    def test_defaults(self) -> None:
        """Verify all default field values on a minimal LLMResponse."""
        resp = LLMResponse(content="hello", input_tokens=10, output_tokens=5)
        assert resp.cache_hit_tokens == 0
        assert resp.cache_miss_tokens == 0
        assert resp.tool_calls == []
        assert resp.cost_usd == 0.0
        assert resp.finish_reason == "stop"
        assert resp.duration_ms == 0
        assert resp.model == ""
        assert resp.raw_response is None


# ---------------------------------------------------------------------------
# call_model dry_run
# ---------------------------------------------------------------------------


class TestCallModelDryRun:
    """Tests for call_model in dry_run mode."""

    def test_dry_run_returns_synthetic_response(self) -> None:
        """dry_run=True returns an LLMResponse without making any API call."""
        resp = asyncio.run(
            call_model(
                "claude-haiku-4-5",
                "test prompt",
                [{"role": "user", "content": "hi"}],
                dry_run=True,
            )
        )
        assert resp.content == '{"dry_run": true}'
        assert resp.input_tokens == 0
        assert resp.output_tokens == 0
        assert resp.model == "claude-haiku-4-5"
