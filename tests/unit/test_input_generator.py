"""Tests for input generator: parsing, provider detection, API key resolution."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agentcost.inputs.generator import (
    _GENERATION_PROMPT_TEMPLATE,
    _parse_response,
    generate_inputs,
)

# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

class TestParseResponse:
    def test_clean_output(self):
        text = "\n".join(f"Input number {i}" for i in range(20))
        result = _parse_response(text, 20)
        assert len(result) == 20
        assert all(s.strip() for s in result)

    def test_numbered_lines_stripped(self):
        text = (
            "1. How do I reset my password?\n"
            "2. What are your hours?\n"
            "3) Tell me about pricing"
        )
        result = _parse_response(text, 10)
        assert result == [
            "How do I reset my password?",
            "What are your hours?",
            "Tell me about pricing",
        ]

    def test_preamble_discarded(self):
        text = (
            "Here are 20 test inputs:\n"
            "How do I reset my password?\n"
            "What are your hours?"
        )
        result = _parse_response(text, 10)
        assert result == [
            "How do I reset my password?",
            "What are your hours?",
        ]

    def test_fewer_than_n_returns_available(self, caplog):
        text = "\n".join(f"Input {i}" for i in range(15))
        with caplog.at_level("WARNING"):
            result = _parse_response(text, 20)
        assert len(result) == 15
        assert "Requested 20" in caplog.text

    def test_more_than_n_truncated(self):
        text = "\n".join(f"Input {i}" for i in range(25))
        result = _parse_response(text, 20)
        assert len(result) == 20

    def test_blank_lines_skipped(self):
        text = "First input\n\n\nSecond input\n\n"
        result = _parse_response(text, 10)
        assert result == ["First input", "Second input"]


# ---------------------------------------------------------------------------
# Provider detection
# ---------------------------------------------------------------------------

def _mock_anthropic_sdk():
    sdk = MagicMock()
    client = AsyncMock()
    content_block = MagicMock()
    content_block.text = "Generated input 1\nGenerated input 2"
    response = MagicMock()
    response.content = [content_block]
    client.messages.create = AsyncMock(return_value=response)
    sdk.AsyncAnthropic.return_value = client
    return sdk


def _mock_openai_sdk():
    sdk = MagicMock()
    client = AsyncMock()
    message = MagicMock()
    message.content = "Generated input 1\nGenerated input 2"
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    client.chat.completions.create = AsyncMock(return_value=response)
    sdk.AsyncOpenAI.return_value = client
    return sdk


class TestProviderDetection:
    @pytest.mark.asyncio
    async def test_anthropic_preferred(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        anthropic_sdk = _mock_anthropic_sdk()
        openai_sdk = _mock_openai_sdk()

        with patch(
            "agentcost.inputs.generator._try_import",
            side_effect=lambda n: (
                anthropic_sdk if n == "anthropic" else openai_sdk
            ),
        ):
            result = await generate_inputs("You are a bot.", n=2)

        anthropic_sdk.AsyncAnthropic.assert_called_once()
        openai_sdk.AsyncOpenAI.assert_not_called()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_openai_fallback(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        openai_sdk = _mock_openai_sdk()

        with patch(
            "agentcost.inputs.generator._try_import",
            side_effect=lambda n: (
                openai_sdk if n == "openai" else None
            ),
        ):
            result = await generate_inputs(
                "You are a bot.", n=2, model="gpt-4o-mini",
            )

        openai_sdk.AsyncOpenAI.assert_called_once()
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_explicit_openai_model_forces_openai(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
        anthropic_sdk = _mock_anthropic_sdk()
        openai_sdk = _mock_openai_sdk()

        with patch(
            "agentcost.inputs.generator._try_import",
            side_effect=lambda n: (
                anthropic_sdk if n == "anthropic" else openai_sdk
            ),
        ):
            await generate_inputs(
                "You are a bot.", n=2, model="gpt-4o-mini",
            )

        openai_sdk.AsyncOpenAI.assert_called_once()
        anthropic_sdk.AsyncAnthropic.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_sdk_raises_import_error(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        with patch(
            "agentcost.inputs.generator._try_import",
            return_value=None,
        ):
            with pytest.raises(ImportError, match="anthropic.*openai"):
                await generate_inputs("You are a bot.", n=2)

    @pytest.mark.asyncio
    async def test_no_api_key_raises_value_error(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        sdk = _mock_anthropic_sdk()

        with patch(
            "agentcost.inputs.generator._try_import",
            side_effect=lambda n: sdk if n == "anthropic" else None,
        ):
            with pytest.raises(ValueError, match="No API key"):
                await generate_inputs("You are a bot.", n=2)

    @pytest.mark.asyncio
    async def test_deepseek_provider_detection(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        openai_sdk = _mock_openai_sdk()

        with patch(
            "agentcost.inputs.generator._try_import",
            side_effect=lambda n: openai_sdk if n == "openai" else None,
        ):
            result = await generate_inputs(
                "You are a bot.", n=2, model="deepseek-v4-flash",
            )

        openai_sdk.AsyncOpenAI.assert_called_once()
        call_kwargs = openai_sdk.AsyncOpenAI.call_args
        assert "api.deepseek.com" in str(call_kwargs)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_deepseek_provider_priority(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
        anthropic_sdk = _mock_anthropic_sdk()
        openai_sdk = _mock_openai_sdk()

        with patch(
            "agentcost.inputs.generator._try_import",
            side_effect=lambda n: (
                anthropic_sdk if n == "anthropic" else openai_sdk
            ),
        ):
            await generate_inputs("You are a bot.", n=2)

        anthropic_sdk.AsyncAnthropic.assert_called_once()
        openai_sdk.AsyncOpenAI.assert_not_called()


# ---------------------------------------------------------------------------
# Meta-prompt content
# ---------------------------------------------------------------------------

class TestMetaPrompt:
    def test_has_required_placeholders(self):
        assert "{system_prompt}" in _GENERATION_PROMPT_TEMPLATE
        assert "{n}" in _GENERATION_PROMPT_TEMPLATE

    def test_mentions_diversity(self):
        lower = _GENERATION_PROMPT_TEMPLATE.lower()
        assert "diverse" in lower or "diversity" in lower
        assert "edge case" in lower or "edge cases" in lower

    def test_mentions_one_per_line(self):
        assert "one per line" in _GENERATION_PROMPT_TEMPLATE.lower()
