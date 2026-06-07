"""Test the StepRecord builder at agents/harness/step_builder.py."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from agentcost.collectors.base import StepRecord
from agents.harness.step_builder import build_embedding_step, build_llm_step
from agents.providers.embeddings import EmbeddingResponse
from agents.providers.llm import LLMResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_response() -> LLMResponse:
    """Minimal LLMResponse for testing build_llm_step."""
    return LLMResponse(
        content='{"result": "test"}',
        input_tokens=100,
        output_tokens=50,
        duration_ms=500,
        model="claude-sonnet-4-6",
        finish_reason="stop",
    )


@pytest.fixture
def mock_llm_response_with_tools() -> LLMResponse:
    """LLMResponse with tool calls and finish_reason='length'."""
    return LLMResponse(
        content="partial output",
        input_tokens=200,
        output_tokens=80,
        duration_ms=1200,
        model="claude-sonnet-4-6",
        finish_reason="length",
        tool_calls=[
            {"id": "call_1", "type": "function", "function": {"name": "search", "arguments": "{}"}},
            {"id": "call_2", "type": "function", "function": {"name": "read", "arguments": "{}"}},
        ],
    )


@pytest.fixture
def mock_embedding_response() -> EmbeddingResponse:
    """Minimal EmbeddingResponse for testing build_embedding_step."""
    return EmbeddingResponse(
        embedding=[0.1] * 1536,
        input_tokens=25,
        duration_ms=100,
        model="text-embedding-3-small",
    )


# ---------------------------------------------------------------------------
# build_llm_step
# ---------------------------------------------------------------------------


class TestBuildLlmStep:
    """Tests for build_llm_step."""

    def test_produces_valid_step_record(self, mock_llm_response: LLMResponse) -> None:
        """build_llm_step returns a StepRecord with all required fields populated."""
        record = build_llm_step(
            step_name="classify",
            response=mock_llm_response,
            system_prompt="You are a classifier.",
            output_format="json",
        )
        assert isinstance(record, StepRecord)
        assert record.step_name == "classify"
        assert record.input_tokens == 100
        assert record.output_tokens == 50
        assert record.duration_ms == 500
        assert record.output_format == "json"
        assert record.is_retry is False
        assert record.parent_step is None
        assert record.timestamp is not None

    def test_step_type_always_llm(self, mock_llm_response: LLMResponse) -> None:
        """step_type is always 'llm' for LLM steps."""
        record = build_llm_step(
            step_name="step_a",
            response=mock_llm_response,
            system_prompt="system",
            output_format="text",
        )
        assert record.step_type == "llm"

    def test_model_resolved_to_canonical(self) -> None:
        """Model aliases are resolved to canonical form."""
        # "claude-sonnet-4" is an alias for "claude-sonnet-4-6"
        response = LLMResponse(
            content="ok",
            input_tokens=10,
            output_tokens=5,
            duration_ms=100,
            model="claude-sonnet-4",
            finish_reason="stop",
        )
        record = build_llm_step(
            step_name="alias_test",
            response=response,
            system_prompt="prompt",
            output_format="text",
        )
        assert record.model == "claude-sonnet-4-6"

    def test_system_prompt_hash_is_hex_digest(self, mock_llm_response: LLMResponse) -> None:
        """system_prompt_hash is a non-empty hex digest."""
        prompt = "You are a helpful assistant."
        record = build_llm_step(
            step_name="hash_test",
            response=mock_llm_response,
            system_prompt=prompt,
            output_format="text",
        )
        assert len(record.system_prompt_hash) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in record.system_prompt_hash)

    def test_system_prompt_hash_consistent(self, mock_llm_response: LLMResponse) -> None:
        """Same system prompt always produces the same hash."""
        prompt = "Consistent prompt text."
        r1 = build_llm_step(
            step_name="a", response=mock_llm_response, system_prompt=prompt, output_format="text"
        )
        r2 = build_llm_step(
            step_name="b", response=mock_llm_response, system_prompt=prompt, output_format="text"
        )
        assert r1.system_prompt_hash == r2.system_prompt_hash
        expected = hashlib.sha256(prompt.encode()).hexdigest()
        assert r1.system_prompt_hash == expected

    def test_system_prompt_tokens_estimated(self, mock_llm_response: LLMResponse) -> None:
        """system_prompt_tokens is estimated as len(prompt) // 4."""
        prompt = "A" * 100  # 100 chars -> 25 tokens
        record = build_llm_step(
            step_name="tok_est",
            response=mock_llm_response,
            system_prompt=prompt,
            output_format="text",
        )
        assert record.system_prompt_tokens == 25

    def test_context_size_equals_input_tokens(self, mock_llm_response: LLMResponse) -> None:
        """context_size is set to response.input_tokens."""
        record = build_llm_step(
            step_name="ctx",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
        )
        assert record.context_size == mock_llm_response.input_tokens

    def test_iteration_defaults_to_one(self, mock_llm_response: LLMResponse) -> None:
        """iteration defaults to 1 when not specified."""
        record = build_llm_step(
            step_name="iter",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
        )
        assert record.iteration == 1

    def test_iteration_custom_value(self, mock_llm_response: LLMResponse) -> None:
        """iteration can be set to a custom value."""
        record = build_llm_step(
            step_name="iter_custom",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
            iteration=3,
        )
        assert record.iteration == 3

    def test_output_truncated_when_finish_reason_length(
        self, mock_llm_response_with_tools: LLMResponse
    ) -> None:
        """output_truncated is True when finish_reason is 'length'."""
        record = build_llm_step(
            step_name="trunc",
            response=mock_llm_response_with_tools,
            system_prompt="s",
            output_format="text",
        )
        assert record.output_truncated is True

    def test_output_not_truncated_when_stop(self, mock_llm_response: LLMResponse) -> None:
        """output_truncated is None when finish_reason is 'stop'."""
        record = build_llm_step(
            step_name="no_trunc",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
        )
        assert record.output_truncated is None

    def test_output_tool_call_count(
        self, mock_llm_response_with_tools: LLMResponse
    ) -> None:
        """output_tool_call_count counts tool_calls in the response."""
        record = build_llm_step(
            step_name="tools",
            response=mock_llm_response_with_tools,
            system_prompt="s",
            output_format="text",
        )
        assert record.output_tool_call_count == 2

    def test_output_tool_call_count_none_when_no_tools(
        self, mock_llm_response: LLMResponse
    ) -> None:
        """output_tool_call_count is None when no tool calls present."""
        record = build_llm_step(
            step_name="no_tools",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
        )
        assert record.output_tool_call_count is None

    def test_round_trip_serialization(self, mock_llm_response: LLMResponse) -> None:
        """StepRecord round-trips through to_dict / from_dict."""
        record = build_llm_step(
            step_name="roundtrip",
            response=mock_llm_response,
            system_prompt="Serialize me.",
            output_format="json",
            iteration=2,
            parent_step="parent_node",
            is_retry=True,
            tool_definitions_tokens=42,
        )
        data = record.to_dict()
        restored = StepRecord.from_dict(data)

        assert restored.step_name == record.step_name
        assert restored.step_type == record.step_type
        assert restored.model == record.model
        assert restored.input_tokens == record.input_tokens
        assert restored.output_tokens == record.output_tokens
        assert restored.context_size == record.context_size
        assert restored.tool_definitions_tokens == record.tool_definitions_tokens
        assert restored.system_prompt_hash == record.system_prompt_hash
        assert restored.system_prompt_tokens == record.system_prompt_tokens
        assert restored.output_format == record.output_format
        assert restored.is_retry == record.is_retry
        assert restored.iteration == record.iteration
        assert restored.parent_step == record.parent_step
        assert restored.duration_ms == record.duration_ms

    def test_v2_fields_pass_through(self, mock_llm_response: LLMResponse) -> None:
        """Extra v2 fields like tool_name pass through to the StepRecord."""
        record = build_llm_step(
            step_name="v2",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
            tool_name="web_search",
            temperature=0.7,
            max_tokens_setting=4096,
        )
        assert record.tool_name == "web_search"
        assert record.temperature == pytest.approx(0.7)
        assert record.max_tokens_setting == 4096

    def test_v2_none_fields_excluded(self, mock_llm_response: LLMResponse) -> None:
        """v2 fields with None values are not passed to StepRecord constructor."""
        record = build_llm_step(
            step_name="v2_none",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
            tool_name=None,
            model_version=None,
        )
        # None v2 fields should remain as their default None
        assert record.tool_name is None
        assert record.model_version is None

    def test_parent_step_and_retry(self, mock_llm_response: LLMResponse) -> None:
        """parent_step and is_retry are properly set."""
        record = build_llm_step(
            step_name="child",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
            parent_step="orchestrator",
            is_retry=True,
        )
        assert record.parent_step == "orchestrator"
        assert record.is_retry is True

    def test_tool_definitions_tokens(self, mock_llm_response: LLMResponse) -> None:
        """tool_definitions_tokens is forwarded to the StepRecord."""
        record = build_llm_step(
            step_name="with_tools",
            response=mock_llm_response,
            system_prompt="s",
            output_format="text",
            tool_definitions_tokens=350,
        )
        assert record.tool_definitions_tokens == 350


# ---------------------------------------------------------------------------
# build_embedding_step
# ---------------------------------------------------------------------------


class TestBuildEmbeddingStep:
    """Tests for build_embedding_step."""

    def test_step_type_always_retrieval(self, mock_embedding_response: EmbeddingResponse) -> None:
        """step_type is always 'retrieval' for embedding steps."""
        record = build_embedding_step(
            step_name="embed_query",
            response=mock_embedding_response,
        )
        assert record.step_type == "retrieval"

    def test_output_tokens_always_zero(self, mock_embedding_response: EmbeddingResponse) -> None:
        """output_tokens is always 0 for embedding steps."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        assert record.output_tokens == 0

    def test_output_format_always_text(self, mock_embedding_response: EmbeddingResponse) -> None:
        """output_format is always 'text' for embedding steps."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        assert record.output_format == "text"

    def test_iteration_always_one(self, mock_embedding_response: EmbeddingResponse) -> None:
        """iteration is always 1 for embedding steps."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        assert record.iteration == 1

    def test_parent_step_always_none(self, mock_embedding_response: EmbeddingResponse) -> None:
        """parent_step is always None for embedding steps."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        assert record.parent_step is None

    def test_input_tokens_from_response(self, mock_embedding_response: EmbeddingResponse) -> None:
        """input_tokens comes from the embedding response."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        assert record.input_tokens == 25

    def test_model_resolved(self) -> None:
        """Model name is resolved to canonical form."""
        response = EmbeddingResponse(
            embedding=[0.0] * 1536,
            input_tokens=10,
            duration_ms=50,
            model="text-embedding-3-small",
        )
        record = build_embedding_step(step_name="embed", response=response)
        assert record.model == "text-embedding-3-small"

    def test_is_retry_false(self, mock_embedding_response: EmbeddingResponse) -> None:
        """is_retry is always False for embedding steps."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        assert record.is_retry is False

    def test_system_prompt_hash_empty(self, mock_embedding_response: EmbeddingResponse) -> None:
        """system_prompt_hash is the SHA-256 of empty bytes."""
        record = build_embedding_step(
            step_name="embed",
            response=mock_embedding_response,
        )
        expected = hashlib.sha256(b"").hexdigest()
        assert record.system_prompt_hash == expected

    def test_produces_valid_step_record(self, mock_embedding_response: EmbeddingResponse) -> None:
        """build_embedding_step returns a proper StepRecord instance."""
        record = build_embedding_step(
            step_name="embed_doc",
            response=mock_embedding_response,
        )
        assert isinstance(record, StepRecord)
        assert record.duration_ms == 100
        assert record.context_size == 25
        assert record.tool_definitions_tokens == 0
        assert record.system_prompt_tokens == 0
