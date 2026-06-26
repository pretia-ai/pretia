"""Centralized StepRecord construction from LLM and embedding responses.

This is the ONLY module that creates StepRecord instances in the agent suite.
All patterns and workflows delegate to these builder functions, guaranteeing
schema compliance with the existing ``pretia.collectors.base.StepRecord``.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import resolve_model

from bt_agents.providers.embeddings import EmbeddingResponse
from bt_agents.providers.llm import LLMResponse


def _prompt_hash(prompt: str) -> str:
    """SHA-256 hex digest of the prompt text."""
    return hashlib.sha256(prompt.encode()).hexdigest()


def build_llm_step(
    *,
    step_name: str,
    response: LLMResponse,
    system_prompt: str,
    output_format: str,
    iteration: int = 1,
    parent_step: str | None = None,
    is_retry: bool = False,
    tool_definitions_tokens: int = 0,
    **v2_fields: Any,
) -> StepRecord:
    """Build a StepRecord from an LLM response.

    Args:
        step_name: Descriptive name for this step.
        response: The LLMResponse returned by ``call_model``.
        system_prompt: The raw system prompt text (before cache-bust substitution).
        output_format: One of ``"json"``, ``"text"``, ``"code"``.
        iteration: Loop iteration number (>= 1).
        parent_step: Name of the parent step if this is a sub-step.
        is_retry: Whether this call is a retry of a previous failure.
        tool_definitions_tokens: Token count of tool schemas, if any.
        **v2_fields: Additional optional fields (``tool_name``,
            ``model_version``, ``temperature``, ``max_tokens_setting``,
            ``output_truncated``, ``output_tool_call_count``, etc.).
    """
    canonical = resolve_model(response.model)
    prompt_tokens = len(system_prompt) // 4

    record = StepRecord(
        step_name=step_name,
        step_type="llm",
        model=canonical,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        context_size=response.input_tokens,
        tool_definitions_tokens=tool_definitions_tokens,
        system_prompt_hash=_prompt_hash(system_prompt),
        system_prompt_tokens=prompt_tokens,
        output_format=output_format,
        is_retry=is_retry,
        iteration=iteration,
        parent_step=parent_step,
        duration_ms=response.duration_ms,
        timestamp=datetime.now(UTC),
        cache_hit_tokens=response.cache_hit_tokens or None,
        cache_miss_tokens=response.cache_miss_tokens or None,
        output_truncated=response.finish_reason == "length" or None,
        output_tool_call_count=len(response.tool_calls) if response.tool_calls else None,
        **{k: v for k, v in v2_fields.items() if v is not None},
    )
    return record


def build_embedding_step(
    *,
    step_name: str,
    response: EmbeddingResponse,
) -> StepRecord:
    """Build a StepRecord for an embedding API call.

    Embedding calls have ``step_type="retrieval"`` and ``output_tokens=0``.
    """
    canonical = resolve_model(response.model)

    return StepRecord(
        step_name=step_name,
        step_type="retrieval",
        model=canonical,
        input_tokens=response.input_tokens,
        output_tokens=0,
        context_size=response.input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash=hashlib.sha256(b"").hexdigest(),
        system_prompt_tokens=0,
        output_format="text",
        is_retry=False,
        iteration=1,
        parent_step=None,
        duration_ms=response.duration_ms,
        timestamp=datetime.now(UTC),
    )
