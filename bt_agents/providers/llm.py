"""Unified LLM call wrapper around LiteLLM.

Every LLM API call across all 14 workflow agents goes through ``call_model``.
Handles cache-bust substitution, usage extraction, retry, and dry-run mode.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import litellm
from litellm import acompletion

from pretia.pricing.tables import calculate_cost, resolve_model

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LiteLLM model string mapping
# ---------------------------------------------------------------------------
# Maps Pretia canonical model names (from pricing/tables.py) to the
# provider-prefixed strings LiteLLM expects for routing.
LITELLM_MODEL_MAP: dict[str, str] = {
    "claude-opus-4-7": "anthropic/claude-opus-4-7",
    "claude-sonnet-4-6": "anthropic/claude-sonnet-4-6",
    "claude-haiku-4-5": "anthropic/claude-haiku-4-5",
    "gpt-5.5": "openai/gpt-5.5",
    "gpt-5.4": "openai/gpt-5.4",
    "gpt-5.4-nano": "openai/gpt-5.4-nano",
    "gpt-4.1": "openai/gpt-4.1",
    "gpt-4.1-nano": "openai/gpt-4.1-nano",
    "deepseek-v4-pro": "deepseek/deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek/deepseek-v4-flash",
    "qwen-turbo": "dashscope/qwen-turbo",
    "qwen3.6-plus": "dashscope/qwen3.6-plus",
    "gemini-2.5-flash": "gemini/gemini-2.5-flash",
    "text-embedding-3-small": "openai/text-embedding-3-small",
    "text-embedding-3-large": "openai/text-embedding-3-large",
}

_CACHE_BUST_PLACEHOLDER = "{{CACHE_BUST_SUFFIX}}"

_MAX_RETRIES = 3
_MAX_RETRIES_RATE_LIMIT = 5


def _retry_delay(attempt: int) -> float:
    """Exponential backoff with jitter to avoid thundering herd."""
    base = min(2**attempt, 30)
    jitter = random.uniform(0, base * 0.5)
    return base + jitter


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect rate-limit errors (HTTP 429 or provider-specific messages)."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "rate_limit" in msg:
        return True
    status = getattr(exc, "status_code", None)
    return status == 429


class LLMCallError(Exception):
    """Raised when all retry attempts for an LLM call are exhausted."""


@dataclass(frozen=True, slots=True)
class LLMResponse:
    """Normalized response from any LLM provider via LiteLLM."""

    content: str
    input_tokens: int
    output_tokens: int
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    finish_reason: str = "stop"
    duration_ms: int = 0
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    cost_usd: float = 0.0
    model: str = ""
    raw_response: Any = None


def _substitute_cache_bust(prompt: str) -> str:
    """Prepend a unique prefix AND replace any inline placeholder.

    DeepSeek's prefix KV cache caches from the start of the prompt, so the
    unique token must appear at the very beginning to defeat caching.
    """
    prefix = f"<!-- req:{uuid.uuid4().hex[:24]} -->\n"
    if _CACHE_BUST_PLACEHOLDER in prompt:
        prompt = prompt.replace(_CACHE_BUST_PLACEHOLDER, uuid.uuid4().hex[:24])
    return prefix + prompt


def _to_litellm_model(canonical: str) -> str:
    """Resolve a canonical model name to a LiteLLM routing string."""
    resolved = resolve_model(canonical)
    litellm_str = LITELLM_MODEL_MAP.get(resolved)
    if litellm_str is None:
        logger.warning(
            "No LiteLLM mapping for %r, using canonical name directly", resolved
        )
        return resolved
    return litellm_str


def _extract_cache_tokens(response: Any) -> tuple[int, int]:
    """Extract cache hit/miss tokens from a LiteLLM response.

    DeepSeek exposes ``prompt_cache_hit_tokens`` in usage. Anthropic exposes
    ``cache_creation_input_tokens`` and ``cache_read_input_tokens``.
    """
    hit, miss = 0, 0
    usage = getattr(response, "usage", None)
    if usage is None:
        return hit, miss

    hit = getattr(usage, "prompt_cache_hit_tokens", 0) or 0
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    hit = hit or cache_read

    cache_create = getattr(usage, "cache_creation_input_tokens", 0) or 0
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
    miss = cache_create or max(0, prompt_tokens - hit)

    return hit, miss


def _extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract tool call dicts from a LiteLLM response."""
    try:
        message = response.choices[0].message
        raw_calls = getattr(message, "tool_calls", None)
        if not raw_calls:
            return []
        calls = []
        for tc in raw_calls:
            calls.append({
                "id": getattr(tc, "id", ""),
                "type": "function",
                "function": {
                    "name": getattr(tc.function, "name", ""),
                    "arguments": getattr(tc.function, "arguments", "{}"),
                },
            })
        return calls
    except (AttributeError, IndexError):
        return []


async def call_model(
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    *,
    max_tokens: int = 4096,
    tools: list[dict[str, Any]] | None = None,
    temperature: float | None = None,
    dry_run: bool = False,
) -> LLMResponse:
    """Call an LLM via LiteLLM with cache-bust, retry, and usage extraction.

    Args:
        model: Canonical model name from the Pretia pricing table.
        system_prompt: The system prompt text. ``{{CACHE_BUST_SUFFIX}}`` is
            replaced with a fresh UUID automatically.
        messages: List of message dicts (role/content) for the user/assistant
            conversation history.
        max_tokens: Maximum output tokens.
        tools: Optional tool/function schemas for function calling.
        temperature: Optional temperature override.
        dry_run: If True, return a synthetic response without making an API call.

    Returns:
        LLMResponse with usage data extracted from the provider's response.

    Raises:
        LLMCallError: If all retry attempts fail.
    """
    canonical = resolve_model(model)

    if dry_run:
        return LLMResponse(
            content='{"dry_run": true}',
            input_tokens=0,
            output_tokens=0,
            model=canonical,
        )

    busted_prompt = _substitute_cache_bust(system_prompt)
    litellm_model = _to_litellm_model(canonical)

    full_messages = [{"role": "system", "content": busted_prompt}, *messages]

    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": full_messages,
        "max_tokens": max_tokens,
    }
    if tools:
        kwargs["tools"] = tools
    if temperature is not None:
        kwargs["temperature"] = temperature

    last_exc: Exception | None = None
    max_attempts = _MAX_RETRIES
    attempt = 0
    while attempt < max_attempts:
        try:
            start = time.monotonic()
            response = await acompletion(**kwargs)
            elapsed_ms = int((time.monotonic() - start) * 1000)
            break
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit_error(exc):
                max_attempts = _MAX_RETRIES_RATE_LIMIT
            logger.warning(
                "LLM call attempt %d/%d failed for %s: %s",
                attempt + 1,
                max_attempts,
                canonical,
                exc,
            )
            attempt += 1
            if attempt < max_attempts:
                await asyncio.sleep(_retry_delay(attempt - 1))
    else:
        raise LLMCallError(
            f"All {max_attempts} attempts failed for {canonical}"
        ) from last_exc

    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    cache_hit, cache_miss = _extract_cache_tokens(response)

    try:
        content = response.choices[0].message.content or ""
    except (AttributeError, IndexError):
        content = ""

    try:
        finish_reason = response.choices[0].finish_reason or "stop"
    except (AttributeError, IndexError):
        finish_reason = "stop"

    tool_calls = _extract_tool_calls(response)

    cost = calculate_cost(
        canonical, input_tokens, output_tokens, cache_hit, cache_miss
    )

    return LLMResponse(
        content=content,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_hit_tokens=cache_hit,
        cache_miss_tokens=cache_miss,
        finish_reason=finish_reason,
        duration_ms=elapsed_ms,
        tool_calls=tool_calls,
        cost_usd=cost,
        model=canonical,
        raw_response=response,
    )


def call_model_sync(
    model: str,
    system_prompt: str,
    messages: list[dict[str, Any]],
    **kwargs: Any,
) -> LLMResponse:
    """Synchronous wrapper around :func:`call_model`."""
    return asyncio.run(call_model(model, system_prompt, messages, **kwargs))
