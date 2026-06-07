"""LLM wrapper for input generation calls.

Uses DeepSeek V4 Flash for short inputs (<4K tokens) and DeepSeek V4 Pro
for long outputs (>4K tokens). Cache-busts generation prompts to avoid
prefix caching producing duplicates across sequential calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_CACHE_BUST_PLACEHOLDER = "{{CACHE_BUST_SUFFIX}}"
_RETRY_DELAYS = [1.0, 2.0, 4.0]

MODEL_FLASH = "deepseek/deepseek-chat"
MODEL_PRO = "deepseek/deepseek-chat"

CONCURRENCY_FLASH = 500
CONCURRENCY_PRO = 100

_flash_semaphore: asyncio.Semaphore | None = None
_pro_semaphore: asyncio.Semaphore | None = None


def _get_semaphore(model: str) -> asyncio.Semaphore:
    """Return a per-model semaphore for concurrency limiting."""
    global _flash_semaphore, _pro_semaphore
    if model == MODEL_PRO:
        if _pro_semaphore is None:
            _pro_semaphore = asyncio.Semaphore(CONCURRENCY_PRO)
        return _pro_semaphore
    if _flash_semaphore is None:
        _flash_semaphore = asyncio.Semaphore(CONCURRENCY_FLASH)
    return _flash_semaphore


@dataclass(frozen=True, slots=True)
class GenerationResponse:
    """Response from a generation LLM call."""

    content: str
    input_tokens: int
    output_tokens: int
    model: str
    duration_ms: int


def _substitute_cache_bust(prompt: str) -> str:
    if _CACHE_BUST_PLACEHOLDER in prompt:
        return prompt.replace(_CACHE_BUST_PLACEHOLDER, uuid.uuid4().hex[:24])
    return prompt


def select_model(target_output_tokens: int) -> str:
    """Select Flash (<4K) or Pro (>=4K) based on expected output length."""
    return MODEL_PRO if target_output_tokens >= 4000 else MODEL_FLASH


async def generate_text(
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 4096,
    model: str | None = None,
    dry_run: bool = False,
) -> GenerationResponse:
    """Generate text via LiteLLM with cache-bust and retry."""
    if model is None:
        model = select_model(max_tokens)

    if dry_run:
        return GenerationResponse(
            content=f"[DRY RUN] Stub response for: {user_prompt[:100]}...",
            input_tokens=0,
            output_tokens=0,
            model=model,
            duration_ms=0,
        )

    from litellm import acompletion

    busted_prompt = _substitute_cache_bust(system_prompt)

    messages = [
        {"role": "system", "content": busted_prompt},
        {"role": "user", "content": user_prompt},
    ]

    sem = _get_semaphore(model)

    last_exc: Exception | None = None
    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            async with sem:
                start = time.monotonic()
                response = await acompletion(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                )
                elapsed_ms = int((time.monotonic() - start) * 1000)

            usage = getattr(response, "usage", None)
            content = response.choices[0].message.content or ""

            return GenerationResponse(
                content=content,
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
                model=model,
                duration_ms=elapsed_ms,
            )
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Generation attempt %d/%d failed: %s", attempt + 1, len(_RETRY_DELAYS), exc
            )
            if attempt < len(_RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)

    raise RuntimeError(
        f"All {len(_RETRY_DELAYS)} generation attempts failed"
    ) from last_exc


def generate_text_sync(
    system_prompt: str,
    user_prompt: str,
    **kwargs: Any,
) -> GenerationResponse:
    """Synchronous wrapper around ``generate_text``."""
    return asyncio.run(generate_text(system_prompt, user_prompt, **kwargs))
