"""Embedding API wrapper around LiteLLM for RAG workflows (W14/W15/W17)."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any

import litellm
from litellm import aembedding

from pretia.pricing.tables import resolve_model

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [1.0, 2.0, 4.0]

_DEFAULT_EMBEDDING_DIM = 1536


class EmbeddingCallError(Exception):
    """Raised when all retry attempts for an embedding call are exhausted."""


@dataclass(frozen=True, slots=True)
class EmbeddingResponse:
    """Normalized response from an embedding API call."""

    embedding: list[float]
    input_tokens: int
    duration_ms: int
    model: str


def _to_litellm_model(canonical: str) -> str:
    from bt_agents.providers.llm import LITELLM_MODEL_MAP

    resolved = resolve_model(canonical)
    return LITELLM_MODEL_MAP.get(resolved, resolved)


async def embed_text(
    text: str,
    model: str = "text-embedding-3-small",
    *,
    dry_run: bool = False,
) -> EmbeddingResponse:
    """Embed a single text string via LiteLLM.

    Returns:
        EmbeddingResponse with the embedding vector and token usage.
    """
    canonical = resolve_model(model)

    if not text.strip():
        return EmbeddingResponse(
            embedding=[0.0] * _DEFAULT_EMBEDDING_DIM,
            input_tokens=0,
            duration_ms=0,
            model=canonical,
        )

    if dry_run:
        return EmbeddingResponse(
            embedding=[0.0] * _DEFAULT_EMBEDDING_DIM,
            input_tokens=0,
            duration_ms=0,
            model=canonical,
        )

    litellm_model = _to_litellm_model(canonical)
    last_exc: Exception | None = None

    for attempt, delay in enumerate(_RETRY_DELAYS):
        try:
            start = time.monotonic()
            response = await aembedding(model=litellm_model, input=[text])
            elapsed_ms = int((time.monotonic() - start) * 1000)
            break
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "Embedding attempt %d/%d failed for %s: %s",
                attempt + 1,
                len(_RETRY_DELAYS),
                canonical,
                exc,
            )
            if attempt < len(_RETRY_DELAYS) - 1:
                await asyncio.sleep(delay)
    else:
        raise EmbeddingCallError(
            f"All {len(_RETRY_DELAYS)} attempts failed for {canonical}"
        ) from last_exc

    embedding_vec = response.data[0]["embedding"]
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    if input_tokens == 0:
        input_tokens = getattr(usage, "total_tokens", 0) or 0

    return EmbeddingResponse(
        embedding=embedding_vec,
        input_tokens=input_tokens,
        duration_ms=elapsed_ms,
        model=canonical,
    )


async def embed_batch(
    texts: list[str],
    model: str = "text-embedding-3-small",
    *,
    dry_run: bool = False,
) -> list[EmbeddingResponse]:
    """Embed multiple texts. Calls embed_text for each (sequential)."""
    results = []
    for text in texts:
        resp = await embed_text(text, model, dry_run=dry_run)
        results.append(resp)
    return results
