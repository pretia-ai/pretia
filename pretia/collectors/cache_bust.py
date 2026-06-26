"""Cache-busting utilities for server-side prompt caching."""

from __future__ import annotations

import uuid

_CACHE_BUSTING_PREFIXES = frozenset(
    {
        "deepseek",
        "claude",
        "haiku",
        "sonnet",
        "opus",
    }
)


def needs_cache_busting(model: str) -> bool:
    """Check if a model has aggressive server-side prompt caching."""
    model_lower = model.lower()
    return any(prefix in model_lower for prefix in _CACHE_BUSTING_PREFIXES)


def cache_bust_prompt(prompt: str, run_id: str | None = None) -> str:
    """Append a unique suffix to break server-side prompt caching."""
    if run_id is None:
        run_id = uuid.uuid4().hex[:12]
    return prompt + f"\n<!-- profiling-run-{run_id} -->"
