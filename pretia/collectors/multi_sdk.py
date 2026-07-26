"""Collect step-level token usage by patching both Anthropic and OpenAI SDKs."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from pretia.collectors.base import _DEFAULT_CONCURRENCY, BaseCollector, StepRecord

logger = logging.getLogger(__name__)


class MultiSDKCollector(BaseCollector):
    """Patch both Anthropic and OpenAI SDK classes simultaneously.

    Runs execute concurrently via asyncio.gather. Each run's captures are
    scoped via ContextVar in the underlying SDK collectors.
    """

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
        concurrency: int | None = None,
    ) -> list[list[StepRecord]]:
        total = len(inputs)
        results: list[list[StepRecord]] = [[] for _ in range(total)]

        patches: list[tuple[Any, str, Any]] = []
        patches.extend(_patch_anthropic())
        patches.extend(_patch_openai())

        if not patches:
            logger.warning(
                "MultiSDKCollector could not patch any SDK. Install anthropic or openai."
            )

        sem = asyncio.Semaphore(concurrency or _DEFAULT_CONCURRENCY)
        anthropic_ctx, openai_ctx = _get_ctx_vars()

        async def _run_one(idx: int, inp: str) -> None:
            captured: list[StepRecord] = []
            lock = asyncio.Lock()
            counters: dict[str, int] = {}
            ctx_val = (captured, lock, counters)

            tokens = []
            if anthropic_ctx is not None:
                tokens.append(anthropic_ctx.set(ctx_val))
            if openai_ctx is not None:
                tokens.append(openai_ctx.set(ctx_val))

            try:
                async with sem:
                    await workflow(inp)
            except Exception:
                logger.error(
                    "Run %d/%d failed on input %.80s",
                    idx + 1,
                    total,
                    inp,
                    exc_info=True,
                )
            finally:
                for tok in tokens:
                    tok.var.reset(tok)

            if not captured:
                logger.warning("Run %d produced 0 steps.", idx + 1)

            results[idx] = captured
            if on_run_complete is not None:
                try:
                    on_run_complete(idx, total, captured)
                except Exception:
                    logger.debug("on_run_complete callback failed", exc_info=True)

        try:
            await asyncio.gather(*[_run_one(i, inp) for i, inp in enumerate(inputs)])
        finally:
            for target, attr, original in patches:
                setattr(target, attr, original)

        return results


def _get_ctx_vars() -> tuple[Any, Any]:
    """Return the ContextVar from each SDK collector, or None if not available."""
    anthropic_ctx = None
    openai_ctx = None
    try:
        from pretia.collectors.anthropic_sdk import _run_ctx as a_ctx

        anthropic_ctx = a_ctx
    except ImportError:
        pass
    try:
        from pretia.collectors.openai_sdk import _run_ctx as o_ctx

        openai_ctx = o_ctx
    except ImportError:
        pass
    return anthropic_ctx, openai_ctx


def _patch_anthropic() -> list[tuple[Any, str, Any]]:
    try:
        import anthropic.resources
    except ImportError:
        return []

    from pretia.collectors.anthropic_sdk import _make_create_wrapper, _make_stream_wrapper

    patches: list[tuple[Any, str, Any]] = []
    for cls_name in ("AsyncMessages", "Messages"):
        cls = getattr(anthropic.resources, cls_name, None)
        if cls is None:
            continue
        is_async = "Async" in cls_name

        original_create = getattr(cls, "create", None)
        if original_create is not None:
            patches.append((cls, "create", original_create))
            cls.create = _make_create_wrapper(original_create, is_async)  # noqa: B010

        original_stream = getattr(cls, "stream", None)
        if original_stream is not None:
            patches.append((cls, "stream", original_stream))
            cls.stream = _make_stream_wrapper(original_stream, is_async)  # noqa: B010

    return patches


def _patch_openai() -> list[tuple[Any, str, Any]]:
    try:
        import openai.resources.chat
    except ImportError:
        return []

    from pretia.collectors.openai_sdk import _make_create_wrapper

    patches: list[tuple[Any, str, Any]] = []
    for cls_name in ("AsyncCompletions", "Completions"):
        target = getattr(openai.resources.chat, cls_name, None)
        if target is None:
            continue
        original_create = getattr(target, "create", None)
        if original_create is None:
            continue
        is_async = "Async" in cls_name
        patches.append((target, "create", original_create))
        target.create = _make_create_wrapper(original_create, is_async)  # noqa: B010

    return patches
