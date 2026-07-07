"""Collect step-level token usage by patching both Anthropic and OpenAI SDKs."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from pretia.collectors.base import BaseCollector, StepRecord

logger = logging.getLogger(__name__)


class MultiSDKCollector(BaseCollector):
    """Patch both Anthropic and OpenAI SDK classes simultaneously.

    Used when a workflow imports both SDKs. Patches are applied per-run
    so each run's captured list is correctly scoped.
    """

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        runs: list[list[StepRecord]] = []
        total = len(inputs)

        for i, inp in enumerate(inputs):
            captured: list[StepRecord] = []
            patches: list[tuple[Any, str, Any]] = []

            patches.extend(_patch_anthropic(captured))
            patches.extend(_patch_openai(captured))

            if not patches:
                logger.warning(
                    "MultiSDKCollector could not patch any SDK. Install anthropic or openai."
                )

            try:
                await workflow(inp)
            except Exception:
                logger.error(
                    "Run %d/%d failed on input %.80s",
                    i + 1,
                    total,
                    inp,
                    exc_info=True,
                )
            finally:
                for target, attr, original in patches:
                    setattr(target, attr, original)

            if not captured:
                logger.warning("Run %d produced 0 steps.", i + 1)

            runs.append(captured)
            try:
                if on_run_complete is not None:
                    on_run_complete(i, total, captured)
            except Exception:
                logger.debug("on_run_complete callback failed", exc_info=True)

        return runs


def _patch_anthropic(captured: list[StepRecord]) -> list[tuple[Any, str, Any]]:
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
            cls.create = _make_create_wrapper(original_create, is_async, captured)  # noqa: B010

        original_stream = getattr(cls, "stream", None)
        if original_stream is not None:
            patches.append((cls, "stream", original_stream))
            cls.stream = _make_stream_wrapper(original_stream, is_async, captured)  # noqa: B010

    return patches


def _patch_openai(captured: list[StepRecord]) -> list[tuple[Any, str, Any]]:
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
        target.create = _make_create_wrapper(original_create, is_async, captured)  # noqa: B010

    return patches
