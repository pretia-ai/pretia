"""Collect step-level token usage by monkey-patching the OpenAI SDK."""

from __future__ import annotations

import hashlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pretia.collectors._utils import get_caller_name
from pretia.collectors.base import BaseCollector, StepRecord

logger = logging.getLogger(__name__)


class OpenAISDKCollector(BaseCollector):
    """Intercept ``openai.chat.completions.create`` calls to capture token usage.

    Handles both sync/async clients and streaming responses.
    """

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        try:
            import openai.resources.chat
        except ImportError as exc:
            raise ImportError(
                "OpenAISDKCollector requires the `openai` package. "
                "Install it with: pip install openai"
            ) from exc

        runs: list[list[StepRecord]] = []
        total = len(inputs)

        for i, inp in enumerate(inputs):
            captured: list[StepRecord] = []
            patches: list[tuple[Any, str, Any]] = []

            for cls_name in ("AsyncCompletions", "Completions"):
                target = getattr(openai.resources.chat, cls_name, None)
                if target is None:
                    continue
                original_create = getattr(target, "create", None)
                if original_create is None:
                    continue
                is_async = "Async" in cls_name
                wrapped = _make_create_wrapper(original_create, is_async, captured)
                patches.append((target, "create", original_create))
                target.create = wrapped  # noqa: B010

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


def _make_create_wrapper(
    original: Any,
    is_async: bool,
    captured: list[StepRecord],
) -> Any:
    if is_async:

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            caller = get_caller_name()
            t0 = time.monotonic_ns()
            stream = kwargs.get("stream", False)
            if stream:
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True
            response = await original(*args, **kwargs)
            if stream:
                return _AsyncStreamCapture(response, t0, captured, caller)
            _record_from_response(response, t0, captured, caller)
            return response

        return async_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        caller = get_caller_name()
        t0 = time.monotonic_ns()
        stream = kwargs.get("stream", False)
        if stream:
            kwargs.setdefault("stream_options", {})
            kwargs["stream_options"]["include_usage"] = True
        response = original(*args, **kwargs)
        if stream:
            return _SyncStreamCapture(response, t0, captured, caller)
        _record_from_response(response, t0, captured, caller)
        return response

    return sync_wrapper


class _AsyncStreamCapture:
    """Wrap an OpenAI async stream to capture usage from the final chunk."""

    def __init__(
        self,
        stream: Any,
        t0_ns: int,
        captured: list[StepRecord],
        caller: str,
    ) -> None:
        self._stream = stream
        self._t0_ns = t0_ns
        self._captured = captured
        self._caller = caller

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aiter__(self) -> Any:
        last_chunk = None
        async for chunk in self._stream:
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            _record_from_chunk(last_chunk, self._t0_ns, self._captured, self._caller)


class _SyncStreamCapture:
    """Wrap an OpenAI sync stream to capture usage from the final chunk."""

    def __init__(
        self,
        stream: Any,
        t0_ns: int,
        captured: list[StepRecord],
        caller: str,
    ) -> None:
        self._stream = stream
        self._t0_ns = t0_ns
        self._captured = captured
        self._caller = caller

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __iter__(self) -> Any:
        last_chunk = None
        for chunk in self._stream:
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            _record_from_chunk(last_chunk, self._t0_ns, self._captured, self._caller)


def _record_from_chunk(
    chunk: Any,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str,
) -> None:
    """Extract usage from a streaming chunk (last chunk with include_usage)."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        logger.warning("OpenAI stream completed without usage data.")
        return
    model = getattr(chunk, "model", "unknown") or "unknown"
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    _build_record(model, input_tokens, output_tokens, t0_ns, captured, caller)


def _record_from_response(
    response: Any,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str = "llm_call",
) -> None:
    """Extract token usage from an OpenAI ChatCompletion response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    model = getattr(response, "model", "unknown") or "unknown"
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0

    cache_hit = getattr(usage, "prompt_tokens_details", None)
    cache_hit_tokens = None
    cache_miss_tokens = None
    if cache_hit is not None:
        cached = getattr(cache_hit, "cached_tokens", None)
        if cached is not None and cached > 0:
            cache_hit_tokens = cached
            cache_miss_tokens = max(0, input_tokens - cached)

    _build_record(
        model,
        input_tokens,
        output_tokens,
        t0_ns,
        captured,
        caller,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
    )


def _build_record(
    model: str,
    input_tokens: int,
    output_tokens: int,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
) -> None:
    duration_ms = (time.monotonic_ns() - t0_ns) // 1_000_000
    record = StepRecord(
        step_name=caller,
        step_type="llm",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash=hashlib.sha256(b"").hexdigest(),
        system_prompt_tokens=0,
        output_format="text",
        is_retry=False,
        iteration=1,
        parent_step=None,
        duration_ms=duration_ms,
        timestamp=datetime.now(UTC),
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
    )
    captured.append(record)
