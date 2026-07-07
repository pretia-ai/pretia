"""Collect step-level token usage by monkey-patching the Anthropic SDK."""

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


class AnthropicCollector(BaseCollector):
    """Intercept ``anthropic.messages.create`` calls to capture token usage.

    Patches both sync/async create and stream methods at the class level.
    """

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        try:
            import anthropic.resources
        except ImportError as exc:
            raise ImportError(
                "AnthropicCollector requires the `anthropic` package. "
                "Install it with: pip install anthropic"
            ) from exc

        runs: list[list[StepRecord]] = []
        total = len(inputs)

        for i, inp in enumerate(inputs):
            captured: list[StepRecord] = []
            patches: list[tuple[Any, str, Any]] = []

            for cls_name in ("AsyncMessages", "Messages"):
                cls = getattr(anthropic.resources, cls_name, None)
                if cls is None:
                    continue
                is_async = "Async" in cls_name

                original_create = getattr(cls, "create", None)
                if original_create is not None:
                    wrapped = _make_create_wrapper(original_create, is_async, captured)
                    patches.append((cls, "create", original_create))
                    cls.create = wrapped  # noqa: B010

                original_stream = getattr(cls, "stream", None)
                if original_stream is not None:
                    wrapped_stream = _make_stream_wrapper(original_stream, is_async, captured)
                    patches.append((cls, "stream", original_stream))
                    cls.stream = wrapped_stream  # noqa: B010

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
            response = await original(*args, **kwargs)
            _record_from_response(response, t0, captured, caller)
            return response

        return async_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        caller = get_caller_name()
        t0 = time.monotonic_ns()
        response = original(*args, **kwargs)
        _record_from_response(response, t0, captured, caller)
        return response

    return sync_wrapper


def _make_stream_wrapper(
    original: Any,
    is_async: bool,
    captured: list[StepRecord],
) -> Any:
    if is_async:

        async def async_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
            caller = get_caller_name()
            t0 = time.monotonic_ns()
            stream = await original(*args, **kwargs)
            return _AsyncStreamCapture(stream, t0, captured, caller)

        return async_stream_wrapper

    def sync_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
        caller = get_caller_name()
        t0 = time.monotonic_ns()
        stream = original(*args, **kwargs)
        return _SyncStreamCapture(stream, t0, captured, caller)

    return sync_stream_wrapper


class _AsyncStreamCapture:
    """Wrap an Anthropic async stream to capture usage from the final message."""

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

    async def __aenter__(self) -> Any:
        result = await self._stream.__aenter__()
        return _AsyncStreamCapture(result, self._t0_ns, self._captured, self._caller)

    async def __aexit__(self, *args: Any) -> Any:
        result = await self._stream.__aexit__(*args)
        final_message = getattr(self._stream, "final_message", None)
        if final_message is None:
            final_message = getattr(self._stream, "response", None)
        if final_message is not None:
            _record_from_response(final_message, self._t0_ns, self._captured, self._caller)
        else:
            logger.warning("Anthropic stream completed without usage data.")
        return result

    async def get_final_message(self) -> Any:
        msg = await self._stream.get_final_message()
        _record_from_response(msg, self._t0_ns, self._captured, self._caller)
        return msg


class _SyncStreamCapture:
    """Wrap an Anthropic sync stream to capture usage from the final message."""

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

    def __enter__(self) -> Any:
        result = self._stream.__enter__()
        return _SyncStreamCapture(result, self._t0_ns, self._captured, self._caller)

    def __exit__(self, *args: Any) -> Any:
        result = self._stream.__exit__(*args)
        final_message = getattr(self._stream, "final_message", None)
        if final_message is None:
            final_message = getattr(self._stream, "response", None)
        if final_message is not None:
            _record_from_response(final_message, self._t0_ns, self._captured, self._caller)
        else:
            logger.warning("Anthropic stream completed without usage data.")
        return result

    def get_final_message(self) -> Any:
        msg = self._stream.get_final_message()
        _record_from_response(msg, self._t0_ns, self._captured, self._caller)
        return msg


def _record_from_response(
    response: Any,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str = "llm_call",
) -> None:
    """Extract token usage from an Anthropic Message response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    model = getattr(response, "model", "unknown") or "unknown"
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0

    cache_read = getattr(usage, "cache_read_input_tokens", None)
    cache_creation = getattr(usage, "cache_creation_input_tokens", None)

    cache_hit_tokens = None
    cache_miss_tokens = None
    if cache_read is not None:
        cache_hit_tokens = cache_read
        cache_miss_tokens = input_tokens - cache_read
        if cache_creation:
            cache_miss_tokens = max(0, cache_miss_tokens - cache_creation)

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
