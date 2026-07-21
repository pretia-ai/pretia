"""Collect step-level token usage by monkey-patching the Anthropic SDK."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from pretia.collectors._utils import get_caller_name
from pretia.collectors.base import BaseCollector, StepRecord

logger = logging.getLogger(__name__)


def _models_match(a: str, b: str) -> bool:
    """Compare models by resolved canonical name, falling back to exact match."""
    from pretia.pricing.tables import UnrecognizedModelError, resolve_model

    try:
        return resolve_model(a) == resolve_model(b)
    except UnrecognizedModelError:
        return a == b


def _step_name_and_iteration(
    kwargs: dict[str, Any],
    captured: list[StepRecord],
) -> tuple[str, int]:
    """Determine step name and iteration for a call."""
    caller = get_caller_name()
    model = kwargs.get("model", "")

    if not isinstance(model, str) or not model:
        iteration = sum(1 for r in captured if r.step_name == caller) + 1
        return caller, iteration

    existing = [(r.step_name, r.model) for r in captured]
    caller_models = {m for s, m in existing if s == caller or s.startswith(f"{caller}_")}

    if not caller_models:
        return caller, 1

    if len(caller_models) == 1 and any(_models_match(model, m) for m in caller_models):
        iteration = sum(1 for r in captured if r.step_name == caller) + 1
        return caller, iteration

    ordinal = len(caller_models) + 1
    name = f"{caller}_step_{ordinal}"
    iteration = sum(1 for r in captured if r.step_name == name) + 1
    return name, iteration


def _extract_kwargs_metadata(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Extract system prompt, tools, and max_tokens from Anthropic kwargs."""
    max_tokens_setting = kwargs.get("max_tokens")

    system = kwargs.get("system", "")
    if isinstance(system, list):
        system_text = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in system
        )
    else:
        system_text = str(system) if system else ""

    system_prompt_hash = hashlib.sha256(system_text.encode()).hexdigest()
    system_prompt_tokens = int(len(system_text.split()) * 1.3) if system_text else 0
    system_prompt_snippet = system_text[:200] if system_text else None

    tools = kwargs.get("tools")
    tool_definitions_tokens = len(str(tools)) // 4 if tools else 0

    return {
        "max_tokens_setting": max_tokens_setting,
        "system_prompt_hash": system_prompt_hash,
        "system_prompt_tokens": system_prompt_tokens,
        "system_prompt_snippet": system_prompt_snippet,
        "tool_definitions_tokens": tool_definitions_tokens,
    }


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
            lock = asyncio.Lock()
            patches: list[tuple[Any, str, Any]] = []

            for cls_name in ("AsyncMessages", "Messages"):
                cls = getattr(anthropic.resources, cls_name, None)
                if cls is None:
                    continue
                is_async = "Async" in cls_name

                original_create = getattr(cls, "create", None)
                if original_create is not None:
                    wrapped = _make_create_wrapper(original_create, is_async, captured, lock)
                    patches.append((cls, "create", original_create))
                    cls.create = wrapped  # noqa: B010

                original_stream = getattr(cls, "stream", None)
                if original_stream is not None:
                    wrapped_stream = _make_stream_wrapper(
                        original_stream, is_async, captured, lock
                    )
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
    lock: asyncio.Lock | None = None,
) -> Any:
    iteration_counters: dict[str, int] = {}

    if is_async:

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if lock is not None:
                async with lock:
                    step_name, _ = _step_name_and_iteration(kwargs, captured)
                    iteration_counters[step_name] = iteration_counters.get(step_name, 0) + 1
                    iteration = iteration_counters[step_name]
            else:
                step_name, iteration = _step_name_and_iteration(kwargs, captured)
            kwargs_meta = _extract_kwargs_metadata(kwargs)
            t0 = time.monotonic_ns()
            response = await original(*args, **kwargs)
            if lock is not None:
                async with lock:
                    _record_from_response(
                        response, t0, captured, step_name, iteration, kwargs_meta=kwargs_meta
                    )
            else:
                _record_from_response(
                    response, t0, captured, step_name, iteration, kwargs_meta=kwargs_meta
                )
            return response

        return async_wrapper

    def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
        step_name, iteration = _step_name_and_iteration(kwargs, captured)
        kwargs_meta = _extract_kwargs_metadata(kwargs)
        t0 = time.monotonic_ns()
        response = original(*args, **kwargs)
        _record_from_response(
            response, t0, captured, step_name, iteration, kwargs_meta=kwargs_meta
        )
        return response

    return sync_wrapper


def _make_stream_wrapper(
    original: Any,
    is_async: bool,
    captured: list[StepRecord],
    lock: asyncio.Lock | None = None,
) -> Any:
    iteration_counters: dict[str, int] = {}

    if is_async:

        async def async_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
            if lock is not None:
                async with lock:
                    step_name, _ = _step_name_and_iteration(kwargs, captured)
                    iteration_counters[step_name] = iteration_counters.get(step_name, 0) + 1
                    iteration = iteration_counters[step_name]
            else:
                step_name, iteration = _step_name_and_iteration(kwargs, captured)
            kwargs_meta = _extract_kwargs_metadata(kwargs)
            t0 = time.monotonic_ns()
            stream = await original(*args, **kwargs)
            return _AsyncStreamCapture(
                stream, t0, captured, step_name, kwargs_meta, iteration, lock
            )

        return async_stream_wrapper

    def sync_stream_wrapper(*args: Any, **kwargs: Any) -> Any:
        step_name, iteration = _step_name_and_iteration(kwargs, captured)
        kwargs_meta = _extract_kwargs_metadata(kwargs)
        t0 = time.monotonic_ns()
        stream = original(*args, **kwargs)
        return _SyncStreamCapture(stream, t0, captured, step_name, kwargs_meta, iteration)

    return sync_stream_wrapper


class _AsyncStreamCapture:
    """Wrap an Anthropic async stream to capture usage from the final message."""

    def __init__(
        self,
        stream: Any,
        t0_ns: int,
        captured: list[StepRecord],
        caller: str,
        kwargs_meta: dict[str, Any] | None = None,
        iteration: int = 1,
        lock: asyncio.Lock | None = None,
    ) -> None:
        self._stream = stream
        self._t0_ns = t0_ns
        self._captured = captured
        self._caller = caller
        self._kwargs_meta = kwargs_meta
        self._iteration = iteration
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    async def __aenter__(self) -> Any:
        result = await self._stream.__aenter__()
        return _AsyncStreamCapture(
            result,
            self._t0_ns,
            self._captured,
            self._caller,
            self._kwargs_meta,
            self._iteration,
            self._lock,
        )

    async def __aexit__(self, *args: Any) -> Any:
        result = await self._stream.__aexit__(*args)
        final_message = getattr(self._stream, "final_message", None)
        if final_message is None:
            final_message = getattr(self._stream, "response", None)
        if final_message is not None:
            if self._lock is not None:
                async with self._lock:
                    _record_from_response(
                        final_message,
                        self._t0_ns,
                        self._captured,
                        self._caller,
                        self._iteration,
                        kwargs_meta=self._kwargs_meta,
                    )
            else:
                _record_from_response(
                    final_message,
                    self._t0_ns,
                    self._captured,
                    self._caller,
                    self._iteration,
                    kwargs_meta=self._kwargs_meta,
                )
        else:
            logger.warning("Anthropic stream completed without usage data.")
        return result

    async def get_final_message(self) -> Any:
        msg = await self._stream.get_final_message()
        if self._lock is not None:
            async with self._lock:
                _record_from_response(
                    msg,
                    self._t0_ns,
                    self._captured,
                    self._caller,
                    self._iteration,
                    kwargs_meta=self._kwargs_meta,
                )
        else:
            _record_from_response(
                msg,
                self._t0_ns,
                self._captured,
                self._caller,
                self._iteration,
                kwargs_meta=self._kwargs_meta,
            )
        return msg


class _SyncStreamCapture:
    """Wrap an Anthropic sync stream to capture usage from the final message."""

    def __init__(
        self,
        stream: Any,
        t0_ns: int,
        captured: list[StepRecord],
        caller: str,
        kwargs_meta: dict[str, Any] | None = None,
        iteration: int = 1,
    ) -> None:
        self._stream = stream
        self._t0_ns = t0_ns
        self._captured = captured
        self._caller = caller
        self._kwargs_meta = kwargs_meta
        self._iteration = iteration

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)

    def __enter__(self) -> Any:
        result = self._stream.__enter__()
        return _SyncStreamCapture(
            result,
            self._t0_ns,
            self._captured,
            self._caller,
            self._kwargs_meta,
            self._iteration,
        )

    def __exit__(self, *args: Any) -> Any:
        result = self._stream.__exit__(*args)
        final_message = getattr(self._stream, "final_message", None)
        if final_message is None:
            final_message = getattr(self._stream, "response", None)
        if final_message is not None:
            _record_from_response(
                final_message,
                self._t0_ns,
                self._captured,
                self._caller,
                self._iteration,
                kwargs_meta=self._kwargs_meta,
            )
        else:
            logger.warning("Anthropic stream completed without usage data.")
        return result

    def get_final_message(self) -> Any:
        msg = self._stream.get_final_message()
        _record_from_response(
            msg,
            self._t0_ns,
            self._captured,
            self._caller,
            self._iteration,
            kwargs_meta=self._kwargs_meta,
        )
        return msg


def _record_from_response(
    response: Any,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str = "llm_call",
    iteration: int = 1,
    kwargs_meta: dict[str, Any] | None = None,
) -> None:
    """Extract token usage from an Anthropic Message response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return

    meta = kwargs_meta or {}

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

    tool_name = None
    content = getattr(response, "content", None)
    if content:
        for block in content:
            if getattr(block, "type", None) == "tool_use":
                tool_name = getattr(block, "name", None)
                break

    duration_ms = (time.monotonic_ns() - t0_ns) // 1_000_000

    record = StepRecord(
        step_name=caller,
        step_type="llm",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=input_tokens,
        tool_definitions_tokens=meta.get("tool_definitions_tokens", 0),
        system_prompt_hash=meta.get("system_prompt_hash", hashlib.sha256(b"").hexdigest()),
        system_prompt_tokens=meta.get("system_prompt_tokens", 0),
        output_format="text",
        is_retry=False,
        iteration=iteration,
        parent_step=None,
        duration_ms=duration_ms,
        timestamp=datetime.now(UTC),
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
        max_tokens_setting=meta.get("max_tokens_setting"),
        system_prompt_snippet=meta.get("system_prompt_snippet"),
        tool_name=tool_name,
    )
    captured.append(record)
