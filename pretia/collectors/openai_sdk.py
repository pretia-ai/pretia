"""Collect step-level token usage by monkey-patching the OpenAI SDK."""

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
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    captured: list[StepRecord],
) -> tuple[str, int]:
    """Determine step name and iteration number for a call.

    Same caller + same model → same step name, incrementing iteration (context growth).
    Same caller + different model → ordinal suffix (step_1, step_2).
    """
    caller = get_caller_name()
    model = kwargs.get("model") or (args[0] if args else None) or ""

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
    """Extract system prompt, tools, and max_tokens from OpenAI kwargs."""
    max_tokens_setting = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")

    system_text = ""
    messages = kwargs.get("messages", [])
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "system":
            content = msg.get("content", "")
            system_text = str(content) if content else ""
            break

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


def _extract_tool_name(response: Any) -> str | None:
    """Extract tool name from an OpenAI ChatCompletion response."""
    choices = getattr(response, "choices", None)
    if not choices:
        return None
    msg = getattr(choices[0], "message", None)
    if msg is None:
        return None
    tool_calls = getattr(msg, "tool_calls", None)
    if not tool_calls:
        return None
    func = getattr(tool_calls[0], "function", None)
    if func is None:
        return None
    return getattr(func, "name", None)


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
            lock = asyncio.Lock()
            patches: list[tuple[Any, str, Any]] = []

            for cls_name in ("AsyncCompletions", "Completions"):
                target = getattr(openai.resources.chat, cls_name, None)
                if target is None:
                    continue
                original_create = getattr(target, "create", None)
                if original_create is None:
                    continue
                is_async = "Async" in cls_name
                wrapped = _make_create_wrapper(original_create, is_async, captured, lock)
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
    lock: asyncio.Lock | None = None,
) -> Any:
    iteration_counters: dict[str, int] = {}

    if is_async:

        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            if lock is not None:
                async with lock:
                    step_name, _ = _step_name_and_iteration(args, kwargs, captured)
                    iteration_counters[step_name] = iteration_counters.get(step_name, 0) + 1
                    iteration = iteration_counters[step_name]
            else:
                step_name, iteration = _step_name_and_iteration(args, kwargs, captured)
            kwargs_meta = _extract_kwargs_metadata(kwargs)
            t0 = time.monotonic_ns()
            stream = kwargs.get("stream", False)
            if stream:
                kwargs.setdefault("stream_options", {})
                kwargs["stream_options"]["include_usage"] = True
            response = await original(*args, **kwargs)
            if stream:
                return _AsyncStreamCapture(
                    response, t0, captured, step_name, kwargs_meta, iteration, lock
                )
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
        step_name, iteration = _step_name_and_iteration(args, kwargs, captured)
        kwargs_meta = _extract_kwargs_metadata(kwargs)
        t0 = time.monotonic_ns()
        stream = kwargs.get("stream", False)
        if stream:
            kwargs.setdefault("stream_options", {})
            kwargs["stream_options"]["include_usage"] = True
        response = original(*args, **kwargs)
        if stream:
            return _SyncStreamCapture(response, t0, captured, step_name, kwargs_meta, iteration)
        _record_from_response(
            response, t0, captured, step_name, iteration, kwargs_meta=kwargs_meta
        )
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

    async def __aiter__(self) -> Any:
        last_chunk = None
        async for chunk in self._stream:
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            if self._lock is not None:
                async with self._lock:
                    _record_from_chunk(
                        last_chunk,
                        self._t0_ns,
                        self._captured,
                        self._caller,
                        kwargs_meta=self._kwargs_meta,
                        iteration=self._iteration,
                    )
            else:
                _record_from_chunk(
                    last_chunk,
                    self._t0_ns,
                    self._captured,
                    self._caller,
                    kwargs_meta=self._kwargs_meta,
                    iteration=self._iteration,
                )


class _SyncStreamCapture:
    """Wrap an OpenAI sync stream to capture usage from the final chunk."""

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

    def __iter__(self) -> Any:
        last_chunk = None
        for chunk in self._stream:
            last_chunk = chunk
            yield chunk
        if last_chunk is not None:
            _record_from_chunk(
                last_chunk,
                self._t0_ns,
                self._captured,
                self._caller,
                kwargs_meta=self._kwargs_meta,
                iteration=self._iteration,
            )


def _record_from_chunk(
    chunk: Any,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str,
    kwargs_meta: dict[str, Any] | None = None,
    iteration: int = 1,
) -> None:
    """Extract usage from a streaming chunk (last chunk with include_usage)."""
    usage = getattr(chunk, "usage", None)
    if usage is None:
        logger.warning("OpenAI stream completed without usage data.")
        return
    model = getattr(chunk, "model", "unknown") or "unknown"
    input_tokens = getattr(usage, "prompt_tokens", 0) or 0
    output_tokens = getattr(usage, "completion_tokens", 0) or 0
    _build_record(
        model,
        input_tokens,
        output_tokens,
        t0_ns,
        captured,
        caller,
        iteration=iteration,
        kwargs_meta=kwargs_meta,
    )


def _record_from_response(
    response: Any,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str = "llm_call",
    iteration: int = 1,
    kwargs_meta: dict[str, Any] | None = None,
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

    tool_name = _extract_tool_name(response)

    _build_record(
        model,
        input_tokens,
        output_tokens,
        t0_ns,
        captured,
        caller,
        iteration=iteration,
        cache_hit_tokens=cache_hit_tokens,
        cache_miss_tokens=cache_miss_tokens,
        kwargs_meta=kwargs_meta,
        tool_name=tool_name,
    )


def _build_record(
    model: str,
    input_tokens: int,
    output_tokens: int,
    t0_ns: int,
    captured: list[StepRecord],
    caller: str,
    iteration: int = 1,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
    kwargs_meta: dict[str, Any] | None = None,
    tool_name: str | None = None,
) -> None:
    meta = kwargs_meta or {}
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
