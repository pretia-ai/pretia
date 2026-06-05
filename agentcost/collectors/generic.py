"""Collect step-level token usage via manual decorator and context manager instrumentation."""

from __future__ import annotations

import functools
import hashlib
import logging
import time
from datetime import UTC, datetime
from typing import Any

from agentcost.collectors.base import BaseCollector, StepRecord

logger = logging.getLogger(__name__)


class StepTracker:
    """Capture timing and token data for a single instrumented step.

    Works as both an async context manager and a decorator factory.
    Not exported — use via ``GenericCollector.step()``.
    """

    def __init__(
        self,
        collector: GenericCollector,
        name: str,
        step_type: str,
        parent_step: str | None,
    ) -> None:
        self._collector = collector
        self._name = name
        self._step_type = step_type
        self._parent_step = parent_step
        self._start_ns: int = 0
        self._iteration: int = 0
        self._recorded: dict[str, Any] | None = None

    def record_llm_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        context_size: int | None = None,
        tool_definitions_tokens: int = 0,
        system_prompt: str = "",
        output_format: str = "text",
        is_retry: bool = False,
        cache_hit_tokens: int | None = None,
        cache_miss_tokens: int | None = None,
    ) -> None:
        """Store LLM call data to be turned into a StepRecord on context exit."""
        self._recorded = {
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "context_size": context_size if context_size is not None else input_tokens,
            "tool_definitions_tokens": tool_definitions_tokens,
            "system_prompt_hash": hashlib.sha256(system_prompt.encode()).hexdigest(),
            "system_prompt_tokens": len(system_prompt) // 4,
            "output_format": output_format,
            "is_retry": is_retry,
            "cache_hit_tokens": cache_hit_tokens,
            "cache_miss_tokens": cache_miss_tokens,
        }

    async def __aenter__(self) -> StepTracker:
        self._iteration = self._collector._next_iteration(self._name)
        self._start_ns = time.monotonic_ns()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        if self._recorded is None:
            return
        duration_ms = (time.monotonic_ns() - self._start_ns) // 1_000_000
        record = StepRecord(
            step_name=self._name,
            step_type=self._step_type,
            model=self._recorded["model"],
            input_tokens=self._recorded["input_tokens"],
            output_tokens=self._recorded["output_tokens"],
            context_size=self._recorded["context_size"],
            tool_definitions_tokens=self._recorded["tool_definitions_tokens"],
            system_prompt_hash=self._recorded["system_prompt_hash"],
            system_prompt_tokens=self._recorded["system_prompt_tokens"],
            output_format=self._recorded["output_format"],
            is_retry=self._recorded["is_retry"],
            iteration=self._iteration,
            parent_step=self._parent_step,
            duration_ms=duration_ms,
            timestamp=datetime.now(UTC),
            cache_hit_tokens=self._recorded.get("cache_hit_tokens"),
            cache_miss_tokens=self._recorded.get("cache_miss_tokens"),
        )
        self._collector._current_run.append(record)

    def __call__(self, fn: Any) -> Any:
        """Wrap an async function so its return value is auto-extracted for token usage."""

        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            async with self:
                result = await fn(*args, **kwargs)
                if self._recorded is None:
                    _try_extract(self, result)
                return result

        return wrapper


def _try_extract(tracker: StepTracker, result: Any) -> None:
    """Attempt to extract token usage from an LLM response object."""
    try:
        # OpenAI/Anthropic: response.usage (attribute on response object)
        usage = getattr(result, "usage", None)
        if usage is None:
            # Dict-style: response["usage"]
            if isinstance(result, dict):
                usage = result.get("usage")
            if usage is None:
                return

        model = (
            getattr(result, "model", None)
            or (result.get("model") if isinstance(result, dict) else None)
            or "unknown"
        )

        if isinstance(usage, dict):
            # Dict-style: usage["prompt_tokens"] (OpenAI) or usage["input_tokens"] (Anthropic)
            input_tokens = usage.get("prompt_tokens") or usage.get("input_tokens")
            output_tokens = usage.get("completion_tokens") or usage.get("output_tokens")
        else:
            # OpenAI: usage.prompt_tokens / usage.completion_tokens
            # Anthropic: usage.input_tokens / usage.output_tokens
            input_tokens = getattr(usage, "prompt_tokens", None) or getattr(
                usage, "input_tokens", None
            )
            output_tokens = getattr(usage, "completion_tokens", None) or getattr(
                usage, "output_tokens", None
            )

        if input_tokens is None or output_tokens is None:
            return

        if isinstance(usage, dict):
            cache_hit = usage.get("prompt_cache_hit_tokens")
            cache_miss = usage.get("prompt_cache_miss_tokens")
        else:
            cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
            cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)

        tracker.record_llm_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
        )
    except (AttributeError, KeyError, TypeError):
        logger.debug("Could not extract token usage from %s", type(result).__name__)


class GenericCollector(BaseCollector):
    """Manual instrumentation collector using decorator and context manager patterns."""

    def __init__(self) -> None:
        self._runs: list[list[StepRecord]] = []
        self._current_run: list[StepRecord] = []
        self._iteration_counters: dict[str, int] = {}

    def _next_iteration(self, step_name: str) -> int:
        count = self._iteration_counters.get(step_name, 0) + 1
        self._iteration_counters[step_name] = count
        return count

    def step(
        self,
        name: str,
        step_type: str = "llm",
        parent_step: str | None = None,
    ) -> StepTracker:
        """Return a ``StepTracker`` usable as both an async context manager and a decorator."""
        return StepTracker(self, name, step_type, parent_step)

    def new_run(self) -> None:
        """Start tracking a new run, clearing the current step buffer and iteration counters."""
        self._current_run = []
        self._iteration_counters = {}

    def end_run(self) -> list[StepRecord]:
        """Finalize the current run, append it to accumulated runs, and return it."""
        run = list(self._current_run)
        self._runs.append(run)
        self._current_run = []
        self._iteration_counters = {}
        return run

    @property
    def all_runs(self) -> list[list[StepRecord]]:
        """Return all completed runs."""
        return self._runs

    def reset(self) -> None:
        """Clear all accumulated runs, the current run, and iteration counters."""
        self._runs = []
        self._current_run = []
        self._iteration_counters = {}

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
    ) -> list[list[StepRecord]]:
        """Run the workflow on each input and return one StepRecord list per run.

        Assumes the workflow is an async callable that uses ``self.step()`` internally.
        """
        runs: list[list[StepRecord]] = []
        for inp in inputs:
            self.new_run()
            await workflow(inp)
            runs.append(list(self._current_run))
        self._runs = runs
        return runs
