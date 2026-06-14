"""Collect step-level token usage from OpenAI Agents SDK workflows via RunHooks."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

try:
    from agents import Runner
    from agents.lifecycle import RunHooksBase
except ImportError:
    raise ImportError(
        "OpenAI Agents SDK not installed. Run: pip install agentcost[openai]"
    ) from None

from agentcost.collectors.base import BaseCollector, StepRecord

logger = logging.getLogger(__name__)

_EMPTY_HASH = hashlib.sha256(b"").hexdigest()


def _estimate_tokens(text: str) -> int:
    return len(text) // 4


def _detect_output_format(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith(("{", "[")):
        try:
            json.loads(stripped)
            return "json"
        except (json.JSONDecodeError, ValueError):
            pass
    if "```" in text:
        return "code"
    return "text"


def _extract_model_name(agent: Any) -> str:
    model = getattr(agent, "model", None)
    if model is None:
        return "unknown"
    if isinstance(model, str):
        return model
    return getattr(model, "model", "unknown")


def _extract_agent_name(agent: Any) -> str:
    return getattr(agent, "name", None) or "agent"


def _extract_tool_name(tool: Any) -> str:
    return getattr(tool, "name", None) or "tool_call"


class AgentCostRunHooks(RunHooksBase):  # type: ignore[type-arg]
    """OpenAI Agents SDK hooks that accumulate StepRecords for one workflow run."""

    def __init__(self) -> None:
        self._steps: list[StepRecord] = []
        self._inflight_llm: dict[str, dict[str, Any]] = {}
        self._inflight_tool: dict[str, dict[str, Any]] = {}
        self._iteration_counts: dict[str, int] = {}
        self._current_agent: str | None = None

    def _next_iteration(self, step_name: str) -> int:
        count = self._iteration_counts.get(step_name, 0) + 1
        self._iteration_counts[step_name] = count
        return count

    def reset(self) -> None:
        """Clear all state between profiling runs."""
        self._steps.clear()
        self._inflight_llm.clear()
        self._inflight_tool.clear()
        self._iteration_counts.clear()
        self._current_agent = None

    @property
    def steps(self) -> list[StepRecord]:
        """Return a copy of the accumulated StepRecords."""
        return list(self._steps)

    async def on_agent_start(
        self,
        context: Any,
        agent: Any,
    ) -> None:
        try:
            self._current_agent = _extract_agent_name(agent)
        except Exception:
            logger.debug("Failed to process on_agent_start", exc_info=True)

    async def on_agent_end(
        self,
        context: Any,
        agent: Any,
        output: Any,
    ) -> None:
        try:
            self._current_agent = None
        except Exception:
            logger.debug("Failed to process on_agent_end", exc_info=True)

    async def on_llm_start(
        self,
        context: Any,
        agent: Any,
        system_prompt: str | None,
        input_items: list[Any],
    ) -> None:
        try:
            agent_name = _extract_agent_name(agent)
            model = _extract_model_name(agent)
            prompt = system_prompt or ""
            prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
            prompt_tokens = _estimate_tokens(prompt)

            context_tokens = prompt_tokens
            for item in input_items:
                content = ""
                if isinstance(item, dict):
                    content = str(item.get("content", ""))
                else:
                    content = str(getattr(item, "content", ""))
                context_tokens += _estimate_tokens(content)

            self._inflight_llm[agent_name] = {
                "model": model,
                "step_name": agent_name,
                "start_ns": time.monotonic_ns(),
                "timestamp": datetime.now(UTC),
                "system_prompt_hash": prompt_hash,
                "system_prompt_tokens": prompt_tokens,
                "context_size": context_tokens,
            }
        except Exception:
            logger.debug("Failed to process on_llm_start", exc_info=True)

    async def on_llm_end(
        self,
        context: Any,
        agent: Any,
        response: Any,
    ) -> None:
        try:
            agent_name = _extract_agent_name(agent)
            inflight = self._inflight_llm.pop(agent_name, None)
            if inflight is None:
                logger.debug(
                    "on_llm_end for unknown agent=%s (missed start event)",
                    agent_name,
                )
                return

            input_tokens = 0
            output_tokens = 0
            usage = getattr(response, "usage", None)
            if usage is not None:
                input_tokens = getattr(usage, "input_tokens", 0) or 0
                output_tokens = getattr(usage, "output_tokens", 0) or 0

            context_size = inflight["context_size"]
            if input_tokens > 0:
                context_size = input_tokens

            output_text = ""
            output_items = getattr(response, "output", []) or []
            for item in output_items:
                text = getattr(item, "text", None)
                if text:
                    output_text += str(text)
                content_parts = getattr(item, "content", None)
                if isinstance(content_parts, list):
                    for part in content_parts:
                        t = getattr(part, "text", None)
                        if t:
                            output_text += str(t)

            output_format = _detect_output_format(output_text) if output_text else "text"

            duration_ms = (time.monotonic_ns() - inflight["start_ns"]) // 1_000_000
            step_name = inflight["step_name"]
            iteration = self._next_iteration(step_name)

            record = StepRecord(
                step_name=step_name,
                step_type="llm",
                model=inflight["model"],
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_size=context_size,
                tool_definitions_tokens=0,
                system_prompt_hash=inflight["system_prompt_hash"],
                system_prompt_tokens=inflight["system_prompt_tokens"],
                output_format=output_format,
                is_retry=False,
                iteration=iteration,
                parent_step=None,
                duration_ms=duration_ms,
                timestamp=inflight["timestamp"],
            )
            self._steps.append(record)
        except Exception:
            logger.debug("Failed to process on_llm_end", exc_info=True)

    async def on_tool_start(
        self,
        context: Any,
        agent: Any,
        tool: Any,
    ) -> None:
        try:
            tool_name = _extract_tool_name(tool)
            self._inflight_tool[tool_name] = {
                "step_name": tool_name,
                "start_ns": time.monotonic_ns(),
                "timestamp": datetime.now(UTC),
            }
        except Exception:
            logger.debug("Failed to process on_tool_start", exc_info=True)

    async def on_tool_end(
        self,
        context: Any,
        agent: Any,
        tool: Any,
        result: str,
    ) -> None:
        try:
            tool_name = _extract_tool_name(tool)
            inflight = self._inflight_tool.pop(tool_name, None)
            if inflight is None:
                logger.debug(
                    "on_tool_end for unknown tool=%s (missed start event)",
                    tool_name,
                )
                return

            duration_ms = (time.monotonic_ns() - inflight["start_ns"]) // 1_000_000
            step_name = inflight["step_name"]
            iteration = self._next_iteration(step_name)

            record = StepRecord(
                step_name=step_name,
                step_type="tool",
                model="",
                input_tokens=0,
                output_tokens=0,
                context_size=0,
                tool_definitions_tokens=0,
                system_prompt_hash=_EMPTY_HASH,
                system_prompt_tokens=0,
                output_format="text",
                is_retry=False,
                iteration=iteration,
                parent_step=None,
                duration_ms=duration_ms,
                timestamp=inflight["timestamp"],
            )
            self._steps.append(record)
        except Exception:
            logger.debug("Failed to process on_tool_end", exc_info=True)

    async def on_handoff(
        self,
        context: Any,
        from_agent: Any,
        to_agent: Any,
    ) -> None:
        try:
            target_name = _extract_agent_name(to_agent)
            step_name = f"handoff_{target_name}"
            iteration = self._next_iteration(step_name)

            record = StepRecord(
                step_name=step_name,
                step_type="tool",
                model="",
                input_tokens=0,
                output_tokens=0,
                context_size=0,
                tool_definitions_tokens=0,
                system_prompt_hash=_EMPTY_HASH,
                system_prompt_tokens=0,
                output_format="text",
                is_retry=False,
                iteration=iteration,
                parent_step=None,
                duration_ms=0,
                timestamp=datetime.now(UTC),
            )
            self._steps.append(record)
        except Exception:
            logger.debug("Failed to process on_handoff", exc_info=True)


def _build_fallback_steps(
    raw_responses: list[Any],
    agent_name: str,
    model: str,
) -> list[StepRecord]:
    """Build StepRecords from RunResult.raw_responses when hooks captured nothing."""
    steps: list[StepRecord] = []
    for i, resp in enumerate(raw_responses):
        usage = getattr(resp, "usage", None)
        if usage is None:
            continue
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        if input_tokens == 0 and output_tokens == 0:
            continue

        steps.append(
            StepRecord(
                step_name=agent_name,
                step_type="llm",
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                context_size=input_tokens,
                tool_definitions_tokens=0,
                system_prompt_hash=_EMPTY_HASH,
                system_prompt_tokens=0,
                output_format="text",
                is_retry=False,
                iteration=i + 1,
                parent_step=None,
                duration_ms=0,
                timestamp=datetime.now(UTC),
            )
        )
    return steps


class OpenAIAgentsCollector(BaseCollector):
    """Auto-instrument OpenAI Agents SDK workflows via RunHooks injection."""

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        """Run the workflow on each input with injected hooks.

        Args:
            workflow: An OpenAI Agents SDK Agent instance.
            inputs: Input strings to run the workflow on.
            on_run_complete: Optional callback after each run.
        """
        runs: list[list[StepRecord]] = []
        total = len(inputs)
        for i, inp in enumerate(inputs):
            hooks = AgentCostRunHooks()

            try:
                result = await Runner.run(workflow, inp, hooks=hooks)
            except Exception:
                logger.warning(
                    "Workflow execution failed for input=%r, skipping",
                    inp[:80],
                    exc_info=True,
                )
                runs.append([])
                continue

            steps = hooks.steps

            if not steps:
                agent_name = _extract_agent_name(workflow)
                model = _extract_model_name(workflow)
                raw_responses = getattr(result, "raw_responses", []) or []
                steps = _build_fallback_steps(raw_responses, agent_name, model)

            runs.append(steps)
            if on_run_complete is not None:
                on_run_complete(i, total, steps)
        return runs
