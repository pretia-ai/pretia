"""Collect step-level token usage from Qwen-Agent workflows via LLM client wrapping."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

try:
    import qwen_agent.agent  # noqa: F401
except ImportError:
    raise ImportError("Qwen-Agent not installed. Run: pip install agentcost[qwen]") from None

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


def _extract_system_prompt(agent: Any) -> str:
    return getattr(agent, "system_message", None) or ""


def _extract_agent_name(agent: Any) -> str:
    name = getattr(agent, "name", None)
    if name:
        return name
    return type(agent).__name__


def _estimate_tool_def_tokens(agent: Any) -> int:
    function_map = getattr(agent, "function_map", None)
    if not function_map:
        return 0
    try:
        schemas = [getattr(fn, "function", {}) for fn in function_map.values()]
        return len(json.dumps(schemas)) // 4
    except Exception:
        return 0


def _extract_usage_from_response(response: Any) -> tuple[int, int]:
    """Extract prompt_tokens and completion_tokens from an OpenAI-format response."""
    usage = getattr(response, "usage", None)
    if usage is not None:
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        return prompt, completion
    return 0, 0


def _extract_usage_from_dashscope_message(msg: Any) -> tuple[int, int]:
    """Extract tokens from DashScope response stored in Message.extra."""
    extra = getattr(msg, "extra", None) or {}
    info = extra.get("model_service_info")
    if info is None:
        return 0, 0
    if isinstance(info, dict):
        usage = info.get("usage", {})
        return (
            usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0),
            usage.get("output_tokens", 0) or usage.get("completion_tokens", 0),
        )
    usage = getattr(info, "usage", None)
    if usage is not None:
        return (
            getattr(usage, "input_tokens", 0) or getattr(usage, "prompt_tokens", 0),
            getattr(usage, "output_tokens", 0) or getattr(usage, "completion_tokens", 0),
        )
    return 0, 0


class _CapturedCall:
    """Data captured from one intercepted LLM call."""

    __slots__ = (
        "model",
        "input_tokens",
        "output_tokens",
        "output_text",
        "start_ns",
        "end_ns",
        "timestamp",
        "is_tool_call",
    )

    def __init__(self) -> None:
        self.model: str = ""
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.output_text: str = ""
        self.start_ns: int = 0
        self.end_ns: int = 0
        self.timestamp: datetime = datetime.now(UTC)
        self.is_tool_call: bool = False


class _InstrumentedChatModel:
    """Wraps a Qwen-Agent BaseChatModel to capture per-call token usage.

    Delegates all attribute access to the original model so the agent's behavior
    is unchanged. Intercepts chat() to record timing and token counts.
    """

    def __init__(self, original: Any, captured: list[_CapturedCall]) -> None:
        object.__setattr__(self, "_original", original)
        object.__setattr__(self, "_captured", captured)

    def __getattr__(self, name: str) -> Any:
        return getattr(object.__getattribute__(self, "_original"), name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in ("_original", "_captured"):
            object.__setattr__(self, name, value)
        else:
            setattr(object.__getattribute__(self, "_original"), name, value)

    def chat(
        self,
        messages: Any,
        functions: Any = None,
        stream: bool = True,
        delta_stream: bool = False,
        extra_generate_cfg: Any = None,
    ) -> Any:
        original = object.__getattribute__(self, "_original")
        captured_list = object.__getattribute__(self, "_captured")

        cap = _CapturedCall()
        cap.model = getattr(original, "model", "") or ""
        cap.start_ns = time.monotonic_ns()
        cap.timestamp = datetime.now(UTC)

        try:
            result = original.chat(
                messages=messages,
                functions=functions,
                stream=stream,
                delta_stream=delta_stream,
                extra_generate_cfg=extra_generate_cfg,
            )
        except Exception:
            logger.debug("Instrumented chat() call failed", exc_info=True)
            raise

        if not stream:
            cap.end_ns = time.monotonic_ns()
            self._extract_from_messages(result, cap)
            captured_list.append(cap)
            return result

        return self._wrap_iterator(result, cap, captured_list)

    def _wrap_iterator(
        self,
        iterator: Iterator[Any],
        cap: _CapturedCall,
        captured_list: list[_CapturedCall],
    ) -> Iterator[Any]:
        """Consume a streaming iterator, capture usage from final chunk, re-yield all."""
        last_output: Any = None
        try:
            for chunk in iterator:
                last_output = chunk
                yield chunk
        finally:
            cap.end_ns = time.monotonic_ns()
            if last_output is not None:
                self._extract_from_messages(last_output, cap)
            captured_list.append(cap)

    @staticmethod
    def _extract_from_messages(messages: Any, cap: _CapturedCall) -> None:
        """Extract token usage and output text from a list of Message objects."""
        if not isinstance(messages, list):
            return
        for msg in messages:
            pt, ct = _extract_usage_from_dashscope_message(msg)
            if pt > 0 or ct > 0:
                cap.input_tokens = pt
                cap.output_tokens = ct

            content = ""
            if isinstance(msg, dict):
                content = str(msg.get("content", ""))
                fc = msg.get("function_call")
                if fc:
                    cap.is_tool_call = True
            else:
                content = str(getattr(msg, "content", ""))
                if getattr(msg, "function_call", None):
                    cap.is_tool_call = True
            if content:
                cap.output_text += content


class QwenAgentCollector(BaseCollector):
    """Auto-instrument Qwen-Agent workflows via LLM client wrapping."""

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
    ) -> list[list[StepRecord]]:
        """Run the workflow on each input with an instrumented LLM client.

        Args:
            workflow: A Qwen-Agent Agent instance (or subclass).
            inputs: Input strings to run the workflow on.
        """
        agent_name = _extract_agent_name(workflow)
        system_prompt = _extract_system_prompt(workflow)
        prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
        prompt_tokens = _estimate_tokens(system_prompt)
        tool_def_tokens = _estimate_tool_def_tokens(workflow)

        original_llm = getattr(workflow, "llm", None)
        runs: list[list[StepRecord]] = []

        for inp in inputs:
            captured: list[_CapturedCall] = []

            if original_llm is not None:
                instrumented = _InstrumentedChatModel(original_llm, captured)
                workflow.llm = instrumented  # type: ignore[assignment]

            try:
                messages = [{"role": "user", "content": inp}]
                response_gen = workflow.run(messages)
                for _ in response_gen:
                    pass
            except Exception:
                logger.warning(
                    "Workflow execution failed for input=%r, skipping",
                    inp[:80],
                    exc_info=True,
                )
                runs.append([])
                continue
            finally:
                if original_llm is not None:
                    workflow.llm = original_llm  # type: ignore[assignment]

            steps = self._build_steps(
                captured,
                agent_name=agent_name,
                prompt_hash=prompt_hash,
                prompt_tokens=prompt_tokens,
                tool_def_tokens=tool_def_tokens,
            )
            runs.append(steps)

        return runs

    @staticmethod
    def _build_steps(
        captured: list[_CapturedCall],
        agent_name: str,
        prompt_hash: str,
        prompt_tokens: int,
        tool_def_tokens: int,
    ) -> list[StepRecord]:
        steps: list[StepRecord] = []
        iteration_counts: dict[str, int] = {}

        for cap in captured:
            if cap.is_tool_call:
                step_type = "tool"
                step_name = f"{agent_name}_tool_call"
            else:
                step_type = "llm"
                step_name = agent_name

            count = iteration_counts.get(step_name, 0) + 1
            iteration_counts[step_name] = count

            context_size = cap.input_tokens if cap.input_tokens > 0 else prompt_tokens
            output_format = _detect_output_format(cap.output_text) if cap.output_text else "text"
            duration_ms = (cap.end_ns - cap.start_ns) // 1_000_000 if cap.end_ns > 0 else 0

            try:
                steps.append(
                    StepRecord(
                        step_name=step_name,
                        step_type=step_type,
                        model=cap.model or "",
                        input_tokens=cap.input_tokens,
                        output_tokens=cap.output_tokens,
                        context_size=context_size,
                        tool_definitions_tokens=tool_def_tokens if step_type == "llm" else 0,
                        system_prompt_hash=prompt_hash,
                        system_prompt_tokens=prompt_tokens,
                        output_format=output_format,
                        is_retry=False,
                        iteration=count,
                        parent_step=None,
                        duration_ms=duration_ms,
                        timestamp=cap.timestamp,
                    )
                )
            except Exception:
                logger.debug("Failed to create StepRecord from captured call", exc_info=True)

        return steps
