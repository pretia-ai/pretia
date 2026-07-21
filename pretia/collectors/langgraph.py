"""Collect step-level token usage from LangGraph workflows via LangChain callbacks."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError:
    raise ImportError(
        "LangGraph support requires langchain-core. Install it with: pip install pretia[langgraph]"
    ) from None

from pretia.collectors.base import BaseCollector, StepRecord

logger = logging.getLogger(__name__)

_EMPTY_HASH = hashlib.sha256(b"").hexdigest()

_WRAPPER_NAMES = frozenset(
    {
        "RunnableSequence",
        "RunnableParallel",
        "RunnableBranch",
        "RunnableMap",
        "RunnableLambda",
        "RunnableBinding",
        "RunnableWithFallbacks",
        "RunnablePassthrough",
        "RunnableAssign",
        "ChannelWrite",
        "ChannelRead",
    }
)


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


def _extract_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _get_message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", message.get("type", "")))
    return str(getattr(message, "role", getattr(message, "type", "")))


class PretiaCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler that accumulates StepRecords for one workflow run."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[StepRecord] = []
        self._inflight: dict[UUID, dict[str, Any]] = {}
        self._step_iterations: dict[str, int] = {}
        self._active_chains: dict[UUID, str] = {}
        self._parent_chain: dict[UUID, UUID | None] = {}

    def _next_iteration(self, step_name: str) -> int:
        count = self._step_iterations.get(step_name, 0) + 1
        self._step_iterations[step_name] = count
        return count

    def _find_node_name(self, run_id: UUID | None) -> str | None:
        """Walk up the parent chain to find the nearest graph node name."""
        seen: set[UUID] = set()
        current = run_id
        while current and current not in seen:
            seen.add(current)
            if current in self._active_chains:
                return self._active_chains[current]
            current = self._parent_chain.get(current)
        return None

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self._parent_chain[run_id] = parent_run_id
        serialized = serialized or {}
        name = (
            kwargs.get("name")
            or (metadata or {}).get("langgraph_node")
            or serialized.get("name")
            or (serialized.get("id", [""])[-1] if serialized.get("id") else None)
        )
        if name and name not in _WRAPPER_NAMES:
            self._active_chains[run_id] = name

    def on_chain_end(
        self,
        outputs: dict[str, Any],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        self._parent_chain.pop(run_id, None)
        self._active_chains.pop(run_id, None)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            model = (
                serialized.get("kwargs", {}).get("model_name")
                or serialized.get("kwargs", {}).get("model")
                or (serialized.get("id", [""])[-1] if serialized.get("id") else None)
                or "unknown"
            )

            llm_class_name = (
                serialized.get("name")
                or kwargs.get("name")
                or (serialized.get("id", [""])[-1] if serialized.get("id") else None)
                or "llm_call"
            )
            step_name = self._find_node_name(parent_run_id) or llm_class_name

            flat_messages = [m for batch in messages for m in batch]

            system_prompt = ""
            for msg in flat_messages:
                if _get_message_role(msg) == "system":
                    system_prompt = _extract_message_content(msg)
                    break

            tool_def_tokens = 0
            tools = kwargs.get("invocation_params", {}).get("tools", [])
            if tools:
                tool_def_tokens = _estimate_tokens(str(tools))

            context_size = sum(
                _estimate_tokens(_extract_message_content(m)) for m in flat_messages
            )

            self._inflight[run_id] = {
                "model": model,
                "step_name": step_name,
                "start_ns": time.monotonic_ns(),
                "timestamp": datetime.now(UTC),
                "system_prompt": system_prompt,
                "tool_definitions_tokens": tool_def_tokens,
                "context_size": context_size,
                "parent_run_id": parent_run_id,
            }
        except Exception:
            logger.debug(
                "Failed to process on_chat_model_start for run_id=%s",
                run_id,
                exc_info=True,
            )

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            inflight = self._inflight.pop(run_id, None)
            if inflight is None:
                logger.debug("on_llm_end for unknown run_id=%s (missed start event)", run_id)
                return

            input_tokens, output_tokens = self._extract_tokens(response)

            context_size = inflight["context_size"]
            if input_tokens > 0:
                context_size = input_tokens

            output_text = self._extract_output_text(response)
            output_format = _detect_output_format(output_text) if output_text else "text"

            system_prompt = inflight["system_prompt"]
            prompt_hash = hashlib.sha256(system_prompt.encode()).hexdigest()
            prompt_tokens = _estimate_tokens(system_prompt)

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
                tool_definitions_tokens=inflight["tool_definitions_tokens"],
                system_prompt_hash=prompt_hash,
                system_prompt_tokens=prompt_tokens,
                output_format=output_format,
                is_retry=False,
                iteration=iteration,
                parent_step=None,
                duration_ms=duration_ms,
                timestamp=inflight["timestamp"],
            )
            self.records.append(record)
        except Exception:
            logger.debug("Failed to process on_llm_end for run_id=%s", run_id, exc_info=True)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        try:
            step_name = serialized.get("name") or kwargs.get("name") or "tool_call"
            self._inflight[run_id] = {
                "step_name": step_name,
                "step_type": "tool",
                "start_ns": time.monotonic_ns(),
                "timestamp": datetime.now(UTC),
            }
        except Exception:
            logger.debug("Failed to process on_tool_start for run_id=%s", run_id, exc_info=True)

    def on_tool_end(
        self,
        output: str,
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        try:
            inflight = self._inflight.pop(run_id, None)
            if inflight is None:
                logger.debug("on_tool_end for unknown run_id=%s (missed start event)", run_id)
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
            self.records.append(record)
        except Exception:
            logger.debug("Failed to process on_tool_end for run_id=%s", run_id, exc_info=True)

    @staticmethod
    def _extract_tokens(response: LLMResult) -> tuple[int, int]:
        """Pull input/output token counts from whichever location LangChain put them."""
        input_tokens = 0
        output_tokens = 0

        # Location 0: message.usage_metadata (cross-provider standard in LangChain)
        try:
            generations = getattr(response, "generations", [])
            if generations and generations[0]:
                msg = getattr(generations[0][0], "message", None)
                if msg:
                    usage_meta = getattr(msg, "usage_metadata", None)
                    if usage_meta:
                        inp = getattr(usage_meta, "input_tokens", 0) or 0
                        out = getattr(usage_meta, "output_tokens", 0) or 0
                        if inp or out:
                            return int(inp), int(out)
        except (IndexError, AttributeError, TypeError):
            pass

        # Location 1: response.llm_output["token_usage"] (OpenAI) or ["usage"] (Anthropic)
        llm_output = getattr(response, "llm_output", None) or {}
        token_usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if token_usage:
            input_tokens = token_usage.get("prompt_tokens", 0) or token_usage.get(
                "input_tokens", 0
            )
            output_tokens = token_usage.get("completion_tokens", 0) or token_usage.get(
                "output_tokens", 0
            )
            if input_tokens or output_tokens:
                return int(input_tokens), int(output_tokens)

        # Location 2: response.generations[0][0].generation_info["usage"]
        try:
            generations = getattr(response, "generations", [])
            if generations and generations[0]:
                gen_info = getattr(generations[0][0], "generation_info", None) or {}
                usage = gen_info.get("usage", {})
                if usage:
                    input_tokens = usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0)
                    output_tokens = usage.get("completion_tokens", 0) or usage.get(
                        "output_tokens", 0
                    )
        except (IndexError, AttributeError, TypeError):
            pass

        return int(input_tokens), int(output_tokens)

    @staticmethod
    def _extract_output_text(response: LLMResult) -> str:
        try:
            generations = getattr(response, "generations", [])
            if generations and generations[0]:
                return str(getattr(generations[0][0], "text", ""))
        except (IndexError, AttributeError, TypeError):
            pass
        return ""


class LangGraphCollector(BaseCollector):
    """Auto-instrument LangGraph workflows via LangChain callback injection."""

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        """Run the workflow on each input with an injected callback handler.

        Args:
            workflow: A LangGraph compiled graph (or any object with ainvoke/invoke).
            inputs: Input strings to run the workflow on.
            on_run_complete: Optional callback after each run.
        """
        runs: list[list[StepRecord]] = []
        total = len(inputs)
        for i, inp in enumerate(inputs):
            handler = PretiaCallbackHandler()
            config: dict[str, Any] = {"callbacks": [handler]}
            payload: Any = inp if isinstance(inp, dict) else {"input": inp}

            try:
                if hasattr(workflow, "ainvoke"):
                    await workflow.ainvoke(payload, config=config)
                elif hasattr(workflow, "invoke"):
                    await asyncio.to_thread(workflow.invoke, payload, config=config)
                else:
                    logger.warning("Workflow has neither ainvoke nor invoke — skipping input")
                    runs.append([])
                    continue
            except Exception:
                logger.error(
                    "Run %d/%d failed on input %.80s",
                    i + 1,
                    total,
                    str(inp)[:80],
                    exc_info=True,
                )

            run_records = list(handler.records)
            runs.append(run_records)
            if on_run_complete is not None:
                on_run_complete(i, total, run_records)
        return runs
