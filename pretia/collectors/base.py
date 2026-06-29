"""Define the StepRecord dataclass and BaseCollector interface."""

from __future__ import annotations

import asyncio
import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

_VALID_STEP_TYPES = frozenset({"llm", "tool", "retrieval"})
_VALID_OUTPUT_FORMATS = frozenset({"json", "text", "code"})
_NON_NEGATIVE_FIELDS = ("input_tokens", "output_tokens", "context_size", "duration_ms")


@dataclass(frozen=True, slots=True)
class StepRecord:
    """One LLM call or tool invocation captured during a workflow run."""

    step_name: str
    step_type: str
    model: str
    input_tokens: int
    output_tokens: int
    context_size: int
    tool_definitions_tokens: int
    system_prompt_hash: str
    system_prompt_tokens: int
    output_format: str
    is_retry: bool
    iteration: int
    parent_step: str | None
    duration_ms: int
    timestamp: datetime
    cache_hit_tokens: int | None = None
    cache_miss_tokens: int | None = None
    # v2 schema additions — optional, backward compatible
    tool_name: str | None = None
    tool_input_tokens: int | None = None
    tool_output_tokens: int | None = None
    tool_success: bool | None = None
    tool_retry_count: int | None = None
    model_version: str | None = None
    temperature: float | None = None
    max_tokens_setting: int | None = None
    output_truncated: bool | None = None
    output_tool_call_count: int | None = None
    step_output_format: str | None = None

    def __post_init__(self) -> None:
        if self.step_type not in _VALID_STEP_TYPES:
            raise ValueError(
                f"step_type must be one of {sorted(_VALID_STEP_TYPES)}, got {self.step_type!r}"
            )
        if self.output_format not in _VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {sorted(_VALID_OUTPUT_FORMATS)}, "
                f"got {self.output_format!r}"
            )
        for name in _NON_NEGATIVE_FIELDS:
            value: int = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.iteration < 1:
            raise ValueError(f"iteration must be >= 1, got {self.iteration}")

    @property
    def total_tokens(self) -> int:
        """Return input_tokens + output_tokens."""
        return self.input_tokens + self.output_tokens

    def cost(self, pricing: dict[str, tuple[float, float]]) -> float:
        """Compute dollar cost from per-token pricing.

        Args:
            pricing: Maps model name to (input_price_per_token, output_price_per_token).

        Returns:
            Dollar cost accounting for cache tokens when available.

        Raises:
            ValueError: If this record's model has no entry in `pricing`.
        """
        if self.model not in pricing:
            raise ValueError(
                f"No pricing for model {self.model!r}. Available models: {sorted(pricing)}"
            )
        input_price, output_price = pricing[self.model]
        if self.cache_hit_tokens is not None and self.cache_miss_tokens is not None:
            from pretia.pricing.tables import _PER_MILLION, MODEL_CACHE_HIT_PRICING

            cache_hit_rate = MODEL_CACHE_HIT_PRICING.get(self.model)
            if cache_hit_rate is not None:
                input_cost = self.cache_miss_tokens * input_price + self.cache_hit_tokens * (
                    cache_hit_rate / _PER_MILLION
                )
                return input_cost + self.output_tokens * output_price
        return self.input_tokens * input_price + self.output_tokens * output_price

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict (timestamp as ISO 8601 string)."""
        return {
            "step_name": self.step_name,
            "step_type": self.step_type,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "context_size": self.context_size,
            "tool_definitions_tokens": self.tool_definitions_tokens,
            "system_prompt_hash": self.system_prompt_hash,
            "system_prompt_tokens": self.system_prompt_tokens,
            "output_format": self.output_format,
            "is_retry": self.is_retry,
            "iteration": self.iteration,
            "parent_step": self.parent_step,
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp.isoformat(),
            "cache_hit_tokens": self.cache_hit_tokens,
            "cache_miss_tokens": self.cache_miss_tokens,
            "tool_name": self.tool_name,
            "tool_input_tokens": self.tool_input_tokens,
            "tool_output_tokens": self.tool_output_tokens,
            "tool_success": self.tool_success,
            "tool_retry_count": self.tool_retry_count,
            "model_version": self.model_version,
            "temperature": self.temperature,
            "max_tokens_setting": self.max_tokens_setting,
            "output_truncated": self.output_truncated,
            "output_tool_call_count": self.output_tool_call_count,
            "step_output_format": self.step_output_format,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepRecord:
        """Deserialize a StepRecord from a dict produced by `to_dict()`."""
        return cls(
            step_name=data["step_name"],
            step_type=data["step_type"],
            model=data["model"],
            input_tokens=data["input_tokens"],
            output_tokens=data["output_tokens"],
            context_size=data["context_size"],
            tool_definitions_tokens=data["tool_definitions_tokens"],
            system_prompt_hash=data["system_prompt_hash"],
            system_prompt_tokens=data["system_prompt_tokens"],
            output_format=data["output_format"],
            is_retry=data["is_retry"],
            iteration=data["iteration"],
            parent_step=data["parent_step"],
            duration_ms=data["duration_ms"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            cache_hit_tokens=data.get("cache_hit_tokens"),
            cache_miss_tokens=data.get("cache_miss_tokens"),
            tool_name=data.get("tool_name"),
            tool_input_tokens=data.get("tool_input_tokens"),
            tool_output_tokens=data.get("tool_output_tokens"),
            tool_success=data.get("tool_success"),
            tool_retry_count=data.get("tool_retry_count"),
            model_version=data.get("model_version"),
            temperature=data.get("temperature"),
            max_tokens_setting=data.get("max_tokens_setting"),
            output_truncated=data.get("output_truncated"),
            output_tool_call_count=data.get("output_tool_call_count"),
            step_output_format=data.get("step_output_format"),
        )


class BaseCollector(ABC):
    """Interface that every framework-specific step collector implements."""

    @abstractmethod
    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        """Run the workflow on each input and return one StepRecord list per run.

        Args:
            workflow: The agent workflow object (framework-specific).
            inputs: Input strings to run the workflow on.
            on_run_complete: Optional callback invoked after each run with
                (run_index, total_runs, records).

        Returns:
            One list of StepRecords per input, in the same order as `inputs`.
        """
        ...

    def collect_sync(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Callable[[int, int, list[StepRecord]], None] | None = None,
    ) -> list[list[StepRecord]]:
        """Run `collect()` to completion synchronously. Convenience wrapper for CLI use."""
        return asyncio.run(self.collect(workflow, inputs, on_run_complete))


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Aggregate metadata for a single profiling run."""

    run_id: str
    steps: list[StepRecord]
    total_cost: float
    started_at: str
    ended_at: str
    active_step_list: list[str] = field(default_factory=list)
    step_execution_order: list[str] = field(default_factory=list)
    loop_exit_reason: str | None = None
    total_tool_calls: int = 0
    input_complexity_tier: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "run_id": self.run_id,
            "steps": [s.to_dict() for s in self.steps],
            "total_cost": self.total_cost,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "active_step_list": list(self.active_step_list),
            "step_execution_order": list(self.step_execution_order),
            "loop_exit_reason": self.loop_exit_reason,
            "total_tool_calls": self.total_tool_calls,
            "input_complexity_tier": self.input_complexity_tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunRecord:
        """Deserialize from a dict produced by `to_dict()`."""
        return cls(
            run_id=data["run_id"],
            steps=[StepRecord.from_dict(s) for s in data["steps"]],
            total_cost=data["total_cost"],
            started_at=data["started_at"],
            ended_at=data["ended_at"],
            active_step_list=data.get("active_step_list", []),
            step_execution_order=data.get("step_execution_order", []),
            loop_exit_reason=data.get("loop_exit_reason"),
            total_tool_calls=data.get("total_tool_calls", 0),
            input_complexity_tier=data.get("input_complexity_tier"),
        )


@dataclass(frozen=True, slots=True)
class WorkflowRecord:
    """Structural metadata about a workflow."""

    workflow_fingerprint: str | None = None
    fingerprint_version: int = 1
    graph_adjacency_list: dict[str, list[str]] | None = None
    graph_edge_types: dict[str, str] | None = None
    step_model_map: dict[str, str] | None = None
    total_step_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow_fingerprint": self.workflow_fingerprint,
            "fingerprint_version": self.fingerprint_version,
            "graph_adjacency_list": self.graph_adjacency_list,
            "graph_edge_types": self.graph_edge_types,
            "step_model_map": self.step_model_map,
            "total_step_count": self.total_step_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WorkflowRecord:
        """Deserialize from a dict produced by `to_dict()`."""
        return cls(
            workflow_fingerprint=data.get("workflow_fingerprint"),
            fingerprint_version=data.get("fingerprint_version", 1),
            graph_adjacency_list=data.get("graph_adjacency_list"),
            graph_edge_types=data.get("graph_edge_types"),
            step_model_map=data.get("step_model_map"),
            total_step_count=data.get("total_step_count"),
        )


def compute_workflow_fingerprint(
    graph_adjacency: dict[str, list[str]],
    step_model_map: dict[str, str],
    prompt_hashes: dict[str, str],
) -> str:
    """SHA-256 of sorted topology + model names + prompt hashes."""
    canonical = json.dumps(
        {
            "adjacency": graph_adjacency,
            "models": step_model_map,
            "prompts": prompt_hashes,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
