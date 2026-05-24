"""Define the StepRecord dataclass and BaseCollector interface."""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass
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
            input_tokens * input_price + output_tokens * output_price.

        Raises:
            ValueError: If this record's model has no entry in `pricing`.
        """
        if self.model not in pricing:
            raise ValueError(
                f"No pricing for model {self.model!r}. Available models: {sorted(pricing)}"
            )
        input_price, output_price = pricing[self.model]
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
        )


class BaseCollector(ABC):
    """Interface that every framework-specific step collector implements."""

    @abstractmethod
    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
    ) -> list[list[StepRecord]]:
        """Run the workflow on each input and return one StepRecord list per run.

        Args:
            workflow: The agent workflow object (framework-specific).
            inputs: Input strings to run the workflow on.

        Returns:
            One list of StepRecords per input, in the same order as `inputs`.
        """
        ...

    def collect_sync(
        self,
        workflow: Any,
        inputs: list[str],
    ) -> list[list[StepRecord]]:
        """Run `collect()` to completion synchronously. Convenience wrapper for CLI use."""
        return asyncio.run(self.collect(workflow, inputs))
