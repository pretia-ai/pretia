"""Workflow agent implementations for Pretia backtesting suite."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pretia.collectors.base import StepRecord


class BaseAgent(ABC):
    """Abstract base for all workflow agents."""

    @abstractmethod
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        """Run the workflow on one input, return StepRecords for every API call."""
