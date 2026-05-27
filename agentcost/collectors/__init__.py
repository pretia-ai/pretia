"""Framework adapters that emit unified StepRecords from agent workflows."""

from __future__ import annotations

from agentcost.collectors.base import BaseCollector, StepRecord
from agentcost.collectors.generic import GenericCollector

__all__ = ["BaseCollector", "GenericCollector", "LangGraphCollector", "StepRecord"]


def __getattr__(name: str) -> type:
    if name == "LangGraphCollector":
        from agentcost.collectors.langgraph import LangGraphCollector

        return LangGraphCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
