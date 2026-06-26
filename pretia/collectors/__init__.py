"""Framework adapters that emit unified StepRecords from agent workflows."""

from __future__ import annotations

from pretia.collectors.base import BaseCollector, StepRecord
from pretia.collectors.generic import GenericCollector

__all__ = [
    "BaseCollector",
    "GenericCollector",
    "LangGraphCollector",
    "OpenAIAgentsCollector",
    "QwenAgentCollector",
    "StepRecord",
]


def __getattr__(name: str) -> type:
    if name == "LangGraphCollector":
        from pretia.collectors.langgraph import LangGraphCollector

        return LangGraphCollector
    if name == "OpenAIAgentsCollector":
        from pretia.collectors.openai_agents import OpenAIAgentsCollector

        return OpenAIAgentsCollector
    if name == "QwenAgentCollector":
        from pretia.collectors.qwen_agent import QwenAgentCollector

        return QwenAgentCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
