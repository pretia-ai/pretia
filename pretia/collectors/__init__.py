"""Framework adapters that emit unified StepRecords from agent workflows."""

from __future__ import annotations

from pretia.collectors.base import BaseCollector, StepRecord
from pretia.collectors.generic import GenericCollector

__all__ = [
    "AnthropicCollector",
    "BaseCollector",
    "GenericCollector",
    "LangGraphCollector",
    "MultiSDKCollector",
    "OpenAIAgentsCollector",
    "OpenAISDKCollector",
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
    if name == "AnthropicCollector":
        from pretia.collectors.anthropic_sdk import AnthropicCollector

        return AnthropicCollector
    if name == "OpenAISDKCollector":
        from pretia.collectors.openai_sdk import OpenAISDKCollector

        return OpenAISDKCollector
    if name == "MultiSDKCollector":
        from pretia.collectors.multi_sdk import MultiSDKCollector

        return MultiSDKCollector
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
