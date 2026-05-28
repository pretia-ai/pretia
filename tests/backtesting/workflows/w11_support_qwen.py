"""Support Agent (Qwen) -- 3-step linear workflow using Qwen models, no loops.

Mirrors W1 structure (classify → retrieve → respond) but uses Qwen-Agent framework
with Qwen-Turbo for classification and Qwen 3.6 Plus for response generation.
Provides a direct cost comparison: W1 (Anthropic) vs W11 (Qwen) on the same task.

Steps:
    classify_intent    (Qwen-Turbo)     -> intent category
    retrieve_context   (tool step)      -> FAQ lookup
    generate_response  (Qwen 3.6 Plus)  -> customer reply
"""

from __future__ import annotations

try:
    import json
    from collections.abc import Iterator
    from typing import Any

    from qwen_agent.agent import Agent
    from qwen_agent.llm.schema import Message
    from qwen_agent.tools.base import BaseTool, register_tool

    from tests.backtesting.workflows._shared import (
        CANNED_FAQ,
        get_qwen_model,
    )

    @register_tool("faq_lookup")
    class FAQLookupTool(BaseTool):
        description = "Look up FAQ content for a given intent category."
        parameters = [
            {
                "name": "category",
                "type": "string",
                "description": (
                    "Intent category: billing, technical, account, "
                    "general, or escalation"
                ),
                "required": True,
            }
        ]

        def call(self, params: str | dict, **kwargs: Any) -> str:
            if isinstance(params, str):
                params = json.loads(params)
            category = params.get("category", "general").strip().lower()
            return CANNED_FAQ.get(category, CANNED_FAQ["general"])

    class SupportQwenAgent(Agent):
        """Simple support agent using Qwen models with FAQ lookup."""

        def _run(
            self,
            messages: list[Message],
            lang: str = "en",
            **kwargs: Any,
        ) -> Iterator[list[Message]]:
            # Step 1: Classify intent using Qwen-Turbo
            classify_messages = [
                Message(
                    role="system",
                    content=(
                        "You are a customer support intent classifier for a SaaS product "
                        "management tool. Classify the user's message into exactly one "
                        "category: billing, technical, account, general, or escalation. "
                        "Respond with only the category name, nothing else."
                    ),
                ),
                messages[-1],
            ]
            intent = "general"
            for output in self.llm.chat(messages=classify_messages, stream=True):
                if output:
                    content = output[-1].content if hasattr(output[-1], "content") else ""
                    raw = str(content).strip().lower()
                    if raw in CANNED_FAQ:
                        intent = raw

            # Step 2: Retrieve FAQ context (tool step, no LLM call)
            context = CANNED_FAQ.get(intent, CANNED_FAQ["general"])

            # Step 3: Generate response using Qwen 3.6 Plus
            user_input = messages[-1].content if hasattr(messages[-1], "content") else ""
            respond_messages = [
                Message(
                    role="system",
                    content=(
                        "You are a helpful customer support agent for ProjectFlow, a SaaS "
                        "project management tool. Use the provided FAQ context to answer the "
                        "user's question. Be concise, friendly, and accurate. If the FAQ "
                        "doesn't cover the question, say so and suggest they contact "
                        "support@projectflow.io."
                    ),
                ),
                Message(
                    role="user",
                    content=f"Customer question: {user_input}\n\nFAQ context:\n{context}",
                ),
            ]
            for output in self.llm.chat(messages=respond_messages, stream=True):
                if output:
                    yield output

    agent = SupportQwenAgent(
        llm={"model": get_qwen_model("plus"), "model_server": "dashscope"},
        system_message=(
            "You are a helpful customer support agent for ProjectFlow. "
            "Classify the user's intent, look up relevant FAQ content, "
            "and provide a concise, friendly response."
        ),
        name="support_qwen",
    )

except ImportError:
    agent = None
