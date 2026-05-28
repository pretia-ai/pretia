"""Support Agent (Simple) -- 3-step linear workflow, no loops.

Steps:
    classify_intent  (Haiku 4.5)  -> intent category
    retrieve_context (tool step)  -> FAQ lookup
    generate_response (Sonnet 4.6) -> customer reply
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        CANNED_FAQ,
        get_anthropic_model,
    )

    # ── State ────────────────────────────────────────────────────────────

    class SupportSimpleState(TypedDict):
        input: str
        intent: str
        context: str
        response: str

    # ── Node functions ───────────────────────────────────────────────────

    def classify_intent(state: SupportSimpleState) -> dict:
        """Classify the user message into one of five intent categories."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=64,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support intent classifier for a SaaS product "
                        "management tool. Classify the user's message into exactly one "
                        "category: billing, technical, account, general, or escalation. "
                        "Respond with only the category name, nothing else."
                    ),
                },
                {"role": "user", "content": state["input"]},
            ]
        )
        text = (
            result.content if isinstance(result.content, str) else str(result.content)
        )
        raw = text.strip().lower()
        intent = raw if raw in CANNED_FAQ else "general"
        return {"intent": intent}

    def retrieve_context(state: SupportSimpleState) -> dict:
        """Look up canned FAQ content for the classified intent."""
        category = state.get("intent", "general")
        context = CANNED_FAQ.get(category, CANNED_FAQ["general"])
        return {"context": context}

    def generate_response(state: SupportSimpleState) -> dict:
        """Generate a customer-facing support response."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support agent for ProjectFlow, a SaaS "
                        "project management tool. Use the provided FAQ context to answer the "
                        "user's question. Be concise, friendly, and accurate. If the FAQ "
                        "doesn't cover the question, say so and suggest they contact "
                        "support@projectflow.io."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer question: {state['input']}\n\n"
                        f"FAQ context:\n{state['context']}"
                    ),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"response": text}

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(SupportSimpleState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_response", generate_response)

    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_response")
    builder.add_edge("generate_response", END)

    graph = builder.compile()

except ImportError:
    graph = None
