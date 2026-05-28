"""Support Agent (Complex) -- 5-step workflow with review loop (1-15 iterations).

Steps:
    classify_intent   (Haiku 4.5)  -> intent category
    retrieve_context  (tool step)  -> FAQ lookup
    generate_draft    (Sonnet 4.6) -> draft response
    review_and_iterate (Opus 4.7)  -> approve or return feedback (loops)
    format_response   (Haiku 4.5)  -> add greeting/sign-off
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

    class SupportComplexState(TypedDict):
        input: str
        intent: str
        context: str
        draft: str
        feedback: str
        response: str
        iteration_count: int

    # ── Node functions ───────────────────────────────────────────────────

    def classify_intent(state: SupportComplexState) -> dict:
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

    def retrieve_context(state: SupportComplexState) -> dict:
        """Look up canned FAQ content for the classified intent."""
        category = state.get("intent", "general")
        context = CANNED_FAQ.get(category, CANNED_FAQ["general"])
        return {"context": context}

    def generate_draft(state: SupportComplexState) -> dict:
        """Draft a customer support response, incorporating feedback if present."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=512,
        )
        feedback = state.get("feedback", "")
        user_content = (
            f"Customer question: {state['input']}\n\n"
            f"FAQ context:\n{state['context']}"
        )
        if feedback:
            user_content += f"\n\nRevision feedback from reviewer:\n{feedback}"

        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful customer support agent for ProjectFlow, a SaaS "
                        "project management tool. Use the provided FAQ context to answer the "
                        "user's question. Be concise, friendly, and accurate. If the FAQ "
                        "doesn't cover the question, say so and suggest they contact "
                        "support@projectflow.io. Keep your response under 200 words."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
        )
        current_count = state.get("iteration_count", 0)
        return {
            "draft": result.content if isinstance(result.content, str) else str(result.content),
            "iteration_count": current_count + 1,
        }

    def review_and_iterate(state: SupportComplexState) -> dict:
        """Review the draft response. Return APPROVED or revision instructions."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a senior customer support quality reviewer. Review the "
                        "following draft response for a customer inquiry.\n\n"
                        "Approve if the response is factually accurate, complete, empathetic, "
                        "and under 200 words. Otherwise, provide specific revision "
                        "instructions. Always approve after reasonable quality is achieved "
                        "-- do not chase perfection.\n\n"
                        "If approved, respond with exactly: APPROVED\n"
                        "If not approved, respond with your revision instructions."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Customer question: {state['input']}\n\n"
                        f"Draft response:\n{state['draft']}"
                    ),
                },
            ]
        )
        feedback_text = result.content if isinstance(result.content, str) else str(result.content)
        return {"feedback": feedback_text}

    def format_response(state: SupportComplexState) -> dict:
        """Add a professional greeting and sign-off to the approved draft."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a customer support formatting assistant. Take the approved "
                        "draft response and add a warm greeting at the top and a professional "
                        "sign-off at the bottom. Do not change the content of the response "
                        "itself. Use 'ProjectFlow Support Team' as the sign-off name."
                    ),
                },
                {"role": "user", "content": state["draft"]},
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"response": text}

    # ── Routing ──────────────────────────────────────────────────────────

    def should_continue_review(state: SupportComplexState) -> str:
        """Route to format_response if approved or iteration cap reached."""
        feedback = state.get("feedback", "")
        iteration_count = state.get("iteration_count", 0)
        if "APPROVED" in feedback.upper() or iteration_count >= 15:
            return "format_response"
        return "generate_draft"

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(SupportComplexState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_context", retrieve_context)
    builder.add_node("generate_draft", generate_draft)
    builder.add_node("review_and_iterate", review_and_iterate)
    builder.add_node("format_response", format_response)

    builder.set_entry_point("classify_intent")
    builder.add_edge("classify_intent", "retrieve_context")
    builder.add_edge("retrieve_context", "generate_draft")
    builder.add_edge("generate_draft", "review_and_iterate")
    builder.add_conditional_edges(
        "review_and_iterate",
        should_continue_review,
        {"format_response": "format_response", "generate_draft": "generate_draft"},
    )
    builder.add_edge("format_response", END)

    graph = builder.compile()

except ImportError:
    graph = None
