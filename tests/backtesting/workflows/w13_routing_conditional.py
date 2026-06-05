"""W13: Conditional routing agent — classifier routes to 3 cost paths.

70% → respond_simple (Haiku, ~$0.002)
20% → research_and_respond (Sonnet, ~$0.04)
10% → escalate_review (Sonnet + Opus, ~$0.20)

Tests step_count_variance and bimodality detectors.
"""

from __future__ import annotations

from typing import Any

try:
    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, StateGraph

    _HAIKU = ChatAnthropic(model="claude-haiku-4-5", max_tokens=512)
    _SONNET = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)
    _OPUS = ChatAnthropic(model="claude-opus-4-7", max_tokens=1024)

    class W13State(dict):
        input: str
        route: str
        response: str

    def classify(state: W13State) -> dict[str, Any]:
        result = _HAIKU.invoke(
            "Classify this customer request as SIMPLE, MODERATE, or COMPLEX. "
            "Respond with only the category name.\n\n"
            f"Request: {state['input']}"
        )
        text = result.content.strip().upper()
        if "COMPLEX" in text:
            route = "COMPLEX"
        elif "MODERATE" in text:
            route = "MODERATE"
        else:
            route = "SIMPLE"
        return {"route": route}

    def route_decision(state: W13State) -> str:
        route = state.get("route", "SIMPLE")
        if route == "COMPLEX":
            return "escalate_review"
        if route == "MODERATE":
            return "research_and_respond"
        return "respond_simple"

    def respond_simple(state: W13State) -> dict[str, str]:
        result = _HAIKU.invoke(f"Provide a brief, helpful response to: {state['input']}")
        return {"response": result.content}

    def research_and_respond(state: W13State) -> dict[str, str]:
        result = _SONNET.invoke(
            "Provide a detailed, researched response to this customer request. "
            "Include relevant details and next steps.\n\n"
            f"Request: {state['input']}"
        )
        return {"response": result.content}

    def escalate_review(state: W13State) -> dict[str, str]:
        draft = _SONNET.invoke(
            "Draft a thorough response to this complex customer issue. "
            "Include investigation steps and resolution.\n\n"
            f"Issue: {state['input']}"
        )
        reviewed = _OPUS.invoke(
            "Review this draft response for accuracy and completeness. "
            "Provide the final version.\n\n"
            f"Draft: {draft.content}"
        )
        return {"response": reviewed.content}

    builder = StateGraph(W13State)
    builder.add_node("classify", classify)
    builder.add_node("respond_simple", respond_simple)
    builder.add_node("research_and_respond", research_and_respond)
    builder.add_node("escalate_review", escalate_review)
    builder.set_entry_point("classify")
    builder.add_conditional_edges("classify", route_decision)
    builder.add_edge("respond_simple", END)
    builder.add_edge("research_and_respond", END)
    builder.add_edge("escalate_review", END)
    graph = builder.compile()

except ImportError:
    graph = None
