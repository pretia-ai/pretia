"""Research Agent (Simple) -- 3-step linear workflow, no loops.

Steps:
    search        (Haiku 4.5)  -> generate queries + canned tool results
    synthesize    (Sonnet 4.6) -> synthesize findings into structured summary
    format_report (Sonnet 4.6) -> format as brief research report
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        CANNED_SEARCH_RESULTS,
        get_anthropic_model,
    )

    # ── State ────────────────────────────────────────────────────────────

    class ResearchSimpleState(TypedDict):
        input: str
        queries: str
        search_results: str
        synthesis: str
        report: str

    # ── Node functions ───────────────────────────────────────────────────

    def search(state: ResearchSimpleState) -> dict:
        """Generate search queries and return canned search results."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=256,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Given a research question, generate "
                        "exactly 3 search queries that would help answer it. Return them as "
                        "a numbered list (1. ..., 2. ..., 3. ...)."
                    ),
                },
                {"role": "user", "content": state["input"]},
            ]
        )
        queries = result.content if isinstance(result.content, str) else str(result.content)
        # Return canned search results as the tool output
        combined_results = "\n\n---\n\n".join(CANNED_SEARCH_RESULTS)
        return {"queries": queries, "search_results": combined_results}

    def synthesize(state: ResearchSimpleState) -> dict:
        """Synthesize search results into a structured summary."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a research analyst. Synthesize the provided search results "
                        "into a structured summary. Identify key themes, data points, and "
                        "insights. Organize by topic and note any contradictions or gaps "
                        "in the data."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research question: {state['input']}\n\n"
                        f"Search queries used:\n{state.get('queries', '')}\n\n"
                        f"Search results:\n{state.get('search_results', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"synthesis": raw}

    def format_report(state: ResearchSimpleState) -> dict:
        """Format the synthesis into a brief research report."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a professional report writer. Format the research synthesis "
                        "into a brief, well-structured research report with sections: "
                        "Executive Summary, Key Findings, Data Points, and Conclusion. "
                        "Keep it concise but informative."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research question: {state['input']}\n\n"
                        f"Synthesis:\n{state.get('synthesis', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"report": raw}

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(ResearchSimpleState)
    builder.add_node("search", search)
    builder.add_node("synthesize", synthesize)
    builder.add_node("format_report", format_report)

    builder.set_entry_point("search")
    builder.add_edge("search", "synthesize")
    builder.add_edge("synthesize", "format_report")
    builder.add_edge("format_report", END)

    graph = builder.compile()

except ImportError:
    graph = None
