"""Research Agent (Complex) -- multi-step workflow with fact-check loop (1-6 iterations).

Steps:
    plan_research     (Sonnet 4.6) -> decompose question into sub-questions
    search_and_gather (Haiku 4.5)  -> generate queries + canned tool results
    synthesize        (Opus 4.7)   -> synthesize into coherent analysis
    fact_check        (Opus 4.7)   -> identify claims needing verification; loop if gaps
    write_report      (Sonnet 4.6) -> write polished report
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

    class ResearchComplexState(TypedDict):
        input: str
        research_plan: str
        sub_questions: str
        gathered_results: str
        synthesis: str
        fact_check_result: str
        report: str
        iteration_count: int

    # ── Node functions ───────────────────────────────────────────────────

    def plan_research(state: ResearchComplexState) -> dict:
        """Decompose the research question into 2-4 sub-questions."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a research strategist. Given a research question, decompose "
                        "it into 2-4 specific sub-questions that together would fully answer "
                        "the original question. For each sub-question, briefly note what type "
                        "of source would best answer it. Return as a numbered list."
                    ),
                },
                {"role": "user", "content": state["input"]},
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"research_plan": raw, "sub_questions": raw, "iteration_count": 0}

    def search_and_gather(state: ResearchComplexState) -> dict:
        """Generate search queries for sub-questions and return canned results."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=256,
        )
        fact_check_gaps = state.get("fact_check_result", "")
        extra = ""
        if fact_check_gaps and state.get("iteration_count", 0) > 0:
            extra = (
                f"\n\nPrevious fact-check found gaps. Focus on these areas:\n"
                f"{fact_check_gaps}"
            )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant. Generate targeted search queries for "
                        "each sub-question below. Return 1-2 queries per sub-question."
                        f"{extra}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research question: {state['input']}\n\n"
                        f"Sub-questions:\n{state.get('sub_questions', '')}"
                    ),
                },
            ]
        )
        _ = result.content  # LLM call generates queries; results come from canned data
        # Return canned search results
        combined = "\n\n---\n\n".join(CANNED_SEARCH_RESULTS)
        prior = state.get("gathered_results", "")
        if prior and state.get("iteration_count", 0) > 0:
            combined = f"{prior}\n\n=== Additional Results ===\n\n{combined}"
        iteration = state.get("iteration_count", 0) + 1
        return {"gathered_results": combined, "iteration_count": iteration}

    def synthesize(state: ResearchComplexState) -> dict:
        """Synthesize all gathered results into a coherent analysis."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a senior research analyst. Synthesize the gathered search "
                        "results into a coherent, well-structured analysis that addresses "
                        "each sub-question. Draw connections between findings, identify "
                        "patterns, and note areas of uncertainty. Support claims with "
                        "specific data points from the results."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research question: {state['input']}\n\n"
                        f"Research plan:\n{state.get('research_plan', '')}\n\n"
                        f"Gathered results:\n{state.get('gathered_results', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"synthesis": raw}

    def fact_check(state: ResearchComplexState) -> dict:
        """Identify claims needing verification. May trigger additional research."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a fact-checking specialist. Review the synthesis below and "
                        "identify any claims that lack sufficient evidence, seem inconsistent, "
                        "or need additional verification. For each claim, note what additional "
                        "information would resolve it. End your response with either "
                        "'GAPS_FOUND: true' if there are significant gaps that require "
                        "additional research, or 'GAPS_FOUND: false' if the analysis is "
                        "sufficiently supported."
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
        return {"fact_check_result": raw}

    def write_report(state: ResearchComplexState) -> dict:
        """Write a polished research report from the verified synthesis."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=2048,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a professional research report writer. Write a polished, "
                        "well-structured report with these sections: Executive Summary, "
                        "Methodology, Key Findings (with data points), Analysis, and "
                        "Recommendations. Use clear, authoritative language. Cite specific "
                        "numbers and facts from the synthesis."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Research question: {state['input']}\n\n"
                        f"Verified synthesis:\n{state.get('synthesis', '')}\n\n"
                        f"Fact-check notes:\n{state.get('fact_check_result', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"report": raw}

    # ── Routing ──────────────────────────────────────────────────────────

    def should_research_more(state: ResearchComplexState) -> str:
        """Decide whether to loop back to search_and_gather or proceed to write_report."""
        iteration = state.get("iteration_count", 0)
        if iteration >= 6:
            return "write_report"
        fact_check = state.get("fact_check_result", "")
        if "gaps_found: true" in fact_check.lower():
            return "search_and_gather"
        return "write_report"

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(ResearchComplexState)
    builder.add_node("plan_research", plan_research)
    builder.add_node("search_and_gather", search_and_gather)
    builder.add_node("synthesize", synthesize)
    builder.add_node("fact_check", fact_check)
    builder.add_node("write_report", write_report)

    builder.set_entry_point("plan_research")
    builder.add_edge("plan_research", "search_and_gather")
    builder.add_edge("search_and_gather", "synthesize")
    builder.add_edge("synthesize", "fact_check")
    builder.add_conditional_edges(
        "fact_check",
        should_research_more,
        {"search_and_gather": "search_and_gather", "write_report": "write_report"},
    )
    builder.add_edge("write_report", END)

    graph = builder.compile()

except ImportError:
    graph = None
