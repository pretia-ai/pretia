"""Code Review (Simple) -- 3-step linear workflow, no loops. All Sonnet 4.6.

Steps:
    analyze_diff      -> structural analysis of the PR diff
    generate_comments -> line-by-line review comments
    summarize         -> overall review summary
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        get_anthropic_model,
    )

    _CODE_REVIEW_SYSTEM = (
        "You are a senior software engineer reviewing a pull request for a Python "
        "web application. Focus on: correctness, edge cases, naming conventions, "
        "error handling, and potential performance issues. Be constructive and "
        "specific. Reference line numbers where relevant."
    )

    # ── State ────────────────────────────────────────────────────────────

    class CodeReviewSimpleState(TypedDict):
        input: str
        analysis: str
        comments: str
        summary: str

    # ── Node functions ───────────────────────────────────────────────────

    def analyze_diff(state: CodeReviewSimpleState) -> dict:
        """Perform structural analysis of the PR diff."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_CODE_REVIEW_SYSTEM}\n\n"
                        "Analyze the diff structurally: identify which files changed, "
                        "what the purpose of the change is, and any patterns you notice. "
                        "Do not write review comments yet."
                    ),
                },
                {"role": "user", "content": f"PR diff:\n\n{state['input']}"},
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"analysis": text}

    def generate_comments(state: CodeReviewSimpleState) -> dict:
        """Generate line-by-line review comments."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_CODE_REVIEW_SYSTEM}\n\n"
                        "Based on the analysis, generate specific line-by-line review "
                        "comments. Format each comment as:\n"
                        "  [file:line] severity (critical/warning/nit): comment\n"
                        "Sort by severity (critical first)."
                    ),
                },
                {
                    "role": "user",
                    "content": (f"PR diff:\n{state['input']}\n\nAnalysis:\n{state['analysis']}"),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"comments": text}

    def summarize(state: CodeReviewSimpleState) -> dict:
        """Write an overall review summary."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_CODE_REVIEW_SYSTEM}\n\n"
                        "Write a concise review summary. Include: overall assessment "
                        "(approve / request changes / comment only), number of issues "
                        "found by severity, and a one-paragraph narrative."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"PR diff:\n{state['input']}\n\nReview comments:\n{state['comments']}"
                    ),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"summary": text}

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(CodeReviewSimpleState)
    builder.add_node("analyze_diff", analyze_diff)
    builder.add_node("generate_comments", generate_comments)
    builder.add_node("summarize", summarize)

    builder.set_entry_point("analyze_diff")
    builder.add_edge("analyze_diff", "generate_comments")
    builder.add_edge("generate_comments", "summarize")
    builder.add_edge("summarize", END)

    graph = builder.compile()

except ImportError:
    graph = None
