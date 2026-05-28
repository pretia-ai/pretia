"""Code Review (Complex) -- 5-step workflow with self-review loop (1-8 iterations).

Steps:
    analyze_diff   (Sonnet 4.6) -> structural analysis
    identify_issues (Opus 4.7)  -> ranked critical vs minor issues
    suggest_fixes   (Opus 4.7)  -> concrete fix suggestions
    self_review     (Opus 4.7)  -> verify suggestions, loop if issues
    format_review   (Sonnet 4.6) -> format as PR comment
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

    class CodeReviewComplexState(TypedDict):
        input: str
        analysis: str
        issues: str
        suggestions: str
        review_feedback: str
        formatted_review: str
        iteration_count: int

    # ── Node functions ───────────────────────────────────────────────────

    def analyze_diff(state: CodeReviewComplexState) -> dict:
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

    def identify_issues(state: CodeReviewComplexState) -> dict:
        """Identify and rank issues as critical or minor."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_CODE_REVIEW_SYSTEM}\n\n"
                        "Based on the analysis, identify all issues in the code. Classify "
                        "each as CRITICAL (bugs, security, data loss) or MINOR (style, "
                        "naming, performance nits). Format as:\n"
                        "  [CRITICAL|MINOR] file:line - description\n"
                        "List critical issues first."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"PR diff:\n{state['input']}\n\n"
                        f"Structural analysis:\n{state['analysis']}"
                    ),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"issues": text}

    def suggest_fixes(state: CodeReviewComplexState) -> dict:
        """Generate concrete fix suggestions for identified issues."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=1024,
        )
        review_feedback = state.get("review_feedback", "")
        user_content = (
            f"PR diff:\n{state['input']}\n\n"
            f"Issues identified:\n{state['issues']}"
        )
        if review_feedback:
            user_content += f"\n\nSelf-review feedback (revise accordingly):\n{review_feedback}"

        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_CODE_REVIEW_SYSTEM}\n\n"
                        "For each identified issue, suggest a concrete fix. Show the "
                        "suggested code change as a diff snippet. Be precise and actionable."
                    ),
                },
                {"role": "user", "content": user_content},
            ]
        )
        current_count = state.get("iteration_count", 0)
        return {
            "suggestions": (
                result.content if isinstance(result.content, str) else str(result.content)
            ),
            "iteration_count": current_count + 1,
        }

    def self_review(state: CodeReviewComplexState) -> dict:
        """Verify the fix suggestions are correct and complete."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are verifying your own code review suggestions. Check that:\n"
                        "1. Each suggestion is syntactically correct\n"
                        "2. Fixes address the actual issue without introducing new bugs\n"
                        "3. No important issues were missed\n\n"
                        "If all suggestions are sound, respond with exactly: APPROVED\n"
                        "Otherwise, explain what needs revision. Be concise."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Original diff:\n{state['input']}\n\n"
                        f"Issues:\n{state['issues']}\n\n"
                        f"Suggested fixes:\n{state['suggestions']}"
                    ),
                },
            ]
        )
        feedback_text = result.content if isinstance(result.content, str) else str(result.content)
        return {"review_feedback": feedback_text}

    def format_review(state: CodeReviewComplexState) -> dict:
        """Format the review as a PR comment."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "Format the code review as a GitHub PR comment using Markdown. "
                        "Include:\n"
                        "1. Summary header with overall verdict (approve/request changes)\n"
                        "2. Critical issues section (if any)\n"
                        "3. Minor issues section (if any)\n"
                        "4. Suggested fixes with code blocks\n"
                        "5. Closing remarks\n\n"
                        "Use proper Markdown formatting with headers, lists, and code blocks."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Issues:\n{state['issues']}\n\n"
                        f"Suggested fixes:\n{state['suggestions']}"
                    ),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"formatted_review": text}

    # ── Routing ──────────────────────────────────────────────────────────

    def should_continue_self_review(state: CodeReviewComplexState) -> str:
        """Route to format_review if approved or iteration cap reached."""
        review_feedback = state.get("review_feedback", "")
        iteration_count = state.get("iteration_count", 0)
        if "APPROVED" in review_feedback.upper() or iteration_count >= 8:
            return "format_review"
        return "suggest_fixes"

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(CodeReviewComplexState)
    builder.add_node("analyze_diff", analyze_diff)
    builder.add_node("identify_issues", identify_issues)
    builder.add_node("suggest_fixes", suggest_fixes)
    builder.add_node("self_review", self_review)
    builder.add_node("format_review", format_review)

    builder.set_entry_point("analyze_diff")
    builder.add_edge("analyze_diff", "identify_issues")
    builder.add_edge("identify_issues", "suggest_fixes")
    builder.add_edge("suggest_fixes", "self_review")
    builder.add_conditional_edges(
        "self_review",
        should_continue_self_review,
        {"format_review": "format_review", "suggest_fixes": "suggest_fixes"},
    )
    builder.add_edge("format_review", END)

    graph = builder.compile()

except ImportError:
    graph = None
