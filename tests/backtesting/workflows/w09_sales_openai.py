"""Sales/Outreach (OpenAI only) -- 3-step linear workflow, no loops.

Cross-provider test using OpenAI models exclusively via langchain_openai.

Steps:
    qualify_lead (GPT-4.1 Nano)     -> score lead, identify selling points
    personalize  (GPT-4.1 standard) -> research company using canned profiles
    draft_email  (GPT-4.1 standard) -> write personalized outreach email
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        CANNED_COMPANY_PROFILES,
        get_openai_model,
    )

    # ── State ────────────────────────────────────────────────────────────

    class SalesOpenAIState(TypedDict):
        input: str
        lead_score: str
        selling_points: str
        personalization: str
        email_draft: str

    # ── Node functions ───────────────────────────────────────────────────

    def qualify_lead(state: SalesOpenAIState) -> dict:
        """Score the lead 1-10 and identify key selling points."""
        llm = ChatOpenAI(
            model=get_openai_model("nano"),
            max_tokens=256,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a sales qualification specialist. Given information about "
                        "a potential lead, score them from 1-10 on likelihood to convert. "
                        "Identify 3-5 key selling points that would resonate with this lead. "
                        "Format your response as:\n"
                        "SCORE: X/10\n"
                        "SELLING POINTS:\n"
                        "1. ...\n2. ...\n3. ..."
                    ),
                },
                {"role": "user", "content": state["input"]},
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        # Split score and selling points
        score = raw
        points = raw
        if "SELLING POINTS" in raw.upper():
            idx = raw.upper().index("SELLING POINTS")
            score = raw[:idx].strip()
            points = raw[idx:]
        return {"lead_score": score, "selling_points": points}

    def personalize(state: SalesOpenAIState) -> dict:
        """Research the company using canned profile data."""
        llm = ChatOpenAI(
            model=get_openai_model("standard"),
            max_tokens=512,
        )
        # Build company context from canned profiles
        profiles_text = "\n\n".join(
            f"[{name}]: {profile}" for name, profile in CANNED_COMPANY_PROFILES.items()
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a sales researcher. Using the company database below, "
                        "identify which company the lead most likely belongs to and extract "
                        "relevant personalization details: company size, pain points, "
                        "technology stack, decision-making process, and recent initiatives. "
                        "If no exact match, infer the best fit."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Lead information: {state['input']}\n\n"
                        f"Lead score: {state.get('lead_score', '')}\n\n"
                        f"Company database:\n{profiles_text}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"personalization": raw}

    def draft_email(state: SalesOpenAIState) -> dict:
        """Write a personalized outreach email."""
        llm = ChatOpenAI(
            model=get_openai_model("standard"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert sales copywriter. Write a personalized cold "
                        "outreach email that:\n"
                        "- Opens with a relevant observation about their company\n"
                        "- Connects their pain points to our solution\n"
                        "- Uses the identified selling points naturally\n"
                        "- Ends with a clear, low-friction CTA\n"
                        "- Is concise (under 200 words)\n"
                        "- Sounds human, not templated"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Lead: {state['input']}\n\n"
                        f"Selling points:\n{state.get('selling_points', '')}\n\n"
                        f"Personalization research:\n{state.get('personalization', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"email_draft": raw}

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(SalesOpenAIState)
    builder.add_node("qualify_lead", qualify_lead)
    builder.add_node("personalize", personalize)
    builder.add_node("draft_email", draft_email)

    builder.set_entry_point("qualify_lead")
    builder.add_edge("qualify_lead", "personalize")
    builder.add_edge("personalize", "draft_email")
    builder.add_edge("draft_email", END)

    graph = builder.compile()

except ImportError:
    graph = None
