"""Sales/Outreach (Mixed providers) -- 5-step workflow with tone review loop (1-4 iterations).

Three different LLM providers: Google Gemini, OpenAI, and Anthropic.

Steps:
    qualify_lead     (Gemini 2.5 Flash)   -> score lead
    research_company (GPT-4.1 standard)   -> research company using canned profiles
    draft_email      (GPT-4.1 standard)   -> write personalized outreach email
    tone_review      (Opus 4.7)           -> review tone; loop to draft_email if issues
    finalize         (Gemini 2.5 Flash)   -> finalize the email
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        CANNED_COMPANY_PROFILES,
        get_anthropic_model,
        get_gemini_model,
        get_openai_model,
    )

    # ── State ────────────────────────────────────────────────────────────

    class SalesMixedState(TypedDict):
        input: str
        lead_score: str
        company_research: str
        email_draft: str
        tone_feedback: str
        final_email: str
        iteration_count: int

    # ── Node functions ───────────────────────────────────────────────────

    def qualify_lead(state: SalesMixedState) -> dict:
        """Score the lead using Gemini Flash."""
        llm = ChatGoogleGenerativeAI(
            model=get_gemini_model("flash"),
            max_output_tokens=256,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a sales qualification specialist. Given information about "
                        "a potential lead, score them from 1-10 on likelihood to convert. "
                        "Provide a brief justification for the score and identify the lead's "
                        "primary pain point and budget tier (startup/mid-market/enterprise)."
                    ),
                },
                {"role": "user", "content": state["input"]},
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"lead_score": raw, "iteration_count": 0}

    def research_company(state: SalesMixedState) -> dict:
        """Research the company using OpenAI GPT-4.1."""
        llm = ChatOpenAI(
            model=get_openai_model("standard"),
            max_tokens=512,
        )
        profiles_text = "\n\n".join(
            f"[{name}]: {profile}" for name, profile in CANNED_COMPANY_PROFILES.items()
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a sales researcher. Using the company database below, "
                        "build a detailed profile of the lead's company. Include: company "
                        "stage, headcount, technology stack, key decision makers, recent "
                        "news/initiatives, and competitive landscape. Identify the best "
                        "angle for outreach."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Lead information: {state['input']}\n\n"
                        f"Lead qualification: {state.get('lead_score', '')}\n\n"
                        f"Company database:\n{profiles_text}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"company_research": raw}

    def draft_email(state: SalesMixedState) -> dict:
        """Draft the outreach email using OpenAI GPT-4.1."""
        llm = ChatOpenAI(
            model=get_openai_model("standard"),
            max_tokens=1024,
        )
        tone_feedback = state.get("tone_feedback", "")
        extra = ""
        if tone_feedback and state.get("iteration_count", 0) > 0:
            extra = f"\n\nPrevious draft had tone issues. Apply this feedback:\n{tone_feedback}"
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert sales copywriter. Write a personalized cold "
                        "outreach email that:\n"
                        "- Opens with a specific, researched observation about their company\n"
                        "- Connects their pain points to our value proposition\n"
                        "- Uses a consultative tone (not pushy or salesy)\n"
                        "- Ends with a clear, low-friction CTA\n"
                        "- Is concise (under 200 words)\n"
                        "- Sounds authentic and human"
                        f"{extra}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Lead: {state['input']}\n\n"
                        f"Lead score: {state.get('lead_score', '')}\n\n"
                        f"Company research:\n{state.get('company_research', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        iteration = state.get("iteration_count", 0) + 1
        return {"email_draft": raw, "iteration_count": iteration}

    def tone_review(state: SalesMixedState) -> dict:
        """Review the email tone using Anthropic Opus."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a communications expert specializing in B2B outreach. "
                        "Review the outreach email draft for tone issues:\n"
                        "- Too aggressive or pushy?\n"
                        "- Too vague or generic?\n"
                        "- Inappropriate formality level for the recipient?\n"
                        "- Claims that seem exaggerated or unsupported?\n"
                        "- Missing personalization opportunities?\n\n"
                        "If the tone is appropriate and professional, respond with "
                        "'TONE_OK: true' at the end. If there are issues, provide specific "
                        "feedback and end with 'TONE_OK: false'."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Lead context: {state['input']}\n\n"
                        f"Lead score: {state.get('lead_score', '')}\n\n"
                        f"Email draft:\n{state.get('email_draft', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"tone_feedback": raw}

    def finalize(state: SalesMixedState) -> dict:
        """Finalize the email using Gemini Flash."""
        llm = ChatGoogleGenerativeAI(
            model=get_gemini_model("flash"),
            max_output_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an email finalization specialist. Take the approved email "
                        "draft and make final polish: fix any grammar issues, ensure "
                        "formatting is clean, add a proper subject line, and ensure the "
                        "signature block is professional. Return the complete, send-ready "
                        "email including Subject line."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Email draft:\n{state.get('email_draft', '')}\n\n"
                        f"Tone review notes:\n{state.get('tone_feedback', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"final_email": raw}

    # ── Routing ──────────────────────────────────────────────────────────

    def should_redraft(state: SalesMixedState) -> str:
        """Decide whether to loop back to draft_email or proceed to finalize."""
        iteration = state.get("iteration_count", 0)
        if iteration >= 4:
            return "finalize"
        tone_feedback = state.get("tone_feedback", "")
        if "tone_ok: false" in tone_feedback.lower():
            return "draft_email"
        return "finalize"

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(SalesMixedState)
    builder.add_node("qualify_lead", qualify_lead)
    builder.add_node("research_company", research_company)
    builder.add_node("draft_email", draft_email)
    builder.add_node("tone_review", tone_review)
    builder.add_node("finalize", finalize)

    builder.set_entry_point("qualify_lead")
    builder.add_edge("qualify_lead", "research_company")
    builder.add_edge("research_company", "draft_email")
    builder.add_edge("draft_email", "tone_review")
    builder.add_conditional_edges(
        "tone_review",
        should_redraft,
        {"draft_email": "draft_email", "finalize": "finalize"},
    )
    builder.add_edge("finalize", END)

    graph = builder.compile()

except ImportError:
    graph = None
