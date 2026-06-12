"""Backtesting workflow configurations for the test workflows."""

from __future__ import annotations

from agentcost.validation.suite import BacktestConfig
from tests.backtesting.workflows._shared import (
    get_anthropic_model,
    get_deepseek_model,
    get_gemini_model,
    get_openai_model,
    get_qwen_model,
)

_ALL_CONFIGS: list[BacktestConfig] = [
    BacktestConfig(
        name="W1-support-simple",
        archetype="support-agent",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w01_support_simple.py",
        description="Classify intent (Haiku) → retrieve FAQ → generate response (Sonnet)",
        expected_models=[get_anthropic_model("haiku"), get_anthropic_model("sonnet")],
        has_loops=False,
        expected_cost_range=(0.005, 0.03),
    ),
    BacktestConfig(
        name="W2-support-complex",
        archetype="support-agent",
        complexity="complex",
        workflow_path="tests/backtesting/workflows/w02_support_complex.py",
        description=(
            "Classify (DeepSeek Flash) → draft loop (DeepSeek Pro, 1-12 iter) "
            "→ conditional review (DeepSeek Pro)"
        ),
        expected_models=[
            get_deepseek_model("flash"),
            get_deepseek_model("pro"),
        ],
        has_loops=True,
        expected_cost_range=(0.005, 0.10),
    ),
    # W3 excluded from active suite — redundant with W4 (both code-review, W3 is simple-only)
    BacktestConfig(
        name="W3-codereview-simple",
        archetype="code-review",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w03_codereview_simple.py",
        description="Analyze diff → generate comments → summarize (all Sonnet)",
        expected_models=[get_anthropic_model("sonnet")],
        has_loops=False,
        expected_cost_range=(0.02, 0.08),
    ),
    BacktestConfig(
        name="W4-compliance-review",
        archetype="compliance-review",
        complexity="complex",
        workflow_path="bt_agents/workflows/w04.py",
        description=(
            "Initial review (DeepSeek V4 Pro) → critique (Qwen Plus) "
            "→ revision loop (DeepSeek V4 Pro, 1-8 pairs)"
        ),
        expected_models=[get_deepseek_model("pro"), get_qwen_model("plus")],
        has_loops=True,
        expected_cost_range=(0.005, 0.15),
    ),
    BacktestConfig(
        name="W5-extraction-simple",
        archetype="data-extraction",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w05_extraction_simple.py",
        description="Parse document (Haiku) → extract fields (Sonnet) → validate (Haiku)",
        expected_models=[get_anthropic_model("haiku"), get_anthropic_model("sonnet")],
        has_loops=False,
        expected_cost_range=(0.005, 0.04),
    ),
    # W6 excluded from active suite — redundant with W5 (both extraction, W6 adds loop)
    BacktestConfig(
        name="W6-extraction-complex",
        archetype="data-extraction",
        complexity="complex",
        workflow_path="tests/backtesting/workflows/w06_extraction_complex.py",
        description=(
            "Parse (Haiku) → extract (Sonnet) → cross-reference (Sonnet) "
            "→ resolve conflicts loop (Opus, 1-5 iter) → format (Haiku)"
        ),
        expected_models=[
            get_anthropic_model("haiku"),
            get_anthropic_model("sonnet"),
            get_anthropic_model("opus"),
        ],
        has_loops=True,
        expected_cost_range=(0.08, 0.50),
    ),
    # W7 excluded from active suite — redundant with W8 (both research, W7 is simple-only)
    BacktestConfig(
        name="W7-research-simple",
        archetype="research-agent",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w07_research_simple.py",
        description="Search queries (Haiku) → synthesize (Sonnet) → format report (Sonnet)",
        expected_models=[get_anthropic_model("haiku"), get_anthropic_model("sonnet")],
        has_loops=False,
        expected_cost_range=(0.02, 0.10),
    ),
    BacktestConfig(
        name="W8-research-complex",
        archetype="research-agent",
        complexity="complex",
        workflow_path="tests/backtesting/workflows/w08_research_complex.py",
        description=(
            "Plan (Sonnet) → search (Haiku) → synthesize (Opus) "
            "→ fact-check loop (Opus, 1-6 iter) → write report (Sonnet)"
        ),
        expected_models=[
            get_anthropic_model("haiku"),
            get_anthropic_model("sonnet"),
            get_anthropic_model("opus"),
        ],
        has_loops=True,
        expected_cost_range=(0.25, 1.80),
    ),
    BacktestConfig(
        name="W9-sales-openai",
        archetype="sales-outreach",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w09_sales_openai.py",
        description=(
            "Qualify lead (GPT-4.1 Nano) → personalize (GPT-4.1) "
            "→ draft email (GPT-4.1) — OpenAI only"
        ),
        expected_models=[get_openai_model("nano"), get_openai_model("standard")],
        has_loops=False,
        expected_cost_range=(0.005, 0.03),
    ),
    BacktestConfig(
        name="W10-sales-mixed",
        archetype="sales-outreach",
        complexity="complex",
        workflow_path="tests/backtesting/workflows/w10_sales_mixed.py",
        description=(
            "Qualify (Gemini Flash) → research (GPT-4.1) → draft (GPT-4.1) "
            "→ tone review loop (Opus, 1-4 iter) → finalize (Gemini Flash) — mixed providers"
        ),
        expected_models=[
            get_gemini_model("flash"),
            get_openai_model("standard"),
            get_anthropic_model("opus"),
        ],
        has_loops=True,
        expected_cost_range=(0.10, 0.70),
    ),
    BacktestConfig(
        name="W11-support-qwen",
        archetype="support-agent",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w11_support_qwen.py",
        description=(
            "Classify intent (Qwen-Turbo) → retrieve FAQ → generate response "
            "(Qwen 3.6 Plus) — Qwen-Agent framework, direct cost comparison with W1"
        ),
        expected_models=[get_qwen_model("turbo"), get_qwen_model("plus")],
        has_loops=False,
        expected_cost_range=(0.001, 0.01),
    ),
    BacktestConfig(
        name="W12-extraction-deepseek",
        archetype="data-extraction",
        complexity="simple",
        workflow_path="tests/backtesting/workflows/w12_extraction_deepseek.py",
        description=(
            "Parse document → extract fields → validate (all DeepSeek V4 Flash) "
            "— direct cost comparison with W5 (Anthropic)"
        ),
        expected_models=[get_deepseek_model("pro")],
        has_loops=False,
        expected_cost_range=(0.001, 0.05),
    ),
    BacktestConfig(
        name="W13-routing-conditional",
        archetype="routing-agent",
        complexity="complex",
        workflow_path="tests/backtesting/workflows/w13_routing_conditional.py",
        description=(
            "Classify (Haiku) → route to one of: respond_simple (Haiku, 70%), "
            "research_and_respond (Sonnet, 20%), escalate_review (Sonnet+Opus, 10%). "
            "Tests step count variance and bimodal cost distribution."
        ),
        expected_models=[
            get_anthropic_model("haiku"),
            get_anthropic_model("sonnet"),
            get_anthropic_model("opus"),
        ],
        has_loops=False,
        expected_cost_range=(0.001, 0.20),
    ),
    BacktestConfig(
        name="W14-simple-pdf-rag",
        archetype="rag-pipeline",
        complexity="complex",
        workflow_path="bt_agents/workflows/w14.py",
        description=(
            "Upload PDF → extract text → chunk → embed (OpenAI) → query → "
            "vector retrieve → generate structured JSON answer (Sonnet). "
            "Tests retrieval size variance and cross-provider accounting."
        ),
        expected_models=[get_openai_model("nano"), get_anthropic_model("sonnet")],
        has_loops=False,
        expected_cost_range=(0.03, 0.65),
    ),
    BacktestConfig(
        name="W15-multihop-rag",
        archetype="rag-pipeline",
        complexity="complex",
        workflow_path="bt_agents/workflows/w15.py",
        description=(
            "PDF ingestion → query → retrieve → assess sufficiency (Gemini Flash) → "
            "re-retrieve if insufficient (1-4 hops) → generate answer (DeepSeek V4). "
            "Exercises all four cost model adjustments simultaneously."
        ),
        expected_models=[
            get_openai_model("nano"),
            get_gemini_model("flash"),
            get_deepseek_model("pro"),
        ],
        has_loops=True,
        expected_cost_range=(0.01, 0.55),
    ),
    BacktestConfig(
        name="W16-map-reduce-pdf",
        archetype="map-reduce",
        complexity="complex",
        workflow_path="bt_agents/workflows/w16.py",
        description=(
            "Upload long PDF → split into N sections (N varies 3-20) → "
            "N parallel Haiku calls → Sonnet aggregation. "
            "Tests fan-out with variable N and parallel execution."
        ),
        expected_models=[get_anthropic_model("haiku"), get_anthropic_model("sonnet")],
        has_loops=False,
        expected_cost_range=(0.02, 0.30),
    ),
    BacktestConfig(
        name="W17-claims-agent",
        archetype="claims-review",
        complexity="complex",
        workflow_path="bt_agents/workflows/w17.py",
        description=(
            "Intake + override check (Haiku) → policy retrieval (OpenAI embed) → "
            "evaluate + decide (Sonnet JSON) → conditional routing (Haiku). "
            "Real-world decision tree with multi-doc RAG and bimodal cost."
        ),
        expected_models=[
            get_anthropic_model("haiku"),
            get_openai_model("nano"),
            get_anthropic_model("sonnet"),
        ],
        has_loops=False,
        expected_cost_range=(0.005, 0.25),
    ),
    BacktestConfig(
        name="W18-long-document",
        archetype="document-processing",
        complexity="simple",
        workflow_path="bt_agents/workflows/w18.py",
        description=(
            "Single long PDF (50-100 pages) processed in one context window. "
            "Summarization, QA, and key findings extraction via DeepSeek V4. "
            "Tests long-context cost scaling (30K-100K input tokens)."
        ),
        expected_models=[get_deepseek_model("pro")],
        has_loops=False,
        expected_cost_range=(0.005, 0.09),
    ),
    BacktestConfig(
        name="W19-multi-turn",
        archetype="conversational",
        complexity="complex",
        workflow_path="bt_agents/workflows/w19.py",
        description=(
            "8-turn customer support conversation via DeepSeek V4. "
            "Each turn accumulates full history as context. "
            "Tests session accumulation and context growth across turns."
        ),
        expected_models=[get_deepseek_model("pro")],
        has_loops=False,
        expected_cost_range=(0.05, 0.65),
    ),
]

_EXCLUDED_NAMES = {
    "W3-codereview-simple",
    "W4-compliance-review",
    "W6-extraction-complex",
    "W7-research-simple",
    "W8-research-complex",
    "W10-sales-mixed",
}

BACKTESTING_CONFIGS: list[BacktestConfig] = [
    c for c in _ALL_CONFIGS if c.name not in _EXCLUDED_NAMES
]
