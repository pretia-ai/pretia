# Directions: System Prompt Generation for All 14 Backtesting Workflows

**Purpose:** This file specifies what Claude Code must produce — one or more production-grade system prompts per workflow, for a total of 14 workflows with ~30 distinct prompts. These prompts are the fixed instructions each agent step uses when processing an input. They do not change between runs. Cost variation comes from how different inputs interact with these fixed prompts and the workflow's control flow (loops, routing, fan-out).

**Context for Claude Code:** You have access to the Pretia codebase, `projection-engine-recommendation-addition-2.md` (engine design, statistical methods, schema, W17 full architecture, PDF pipeline), `cross-cutting-robustness.md` (robustness constraints, detector matrix, failure modes), and the technical spec. The system described in the system guide is already implemented. This file provides the per-workflow specifications you need to generate the prompts. Treat it as the authoritative source for workflow structure, step architecture, model assignments, and prompt requirements.

**What you are generating:** For each workflow step, a complete system prompt — full text, not a stub. The prompt is what gets sent as the `system` message (or equivalent) in the API call. It defines the agent's role, constraints, output format, and behavioral rules for that step.

---

## General Constraints (Apply to Every Prompt)

### Length and Realism

Every main-step prompt must be 1,000–2,500 tokens. Classification/routing steps can be shorter (400–800 tokens). This is not padding — production agent prompts are this long because they include persona definition, behavioral rules, output schemas, examples, and edge case handling. A 200-token prompt is a toy; a 2,500-token prompt is what ships.

### Output Format Enforcement

Any step requiring structured output must include the full JSON schema in the prompt, with field names, types, descriptions, and required/optional flags. Include the instruction: "Respond with valid JSON only. No explanation, no markdown fences, no preamble." Reinforce for non-Anthropic models: "Your entire response must be a single JSON object. Do not include any text before or after the JSON."

### Cache-Busting Placeholder

Every prompt for a DeepSeek workflow (W4, W12, W18, W19) must end with: `\n\n<!-- session: {{CACHE_BUST_SUFFIX}} -->`. This placeholder is replaced at runtime with a unique UUID per API call. Place it at the END of the prompt to defeat prefix caching. For W19 (multi-turn), each turn gets a fresh suffix.

Anthropic workflows (W1, W2, W5, W13, W14, W16, W17) must include the same placeholder for cache-busting during rapid profiling runs.

### Cost-Critical Elements

Elements marked **[COST-CRITICAL]** in the per-workflow specs below are load-bearing for the cost distribution. They control iteration counts, routing decisions, output length, or model selection. When generating prompts:
- Include these elements exactly as specified.
- Add an inline comment in the prompt (as a non-functional annotation) marking them: `<!-- COST-CRITICAL: loop termination -->`.
- Do not paraphrase or soften these instructions.

Elements not marked cost-critical are important for realism but can be adapted without invalidating the cost profile.

### What Prompts Must NOT Do

- Must not leak internal reasoning markers to the user (no "thinking out loud" unless it's a self-reflection step designed for it).
- Must not invent business rules, policies, or product features not specified in the prompt itself. The prompt must be self-contained — no assumption that the model has external knowledge about the fictional company or domain.
- Must not include instructions that suppress cost variance (e.g., "always respond in exactly 50 words" flattens the distribution). Output length constraints should define a range, not a point.
- Must not include meta-instructions about Pretia, profiling, or backtesting. The prompt should read as a genuine production prompt for an agent that doesn't know it's being profiled.

### Domain Content

Each workflow operates in a fictional but specific domain. The prompts must include enough domain content (product names, policy rules, company context) to produce realistic agent behavior. Use the following fictional domains:

- **W1, W2, W11:** TechFlow — a B2B SaaS platform for workflow automation. Products: TechFlow Core (workflow builder, $49/mo starter, $199/mo pro, $499/mo enterprise), TechFlow Analytics (reporting dashboard, included in Pro+), TechFlow Connect (third-party integrations, 50+ connectors). Support hours: 9 AM–6 PM ET weekdays, emergency 24/7 for Enterprise.
- **W4:** Compliance review — generic business documents (contracts, HR policies, regulatory filings). No single company; the agent reviews documents submitted to it.
- **W5:** Data extraction — invoices, receipts, forms, business cards. Generic commercial context.
- **W9:** NovaCRM — a fictional sales intelligence platform. The agent qualifies leads and drafts outreach for NovaCRM's sales team.
- **W12:** Data extraction — same as W5 but text-only. Corporate documents, structured reports.
- **W13:** General-purpose assistant — routed by query complexity. No specific domain.
- **W14, W15:** Health insurance — policy documents, coverage questions. Providers: United Healthcare, Aetna, BlueCross BlueShield.
- **W16:** Corporate analysis — annual reports, regulatory filings, research papers.
- **W17:** Health insurance claims — full claims processing. Providers: United Healthcare, Aetna, Cigna. (See W17 full architecture in `projection-engine-recommendation-addition-2.md`.)
- **W18:** Long-document analysis — annual reports, legal depositions, technical specifications.
- **W19:** CloudOps — a fictional cloud infrastructure platform. Multi-turn support conversations about deployments, monitoring, scaling.

---

## Per-Workflow Specifications

---

### W1 — Customer Support (Simple)

**What it does:** Single-turn customer support. User asks a question, agent classifies intent and responds. The classifier routes between Haiku (simple) and Sonnet (complex) — but this routing is handled by the workflow orchestrator based on the classification output, not by the prompt itself.

**Steps:**
1. **Classify + Respond** — Haiku 4.5 or Sonnet 4.6 (routed by orchestrator based on detected complexity)

**Step 1 prompt specification:**

- **Role:** Customer support agent for TechFlow, a B2B SaaS platform.
- **Persona:** Professional, helpful, concise. First name: "Alex."
- **Domain knowledge to include in prompt:** TechFlow product line (Core: workflow builder, $49/mo starter, $199/mo pro, $499/mo enterprise; Analytics: reporting dashboard, included in Pro+; Connect: third-party integrations, 50+ connectors). Support hours: 9 AM–6 PM ET weekdays, emergency support 24/7 for Enterprise. Common issues: password resets, billing questions, integration setup, feature requests, outage reports.
- **Behavioral rules:**
  - Greet the customer by acknowledging their issue (not with a generic "Hello!").
  - **[COST-CRITICAL]** For simple factual questions (hours, pricing, feature availability): respond in 100–200 words.
  - **[COST-CRITICAL]** For complex issues (multi-part complaints, technical troubleshooting, billing disputes): respond in 200–400 words.
  - If the question is outside TechFlow's domain, say so and do not fabricate an answer.
  - If the issue requires account access or actions the agent cannot take, escalate: "Let me connect you with a specialist who can access your account."
  - Never promise refunds, credits, or policy exceptions without explicit authorization language.
- **Output format:** Plain text response. No JSON. The response goes directly to the customer.
- **What the prompt must include:** The full product/pricing reference (this is the domain knowledge the agent draws from — without it, responses are generic and short, suppressing output variance). Escalation criteria. Tone instructions.
- **What the prompt must NOT include:** No routing logic (the orchestrator handles model selection). No internal classification labels in the output.
- **Token budget:** 1,200–1,600 tokens.
- **Cost drivers this prompt enables:** Input length variance (short question → short response, long complaint → long response). The word count ranges are the primary output cost lever. Model routing (Haiku vs Sonnet) is handled externally.

**Quality check:** Generate 3 test inputs (one simple, one complex, one edge) and verify the prompt would produce responses of varying length. The simple input should elicit ~150 words; the complex input should elicit ~350 words.

---

### W2 — Customer Support (Complex, with Loops)

**What it does:** Multi-step support with a research-and-draft loop. The agent takes a customer question, classifies it, enters a research/draft/self-assess loop (3–12 iterations), and optionally escalates to a final Opus review for complex cases.

**Steps:**
1. **Intake + Classify** — Haiku 4.5
2. **Research + Draft Loop** — Sonnet 4.6, iterates 3–12 times
3. **Final Review** — Opus 4.7, conditional (triggered when classification = "complex" AND iteration count ≥ 4)

**Step 1 prompt specification:**

- **Role:** Intake classifier for TechFlow support pipeline.
- **Task:** Read the customer message. Classify the question type and determine whether it needs research.
- **Classification schema:** `billing`, `technical`, `feature_request`, `account_access`, `outage_report`, `general_inquiry`, `complaint`.
- **[COST-CRITICAL]** Complexity determination: output `complexity: "simple"` or `complexity: "complex"`. Simple = single factual question answerable from product knowledge. Complex = multi-part question, requires cross-referencing policies, involves troubleshooting, emotional/escalated language, or references prior interactions.
- **Output format (JSON):**
```json
{
  "question_type": "string — one of the classification values",
  "complexity": "simple | complex",
  "needs_research": true,
  "key_topics": ["list of 1-3 topic tags"],
  "customer_sentiment": "neutral | frustrated | urgent"
}
```
- **Token budget:** 600–900 tokens.
- **Cost driver:** This step is cheap (Haiku, short output). Its cost contribution is minimal. Its purpose is to set the routing for downstream steps.

**Step 2 prompt specification:**

- **Role:** Research and drafting agent for TechFlow support. You are working iteratively to produce the best possible response.
- **Behavioral rules:**
  - You receive the customer's original question plus the classification from Step 1.
  - Each iteration, you produce a draft response AND a self-assessment.
  - **[COST-CRITICAL]** Self-assessment criteria (all five must be evaluated per iteration):
    1. Does the response address every part of the customer's question?
    2. Is every claim grounded in TechFlow product knowledge (no fabrication)?
    3. Is the tone appropriate to the customer's sentiment?
    4. Are next steps clear and actionable?
    5. Confidence score: 0.0–1.0.
  - **[COST-CRITICAL]** Loop termination: Stop iterating when confidence ≥ 0.9 OR when you have completed 12 iterations, whichever comes first. If confidence is below 0.7 after 6 iterations, flag for escalation.
  - Each iteration should meaningfully improve the response. Do not repeat the same draft with trivial changes.
  - The draft must include the TechFlow product context where relevant (pricing, features, policies).
- **Output format (JSON):**
```json
{
  "draft_response": "string — the customer-facing response",
  "self_assessment": {
    "addresses_all_parts": true,
    "factually_grounded": true,
    "tone_appropriate": true,
    "next_steps_clear": true,
    "confidence": 0.85
  },
  "should_continue": false,
  "iteration_notes": "string — what changed this iteration"
}
```
- **Context growth behavior:** Each iteration receives the FULL conversation so far: original question + classification + all prior drafts and assessments. This means context grows by ~500–1,000 tokens per iteration. By iteration 8, context is ~6,000–10,000 tokens. This is the primary cost driver for W2 and is by design — the context growth detector (Pearson/Spearman) must fire on this workflow.
- **What the prompt must include:** The full self-assessment rubric (five criteria). The confidence threshold. The maximum iteration cap. TechFlow domain knowledge (same as W1 but embedded in this prompt — the loop step doesn't have access to a separate knowledge base).
- **What the prompt must NOT include:** No instruction to "be brief" or "minimize iterations" — that would suppress loop variance. The prompt should encourage thoroughness.
- **Token budget:** 1,800–2,400 tokens (the longest prompt in the suite, because it includes domain knowledge + iteration rules + self-assessment rubric + output schema).
- **[COST-CRITICAL]** The escalation trigger at iteration 4 is what gates Opus usage. The prompt must include: "If after 4 iterations your confidence is below 0.75, flag the response for senior review."

**Step 3 prompt specification:**

- **Role:** Senior quality reviewer for TechFlow support responses.
- **Task:** You are reviewing a draft response that the support agent was not fully confident about. Assess quality and either approve, revise, or flag for human escalation.
- **Review criteria:** Accuracy, completeness, tone, professionalism, risk of misunderstanding.
- **Output format (JSON):**
```json
{
  "decision": "approve | revise | escalate_to_human",
  "revised_response": "string or null — only if decision is revise",
  "review_notes": "string — what you found",
  "quality_score": 0.92
}
```
- **Token budget:** 800–1,200 tokens.
- **Cost driver:** Opus 4.7 pricing makes this the most expensive per-token step. It runs conditionally (only on complex cases that didn't converge). The prompt should encourage thorough review (longer output) rather than terse approval, to produce realistic Opus output variance.

---

### W4 — Compliance/Document Review (Self-Reflection Loops)

**What it does:** Reviews a business document for compliance issues. Generates initial findings, self-critiques, revises. 2–8 iterations. Models: DeepSeek V4 (review and revision) + Qwen 3.6 Plus (critique).

**Steps:**
1. **Initial Review** — DeepSeek V4
2. **Self-Critique** — Qwen 3.6 Plus
3. **Revision** — DeepSeek V4. Loop to Step 2 until satisfied or max iterations.

**Step 1 prompt specification:**

- **Role:** Compliance analyst reviewing a business document.
- **Task:** Read the provided document. Identify all compliance issues, regulatory risks, ambiguous clauses, and missing required elements. Generate structured findings.
- **Compliance criteria to include in the prompt (the checklist the agent works from):**
  - For contracts: party identification, term/termination clarity, liability/indemnification, governing law, force majeure, confidentiality, data protection (GDPR/CCPA references), intellectual property, payment terms, dispute resolution.
  - For HR policies: non-discrimination compliance, leave policies (FMLA alignment), harassment definitions, reporting procedures, disciplinary process, at-will employment statement.
  - For regulatory filings: completeness of required disclosures, accuracy of numerical data, proper dating and signatures, cross-reference consistency.
- **[COST-CRITICAL]** Thoroughness instruction: "Check every section of the document, including boilerplate and appendices. Do not skip standard clauses — compliance issues often hide in boilerplate."
- **Output format (JSON):**
```json
{
  "findings": [
    {
      "id": "F1",
      "section_reference": "Section 4.2, paragraph 3",
      "issue": "string — description of the compliance issue",
      "severity": "critical | major | minor | observation",
      "regulatory_basis": "string — which regulation or standard this relates to",
      "recommendation": "string — suggested remediation",
      "confidence": 0.9
    }
  ],
  "document_summary": {
    "document_type": "contract | hr_policy | regulatory_filing | other",
    "total_sections_reviewed": 12,
    "overall_compliance_assessment": "compliant | minor_issues | major_issues | non_compliant"
  }
}
```
- **Token budget:** 1,400–1,800 tokens.
- **Cost driver:** Document length drives input tokens. Number of findings drives output tokens. Both vary with document complexity.

**Step 2 prompt specification:**

- **Role:** Compliance review quality assessor.
- **Task:** Review the findings generated by the initial review. Critique them on four dimensions.
- **[COST-CRITICAL]** Critique dimensions (all four must be evaluated):
  1. Missed issues: Are there compliance risks in the document that the initial review did not identify?
  2. Severity accuracy: Are the severity ratings appropriate? Is a "minor" issue actually "major"?
  3. Evidence quality: Is each finding supported by a specific section reference? Are the citations accurate?
  4. Recommendation quality: Are the recommendations actionable and proportionate?
- **[COST-CRITICAL]** Completion criterion: "If no critical or major issues are found with the review AND you identify no missed issues, output `satisfied: true`. Otherwise, output `satisfied: false` with specific feedback." This controls loop termination — a strict criterion means more iterations; a lenient one means fewer.
- **Output format (JSON):**
```json
{
  "satisfied": false,
  "critique": {
    "missed_issues": ["string — descriptions of issues the review missed"],
    "severity_corrections": [{"finding_id": "F1", "current": "minor", "suggested": "major", "reason": "string"}],
    "evidence_gaps": [{"finding_id": "F2", "issue": "string"}],
    "recommendation_improvements": [{"finding_id": "F3", "suggestion": "string"}]
  },
  "iteration_recommendation": "continue | finalize"
}
```
- **Token budget:** 1,000–1,400 tokens.

**Step 3 prompt specification:**

- **Role:** Compliance analyst (same as Step 1) incorporating review feedback.
- **Task:** Update the findings based on the critique. Only modify findings that the critique identified as problematic. Add any newly identified issues. Do not remove findings unless the critique explicitly states they are incorrect.
- **Output format:** Same JSON schema as Step 1.
- **Context growth behavior:** Each iteration appends the prior review + critique to context. By iteration 6, context includes the original document + 5 rounds of review/critique pairs. This is the cost driver.
- **Token budget:** 1,200–1,600 tokens (slightly shorter than Step 1 — the revision prompt doesn't need to re-state the full compliance checklist, just reference it).

**Cache-busting:** All three steps use the `{{CACHE_BUST_SUFFIX}}` placeholder (DeepSeek and Qwen).

---

### W5 — Multimodal Extraction + Structured Output

**What it does:** Extracts structured data from mixed-modality inputs (text, images, PDFs with embedded charts/tables). Single step. Output is JSON conforming to a schema.

**Steps:**
1. **Extract** — Sonnet 4.6 with vision enabled

**Step 1 prompt specification:**

- **Role:** Data extraction agent. You receive documents or images of business artifacts (invoices, receipts, business cards, forms, screenshots of tables) and extract structured data.
- **Per-modality instructions (must be in the prompt):**
  - For text documents: Extract fields directly from the text.
  - For images: Describe what you see first (document type, layout, quality), then extract fields. If text is handwritten, note confidence per field.
  - For tables (in images or PDFs): Extract row by row, preserving column headers as field names.
  - For mixed documents (text + embedded images): Process text sections by extraction, image sections by description + extraction. Combine into a single output.
- **Extraction schema (must be in the prompt — this is the schema the output conforms to):**
```json
{
  "document_type": "invoice | receipt | business_card | form | table | other",
  "extracted_fields": [
    {
      "field_name": "string",
      "field_value": "string",
      "confidence": 0.95,
      "source": "text | image | table",
      "location": "string — page/section/region description"
    }
  ],
  "metadata": {
    "total_fields_extracted": 8,
    "low_confidence_fields": 1,
    "modalities_present": ["text", "image"],
    "extraction_notes": "string — any issues or ambiguities"
  }
}
```
- **Common field names to include as examples in the prompt:** vendor_name, invoice_number, date, due_date, line_items (nested: description, quantity, unit_price, total), subtotal, tax, total_amount, payment_terms, billing_address, contact_name, contact_email, contact_phone, company_name.
- **[COST-CRITICAL]** JSON enforcement: "Respond ONLY with valid JSON. No explanation, no markdown fences, no preamble. Your entire response must be a single JSON object matching the schema above."
- **What the prompt must NOT include:** No instruction to limit the number of fields extracted (that would suppress output variance — a 5-field invoice is cheap, a 15-field receipt is expensive, and we need both).
- **Token budget:** 1,400–1,800 tokens.
- **Cost driver:** Input modality. Text-only inputs: ~300–800 input tokens. Image inputs: ~1,500–6,400 input tokens per image (vision token calculation depends on resolution and tiling). Number of fields to extract drives output length.

---

### W9 — Sales/Outreach (OpenAI)

**What it does:** Lead qualification and personalized email drafting. Two steps, both using OpenAI models.

**Steps:**
1. **Qualify** — GPT-5.4 Nano
2. **Draft Email** — GPT-5.4

**Step 1 prompt specification:**

- **Role:** Lead qualification analyst for NovaCRM's sales team.
- **Task:** Score the incoming lead as hot, warm, or cold based on their profile data.
- **Qualification criteria (must be in the prompt):**
  - Company size: >500 employees = +2, 100–500 = +1, <100 = 0
  - Industry: SaaS, fintech, healthcare = +2 (high-fit verticals), manufacturing, retail = +1, other = 0
  - Recent signals: funding round in last 6 months = +2, job postings for "AI/ML" roles = +1, tech blog posts = +1
  - Tech stack indicators: uses Salesforce or HubSpot = +1 (CRM-aware), uses competitor product = +2 (switching opportunity)
  - Engagement: visited pricing page = +2, downloaded whitepaper = +1, attended webinar = +1
  - Scoring: 7+ = hot, 4–6 = warm, 0–3 = cold
- **Output format (JSON):**
```json
{
  "score": 7,
  "rating": "hot | warm | cold",
  "score_breakdown": {"company_size": 2, "industry": 2, "recent_signals": 2, "tech_stack": 1, "engagement": 0},
  "key_talking_points": ["Recent Series B funding", "Active Salesforce users"],
  "recommended_approach": "string — brief strategy note"
}
```
- **Token budget:** 800–1,100 tokens.

**Step 2 prompt specification:**

- **Role:** Sales development representative for NovaCRM. Your name is Jordan. Your title is "Senior Account Executive."
- **Task:** Draft a personalized outreach email based on the lead profile and qualification score.
- **Personalization rules:**
  - Hot leads: Direct, confident, reference specific pain points. Include a specific calendar link CTA.
  - Warm leads: Educational, value-first. Reference a relevant case study or ROI metric.
  - Cold leads: Soft touch, industry-relevant insight, no hard CTA. Position as "thought leadership."
- **[COST-CRITICAL]** Length constraints:
  - Hot: 120–180 words. Direct and action-oriented.
  - Warm: 150–220 words. Value proposition focused.
  - Cold: 80–120 words. Brief and non-pushy.
- **Email structure:** Subject line, greeting (use first name if available, "Hi there" if not), 2–3 paragraphs, CTA, sign-off.
- **What the prompt must include:** The personalization rules with concrete examples of each approach. NovaCRM's value propositions: "40% faster pipeline velocity," "AI-powered lead scoring," "integrates with Salesforce and HubSpot in under 10 minutes."
- **Output format (JSON):**
```json
{
  "subject_line": "string",
  "email_body": "string — the full email text",
  "personalization_elements_used": ["string — which profile data points were referenced"],
  "word_count": 165
}
```
- **Token budget:** 1,000–1,400 tokens.
- **Cost driver:** Lead profile length (input tokens), email length by rating tier (output tokens). Hot leads have shorter emails but more complete profiles; cold leads have sparse profiles but shorter emails.

---

### W11 — Support (Qwen)

**What it does:** Same structure and content as W1, but using Qwen models. Tests Qwen pricing and tokenizer differences. This is the cross-provider twin of W1 — running identical inputs through both enables direct cost comparison.

**Steps:**
1. **Classify + Respond** — Qwen-Turbo or Qwen 3.6 Plus (routed by orchestrator)

**Prompt specification:** Generate the same prompt as W1 with these adaptations for Qwen models:
- Reinforce JSON compliance if structured output is involved (Qwen models sometimes prepend explanatory text).
- Reinforce output language: "Respond in English only" (Qwen models may default to Chinese for certain prompt patterns).
- The domain content (TechFlow), behavioral rules, and word count constraints must be identical to W1 to enable valid cross-provider comparison. The only differences should be model-specific instruction reinforcement.
- **Token budget:** Same as W1 (1,200–1,600 tokens). Note that Qwen's tokenizer may produce a different token count from the same text — this is expected and is part of what the cross-provider comparison measures.

---

### W12 — Data Extraction (DeepSeek)

**What it does:** Simple text extraction pipeline. Text document in, structured fields out. Uses DeepSeek V4 Flash. Tests DeepSeek pricing and cache behavior. No vision — text only.

**Steps:**
1. **Extract** — DeepSeek V4 Flash

**Step 1 prompt specification:**

- **Role:** Data extraction agent. You receive text documents (corporate reports, structured business communications, financial summaries) and extract key entities and structured data.
- **Extraction schema (similar to W5 but text-only, more focused on entity extraction):**
```json
{
  "document_type": "report | memo | financial_summary | correspondence | other",
  "entities": [
    {
      "entity_type": "person | organization | date | monetary_value | percentage | location | product",
      "value": "string",
      "context": "string — the sentence or phrase where this entity appears",
      "confidence": 0.95
    }
  ],
  "key_facts": [
    {
      "fact": "string — a key assertion or data point",
      "source_section": "string — where in the document this comes from"
    }
  ],
  "summary": "string — 2-3 sentence document summary"
}
```
- **[COST-CRITICAL]** JSON enforcement (especially important for DeepSeek, which can be verbose): "Your entire response must be a single JSON object. No explanation. No markdown. No commentary outside the JSON structure."
- **What the prompt must include:** The entity type taxonomy. Examples of each entity type in context. Instructions for handling ambiguous entities ("if a value could be either a person or an organization, include it as both with a note").
- **Token budget:** 1,000–1,400 tokens.
- **Cache-busting:** `{{CACHE_BUST_SUFFIX}}` at end of prompt.
- **Cost driver:** Document length (input tokens, dominant cost at DeepSeek pricing). Output length is relatively constrained by the JSON schema. This workflow tests the simplest cost model: input-dominated, low variance, linear scaling.

---

### W13 — Routing/Conditional Agent

**What it does:** Classifies input, routes to one of three paths with very different cost profiles. Tests step count variance and bimodality detection.

**Steps:**
1. **Classify** — Haiku 4.5
2. **Path A — Simple Response** — Haiku 4.5 (target: 70% of inputs)
3. **Path B — Moderate Analysis** — Sonnet 4.6 (target: 20% of inputs)
4. **Path C — Complex with Tools** — Sonnet 4.6 + tool calls (target: 10% of inputs)

**Step 1 prompt specification:**

- **Role:** Query router. You classify incoming questions to determine the appropriate processing path.
- **[COST-CRITICAL]** Classification criteria (these control the 70/20/10 routing split):
  - **TIER_1 (Path A):** Simple factual question. Answerable in 1–2 sentences. Single data point requested. Examples: "What time does the store close?", "What's the capital of France?", "How many days in February?"
  - **TIER_2 (Path B):** Analytical question requiring reasoning. Involves comparison, explanation, or multi-step logic. Examples: "Compare the pros and cons of electric vs hybrid cars", "Explain why the stock market dropped today", "What are the tax implications of selling a rental property?"
  - **TIER_3 (Path C):** Complex question requiring external information lookup or multi-source synthesis. Involves current data, specific calculations, or information the model doesn't have. Examples: "What's the current exchange rate for USD to EUR and how has it changed this month?", "Find the cheapest flight from NYC to London next Tuesday", "Calculate the compound interest on $10,000 at 4.5% over 7 years with monthly compounding."
- **Output format (JSON):**
```json
{
  "tier": "TIER_1 | TIER_2 | TIER_3",
  "reasoning": "string — one sentence explaining the classification",
  "confidence": 0.95
}
```
- **What the prompt must include:** At least 3 examples per tier. The boundary between tiers must be unambiguous for clear-cut cases (the inputs are designed to trigger specific tiers — the routing must be deterministic for non-edge inputs).
- **Token budget:** 800–1,100 tokens.

**Step 2 (Path A) prompt specification:**

- **Role:** Quick-answer assistant. Provide a concise, direct answer.
- **[COST-CRITICAL]** Length constraint: Respond in 1–3 sentences. Maximum 75 words. Do not provide analysis, context, or caveats unless the question specifically asks for them.
- **Output format:** Plain text.
- **Token budget:** 400–600 tokens.

**Step 3 (Path B) prompt specification:**

- **Role:** Analytical assistant. Provide a thorough, reasoned response.
- **Behavioral rules:** Structure your response with clear reasoning. Consider multiple perspectives where relevant. Provide evidence or examples to support your analysis.
- **[COST-CRITICAL]** Length constraint: Respond in 150–350 words. Use paragraphs, not bullet points.
- **Output format:** Plain text.
- **Token budget:** 600–900 tokens.

**Step 4 (Path C) prompt specification:**

- **Role:** Research assistant with tool access. You have access to external tools to look up information.
- **Tool schemas (must be included in the prompt as function definitions):**
```json
[
  {
    "name": "web_search",
    "description": "Search the web for current information",
    "parameters": {"query": "string"}
  },
  {
    "name": "calculator",
    "description": "Perform mathematical calculations",
    "parameters": {"expression": "string — mathematical expression"}
  },
  {
    "name": "unit_converter",
    "description": "Convert between units",
    "parameters": {"value": "number", "from_unit": "string", "to_unit": "string"}
  }
]
```
- **Behavioral rules:** Use tools when the question requires information you don't have or calculations that must be precise. Explain your reasoning and cite the tool results in your response.
- **[COST-CRITICAL]** Length constraint: Respond in 200–500 words including tool result integration.
- **Output format:** Plain text with tool calls as needed.
- **Token budget:** 900–1,300 tokens.
- **Cost driver:** Path C includes tool call schemas in the prompt (~200–400 tokens of tool definitions), and tool calls add to both input and output tokens. The cost gap between Path A (~$0.003) and Path C (~$0.08) is what creates bimodality in the cost distribution.

---

### W14 — Simple PDF RAG + Structured Output

**What it does:** User uploads PDFs, asks a question, system retrieves relevant chunks and generates a structured answer. No loops. Cross-provider: OpenAI embeddings + Sonnet generation.

**Steps:**
1. **PDF processing** — non-LLM for text pages, Claude Vision for scanned/image pages (chunking and extraction)
2. **Embed query** — OpenAI text-embedding-3-small (no system prompt needed)
3. **Retrieve** — vector search, no LLM (no system prompt needed)
4. **Generate answer** — Sonnet 4.6 in JSON mode

**Step 4 prompt specification (the only step requiring a system prompt):**

- **Role:** Insurance policy analyst. You answer questions about health insurance policies using ONLY the provided context from retrieved document sections.
- **Behavioral rules:**
  - Answer using ONLY the provided context. If the context does not contain sufficient information, say so explicitly — do not fabricate coverage details, dollar amounts, or policy rules.
  - Cite specific sources: "[Source: {document_name}, page {X}]" for every claim.
  - If the context contains conflicting information across sources, note the conflict and present both positions.
  - If the question asks about a specific insurance provider, only cite that provider's documents.
- **[COST-CRITICAL]** The retrieved context is injected into the user message, not the system prompt. The system prompt defines behavior only. Context size varies from ~500 tokens (1–2 chunks, easy) to ~20,000 tokens (15+ chunks, hard) — this is the primary cost driver.
- **Output format (JSON):**
```json
{
  "answer": "string — the response to the question",
  "confidence": "high | medium | low",
  "citations": [
    {
      "document": "string — document name",
      "page": "number",
      "relevant_text_summary": "string — brief paraphrase of the cited section"
    }
  ],
  "missing_info": "string | null — what information would improve the answer"
}
```
- **What the prompt must NOT include:** No policy content in the system prompt (it comes from retrieval). No instruction to limit citation count (more citations = more output tokens = cost variance).
- **Token budget:** 1,000–1,400 tokens.

---

### W15 — Agentic Multi-Hop PDF RAG

**What it does:** Multi-hop retrieval over PDFs. Retrieves, assesses sufficiency, re-retrieves if needed. 1–4 hops. Three providers: OpenAI embeddings + Gemini 2.5 Flash (sufficiency assessment) + DeepSeek V4 (final generation).

**Steps:**
1. **PDF processing + embedding** — same as W14
2. **Initial retrieval** — vector search (no prompt)
3. **Assess sufficiency** — Gemini 2.5 Flash (iterates with step 4 for up to 4 hops)
4. **Re-retrieve** — vector search with reformulated query (no prompt)
5. **Generate answer** — DeepSeek V4

**Step 3 prompt specification (sufficiency assessment):**

- **Role:** Information sufficiency assessor. You determine whether the retrieved context is sufficient to answer the user's question comprehensively.
- **Task:** Given the question and the retrieved context so far, evaluate:
  1. Do you have sufficient information to answer comprehensively? (yes/no)
  2. If no, what specific information is missing?
  3. Generate a refined search query targeting the missing information.
- **[COST-CRITICAL]** Sufficiency threshold: "Answer 'sufficient: true' only when ALL parts of the question can be answered with specific evidence from the context. If any part requires speculation or inference beyond what the context states, answer 'sufficient: false'." This threshold controls hop count — a strict threshold means more hops.
- **[COST-CRITICAL]** Maximum hops: "If this is the 4th retrieval round, answer 'sufficient: true' regardless and note the information gaps in your response." This prevents infinite retrieval loops.
- **Output format (JSON):**
```json
{
  "sufficient": false,
  "assessment": "string — what the context covers and what it lacks",
  "missing_info": ["string — specific information gaps"],
  "refined_query": "string — a search query targeting the missing information",
  "hop_number": 2
}
```
- **Token budget:** 800–1,100 tokens.
- **Context growth:** Each hop appends all prior retrieved chunks to the context. By hop 4, context may contain 4 rounds of retrieval results. The assessment step receives all accumulated context — this is where context growth cost compounds.

**Step 5 prompt specification (final generation):**

- **Role:** Insurance policy analyst. Same behavioral rules as W14 Step 4, but with a multi-retrieval context.
- **Additional instruction:** "You are receiving context from {N} retrieval rounds. Some information may be redundant across rounds. Synthesize into a single coherent answer. Do not repeat information."
- **Output format (JSON):**
```json
{
  "retrieval_rounds_used": 3,
  "answer": "string",
  "confidence": "high | medium | low",
  "citations": [
    {"document": "string", "page": "number", "relevant_text_summary": "string"}
  ],
  "missing_info": "string | null"
}
```
- **Token budget:** 1,000–1,400 tokens.
- **Cache-busting:** `{{CACHE_BUST_SUFFIX}}` on the DeepSeek generation step.

---

### W16 — Map-Reduce PDF Analysis

**What it does:** Splits a long PDF into N sections, processes each in parallel (Haiku), aggregates results (Sonnet). N varies with document length (3–20).

**Steps:**
1. **Split** — Sonnet 4.6
2. **Process** — Haiku 4.5 (N parallel calls)
3. **Aggregate** — Sonnet 4.6

**Step 1 prompt specification:**

- **Role:** Document structure analyst.
- **Task:** Read the provided document and identify its major sections. For each section, provide: title, page range, and a one-sentence description of its content.
- **Rules:**
  - Identify natural section boundaries (headings, topic shifts, chapter breaks).
  - Minimum section size: 1 page. Maximum: 10 pages.
  - If the document has no clear structure, divide into roughly equal segments of 3–5 pages each.
  - **[COST-CRITICAL]** The number of sections you identify determines the number of parallel processing calls. Err toward more granular sections for long documents (>30 pages) and fewer for short ones (<10 pages).
- **Output format (JSON):**
```json
{
  "sections": [
    {
      "id": 1,
      "title": "string",
      "page_range": [1, 5],
      "description": "string — one sentence"
    }
  ],
  "total_sections": 8,
  "document_structure_quality": "well_structured | partially_structured | unstructured"
}
```
- **Token budget:** 800–1,100 tokens.

**Step 2 prompt specification (per-section processing):**

- **Role:** Section analyst. You are analyzing one section of a larger document. Other sections are being analyzed in parallel.
- **Task:** Extract key findings, data points, risks identified, and recommendations from this section. Be thorough but concise — your output will be combined with analyses of other sections.
- **Output format (JSON):**
```json
{
  "section_title": "string",
  "key_findings": ["string — each a substantive finding"],
  "data_points": [{"metric": "string", "value": "string", "context": "string"}],
  "risks": ["string — each a specific risk"],
  "recommendations": ["string — each an actionable recommendation"],
  "section_summary": "string — 2-3 sentences"
}
```
- **What the prompt must NOT include:** No reference to other sections (the processor only sees its section). No instruction to cross-reference (that's the aggregator's job).
- **Token budget:** 600–900 tokens.
- **Cost driver:** Section length (input tokens per parallel call) × N (number of parallel calls). Short documents → few calls with short input. Long documents → many calls with longer input.

**Step 3 prompt specification (aggregation):**

- **Role:** Senior analyst producing a final report. You are receiving analyses of {N} document sections from parallel reviewers.
- **Task:** Combine all section analyses into a coherent summary. Resolve contradictions between sections. Highlight the 3 most important findings across the entire document. Identify cross-section themes.
- **Output format (JSON):**
```json
{
  "executive_summary": "string — 200-400 words",
  "top_findings": [
    {
      "finding": "string",
      "source_sections": [1, 3],
      "importance": "critical | high | medium"
    }
  ],
  "cross_section_themes": ["string"],
  "contradictions": [
    {
      "section_a": 2,
      "section_b": 5,
      "description": "string"
    }
  ],
  "overall_assessment": "string — one paragraph"
}
```
- **Token budget:** 1,000–1,400 tokens.
- **Cost driver:** The aggregation step receives ALL N section analyses as input. For N=15, this could be 10,000+ tokens of input. This is the second cost lever after the fan-out itself.

---

### W17 — Insurance Claims Review Agent

**Full architecture is specified in `projection-engine-recommendation-addition-2.md`.** Refer to that document for the complete step-by-step design, override rules, claim type routing, and function call schema. Generate prompts consistent with that specification.

**Steps:**
1. **Intake + Override Check** — Haiku 4.5
2. **Policy Retrieval** — OpenAI embeddings + vector search (no generation prompt)
3. **Evaluate + Decide** — Sonnet 4.6 with JSON mode
4. **Conditional Routing** — Haiku 4.5 (only when flagged)

**Step 1 prompt specification:**

- **Role:** Claims intake processor.
- **Task:** Read the incoming claim JSON. Run four override checks before any policy consultation.
- **[COST-CRITICAL]** Override rules (these control short-circuit behavior — they determine whether the pipeline ends at Step 1 or continues to Steps 2–4):
  1. `member_status == "inactive"` → short-circuit to `deny_claim()` with reason "Plan inactive at time of service." Pipeline ends.
  2. Essential documentation missing (no `clinical_note` for pre-approval claims, no `itemized_bill` for standard claims) → short-circuit to `request_missing_documentation()` with a list of what's needed. Pipeline ends.
  3. `claimed_amount > 5000` → flag for senior reviewer routing at Step 4. Pipeline continues.
  4. Diagnosis code / procedure code inconsistency (e.g., diagnosis is "broken arm" but procedure is "cardiac MRI") → flag for coding review routing at Step 4. Pipeline continues.
- **If no short-circuit fires:** Classify claim type as `pre_approval`, `standard`, or `appeal` based on the claim data.
- **Output format (JSON):**
```json
{
  "overrides_triggered": [
    {"rule": "inactive_member | missing_docs | high_amount | code_inconsistency", "action": "string", "details": "string"}
  ],
  "short_circuit": true,
  "short_circuit_action": {"action": "deny_claim", "reason": "Plan inactive at time of service."},
  "claim_type": "pre_approval | standard | appeal | null",
  "proceed_to_retrieval": false,
  "flags": ["high_amount"]
}
```
- **Token budget:** 800–1,200 tokens.

**Step 3 prompt specification:**

- **Role:** Claims evaluation specialist. You assess insurance claims against policy evidence.
- **Task:** Given the claim JSON and retrieved policy sections, evaluate the claim and decide the next action.
- **Claim-type-specific evaluation logic (must be in the prompt):**
  - **Pre-approval:** Check coverage criteria, verify documentation completeness (clinical notes, prior treatment history), assess medical necessity against policy standard ("Is this procedure medically necessary based on the documented diagnosis and treatment history?"). Decide: `approve_pre_authorization()` or `deny_claim()`.
  - **Standard:** Validate coverage, check procedure/diagnosis code consistency, verify no exclusions apply (pre-existing condition exclusions, experimental treatment exclusions), confirm in-network status if relevant. Decide: `approve_claim_payment(payment_in_dollars)` or `deny_claim()`.
  - **Appeal:** Retrieve prior denial reason from claim data. Assess whether the appeal documentation is (a) new, (b) sufficient, and (c) relevant to the denial reason. Re-evaluate medical justification with new evidence. Decide: `approve_claim_payment()` (overturn) or `deny_claim()` (uphold).
- **[COST-CRITICAL]** Function call schema (all 6 actions — must be included as tool/function definitions in the prompt):
```json
[
  {"name": "approve_pre_authorization", "parameters": {}},
  {"name": "approve_claim_payment", "parameters": {"payment_in_dollars": "float"}},
  {"name": "deny_claim", "parameters": {}},
  {"name": "request_missing_documentation", "parameters": {"documents": ["string"]}},
  {"name": "route_to_senior_reviewer", "parameters": {}},
  {"name": "route_to_coding_review", "parameters": {}}
]
```
- **Output format (JSON):**
```json
{
  "action": "approve_claim_payment",
  "action_params": {"payment_in_dollars": 2200.0},
  "reason": "string — grounded in policy evidence",
  "evidence": ["string — specific policy section citations"],
  "confidence": "high | medium | low",
  "flags": []
}
```
- **Token budget:** 1,800–2,400 tokens (the longest prompt in W17, because it includes all three claim-type evaluation paths + all 6 function schemas).
- **Cost driver:** Retrieved policy context size (500–4,000 tokens) drives input tokens. Evaluation complexity drives output tokens (simple approve vs. detailed denial reasoning).

**Step 4 prompt specification:**

- **Role:** Claims routing decision wrapper.
- **Task:** The evaluation step has produced a decision, but one or more flags were raised during intake (high amount, code inconsistency). Wrap the evaluation decision with the appropriate routing action.
- **Rules:** If `high_amount` flag → add `route_to_senior_reviewer()` to the output alongside the evaluation decision. If `code_inconsistency` flag → add `route_to_coding_review()`. If both → route to senior reviewer (takes precedence).
- **Output format (JSON):**
```json
{
  "evaluation_decision": { "...same as Step 3 output..." },
  "routing_action": "route_to_senior_reviewer",
  "routing_reason": "Claimed amount exceeds $5,000 threshold"
}
```
- **Token budget:** 500–800 tokens.

---

### W18 — Long-Document Single-Pass

**What it does:** Processes an entire long PDF (30–100 pages, 30K–100K tokens) in a single context window. No splitting, no loops. The simplest workflow architecture at the highest token scale.

**Steps:**
1. **Process** — DeepSeek V4

**Step 1 prompt specification:**

- **Role:** Senior analyst performing comprehensive document review.
- **Task:** You are given a complete long-form document. Perform all of the following:
  1. **Executive summary** (200–400 words): The most important takeaways.
  2. **Key findings** (5–10 items): Specific, quantified findings from the document. Cite page numbers.
  3. **Risks and concerns** (3–7 items): Risks, issues, or gaps identified.
  4. **Recommendations** (3–5 items): Actionable next steps.
- **[COST-CRITICAL]** Output length: The output should be 800–2,000 tokens regardless of document length. A 30-page document and a 100-page document should produce similarly-sized output. This means cost is almost entirely input-token-dominated.
- **Output format (JSON):**
```json
{
  "executive_summary": "string",
  "key_findings": [
    {"finding": "string", "page_reference": "number", "importance": "high | medium"}
  ],
  "risks": [
    {"risk": "string", "page_reference": "number", "severity": "critical | major | minor"}
  ],
  "recommendations": [
    {"recommendation": "string", "related_findings": ["F1", "F3"]}
  ],
  "document_metadata": {
    "estimated_pages": 45,
    "primary_topic": "string",
    "document_type": "annual_report | legal_deposition | technical_spec | regulatory_filing"
  }
}
```
- **What the prompt must NOT include:** No instruction to summarize section-by-section (that would make output length proportional to input length, changing the cost model from input-dominated to mixed).
- **Token budget:** 1,000–1,400 tokens.
- **Cache-busting:** `{{CACHE_BUST_SUFFIX}}` at end of prompt.
- **Cost driver:** Document token count (30K–100K). This is a pure input-scaling test. The projection engine must handle the variance in input tokens correctly — a 3× range in input tokens should produce a ~3× range in per-run costs.

---

### W19 — Multi-Turn Conversation (8 Turns)

**What it does:** Simulates an 8-turn support conversation. Each turn accumulates full conversation history in context. Uses DeepSeek V4.

**Steps (repeated 8 times):**
1. **Respond** — DeepSeek V4

**Step 1 prompt specification (same prompt used for all 8 turns):**

- **Role:** Customer support agent for CloudOps, a cloud infrastructure management platform.
- **CloudOps domain knowledge to include:**
  - Products: CloudOps Dashboard (monitoring, real-time alerts, 99.9% uptime SLA), CloudOps Deploy (CI/CD pipelines, blue-green deployments, rollback in <30 seconds), CloudOps Scale (auto-scaling, predictive scaling based on traffic patterns, supports Kubernetes and ECS), CloudOps Guard (security scanning, IAM management, compliance reports).
  - Pricing: Starter $99/mo (up to 5 services monitored), Pro $399/mo (up to 25 services, priority support), Enterprise custom pricing (unlimited services, dedicated support engineer, SLA guarantees).
  - Common issues: deployment failures (rolling back, checking logs), scaling alerts (threshold tuning, burst capacity), monitoring gaps (adding custom metrics, dashboard setup), integration setup (AWS, GCP, Azure connectors), billing questions, incident response procedures.
  - Architecture concepts agents should discuss knowledgeably: containers, Kubernetes pods/services/deployments, load balancers (ALB/NLB), CDN configuration, database replication (primary-replica), CI/CD pipeline stages, environment variables, secrets management.
- **Behavioral rules:**
  - Maintain full context from prior turns. Reference previous statements when relevant ("As I mentioned earlier...").
  - If the customer's latest message contradicts something from earlier, note the discrepancy politely.
  - Track the state of the issue across turns: open → investigating → identified → resolved.
  - **[COST-CRITICAL]** Response length: keep responses under 200 words per turn. This constrains output tokens so the dominant cost driver is input tokens (accumulated history), which grow linearly across turns.
  - Be helpful but do not over-promise. If an issue requires engineering escalation, say so.
- **Context growth behavior:** The prompt is sent with every turn. Turn 1 context: system prompt (~1,200 tokens) + user message (~200 tokens) ≈ 1,400 tokens. Turn 8 context: system prompt + 7 prior exchanges (~8,000–10,000 tokens) + user message ≈ 9,200–11,200 tokens. Input tokens grow linearly with turn number. This linear growth is the primary pattern the context growth detector must identify.
- **What the prompt must include:** Enough domain knowledge to generate substantive, varied responses (not just "I'll look into that"). The agent needs to discuss CloudOps features, troubleshooting steps, and architecture concepts to produce realistic response lengths and avoid collapsing into generic filler.
- **What the prompt must NOT include:** No instruction to "keep context short" or "summarize prior conversation." The full history must be re-sent every turn — this is the cost pattern being tested.
- **Token budget:** 1,200–1,600 tokens.
- **Cache-busting:** `{{CACHE_BUST_SUFFIX}}` at end of prompt, unique per turn (8 different suffixes per conversation). The harness must generate a new suffix for each API call, not reuse one suffix across the 8 turns of a single conversation.

---

## Quality Verification Protocol

After generating all prompts, Claude Code must verify each prompt against these criteria:

### Structural Checks (automated)

1. **Token count:** Measure each prompt's token count using the target model's tokenizer (or tiktoken cl100k_base as proxy). Verify it falls within the specified budget range. Flag any prompt outside range.
2. **JSON schema presence:** For every step that requires JSON output, verify the prompt contains a complete JSON schema example with all required fields.
3. **Cost-critical elements:** Verify every element marked `[COST-CRITICAL]` in this spec appears in the generated prompt. List any missing elements.
4. **Cache-bust placeholder:** Verify every DeepSeek and Anthropic prompt ends with the `<!-- session: {{CACHE_BUST_SUFFIX}} -->` placeholder.
5. **Domain content:** Verify each prompt contains the specified domain content (product names, pricing, features). A prompt that says "your company's products" instead of naming TechFlow/NovaCRM/CloudOps specific products will produce generic, short responses that suppress cost variance.

### Behavioral Checks (manual review of 3 prompts minimum)

6. **Output length variance potential:** For each prompt, mentally simulate a simple input and a complex input. Would the prompt produce meaningfully different output lengths? If the prompt over-constrains output (e.g., "always respond in exactly 100 words"), it will suppress variance.
7. **Loop termination clarity:** For W2 and W4, verify the loop termination condition is unambiguous. The model must be able to determine, from the prompt alone, when to stop iterating.
8. **Routing determinism:** For W13 and W17, verify the classification criteria are specific enough that clear-cut inputs route deterministically. Ambiguity should only occur on deliberately ambiguous edge-case inputs.
9. **Self-containment:** Each prompt must be usable without any external knowledge. If the model needs to know TechFlow's pricing to answer a billing question, that pricing must be in the prompt, not assumed.

### Cross-Workflow Consistency Checks

10. **W1 ↔ W11:** These two workflows must have identical prompt content (domain knowledge, behavioral rules, word count constraints) to enable valid cross-provider cost comparison. The only differences should be model-specific instruction reinforcement (JSON compliance, language enforcement).
11. **W14 ↔ W15 generation steps:** Both use the same domain (health insurance) and similar output schemas. The generation prompts should be consistent in citation format and behavioral rules, differing only in multi-hop context handling instructions.
12. **W17 consistency with spec:** The W17 prompts must match the full architecture in `projection-engine-recommendation-addition-2.md`. Cross-check: override rules (4 conditions), claim types (3 types), function call schema (6 actions), routing logic.

---

## Expected Detector Activation by Workflow

This table (from `cross-cutting-robustness.md`) is the acceptance criterion for whether the system prompts and inputs together produce the right cost patterns. If the prompts suppress a pattern that should appear, they need revision.

| Detector | Must fire | Must NOT fire |
|---|---|---|
| Context growth (Pearson/Spearman) | W2, W4, W19 (strong), W15 (moderate) | W1, W5, W9, W11, W12, W18 |
| Loop count variance | W2, W4, W15 | W1, W5, W9, W11, W12, W14, W18 |
| High token variance | W5 (vision tokens), W18 (input length), W16 (variable N) | W9 (constrained), W12 (constrained) |
| Step count variance | W2, W13, W15, W16, W17 | W1, W5, W9, W11, W12, W18, W19 |
| Bimodality (BIC) | W13 (strong), W17 (strong), W15 (moderate) | W1, W9, W11, W12, W18 |
| Linear projection (no patterns) | W1, W9, W11, W12, W18 | W2, W4, W13, W15, W17, W19 |

If a prompt's design would suppress an expected detector (e.g., a W2 prompt that terminates loops too eagerly, producing only 3 iterations every time), the prompt must be revised before the pilot.

---

## File Manifest

Generate the following files:

```
prompts/
  w01_support_simple/
    step1_classify_respond.txt
  w02_support_complex/
    step1_intake_classify.txt
    step2_research_draft_loop.txt
    step3_final_review.txt
  w04_compliance_review/
    step1_initial_review.txt
    step2_self_critique.txt
    step3_revision.txt
  w05_multimodal_extraction/
    step1_extract.txt
  w09_sales_outreach/
    step1_qualify.txt
    step2_draft_email.txt
  w11_support_qwen/
    step1_classify_respond.txt
  w12_extraction_deepseek/
    step1_extract.txt
  w13_routing_agent/
    step1_classify.txt
    step2_path_a_simple.txt
    step3_path_b_moderate.txt
    step4_path_c_complex.txt
  w14_simple_rag/
    step4_generate_answer.txt
  w15_multihop_rag/
    step3_assess_sufficiency.txt
    step5_generate_answer.txt
  w16_map_reduce/
    step1_split.txt
    step2_process_section.txt
    step3_aggregate.txt
  w17_claims_agent/
    step1_intake_override.txt
    step3_evaluate_decide.txt
    step4_conditional_routing.txt
  w18_long_document/
    step1_process.txt
  w19_multi_turn/
    step1_respond.txt
```

Each `.txt` file contains the raw prompt text — no metadata wrapper, no markdown, just the system prompt content that gets sent to the API. Include the `<!-- COST-CRITICAL -->` inline annotations and the `<!-- session: {{CACHE_BUST_SUFFIX}} -->` placeholder as specified.

Generate a companion `prompts/manifest.json` listing every prompt file with: workflow ID, step name, target model, measured token count (actual, from tokenizer), token budget range, and the list of cost-critical elements present in that prompt.
