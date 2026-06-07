# Directions: Input Generation for All 14 Backtesting Workflows

**Purpose:** This file specifies how Claude Code must build the input generation system — the code that produces two distinct input sets per workflow: a **profiling set** (n=50) and a **ground truth set** (n=500). The designed drift between these two sets is what the backtesting measures. If the projection engine works, it should project accurately from the profiling set; if it's fragile, the drift will expose that.

**Context for Claude Code:** You have the AgentCost codebase, `projection-engine-recommendation-addition-2.md` (engine design, stratified reweighting A5, `--traffic-mix` flag), `cross-cutting-robustness.md` (dirty input table, non-uniformity requirements, detector validation matrix), and the technical spec. This file provides the per-workflow input specifications and the drift design. Treat it as authoritative for what inputs look like.

**What you are building:** A generator module per workflow that takes a `profile` parameter (`"profiling"` or `"ground_truth"`) and produces tagged, diverse inputs. Each input carries a tier label, a token count, and a structural descriptor. The generator is deterministic given a seed — same seed, same inputs.

---

## Part 1: Dual Distribution Framework

### Why Two Distributions

In production, profiling inputs are curated (the user picks representative examples). Production traffic is organic (users send whatever they want). The distribution shift between profiling and production is the primary source of projection error. The backtesting quantifies this error by designing a known, measurable drift and checking whether the engine handles it.

### Profiling Distribution (n=50)

Represents the "curated test" a careful user would run. Inputs are:
- Clean, well-formed, grammatically correct
- Moderate in length (not abnormally short or long)
- Distributed across difficulty tiers at weights that over-represent harder cases (to get signal on expensive paths)

**Tier weights:** 40% easy / 35% medium / 20% hard / 5% edge = 4 tiers, 50 inputs.

That gives: 20 easy, 18 medium (round from 17.5), 10 hard, 2 edge.

### Ground Truth Distribution (n=500)

Represents realistic production traffic. Inputs are:
- Messier (typos, informal phrasing, mixed formatting)
- Longer on average (users in production write more than test users)
- Skewed toward the easy end (most production traffic is simple)
- Includes a 5th "extreme" tier that profiling never saw

**Tier weights:** 55% easy / 25% medium / 12% hard / 5% edge / 3% extreme = 5 tiers, 500 inputs.

That gives: 275 easy, 125 medium, 60 hard, 25 edge, 15 extreme.

### The Extreme Tier

The 5th tier exists only in the ground truth. These are production outliers the profiling user would never think to test: context-window-filling documents, pathological loop triggers, combinations that exercise the most expensive code path. They represent ~3% of production traffic but can account for 15–30% of monthly cost. The projection engine must handle them through tail-risk detection (CVaR, A4) — it won't have seen them during profiling.

### Tier Labels and Reweighting

Every generated input must carry a `tier` label: `easy`, `medium`, `hard`, `edge`, or `extreme`. These labels feed the stratified analysis (A5) and the reweighting validation. The reweighting test tells the engine: "production traffic has weights 55/25/12/5/3 instead of 40/35/20/5" and checks if the reweighted projection from the profiling set closes the accuracy gap with the ground truth. If it does, the engine's `--traffic-mix` flag works. If it doesn't, the drift is structural, not just distributional.

---

## Part 2: Three Drift Dimensions

All three dimensions apply uniformly across workflows unless a per-workflow exception is noted.

### Dimension 1: Tier Weight Shift

Profiling: 40/35/20/5 (over-represents hard cases).
Ground truth: 55/25/12/5/3 (production skew toward easy, adds extreme tier).

This is the primary drift lever. It changes the mix of cheap vs. expensive runs. The projection engine should detect this through stratified analysis (A5) — if it can reweight, it compensates.

### Dimension 2: Tone and Style Shift

**Profiling inputs:** Clean. Proper grammar, consistent formatting, no artifacts. Like a QA engineer wrote them.

**Ground truth inputs:** 30% remain clean; 70% have one or more of:
- Casual/informal phrasing ("hey can u help me with this")
- Minor typos and misspellings (1–3 per input)
- Mixed case, inconsistent punctuation
- Run-on sentences or sentence fragments
- Excessive whitespace, trailing characters

This affects cost because messier inputs may produce longer model responses (the model asks for clarification or restates the question), trigger different classification paths, or increase iteration counts in self-reflection loops.

The style shift is a property of the generator, not the tier. Within each tier, ground truth inputs are messier than profiling inputs at the same tier.

### Dimension 3: Token Length Stretch

**Profiling inputs:** Token counts at the middle of each tier's range.

**Ground truth inputs:** Token counts stretched to 1.5–2× the profiling average for the same tier. This is the simplest drift — production users write longer messages than test users.

This is implemented by expanding the token count target range for each tier in the ground truth generator. See per-workflow specs for the exact ranges.

### Workflow-Specific Structural Drift (Exceptions)

Two workflows have a structural dimension orthogonal to tier weights:

**W5 — Modality ratio shift:** Profiling inputs are 70% text-only / 30% image-containing. Ground truth inputs are 40% text-only / 60% image-containing. Image inputs cost 3–10× more due to vision tokens, and this ratio isn't captured by difficulty tiers (a "hard" text invoice and an "easy" image receipt have different cost profiles for different reasons).

**W19 — Session depth shift:** Profiling conversations average 5 substantive turns out of 8 (3 turns are filler — "thanks", "ok", "got it"). Ground truth conversations average 7 substantive turns out of 8 (only 1 filler). More substantive turns = longer per-turn responses = more accumulated context = higher cost at turns 6–8.

These two exceptions are the only per-workflow structural drift in v1. All other workflows use the three uniform dimensions only.

---

## Part 3: General Input Requirements

### Non-Uniformity (Acceptance Criteria)

These are post-generation checks, not generation instructions. After generating a set, verify:

1. **Token count entropy:** Compute entropy of token counts across all inputs (binned into 10 bins). Must be >60% of maximum entropy (log(10)). This prevents clustering at tier midpoints.
2. **Range coverage:** Within each tier, the token count range must span at least 70% of the tier's specified bounds. No tier should have all inputs clustered at one end.
3. **Duplicate detection:** No two inputs within a tier may be structurally identical (same template, same slot values). Textual similarity (cosine on TF-IDF) between any two inputs in the same tier must be <0.85.
4. **Tier allocation verification:** Actual tier counts must match target weights within ±2 inputs.

If any check fails, regenerate the failing tier. These criteria are already in `cross-cutting-robustness.md` — reference that document for the verification code patterns.

### Dirty Inputs (5% of Each Set)

Across all workflows, ~5% of inputs are deliberately malformed. For the profiling set (n=50), that's 2–3 dirty inputs. For the ground truth set (n=500), that's ~25.

Dirty inputs are distributed across tiers (they don't all go in the edge tier). A dirty input at the easy tier tests whether a typo-laden simple question still routes correctly and costs roughly the same.

The dirty input types per workflow are specified in `cross-cutting-robustness.md` Section 8 (Dirty Input Specification). The types are:

| Input Type | Workflows |
|---|---|
| Typos and misspellings | W1, W2, W9, W11, W13, W17 |
| Mixed Unicode/encoding (CJK, emoji, RTL) | W1, W11, W12, W19 |
| Copy-pasted artifacts (HTML tags, excessive whitespace, markdown in plain text) | W2, W4, W14, W15 |
| Near-empty inputs (5-word questions, 1-sentence documents) | W1, W9, W11, W14 |
| Near-limit inputs (95% of context window) | W18, W19 |
| Adversarial routing (designed to confuse classifiers) | W13, W17 |

### Input Tagging Schema

Every generated input must include these metadata fields:

```json
{
  "id": "w01_prof_easy_007",
  "workflow": "W1",
  "profile": "profiling | ground_truth",
  "tier": "easy | medium | hard | edge | extreme",
  "token_count": 142,
  "is_dirty": false,
  "dirty_type": null,
  "structural_descriptor": {
    "...workflow-specific fields..."
  },
  "input_data": {
    "...the actual input content..."
  }
}
```

The `structural_descriptor` carries workflow-specific metadata that enables post-hoc analysis: for W13, which path the input is designed to trigger; for W2, the expected iteration range; for W5, the modality type; etc.

---

## Part 4: Per-Workflow Input Specifications

For each workflow: what the input looks like as data, what defines each tier, token count ranges per tier for both distributions, the structural descriptor fields, and generation approach.

---

### W1 — Customer Support (Simple)

**Input schema:**
```json
{
  "customer_message": "string — the customer's support question"
}
```

**Tier definitions:**

| Tier | Description | Profiling tokens | Ground truth tokens | Expected routing |
|---|---|---|---|---|
| Easy | Single factual question. Pricing, hours, feature availability. | 10–40 | 20–70 | Haiku |
| Medium | Multi-sentence question. Return policy with order context, integration help with details. | 40–100 | 70–180 | Haiku or Sonnet |
| Hard | Multi-paragraph complaint. Multiple issues, order history, emotional language. | 150–400 | 200–600 | Sonnet |
| Edge | Empty message, non-English, prompt injection, competitor question. | 5–500 | 5–500 | Unpredictable |
| Extreme (GT only) | Extremely long complaint with embedded support chat transcript, 3+ separate issues. | — | 500–1,200 | Sonnet |

**Structural descriptor fields:**
```json
{
  "question_type": "billing | technical | feature_request | account_access | outage | general | complaint",
  "expected_model": "haiku | sonnet",
  "issue_count": 1
}
```

**Generation approach:** LLM-generated. Prompt the generation model with: the TechFlow domain (products, pricing, features from the system prompt spec), the target tier, and a topic sampled from the question_type list. Instruct it to produce a realistic customer message at the specified token range. Vary topics across the question_type taxonomy — no more than 30% of inputs in any one question_type.

**W1-specific notes:**
- Inputs must reference TechFlow-specific products and features by name (not generic "your product"). The agent prompt contains TechFlow knowledge, so inputs must exercise it.
- Medium-tier inputs should include enough context (order details, account info) to push the response toward the 200–400 word range.
- Hard-tier inputs should include emotional markers ("I've been waiting for three days," "this is unacceptable") that trigger longer, more empathetic responses.

---

### W2 — Customer Support (Complex, with Loops)

**Input schema:**
```json
{
  "customer_message": "string — the customer's support question"
}
```

Same input format as W1, but the inputs are designed to trigger varying iteration counts in the research-and-draft loop.

**Tier definitions:**

| Tier | Description | Profiling tokens | GT tokens | Expected iterations | Opus triggered? |
|---|---|---|---|---|---|
| Easy | Clear single-issue question with enough context for quick resolution. | 40–100 | 60–180 | 3–4 | No |
| Medium | Multi-part question requiring cross-referencing product knowledge. | 80–200 | 120–350 | 5–7 | No |
| Hard | Complex complaint with multiple issues, contradictory info, or policy edge cases. | 150–400 | 250–600 | 8–12 | Yes (conditional) |
| Edge | Ambiguous question that can never be fully resolved, or extremely long input. | 20–600 | 20–800 | 4–12 | Maybe |
| Extreme (GT only) | Multi-issue complaint referencing prior interactions the agent doesn't have, requesting actions outside scope. | — | 400–1,000 | 10–12 | Yes |

**Structural descriptor fields:**
```json
{
  "question_type": "billing | technical | feature_request | account_access | outage | complaint",
  "expected_iteration_range": [5, 7],
  "expected_opus_trigger": false,
  "issue_count": 2,
  "includes_prior_interaction_reference": false
}
```

**Generation approach:** LLM-generated. The key design challenge is that iteration count is driven by the **interaction** between input complexity and the self-assessment confidence threshold in the system prompt, not solely by input length. To control iteration count:
- Easy inputs: single clear issue, all necessary context provided → agent converges quickly.
- Medium inputs: 2–3 related issues that require the agent to address each one → more iterations to reach confidence ≥ 0.9.
- Hard inputs: contradictory information, ambiguous policy situations, emotional tone that requires careful handling → many iterations before confidence converges.
- The prompt must instruct the generator: "Create a customer message that a support agent would need approximately {N} research-and-revise cycles to address confidently."

**W2-specific notes:**
- Hard-tier inputs must include enough ambiguity that the agent's self-assessment stays below 0.9 for multiple iterations. Examples: "I was told by a previous agent that X, but your website says Y" (forces the agent to reconcile conflicting info), or multi-issue tickets where resolving one issue depends on the other.
- The Opus trigger fires when classification = "complex" AND iteration count ≥ 4, so hard-tier inputs should be classifiable as "complex" by the intake step.

---

### W4 — Compliance/Document Review (Self-Reflection Loops)

**Input schema:**
```json
{
  "document_text": "string — the full text of the business document to review",
  "document_type": "contract | hr_policy | regulatory_filing"
}
```

**Tier definitions:**

| Tier | Description | Profiling tokens | GT tokens | Expected iterations | Compliance issues |
|---|---|---|---|---|---|
| Easy | Short, clean document with 0–1 minor issues. | 500–1,500 | 800–2,500 | 2 | 0–1 minor |
| Medium | Moderate document with 3–4 issues across severity levels. | 1,500–4,000 | 2,500–7,000 | 3–4 | 3–4, mixed severity |
| Hard | Long document with 8+ issues, ambiguous clauses, missing sections. | 4,000–10,000 | 6,000–15,000 | 6–8 | 8+, including critical |
| Edge | Non-standard format, contradictory sections, 1-paragraph document. | 100–12,000 | 100–15,000 | 2–8 | Unpredictable |
| Extreme (GT only) | Dense regulatory filing at near-context-limit, subtle issues buried in boilerplate. | — | 12,000–25,000 | 7–8 | 10+, mostly subtle |

**Structural descriptor fields:**
```json
{
  "document_type": "contract | hr_policy | regulatory_filing",
  "planted_issues": [
    {"severity": "critical", "type": "missing_clause", "section": "Section 4.2"},
    {"severity": "minor", "type": "ambiguous_language", "section": "Section 7.1"}
  ],
  "expected_iteration_range": [3, 4],
  "total_sections": 8
}
```

**Generation approach:** LLM-generated documents with planted compliance issues. The generator should:
1. Generate a structurally complete business document of the target type and length.
2. Plant specific compliance issues at known locations (missing clauses, ambiguous language, incorrect references, missing GDPR provisions, etc.).
3. Tag each planted issue in the structural descriptor so the pilot can verify the agent finds them.

Distribute document types roughly evenly: ~35% contracts, ~35% HR policies, ~30% regulatory filings.

**W4-specific notes:**
- The iteration count is driven by the Qwen critique step's satisfaction criterion. Easy documents have few or no issues → critique is satisfied after 2 iterations. Hard documents have many issues → critique keeps finding problems through iteration 6–8.
- The compliance checklist in the system prompt (contracts: 10 items, HR: 6 items, regulatory: 4 items) must be exercised across the input set. Each checklist item should appear as a planted issue in at least 2 inputs.
- Ground truth style shift: documents have more formatting issues (inconsistent numbering, missing headers, typographical errors in section references) that make the review harder without adding substantive compliance issues.

---

### W5 — Multimodal Extraction + Structured Output

**Input schema:**
```json
{
  "modality": "text | image | mixed",
  "content": "string — text content, or base64-encoded image, or both",
  "document_subtype": "invoice | receipt | business_card | form | table"
}
```

**Tier definitions:**

| Tier | Description | Profiling tokens | GT tokens | Fields to extract |
|---|---|---|---|---|
| Easy | Text-only invoice with 5 clear fields. Clean formatting. | 300–800 | 500–1,200 | 4–6 |
| Medium | PDF with embedded table, 10 fields. Some fields in table cells. | 800–2,000 | 1,200–3,500 | 8–12 |
| Hard | Image of handwritten receipt or scanned form. 15+ fields. | 1,500–6,400 | 2,000–8,000 | 13–20 |
| Edge | Blurry image, rotated document, mixed-language, blank input. | 100–6,400 | 100–8,000 | 0–20 |
| Extreme (GT only) | Multi-page scanned document with mixed handwriting and print, tables across page breaks. | — | 5,000–12,000 | 20+ |

**Structural descriptor fields:**
```json
{
  "modality": "text | image | mixed",
  "document_subtype": "invoice | receipt | business_card | form | table",
  "field_count": 10,
  "image_resolution": "high | medium | low",
  "handwriting_present": false
}
```

**Generation approach:** Hybrid.
- Text-only inputs: LLM-generated invoices/receipts as structured text.
- Image inputs: Generate text content via LLM, render to image using reportlab or PIL (vary font, layout, rotation, noise). For "scanned" inputs, rasterize clean text to images with degradation (blur, skew, noise).
- Mixed inputs: Combine text sections with embedded images.

**W5-specific drift (modality ratio):**
- Profiling: 70% text-only, 30% image-containing.
- Ground truth: 40% text-only, 60% image-containing.

This is the structural drift exception for W5. The modality ratio drives cost independently of difficulty tier because vision tokens have a different pricing model than text tokens. A "hard" text invoice (many fields, complex layout) costs less than a "medium" image receipt (fewer fields but vision token overhead).

**W5-specific notes:**
- The extraction schema in the system prompt lists 15+ common field names. Each tier should exercise a different subset — easy inputs use the 5 most common fields, hard inputs use obscure fields too (payment_terms, billing_address, PO_number).
- Document subtypes should be evenly distributed across the set. Don't make all easy inputs invoices and all hard inputs receipts.

---

### W9 — Sales/Outreach (OpenAI)

**Input schema:**
```json
{
  "lead_profile": {
    "company_name": "string",
    "industry": "string",
    "employee_count": 350,
    "tech_stack": ["Salesforce", "Slack"],
    "recent_signals": ["Series B funding", "Hiring ML engineers"],
    "engagement": ["visited_pricing_page", "downloaded_whitepaper"],
    "contact": {
      "first_name": "string",
      "last_name": "string",
      "title": "string",
      "email": "string"
    }
  }
}
```

**Tier definitions:**

| Tier | Description | Profiling tokens | GT tokens | Expected rating | Email length |
|---|---|---|---|---|---|
| Easy | Complete profile, high-fit industry, strong signals. | 150–250 | 200–400 | Hot | 120–180 words |
| Medium | Partial profile, moderate industry fit, some signals. | 100–200 | 150–350 | Warm | 150–220 words |
| Hard | Sparse profile, low-fit industry, few signals. | 60–150 | 80–250 | Cold | 80–120 words |
| Edge | Empty fields, contradictory signals, non-English company. | 30–200 | 30–300 | Unpredictable | Variable |
| Extreme (GT only) | Extremely detailed profile (15+ data points), niche industry not in scoring rubric. | — | 300–600 | Variable | Variable |

**Structural descriptor fields:**
```json
{
  "expected_rating": "hot | warm | cold",
  "expected_score": 7,
  "profile_completeness": 0.85,
  "industry_fit": "high | medium | low"
}
```

**Generation approach:** Template-based with LLM embellishment. Generate lead profiles by:
1. Sampling from pools of: company names, industries, tech stacks, signal types, engagement events, contact names.
2. Filling the JSON schema with sampled values.
3. Controlling the score by choosing field values that sum to the target score range per the qualification rubric in the system prompt.
4. Using an LLM to generate the `recent_signals` descriptions (these are the free-text fields that vary in length and style).

**W9-specific notes:**
- The scoring rubric in the system prompt is deterministic: company size + industry + signals + tech stack + engagement = score, score maps to hot/warm/cold. The input generator must produce inputs that score correctly per this rubric — don't generate a lead with employee_count=800, industry="SaaS", funding=recent, and label it "cold."
- Hot leads have more complete profiles (more tokens), cold leads have sparser profiles (fewer tokens). But cold lead emails are shorter. So the total cost per run inverts slightly across tiers.
- Ground truth length stretch: lead profile descriptions are more verbose (full paragraphs about the company instead of bullet points).

---

### W11 — Support (Qwen)

**Same inputs as W1.** Use the exact same generated inputs for both W1 and W11 to enable valid cross-provider cost comparison. The only difference is the model processing them.

Generate inputs once, use for both workflows. Tag with both `W1` and `W11` workflow IDs.

---

### W12 — Data Extraction (DeepSeek)

**Input schema:**
```json
{
  "document_text": "string — the text document to extract from"
}
```

**Tier definitions:**

| Tier | Description | Profiling tokens | GT tokens | Expected entities |
|---|---|---|---|---|
| Easy | Short memo or correspondence. 3–5 entities. | 200–600 | 300–1,000 | 3–5 |
| Medium | Business report with tables and data. 8–12 entities. | 600–2,000 | 1,000–3,500 | 8–12 |
| Hard | Long financial summary with nested data. 15+ entities. | 2,000–5,000 | 3,000–8,000 | 15–25 |
| Edge | Formatted oddly (all caps, no punctuation, mixed encoding). | 100–5,000 | 100–8,000 | Variable |
| Extreme (GT only) | Dense regulatory or financial document approaching DeepSeek's sweet spot for cache testing. | — | 6,000–15,000 | 30+ |

**Structural descriptor fields:**
```json
{
  "document_type": "report | memo | financial_summary | correspondence",
  "entity_count": 10,
  "has_tables": false,
  "has_numerical_data": true
}
```

**Generation approach:** LLM-generated documents. The generator should produce realistic business documents with known entities that can be verified post-extraction. Vary document types evenly.

**W12-specific notes:**
- This workflow tests the simplest cost model: input-token-dominated, low output variance. The input set should produce a smooth, roughly linear relationship between input tokens and cost.
- Ground truth length stretch is the primary drift lever (longer documents = more input tokens = more cost).
- Extreme tier tests whether very long documents interact with DeepSeek's caching behavior in unexpected ways.

---

### W13 — Routing/Conditional Agent

**Input schema:**
```json
{
  "user_query": "string — the question to be routed"
}
```

**Tier definitions (mapped to routing paths):**

| Tier | Path | Description | Profiling tokens | GT tokens | Profiling weight | GT weight |
|---|---|---|---|---|---|---|
| Easy | Path A | Simple factual question, 1–2 sentence answer. | 10–30 | 15–50 | 70% | 55% |
| Medium | Path B | Analytical question requiring reasoning, multi-paragraph answer. | 30–80 | 50–140 | 20% | 25% |
| Hard | Path C | Complex question requiring tool calls. | 40–120 | 60–200 | 10% | 15% |
| Edge | Any | Ambiguous input that could route to multiple paths. | 10–120 | 10–200 | — | 5% |

**Note on W13 tier mapping:** Unlike other workflows where tiers map to complexity within a single processing pipeline, W13 tiers map directly to routing outcomes. The "easy" tier IS Path A, the "medium" tier IS Path B, the "hard" tier IS Path C. The tier weight shift changes the routing ratio, which is the primary cost driver (Path A costs ~$0.003, Path C costs ~$0.08 — a ~27× gap that creates bimodality).

**W13-specific profiling distribution:** 35 Path A, 10 Path B, 5 Path C (70/20/10). No edge tier in profiling — all inputs route deterministically.

**W13-specific ground truth distribution:** 275 Path A, 125 Path B, 75 Path C, 25 edge (55/25/15/5). The edge tier includes ambiguous inputs that may route to Path B or C unpredictably, adding noise to the cost distribution.

**Structural descriptor fields:**
```json
{
  "target_path": "A | B | C",
  "classification_confidence": "high | low",
  "requires_tools": false
}
```

**Generation approach:** LLM-generated, constrained by the classification criteria in the W13 system prompt. For each input:
1. Select the target path.
2. Generate a question that unambiguously matches that path's criteria (for non-edge inputs) or deliberately straddles two paths (for edge inputs).
3. Verify by running the classification prompt on a sample — if >5% of non-edge inputs misroute, the generation criteria need tightening.

**W13-specific notes:**
- Path A questions must be simple enough that Haiku answers in <75 words. Don't generate Path A questions that accidentally require analysis.
- Path C questions must reference current data, calculations, or lookups that justify tool use. "What's the weather?" is a tool question. "Why is the sky blue?" is not.
- The 70/20/10 → 55/25/15/5 shift is smaller than the general tier weight shift because W13's routing ratio is the cost-critical dimension, and a 70→55% shift in cheap queries is already a significant cost impact when the expensive path is 27× more costly.

---

### W14 — Simple PDF RAG + Structured Output

**Input schema:**
```json
{
  "query": "string — the insurance question to answer",
  "pdf_corpus": ["list of PDF document references available for retrieval"]
}
```

The PDFs themselves are specified in `directions-pdf-generation.md`. This file specifies the queries.

**Tier definitions:**

| Tier | Description | Profiling query tokens | GT query tokens | Expected retrieval size |
|---|---|---|---|---|
| Easy | Specific factoid query matching 1–2 chunks. | 15–40 | 20–60 | ~500 tokens |
| Medium | Comparison query matching 3–5 chunks. | 30–80 | 50–130 | ~3,000 tokens |
| Hard | Synthesis query matching 8–15 chunks across multiple documents. | 40–120 | 60–200 | ~12,000 tokens |
| Edge | No-match query, non-English query, extremely long query. | 5–500 | 5–600 | 0–20,000 tokens |
| Extreme (GT only) | Broad query that matches across all documents, triggering maximum retrieval. | — | 80–300 | ~20,000 tokens |

**Structural descriptor fields:**
```json
{
  "query_type": "factoid | comparison | synthesis | edge",
  "expected_chunk_count": 4,
  "target_providers": ["United Healthcare"],
  "answerable": true
}
```

**Generation approach:** LLM-generated queries. The generator must know the content of the generated PDF corpus (from `directions-pdf-generation.md`) to produce queries that will match specific chunks. This means PDF generation runs first, then query generation runs against the PDF content.

**W14-specific notes:**
- Easy queries should reference a single specific concept ("What is the deductible for in-network visits under the Gold plan?").
- Medium queries should require comparing across plan types or coverage areas ("Compare MRI coverage across plan types").
- Hard queries should require synthesizing information from multiple sections and documents ("Summarize all exclusions that apply to pre-existing conditions across all providers").
- The retrieval size is the primary cost driver (500 vs. 20,000 tokens of context injected into the generation prompt).

---

### W15 — Agentic Multi-Hop PDF RAG

**Input schema:** Same as W14.

**Tier definitions:**

| Tier | Description | Profiling query tokens | GT query tokens | Expected hops |
|---|---|---|---|---|
| Easy | Answerable in 1 hop. Specific factoid in a single section. | 15–40 | 20–60 | 1 |
| Medium | Requires cross-referencing 2 sources. 2 hops. | 30–80 | 50–130 | 2 |
| Hard | Synthesis across many sections. 3–4 hops. | 40–120 | 60–200 | 3–4 |
| Edge | Unanswerable question (4 hops of futile searching), contradictory sources. | 20–150 | 20–200 | 4 (max) |
| Extreme (GT only) | Multi-part question requiring 4 hops where each hop adds substantial context. | — | 80–300 | 4 |

**Structural descriptor fields:**
```json
{
  "expected_hop_count": 2,
  "query_type": "factoid | cross_reference | synthesis | unanswerable",
  "information_spread": "single_section | multi_section | multi_document"
}
```

**Generation approach:** LLM-generated queries designed to require specific hop counts. The key is that multi-hop queries ask for information spread across non-adjacent sections or documents, so the first retrieval won't be sufficient.

**W15-specific notes:**
- Easy queries must be answerable from a single retrieval round — the sufficiency assessor should say "sufficient: true" after the first hop.
- Hard queries must require information from 3–4 different locations in the corpus. Design them as compound questions: "What are the scenarios where a claim would be denied, and for each, what appeal options exist?" — answering requires denial conditions (one section) + appeal procedures (another section) + exceptions (a third section).
- The sufficiency threshold in the system prompt is strict ("ALL parts must be answerable with specific evidence"). This means even medium queries may trigger 2 hops if the first retrieval misses one part.
- Context growth across hops is the cost driver. By hop 4, the accumulated context from all prior retrievals can be 15,000+ tokens.

---

### W16 — Map-Reduce PDF Analysis

**Input schema:**
```json
{
  "pdf_document": "reference to the PDF to analyze"
}
```

The input IS the PDF. PDFs are specified in `directions-pdf-generation.md`. The per-tier definitions here describe what PDFs to generate for each tier.

**Tier definitions:**

| Tier | Description | Profiling pages | GT pages | Expected sections (N) |
|---|---|---|---|---|
| Easy | Short memo or brief. | 3–8 | 5–12 | 3–4 |
| Medium | Standard report. | 10–25 | 15–40 | 6–10 |
| Hard | Long annual report or filing. | 30–60 | 45–80 | 12–18 |
| Edge | 1-page document (degenerate) or 100-page document (max fan-out). | 1–100 | 1–100 | 1–20 |
| Extreme (GT only) | 80–100 page dense document with poor section structure. | — | 80–100 | 15–20 |

**Structural descriptor fields:**
```json
{
  "page_count": 25,
  "expected_section_count": 8,
  "has_clear_structure": true,
  "content_type": "annual_report | regulatory_filing | research_paper | meeting_transcript"
}
```

**W16-specific notes:**
- The section count N is the primary cost driver (N parallel Haiku calls + aggregation of N outputs). The input set must produce a wide range of N values.
- Ground truth PDFs should have less clear section structure than profiling PDFs (fewer headings, more gradual topic transitions) so the splitter step produces more variable N values.
- Content types should be distributed evenly.

---

### W17 — Insurance Claims Review Agent

**Input schema:**
```json
{
  "claim": {
    "claim_id": "CLM-2024-001234",
    "member_id": "MEM-789456",
    "member_status": "active | inactive",
    "claim_type": "pre_approval | standard | appeal",
    "provider": "United Healthcare | Aetna | Cigna",
    "diagnosis_code": "M54.5",
    "procedure_code": "72148",
    "claimed_amount": 2200.00,
    "service_date": "2024-03-15",
    "clinical_note": "string | null",
    "itemized_bill": "string | null",
    "prior_denial_reason": "string | null",
    "appeal_documentation": "string | null"
  }
}
```

**Tier definitions (mapped to pipeline outcomes):**

| Tier | Pipeline outcome | Profiling weight | GT weight | Claim characteristics |
|---|---|---|---|---|
| Easy | Short-circuit: inactive member or missing docs | 15% | 20% | `member_status="inactive"` or missing required docs |
| Medium | Full pipeline, standard claim, straightforward approve/deny | 40% | 35% | Standard claim, all docs present, clear coverage |
| Hard | Full pipeline, complex evaluation (pre-approval or appeal) | 30% | 25% | Pre-approval with clinical review, or appeal with new evidence |
| Edge | Full pipeline + routing flag (high amount or code inconsistency) | 15% | 15% | `claimed_amount > 5000` or diagnosis/procedure mismatch |
| Extreme (GT only) | Appeal of previously denied pre-approval, high amount, routed | — | 5% | Complex appeal + high amount + edge documentation |

**Structural descriptor fields:**
```json
{
  "claim_type": "pre_approval | standard | appeal",
  "overrides_expected": ["inactive_member"],
  "short_circuit_expected": true,
  "routing_expected": false,
  "pipeline_depth": "intake_only | full_pipeline | full_pipeline_routed"
}
```

**Generation approach:** Template-based with controlled field values. Each claim is a JSON object with specific field combinations that trigger known pipeline behaviors:
- `member_status = "inactive"` → short-circuit deny (test override rule 1)
- `clinical_note = null` on a pre-approval claim → short-circuit request docs (test override rule 2)
- `claimed_amount > 5000` → full pipeline + routing flag (test override rule 3)
- Mismatched diagnosis/procedure codes → full pipeline + coding review flag (test override rule 4)

**W17-specific notes:**
- The bimodality in W17 comes from short-circuit claims (cheap: intake only, ~$0.01) vs. full-pipeline claims (expensive: intake + retrieval + evaluation + maybe routing, ~$0.05–0.15). The tier distribution must produce both modes with clear separation.
- Claim types must be distributed across providers (United Healthcare, Aetna, Cigna) so the retrieval step pulls from different policy documents. This exercises cross-provider cost accounting.
- The 6 function call actions in the system prompt (approve_pre_authorization, approve_claim_payment, deny_claim, request_missing_documentation, route_to_senior_reviewer, route_to_coding_review) must each be triggered by at least 3 inputs in the profiling set and at least 15 in the ground truth set.
- Override rule 4 (code inconsistency) requires generating realistic diagnosis/procedure code pairs. Use real ICD-10 and CPT code formats (e.g., M54.5 for low back pain, 72148 for lumbar MRI) but ensure some pairs are intentionally inconsistent (e.g., diagnosis S52.501A for forearm fracture paired with procedure 72148 for lumbar MRI).

---

### W18 — Long-Document Single-Pass

**Input schema:**
```json
{
  "pdf_document": "reference to the PDF to analyze"
}
```

The input IS the PDF. PDFs are specified in `directions-pdf-generation.md`.

**Tier definitions:**

| Tier | Description | Profiling token count | GT token count |
|---|---|---|---|
| Easy | 30-page text-only report, clear structure. | 30,000–40,000 | 30,000–50,000 |
| Medium | 50–60 page report with tables. | 40,000–60,000 | 50,000–75,000 |
| Hard | 80–100 page dense regulatory filing. | 60,000–80,000 | 70,000–95,000 |
| Edge | Near context limit (100K tokens). | 80,000–100,000 | 85,000–100,000 |
| Extreme (GT only) | At context limit with complex structure and dense numerical data. | — | 90,000–100,000 |

**Structural descriptor fields:**
```json
{
  "estimated_token_count": 55000,
  "page_count": 60,
  "content_type": "annual_report | legal_deposition | technical_spec | regulatory_filing",
  "has_tables": true,
  "has_numerical_data": true
}
```

**W18-specific notes:**
- This is a pure input-scaling test. Output is fixed (~800–2,000 tokens regardless of input length). The input set must produce a smooth, wide range of input token counts.
- Token count is the only cost-relevant dimension. The content type doesn't matter for cost, but it matters for realism.
- Ground truth length stretch is the primary drift lever — ground truth documents are longer at every tier.
- Near-context-limit documents (extreme tier) test whether the model's behavior changes near the limit (truncation, quality degradation, latency spikes).

---

### W19 — Multi-Turn Conversation (8 Turns)

**Input schema:**
```json
{
  "conversation_script": [
    {"turn": 1, "user_message": "string"},
    {"turn": 2, "user_message": "string"},
    {"turn": 3, "user_message": "string"},
    {"turn": 4, "user_message": "string"},
    {"turn": 5, "user_message": "string"},
    {"turn": 6, "user_message": "string"},
    {"turn": 7, "user_message": "string"},
    {"turn": 8, "user_message": "string"}
  ]
}
```

Each "input" is a full 8-turn user-side conversation script. The agent generates responses at runtime. The cost of each turn depends on accumulated context from all prior turns.

**Tier definitions:**

| Tier | Description | Profiling per-turn tokens | GT per-turn tokens | Substantive turns |
|---|---|---|---|---|
| Easy | Simple Q&A, topic resolved by turn 4, remaining turns are filler. | 20–60 | 30–100 | 4–5 |
| Medium | Moderate complexity, some topic shifts, medium-length messages. | 40–100 | 60–180 | 5–6 |
| Hard | Complex multi-issue conversation, long messages, cross-references. | 80–200 | 120–350 | 7–8 |
| Edge | Contradictions across turns, language switching, "forget this" requests. | 10–250 | 10–350 | Variable |
| Extreme (GT only) | Every turn is substantive and long, pushing context to near-limit by turn 8. | — | 200–500 | 8 |

**Structural descriptor fields:**
```json
{
  "substantive_turn_count": 6,
  "topic_shifts": 1,
  "includes_contradiction": false,
  "avg_user_message_tokens": 85,
  "estimated_turn_8_context_tokens": 9500
}
```

**W19-specific drift (session depth):**
- Profiling: average 5 substantive turns (3 filler turns like "ok", "thanks", "got it" — short, don't trigger detailed responses).
- Ground truth: average 7 substantive turns (only 1 filler). More substantive turns = longer agent responses per turn = more accumulated context = higher cost at turns 6–8.

This is the structural drift exception for W19. The tier weight shift also applies (more easy conversations in ground truth), but the session depth shift is the cost-driving dimension that tier weights alone don't capture.

**Generation approach:** LLM-generated conversation scripts. The generator should:
1. Choose a CloudOps topic (deployment failure, scaling alert, monitoring gap, integration setup, billing question, incident response).
2. Generate 8 user messages that form a coherent conversation arc.
3. Control substantive turn count by inserting filler messages ("ok", "makes sense", "thanks, let me try that") at designated positions.
4. For hard tier: include cross-references to earlier turns ("going back to what you said about the load balancer...") and multi-part questions.

**W19-specific notes:**
- The cost pattern is linear context growth: Turn 1 ≈ 1,400 tokens input, Turn 8 ≈ 9,200–11,200 tokens input. The context growth detector must fire on this workflow.
- Filler turns are cheap (short user message → short agent response → minimal context addition). Substantive turns are expensive. The session depth shift changes the ratio.
- Topics must be distributed across CloudOps's product line. Conversations should reference specific CloudOps features by name (Dashboard, Deploy, Scale, Guard).

---

## Part 5: Generation Protocol

### Execution Order

1. **PDF generation** (W14, W15, W16, W17, W18) — must run first because query generation for W14/W15 depends on PDF content.
2. **Structured inputs** (W9, W17) — template-based, generated programmatically. No LLM dependency.
3. **Text inputs** (W1/W11, W2, W4, W12, W13, W18, W19) — LLM-generated with tier and style constraints.
4. **Query inputs** (W14, W15) — LLM-generated against the PDF corpus.
5. **Multimodal inputs** (W5) — hybrid: LLM-generated content rendered to images.

### Seeding and Reproducibility

Each generator takes a `seed` parameter. Same seed + same profile → same inputs. This enables:
- Debugging: reproduce a specific input that caused a problem.
- Comparison: run the same profiling inputs through W1 and W11 for cross-provider comparison.

Use the seed to initialize all random choices (tier assignment, topic selection, template slot filling) via a deterministic PRNG.

### Batch Generation

Generate full sets in one pass per workflow per profile:
- `generate_inputs(workflow="W1", profile="profiling", n=50, seed=42)` → 50 tagged inputs.
- `generate_inputs(workflow="W1", profile="ground_truth", n=500, seed=42)` → 500 tagged inputs.

The generator should log: tier distribution (actual vs. target), token count statistics (mean, std, min, max per tier), dirty input count, and any regeneration events (inputs that failed non-uniformity checks and were regenerated).

### LLM Generation Model

Use **DeepSeek V4 Flash** for short-to-medium inputs: customer questions, queries, lead profiles, conversation scripts, claims JSON, and any text document under ~4,000 tokens. Flash's instruction-following is more than sufficient for "generate a customer complaint about billing at 200 tokens."

Use **DeepSeek V4 Pro** for long document generation: W4 compliance documents at the hard/extreme tiers (6K–25K tokens), W12 extreme tier documents (6K–15K tokens), and any generated text exceeding ~4,000 tokens. Pro maintains coherence, consistent cross-references, and correctly planted issues (W4) across long outputs where Flash hallucinates or drifts. For documents over ~8K tokens, use sectional generation: Pro generates the outline (section titles, planted issues, key data points per section), then Pro generates each section independently (2–5 pages per call). See `directions-pdf-generation.md` for the same pattern applied to PDFs.

Do not use Haiku (quality too low for diverse, realistic inputs) or Opus/Sonnet (budget doesn't support it at n=500 per workflow across 14 workflows).

**Multimodal note:** No generation step requires vision. Even W5 (multimodal extraction) generates text content via the LLM, then renders it to images programmatically using reportlab/PIL. The generation model never sees or produces images — it produces the invoice/receipt/form text that the rendering code converts. Similarly, W14/W15 query generation works against the text content used to create PDFs, not against the rendered PDFs themselves. DeepSeek V4 Flash/Pro (text-only) covers every workflow's generation needs.

**Cache-busting during generation:** Since you are making many sequential calls to DeepSeek for generation, use the same `{{CACHE_BUST_SUFFIX}}` pattern in the generation prompts to avoid prefix caching producing duplicate outputs.

---

## Part 6: Acceptance Criteria

After generating all input sets, verify:

### Per-Set Checks

1. **Tier distribution:** Actual counts within ±2 of target per tier.
2. **Token count entropy:** >60% of maximum entropy across all inputs.
3. **Range coverage:** Each tier spans ≥70% of its specified token range.
4. **Duplicate detection:** No two inputs in the same tier with cosine similarity >0.85.
5. **Dirty input count:** 2–3 per profiling set, 20–30 per ground truth set.
6. **Dirty input distribution:** Dirty inputs spread across tiers, not concentrated in edge.

### Cross-Set Checks (Profiling vs. Ground Truth)

7. **Tier weight shift verified:** Ground truth tier distribution matches 55/25/12/5/3 (±2%).
8. **Token length stretch verified:** Mean token count per tier in ground truth is 1.5–2× profiling mean for the same tier.
9. **Style shift verified:** ≥70% of ground truth inputs have at least one style artifact (typo, informal phrasing, formatting issue). ≤10% of profiling inputs have style artifacts.
10. **Structural drift verified (W5 only):** Ground truth modality ratio is 40/60 text/image (±5%).
11. **Structural drift verified (W19 only):** Ground truth mean substantive turns is ≥6.5 (vs. profiling ≤5.5).

### Detector Activation Pre-Check

Before running the full pipeline, verify that the input design should trigger the expected detectors by checking structural properties:

- **W2, W4:** Hard-tier inputs should produce iteration count variance (range ≥ 5 across the set). If all hard-tier inputs would trigger the same iteration count, add diversity.
- **W5:** Vision token variance should span at least 5× (cheapest image input vs. most expensive).
- **W13:** The 70/20/10 routing split, combined with the 27× cost gap between Path A and Path C, should produce ΔBIC > 6 (bimodality). Verify this holds after generation by computing the expected cost distribution.
- **W17:** Short-circuit claims (15–20% of inputs) vs. full-pipeline claims should produce two distinct cost modes.
- **W19:** Token counts across 8 turns should show Pearson correlation > 0.9 with turn number for hard-tier conversations.

---

## File Manifest

Generate the following:

```
inputs/
  generators/
    w01_support_simple.py
    w02_support_complex.py
    w04_compliance_review.py
    w05_multimodal_extraction.py
    w09_sales_outreach.py
    w12_extraction_deepseek.py
    w13_routing_agent.py
    w14_simple_rag_queries.py
    w15_multihop_rag_queries.py
    w16_map_reduce.py       # (delegates to PDF generator, adds metadata)
    w17_claims_agent.py
    w18_long_document.py    # (delegates to PDF generator, adds metadata)
    w19_multi_turn.py
  generated/
    profiling/
      w01/ ... (50 JSON files, one per input)
      w02/ ...
      ...
    ground_truth/
      w01/ ... (500 JSON files, one per input)
      w02/ ...
      ...
  validation/
    verify_inputs.py        # Runs all acceptance criteria checks
    report_template.json    # Schema for the validation report
```

W11 does not get a separate generator — it reuses W1's inputs. The W11 directory under `generated/` contains symlinks or copies of W1's inputs.

W16 and W18 generators are thin wrappers that call the PDF generator (from `directions-pdf-generation.md`) and attach the tier metadata.

Each generator must be runnable standalone: `python w01_support_simple.py --profile profiling --n 50 --seed 42 --output-dir generated/profiling/w01/`.
