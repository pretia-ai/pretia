# AgentCost v2 Additions — Workflow Expansion & PDF Pipeline

Companion to the v1 recommendation doc. Contains only what changed: 5 new/redesigned workflows, the W17 claims agent architecture, the PDF processing pipeline, and the revised budget.

---

## Updated Test Suite (13 Workflows)

**Cut:** W10 (artificial mixed sales) — replaced by cross-provider RAG workflows.

**Redesigned:** W5 — multimodal extraction with structured output.

**Added:** W14, W15, W16, W17.

| # | Workflow | Complexity | Ground truth | Est. cost | Key patterns tested |
|---|----------|-----------|-------------|-----------|-------------------|
| W1 | Support agent | Simple | 200 | $4 | Baseline |
| W2 | Support agent | Complex (3–12 iter) | 300 | $109 | Loop variance, context growth |
| W4 | Code review | Complex (2–8 iter) | 300 | $218 | Self-reflection, context growth |
| **W5** | **Multimodal extraction + structured output** | Simple | 220 | $18 | Vision tokens, JSON mode |
| W8 | Research agent | Complex (2–6 iter) | 300 | $330 | Tool use, loop variance |
| W9 | Sales/outreach (OpenAI) | Simple | 200 | $4 | OpenAI generation pricing |
| W11 | Support (Qwen) | Simple | 200 | $1 | Qwen pricing |
| W12 | Extraction (DeepSeek) | Simple | 200 | $1 | DeepSeek pricing, cache |
| W13 | Routing agent | Conditional | 300 | $22 | Step count variance, bimodality |
| **W14** | **Simple PDF RAG + structured output** | RAG | 300 | $38 | Retrieval variance, cross-provider |
| **W15** | **Agentic multi-hop PDF RAG** | RAG + loops | 300 | $208 | All 3 cost models combined |
| **W16** | **Map-reduce PDF analysis** | Parallel | 300 | $19 | Fan-out, variable N, parallel |
| **W17** | **Insurance claims agent** | Conditional + RAG | 300 | $16 | Real-world decision tree, multi-doc RAG |

Provider coverage: Anthropic (W1/2/4/5/8/14/15/16/17), OpenAI generation (W9) + embeddings (W14/15/17), Gemini (W15), Qwen (W11), DeepSeek (W12).

---

## New Workflow Designs

### W5 (Redesign): Multimodal Extraction with Structured Output

Accepts a mix of text documents, images (screenshots, scanned receipts, photos), and PDFs with embedded charts/tables. Output is structured JSON conforming to an extraction schema (using JSON mode). Models: Sonnet 4.6 with vision. Cost driver: input modality — image-heavy inputs cost 3–10× more than text. Tests vision token capture by the collector and structured output token behavior.

### W14: Simple PDF RAG with Structured Output

The most common production RAG pattern. Upload PDF(s) → extract text from text pages (non-LLM) → vision model for image-heavy/scanned pages → chunk → embed (OpenAI text-embedding-3-small) → query → vector retrieve (variable chunk count) → generate structured JSON answer (Sonnet 4.6 JSON mode). No loops. Cross-provider (OpenAI embeddings + Anthropic generation). Cost driver: retrieval size variance (500–20K tokens). Input set: 30–50 diverse PDFs as corpus, 50+ queries spanning sparse-to-dense retrieval.

### W15: Agentic Multi-Hop PDF RAG

Same PDF ingestion as W14 but with multi-hop retrieval. Query → retrieve → assess sufficiency (Gemini 2.5 Flash) → re-retrieve if insufficient → repeat 1–4 hops → generate comprehensive answer (Opus 4.7) with all accumulated context. This is the first workflow exercising all three cost model adjustments simultaneously: retrieval size variance × loop count variance × context growth. Multi-provider: OpenAI embeddings + Gemini assessment + Opus generation + Claude Vision for image pages. Expected patterns: loop variance, context growth, high token variance, possibly bimodality.

### W16: Map-Reduce PDF Analysis

Upload long PDF → splitting step identifies N sections (N varies 3–20 with document length) → N parallel Haiku 4.5 calls process each section → Sonnet 4.6 aggregation combines all results. Tests parallel fan-out with variable N, collector behavior with parallel execution, and aggregation cost scaling. Expected patterns: step count variance (N varies), high token variance.

---

## W17: Insurance Claims Review Agent — Full Architecture

Based on a real production use case (health insurance claims processing). This is the integration test of the suite — a single workflow that combines conditional routing, multi-document RAG with PDF, structured output, function calling, and variable step topology.

### System Context

The agent receives a structured claim JSON and must consult unstructured policy PDFs (one per insurance provider) to decide the next action. The action set is fixed:

- `approve_pre_authorization()`
- `approve_claim_payment(payment_in_dollars: float)`
- `deny_claim()`
- `request_missing_documentation(documents: list[str])`
- `route_to_senior_reviewer()`
- `route_to_coding_review()`

Output is structured JSON containing the action, a free-text reason grounded in policy evidence, and any additional metadata.

### Step 1 — Intake & Override Check

**Model:** Haiku 4.5. **Input:** claim JSON only (no policy retrieval yet).

Four override checks run before any policy consultation:

1. `member_status == "inactive"` → short-circuit to `deny_claim()` with reason "Plan inactive at time of service." Pipeline ends. Cost: ~$0.002.
2. Essential documentation missing (e.g., no clinical_note for pre-approval) → short-circuit to `request_missing_documentation()`. Pipeline ends after 1–2 steps. Cost: ~$0.005.
3. `claimed_amount > 5000` → flag for senior reviewer routing at Step 4. Pipeline continues.
4. Diagnosis/procedure code inconsistency → flag for coding review routing at Step 4. Pipeline continues.

If no short-circuit override fires, classify claim type (pre-approval / standard / appeal) and pass to Step 2.

Output schema: `{ "overrides": [...], "claim_type": "...", "proceed_to_retrieval": bool }`

### Step 2 — Policy Retrieval (Multi-Document RAG)

The agent must retrieve from the correct provider's policy PDF — United Healthcare claims use the UHC policy, Aetna claims use the Aetna policy, etc. This is multi-document RAG where document selection is part of the retrieval logic.

**Embedding model:** OpenAI text-embedding-3-small. Each policy PDF is pre-processed into a vector store at profiling setup time (one-time cost, not per-run). At runtime, the retrieval query is constructed from the claim's procedure code, diagnosis code, and claim type, scoped to the correct provider's document.

Retrieval size varies: a simple office visit claim matches 1–2 policy sections (~500 tokens). A complex biologic injection pre-approval matches 5–8 sections covering coverage criteria, prior authorization requirements, medical necessity standards, and exclusions (~4,000 tokens).

### Step 3 — Evaluate & Decide

**Model:** Sonnet 4.6 with JSON mode. **Input:** claim JSON + retrieved policy sections + workflow rules for the classified claim type.

Type-specific logic:

**Pre-approval claims:** Check coverage criteria (is the diagnosis covered?). Verify documentation completeness (clinical note present? prior treatment documented?). Assess medical necessity against the policy's medical necessity standard. Decide approve or deny authorization.

**Standard claims:** Validate coverage (is the procedure covered under the plan?). Check procedure/diagnosis code consistency. Verify no exclusions apply. Decide approve with payment amount or deny.

**Appeal claims:** Retrieve the prior denial reason from claim data. Assess whether the appeal documentation is new, sufficient, and relevant to the denial reason. Re-evaluate medical justification. Decide to overturn or uphold.

Output schema (structured JSON, function call format):

```json
{
  "action": "approve_claim_payment",
  "action_params": { "payment_in_dollars": 2200.0 },
  "reason": "MRI lumbar spine covered under policy §5.1. Conservative treatment attempted per clinical note. Prior authorization approved.",
  "evidence": [
    "Policy §5.1: MRI covered when covered diagnosis present AND conservative treatment attempted AND PA approved.",
    "Clinical note: Persistent low back pain for 6 months, conservative therapy failed."
  ],
  "confidence": "high",
  "flags": []
}
```

### Step 4 — Conditional Routing (When Flagged)

**Model:** Haiku 4.5. If `claimed_amount > 5000` or code inconsistency was flagged in Step 1, wrap the Step 3 decision with a routing action: `route_to_senior_reviewer()` or `route_to_coding_review()`, including the Step 3 evaluation as context for the human reviewer.

### Cost Per Claim Scenario

| Claim scenario | Steps | Path | Est. cost |
|---------------|-------|------|-----------|
| Inactive member | 1 | Intake → deny | ~$0.002 |
| Missing docs (no policy needed) | 1–2 | Intake → request docs | ~$0.005 |
| Simple standard (low amount) | 3 | Intake → retrieve → evaluate | ~$0.04 |
| Standard with code mismatch | 4 | Intake → retrieve → evaluate → route coding | ~$0.06 |
| Pre-approval (full workflow) | 3 | Intake → retrieve → evaluate | ~$0.08 |
| High-amount pre-approval | 4 | Intake → retrieve → evaluate → route senior | ~$0.10 |
| Appeal (full workflow) | 3 | Intake → retrieve → evaluate | ~$0.10 |
| Complex appeal, high amount | 4 | Intake → retrieve → evaluate → route senior | ~$0.15 |

### Expected Patterns

- **Step count variance:** 1–4 steps, CV likely > 0.5 → DANGER
- **Bimodality:** short-circuit claims at ~$0.003 vs. full-workflow claims at ~$0.08 — two distinct cost modes
- **High token variance:** retrieval size varies 500–4,000 tokens
- **No context growth:** single-pass, not iterative
- **Structured output:** JSON mode affects output token count

### Input Set Design (40–50 Claims)

Seed from the 8 sample claims in the case study. Expand to:

- 5 inactive-member claims (override short-circuit test)
- 5 missing-docs claims (early exit test)
- 10 simple standard claims (bread-and-butter path)
- 5 standard with code mismatches (coding review routing)
- 8 pre-approval claims (varying documentation quality — some complete, some borderline)
- 5 high-amount claims (>$5,000 threshold — senior reviewer routing)
- 7 appeals (mix of new evidence quality — some strong, some insufficient)
- 5 edge cases (conflicting evidence, ambiguous provider notes, amounts near $5K boundary)

Distribute across 3–4 insurance providers.

### Policy Documents

The provided sample (HorizonCare Silver PPO 2500) plus 2–3 additional synthetic policy PDFs with meaningfully different coverage rules. Each provider's policy should differ on at least one rule (e.g., one covers biologics, another excludes them; one requires prior auth for MRI, another doesn't) so that selecting the wrong provider's policy produces a wrong decision — testing multi-document RAG correctness.

---

## PDF Processing Pipeline

Shared infrastructure across W14, W15, W16, and W17. Design goal: friction-free. User uploads a PDF, the system handles everything. No manual conversion.

### Page-Level Classification

Each PDF page is classified before processing:

1. **Text-extractable pages** (most common). Use `pdfplumber` or `PyMuPDF` for text extraction. Zero LLM cost. Reliable for digitally-created PDFs.

2. **Image-heavy pages** (charts, tables as images, diagrams). Detected by low text-to-area ratio or large embedded images. Processed via vision model (Claude Vision or GPT-4V). Cost: ~$0.003–0.01 per page.

3. **Scanned pages** (no selectable text). Detected by zero extractable text. Full-page vision OCR. Cost: ~$0.005–0.015 per page.

4. **Mixed pages** (some text, some embedded images). Extract text normally, send images to vision model separately, merge. Moderate cost.

Classification heuristic:

```
if extractable_text_length > 100 chars:
    if embedded_image_area > 40% of page:
        → mixed (text extraction + vision for images)
    else:
        → text-extractable (pdfplumber only, $0)
else:
    → scanned (full-page vision model)
```

### Chunking

After extraction, text is split for embedding. Strategy: semantic chunking by section headers when detectable, falling back to fixed-size overlapping windows (512 tokens, 64-token overlap). Chunk metadata includes source PDF name, page number(s), and section header (if any). This metadata enables W17's multi-document RAG to scope retrieval to the correct provider's policy.

### Robustness — Edge Cases

| Edge case | Detection | Handling | Profiling impact |
|-----------|-----------|----------|-----------------|
| Corrupted/malformed PDF | pdfplumber exception on open | Skip file, log warning, continue | Reduces corpus size, doesn't crash |
| Password-protected PDF | PDFPasswordIncorrect error | Skip, log "cannot process" | Graceful skip |
| Very large PDF (100+ pages) | Page count check | Process in 20-page batches. Cap retrieval to top-K chunks regardless of size. | Linear cost scaling captured correctly |
| Empty/blank pages | No text AND no images | Skip silently | No impact (common in scanned docs) |
| Non-English text | langdetect on extracted text | Process normally (embeddings + vision handle multilingual) | Token counts may differ from English |
| Tables spanning pages | pdfplumber extract_tables() | Extract row by row, attempt cross-page merge by column structure | Imperfect extraction is realistic |
| Very low resolution scans | Vision model returns garbled text | Include as-is with low confidence | Increases retrieval noise, realistic |
| All-image PDF (slide decks) | All pages classified as scanned | Every page through vision model | 10–50× higher cost than text-only PDFs |

### Processing Cost by PDF Type

| PDF type (20 pages) | Processing cost | Notes |
|---------------------|----------------|-------|
| Text-only | ~$0.00 | pdfplumber only |
| Mixed (5 image pages) | ~$0.025 | 5 × $0.005 vision |
| Scanned (all pages) | ~$0.15 | 20 × $0.0075 vision |
| Slide deck (30 pages, all images) | ~$0.30 | Highest cost per document |

This cost variance is a first-order cost driver for W14/15/16/17. The projection engine handles it through the high token variance detector.

---

## Updated Budget

| Change from v1 | Delta |
|----------------|-------|
| Cut W10 (mixed sales) | −$128 |
| W5 redesign (multimodal + structured output) | +$13 |
| W14 (simple PDF RAG) | +$38 |
| W15 (agentic multi-hop RAG) | +$208 |
| W16 (map-reduce) | +$19 |
| W17 (insurance claims agent) | +$16 |
| **Net change** | **+$166** |
| **New subtotal** | **~$993** |
| Contingency / skewed variant | ~$100 |
| **Total budget** | **~$1,093** |

If over budget: reduce W15 to 200 ground truth runs (saves ~$70) or W4 to 200 (saves ~$73).

---

## Additional Pre-Backtesting Items (v2)

These are new items added to the action list on top of v1's 15 pre-backtesting items:

| # | Action | Effort | Why it can't wait |
|---|--------|--------|-------------------|
| 5 | Build W14 simple PDF RAG workflow | 1 day | Core RAG — most common production pattern |
| 6 | Build W15 agentic multi-hop RAG workflow | 1–2 days | Tests all 3 cost models simultaneously |
| 7 | Build W16 map-reduce workflow | Half day | Tests parallel fan-out and variable step count |
| 8 | Build W17 insurance claims agent | 1–2 days | Real-world integration test |
| 9 | Redesign W5 with multimodal inputs + structured output | Half day | Tests vision tokens and JSON mode |
| 10 | Curate shared PDF corpus (30–50 PDFs, tagged) | Half day | Required for W14/15/16/17 inputs |
| 11 | Collector unit test: vision/image token capture | 2 hrs | Required before W5/W14/W15 |
| 12 | Collector unit test: parallel execution handling | 2 hrs | Required before W16 |
| 13 | Collector unit test: structured output (JSON mode) | 1 hr | Required before W5/W14/W17 |
| 25 | Verify W4 has self-reflection structure | 1 hr | Confirms critique-loop coverage |

Total additional engineering: ~5–6 days on top of v1's 7–9 days.
