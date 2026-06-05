# AgentCost v2 Additions — Workflows, Statistical Methods, Schema & PDF Pipeline

Companion to the v1 recommendation doc. Contains only what changed: model-optimized workflow suite (14 workflows, $414 subtotal), 6 statistical method upgrades (A1–A6), 5 data schema additions (B1–B4, B7), profiling tier revision (n=50 default, n=20 dropped), subsampling calibration methodology, the W17 claims agent architecture, and the PDF processing pipeline.

---

## Part A — Statistical Method Changes

### A1. Replace confidence tier scoring with conformal prediction intervals (jackknife+)

**Effort:** 2–3 days. **Priority:** Highest.

Replace the hand-tuned point score system with jackknife+ prediction intervals (Barber et al., 2021, *Annals of Statistics*). Jackknife+ provides finite-sample coverage guarantees: targeting 1−α coverage, you get at least 1−2α for any sample size, any distribution, no model assumptions.

The approach uses leave-one-out residuals from the bootstrap. For each profiling run, compute the prediction error when that run is held out. The quantile of those residuals defines the prediction interval width.

**Monthly projection propagation (critical).** Jackknife+ gives prediction intervals for per-run costs. For monthly totals, propagate through the CLT correction — do NOT simply multiply the per-run interval by N (that's the variance inflation bug). The correct approach: compute the jackknife+ interval on per-run cost to get per-run SE, then monthly interval = N × point_estimate ± z × √N × per_run_SE. This preserves the coverage guarantee asymptotically.

The MAPIE Python library implements jackknife+, CV+, and split conformal out of the box. A from-scratch implementation is ~200 lines.

Derive tiers from interval width instead of from a point score:

| Interval width relative to p50 | Tier | Label |
|-------------------------------|------|-------|
| < 2× | HIGH | "projected" |
| 2–5× | MODERATE | "estimated" |
| 5–10× | LOW | "estimated (wide range)" |
| > 10× | VERY LOW | "order of magnitude" |

The report output changes from "Confidence: MODERATE (score: 64)" to "Projected monthly cost: $4,100–$11,700 (90% prediction interval, MODERATE confidence)." The interval is the ground truth; the tier is a derived UX convenience.

**Keep the n_eff entropy gate as a separate diagnostic** for gaming detection. It no longer drives the confidence tier — it warns about profiling uniformity.

**Reference:** Barber, R.F., Candes, E.J., Ramdas, A., & Tibshirani, R.J. (2021). Predictive inference with the jackknife+. *Annals of Statistics*, 49(1), 486–507.

### A2. Replace Hartigan's dip test with BIC-based GMM + Monte Carlo integration

**Effort:** 1 day. **Priority:** High.

For bimodality detection, fit 1-component and 2-component Gaussian mixture models (scikit-learn GaussianMixture) to per-run costs. If ΔBIC > 6 (strong evidence for 2 components), declare bimodality and extract mixture parameters: means, standard deviations, and weights.

This changes the cost adjustment mapping. The v1 design said bimodality has no cost model because whole-run sampling handles it. That's wrong at small n. At n=20 with a 70/30 split, you have ~6 observations from the expensive mode. Resampling from 6 points gives poor tail coverage. Sampling from a fitted Gaussian per component is more stable. **Bimodality now gets a cost model: when detected, Monte Carlo samples from the fitted 2-component mixture instead of from raw observations.**

Updated detector-to-cost-adjustment mapping:

| Detector | Cost adjustment in Monte Carlo? |
|----------|-------------------------------|
| Context growth (linear, Pearson) | **Yes** — linear + log average |
| Context growth (non-linear, Spearman) | **Yes** — power-law k^α |
| Loop count variance | **Yes** — sample iter count × per-iter costs |
| **Bimodality (GMM)** | **Yes** — sample from fitted 2-component mixture |
| High token variance | No — triggers MC mode; variance in data |
| Step count variance | No — whole-run sampling preserves topology |

Implementation: ~50 lines. Fit k=1 and k=2 models, compare BIC, extract parameters, feed into MC.

**Reference:** Freeman, J.B. & Dale, R. (2013). Assessing bimodality to detect the presence of a dual cognitive process. *Behavior Research Methods*, 45(1), 83–97.

### A3. BCa bootstrap upgrade

**Effort:** 0.5 days. **Priority:** Medium.

Replace percentile bootstrap with bias-corrected and accelerated (BCa) bootstrap for confidence interval computation. BCa corrects for both bias and skewness in the bootstrap distribution — critical for right-skewed cost distributions. Better small-n coverage properties than the percentile method. Scipy supports BCa natively since version 1.9.

This replaces the ad-hoc tail inflation factor (1 + 2/√n) with a principled correction. If jackknife+ (A1) is implemented, BCa applies to the ground truth bootstrapping in the calibration framework rather than to the user-facing intervals.

**Reference:** Efron, B. & Tibshirani, R.J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall, Chapter 14.

### A4. CVaR (Expected Shortfall) reporting

**Effort:** 0.5 days. **Priority:** Medium.

Add Conditional Value-at-Risk to projection output alongside percentiles. CVaR at level α is the expected cost given that you exceed the (1−α) percentile. Computed trivially from Monte Carlo samples as the mean of the top 5% of simulated monthly totals.

Instead of only "p95 = $15,000/month," also report "when monthly cost exceeds $15,000, the average overshoot is $22,000." This directly answers the budgeting question: "how bad does it get when it gets bad?"

CVaR is a coherent risk measure (Artzner et al., 1999) — it satisfies subadditivity. Step-level CVaR values add up to workflow-level CVaR correctly. Percentiles don't have this property, which is why step-level p95 decomposition can be inconsistent with workflow-level p95. Worth noting in the report output.

**Reference:** Artzner, P., Delbaen, F., Eber, J.-M., & Heath, D. (1999). Coherent measures of risk. *Mathematical Finance*, 9(3), 203–228.

### A5. Stratified profiling analysis

**Effort:** 2 days. **Priority:** Medium.

Tag each profiling run with its input complexity tier (easy/medium/hard/adversarial) from the auto-generation step. Compute per-tier cost distributions separately. When projecting, mix strata with user-specified weights or default to the generation weights (35/30/20/10/5).

This partially mitigates the input distribution mismatch problem (the #1 silent failure mode). It enables:

- **Guaranteed tail representation.** 5% adversarial inputs means at least 1 extreme case at n=20.
- **Reduced variance** of percentile estimates through stratified estimation.
- **Post-profiling reweighting.** "What if production is 60% easy instead of 35%?" without re-profiling. A `--traffic-mix` CLI flag lets users specify their production difficulty distribution.

The auto-generator already produces tiered inputs. The change is to label the runs, compute per-tier means and variances, and combine using the standard stratified estimator.

### A6. Robust estimator substitutions

**Effort:** 1 day. **Priority:** Low-medium.

Replace CV (σ / μ) with MAD-based coefficient of variation (1.4826 × MAD / median) in pattern detection thresholds. Standard CV is sensitive to a single extreme run; MAD-based measures are resistant.

Changes:

| Detector | Current | Replacement |
|----------|---------|-------------|
| Loop count variance | CV > 0.5 | robust_cv > 0.5 (MAD/median) |
| Step count variance | CV > 0.3 / 0.6 | robust_cv > 0.3 / 0.6 |

Threshold values may need recalibration against the synthetic distribution test suite after switching.

---

## Part B — Data Schema Additions

The principle: record raw signals at collection time, derive features at training time. Data you discard in v1 is data you can never recover for future model training. These fields cost near-zero in runtime and storage.

### B1. StepRecord additions

| Field | Type | Purpose |
|-------|------|---------|
| tool_name | optional str | Identifies which tools drive cost. Needed for tool-filtering recommendations. |
| tool_input_tokens | optional int | Measures what the agent sends to tools. |
| tool_output_tokens | optional int | Measures tool response size — the variable that drives RAG cost variance. |
| tool_success | optional bool | Tool failure triggers retries = cost multiplier. |
| tool_retry_count | optional int | Direct cost multiplier signal. |
| cache_hit_tokens | optional int | For DeepSeek/Anthropic cache-aware pricing. |
| cache_miss_tokens | optional int | The expensive tokens. |
| model_version | optional str | Full version string for drift detection. |
| temperature | optional float | Higher temperature = more variable output = wider cost distribution. |
| max_tokens_setting | optional int | Captures whether output was constrained. |
| output_truncated | optional bool | True if output hit max_tokens — suggests model wanted to generate more. |
| output_tool_call_count | optional int | Number of tool calls in the output. |
| output_format | optional str | "json" / "text" / "function_call" — structured output affects token count. |

### B2. RunRecord additions

| Field | Type | Purpose |
|-------|------|---------|
| active_step_list | list[str] | Which steps actually executed in this run. Essential for step count variance detection. |
| step_execution_order | list[str] | Ordered list of steps as executed. Captures routing decisions. |
| loop_exit_reason | optional str | Why did the loop stop? "max_iterations" / "confidence_threshold" / "task_complete". Explains cost variance. |
| total_tool_calls | int | Sum across all steps. Quick cost attribution signal. |
| input_complexity_tier | optional str | "easy" / "medium" / "hard" / "adversarial" — for stratified analysis (A5). |

### B3. ProfileSession additions

| Field | Type | Purpose |
|-------|------|---------|
| python_version | str | Environment reproducibility. |
| sdk_versions | dict | Framework versions (langchain, openai, anthropic SDK). |
| api_endpoints | dict | Per-provider endpoint URLs. Catches region differences. |
| git_commit_hash | optional str | Links profile to code version. Essential for GitHub Action. |
| git_branch | optional str | PR branch identification. |
| git_diff_summary | optional str | Brief description of changes for CI context. |
| profiling_start_time | datetime | Session timing. |
| profiling_end_time | datetime | Session duration. |
| inter_request_delay_ms | optional int | Captures rate-limiting behavior. |

### B4. WorkflowRecord additions

| Field | Type | Purpose |
|-------|------|---------|
| workflow_fingerprint | str | SHA-256 hash of graph topology + model names + prompt hashes. Primary key for cross-session tracking and staleness detection. Replaces separate age/step-hash/prompt-hash checks with one canonical identifier. |
| fingerprint_version | int | Version of the fingerprinting algorithm. Increment when hash inputs change. |
| graph_adjacency_list | dict | Step-to-step connections. Captures workflow structure. |
| graph_edge_types | dict | "sequential" / "conditional" / "parallel" per edge. |
| step_model_map | dict | Step name → model name. Quick lookup for model swap recommendations. |
| total_step_count | int | Number of defined steps (not active steps per run). |

**Fingerprint design guidance.** Too sensitive (changes on irrelevant edits) and profiles can't be linked across sessions. Too coarse (groups different workflows) and comparison data gets contaminated. The hash should include graph topology, model identities, and prompt hashes, but NOT file paths, variable names, or comments.

### B7. What NOT to store

| Don't store | Store instead | Reason |
|-------------|---------------|--------|
| Raw input text | Token count + tier label + vocabulary richness | Privacy |
| Raw output text | Token count + format + truncation flag + tool call count | Privacy and storage |
| Raw system prompt (default) | Hash + token count. Offer raw text as opt-in. | Privacy |
| Full conversation histories | Turn count + accumulated context per turn + growth rate | Privacy |
| API keys or credentials | Nothing | Security |

---

## Model Choice Optimization

The backtesting validates the projection engine's math, not the models. The engine processes cost distributions — it doesn't care whether a data point is $0.003 (DeepSeek) or $0.50 (Opus). Statistical properties (skew, variance, multimodality) exist at any price point.

What expensive models are actually needed for:

- Validating the collector parses that provider's API response format correctly → needs ≥1 workflow per provider
- Validating the pricing table for that provider → needs ≥1 workflow per provider
- Testing provider-specific cost patterns (reasoning tokens, caching, tiered pricing) → needs ≥1 workflow per provider

Everything else — engine math, pattern detection, Monte Carlo, calibration — is model-agnostic.

**Strategy: use the cheapest viable models for heavy computation, maintain provider diversity with cheap workflows.**

Anthropic coverage is already strong with 5 workflows (W1, W2, W5, W16, W17). W2 includes Opus in loops — that's sufficient for Opus validation. W4 and W15 don't need expensive Anthropic models; swap to DeepSeek/Qwen for the heavy generation steps. This also improves DeepSeek coverage from 1 workflow (W12, simple) to 5 workflows including complex patterns.

---

## Cache-Cold Profiling Strategy

DeepSeek cache-hit input pricing is $0.0028/MTok vs. $0.14/MTok cache-miss — a 50× difference. When profiling 50+ runs in rapid succession, the provider's prompt cache stays warm for runs 2+. If 60% of input tokens are in the cacheable system prompt prefix and input tokens are 70% of total cost, the per-run cost ratio between cache-warm and cache-cold profiling is ~21×. A projection based on cache-warm profiling that says "$500/month" could be "$10,000/month" in production with sparse traffic.

**Default: cache-cold (cache-busting) for all DeepSeek workflows.** Append a unique random suffix to the system prompt for each profiling run. This ensures every run pays full cache-miss pricing, producing projections that are correct regardless of production traffic volume. If production traffic is high and the cache stays warm, actual costs are lower than projected — the safe direction for a budgeting tool.

**Cache-warm comparison run.** Additionally, run 50 cache-warm (no cache-busting) runs on W12 (cheapest DeepSeek workflow, ~$0.50). Compare per-run costs between cache-warm and cache-cold. The ratio becomes a reportable cache discount factor: "Projected at full pricing. If production traffic keeps the prompt cache warm, apply an estimated X× discount on input costs."

**Applies to Anthropic too.** Anthropic auto-caches repeated prefixes (90% cheaper for cached reads). The same cache-busting strategy should apply to rapid Anthropic profiling runs (W1, W2, W5, W13, W14, W16, W17). The impact is smaller (90% discount vs. DeepSeek's 98% discount) but still material at scale.

**B1 schema integration.** The cache_hit_tokens and cache_miss_tokens fields (B1) capture actual cache behavior per request regardless of mode. Long-term, this enables cache-aware projection: price each request using its actual cache breakdown. For v1, cache-cold default with the comparison ratio is simpler and safer.

---

## Profiling Tiers (Revised)

**n=20 is dropped entirely.** The 36% probability of missing the true p95 is unacceptable for a tool that informs budget decisions. n=20 doesn't give enough data for the jackknife+ conformal intervals (A1) to produce useful prediction intervals either.

**n=50 is the default.** At n=50: p50 ±12–18%, p95 miss probability 8%, rare events >6% captured, minimum detectable regression ~20%. Profiling cost at $0.10/run: $5. Trivial for business users projecting $5K–$50K/month production costs.

**n=100 is budget grade.** For CI regression gating, formal budgeting, and optimization validation where precision matters.

| Tier | Sample size | When to use |
|------|------------|-------------|
| Standard (default) | n=50 | Pre-deployment budgeting, cost estimation |
| Budget grade | n=100+ | CI regression detection, formal procurement |

---

## Updated Test Suite (14 Workflows)

**Cut from v1:** W3 (simple code review), W6 (complex extraction), W7 (simple research) — redundant. W10 (artificial mixed sales) — replaced by cross-provider RAG. W8 (generic research) — redundant with W15 (agentic multi-hop RAG). Saves $458 total.

**Redesigned:** W4 — from code review to compliance/document review, swapped to DeepSeek V4 for generation (massive cost reduction, identical pattern testing). W5 — multimodal extraction with structured output.

**Model-optimized:** W4 and W15 swapped to DeepSeek/Qwen for heavy generation. Savings used to increase ground truth to 500 on key workflows and add two previously-deferred workflows (W18 long-document, W19 multi-turn sessions).

| # | Workflow | Models | Ground truth | Est. cost | Key patterns tested |
|---|----------|--------|-------------|-----------|-------------------|
| W1 | Support agent | Haiku 4.5, Sonnet 4.6 | 200 | $4 | Baseline |
| W2 | Support agent (complex) | Haiku 4.5, Sonnet 4.6, Opus 4.7 | **500** | $182 | Loop variance, context growth, Opus validation |
| **W4** | **Compliance/document review** | **DeepSeek V4, Qwen 3.6 Plus** | **500** | **$15** | Self-reflection loops, context growth |
| **W5** | **Multimodal extraction + structured output** | Sonnet 4.6 (vision) | 220 | $18 | Vision tokens, JSON mode |
| W9 | Sales/outreach (OpenAI) | GPT-5.4 Nano, GPT-5.4 | 200 | $4 | OpenAI generation pricing |
| W11 | Support (Qwen) | Qwen-Turbo, Qwen 3.6 Plus | 200 | $1 | Qwen pricing |
| W12 | Extraction (DeepSeek) | DeepSeek V4 Flash | 200 | $2 | DeepSeek pricing, cache |
| W13 | Routing agent | Haiku 4.5, Sonnet 4.6 | 300 | $22 | Step count variance, bimodality |
| **W14** | **Simple PDF RAG + structured output** | OpenAI embed + Sonnet 4.6 | 300 | $38 | Retrieval variance, cross-provider |
| **W15** | **Agentic multi-hop PDF RAG** | OpenAI embed + Gemini 2.5 Flash + **DeepSeek V4** | **500** | **$55** | All 4 cost models combined |
| **W16** | **Map-reduce PDF analysis** | Haiku 4.5, Sonnet 4.6 | 300 | $19 | Fan-out, variable N, parallel |
| **W17** | **Insurance claims agent** | Haiku 4.5, OpenAI embed, Sonnet 4.6 | **500** | **$27** | Real-world decision tree, multi-doc RAG |
| **W18** | **Long-document single-pass** | **DeepSeek V4** | **500** | **$9** | Long context (50K–100K tokens), cost scaling |
| **W19** | **Multi-turn conversation (8 turns)** | **DeepSeek V4** | **500** | **$65** | Session accumulation, context growth across turns |

All DeepSeek workflows (W4, W12, W15, W18, W19) profiled in cache-cold mode. Costs reflect cache-miss pricing.

Provider coverage: Anthropic (W1/2/5/13/14/16/17 — 7 workflows including Opus in W2), OpenAI generation (W9) + embeddings (W14/15/17), Gemini (W15), Qwen (W4/W11), DeepSeek (W4/W12/W15/W18/W19 — 5 workflows including complex patterns and long context).

---

## Backtesting Methodology (Revised)

### Subsampling calibration (zero additional API cost)

With ground truth at 500 runs on key workflows, validate projections at both n=50 and n=100 via subsampling:

1. Draw 200 random subsamples of n=50 from the 500 ground truth runs. Run the full projection engine (pattern detection → mode selection → projection) on each subsample. Measure all calibration metrics against the full 500.
2. Draw 200 random subsamples of n=100. Same procedure.
3. Plot calibration curves: metric pass rate vs. sample size, per workflow.

This gives statistically robust calibration (200 independent trials per tier per workflow) instead of a single pass/fail coin flip. Report empirically: "at n=50, p50 ratio passes 94% of subsamples. At n=100, 98%."

### Tightened calibration thresholds for n=50

n=50 projections are tighter than n=20, so thresholds should tighten correspondingly. Calibrate exact values from the subsample analysis — pick the tightest threshold where ≥90% of subsamples pass.

Starting thresholds (to be refined empirically):

| Metric | n=50 threshold | Rationale |
|--------|---------------|-----------|
| p50 ratio | 0.8–1.7× | Tighter than n=20's 0.7–2.0×. n=50 should estimate median within ±30% worst case. |
| p95 coverage (simple) | ≥88% | Up from 85%. Stable distributions should hit 90%+ at n=50. |
| p95 coverage (complex) | ≥80% | Up from 75%. Still tolerant of heavy-tailed noise. |
| Range ratio (simple) | <2.5× | Down from 3×. Simple workflows should produce tight ranges at n=50. |
| Range ratio (complex) | <6× | Down from 8×. n=50 captures more of the distribution. |
| Top step correct | Yes (25% co-dominant) | Down from 30%. n=50 better distinguishes close steps. |

---

## New Workflow Designs

### W4 (Redesign): Compliance/Document Review with Self-Reflection

Same self-reflection loop pattern as the original code review (generate draft → critique → revise, output feeds back as context), applied to business documents (contracts, compliance reports, policy drafts). **Models swapped to DeepSeek V4 + Qwen 3.6 Plus** — identical pattern testing at ~$0.015/run instead of ~$0.68/run. Self-reflection and context growth are model-agnostic behaviors. Anthropic is already validated by W1/W2/W5. Expected patterns: context growth (critique accumulates), loop variance (2–8 revision iterations).

### W5 (Redesign): Multimodal Extraction with Structured Output

Accepts a mix of text documents, images (screenshots, scanned receipts, photos), and PDFs with embedded charts/tables. Output is structured JSON conforming to an extraction schema (using JSON mode). Models: Sonnet 4.6 with vision. Cost driver: input modality — image-heavy inputs cost 3–10× more than text. Tests vision token capture by the collector and structured output token behavior.

### W14: Simple PDF RAG with Structured Output

The most common production RAG pattern. Upload PDF(s) → extract text from text pages (non-LLM) → vision model for image-heavy/scanned pages → chunk → embed (OpenAI text-embedding-3-small) → query → vector retrieve (variable chunk count) → generate structured JSON answer (Sonnet 4.6 JSON mode). No loops. Cross-provider (OpenAI embeddings + Anthropic generation). Cost driver: retrieval size variance (500–20K tokens). Input set: 30–50 diverse PDFs as corpus, 50+ queries spanning sparse-to-dense retrieval.

### W15: Agentic Multi-Hop PDF RAG

Same PDF ingestion as W14 but with multi-hop retrieval. Query → retrieve → assess sufficiency (Gemini 2.5 Flash) → re-retrieve if insufficient → repeat 1–4 hops → generate comprehensive answer (**DeepSeek V4**) with all accumulated context. This is the first workflow exercising all four cost model adjustments simultaneously: retrieval size variance × loop count variance × context growth × bimodality (1-hop easy queries vs. 4-hop hard queries). Multi-provider: OpenAI embeddings + Gemini assessment + DeepSeek generation + Claude Vision for image pages. **Model swap from Opus to DeepSeek drops cost from ~$0.65/run to ~$0.07/run, enabling 500 ground truth runs for $35.**

### W16: Map-Reduce PDF Analysis

Upload long PDF → splitting step identifies N sections (N varies 3–20 with document length) → N parallel Haiku 4.5 calls process each section → Sonnet 4.6 aggregation combines all results. Tests parallel fan-out with variable N, collector behavior with parallel execution, and aggregation cost scaling. Expected patterns: step count variance (N varies), high token variance.

### W18: Long-Document Single-Pass Processing

Previously deferred due to cost — profiling 50K–100K token inputs was expensive with Anthropic/OpenAI models. DeepSeek V4 at $0.14/MTok input makes it $0.01/run.

Input: a single long PDF (50–100 pages) processed in one context window without map-reduce splitting. Tasks: full-document summarization, specific question answering over the entire document, key findings extraction. Model: DeepSeek V4 (128K context window). Input tokens: 30K–100K per run depending on document length.

This tests: long-context cost scaling (does the projection extrapolate correctly when per-run input tokens are 100× higher than typical workflows?), whether the high token variance detector fires appropriately on documents of varying length, and the known limitation around tiered pricing boundaries (validated indirectly — DeepSeek doesn't tier, but the engine math is tested at high token counts).

Input set: PDFs from the shared corpus, filtered to 30+ pages. Mix of text-heavy reports (30K tokens), dense annual reports (60K tokens), and long transcripts/depositions (80K–100K tokens).

Expected patterns: high token variance (document length drives cost variance). No loops, no routing. Cost structure is dominated by input tokens — the simplest possible cost model at the highest possible token scale.

### W19: Multi-Turn Conversation Session (8 Turns)

Previously deferred as a documentation-only limitation ("session accumulation causes 2–3× underestimate for conversational agents"). DeepSeek V4 makes it testable: an 8-turn conversation costs ~$0.08/run.

The workflow simulates a realistic customer support conversation: user sends message → agent responds → user follows up → agent responds (with full conversation history in context) → repeat for 8 turns total. Each turn accumulates the entire conversation history as context, so input tokens grow roughly linearly with turn number.

Turn-level cost structure:
- Turn 1: system prompt (500 tokens) + user message (200 tokens) = 700 input tokens. Cost: ~$0.0001.
- Turn 4: system prompt + turns 1–3 history (~3,000 tokens) + user message = ~3,700 input tokens. Cost: ~$0.0005.
- Turn 8: system prompt + turns 1–7 history (~8,000 tokens) + user message = ~8,700 input tokens. Cost: ~$0.0012.
- Total 8-turn session: ~$0.005 in input tokens + output tokens.

The profiling captures per-turn costs within each session. The engine can then:
1. Compare single-turn projection (what the engine would produce from profiling turn 1 only) to actual 8-turn session cost.
2. Calibrate the session multiplier empirically: actual_session_cost / single_turn_cost. This ratio becomes the `--session-depth` correction factor.
3. Validate that the context growth detector fires across turns (iteration = turn number, context_size = accumulated history).

**This converts a documented limitation into a validated feature.** Instead of shipping with "multi-turn sessions may cost 2–3× more," ship with "for 8-turn sessions, apply a 2.4× correction factor (empirically calibrated from W19)."

Expected patterns: context growth (linear across turns — Pearson should catch this), high token variance (early turns are cheap, late turns are expensive). The session multiplier is the primary output — it feeds the `--session-depth` flag.

---

## W17: Insurance Claims Review Agent — Full Architecture

Based on a real production use case (health insurance claims processing). This is the integration test of the suite — a single workflow combining conditional routing, multi-document RAG with PDF, structured output, function calling, and variable step topology.

### System Context

The agent receives a structured claim JSON and consults unstructured policy PDFs (one per insurance provider) to decide the next action. The action set is fixed:

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

The agent retrieves from the correct provider's policy PDF — United Healthcare claims use the UHC policy, Aetna claims use the Aetna policy. Multi-document RAG where document selection is part of the retrieval logic.

**Embedding model:** OpenAI text-embedding-3-small. Each policy PDF is pre-processed into a vector store at profiling setup time (one-time cost). At runtime, the retrieval query is constructed from the claim's procedure code, diagnosis code, and claim type, scoped to the correct provider's document.

Retrieval size varies: simple office visit matches 1–2 policy sections (~500 tokens). Complex biologic injection pre-approval matches 5–8 sections (~4,000 tokens).

### Step 3 — Evaluate & Decide

**Model:** Sonnet 4.6 with JSON mode. **Input:** claim JSON + retrieved policy sections + workflow rules for the classified claim type.

Type-specific logic:

**Pre-approval:** Check coverage criteria, verify documentation completeness, assess medical necessity against policy standard, decide approve or deny.

**Standard:** Validate coverage, check procedure/diagnosis code consistency, verify no exclusions apply, decide approve with payment or deny.

**Appeal:** Retrieve prior denial reason from claim data, assess whether appeal documentation is new, sufficient, and relevant to denial reason, re-evaluate medical justification, decide overturn or uphold.

Output schema:
```json
{
  "action": "approve_claim_payment",
  "action_params": { "payment_in_dollars": 2200.0 },
  "reason": "MRI lumbar spine covered under policy §5.1. Conservative treatment attempted per clinical note.",
  "evidence": ["Policy §5.1: MRI covered when...", "Clinical note: Persistent low back pain..."],
  "confidence": "high",
  "flags": []
}
```

### Step 4 — Conditional Routing (When Flagged)

**Model:** Haiku 4.5. If `claimed_amount > 5000` or code inconsistency was flagged, wrap the Step 3 decision with a routing action.

### Cost Per Claim Scenario

| Claim scenario | Steps | Est. cost |
|---------------|-------|-----------|
| Inactive member | 1 | ~$0.002 |
| Missing docs | 1–2 | ~$0.005 |
| Simple standard (low amount) | 3 | ~$0.04 |
| Standard with code mismatch | 4 | ~$0.06 |
| Pre-approval (full) | 3 | ~$0.08 |
| High-amount pre-approval | 4 | ~$0.10 |
| Appeal (full) | 3 | ~$0.10 |
| Complex appeal, high amount | 4 | ~$0.15 |

### Expected Patterns

- **Step count variance:** 1–4 steps, CV > 0.5 → DANGER
- **Bimodality:** short-circuits ~$0.003 vs. full workflows ~$0.08 → GMM detection (A2)
- **High token variance:** retrieval 500–4,000 tokens
- **Structured output:** JSON mode affects output tokens

### Input Set (40–50 Claims)

5 inactive-member, 5 missing-docs, 10 simple standard, 5 code mismatch, 8 pre-approval (varying doc quality), 5 high-amount, 7 appeals (varying evidence quality), 5 edge cases (conflicting evidence, ambiguous notes, borderline $5K). Distribute across 3–4 insurance providers, each with a different policy PDF.

---

## PDF Processing Pipeline

Shared across W14, W15, W16, W17. Design goal: friction-free — user uploads PDF, system handles everything.

### Page-Level Classification

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

Semantic chunking by section headers when detectable, falling back to 512-token overlapping windows (64-token overlap). Metadata: source PDF, page numbers, section header. Enables W17's multi-document RAG to scope retrieval to the correct provider.

### Robustness — Edge Cases

| Edge case | Detection | Handling |
|-----------|-----------|----------|
| Corrupted/malformed PDF | pdfplumber exception | Skip, log, continue |
| Password-protected PDF | PDFPasswordIncorrect | Skip, log |
| Very large PDF (100+ pages) | Page count check | Batch in 20-page groups |
| Empty/blank pages | No text AND no images | Skip silently |
| Non-English text | langdetect | Process normally |
| Tables spanning pages | extract_tables() | Row-by-row, cross-page merge attempt |
| Very low resolution scans | Garbled vision output | Include as-is (realistic) |
| All-image PDF (slide decks) | All pages scanned | Every page through vision (10–50× cost) |

### Processing Cost by PDF Type (20 pages)

| Type | Cost |
|------|------|
| Text-only | ~$0.00 |
| Mixed (5 image pages) | ~$0.025 |
| Scanned (all pages) | ~$0.15 |
| Slide deck (30 pages) | ~$0.30 |

---

## Updated Budget

### Model optimization + cache-cold pricing

| Item | Cost |
|------|------|
| W1 (200 runs, Haiku+Sonnet) | $4 |
| W2 (550 runs, Haiku+Sonnet+Opus) | $182 |
| W4 (550 runs, DeepSeek+Qwen, cache-cold) | $15 |
| W5 (270 runs, Sonnet vision) | $18 |
| W9 (250 runs, GPT-5.4) | $4 |
| W11 (250 runs, Qwen) | $1 |
| W12 (250 runs, DeepSeek, cache-cold) | $2 |
| W13 (350 runs, Haiku+Sonnet) | $22 |
| W14 (350 runs, OpenAI embed + Sonnet) | $38 |
| W15 (550 runs, OpenAI embed + Gemini + DeepSeek, cache-cold) | $55 |
| W16 (350 runs, Haiku+Sonnet) | $19 |
| W17 (550 runs, Haiku + OpenAI embed + Sonnet) | $27 |
| W18 (550 runs, DeepSeek, cache-cold) | $9 |
| W19 (550 runs, DeepSeek, cache-cold) | $65 |
| W12 cache-warm comparison (50 runs) | $1 |
| Pricing validation | $5 |
| **Subtotal** | **~$467** |
| Skewed-distribution variants (W2, W4, W15) | ~$80 |
| Contingency / re-runs | ~$150 |
| **Total budget** | **~$697** |

**$697 total — 14 workflows, 500 ground truth on 7 key workflows, cache-cold on all DeepSeek runs, 73% of the original $950 ceiling.** The remaining $253 of headroom absorbs cost estimation errors and allows additional exploratory profiling.

---

## Additional Pre-Backtesting Items

New items on top of v1's existing pre-backtesting list:

**Workflow engineering:**

| # | Action | Effort |
|---|--------|--------|
| 1 | Build W14 simple PDF RAG workflow | 1 day |
| 2 | Build W15 agentic multi-hop RAG workflow (DeepSeek generation) | 1–2 days |
| 3 | Build W16 map-reduce workflow | Half day |
| 4 | Build W17 insurance claims agent | 1–2 days |
| 5 | Build W18 long-document single-pass workflow | Half day |
| 6 | Build W19 multi-turn conversation workflow (8 turns, per-turn recording) | 1 day |
| 7 | Redesign W5 with multimodal inputs + structured output | Half day |
| 8 | Reframe W4 as compliance/document review with DeepSeek+Qwen | Half day |
| 9 | Curate shared PDF corpus (30–50 PDFs, tagged, including 30+ page docs for W18) | Half day |
| 10 | Collector unit test: vision/image token capture | 2 hrs |
| 11 | Collector unit test: parallel execution handling | 2 hrs |
| 12 | Collector unit test: structured output (JSON mode) | 1 hr |
| 13 | Collector unit test: per-turn token recording for multi-turn sessions | 2 hrs |
| 14 | Implement cache-busting for all DeepSeek profiling runs (unique suffix per run) | 2 hrs |
| 15 | Implement cache-busting for Anthropic profiling runs (same mechanism) | 1 hr |
| 16 | Verify W4 self-reflection structure (output feeds back as context) | 1 hr |

**Statistical method implementation:**

| # | Action | Effort |
|---|--------|--------|
| 12 | Implement jackknife+ conformal prediction intervals (A1) | 2–3 days |
| 13 | Implement BIC-based GMM bimodality detection + MC integration (A2) | 1 day |
| 14 | Implement BCa bootstrap (A3) | Half day |
| 15 | Implement CVaR computation from Monte Carlo samples (A4) | Half day |
| 16 | Implement stratified profiling analysis with --traffic-mix flag (A5) | 2 days |
| 17 | Replace CV with robust_cv (MAD/median) in detectors (A6) | 1 day |

**Schema additions:**

| # | Action | Effort |
|---|--------|--------|
| 18 | Add B1 fields to StepRecord (tool details, cache, model_version, truncation) | Half day |
| 19 | Add B2 fields to RunRecord (active_step_list, loop_exit_reason, complexity_tier) | Half day |
| 20 | Add B3 fields to ProfileSession (environment snapshot, git context) | Half day |
| 21 | Add B4 fields to WorkflowRecord (fingerprint, graph topology, step_model_map) | 1 day |
| 22 | Document B7 privacy guidelines (what not to store) | 2 hrs |

**Profiling and calibration changes:**

| # | Action | Effort |
|---|--------|--------|
| 23 | Drop n=20 tier entirely; set n=50 as default, n=100 as budget grade | 1 hr |
| 24 | Implement subsampling calibration: 200 subsamples at n=50 and n=100 per workflow | Half day |
| 25 | Implement tightened calibration thresholds for n=50 (p50 0.8–1.7×, p95 cov ≥80–88%) | 2 hrs |
| 26 | Update synthetic test suite to validate conformal interval coverage | Half day |
| 27 | Update calibration metrics to include conformal intervals and CVaR | Half day |
| 28 | Validate GMM-BIC on W13/W17 (should detect bimodality) and W1 (should not) | 2 hrs |
| 29 | Validate stratified reweighting doesn't degrade p50 accuracy | 2 hrs |
| 30 | Plot calibration curves (accuracy vs. sample size) from subsampling results | 2 hrs |
| 31 | Run W12 cache-warm comparison (50 runs without cache-busting), compute cache impact ratio | 2 hrs |
| 32 | Calibrate session multiplier from W19 (single-turn projection vs. 8-turn actual) | Half day |
| 33 | Implement --session-depth flag using empirical W19 multiplier | Half day |
| 34 | Implement --cache-mode flag (cold default, warm option) with empirical cache discount from W12 | Half day |

**Total additional engineering:** ~18–20 days on top of v1's engine work (~7–9 days). Full v1 pre-backtesting effort: ~25–29 days (5–6 weeks for one person).

---

## Prompt & Input Design

System prompts and input generation strategy for all 14 workflows are specified in a separate deliverable (`agentcost-prompt-engineering-prompt.md` → output file). This covers:

- Production-grade system prompts for each workflow step (full text, not stubs)
- Input generation strategy targeting cost-driving dimensions per workflow (not semantic difficulty)
- Pilot calibration protocol (10 runs per tier before committing to full profiling)
- Robustness to model updates, cross-model transfer, prompt sensitivity, and input drift
- Stratified input tagging for A5 (stratified profiling analysis)

**Dependency:** prompt and input design must be finalized before running Layer 3 (real workflow profiling). Layers 1 and 2 (synthetic distributions + SWE-bench) don't depend on prompts.
