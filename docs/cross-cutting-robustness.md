# Cross-Cutting Robustness Framework

This framework applies to all 14 test workflows. It synthesizes the per-workflow robustness notes into unified policies that Claude Code should enforce across the entire backtesting suite.

---

## 1. Model Update Resilience

### The Problem

Every LLM provider ships model updates that change output behavior without changing the model name or API endpoint. A "Sonnet 4.6" call today may produce 20% longer outputs than the same call three months ago. These silent updates invalidate existing cost profiles.

### Impact by Workflow Category

| Category | Workflows | Model Update Impact | Severity |
|----------|-----------|-------------------|----------|
| Output-length-sensitive | W1, W2, W9, W11, W19 | Verbosity changes shift mean cost ±15–25%. Multi-turn (W19) compounds: +20% per-turn verbosity → +40% turn-8 cost. | HIGH |
| Loop-termination-sensitive | W2, W4, W15 | Changed reasoning patterns alter when models decide "resolved" vs "continue." W2 escalation threshold (iteration 4 → Opus) makes this acute. | HIGH |
| Routing/classification-sensitive | W13, W17 | Classification boundary shifts change the traffic split between cheap and expensive paths. A 10% shift in W17's override rate moves 10% of traffic between $0.0005 and $0.008. | HIGH |
| Input-dominated (low sensitivity) | W5, W12, W14, W16, W18 | Output variance is a small fraction of total cost. Model updates have minimal impact. | LOW |

### Mitigation Protocol

**Model version pinning.** Record the exact model version string (not just the model name) with every profiling run. The API response headers or metadata often include a version hash — capture and store it.

**Re-profiling trigger.** When a provider announces a model update, run the 10-run pilot for each affected workflow BEFORE running the full ground truth. Compare pilot metrics (mean cost, CV, loop counts, classification distribution) against the stored baseline. If any metric shifts >15%, the existing profile is invalidated.

**Version-specific baselines.** Store ground truth results indexed by model version. This allows longitudinal tracking: "Sonnet 4.6 v2026.03 costs $X; Sonnet 4.6 v2026.06 costs $Y."

**Prompt-level constraints as shock absorbers.** Several workflow prompts include explicit output length limits (W1: "200/400 words," W9: "120–180 words," W19: "under 200 words"). These dampen model verbosity changes. Workflows without length limits (W4, W15) are more exposed.

---

## 2. Prompt Sensitivity

### The Problem

Small prompt edits can cause large cost distribution shifts. This is especially dangerous because users can customize the shipped prompt templates.

### High-Sensitivity Prompt Elements

**Loop termination criteria.** In W2, the phrase "resolved to the customer's satisfaction" controls when the loop exits. Rephrasing to "fully resolved with all edge cases addressed" could add 2–3 extra iterations. In W4, the critique threshold ("no critical issues remain") controls the self-reflection loop depth.

**Output format constraints.** Switching from "JSON only" to allowing natural language preambles adds ~50–200 tokens per call. Over 8 iterations in W4, that's 400–1,600 extra output tokens.

**Routing thresholds in W13 and W17.** W13's tier definitions ("single data point" → TIER_1, "comparing across dimensions" → TIER_2) control the 70/20/10 traffic split. W17's override rules ($500 auto-approve threshold) are explicitly parametric.

**Word/token count limits.** W1's "200 words (simple) / 400 words (complex)" and W9's "120–180 words" directly cap output cost. Removing these limits could 2× output tokens.

### Mitigation Protocol

**Prompt hashing.** Compute a SHA-256 hash of each prompt template and store it alongside the profiling results. Any prompt modification (even whitespace changes) generates a new hash, flagging potential invalidation.

**Sensitivity testing.** For each workflow, identify the 2–3 cost-critical prompt elements (listed in each workflow's robustness notes). During pilot calibration, run variants:
- Remove output length constraints → measure output token increase
- Relax loop termination criteria → measure iteration count increase
- Shift routing thresholds → measure path distribution change

**Cost-critical elements registry.** Maintain a per-workflow list of prompt elements that, if changed, require re-profiling. Ship this as metadata alongside the prompt templates.

| Workflow | Cost-Critical Prompt Elements |
|----------|------------------------------|
| W1 | Word count limits (200/400), response detail level instruction |
| W2 | Loop exit condition phrasing, escalation iteration threshold (4), Opus trigger criteria |
| W4 | Critique satisfaction threshold, maximum iteration cap |
| W5 | JSON schema field count (more fields → longer output) |
| W9 | Word count limits (120–180, 80–100) |
| W11 | Same as W1 (shared prompt) |
| W12 | JSON schema complexity, entity extraction field list |
| W13 | Tier classification definitions, tier routing rules |
| W14 | Retrieved context window size (top_k parameter), answer length constraints |
| W15 | Sufficiency assessment threshold ("sufficient" vs "need more"), max hops |
| W16 | Chunk splitting strategy, map output length constraints |
| W17 | Override rule thresholds ($500, 365 days), evaluation detail level |
| W18 | Task count in prompt (4 tasks), output format complexity |
| W19 | Response length limit ("under 200 words"), conversation rules affecting verbosity |

---

## 3. Input Distribution Drift

### The Problem

Profiling inputs are static. Production inputs evolve. A profiling suite built with today's customer questions becomes unrepresentative six months later when the product ships new features, customer demographics shift, or seasonal patterns emerge.

### Drift Vectors by Workflow

**Semantic drift.** New product features generate question categories not in the profiling set. W1/W11 customer support questions about features that didn't exist during profiling.

**Structural drift.** Documents get longer over time (regulatory filings, compliance documents). W18's 30K–100K token range may not capture future 150K-token documents. W16's PDF corpus ages.

**Traffic mix drift.** W13's assumed 70/20/10 routing split or W17's 40/60 override rate may shift as customer demographics change. The stratified profiling design handles this via reweighting, but only if the profiling set covers the new mix adequately.

**Modality drift.** W5 assumes a specific image-to-text ratio. If production shifts toward more image-heavy documents, the vision token distribution shifts.

### Mitigation Protocol

**Tier tag reweighting.** This is the primary defense. Every input is tagged with a cost tier. When production traffic mix shifts, the projection engine can reweight without re-profiling — IF the tiers are defined by structural characteristics (document length, complexity class) rather than semantic content.

**Monitoring hooks.** The profiling harness should log the structural characteristics of each production input (token count, image count, document length). Periodically compare the production distribution against the profiling distribution. Alert when:
- Production mean token count drifts >20% from profiling mean
- A new structural category appears (e.g., video inputs in W5)
- The routing/override distribution in W13/W17 shifts >15 percentage points

**Refresh cadence.** Re-run the 10-run pilot quarterly, using recent production inputs (sampled from the monitoring logs) instead of the original synthetic inputs. If pilot costs deviate >20% from the stored baseline, re-run the full ground truth.

**Input set versioning.** Version the input sets alongside the prompt templates. Each input set has a creation date and a "valid until" recommendation (default: 6 months).

---

## 4. Cross-Model Transfer

### The Problem

Prompts designed for one model family may behave differently on another. W1 (Anthropic) and W11 (Qwen) use the same prompt text, but tokenizer differences, instruction-following behavior, and output style differ.

### Known Transfer Issues

**Tokenizer differences.** The same English text produces different token counts across providers:
- Anthropic (BPE): ~1.3 tokens per word
- OpenAI (BPE, tiktoken): ~1.3 tokens per word
- DeepSeek (BPE, custom): ~1.2–1.4 tokens per word (CJK-optimized, English varies)
- Qwen (BPE, custom): ~1.2–1.3 tokens per word (CJK-optimized)
- Google (SentencePiece): ~1.1–1.3 tokens per word

This means the prompt's token count (and therefore cost) varies by ±15% across providers even before the model generates a single output token.

**Instruction following fidelity.** Some models follow "JSON only" instructions reliably; others prepend explanatory text. This adds ~50–200 tokens of output waste. The prompt's `JSON only.` instruction is sufficient for Anthropic models but may need reinforcement (e.g., "Respond with valid JSON only. No explanation, no markdown fences.") for other providers.

**Output verbosity.** Different model families have different default verbosity. Qwen models tend to be more concise than Anthropic models for the same prompt. DeepSeek V4 can be verbose when not constrained.

### Mitigation Protocol

**Provider-specific prompt variants.** For workflows that span providers (W14, W15, W17) or have cross-provider twins (W1/W11), maintain provider-specific prompt adjustments if pilot testing shows >20% cost deviation from the reference provider.

**Tokenizer-aware token counting.** The input generation strategy should count tokens using each provider's actual tokenizer, not a universal approximation. Store both the "universal" token count and the provider-specific token count in input metadata.

**The W1↔W11 comparison as a canary.** Running identical inputs through Anthropic and Qwen models provides a direct measurement of cross-provider transfer cost. If the cost ratio deviates significantly from the pricing ratio, the prompt is behaving differently. This comparison should be run first during pilot calibration.

---

## 5. Anti-Gaming

### The Problem

The system guide describes an anti-gaming mechanism: entropy-based effective sample size (n_eff) that discounts confidence when inputs are too homogeneous. An adversarial user could deliberately choose inputs to produce misleadingly narrow cost distributions.

### Gaming Attack Vectors

**Homogeneous inputs.** Running 50 identical inputs produces a tight distribution that looks confident but doesn't represent production variance. The n_eff mechanism should catch this (n_eff << n_actual), but verify.

**Tier manipulation.** Submitting only cheap-tier inputs and claiming they represent production traffic. The stratified profiling design mitigates this (users tag inputs by tier), but if the user mis-tags intentionally, the reweighting is wrong.

**Cherry-picked inputs.** Choosing inputs that avoid expensive code paths (no escalations in W2, no TIER_3 in W13, no override failures in W17). The projection will underestimate because it never observed the expensive tail.

**Prompt manipulation.** Adding aggressive output limits ("respond in exactly 10 words") to suppress output variance. Or removing loop termination criteria to inflate costs.

### Mitigation Protocol

**n_eff validation.** The backtesting should verify that n_eff correctly identifies homogeneous input sets. Test: run W1 with 50 identical inputs → n_eff should be near 1.0, and the confidence tier should drop to LOW.

**Input diversity scoring.** Before accepting a profiling run, compute diversity metrics on the input set:
- Token count entropy across inputs (should be >60% of maximum entropy)
- Tier distribution matches the declared distribution (±10 percentage points)
- No single input appears more than 2× (duplicate detection)

**Structural coverage verification.** For workflows with branching (W2, W13, W17), verify that profiling runs actually exercised all code paths. Check:
- W2: at least 10% of runs hit escalation (iteration 4+)
- W13: all three tiers received traffic
- W17: both override and full-eval paths were taken

**Output distribution sanity checks.** Flag profiling results where:
- CV < 0.05 (suspiciously narrow)
- All runs have identical step counts (branching logic not exercised)
- n_eff / n_actual < 0.3 (severe homogeneity)

---

## 6. Cache Behavior

### The Problem

DeepSeek offers 90% input token discount on cache hits. If the backtesting accidentally runs with cache warm (repeated prompts without cache-busting), the measured costs will be up to 10× lower than production cache-cold costs.

### Affected Workflows

| Workflow | Cache Risk | Cache-Busting Method |
|----------|-----------|---------------------|
| W4 | HIGH | `{{CACHE_BUST_SUFFIX}}` per run |
| W12 | CRITICAL | `{{CACHE_BUST_SUFFIX}}` per run + verification assertion |
| W18 | HIGH | `{{CACHE_BUST_SUFFIX}}` per run |
| W19 | CRITICAL | `{{CACHE_BUST_SUFFIX}}` per TURN (8 unique suffixes per conversation) |
| All others | LOW | Non-DeepSeek models, or models without aggressive caching |

### Mitigation Protocol

**Automated cache verification.** Every DeepSeek API call must be checked:
```python
assert response.usage.prompt_cache_hit_tokens == 0, \
    f"Cache hit detected: {response.usage.prompt_cache_hit_tokens} tokens cached"
```

**Unique suffix generation.** Each `{{CACHE_BUST_SUFFIX}}` is a UUID-v4 string, generated fresh per API call (not per run, not per conversation — per call).

**Suffix placement.** The suffix must appear at the END of the prompt (after all variable content) to prevent prefix caching. Placing it at the beginning only busts the cache for the first few tokens; the remainder may still cache-hit via prefix matching.

**Reporting.** Include cache-hit rate as a mandatory field in profiling results. Any rate >0% in cache-cold mode is a bug.

---

## 7. Cross-Provider Accounting

### The Problem

Five of the 14 workflows use multiple providers in the same pipeline (W14, W15, W17 use OpenAI embeddings + other LLMs; W4 uses DeepSeek + Qwen). Token counting, pricing, and API response formats differ across providers.

### Common Accounting Errors

**Token count misattribution.** Applying Anthropic's token counter to OpenAI embedding calls, or vice versa. Each provider's `usage` field uses their own tokenizer.

**Pricing table misapplication.** Using the wrong per-token rate. OpenAI embedding tokens are ~10× cheaper per token than Anthropic Sonnet input tokens. Mixing them up inflates or deflates cost by an order of magnitude.

**Missing provider in accounting.** Forgetting to count the embedding cost in RAG workflows (W14, W15, W17). The embedding cost is small but non-zero, and omitting it systematically biases the projection downward.

**Currency/unit confusion.** Some pricing tables use $/1K tokens, others $/1M tokens. A 1000× error is possible.

### Mitigation Protocol

**Provider-tagged StepRecords.** Every StepRecord must include the provider name, model name, and the provider's own reported token counts. Never convert between providers' token counts.

**Price table pinning.** Store the exact per-token prices used for each profiling run. Record the date prices were fetched. Providers change prices without notice.

**Cross-provider reconciliation.** For multi-provider workflows, verify:
- Total cost = Σ (per_provider_cost) — no double-counting, no omissions
- Each provider's tokens sum to the expected total for that provider
- Embedding costs appear as separate line items, not folded into LLM costs

**W15 as the acid test.** W15 (three providers: OpenAI + Gemini + DeepSeek) is the most complex accounting scenario. If the profiler handles W15 correctly, it handles everything. Run W15 pilot first when debugging accounting issues.

---

## 8. Synthetic Corpus Generation

### Why Synthetic Over Real Documents

Six workflows depend on PDF documents (W5, W14, W15, W16, W17, W18). Rather than sourcing PDFs from SEC EDGAR, GAO, or other public repositories, all documents are generated synthetically. This solves three problems real corpora create:

1. **Homogeneity.** SEC filings and government reports follow standardized templates. A corpus of 75 financial documents produces nearly identical token density, section structure, and vocabulary patterns. Synthetic generation produces five distinct document types per workflow, each with different structural characteristics.
2. **Uncontrolled token counts.** Real documents have whatever token count they happen to have. Synthetic generation hits exact token targets (±5%) using the target model's tokenizer, ensuring tier boundaries are precise.
3. **Reproducibility.** Real corpus URLs break, document versions change, and downloads are fragile. Synthetic generators ship as code with deterministic seeds — anyone can reproduce the exact corpus.

### Generator Architecture

Each PDF-dependent workflow has its own generator spec (`pdf-generators/W{xx}-pdf-gen.md`) that defines:
- Per-tier structural parameters (the cost levers specific to that workflow)
- Document type rotation (5 types per workflow, distributed across all tiers)
- Pathological input specifications (5% of inputs)
- Token-precise generation using the target model's tokenizer

The generators DO NOT share a corpus. Each workflow generates documents optimized for its own cost driver — W14 controls information scatter, W15 controls reasoning chain depth, W16 controls section count, W18 controls raw token count. Sharing would force compromises that weaken the cost signal.

### Generator Versioning

Each generator is deterministic given a seed. The generator version (code hash) plus the seed constitutes a full corpus specification. Store both alongside profiling results:

```python
corpus_spec = {
    "generator_version": "sha256:abc123...",  # hash of generator code
    "seed": 42,
    "n_inputs": 75,
    "tokenizer": "deepseek-v4",  # which tokenizer was used for token counting
}
```

---

## 9. Non-Uniformity Requirements

### The Problem

Synthetic inputs default to uniformity. An LLM generating "75 diverse customer support questions" produces questions that are diverse by topic but structurally identical — similar token counts, similar sentence structures, similar complexity levels. This suppresses cost variance and makes the projection engine look artificially good.

### Non-Uniformity Mandate

Every workflow's input set MUST satisfy these diversity requirements:

**Structural diversity within tiers.** Within each tier, inputs must vary on the structural dimension that drives cost:
- Token count entropy > 60% of maximum entropy (no clustering at tier midpoint)
- No two inputs in the same tier with identical structural parameters
- Section lengths, entity counts, and complexity metrics must span each tier's full range

**5% pathological inputs (4 per workflow).** Each workflow includes inputs specifically designed to stress-test edge cases:
- Extremely short or empty inputs (minimum viable input)
- Extremely long inputs near the context limit
- Malformed or "dirty" inputs (typos, mixed encoding, non-ASCII, copy-pasted artifacts)
- Adversarial inputs that could confuse routing/classification logic

**Document type rotation.** For workflows with document types, every tier includes all available types. Don't cluster invoices in cheap and legal briefs in expensive — cost is driven by structural parameters, not domain.

**No synthetic uniformity.** If using an LLM to generate inputs, verify structural diversity post-generation:

```python
def verify_non_uniformity(inputs: list, tier: str) -> bool:
    """Reject input sets that are too uniform."""
    token_counts = [count_tokens(inp) for inp in inputs]
    
    # Token count entropy check
    entropy = compute_entropy(token_counts, bins=10)
    max_entropy = math.log(10)
    if entropy / max_entropy < 0.6:
        return False  # Too clustered
    
    # Duplicate detection
    if len(set(hash(inp) for inp in inputs)) < len(inputs) * 0.98:
        return False  # Duplicates
    
    # Range coverage
    tier_range = TIER_RANGES[tier]
    if max(token_counts) < tier_range[0] + 0.7 * (tier_range[1] - tier_range[0]):
        return False  # Not spanning the range
    
    return True
```

### Dirty Input Specification

Across all 14 workflows, ~4 inputs per workflow (56 total) are deliberately "dirty":

| Input Type | Purpose | Workflows |
|-----------|---------|-----------|
| Typos and misspellings | Tests tokenizer behavior on misspelled words | W1, W2, W9, W11, W13, W17 |
| Mixed Unicode/encoding | CJK characters, emoji, RTL text mixed with Latin | W1, W11, W12, W19 |
| Copy-pasted artifacts | Excessive whitespace, HTML tags, markdown in plain text | W2, W4, W14, W15 |
| Near-empty inputs | 5-word questions, 1-sentence documents | W1, W9, W11, W14 |
| Near-limit inputs | Documents at 95% of context window | W18, W19 |
| Adversarial routing | Inputs designed to confuse classifiers | W13, W17 |

---

## 9. Detector Validation Matrix

Each workflow is designed to trigger specific pattern detectors. This matrix maps workflows to expected detector firings. If a detector doesn't fire where expected, there's a bug in the detector or the input set.

| Detector | Expected to Fire | Expected NOT to Fire |
|----------|-----------------|---------------------|
| Context growth (Pearson + Spearman) | W2, W4, W19 (strong), W15 (moderate) | W1, W5, W9, W11, W12, W18 |
| Loop count variance | W2, W4, W15 | W1, W5, W9, W11, W12, W14, W18 |
| High token variance | W5 (vision tokens), W18 (input length), W16 (variable N) | W9 (constrained output), W12 (constrained output) |
| Step count variance | W2, W13, W15, W16, W17 | W1, W5, W9, W11, W12, W18, W19 |
| Bimodality (BIC) | W13 (strong), W17 (strong), W15 (moderate) | W1, W9, W11, W12, W18 |
| Linear projection (no patterns) | W1, W9, W11, W12, W18 | W2, W4, W13, W15, W17, W19 |

**Validation protocol.** After running the ground truth for each workflow, check every detector against this matrix. A false negative (detector doesn't fire where expected) indicates insufficient input diversity or a detector bug. A false positive (detector fires where not expected) indicates an unexpected cost pattern worth investigating.

---

## 10. Failure Mode Catalogue

### Catastrophic Failures (invalidate entire profiling run)

1. **Cache-busting failure on DeepSeek.** All DeepSeek costs are 10× too low. Detected by: `prompt_cache_hit_tokens > 0`.
2. **Wrong pricing table.** Applying provider A's prices to provider B's tokens. Detected by: cross-provider cost reconciliation.
3. **Template injection failure.** `{{PLACEHOLDER}}` appears literally in the API call instead of being replaced. Detected by: API error (malformed JSON) or anomalous token counts.
4. **History not accumulating in W19.** All 8 turns have the same input token count. Detected by: constant step cost across turns.

### Degraded Accuracy (bias results but don't crash)

5. **Input tier imbalance.** Too many cheap inputs, too few expensive. Detected by: tier distribution check.
6. **Model version mismatch.** Pilot run on model v1, ground truth on model v2. Detected by: version string comparison.
7. **Output truncation.** Model hits max_tokens limit, producing shorter-than-natural output. Detected by: `finish_reason == "length"` in API response.
8. **Thinking tokens not accounted.** DeepSeek V4 reasoning tokens billed but not visible. Detected by: `completion_tokens > visible_output_tokens`.

### Silent Failures (produce plausible but wrong results)

9. **Input distribution mismatch with production.** Profiling inputs don't match production traffic. Detected by: only production monitoring can catch this.
10. **Prompt sensitivity to formatting.** Minor whitespace/formatting differences between the template and the actual API call change tokenization. Detected by: token count verification against expected values.
11. **Synthetic corpus token count drift.** Generator produces documents at the wrong token count due to tokenizer version mismatch. Detected by: post-generation token count audit against target ±5%.
12. **Uniform synthetic inputs.** Generator produces structurally identical inputs within tiers, suppressing cost variance. Detected by: non-uniformity verification (entropy check, range coverage).
