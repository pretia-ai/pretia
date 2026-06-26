# Pretia v2 Additions: Drift Specification, Pilot Calibration, and Backtest Validation

**Purpose:** This file extends `projection-engine-recommendation-addition-2.md` with three components needed to execute the backtesting suite: (1) the drift design between profiling and ground truth distributions, (2) the pilot calibration checklist that runs before committing the ~$400+ ground truth budget, and (3) the backtest validation protocol that evaluates whether the projection engine works and how fragile it is.

**Context for Claude Code:** You have `projection-engine-recommendation-addition-2.md` (engine design, statistical methods A1–A6, workflow table with per-workflow ground truth sample sizes and budgets, W17 architecture, PDF pipeline), `cross-cutting-robustness.md` (detector matrix, failure modes, non-uniformity requirements), and the technical spec. This file provides the operational procedures that connect the generated prompts and inputs (from the other directions files) to the actual backtesting runs.

**Relationship to other directions files:**
- `directions-system-prompts.md` → what the agent does (fixed prompts)
- `directions-input-generation.md` → what the agent processes (variable inputs with drift)
- `directions-pdf-generation.md` → what documents the agent reads (variable PDFs with drift)
- **This file** → how to verify the whole thing works

---

## Part 1: Drift Specification

### What Drift Measures

The projection engine profiles a workflow with n=50 runs and projects monthly cost. Production traffic differs from profiling inputs. The drift between profiling and ground truth distributions simulates this difference. The backtesting answers: how much does projection accuracy degrade when the input distribution shifts?

If the engine handles drift well (projection stays within ±15% of ground truth despite distribution shift), it's robust. If it doesn't, the `--traffic-mix` reweighting flag should recover accuracy. If reweighting doesn't help either, the drift is structural and the engine needs the user to re-profile.

### Three Uniform Drift Dimensions

These apply to all 14 workflows unless a per-workflow exception is noted.

#### Dimension 1: Tier Weight Shift

| Tier | Profiling weight | Ground truth weight |
|---|---|---|
| Easy | 40% | 55% |
| Medium | 35% | 25% |
| Hard | 20% | 12% |
| Edge | 5% | 5% |
| Extreme | — | 3% |

The ground truth adds a 5th "extreme" tier (3%) that profiling never saw. These are production outliers: context-window-filling documents, pathological loop triggers, maximum-cost code paths. They represent ~3% of traffic but can account for 15–30% of monthly cost. The engine must handle them through tail-risk detection (CVaR, A4).

**What this tests:** The projection engine's ability to extrapolate from a distribution that over-represents hard cases (profiling) to one that under-represents them (production). The stratified reweighting (A5, `--traffic-mix` flag) should compensate for this shift. The extreme tier tests tail-risk estimation from zero-shot data.

**Reweighting specification:** Tell the engine: "production traffic weights are 55/25/12/5/3." The engine reweights the per-tier cost distributions from the n=50 profiling run accordingly. Compare the reweighted projection against ground truth. If the gap between raw and reweighted projection accuracy is large, reweighting works and is essential. If small, the engine is robust to tier shifts without reweighting.

#### Dimension 2: Tone and Style Shift

**Profiling:** All inputs clean — proper grammar, consistent formatting, no artifacts. Like a QA engineer wrote them.

**Ground truth:** 70% of inputs have one or more style artifacts:
- Casual/informal phrasing
- 1–3 typos or misspellings per input
- Mixed case, inconsistent punctuation
- Run-on sentences or fragments
- Excessive whitespace, trailing characters

**What this tests:** Whether messy inputs change cost behavior. Messier inputs may cause longer model responses (clarification, restating the question), different classification outcomes, or more self-reflection loop iterations. If style shift doesn't affect cost, that's a finding (robustness). If it does, the per-tier cost distributions in profiling are biased (they only saw clean inputs).

**Implementation:** The style shift is a parameter of the input generator, not the tier assignment. A ground truth "easy" input is the same difficulty as a profiling "easy" input, just messier. Applied within each tier, not across tiers.

#### Dimension 3: Token Length Stretch

**Profiling:** Token counts at the middle of each tier's range (as specified in `directions-input-generation.md`).

**Ground truth:** Token counts stretched to 1.5–2× the profiling average for the same tier. The per-tier token ranges in `directions-input-generation.md` specify both profiling and ground truth ranges.

**What this tests:** Whether the projection engine correctly models cost as a function of input length within a tier. If it does, longer inputs produce proportionally higher costs and the projection scales. If it doesn't (e.g., the model always produces the same length output regardless of input length), the cost model is wrong.

### Two Workflow-Specific Structural Drift Exceptions

These apply in addition to the three uniform dimensions.

#### W5 — Modality Ratio Shift

**Profiling:** 70% text-only inputs, 30% image-containing inputs.
**Ground truth:** 40% text-only, 60% image-containing.

**Rationale:** The text-vs-image split drives cost independently of difficulty. An "easy" image input (simple invoice photo, few fields) costs more than a "hard" text input (complex invoice text, many fields) because vision tokens are priced differently. Tier weight shift alone doesn't capture this because modality isn't correlated with difficulty.

#### W19 — Session Depth Shift

**Profiling:** Average 5 substantive turns out of 8 (3 filler turns).
**Ground truth:** Average 7 substantive turns out of 8 (1 filler turn).

**Rationale:** Substantive turns produce longer agent responses and accumulate more context for subsequent turns. The cost difference between a 5-substantive-turn and 7-substantive-turn conversation is ~40% at turn 8 (due to linear context growth). Tier weights shift per-turn complexity, but session depth is orthogonal — a "medium" conversation with 7 substantive turns costs more than a "medium" conversation with 5 substantive turns.

### Drift Summary Table

| Workflow | Tier shift | Style shift | Length stretch | Structural drift |
|---|---|---|---|---|
| W1 | Yes | Yes | Yes | — |
| W2 | Yes | Yes | Yes | — |
| W4 | Yes | Yes | Yes | — |
| W5 | Yes | Yes | Yes | Modality ratio 70/30 → 40/60 |
| W9 | Yes | Yes | Yes | — |
| W11 | Yes | Yes | Yes | — |
| W12 | Yes | Yes | Yes | — |
| W13 | Yes (routing ratio shift) | Yes | Yes | — |
| W14 | Yes | Yes | Yes | — |
| W15 | Yes | Yes | Yes | — |
| W16 | Yes | Yes | Yes | — |
| W17 | Yes (pipeline outcome shift) | Yes | Yes | — |
| W18 | Yes | Yes | Yes | — |
| W19 | Yes | Yes | Yes | Session depth 5→7 substantive turns |

---

## Part 2: Pilot Calibration Protocol

### Purpose

Before committing to the full ground truth budget (~$400+), run a 10-run pilot per workflow to verify: (a) the infrastructure works (no catastrophic failures), (b) the cost distributions are plausible (inputs produce the right cost patterns), and (c) the expected detector activations are present.

The pilot uses profiling-distribution inputs only. Budget: ~10% of the ground truth budget per workflow (typically $0.50–$5.00 per workflow, ~$40 total across 14 workflows).

### Layer 1: Infrastructure Assertions (Binary Pass/Fail)

These checks are non-negotiable. Any failure blocks the full run for that workflow.

#### Check 1: Cache-Busting Verification (DeepSeek/Anthropic workflows)

**Applies to:** W1, W2, W4, W5, W12, W13, W14, W16, W17, W18, W19

After every API call, assert:
```
assert "{{CACHE_BUST_SUFFIX}}" not in actual_prompt  # Template was substituted
assert response.usage.prompt_cache_hit_tokens == 0    # No prefix cache hit (DeepSeek)
```

For W19: verify 8 different cache-bust suffixes were used across the 8 turns. Log all 8 suffixes per conversation.

**Failure action:** If `prompt_cache_hit_tokens > 0` on any DeepSeek call, the cache-busting mechanism is broken. Do not proceed. Debug the suffix injection.

#### Check 2: Template Substitution

**Applies to:** All workflows

Verify no `{{PLACEHOLDER}}` strings appear in the actual API request (system prompt or user message). Scan the raw request payload for `{{` and `}}`.

**Failure action:** If any placeholder is found, the template engine is broken. Do not proceed.

#### Check 3: Cross-Provider Cost Accounting

**Applies to:** W4 (DeepSeek + Qwen), W14 (OpenAI + Anthropic), W15 (OpenAI + Gemini + DeepSeek), W17 (OpenAI + Anthropic)

For each API call, verify:
1. The correct provider's pricing was applied (check the price table lookup against the model name in the API call).
2. The token count came from the API response (`response.usage`), not from local tokenization.
3. The per-call cost equals `input_tokens × input_price + output_tokens × output_price` within ±1%.

**Failure action:** If any per-call cost is off by >1%, the pricing table or token counting is wrong. Fix before proceeding.

#### Check 4: W19 History Accumulation

**Applies to:** W19 only

Verify that input token counts increase monotonically across the 8 turns of a conversation:
```
assert turn_2_input_tokens > turn_1_input_tokens
assert turn_8_input_tokens > turn_7_input_tokens
assert turn_8_input_tokens > 5 * turn_1_input_tokens  # At least 5× growth
```

**Failure action:** If input tokens are constant across turns, the conversation history is not being re-sent. The system prompt is being cached or the history append is broken.

#### Check 5: PDF Validity (W14, W15, W16, W17, W18)

For each PDF used in the pilot:
1. Opens with pdfplumber without exception.
2. Text extraction produces >100 characters for text pages.
3. `extract_tables()` returns at least one table for documents that should contain tables.
4. For W14/W15: any scanned pages in the ground truth corpus are verified to produce <10 characters of text extraction (confirming they trigger the vision path).

**Failure action:** If any PDF fails to open or produces no extractable text (for text-based pages), the PDF generation pipeline is broken.

#### Check 6: Output Schema Validation

**Applies to:** All workflows with JSON output (W2, W4, W5, W9, W12, W13 step 1, W14, W15, W16, W17)

Parse every model output as JSON. Verify all required fields are present per the schema in `directions-system-prompts.md`.

**Failure action:** If >20% of outputs fail JSON parsing, the system prompt's format enforcement is insufficient. Revise the prompt.

#### Check 7: Finish Reason

**Applies to:** All workflows

For every API call, check `finish_reason`:
- If `finish_reason == "length"` on any call, the output was truncated. The `max_tokens` setting is too low, or the prompt encourages overly long responses.
- Log the percentage of truncated outputs per workflow.

**Failure action:** If >5% of outputs are truncated, increase `max_tokens` or revise the output length constraints in the system prompt.

### Layer 2: Cost Plausibility Checks (Quantitative)

These checks verify the inputs and prompts produce the intended cost distribution. Failures indicate design problems, not infrastructure bugs.

#### Check 8: Tier Separation

**Applies to:** All workflows

Across the 10 pilot runs, compare per-run costs across tiers. The most expensive run must be at least 2× the cheapest run. For workflows with strong tier separation (W2, W13, W17), the ratio should be ≥5×.

| Workflow group | Minimum cost ratio (max/min) |
|---|---|
| W1, W9, W11, W12 (linear) | ≥2× |
| W2, W4, W15, W19 (loop/growth) | ≥5× |
| W5, W18 (token variance) | ≥3× |
| W13, W17 (bimodal) | ≥10× |
| W14, W16 (retrieval/fan-out) | ≥3× |

**Failure action:** If tiers don't separate, either the tier definitions are too similar, or the system prompt is suppressing cost variance (e.g., fixed output length regardless of input complexity). Investigate which step's cost is not varying.

#### Check 9: Routing Ratio Verification

**Applies to:** W1 (Haiku/Sonnet), W2 (Opus trigger), W13 (Path A/B/C), W17 (short-circuit/full/routed)

Verify the actual routing matches the designed tier distribution within ±15%:
- W13: 70% of easy inputs route to Path A, 20% of medium to Path B, 10% of hard to Path C.
- W17: ~15% of inputs short-circuit (intake only), ~70% go full pipeline, ~15% trigger routing.
- W2: ≥1 of the 10 pilot runs should trigger the Opus review step.

**Failure action:** If routing ratios are off by >15%, the classification criteria in the system prompt don't match the input design. Either the inputs are ambiguous or the classification boundaries are wrong. Cross-check the input tier labels against the actual classification output.

#### Check 10: Per-Run Cost Plausibility

**Applies to:** All workflows

Compare each pilot run's total cost against the expected per-run cost from `projection-engine-recommendation-addition-2.md` (derive from total budget ÷ ground truth sample size). Each run should fall within 0.5–5× of the expected per-run cost.

| Workflow | Expected per-run cost (from v2 additions) | Plausible range |
|---|---|---|
| W1 | ~$0.02 | $0.01–$0.10 |
| W2 | ~$0.36 | $0.08–$1.80 |
| W4 | ~$0.03 | $0.005–$0.15 |
| W5 | ~$0.08 | $0.02–$0.40 |
| W9 | ~$0.02 | $0.005–$0.10 |
| W11 | ~$0.005 | $0.001–$0.025 |
| W12 | ~$0.01 | $0.002–$0.05 |
| W13 | ~$0.07 | $0.003–$0.35 |
| W14 | ~$0.13 | $0.03–$0.65 |
| W15 | ~$0.11 | $0.01–$0.55 |
| W16 | ~$0.06 | $0.02–$0.30 |
| W17 | ~$0.05 | $0.005–$0.25 |
| W18 | ~$0.018 | $0.005–$0.09 |
| W19 | ~$0.13 | $0.05–$0.65 |

**Failure action:** If any run falls outside 0.5–5× of expected, investigate. Common causes: wrong model (e.g., Opus instead of Sonnet), pricing table error, unexpectedly long output, cache-busting failure (costs too low).

#### Check 11: Detector Pre-Activation

**Applies to:** Workflows with expected detector activations (from `cross-cutting-robustness.md` Section 9)

With only 10 runs, detectors won't fire statistically. But you can check prerequisites:

- **Context growth (W2, W4, W19):** Plot input tokens vs. step number (for loops) or turn number (W19). Must show a positive trend. If flat, context isn't accumulating.
- **Loop count variance (W2, W4):** Across 10 runs, loop counts must span a range of at least 3 (e.g., 3–6, not 4–4–4–5–4–4–5–4–4–4).
- **Bimodality (W13, W17):** The 10 runs should include at least 2 runs in the "expensive" mode and at least 5 in the "cheap" mode. If all 10 are in one mode, the input distribution is wrong.
- **Step count variance (W13, W16, W17):** At least 2 different step counts must appear in the 10 runs.

**Failure action:** If a prerequisite isn't met, the input set doesn't exercise the intended cost pattern. Revise the inputs or check the system prompt for constraints that suppress variance.

### Pilot Execution Order

Run workflows cheapest-first to establish infrastructure before committing to expensive workflows:

1. **W11, W12** (~$0.05 total) — cheapest, validates DeepSeek and Qwen infrastructure
2. **W1, W9** (~$0.20 total) — validates Anthropic and OpenAI infrastructure
3. **W13** (~$0.70) — validates routing, tests bimodality prerequisite
4. **W4, W14, W16** (~$2.50 total) — validates PDF pipeline, self-reflection loops
5. **W5** (~$0.80) — validates vision token processing
6. **W17** (~$0.50) — validates the full integration test (multi-provider, function calls, override rules)
7. **W18, W19** (~$1.50 total) — validates long-context and multi-turn
8. **W2** (~$3.60) — most expensive, runs last (Opus usage)
9. **W15** (~$1.10) — multi-provider integration, runs after all individual providers are validated

Total pilot budget: ~$11. If any workflow fails, fix and re-pilot only that workflow.

### Pilot Report

After the pilot, produce a structured report:

```json
{
  "pilot_date": "2026-06-XX",
  "model_versions": {
    "claude-haiku-4.5": "...",
    "claude-sonnet-4.6": "...",
    "claude-opus-4.7": "...",
    "deepseek-v4": "...",
    "qwen-3.6-plus": "...",
    "gpt-5.4-nano": "...",
    "gpt-5.4": "...",
    "gemini-2.5-flash": "..."
  },
  "per_workflow": {
    "W1": {
      "infrastructure_checks": {"cache_bust": "PASS", "template_sub": "PASS", "schema_validation": "PASS", "finish_reason": "PASS"},
      "cost_plausibility": {"min_cost": 0.012, "max_cost": 0.065, "ratio": 5.4, "expected_ratio": ">=2x", "status": "PASS"},
      "routing_check": {"haiku_pct": 60, "sonnet_pct": 40, "expected": "varies", "status": "PASS"},
      "detector_prereqs": {"context_growth": "N/A", "loop_variance": "N/A", "bimodality": "N/A"},
      "per_run_costs": [0.012, 0.018, 0.023, 0.031, 0.014, 0.065, 0.019, 0.027, 0.015, 0.022],
      "status": "PASS"
    }
  },
  "blocked_workflows": [],
  "total_pilot_cost": 10.85,
  "proceed_to_ground_truth": true
}
```

---

## Part 3: Backtest Validation Protocol

### The Three Comparisons

The backtest runs three configurations and compares them. Each configuration uses the same system prompts and engine settings — only the input distribution changes.

#### Comparison A: No-Drift Baseline

**Profiling set:** n=50 from the profiling distribution (40/35/20/5 tiers, clean style, moderate length).
**Ground truth:** n=GT (per-workflow sample size from v2 additions table) from the **same** profiling distribution.

**What this measures:** Engine accuracy when there's no distribution mismatch. If the engine can't project accurately from 50 to GT runs drawn from the same distribution, there's a fundamental engine or input problem.

**Expected result:** Projection within ±10% of ground truth mean, ±15% of ground truth p75, confidence interval coverage ≥85%.

#### Comparison B: Drifted

**Profiling set:** n=50 from the profiling distribution (same as Comparison A).
**Ground truth:** n=GT from the ground truth distribution (55/25/12/5/3 tiers, messy style, stretched length, plus W5 modality shift and W19 session depth shift).

**What this measures:** Engine accuracy when the input distribution shifts. This is the core backtesting question. The profiling saw one distribution; production (ground truth) looks different.

**Expected result:** Projection accuracy degrades relative to Comparison A. The question is how much. If degradation is <5 percentage points (e.g., from ±10% to ±15%), the engine is robust. If >15 percentage points, the drift is impactful and reweighting is needed.

#### Comparison C: Drifted + Reweighted

**Profiling set:** Same as Comparison B.
**Ground truth:** Same as Comparison B.
**Engine setting:** `--traffic-mix 55/25/12/5/3` — tells the engine the production tier weights.

**What this measures:** Whether the `--traffic-mix` reweighting flag recovers accuracy lost to drift. The engine reweights the per-tier cost distributions from the profiling run using the specified production weights.

**Expected result:** Accuracy recovery. If Comparison B shows ±20% error and Comparison C shows ±12%, reweighting recovered ~8 percentage points. If Comparison C shows no improvement over B, the drift is structural (style, length, modality), not distributional (tier weights), and reweighting doesn't help.

### Accuracy Metrics

For each comparison, compute:

| Metric | Definition | Target (no-drift) | Target (drifted) |
|---|---|---|---|
| Mean error | \|projected_mean − gt_mean\| / gt_mean | <10% | <20% |
| P75 error | \|projected_p75 − gt_p75\| / gt_p75 | <15% | <25% |
| CI coverage | % of gt runs within the projected 90% CI | ≥85% | ≥75% |
| Monthly projection error | \|projected_monthly − gt_monthly\| / gt_monthly | <10% | <20% |
| Tail accuracy (CVaR) | \|projected_cvar95 − gt_cvar95\| / gt_cvar95 | <25% | <40% |

Compute all five metrics per workflow. A workflow "passes" if all five metrics are within target for the relevant comparison.

### Detector Validation

After the ground truth runs for Comparison B, check every detector against the matrix in `cross-cutting-robustness.md` Section 9:

**Expected detector firings:**
- Context growth: W2, W4, W19 (strong), W15 (moderate)
- Loop count variance: W2, W4, W15
- High token variance: W5, W18, W16
- Step count variance: W2, W13, W15, W16, W17
- Bimodality: W13 (strong), W17 (strong), W15 (moderate)
- Linear (no patterns): W1, W9, W11, W12, W18

**Classification:**
- **True positive:** Detector fires where expected. Good.
- **False negative:** Detector doesn't fire where expected. Either the inputs don't produce the pattern (input generation problem) or the detector threshold is wrong (engine problem). Investigate.
- **True negative:** Detector doesn't fire where not expected. Good.
- **False positive:** Detector fires where not expected. Unexpected cost pattern. Investigate — this may reveal a genuine cost behavior the design didn't anticipate.

Log all detector results in a matrix. Any false negative blocks the workflow from "passing" the backtest.

---

## Part 4: Failure Attribution

When a workflow fails any metric in the backtest, classify the failure into one of three buckets.

### Bucket 1: Engine or Infrastructure Problem

**Symptoms:**
- Comparison A (no-drift baseline) fails. The engine can't project accurately even without distribution mismatch.
- Infrastructure assertions failed during pilot (should not happen if pilot was thorough, but may emerge at scale).
- Detector false negatives on workflows with obvious cost patterns (e.g., W19 shows no context growth signal despite clearly accumulating history).

**Action:** Fix the engine or infrastructure. This is not an input or drift design problem. Common causes: pricing table error, token counting bug, detector threshold miscalibration, Monte Carlo convergence issue at n=50.

### Bucket 2: Working as Designed (Drift Sensitivity)

**Symptoms:**
- Comparison A passes but Comparison B fails. The engine works on clean distributions but not drifted ones.
- Comparison C (reweighted) recovers most of the lost accuracy. The reweighting flag works.

**Action:** This is the expected outcome for some workflows. The engine correctly identifies that production differs from profiling, and reweighting compensates. Document the accuracy gap and the reweighting recovery. The user-facing recommendation is: "Use `--traffic-mix` to specify your production distribution for best accuracy."

### Bucket 3: Flag for Investigation

**Symptoms:**
- Comparison A passes, Comparison B fails, and Comparison C doesn't recover accuracy. The drift isn't just distributional — it's structural.
- Or: Comparison A passes with marginal accuracy (just under 10% mean error), and any drift pushes it over.
- Or: A specific detector produces a false positive, indicating an unexpected cost pattern.

**Action:** Investigate per-step. Break down the cost error by workflow step to identify which step's cost is misprojected. Common causes:
- Style shift affects a specific step disproportionately (e.g., messy inputs cause more loop iterations in W2 Step 2, but the engine's loop model doesn't account for input quality).
- Length stretch interacts nonlinearly with a specific model's output behavior (e.g., DeepSeek produces disproportionately longer outputs for longer inputs, but the engine assumes linear scaling).
- The extreme tier (absent from profiling) produces cost outliers that dominate the ground truth mean but weren't in the profiling data.

These are genuine findings about the engine's limitations. Document them as known limitations with specific conditions under which projection accuracy degrades.

### Failure Attribution Flowchart

```
Does Comparison A pass?
├── NO → Bucket 1 (engine/infrastructure)
└── YES → Does Comparison B pass?
    ├── YES → All good. Workflow passes.
    └── NO → Does Comparison C recover ≥50% of lost accuracy?
        ├── YES → Bucket 2 (drift sensitivity, reweighting works)
        └── NO → Bucket 3 (investigate structural drift)
```

---

## Part 5: Ground Truth Execution

### Execution Order

Run the full ground truth in the same order as the pilot (cheapest-first), with one modification: run Comparison A (no-drift) for ALL workflows first, then run Comparison B (drifted) for all workflows, then run Comparison C (reweighted) for workflows that failed B.

This allows early detection of engine problems (if Comparison A fails across multiple workflows, it's systemic — stop and fix before spending on B and C).

### Budget Checkpoints

| Checkpoint | Cumulative spend | Decision |
|---|---|---|
| After Comparison A, all workflows | ~$200 | If ≥3 workflows fail Comparison A, stop. Engine problem. |
| After Comparison B, cheap workflows (W1/W9/W11/W12) | ~$210 | Sanity check: linear workflows should pass B easily. If they don't, something is wrong with the drift design. |
| After Comparison B, all workflows | ~$400+ | Evaluate which workflows need Comparison C. |
| After Comparison C | ~$430+ | Final results. |

### Per-Workflow Ground Truth Sample Sizes

Use the sample sizes from `projection-engine-recommendation-addition-2.md`:

| Workflow | Ground truth runs | Estimated cost |
|---|---|---|
| W1 | 200 | $4 |
| W2 | 500 | $182 |
| W4 | 500 | $15 |
| W5 | 220 | $18 |
| W9 | 200 | $4 |
| W11 | 200 | $1 |
| W12 | 200 | $2 |
| W13 | 300 | $22 |
| W14 | 300 | $38 |
| W15 | 500 | $55 |
| W16 | 300 | $19 |
| W17 | 500 | $27 |
| W18 | 500 | $9 |
| W19 | 500 | $65 |

**Note:** These are per-comparison sample sizes. Comparison A and Comparison B each use this many runs. Comparison C reuses Comparison B's ground truth data — it only re-runs the projection engine with reweighting, no new API calls.

Total ground truth budget: ~$400–450 for Comparisons A + B combined (Comparison A uses profiling-distribution inputs, which are cheaper on average). Comparison C costs nothing — it re-analyzes existing data.

---

## Part 6: Results Reporting

### Per-Workflow Results Card

```json
{
  "workflow": "W13",
  "comparison_a": {
    "mean_error_pct": 7.2,
    "p75_error_pct": 11.8,
    "ci_coverage_pct": 89,
    "monthly_error_pct": 6.5,
    "cvar95_error_pct": 18.3,
    "pass": true
  },
  "comparison_b": {
    "mean_error_pct": 16.4,
    "p75_error_pct": 22.1,
    "ci_coverage_pct": 78,
    "monthly_error_pct": 15.8,
    "cvar95_error_pct": 34.2,
    "pass": false
  },
  "comparison_c": {
    "mean_error_pct": 9.1,
    "p75_error_pct": 14.3,
    "ci_coverage_pct": 86,
    "monthly_error_pct": 8.7,
    "cvar95_error_pct": 22.1,
    "pass": true,
    "recovery_pct": 79
  },
  "failure_bucket": "bucket_2_drift_sensitivity",
  "detectors": {
    "context_growth": {"expected": false, "actual": false, "status": "true_negative"},
    "loop_variance": {"expected": false, "actual": false, "status": "true_negative"},
    "high_token_variance": {"expected": false, "actual": false, "status": "true_negative"},
    "step_count_variance": {"expected": true, "actual": true, "status": "true_positive"},
    "bimodality": {"expected": true, "actual": true, "status": "true_positive"},
    "linear": {"expected": false, "actual": false, "status": "true_negative"}
  },
  "notes": "Drift sensitivity driven by routing ratio shift (70/20/10 → 55/25/15/5). Reweighting recovers 79% of lost accuracy. Engine correctly identifies bimodality in both distributions."
}
```

### Summary Table

After all workflows complete, produce a summary:

| Workflow | A: Pass? | B: Pass? | C: Pass? | Bucket | Reweighting recovery |
|---|---|---|---|---|---|
| W1 | ✓ | ✓ | — | — | N/A |
| W2 | ✓ | ✗ | ✓ | Bucket 2 | 72% |
| W13 | ✓ | ✗ | ✓ | Bucket 2 | 79% |
| ... | ... | ... | ... | ... | ... |

### Aggregate Findings

Across all 14 workflows, report:
1. **How many pass all three comparisons?** (Engine is robust.)
2. **How many need reweighting?** (Engine works but needs `--traffic-mix`.)
3. **How many have unresolved structural drift sensitivity?** (Known limitations.)
4. **Any systematic patterns?** (e.g., "All loop-based workflows are drift-sensitive" → the loop model needs improvement.)
5. **Detector reliability:** % true positive rate, % false negative rate across all workflows.

These findings feed directly into Pretia's documentation and the user-facing guidance for the `--traffic-mix` flag.
