# AgentCost Projection Engine: Comprehensive Recommendations

## Executive Summary

The projection engine is fundamentally sound. The core approach — profile N runs, compute distributional statistics, project at volume via linear or Monte Carlo methods — is correct and delivers on the value proposition of "significantly better than napkin math." The backtesting plan, with targeted fixes, will produce defensible calibration data.

This document synthesizes findings from an 11-area adversarial review plus a subsequent deep-dive on statistical methods, detection gaps, and validation strategy. It identifies **one critical math bug**, **one structural coverage gap**, **three new pattern detectors**, **an anti-gaming confidence gate**, and **a three-layer validation strategy** (synthetic distributions, SWE-bench trajectory data, and real workflow profiling). The projection engine needs approximately 7–9 days of pre-backtesting engineering work and a revised ~$850 budget allocation across 10 workflows.

**Key post-review additions:** dual Pearson+Spearman context growth detection (catches non-linear growth), step count variance and cost bimodality detectors, effective sample size gate (prevents gaming via LLM-generated inputs), tiered profiling recommendations (n=20 quick scan / n=50 standard / n=100+ budget grade), and a synthetic distribution testing layer that validates the engine across 500+ distribution shapes before spending API dollars.

---

## Part 1: Projection Engine Fixes

These are changes to the projection engine itself, ordered by severity.

### 1.1 — CRITICAL: CLT Variance Inflation Bug

**The problem.** The Monte Carlo simulator samples one run cost and multiplies by N (daily volume × 30) to get monthly cost. This inflates the variance of the monthly cost distribution by a factor of N.

The simulated monthly variance is N² × σ² (run variance). The true monthly variance is N × σ² (sum of N independent draws). At 10,000 requests/day (N = 300,000), the simulated standard deviation is ~548× too wide. A workflow costing $30,000 ± $170/month would be projected as $30,000 ± $90,000.

This means every high-volume projection produces an absurdly wide range, and p95 projections are 3–6× too high at scale. At low volumes, the overestimation is smaller but still present.

**The fix.** Replace single-sample-multiply with CLT-corrected projection. In each Monte Carlo simulation:

1. Sample K run costs (K = min(N, 1000)) with replacement from observed data.
2. Compute sample mean (μ̂) and sample variance (σ̂²) of the K samples.
3. Monthly cost for this simulation = N × μ̂ + z × √(N × σ̂²), where z is drawn from a standard normal.

Alternatively, the simpler approach: sample K costs, sum them, scale by N/K. At K = 1000, CLT convergence is reliable.

After all 10,000 simulations, compute percentiles of the monthly cost distribution as usual.

**Effort:** Half a day. **Priority:** Fix before any backtesting runs.

### 1.2 — Bootstrap Tail Truncation

**The problem.** With 20 observed runs, Monte Carlo can only sample values that were observed. The simulated p95 is bounded above by the observed maximum. There is a 36% probability that the observed maximum from 20 draws falls below the true p95 of a heavy-tailed distribution. The engine systematically underestimates the tail.

**The fix (v1 — launch).** Apply a conservative tail inflation factor for small samples. Multiply the Monte Carlo p95 by (1 + 2/√n). At n = 20, this inflates the p95 by 45%. The factor is a heuristic derived from the convergence rate of bootstrap percentile estimates for heavy-tailed distributions, which is O(1/√n) (Hall, 1988). The multiplier of 2 covers approximately 2–3 standard errors of the p95 estimate — deliberately conservative because underestimating the tail is more harmful than overestimating it for a budgeting tool. The exact factor should be calibrated empirically from the synthetic distribution testing (§5B, Layer 1) across 500+ distribution shapes before launch.

**The fix (v1.1).** Replace discrete bootstrap with a smoothed bootstrap via log-normal KDE. Log-transform observed costs, fit a Gaussian KDE with Silverman bandwidth, sample from the continuous density, exponentiate. This fills gaps between observations, prevents negative costs, and allows modest tail extrapolation. Approximately 15 lines of code with scipy.

**Silverman bandwidth explained.** KDE places a Gaussian "bump" at each observation and sums them into a smooth density. The bandwidth h controls bump width — too narrow reverts to discrete bootstrap, too wide smears away real structure. Silverman's rule (1986) sets h = 0.9 × min(σ̂, IQR/1.34) × n^(−1/5), where the min() provides robustness against heavy tails and bimodality. Applied in log-space, this naturally handles right-skew and prevents negative cost samples. At n=20 with Silverman bandwidth in log-space, the KDE typically allows extrapolation of 15–30% beyond the observed maximum — exactly the modest tail extension needed to compensate for bootstrap truncation.

**The fix (v2+).** Train a mixture density network on synthetic and real workflow cost distributions. Input: summary statistics of observed runs. Output: parameters of a predicted cost distribution. This enables principled extrapolation beyond the observed range.

**Effort:** 30 minutes for v1 inflation factor; half a day for v1.1 KDE. **Priority:** v1 inflation factor before backtesting. KDE after reviewing backtesting results.

### 1.3 — Step-Level Independence Assumption

**The problem.** When the Monte Carlo applies pattern-specific models (context growth, loop variance), it samples per-step costs independently, breaking the correlation structure between steps. This can produce impossible simulations — e.g., both the cheap path and expensive path executing in the same simulated run when they're mutually exclusive.

For workflows with conditional branching, independent step sampling overestimates the tail (creates impossible expensive combinations). For workflows with positively correlated steps (shared context), it underestimates the tail (misses the "everything is expensive" scenario).

**The fix.** Verify the base case (no detected patterns) samples whole runs with replacement. For the pattern-specific simulation paths, sample non-pattern steps from a whole observed run, then apply growth/loop models only to the flagged steps. This partially preserves inter-step correlation.

**Effort:** Half a day, depending on current implementation. **Priority:** Fix before backtesting.

### 1.4 — Context Growth Extrapolation

**The problem.** The context growth model (linear and logarithmic average) extrapolates beyond observed iteration counts. If profiling observes iterations 3–8 and production runs to iteration 15, the linear model extrapolates 87% beyond the data range. Real context growth is often stepwise (summarization steps compress context periodically), and context windows have hard limits that cap or crash the workflow.

**The fix.** Cap the growth model extrapolation at the observed maximum iteration count. When Monte Carlo samples an iteration count beyond the observed max, use the cost of the observed max rather than extrapolating. Add a warning when more than 10% of simulations sample beyond the observed range.

**Effort:** 1 hour. **Priority:** Fix before backtesting.

### 1.5 — Monte Carlo Seed Stability

**The problem.** At 10,000 simulations with 20 observed values, there is potential for noticeable variation across random seeds, particularly for heavy-tailed distributions.

**The fix.** Verify by running the same profile 5 times with different seeds. If p50 and p95 vary by more than 5%, increase to 50,000 simulations. The compute cost is negligible (resampling, not API calls).

**Effort:** 30 minutes to verify. **Priority:** Quick check before backtesting.

---

## Part 2: Pattern Detection Fixes

### 2.1 — CRITICAL: Context Growth Detection Overhaul

**Problem 1: False positives at low n.** The context growth detector requires r² > 0.7 with a minimum of 3 data points. With 3 data points drawn from pure noise (no real growth), r² follows a uniform distribution on [0, 1], giving a 30% false positive rate. This means nearly one in three steps with 3 iterations gets flagged as having context growth when it doesn't.

This directly affects recommendations (Sprint 4) — users receive incorrect "add context compaction" advice — and Monte Carlo mode selection — a false context growth flag forces Monte Carlo mode unnecessarily.

**Problem 2: Blind to non-linear growth.** The original detector uses Pearson r², which measures *linear* correlation. If context grows sub-linearly (logarithmic) or super-linearly (exponential), Pearson r² underestimates the relationship strength because the data curves away from the best-fit line. A workflow with perfect quadratic context growth might produce Pearson r² = 0.6 (below threshold) while Spearman ρ = 1.0 (perfect monotonic relationship). The detector misses it, and the engine uses linear mode on a super-linear growth pattern — underestimating costs at high iterations.

**The fix — dual correlation with p-value gating.** Run both Pearson and Spearman on each step:

1. Raise minimum data points from 3 to 5.
2. Compute Pearson r² and Spearman ρ². Compute p-values for both using their respective t-tests (Pearson: t = r × √(n−2) / √(1−r²); Spearman: same formula with ρ substituted).
3. Flag if **either** exceeds its threshold with p < 0.05:
   - Pearson r² > 0.7 → linear growth detected. Use linear+logarithmic model average in Monte Carlo.
   - Spearman ρ² > 0.7 but Pearson r² < 0.7 → non-linear monotonic growth detected. Use logarithmic model only (more conservative for sub-linear; fit a power-law model for super-linear).
4. The *gap* between Spearman ρ and Pearson r is diagnostic: large gap = highly non-linear growth. Report this in recommendations.

**Severity assignment:**
- Pearson r² > 0.85 OR Spearman ρ² > 0.85 (with p < 0.05) → DANGER
- Either r² or ρ² in (0.7, 0.85] (with p < 0.05) → WARNING

**Effort:** 2 hours (up from 1 hour due to dual correlation). **Priority:** Fix before backtesting.

### 2.2 — Token Variance Detector Sensitivity at Small n

**The problem.** The high token variance detector uses p95/p50 > 3.0. At n = 20, the p95 estimate is noisy (it's essentially the 19th ordered observation), producing a 5–10% false positive rate for moderately skewed distributions.

**The fix.** For n < 30, use p90/p50 instead of p95/p50, or raise the warning threshold from 3.0 to 4.0. The p90 is more stable (18th observation instead of 19th).

**Effort:** 15 minutes. **Priority:** Before backtesting.

### 2.3 — Loop Count Variance Detector

**Status:** The CV > 0.5 threshold is well-calibrated. False positive rate is low. No changes needed.

### 2.4 — NEW: Step Count Variance Detector

**The problem.** The existing detectors assume a fixed set of steps with variable behavior *within* each step. Real-world routing agents have a variable number of *active* steps per run — some runs execute 3 steps, others execute 7. This is the workflow-level analog of loop count variance and directly indicates conditional branching. None of the existing detectors catch this.

**Detection logic:**
1. For each profiling run, count the number of steps with non-zero cost (active steps).
2. Compute the coefficient of variation of active step counts across runs.
3. If CV > 0.3 → WARNING. If CV > 0.6 or max active steps > 2× min active steps → DANGER.

The lower CV threshold (0.3 vs. 0.5 for loop variance) reflects that step count variation has a larger impact on total cost than iteration count variation — an entire step being present or absent changes the cost structure more than one extra iteration of an existing step.

**Impact on projection:** When step count variance is detected, Monte Carlo should sample whole runs (preserving which steps are active) rather than sampling per-step costs independently. This is already the recommendation from §1.3 but the detector provides the signal.

**Effort:** 1–2 hours. **Priority:** Before backtesting (required for W13 routing workflow validation).

### 2.5 — NEW: Cost Bimodality Detector

**The problem.** A workflow with two distinct cost modes (happy path at $0.02, error/escalation path at $0.40) produces a bimodal cost distribution with nothing in between. The token variance detector catches the high p90/p50 ratio, but doesn't distinguish bimodal from unimodal-heavy-tailed. This distinction matters for reporting: "typical $600/month, worst case $12,000/month" is more informative than "p50 $600, p95 $12,000" when there's an empty gap between the modes.

**Detection logic:** Apply Hartigan's dip test to the per-run cost distribution. The dip test measures the maximum difference between the empirical distribution and the best-fitting unimodal distribution. If the dip statistic exceeds the critical value at p < 0.05, the distribution is significantly bimodal.

Alternatively, for a simpler implementation: fit a 2-component Gaussian mixture model (GMM) to the log-costs. If the BIC of the 2-component model is lower than the 1-component model by > 6 (strong evidence), flag bimodality. Report the two modes, their mixing proportions, and the per-mode cost ranges.

**Severity:** WARNING always (bimodality is informational, not dangerous — Monte Carlo handles it correctly as long as both modes are observed in the profiling data).

**Impact on reporting:** When bimodality is detected, report per-mode statistics instead of (or in addition to) global percentiles: "Mode A (70% of runs): $0.02/run. Mode B (30% of runs): $0.40/run." This is more actionable for the user than a single p50 that falls between the modes.

**Effort:** 2–3 hours (Hartigan's dip test is available in scipy; GMM in sklearn). **Priority:** Before backtesting (validates W13 routing workflow which should produce bimodal costs).

### 2.6 — Detector-to-Cost-Adjustment Mapping

Not every detector needs a cost adjustment model in Monte Carlo. Cost adjustments are needed only when the observed data can't represent the cost mechanism through simple whole-run resampling. Context growth needs a model because it extrapolates beyond observed iterations. Loop variance needs a model because cost is the product of two independent variables (iteration count × per-iteration cost) that must be decomposed. The remaining detectors flag patterns already faithfully represented in the observed data — whole-run resampling handles them.

| Detector | Cost adjustment in Monte Carlo? | What it does instead |
|----------|-------------------------------|---------------------|
| Context growth (linear, Pearson r² > 0.7) | **Yes** — linear + log average model | — |
| Context growth (non-linear, Spearman ρ² > 0.7 but Pearson r² < 0.7) | **Yes** — power-law model (see below) | — |
| Loop count variance (CV > 0.5) | **Yes** — sample iteration count × sample per-iteration costs | — |
| High token variance (p90/p50 > 3.0) | No | Triggers MC mode; reporting; confidence deduction |
| Step count variance (CV > 0.3) | No | Triggers MC mode; enforces whole-run sampling; reporting |
| Bimodality (dip test p < 0.05) | No | Per-mode reporting; recommendations |

**Non-linear context growth model (power-law):**

When Spearman ρ significantly exceeds Pearson r, the growth is non-linear monotonic. Estimate the exponent α via log-log regression: ln(context) = ln(base) + α × ln(k), where k is iteration index.

- Sub-linear (α < 1, growth decelerating): Use average of logarithmic and power-law model. Both are conservative.
- Super-linear (α > 1, growth accelerating): Use average of power-law and a capped version (cap at model's max_context_window or 3× observed max context, whichever is lower). Prevents runaway extrapolation while modeling acceleration.
- The iteration-count cap from §1.4 still applies independently.

Detect which case by comparing growth rates in the first vs. second half of the iteration range: first_half_slope > second_half_slope → sub-linear; first_half_slope < second_half_slope → super-linear.

**Loop variance cost adjustment — full procedure:**

For one flagged step within one Monte Carlo simulation:
1. From profiling data, collect the distribution of max iteration counts per run: {3, 4, 5, 4, 12, 3, 7, 4, 8, 5, ...}.
2. Sample one iteration count K from this distribution with replacement. Say K = 12.
3. From profiling data, collect all observed per-iteration costs (a flat pool across all runs and iterations): {$0.03, $0.04, $0.03, $0.05, ...}.
4. Sample K per-iteration costs from this pool with replacement.
5. Sum them: step_cost = Σ of K sampled costs.

Why decompose rather than sampling whole-step costs? Because a run with 3 iterations and a run with 12 iterations are both in the pool. Sampling a 3-iteration whole-step cost in a simulation modeling 12 iterations produces the wrong cost. Decomposition correctly models: more iterations → proportionally more cost.

**Interaction between loop variance and context growth:** If both patterns are detected on the same step, context growth takes precedence for the per-iteration cost model (each later iteration costs more), while the iteration count K is sampled from the loop variance distribution. The two compose naturally: sample K from loop distribution, then compute cost for iterations 1..K using the growth model.

---

## Part 3: Calibration and Metrics Fixes

### 3.1 — Revised Metric Thresholds

| Metric | Current | Proposed | Rationale |
|--------|---------|----------|-----------|
| p50 ratio | 0.5–2.0× | **0.7–2.0×** | Underestimates are more harmful than overestimates for a budgeting tool. A 2× underestimate (0.5×) is genuinely misleading. |
| p95 coverage | ≥80% | **≥85% simple / ≥75% complex** | Ground truth p95 has 8–15% uncertainty for complex workflows. 75% accounts for this noise. |
| Range ratio | <5× | **<3× simple / <8× complex** | Simple workflows with wide ranges indicate a broken projection. Complex workflows genuinely have wide cost distributions. |
| Top step correct | 20% co-dominant | **30% co-dominant** | At n = 20, steps within 30% are statistically indistinguishable. |
| Step ranking | Spearman r > 0.8 | **Drop for ≤3 steps; r > 0.7 for 4+ steps** | With 3 steps, only a perfect ranking (r = 1.0) passes at 0.8. One adjacent swap fails. This tests noise, not engine quality. |

### 3.2 — Revised Launch Gate

**Current:** All 12 workflows pass all metrics.

**Proposed:** All workflows pass p50 ratio AND top step correct. At least 80% of workflows pass each remaining metric. P50 accuracy and step identification are the core value propositions and serve as hard gates. The tail metrics (p95 coverage, range ratio, ranking correlation) are diagnostic — they inform engine quality but shouldn't individually block launch given the multiple-testing problem (60+ metric-workflow pairs).

### 3.3 — Bootstrap Ground Truth Confidence Intervals

After running the ground truth (300 runs for complex workflows, 200 for simple), compute bootstrap CIs on all calibration-relevant percentiles. Draw 1,000 bootstrap samples, compute p50 and p95 for each, and use the resulting 90% CI as the comparison target instead of point estimates. This makes calibration results statistically honest and prevents spurious failures from ground truth noise.

**Effort:** 2–3 hours of implementation. No API cost.

### 3.4 — Directional Bias Meta-Check

Track whether the projected p95 is above or below the ground truth p95 across all workflows. If 8+ out of 10 workflows show projected p95 below ground truth p95, the engine has a systematic conservative-bias deficit. Individual workflows can pass calibration while a global directional trend signals a real problem.

**Effort:** 1 hour.

### 3.5 — NEW: Effective Sample Size Gate (Anti-Gaming)

**The problem.** The confidence tier's sample size gate uses raw n. This is trivially gameable: generate 200 synthetic inputs with an LLM, profile them, and jump from −20 (n=20) to +10 (n≥200) — a 30-point swing. Worse, LLM-generated inputs cluster around medium difficulty and length, producing artificially low variance. The system *rewards* gaming by producing tighter confidence intervals on a projection that systematically underestimates the tail.

**The fix.** Replace raw n with effective sample size (n_eff) in the confidence tier scoring. Compute the entropy of the per-run cost distribution:

```
H = −Σ pᵢ × log(pᵢ)
```

where pᵢ is the proportion of runs in each cost bin (use √n bins via Sturges' rule). Maximum entropy is H_max = log(n_bins). The effective sample size is:

```
n_eff = n × (H / H_max)
```

If all 200 runs cost exactly the same, H = 0, n_eff = 0. If costs are uniformly spread across bins, H = H_max, n_eff = n. An LLM generating 200 "diverse" inputs that all exercise the same code path and produce near-identical costs gets n_eff ≈ 10–20 despite n = 200.

The confidence tier uses n_eff instead of raw n for the sample size deduction:
- n_eff < 10: −40
- n_eff 10–29: −20
- n_eff 30–99: −10
- n_eff ≥ 100: no deduction

**Effort:** 1 hour (15 lines of code). **Priority:** Before backtesting.

### 3.6 — NEW: Tiered Profiling Recommendations

**The problem.** n=20 was treated as a reasonable default for all use cases. After statistical analysis, n=20 is defensible for p50 estimation (±20–30%) but genuinely poor for tail estimation (36% chance of missing the true p95 entirely) and insufficient for regression detection (can only detect >40% changes).

**The fix.** Define three profiling tiers, communicated in the CLI and documentation:

| Tier | Sample size | p50 accuracy | Rare event coverage | Confidence cap | Use case |
|------|------------|-------------|-------------------|---------------|----------|
| **Quick scan** | n=20 | ±20–30% | Events >14% captured | MODERATE max | Order-of-magnitude screening. "Is this $500/month or $50,000/month?" |
| **Standard** (default) | n=50 | ±15% | Events >6% captured | HIGH eligible | Pre-deployment budgeting. The recommended default. |
| **Budget grade** | n=100+ | ±10% | Events >3% captured | HIGH eligible | CI regression detection, optimization validation, formal budgeting. |

At n < 20 (quick scan threshold), suppress p95 entirely. Show only p50 with a 0.5×–3× range qualifier.

At n < 50, label projections as "estimate" regardless of other confidence factors.

The engine works identically at all sample sizes — this is a presentation and documentation change, not an engine change.

**Effort:** 2 hours (CLI flag + output formatting + docs). **Priority:** Before launch (can be after backtesting).

---

## Part 4: Data Collection and Validation Fixes

### 4.1 — CRITICAL: Mock Response Unit Tests for All Collectors

**The problem.** Each provider returns usage metadata in different fields. If a collector doesn't extract reasoning tokens (OpenAI `completion_tokens_details.reasoning_tokens`), thinking tokens (Anthropic extended thinking), or cache tokens (DeepSeek `prompt_cache_hit_tokens`), the profiled cost is silently wrong. The projection "succeeds" with incorrect data.

**The fix.** Write unit tests with mocked API responses for every provider and response variant:

- OpenAI standard (GPT-5.4)
- OpenAI reasoning (o-series with `reasoning_tokens`)
- Anthropic standard (Sonnet 4.6)
- Anthropic extended thinking
- Anthropic cached (with `cache_creation_input_tokens` and `cache_read_input_tokens`)
- DeepSeek with cache hit/miss breakdown
- Qwen DashScope-native format
- Qwen OpenAI-compatible format
- Gemini with `thoughtsTokenCount`

Assert extracted token counts match expected values for each variant.

**Effort:** Half a day. Zero API cost. **Priority:** Non-negotiable before backtesting.

### 4.2 — CRITICAL: Pricing Table Validation

**The problem.** If the pricing table has a decimal point error on any model, both the projection and the ground truth use the same wrong price. Backtesting passes. Users get confident, wrong projections. This is undetectable from backtesting alone.

**The fix.** For each model in the test suite, submit one known request. Record the token count from the API response. Compute cost using the pricing table. Compare against the provider's billing dashboard charge. One request per model, manual cross-check.

**Effort:** 2–3 hours. ~$5 API cost. **Priority:** Before backtesting.

### 4.3 — DeepSeek Cache-Busting During Profiling

**The problem.** DeepSeek cache-hit input costs $0.0028/MTok vs. $0.14/MTok cache-miss — a 50× difference. If 20 rapid profiling runs all hit the prompt cache, the profiled cost is up to 50× lower than a cold-start production request.

**The fix.** Parse DeepSeek's `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens` from the response. Price them at their respective rates. Default to cache-busting during profiling (append a unique suffix to break the cache). Provide a `--allow-cache` flag for users who know their production workload has warm caches.

**Effort:** Half a day. **Priority:** Before running W12 backtesting.

### 4.4 — Zero-Token Step Validation

After profiling, check that every step has non-zero token counts across all runs. If any step records zero tokens in every run, flag it as a data collection error rather than silently including a free step in the projection. This catches streaming-usage-dropped scenarios and any collector bug.

**Effort:** 30 minutes.

### 4.5 — Unrecognized Model Handling

When a StepRecord contains a model name not in the pricing table, raise a visible error. Do not silently fall back to a default price or zero. Provide an `--add-model` flag for user-specified pricing.

**Effort:** 1 hour.

---

## Part 5: Workflow Coverage Revisions

### 5.1 — Revised Test Suite (10 Workflows)

**Cut:** W3 (simple code review), W6 (complex extraction), W7 (simple research). These are redundant with other workflows in the same complexity tier. See Area 11 for detailed justification.

**Add:** W13 (conditional routing agent). A classifier routes to one of three paths with different models and costs (70% cheap / 20% moderate / 10% expensive). This tests zero-inflated step distributions, bimodal run costs, and Monte Carlo behavior on rare expensive paths.

| # | Workflow | Complexity | Ground truth runs | Est. cost |
|---|----------|-----------|-------------------|-----------|
| W1 | Support agent | Simple | 200 | $4 |
| W2 | Support agent | Complex | 300 | $109 |
| W4 | Code review | Complex | 300 | $218 |
| W5 | Data extraction | Simple | 200 | $5 |
| W8 | Research agent | Complex | 300 | $330 |
| W9 | Sales/outreach (OpenAI) | Simple | 200 | $4 |
| W10 | Sales/outreach (mixed) | Complex | 300 | $128 |
| W11 | Support (Qwen) | Simple | 200 | $1 |
| W12 | Extraction (DeepSeek) | Simple | 200 | $1 |
| W13 | Routing agent (NEW) | Conditional | 300 | $22 |
| — | Pricing validation | — | ~15 requests | $5 |
| | | | **Total** | **~$827** |

Run 20-sample projections against the ground truth for each workflow. Drop the 100-run intermediate tier — it doesn't change any launch decision. Subsample from ground truth data (draw n = 50, 100 from the 300 runs) to build the accuracy-vs-sample-size curve without additional API spend.

### 5.2 — Skewed Input Distribution Variant

For W2 and W8, create a second input set with 80/15/5 distribution (easy/medium/hard) alongside the primary 30/30/20/10/10 set. Run the 20-vs-ground-truth calibration on both. This tests whether Monte Carlo handles production-realistic skewed distributions. If calibration passes on uniform but fails on skewed, the engine has a real problem that wouldn't surface in the primary backtesting.

**Cost:** ~$100 additional if budget allows. Zero additional implementation cost — just different input allocation.

### 5.3 — Input Set Design Principles

- Reduce adversarial inputs from 10% to 5%. Shift the freed 5% to easy cases (35% easy total). Better matches production while still exercising error paths.
- Within each difficulty bucket, vary input length independently of difficulty. Include some easy-but-long and hard-but-short inputs. This prevents confounding length with difficulty in pattern detection.
- For each workflow, ensure at least 2 inputs that trigger every declared conditional branch or error-handling path. This reduces the probability of zero-execution steps in the profiling data.

---

## Part 5B: Three-Layer Validation Strategy

The backtesting validates the projection engine, but relying solely on 10 real workflows (limited by API budget) leaves the statistical validation thin. A three-layer approach provides comprehensive coverage at minimal additional cost.

### Layer 1: Synthetic Distribution Testing (before spending API dollars)

Generate 500+ synthetic "workflows" with known statistical properties and feed them directly into the projection engine as if they were observed runs. This validates the Monte Carlo, percentile estimation, tail inflation, and pattern detection across a much wider range of distribution shapes than 10 real workflows can provide.

**Distribution shapes to generate:**
- Log-normal: varying σ from 0.2 (mild skew) to 1.5 (extreme skew)
- Bimodal: varying mode separation (2× to 20×) and mixing ratio (90/10 to 50/50)
- Pareto-tailed: varying shape parameter α from 1.5 (very heavy) to 5.0 (moderate)
- Zero-inflated: conditional branching simulation, varying trigger probability from 5% to 40%
- Uniform (control): no variance, no patterns

For each synthetic workflow, draw samples at n = 20, 50, 100, 300. Run the projection engine. Measure calibration metrics against the known true distribution parameters.

**What this tests:** Engine math correctness (CLT fix, tail inflation, Monte Carlo sampling), pattern detector behavior across distribution shapes, confidence tier calibration, and the effective sample size gate. It does NOT test data collection, pricing, or end-to-end pipeline.

**Effort:** 1–2 days to build the synthetic framework + analysis. Zero API cost. **Priority:** Run before spending API dollars on real profiling. If the engine fails on known distributions, fix it before running $850 of real workflows.

### Layer 2: SWE-bench Trajectory Analysis (free, real-world distribution shapes)

The SWE-bench experiments repository open-sources per-instance execution trajectories with token usage data from coding agent runs. Download per-instance trajectories, extract token usage, and group by repository (e.g., all Django tasks, all Flask tasks). Within each repo group, tasks share a similar codebase and agent workflow structure — the per-task cost variation approximates what you'd see from running the same workflow with diverse inputs.

**What you get:** 50–100 "runs" per repo group from SWE-bench Verified, with real heavy-tailed cost distributions driven by task complexity. These are real-world distribution shapes, not synthetic ones.

**How to use it:** Feed the per-repo cost distributions into the synthetic testing framework as additional test cases. The engine doesn't know these came from SWE-bench vs. synthetic generation. Measure the same calibration metrics. This gives you real-world distribution shapes without API cost.

**Limitation:** SWE-bench tasks are not repeated runs of the same workflow — they're different tasks run once each. The cost variation is driven by task diversity, not input diversity within a single task. The distribution shape is realistic, but the structure (per-step breakdown) may not perfectly match an AgentCost profiling run.

**Effort:** Half a day to download, parse, and integrate. Zero API cost.

### Layer 3: Real Workflow Profiling (validates end-to-end pipeline)

The 10 real workflows (§5.1) at the ~$850 budget. This is the only layer that validates the complete pipeline: data collection → pricing → pattern detection → projection → presentation. Layers 1 and 2 validate the engine math on injected data; Layer 3 validates the entire system including collector correctness, API response parsing, and pricing table accuracy.

**Effort:** ~$850 API cost + profiling time.

### How the Layers Complement Each Other

| Aspect validated | Layer 1 (synthetic) | Layer 2 (SWE-bench) | Layer 3 (real profiling) |
|-----------------|--------------------|--------------------|------------------------|
| Monte Carlo math | ✓ (primary) | ✓ | ✓ |
| Tail estimation | ✓ (primary) | ✓ | ✓ |
| Pattern detection | ✓ | Partial | ✓ |
| Distribution shape coverage | ✓ (500+ shapes) | ✓ (real shapes) | Limited (10 workflows) |
| Data collection | — | — | ✓ (primary) |
| Pricing accuracy | — | — | ✓ (primary) |
| Cross-provider | — | — | ✓ (primary) |
| End-to-end pipeline | — | — | ✓ (primary) |

The synthetic layer is the highest-value addition because it converts the validation from "we tested on 10 distributions and it worked" to "we tested on 500+ distributions spanning every shape class and calibrated the tail inflation factor empirically." The latter is publishable-quality validation. The former is a spot check.

---

## Part 6: Visibility and Warning Systems

These features turn silent failures into visible warnings. They are inexpensive to implement and address the largest class of real-world projection errors — input quality, environment mismatch, and stale data.

### 6.1 — Input Distribution Statistics (v1)

Report alongside every projection: input token count p50, p95, max, and CV across profiling runs. This makes the "garbage in" assumption visible. Users who profile with 200-token inputs and deploy with 2,000-token production traffic can see the mismatch.

**Effort:** 1–2 hours.

### 6.2 — Uniform Input Heuristic Flags (v1)

Two flags:

1. If a workflow has loops or conditional branches but all profiling runs show identical iteration counts, warn: "All runs executed the same iteration count. If production inputs trigger variable loop counts, re-profile with more diverse inputs."
2. If input token CV across all runs is < 0.1, warn: "Input length is very uniform. Production traffic with variable input lengths may produce different cost distributions."

These catch the accidental version of cherry-picked inputs, which is far more common than the deliberate version.

**Effort:** 1–2 hours.

### 6.3 — Zero-Execution Step Detection (v1)

If a workflow step appears in the workflow definition but was never triggered across any profiling run, flag: "Step X was never triggered during profiling. If this step runs in production, the projection underestimates costs."

**Effort:** 1–2 hours.

### 6.4 — Stale Profile Warnings (v1)

Record profiling timestamp, step name list hash, and system prompt hash per step. On loading a profile for display, comparison, or CI:

- Age > 30 days: "Profile may be stale. Consider re-profiling."
- Age > 90 days: "Profile is stale. Re-profile recommended." Degrade confidence tier by one level.
- Step list hash mismatch: "Workflow structure has changed since profiling."
- Prompt hash mismatch: "System prompt has changed since profiling."

**Effort:** 2–3 hours total.

### 6.5 — Pricing Staleness Warning (v1)

Timestamp the pricing table. If older than 30 days when generating a projection, warn: "Pricing data may be outdated. Run `agentcost update-pricing` to refresh."

**Effort:** 1 hour.

### 6.6 — Sample Coverage Statement (v1)

Report with every projection: "With N profiling runs, events occurring less than ~X% of the time may not be represented." Where X = 1 − 0.05^(1/N) × 100. At n = 20, X ≈ 14%. At n = 100, X ≈ 3%.

**Effort:** 30 minutes.

### 6.7 — Small-Sample p95 Suppression (v1)

At n < 10, suppress the p95 and report only p50 with a range multiplier: "Estimated $X/month. At this sample size, actual costs could be 0.5×–3× this estimate." Showing p50 and p95 at n = 5 implies distributional precision that doesn't exist.

**Effort:** 30 minutes.

---

## Part 7: Downstream System Recommendations

### 7.1 — Baseline Diffs (Sprint 3)

Add a Mann-Whitney U significance test to the diff output. Report the p-value and label changes as "significant" (p < 0.05), "possibly significant" (p < 0.1), or "not statistically significant" (p ≥ 0.1). Include a 95% bootstrap CI on the cost change percentage.

At n = 20 with moderate variance, changes below ~15% are indistinguishable from noise. Document this explicitly: "To reliably detect 10–20% cost changes, profile with n ≥ 100."

Warn when the baseline and new profile have sample sizes differing by more than 2×.

**Effort:** 2–3 hours.

### 7.2 — GitHub Action (Sprint 5)

Default to **report mode**, not blocking mode. The Action comments on the PR with cost diff, confidence interval, and significance flag. It does not block. After 2–4 weeks of report mode, teams can switch to blocking mode with an empirically validated threshold.

Implement adaptive thresholds: `effective_threshold = max(user_threshold, 2.5 × SE_of_diff)`. If the user sets 20% but the minimum detectable change at the profiled sample size is 35%, the effective threshold is 35%, with a message explaining why.

Require contemporaneous profiling: both baseline and PR branches profiled in the same CI run, same model version, same environment. Stale baselines cause model-drift false positives.

**Effort:** 2–3 hours for adaptive thresholds. Architectural decision on contemporaneous profiling.

### 7.3 — Recommendations (Sprint 4)

Fix the context growth false positive rate (§2.1) before recommendations ship. With dual Pearson+Spearman detection and p-value gating, recommendations become trustworthy. The new detectors (§2.4 step count variance, §2.5 bimodality) enable richer recommendations:

- Step count variance detected → "This workflow has conditional branches. Consider profiling with more diverse inputs to ensure all paths are exercised."
- Bimodality detected → "Cost has two distinct modes: [Mode A stats] and [Mode B stats]. Investigate what triggers the expensive mode."
- Non-linear context growth (Spearman ≫ Pearson) → "Context growth is non-linear. Consider adding a summarization step to compress context."

Add a caveat to all pattern-based recommendations: "Detected from N profiling runs (n_eff = X). Verify with a larger sample or production data before making architectural changes." Recommendations should prompt investigation, not dictate action.

---

## Part 8: Known Limitations to Document

These are real gaps that affect projection accuracy in specific scenarios. They are not engine bugs — they are fundamental to the "profile a sample, project to production" methodology. Document them prominently, not buried in footnotes.

1. **Session context accumulation.** Single-turn profiling underestimates multi-turn conversational agent costs by 2–3× for an 8-turn average session. If the workflow maintains conversation history, recommend users profile with mid-session context or use the `--session-depth` correction (v2).

2. **Input distribution mismatch.** Projections are conditioned on the input distribution used during profiling. If production inputs are longer, more complex, or trigger different code paths, the projection does not account for this. The input distribution statistics (§6.1) make this assumption visible.

3. **Tool response size mismatch.** If profiling uses test stubs or toy databases, tool responses may be 10–100× smaller than production. Every step downstream of a tool call will have understated context size. Recommend profiling against production-representative tools.

4. **Provider-side caching.** Rapid profiling may benefit from prompt cache hits that low-traffic production workloads don't get. For DeepSeek workflows, this can cause up to 50× underestimation of input token costs. Use `--cache-mode cold` for conservative estimates.

5. **Tiered pricing boundaries.** v1 uses standard-tier pricing for all providers. Qwen workflows exceeding 200K input tokens per request may be underpriced. Warn when observed context sizes exceed 150K tokens.

6. **Model drift.** Model providers update models within version lines. Token counts and costs may shift 5–25% between updates. Profiles older than 30 days may not reflect current model behavior.

7. **Batch vs. real-time pricing.** v1 uses real-time API pricing. Users deploying via batch APIs (OpenAI Batch, Anthropic Message Batches) will see ~50% lower actual costs than projected.

---

## Part 9: Prioritized Action List

### Before Running Backtesting (7–9 days)

| # | Action | Section | Effort | Why it can't wait |
|---|--------|---------|--------|-------------------|
| 1 | Fix CLT variance inflation in Monte Carlo | §1.1 | Half day | Every high-volume projection is wrong without this |
| 2 | Write mock response unit tests for all collectors | §4.1 | Half day | Profiling on bad token data wastes entire budget |
| 3 | Validate pricing table against billing dashboards | §4.2 | 2–3 hrs | Ground truth is wrong if pricing table is wrong |
| 4 | Build W13 routing workflow | §5.1 | 1–2 days | Tests the most important structural gap |
| 5 | Implement dual Pearson+Spearman context growth detection with p-value gating | §2.1 | 2 hrs | 30% false positive rate; misses non-linear growth |
| 6 | Implement step count variance detector | §2.4 | 1–2 hrs | Required for W13 routing workflow validation |
| 7 | Implement cost bimodality detector | §2.5 | 2–3 hrs | Required for W13; improves reporting for conditional workflows |
| 8 | Implement effective sample size gate (n_eff) | §3.5 | 1 hr | Prevents gaming via uniform/LLM-generated inputs |
| 9 | Implement revised calibration metric thresholds | §3.1–3.2 | 2–3 hrs | Running backtesting with broken metrics wastes time |
| 10 | **Build synthetic distribution testing framework** | §5B.L1 | 1–2 days | Validates engine math on 500+ shapes before spending $850 |
| 11 | Prepare skewed-distribution input sets for W2, W8 | §5.2 | 2–3 hrs | Tests the distribution shape Monte Carlo depends on |
| 12 | Implement DeepSeek cache-busting | §4.3 | Half day | W12 results are meaningless without this |
| 13 | Add conservative tail inflation for n < 30 | §1.2 | 30 min | Reduces systematic p95 underestimation |
| 14 | Cap context growth extrapolation at observed max | §1.4 | 1 hour | Prevents unbounded extrapolation errors |
| 15 | Download and parse SWE-bench trajectory data | §5B.L2 | Half day | Free real-world distribution shapes for engine validation |

### During Backtesting

| # | Action | Section | Effort |
|---|--------|---------|--------|
| 16 | Run synthetic distribution calibration (500+ shapes × 4 sample sizes) | §5B.L1 | Automated |
| 17 | Run SWE-bench trajectory calibration | §5B.L2 | Automated |
| 18 | Bootstrap ground truth percentiles; use CIs as targets | §3.3 | 2–3 hrs |
| 19 | Run skewed-distribution variant on W2 or W8 | §5.2 | ~$100 |
| 20 | Check directional bias on p95 across all workflows | §3.4 | 1 hour |
| 21 | Profile W12 in cache-cold mode | §4.3 | Minimal |
| 22 | Run 10 W8 runs first to calibrate actual cost/run before committing to 300 | §5.1 | ~$10 |
| 23 | Calibrate tail inflation factor empirically from synthetic + real results | §1.2 | Half day |

### After Backtesting, Before Launch

| # | Action | Section | Effort |
|---|--------|---------|--------|
| 24 | Implement tiered profiling recommendations (n=20/50/100+) in CLI + docs | §3.6 | 2 hrs |
| 25 | Add zero-execution step detection | §6.3 | 1–2 hrs |
| 26 | Add input token distribution stats to output | §6.1 | 1–2 hrs |
| 27 | Add uniform-input heuristic flags | §6.2 | 1–2 hrs |
| 28 | Add stale profile age + hash warnings | §6.4 | 2–3 hrs |
| 29 | Add pricing staleness warning | §6.5 | 1 hour |
| 30 | Add sample coverage statement | §6.6 | 30 min |
| 31 | Suppress p95 at n < 20 (aligned with tiered profiling) | §6.7 | 30 min |
| 32 | Add significance testing to baseline diff | §7.1 | 1 hour |
| 33 | Add zero-token step validation | §4.4 | 30 min |
| 34 | Add unrecognized model error handling | §4.5 | 1 hour |
| 35 | Document all known limitations prominently | §8 | 2–3 hrs |
| 36 | Use p90/p50 for token variance detector at n < 30 | §2.2 | 15 min |

### Deferred to v2+

| Action | Section |
|--------|---------|
| Log-normal KDE for Monte Carlo smoothed bootstrap | §1.2 |
| Mixture density network for cost distribution modeling | §1.2 |
| `--session-depth` analytical correction for multi-turn | §8.1 |
| Tool response multiplier | §8.3 |
| Adaptive GitHub Action thresholds | §7.2 |
| `--cache-mode warm/cold/mixed` flag (generalized) | §8.4 |
| Input difficulty prediction for distribution shift detection | ML |
| Learned confidence tier calibration via logistic regression | ML |
| Dynamic pricing table fetching from provider APIs | §6.5 |
| Batch API pricing tier flag | §8.7 |
| Parallel fan-out collector validation | §5.1 |
| Time-series cost trend forecasting | ML |
| Anomaly detection on profiling runs (isolation forest / autoencoder) | ML |
| Conditional per-step distribution modeling | ML |
| Inter-step cost ratio shift detector | Future |
| Output token explosion detector (max_tokens cap detection) | Future |
| Cost-per-token vs. input-size correlation detector (super-linear pricing) | Future |

---

## Part 10: Budget Summary

| Item | Cost |
|------|------|
| W1 (simple support, 220 runs) | $4 |
| W2 (complex support, 320 runs) | $109 |
| W4 (complex code review, 320 runs) | $218 |
| W5 (simple extraction, 220 runs) | $5 |
| W8 (complex research, 320 runs) | $330 |
| W9 (simple sales OpenAI, 220 runs) | $4 |
| W10 (complex mixed, 320 runs) | $128 |
| W11 (simple Qwen, 220 runs) | $1 |
| W12 (simple DeepSeek, 220 runs) | $1 |
| W13 (routing agent, 320 runs) | $22 |
| Pricing validation (~15 requests) | $5 |
| **Subtotal** | **$827** |
| Contingency / skewed variant | ~$120 |
| **Total budget** | **~$950** |

Synthetic distribution testing (Layer 1) and SWE-bench trajectory analysis (Layer 2) add zero API cost. They are engineering time investments (2–3 days combined) that dramatically increase validation coverage.

Run 10 exploratory runs of W8 first (~$10) to validate cost-per-run estimates. If W8 comes in at the high end of its range, reduce W8 ground truth from 300 to 200 and reallocate.

---

## Conclusion

The projection engine works. The Monte Carlo approach with pattern detection is the right architecture for this problem. The post-review design adds meaningful robustness: dual Pearson+Spearman context growth detection catches non-linear patterns the original design missed, step count variance and bimodality detectors handle the conditional routing workflows that are increasingly common in production, and the effective sample size gate prevents the most obvious gaming vector.

The three-layer validation strategy is the most important structural improvement. Synthetic distribution testing (500+ distribution shapes, zero API cost) converts the validation from a 10-workflow spot check into a statistically comprehensive calibration exercise. SWE-bench trajectory data provides real-world distribution shapes for free. Together, these layers mean the $850 of real workflow profiling is confirming results you've already validated synthetically, not discovering problems for the first time.

The tiered profiling recommendation (n=20 quick scan / n=50 standard / n=100+ budget grade) is an honest acknowledgment that sample size matters more than engine sophistication for projection quality. The engine is correct at any n. The question is whether the user's sample is large enough to characterize their workflow's cost distribution. Setting the default to n=50 and communicating the tradeoffs explicitly prevents the "n=20 projection says $3,400 and I budgeted $3,400" failure mode.

The larger insight from this review remains: most real-world projection failures will come from outside the engine — input quality, environment mismatch, stale data, and collector bugs. The visibility features (§6) and the effective sample size gate (§3.5) are disproportionately valuable relative to their implementation cost. A $3,000/month projection that says "based on 45 effective samples averaging 200 input tokens, profiled 12 days ago, with 6%+ rare-event coverage" lets users calibrate trust. A bare "$3,000/month (moderate confidence)" invites blind trust or blind distrust, neither of which serves them.

Build the synthetic testing framework first (it catches engine bugs for free), then run the real profiling suite, then ship.
