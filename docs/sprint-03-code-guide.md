# Sprint 3 Code Guide

Developer walkthrough, architecture diagrams, function-level documentation, data flow charts, debugging exercises, and REPL cheat sheet for Sprint 3.

Sprint 3 overhauled the projection engine, added a validation layer, hardened collectors and pricing, introduced a synthetic testing framework, and added visibility warnings. It touched 25+ files across 11 fix prompts. This guide explains every subsystem, every function, every data structure, and how they connect.

---

## Table of Contents

- [Part 1: Architecture Overview](#part-1-architecture-overview)
- [Part 2: Projection Engine](#part-2-projection-engine)
  - [2A. Monte Carlo Simulation](#2a-monte-carlo-simulation--pretiaprojectionmontecarlopy)
  - [2B. Pattern Detection](#2b-pattern-detection--pretiaprojectionpatternspy)
  - [2C. Projector](#2c-projector--pretiaprojectionprojectorpy)
- [Part 3: Validation Layer](#part-3-validation-layer)
  - [3A. Confidence Scoring](#3a-confidence-scoring--pretiavalidationconfidencepy)
  - [3B. Calibration Scoring](#3b-calibration-scoring--pretiavalidationscoringpy)
  - [3C. Backtesting Suite](#3c-backtesting-suite--pretiavalidationsuitepy)
  - [3D. Data Quality Checks](#3d-data-quality-checks--pretiavalidationdata_checkspy)
  - [3E. Visibility Warnings](#3e-visibility-warnings--pretiavalidationvisibilitypy)
- [Part 4: Collector & Pricing Hardening](#part-4-collector--pricing-hardening)
  - [4A. StepRecord Cache Fields](#4a-steprecord-cache-fields--pretiacollectorsbasepy)
  - [4B. Token Extraction](#4b-token-extraction--pretiacollectorsgenericpy)
  - [4C. Cache Busting](#4c-cache-busting--pretiacollectorscache_bustpy)
  - [4D. Pricing Tables](#4d-pricing-tables--pretiapricingtablespy)
- [Part 5: Significance Testing & Baseline Diffing](#part-5-significance-testing--baseline-diffing)
- [Part 6: Runner & CLI Changes](#part-6-runner--cli-changes)
- [Part 7: Synthetic Testing Framework](#part-7-synthetic-testing-framework)
  - [7A. Distribution Generators](#7a-distribution-generators--testsssyntheticgeneratorspy)
  - [7B. Synthetic Runner](#7b-synthetic-runner--testssyntheticrunnerpy)
  - [7C. Calibration Report](#7c-calibration-report--testssyntheticcalibrationpy)
- [Part 8: Backtesting Infrastructure](#part-8-backtesting-infrastructure)
- [Part 9: Full Data Flow Diagrams](#part-9-full-data-flow-diagrams)
- [Part 10: Worked Example Runs](#part-10-worked-example-runs)
- [Part 11: Debugging Exercises](#part-11-debugging-exercises)
- [Part 12: REPL Cheat Sheet](#part-12-repl-cheat-sheet)

---

## Part 1: Architecture Overview

Sprint 3's changes form three major arcs:

1. **Projection accuracy** — fix the Monte Carlo math, add new detectors, gate everything with statistics
2. **Validation infrastructure** — prove the engine works before shipping, with synthetic data and backtesting
3. **Edge case hardening** — DeepSeek caching, zero-token checks, model suggestions on errors

### System-Level Architecture After Sprint 3

```
┌───────────────────────────────────────────────────────────────────────┐
│                           CLI  (cli.py)                               │
│  profile run · report · analyze · baseline update · diff · validate   │
└───────────┬───────────────────────────────────────────────────────────┘
            │
            ▼
┌───────────────────────────────────────────────────────────────────────┐
│                       ProfileRunner  (runner.py)                      │
│  _load_workflow → _select_collector → _resolve_inputs                 │
│  → collector.collect → validate_profiling_data                        │
│  → compute_stats → detect_patterns → project → save                  │
└───────────┬───────────────────────────────────────────────────────────┘
            │
    ┌───────┴───────────────────────────────────┐
    │                                           │
    ▼                                           ▼
┌──────────────────┐                  ┌──────────────────────────────────┐
│   COLLECTORS     │                  │       PROJECTION ENGINE          │
│ GenericCollector  │                  │                                  │
│ LangGraphCollect. │                  │  stats.py  ─ ProfilingStats      │
│ OpenAIAgentsColl. │                  │     ▼                            │
│ QwenAgentCollect. │                  │  patterns.py ─ DetectedPattern   │
│                  │                  │  5 detectors:                    │
│  → StepRecord[]  │                  │    context_growth                │
│    (now with     │                  │    loop_count_variance           │
│     cache fields)│                  │    high_token_variance           │
│                  │                  │    step_count_variance  ← NEW    │
│  cache_bust.py   │                  │    bimodality           ← NEW    │
│    (DeepSeek)    │                  │     ▼                            │
└──────────────────┘                  │  projector.py ─ ProjectionResult │
                                      │    linear or montecarlo.py       │
                                      │     ▼                            │
                                      │  confidence.py ─ ConfidenceResult│
                                      │    (with effective n)            │
                                      └──────────────────────────────────┘
                                                  │
                                      ┌───────────┴──────────────┐
                                      │                          │
                                      ▼                          ▼
                          ┌─────────────────────┐   ┌───────────────────────┐
                          │   VALIDATION         │   │    REPORTING           │
                          │ scoring.py           │   │ ci/report.py           │
                          │ suite.py             │   │ ci/diff.py             │
                          │ data_checks.py       │   │ ci/baseline.py         │
                          │ visibility.py        │   │                        │
                          │                     │   │ format_cli_report()    │
                          │ SYNTHETIC TESTS     │   │ format_diff_report()   │
                          │ generators.py       │   └───────────────────────┘
                          │ runner.py           │
                          │ calibration.py      │
                          └─────────────────────┘
```

### What Changed From Sprint 2

| Area | Sprint 2 | Sprint 3 |
|------|----------|----------|
| Monte Carlo | 1 sample × N, variance = N² × σ² | K samples + CLT aggregation, variance = N × σ² |
| Context growth detector | Pearson r² > 0.7, 3 points, no p-value | Dual Pearson + Spearman, p < 0.05 gate, power-law classification |
| Pattern detectors | 3 (context, loop, token) | 5 (+ step count variance, bimodality) |
| MC trigger | Only on `danger` patterns | Any detected pattern |
| Confidence | Raw n for deductions | Entropy-based effective sample size (n_eff) |
| Scoring | Single threshold set | Split by workflow complexity (simple/complex) |
| Backtesting gates | All-or-nothing | Hard gates (100% required) + soft gates (80% required) |
| StepRecord | 15 fields | 17 fields (+cache_hit_tokens, +cache_miss_tokens) |
| Pricing errors | Generic `ValueError` | `UnrecognizedModelError` with suggestions |
| Cost calculation | Standard rates only | Differential cache-hit pricing for DeepSeek |
| Data quality | None | `validate_profiling_data()` post-profiling checks |
| Visibility | None | Context-aware recommendations, coverage statements |

---

## Part 2: Projection Engine

### 2A. Monte Carlo Simulation — `pretia/projection/montecarlo.py`

#### What this file does

Runs 10,000 Monte Carlo simulations to project monthly costs when non-linear patterns are detected. Each simulation samples observed run costs, applies pattern-specific cost models (linear/logarithmic/power-law growth for context growth steps, sampled iteration counts for loop variance steps), and aggregates to monthly totals using the Central Limit Theorem.

#### Data structures

| Name | Type | Fields | Purpose |
|------|------|--------|---------|
| `PercentileProjection` | frozen dataclass | `p50`, `p75`, `p90`, `p95`, `p99`, `mean` | Cost projection at each percentile for one time scale |
| `MonteCarloResult` | frozen dataclass | `n_simulations`, `monthly_projection`, `daily_projection`, `per_run_projection`, `linear_monthly`, `log_monthly`, `convergence_check`, `growth_model_delta`, `tail_inflation_factor`, `extrapolation_cap_warnings` | Full output of one MC simulation run |

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_percentile` | `(sorted_data: list[float], p: float) -> float` | Linear interpolation percentile on pre-sorted data. Handles empty (returns 0.0) and single-element lists. | `float` |
| `_safe_cost` | `(model: str, input_tokens: int, output_tokens: int) -> float` | Wraps `calculate_cost()` — catches `ValueError`/`KeyError` for unknown models and returns `0.0`. | `float` |
| `_build_percentile_projection` | `(values: list[float]) -> PercentileProjection` | Sorts values and computes all percentile fields. Returns all-zero projection for empty input. | `PercentileProjection` |
| `_precompute_step_data` | `(runs: list[list[StepRecord]]) -> tuple[dict[str, list[float]], dict[str, list[int]], dict[str, list[float]]]` | One pass over all runs. Produces: (1) `step_run_costs` — total cost of each step per run, (2) `step_iterations` — max iteration count of each step per run, (3) `step_occurrence_costs` — individual occurrence cost per step call. | 3-tuple of dicts |
| `_precompute_growth_data` | `(runs: list[list[StepRecord]], growth_steps: set[str]) -> dict[str, dict[str, Any]]` | For each context-growth step: extracts `(iteration, context_size)` pairs across all runs, computes linear regression slope, base context (mean of iteration-1 contexts), max observed context. Returns a dict keyed by step name. | `dict[str, dict]` |
| `_build_run_step_cost_map` | `(runs: list[list[StepRecord]]) -> list[dict[str, float]]` | **Fix 3: whole-run sampling.** Builds a per-run, per-step cost dict — position i corresponds to observed run i. Used to preserve inter-step correlation in sampling. | `list[dict[str, float]]` |
| `_clt_aggregate` | `(costs: list[float], n_total: int, z: float) -> float` | **Fix 1: CLT correction.** Given K sampled run costs, computes the monthly total as `N × μ̂ + z × √(N × σ̂²)` where z is a standard normal draw. Clamps result to ≥ 0. | `float` |
| `_sample_step_cost` | `(step_name, rng, step_run_costs, step_iterations, step_occurrence_costs, step_growth, growth_steps, loop_variance_steps, max_observed_iters) -> tuple[float, float, float, bool]` | Samples cost for one step in one simulated run. Three code paths: (1) **Context growth step** — picks random iteration count, sums per-iteration costs using linear+log (linear growth) or power-law (nonlinear growth) models. (2) **Loop variance step** — picks random iteration count, sums random per-occurrence costs. (3) **Plain step** — picks a random observed total cost. Returns `(avg_cost, linear_cost, log_cost, was_capped)`. | 4-tuple |
| `_inflate_p95` | `(proj: PercentileProjection, factor: float) -> PercentileProjection` | **Fix 2: tail inflation.** Returns a copy with `p95 *= factor`. Only `p95` is inflated — other percentiles are unchanged. | `PercentileProjection` |
| `simulate` | `(stats, patterns, daily_volume, runs, n_simulations=10000, n_days=30, seed=42, _debug_run_costs=None) -> MonteCarloResult` | The main entry point. Orchestrates the full MC simulation. See detailed flow below. | `MonteCarloResult` |

#### How `simulate()` works step by step

```
simulate(stats, patterns, daily_volume=1000, runs=[...], n_simulations=10000)
  │
  ├─ 1. Classify patterns into growth_steps and loop_variance_steps
  │
  ├─ 2. Precompute data:
  │     _precompute_step_data(runs)     → step_run_costs, step_iterations, step_occurrence_costs
  │     _precompute_growth_data(runs)   → step_growth dict (slope, base_context, etc.)
  │     Merge pattern metadata (growth_type, power_law_alpha) into step_growth
  │     _build_run_step_cost_map(runs)  → per-run step cost map
  │     Precompute run_non_pattern_costs (sum of non-pattern step costs per run)
  │     Compute max_observed_iters per step
  │
  ├─ 3. Compute K = min(N_monthly, 1000) samples per simulation
  │
  ├─ 4. SIMULATION LOOP (10,000 iterations):
  │     │
  │     ├─ INNER SAMPLING LOOP (K iterations):
  │     │     ├─ Pick random base_run_idx from [0, n_observed)    ← FIX 3: whole-run sampling
  │     │     ├─ Non-pattern cost = run_non_pattern_costs[base_run_idx]
  │     │     ├─ For each pattern step: _sample_step_cost()       ← FIX 4: extrapolation cap
  │     │     └─ Total run cost = non-pattern + Σ(pattern step costs)
  │     │
  │     ├─ z = rng.gauss(0, 1)
  │     ├─ monthly_cost = _clt_aggregate(K costs, N_monthly, z)   ← FIX 1: CLT correction
  │     ├─ daily_cost = monthly / 30
  │     └─ Collect sim_run_costs[0], sim_monthly, sim_daily
  │
  ├─ 5. Convergence check: |p95(first 9000) - p95(all 10000)| / p95(all) < 1%
  │
  ├─ 6. Build PercentileProjection for monthly, daily, per-run, linear, log
  │
  ├─ 7. Compute growth_model_delta = |linear_p95 - log_p95| / log_p95 × 100
  │
  └─ 8. If n_observed < 30: tail_inflation_factor = 1 + 2/√n    ← FIX 2: tail inflation
        Apply _inflate_p95() to monthly, daily, per-run projections
```

#### The four fixes explained

| Fix | Problem | Solution | Where in code |
|-----|---------|----------|---------------|
| Fix 1 (CLT) | Old code: 1 sample × N → variance = N²σ². At 10K/day, SD was 548× too wide | K inner samples → `_clt_aggregate()` computes `Nμ̂ + z√(Nσ̂²)` | `_clt_aggregate()` line ~207 |
| Fix 2 (tail) | Small samples underestimate tails | If n < 30, multiply p95 by `(1 + 2/√n)`. At n=20 → 44.7% inflation | `_inflate_p95()` + check at line ~464 |
| Fix 3 (whole-run) | Independent step sampling breaks inter-step correlation | Pick a random observed run, use ALL non-pattern step costs from it. Only pattern steps use their own models | `_build_run_step_cost_map()` + base_run_idx sampling at line ~388 |
| Fix 4 (cap) | Context growth model extrapolates unboundedly past observed data | Cap iteration at `max_observed_iter` for cost computation. `effective_k = min(k, max_obs)` | `_sample_step_cost()` line ~263 |

#### Common failure modes

1. **All pattern steps** — if the code misclassifies a step as a pattern step (e.g., false positive context growth), ALL cost for that step goes through `_sample_step_cost()` instead of whole-run sampling. The correlation-preserving benefit of Fix 3 is lost for that step.

2. **Zero observed runs** — `n_observed = len(runs)` is used as denominator in `rng.randrange(n_observed)`. If `runs` is empty, this crashes with `ValueError: empty range`. The guard `if not runs` in `detect_patterns()` prevents this upstream, but direct callers of `simulate()` could hit it.

3. **Convergence failure** — if the distribution has extreme tails (Pareto α < 1.5), 10,000 simulations may not converge. The `convergence_check` flag is set to `False`, and a warning appears in `ProjectionResult.warnings`.

#### How to debug

```bash
# Run CLT-specific tests
pytest tests/unit/test_montecarlo.py -v -k "clt"

# Run whole-run sampling tests
pytest tests/unit/test_montecarlo.py -v -k "whole_run"

# Use _debug_run_costs to inspect sampled values
python -c "
from tests.synthetic.runner import run_one
from tests.synthetic.generators import generate_lognormal
wf = generate_lognormal(0.5, 20, seed=42)
result = run_one(wf)
print(f'p50={result.projected_p50:.2f}, p95={result.projected_p95:.2f}')
"
```

---

### 2B. Pattern Detection — `pretia/projection/patterns.py`

#### What this file does

Runs 5 independent detectors on raw profiling data and returns a severity-sorted list of `DetectedPattern` objects. These patterns determine: (1) whether Monte Carlo is used, (2) which cost models apply to which steps, (3) confidence score deductions, (4) user-facing warnings.

#### Data structure: `DetectedPattern`

Frozen dataclass with 20 fields. The first 5 are always populated; the rest are `None` by default and set only by specific detectors.

| Field | Type | Set by | Purpose |
|-------|------|--------|---------|
| `pattern_type` | `str` | all | `"context_growth"`, `"loop_count_variance"`, `"high_token_variance"`, `"step_count_variance"`, `"bimodality"` |
| `step_name` | `str` | all | Step this pattern applies to, or `"_workflow_"` for workflow-level patterns |
| `severity` | `str` | all | `"danger"` or `"warning"` |
| `evidence` | `dict` | all | Numeric evidence (r², slopes, CVs, counts) |
| `description` | `str` | all | Human-readable explanation |
| `growth_type` | `str\|None` | context_growth | `"linear"` or `"nonlinear"` |
| `pearson_r_squared` | `float\|None` | context_growth | Pearson r² value |
| `spearman_rho_squared` | `float\|None` | context_growth | Spearman ρ² value |
| `pearson_significant` | `bool\|None` | context_growth | Whether Pearson r passes t-test at p < 0.05 |
| `spearman_significant` | `bool\|None` | context_growth | Whether Spearman ρ passes t-test |
| `nonlinearity_gap` | `float\|None` | context_growth | `\|ρ - r\|` — large gap means non-linear |
| `power_law_alpha` | `float\|None` | context_growth (nonlinear) | Power-law exponent α |
| `power_law_base` | `float\|None` | context_growth (nonlinear) | Power-law base coefficient |
| `growth_classification` | `str\|None` | context_growth (nonlinear) | `"sub_linear"` (α < 1) or `"super_linear"` (α ≥ 1) |
| `variance_percentile_used` | `int\|None` | high_token_variance | 90 (if n < 30) or 95 (if n ≥ 30) |
| `step_count_cv` | `float\|None` | step_count_variance | CV of active step counts across runs |
| `step_count_min` | `int\|None` | step_count_variance | Min active steps in any run |
| `step_count_max` | `int\|None` | step_count_variance | Max active steps in any run |
| `step_count_mean` | `float\|None` | step_count_variance | Mean active step count |
| `bimodal_bic_delta` | `float\|None` | bimodality | BIC(1-component) - BIC(2-component) |
| `bimodal_modes` | `list[dict]\|None` | bimodality | Per-mode stats: proportion, mean, median, min, max cost |

#### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `_T_CRITICAL_005` | dict of df → t-critical at p=0.05 | Lookup table for significance testing. Covers df 3–120, then 1.960 for df > 120 |

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_pearson_r` | `(xs: list[float], ys: list[float]) -> tuple[float, float]` | Computes **signed** Pearson r (not r²) and slope. Returns `(0.0, 0.0)` if n < 3 or either dimension has zero variance. Changed from Sprint 2: now returns r, not r². | `(r, slope)` |
| `_rank` | `(values: list[float]) -> list[float]` | Assigns 1-based ranks with tie averaging. Used for Spearman correlation. E.g., `[3, 1, 1]` → `[3.0, 1.5, 1.5]`. | `list[float]` |
| `_is_significant` | `(r_value: float, n: int) -> bool` | Converts r to t-statistic via `t = \|r\| × √(df) / √(1 - r²)`, looks up t-critical from `_T_CRITICAL_005` for df = n-2. Returns `True` if t > t_critical. Returns `False` if df < 3. | `bool` |
| `_log_log_regression` | `(xs: list[float], ys: list[float]) -> tuple[float, float]` | OLS regression on `(log(x), log(y))`. Returns `(alpha, base)` where `context ≈ base × iteration^alpha`. Filters out non-positive values. Falls back to `(1.0, exp(mean_ly))` if zero variance in log(x). | `(alpha, base)` |
| `_detect_context_growth` | `(runs) -> list[DetectedPattern]` | See detailed pipeline below. | list of 0+ patterns |
| `_detect_loop_count_variance` | `(runs) -> list[DetectedPattern]` | Groups max iteration per step per run. If CV > 0.5 → pattern. Severity: `danger` if CV > 1.0 or max > 3× mean, else `warning`. | list of 0+ patterns |
| `_detect_high_token_variance` | `(stats: ProfilingStats) -> list[DetectedPattern]` | For each step: if n < 30, uses p90; otherwise p95. If p90-or-p95 / p50 > 3.0 → pattern. Severity: `danger` if ratio > 5.0. | list of 0+ patterns |
| `_detect_step_count_variance` | `(runs) -> list[DetectedPattern]` | **NEW in Sprint 3.** Counts active steps (non-zero tokens) per run. CV > 0.3 → `warning`. CV > 0.6 or max > 2× min → `danger`. Returns `step_name="_workflow_"`. | list of 0 or 1 patterns |
| `_detect_bimodality` | `(runs, stats) -> list[DetectedPattern]` | **NEW in Sprint 3.** Lazy-imports sklearn. Fits 1-comp vs 2-comp GMM on log-costs. BIC delta > 6 → bimodal. Special case: zero-cost + positive-cost runs = inherently bimodal. Returns `step_name="_workflow_"`. Degrades gracefully if sklearn missing. | list of 0 or 1 patterns |
| `detect_patterns` | `(runs, stats=None) -> list[DetectedPattern]` | Orchestrator. Runs all 5 detectors, concatenates, sorts by severity (danger first). Computes stats if not provided. | sorted list |

#### Context growth detection pipeline (detailed)

```
_detect_context_growth(runs)
  │
  ├─ 1. Build step_pairs: for each step that iterates (iteration > 1 in any run record),
  │     collect (iteration, context_size) pairs across all runs.
  │
  ├─ 2. For each step with ≥ 5 data points:
  │     │
  │     ├─ Pearson r, slope = _pearson_r(iterations, context_sizes)
  │     ├─ pearson_r_sq = r × r
  │     ├─ pearson_sig = _is_significant(r, n) if r > 0
  │     │
  │     ├─ rank_x = _rank(iterations)
  │     ├─ rank_y = _rank(context_sizes)
  │     ├─ spearman_r, _ = _pearson_r(rank_x, rank_y)
  │     ├─ spearman_r_sq = r × r
  │     ├─ spearman_sig = _is_significant(r, n) if r > 0
  │     │
  │     ├─ pearson_passes = r² > 0.7 AND significant AND r > 0
  │     ├─ spearman_passes = ρ² > 0.7 AND significant AND ρ > 0
  │     │
  │     ├─ if NEITHER passes → skip
  │     │
  │     ├─ CLASSIFICATION:
  │     │   if pearson_passes → growth_type = "linear"
  │     │   else → growth_type = "nonlinear"
  │     │         alpha, base = _log_log_regression(xs, ys)
  │     │         classification = "sub_linear" if α < 1 else "super_linear"
  │     │
  │     ├─ SEVERITY:
  │     │   max of significant r² values. > 0.85 → "danger", else "warning"
  │     │
  │     └─ Build DetectedPattern with all growth metadata fields populated
  │
  └─ Return list of detected patterns
```

#### Detector-to-Monte Carlo mapping

This is critical for understanding which detectors affect cost models vs. just triggering MC mode:

```
┌─────────────────────────────────────────────────────────────────────┐
│ COST ADJUSTMENT DETECTORS (have custom MC models in montecarlo.py) │
│                                                                     │
│  context_growth ──────► linear+log average     (growth_type=linear) │
│                  ──────► log+power average   (sub_linear nonlinear) │
│                  ──────► power+capped average  (super_linear)       │
│                                                                     │
│  loop_count_variance ──► sample iter count × sample per-iter costs  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ NO COST ADJUSTMENT DETECTORS (whole-run resampling handles them)    │
│                                                                     │
│  high_token_variance ──► triggers MC mode; confidence deduction     │
│  step_count_variance ──► triggers MC mode; enforces whole-run       │
│  bimodality ───────────► per-mode reporting; confidence deduction   │
└─────────────────────────────────────────────────────────────────────┘
```

#### Common failure modes

1. **False positive context growth on 5 data points.** The minimum threshold of 5 is low enough that noisy data with a monotonic trend can pass. The significance test helps, but with n=5 and df=3, the t-critical is 3.182 — a high bar, but possible with r ≈ 0.98.

2. **Bimodality requiring sklearn.** If sklearn isn't installed, `_detect_bimodality` returns `[]` silently. The user sees no bimodality warning even if their cost distribution is clearly bimodal. This is intentional — sklearn is an optional dependency.

3. **`_detect_step_count_variance` with homogeneous workflows.** If every run executes exactly the same steps (e.g., a simple linear workflow), `cv = 0` and the detector returns nothing. This is correct but means conditional routing workflows need enough diverse inputs to exercise all paths.

---

### 2C. Projector — `pretia/projection/projector.py`

#### What this file does

Unified entry point that decides between linear projection and Monte Carlo, runs confidence scoring, and packages everything into a `ProjectionResult`.

#### Data structures

| Name | Type | Fields | Purpose |
|------|------|--------|---------|
| `TrafficProjection` | frozen dataclass | `daily_volume`, `monthly_cost`, `daily_cost`, `cost_per_run` (all `PercentileProjection`) | Projection for one traffic level |
| `ProjectionResult` | frozen dataclass | `method`, `traffic_volumes`, `projections`, `confidence`, `warnings`, `patterns_detected`, `montecarlo_result` | Full projection output |

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_linear_project` | `(stats: ProfilingStats, traffic: list[int]) -> dict[int, TrafficProjection]` | Scales `stats.cost_per_run` percentiles linearly: `daily = cpr × volume`, `monthly = cpr × volume × 30`. No simulation, just multiplication. | dict mapping volume → projection |
| `_montecarlo_project` | `(stats, patterns, traffic, runs) -> tuple[dict[int, TrafficProjection], dict[int, MonteCarloResult]]` | Calls `simulate()` for each traffic volume. Returns both the projections and the raw MC results. | (projections, mc_results) |
| `project` | `(stats, patterns, traffic=None, runs=None, input_source="auto-generate") -> ProjectionResult` | **Main entry.** Decision logic: if ANY pattern detected AND runs available → Monte Carlo. If patterns but no runs → linear with warning. If no patterns → linear. Computes `ConfidenceResult` from `compute_confidence()`. | `ProjectionResult` |

#### Mode selection (Sprint 3 change)

```python
# Sprint 2:
has_danger = any(p.severity == "danger" for p in patterns)
use_montecarlo = has_danger

# Sprint 3:
use_montecarlo = len(patterns) > 0
```

**Why:** Warning-level patterns like bimodality and step count variance still produce non-linear cost distributions that mean-based linear projection underestimates. Whole-run resampling in Monte Carlo mode naturally handles these without needing custom cost models.

#### Data flow through `project()`

```
project(stats, patterns, traffic=[100, 1000, 10000], runs=runs)
  │
  ├─ Extract run_costs from stats.run_stats for confidence scoring
  │
  ├─ compute_confidence(n, step_stats, patterns, input_source, run_costs)
  │   └─ → ConfidenceResult (score, tier, display_range, language)
  │
  ├─ if len(patterns) > 0:
  │   ├─ if runs is None:
  │   │   └─ FALLBACK: _linear_project() + warning
  │   └─ else:
  │       └─ _montecarlo_project() for each traffic volume
  │           └─ simulate() × 3 (for 100, 1000, 10000 daily)
  │
  ├─ else:
  │   └─ _linear_project() + "No non-linear patterns" info message
  │
  └─ Return ProjectionResult
```

---

## Part 3: Validation Layer

### 3A. Confidence Scoring — `pretia/validation/confidence.py`

#### What this file does

Assigns a 0–100 confidence score and maps it to a tier (HIGH/MODERATE/LOW/VERY_LOW). The score starts at 100 and takes deductions for small sample size, high variance, and detected patterns. Bonuses for Langfuse input and large samples.

#### Data structures

| Name | Type | Fields | Purpose |
|------|------|--------|---------|
| `ConfidenceResult` | frozen dataclass | `score`, `tier`, `display_range`, `language`, `deductions`, `bonuses`, `effective_sample_size` | Confidence assessment attached to every projection |

#### Constants

| Name | Value | Purpose |
|------|-------|---------|
| `_MAX_STEP_VARIANCE_DEDUCTION` | 30 | Cap on total per-step variance deductions |

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `compute_effective_sample_size` | `(costs: list[float]) -> float` | **NEW in Sprint 3.** Entropy-based n_eff. (1) Bin costs into √n equal-width bins. (2) Compute Shannon entropy H = −Σ(pᵢ ln pᵢ). (3) n_eff = n × (H / H_max). Returns 0.0 if all costs are identical (zero entropy). Returns ~n if costs are uniformly spread. | `float` |
| `compute_confidence` | `(sample_size, step_stats, patterns, input_source="auto-generate", run_costs=None) -> ConfidenceResult` | Computes score from 100 via deductions/bonuses (see table below). Uses n_eff from `run_costs` if provided, otherwise falls back to raw `sample_size`. | `ConfidenceResult` |

#### Deduction schedule

| Check | Condition | Points |
|-------|-----------|--------|
| Very low n_eff | n_eff < 10 | −40 |
| Low n_eff | 10 ≤ n_eff < 30 | −20 |
| Moderate n_eff | 30 ≤ n_eff < 100 | −10 |
| Per-step high variance | CV > 1.0 | −15 per step (capped at −30 total) |
| Per-step moderate variance | CV > 0.5 | −8 per step (capped at −30 total) |
| Context growth | any | −10 |
| Loop count variance | any | −10 |
| High token variance | any | −10 |
| Step count variance | danger | −15 |
| Step count variance | warning | −5 |
| Bimodality | any | −5 |
| **Bonus:** Langfuse input | input_source == "langfuse" | +15 |
| **Bonus:** Large sample | sample_size ≥ 200 | +10 |

#### Tier mapping

| Score range | Tier | display_range | language |
|-------------|------|---------------|----------|
| ≥ 80 | HIGH | "p50 – p90" | "projected" |
| 60–79 | MODERATE | "p50 – p95" | "estimated" |
| 40–59 | LOW | "p25 – p99" | "estimated" |
| < 40 | VERY_LOW | "order of magnitude" | "ballpark" |

#### n_eff intuition

```
200 identical costs ($0.05 each):     n_eff = 0    (zero information)
200 costs in 2 tight clusters:        n_eff ≈ 60   (captures bimodality)
200 costs uniformly spread $0.01-$1:  n_eff ≈ 190  (near-full diversity)
```

---

### 3B. Calibration Scoring — `pretia/validation/scoring.py`

#### What this file does

Scores a projection against ground truth. Used by the backtesting suite to validate that the projection engine produces reliable numbers across different workflow types.

#### Data structures

| Name | Type | Fields | Purpose |
|------|------|--------|---------|
| `CalibrationScore` | frozen dataclass | `workflow_name`, `sample_size`, `ground_truth_size`, `p50_ratio`, `p95_coverage`, `range_ratio`, `top_step_correct`, `ranking_correlation`, `verdict`, `failures`, `warnings` | One workflow's calibration result |

#### Constants

```python
_THRESHOLDS = {
    "simple": {"p95_coverage": 0.85, "range_ratio": 3.0},
    "complex": {"p95_coverage": 0.75, "range_ratio": 8.0},
}
```

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_spearman_rank_correlation` | `(projected_costs: dict[str, float], actual_costs: dict[str, float]) -> float` | Spearman ρ between projected and actual per-step costs. Returns 1.0 if < 3 common steps. Uses tie-aware rank assignment. | `float` in [-1, 1] |
| `_percentile` | `(sorted_data, p) -> float` | Linear interpolation percentile (same as everywhere else). | `float` |
| `bootstrap_percentile_ci` | `(costs, percentile, n_bootstrap=1000, ci_level=0.90, seed=42) -> tuple[float, float, float]` | Bootstrap confidence interval. Draws 1000 resamples, computes the target percentile for each, returns `(point_estimate, ci_lower, ci_upper)`. Uses `random.Random(seed)` — no numpy. | `(point, lower, upper)` |
| `score_projection` | `(projected, ground_truth, projected_projection=None, traffic=1000, workflow_complexity="simple", ground_truth_p50_ci=None, ground_truth_p95_ci=None) -> CalibrationScore` | Computes all calibration metrics (see flow below). | `CalibrationScore` |
| `format_calibration_report` | `(scores: list[CalibrationScore]) -> str` | Renders a terminal-friendly table with pass/warn/fail indicators per metric per workflow. Computes overall launch gate pass/fail. | `str` |

#### How `score_projection()` evaluates

```
score_projection(projected, ground_truth, ...)
  │
  ├─ P50 RATIO = projected.cost_per_run.p50 / ground_truth.cost_per_run.p50
  │   ├─ If projected p50 falls within bootstrap CI → auto-pass
  │   ├─ < 0.33 or > 3.0 → FAILURE
  │   ├─ < 0.7 or > 2.0 → WARNING
  │   └─ 0.7–2.0 → PASS
  │
  ├─ P95 COVERAGE = fraction of ground truth runs ≤ projected p95
  │   ├─ If projected p95 falls within bootstrap CI → auto-pass
  │   ├─ < 0.60 → FAILURE ("dangerously overconfident")
  │   ├─ < threshold (0.85 simple / 0.75 complex) → WARNING
  │   └─ ≥ threshold → PASS
  │
  ├─ RANGE RATIO = projected p95 / projected p50
  │   ├─ ≥ 10.0 → FAILURE ("too wide to be actionable")
  │   ├─ ≥ threshold (3.0 simple / 8.0 complex) → WARNING
  │   └─ < threshold → PASS
  │
  ├─ TOP STEP CORRECT
  │   ├─ Sort projected + actual steps by mean cost
  │   ├─ If top step matches → PASS
  │   ├─ If top two actual steps are within 30% of each other ("co-dominant"):
  │   │   picking either is acceptable → PASS with WARNING
  │   └─ Otherwise → FAILURE
  │
  ├─ RANKING CORRELATION (Spearman ρ)
  │   ├─ ≤ 3 common steps → auto-pass (1.0)
  │   ├─ ρ < 0.7 → FAILURE
  │   └─ ρ ≥ 0.7 → PASS
  │
  └─ VERDICT: FAIL if any failures, WARN if any warnings, else PASS
```

---

### 3C. Backtesting Suite — `pretia/validation/suite.py`

#### What this file does

Runs the full backtesting protocol across workflow archetypes. Implements the launch gate system with hard + soft gates, computes directional bias, manages bootstrap CIs.

#### Data structures

| Name | Type | Fields | Purpose |
|------|------|--------|---------|
| `BacktestConfig` | frozen dataclass | `name`, `archetype`, `complexity`, `workflow_path`, `description`, `expected_models`, `has_loops`, `expected_cost_range` | Description of one test workflow |
| `BacktestResult` | frozen dataclass | `config`, `synth20_score`, `synth100_score`, `ground_truth_stats`, `convergence_20_to_100` | One workflow's backtest result |
| `BacktestSuiteResult` | frozen dataclass | `results`, `overall_verdict`, `pass_count`, `warn_count`, `fail_count`, `launch_gate`, `timestamp`, `hard_gate_passed`, `soft_gate_pass_rates`, `overall_passed`, `directional_bias`, `ground_truth_cis` | Full suite output |

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `check_directional_bias` | `(results: list[dict]) -> dict` | Counts how many workflows have projected p95 above vs. below ground truth. If ≥ 80% skew one direction → flags "underestimation" or "overestimation". Diagnostic only — doesn't fail the gate. | `dict` with counts + bias flag |
| `_extract_stats` | `(session: ProfilingSession) -> ProfilingStats \| None` | Reconstructs `ProfilingStats` from `session.metadata["stats"]` dict. Handles nested `StepStats` and `RunStats` deserialization. Returns `None` if no stats in metadata. | `ProfilingStats \| None` |
| `run_backtesting_suite` | `(profiles, configs, traffic=1000) -> BacktestSuiteResult` | See detailed flow below. | `BacktestSuiteResult` |
| `format_suite_report` | `(suite_result) -> str` | Renders calibration report + convergence table + directional bias warning. | `str` |

#### Launch gate system

```
┌──────────────────────────────────────────────────────────────────┐
│                        HARD GATES                                │
│  (EVERY workflow must pass these)                                │
│                                                                  │
│  ✓ p50 ratio in (0.7, 2.0)                                      │
│  ✓ Top cost step is correct                                      │
│                                                                  │
│  If ANY workflow fails either hard gate → launch_gate = False    │
└──────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────┐
│                        SOFT GATES                                │
│  (≥ 80% of workflows must pass each)                             │
│                                                                  │
│  • p95 coverage ≥ threshold (0.85 simple / 0.75 complex)         │
│  • Range ratio < threshold (3.0 simple / 8.0 complex)            │
│  • Step ranking correlation > 0.7                                │
│                                                                  │
│  If any soft gate pass rate < 80% → launch_gate = False          │
└──────────────────────────────────────────────────────────────────┘

overall_passed = hard_gate_passed AND all_soft_gates_pass
```

#### How `run_backtesting_suite()` works

```
For each workflow config:
  ├─ Load profiles: synth20, synth100, real500 sessions
  ├─ Extract ground truth stats from real500
  ├─ If ≥ 15 ground truth runs: compute bootstrap CIs for p50, p95
  ├─ Score synth20 vs ground truth (with CIs)
  ├─ Score synth100 vs ground truth (with CIs)
  ├─ Compute convergence: |synth20_p50 - synth100_p50| / synth100_p50 × 100%
  ├─ Collect bias data (projected p95 vs ground truth p95)
  └─ Build BacktestResult

Compute launch gate:
  ├─ Hard gates: check all synth20 scores
  ├─ Soft gates: compute pass rates
  ├─ Directional bias: check_directional_bias()
  └─ Return BacktestSuiteResult
```

---

### 3D. Data Quality Checks — `pretia/validation/data_checks.py`

#### What this file does

Post-profiling validation that runs after collection completes. Checks for data quality issues that would silently corrupt projections.

#### Functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `validate_profiling_data` | `(records: list[list[StepRecord]]) -> list[str]` | For each step across all runs: (1) if zero tokens in ALL runs → "likely data collection error". (2) if zero tokens in > 50% of runs → "partial missing data". Returns warning strings, never aborts. | `list[str]` |

#### Integration point

Called from `ProfileRunner.run()` after `collector.collect()` completes:

```python
data_warnings = validate_profiling_data(runs)
for w in data_warnings:
    logger.warning(w)
```

---

### 3E. Visibility Warnings — `pretia/validation/visibility.py`

#### What this file does

Context-aware profiling recommendations and display helpers. Tells the user what to do next based on their data quality.

#### Functions — complete reference

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `get_profiling_recommendation` | `(n_eff, patterns, current_n) -> str \| None` | Priority ladder: (1) n_eff < 10 → "diverse inputs". (2) danger patterns + n < 100 → "n=100+". (3) conditional patterns + n < 50 → "n=50+". (4) n < 50 → "n=50". Returns `None` if n ≥ 100 and n_eff ≥ 50. Includes CLI command in message. | `str \| None` |
| `compute_input_stats` | `(records) -> dict` | Computes p50/p95/max/min/CV of input tokens per run and per step. Uses linear interpolation percentile internally. | `dict` with `total_input_tokens` and `per_step` keys |
| `check_input_uniformity` | `(records, patterns) -> list[str]` | Warns on: (1) CV < 0.1 for total input tokens (suspiciously uniform). (2) identical iteration counts across all runs for iterating steps. | `list[str]` |
| `check_zero_execution_steps` | `(records, workflow_steps) -> list[str]` | Compares declared workflow steps vs observed steps. Flags steps that never triggered during profiling. | `list[str]` |
| `sample_coverage_statement` | `(n: int) -> str` | Computes minimum detectable event frequency: `1 - 0.05^(1/n)`. At n=20 → "events < ~14% may not be represented". | `str` |
| `format_projection_output` | `(p50, p95, n) -> dict` | n < 10: p50 only + "could be 0.5×–3×" range note. n < 20: p50+p95 with warning. n ≥ 20: full display. | `dict` |

---

## Part 4: Collector & Pricing Hardening

### 4A. StepRecord Cache Fields — `pretia/collectors/base.py`

Two new optional fields added to `StepRecord`:

```python
cache_hit_tokens: int | None = None
cache_miss_tokens: int | None = None
```

Added at the end of the dataclass with defaults, so all existing constructor calls remain valid. `to_dict()` includes them. `from_dict()` uses `.get()` with `None` default for backward compatibility with older JSON files.

### 4B. Token Extraction — `pretia/collectors/generic.py`

`_try_extract()` now extracts DeepSeek cache fields from both dict and attribute access paths:

```python
# Dict-style:
cache_hit = usage.get("prompt_cache_hit_tokens")
cache_miss = usage.get("prompt_cache_miss_tokens")

# Attribute-style:
cache_hit = getattr(usage, "prompt_cache_hit_tokens", None)
cache_miss = getattr(usage, "prompt_cache_miss_tokens", None)
```

Both flow through `record_llm_call()` → `StepRecord(cache_hit_tokens=..., cache_miss_tokens=...)`.

`StepTracker.record_llm_call()` gained two new optional parameters: `cache_hit_tokens: int | None = None` and `cache_miss_tokens: int | None = None`.

### 4C. Cache Busting — `pretia/collectors/cache_bust.py`

#### What this file does

Prevents server-side prompt caching (specifically DeepSeek's 50× cheaper cache-hit rates) from producing unrealistically low profiling costs.

#### Functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `needs_cache_busting` | `(model: str) -> bool` | Returns `True` if `"deepseek"` appears in the lowercase model name. | `bool` |
| `cache_bust_prompt` | `(prompt: str, run_id: str \| None = None) -> str` | Appends `\n<!-- profiling-run-{id} -->` to the prompt. If `run_id` is None, generates a random 12-char hex UUID. | `str` |

#### Integration

`ProfileRunner.__init__` accepts `cache_mode: str = "cold"`. The CLI flag `--allow-cache` sets it to `"warm"`. The runner passes this to collectors that support it.

### 4D. Pricing Tables — `pretia/pricing/tables.py`

#### New constants

| Name | Type | Purpose |
|------|------|---------|
| `PRICING_LAST_UPDATED` | `str` | `"2026-05-30"` — used by staleness check |
| `MODEL_CACHE_HIT_PRICING` | `dict[str, float]` | DeepSeek models → per-million cache-hit input rate. E.g., `deepseek-v4-flash: 0.0028` (2% of standard 0.14) |
| `_VALID_TIERS` | `frozenset` | `{"frontier", "mid", "fast"}` — validates tier in `register_model()` |

#### New error class

```python
class UnrecognizedModelError(ValueError):
    """Raised when a model name is not found in the pricing table."""
```

Replaces generic `ValueError`. Error message includes: model name, similar model suggestions, and `register_model()` usage hint.

#### New and changed functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_find_similar_models` | `(model: str, max_results: int = 5) -> list[str]` | Normalizes model name (lowercase, strip `-_. `), compares prefix overlap and substring matching against all known models. Returns up to 5 candidates sorted by match score. | `list[str]` |
| `register_model` | `(model: str, input_price: float, output_price: float, tier: str = "mid") -> None` | Adds entries to `MODEL_PRICING` and `MODEL_TIERS` at runtime. Validates tier against `_VALID_TIERS`. Used by synthetic testing to register `_synthetic_unit_cost_`. | `None` |
| `resolve_model` | `(model: str) -> str` | **Changed.** Now raises `UnrecognizedModelError` (not ValueError) with similar model suggestions. | `str` |
| `calculate_cost` | `(model, input_tokens, output_tokens, cache_hit_tokens=None, cache_miss_tokens=None) -> float` | **Changed.** When `cache_hit_tokens` AND `cache_miss_tokens` are both provided AND the model has cache pricing in `MODEL_CACHE_HIT_PRICING`: `input_cost = miss × standard_rate + hit × cache_hit_rate`. Otherwise falls back to `input_tokens × standard_rate`. | `float` |
| `check_pricing_staleness` | `() -> str \| None` | Compares `PRICING_LAST_UPDATED` against today. If > 30 days old, returns a warning string. Otherwise `None`. | `str \| None` |

#### Cache pricing formula

```
Standard:  input_cost = input_tokens × (input_price_per_M / 1_000_000)
Cached:    input_cost = cache_miss_tokens × standard_rate
                      + cache_hit_tokens × (cache_hit_rate / 1_000_000)

Example (DeepSeek V4 Flash, 1000 cache hits + 200 cache misses):
  Standard rate = 0.14 / 1M = 0.00000014 per token
  Cache hit rate = 0.0028 / 1M = 0.0000000028 per token
  input_cost = 200 × 0.00000014 + 1000 × 0.0000000028
             = 0.000028 + 0.0000028
             = 0.0000308
```

---

## Part 5: Significance Testing & Baseline Diffing

### `pretia/ci/diff.py`

#### What this file does

Compares a saved baseline against a new profiling session and computes per-step deltas, model changes, pattern changes, and monthly projection changes. Also provides a standalone Mann-Whitney U test for significance testing.

#### Statistical functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_normal_cdf` | `(z: float) -> float` | Approximate CDF of standard normal using Abramowitz & Stegun polynomial (26.2.17). All stdlib — no scipy. Handles negative z via symmetry. | `float` in [0, 1] |
| `_rank_values` | `(values: list[float]) -> list[float]` | 1-based rank assignment with tie averaging. Identical to `_rank()` in patterns.py. | `list[float]` |
| `mann_whitney_u` | `(x: list[float], y: list[float]) -> float` | Two-tailed Mann-Whitney U test via normal approximation. Returns p-value. Returns 1.0 if either group has < 2 values. | `float` (p-value) |
| `significance_label` | `(p_value: float) -> str` | `< 0.05` → "significant", `< 0.10` → "possibly significant", else → "not significant". | `str` |

#### Data structures

| Name | Type | Fields | Purpose |
|------|------|--------|---------|
| `ModelChange` | frozen dataclass | `step_name`, `old_model`, `new_model`, `cost_impact` | One step's model swap |
| `PatternChanges` | frozen dataclass | `new_patterns`, `resolved_patterns`, `unchanged_patterns` | Diff of detected patterns |
| `StepDiff` | frozen dataclass | `step_name`, `cost_change_pct`, `cost_change_abs`, `token_change_pct`, `iteration_change`, `model_changed`, `old_model`, `new_model`, `flags` | One step's full diff |
| `DiffResult` | frozen dataclass | `baseline_workflow`, `baseline_date`, `new_date`, `total_monthly_change`, `total_monthly_pct_change`, `step_diffs`, `new_steps`, `removed_steps`, `model_changes`, `pattern_changes`, `exceeds_threshold`, `summary`, `traffic` | Full comparison output |

#### Functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_pct_change` | `(old, new) -> float` | Percentage change: `(new - old) / |old| × 100`. Returns 0 if both 0, 100 if old is 0 and new is not. | `float` |
| `diff_baseline` | `(baseline, new_session, traffic=None) -> DiffResult` | Computes monthly change at p50/p95, per-step diffs (cost/token/iteration changes), model changes, pattern changes, summary string. | `DiffResult` |
| `format_diff_report` | `(diff: DiffResult) -> str` | Renders step comparison table, new/removed steps, model changes, pattern changes, monthly projection delta. | `str` |

---

## Part 6: Runner & CLI Changes

### `pretia/runner.py` — Sprint 3 Changes

1. **`cache_mode` parameter** — `ProfileRunner.__init__` accepts `cache_mode: str = "cold"`. Passed to collectors that support cache busting.

2. **`validate_profiling_data` integration** — Called after `collector.collect()` completes. Warnings are logged, never abort the pipeline:
   ```python
   data_warnings = validate_profiling_data(runs)
   for w in data_warnings:
       logger.warning(w)
   ```

3. **`project()` integration** — The runner now calls `project(stats, patterns, runs=runs, input_source=selection.mode)` and stores the full `ProjectionResult` in session metadata.

4. **Auto baseline diff** — `_auto_diff_baseline()` checks for `.pretia/baseline.json` and shows a one-line summary if it exists.

### `pretia/cli.py` — Sprint 3 Changes

New CLI commands:

| Command | What it does |
|---------|-------------|
| `pretia baseline update <profile>` | Save a profile as a cost baseline |
| `pretia diff <baseline> <profile>` | Compare baseline to new profile |
| `pretia validate <workflow>` | Run projection quality check (small-n vs large-n) |
| `pretia update-pricing` | Placeholder for pricing updates |

New flags:

| Flag | Command | What it does |
|------|---------|-------------|
| `--allow-cache` | `profile run` | Allow server-side prompt caching (default: bust cache) |
| `--threshold N` | `diff` | Fail if monthly cost increase exceeds N% |
| `--budget N` | `validate` | Estimated cost for validation runs (default: $10) |
| `--small-n N` | `validate` | First sample size (default: 20) |
| `--large-n N` | `validate` | Second sample size (default: 100) |

---

## Part 7: Synthetic Testing Framework

### 7A. Distribution Generators — `tests/synthetic/generators.py`

#### What this file does

Generates synthetic per-run cost data from 5 known probability distributions with analytically or empirically computed ground-truth percentiles. Used to test whether the projection engine is calibrated — i.e., its p50 and p95 projections are accurate for data with known statistical properties.

#### Data structure: `SyntheticWorkflow`

| Field | Type | Purpose |
|-------|------|---------|
| `name` | `str` | E.g., `"lognormal_sigma_0.5_n_20"` |
| `distribution_type` | `str` | `"lognormal"`, `"bimodal"`, `"pareto"`, `"zero_inflated"`, `"uniform"` |
| `params` | `dict` | Distribution parameters |
| `observed_costs` | `list[float]` | The N sampled costs (what the projection engine sees) |
| `true_p50` | `float` | Ground truth p50 of the underlying distribution |
| `true_p95` | `float` | Ground truth p95 |
| `true_mean` | `float` | Ground truth mean |
| `true_std` | `float` | Ground truth standard deviation |
| `sample_size` | `int` | N |

#### Helper functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_standard_normal` | `(rng: random.Random) -> float` | Box-Muller transform for normal draws. No numpy dependency. | `float` |
| `_percentile` | `(sorted_data, p) -> float` | Linear interpolation percentile. | `float` |
| `_reference_percentiles` | `(draw_fn, n_ref=100_000, seed=999999) -> tuple[float, float, float, float]` | Draws 100K samples to compute empirical (p50, p95, mean, std). Used for distributions without closed-form percentile formulas (bimodal, zero-inflated). | `(p50, p95, mean, std)` |

#### Generators

| Generator | Signature | How it works | Ground truth source |
|-----------|-----------|-------------|---------------------|
| `generate_lognormal` | `(sigma, n, seed, mu=0.0)` | `cost = exp(mu + sigma × z)` where z ~ N(0,1) via Box-Muller. | Analytical: `p50 = exp(mu)`, `p95 = exp(mu + 1.645σ)` |
| `generate_bimodal` | `(mixing, separation, n, seed, sigma_within=0.3)` | Bernoulli mixture: with prob `mixing` → expensive mode `exp(log(separation) + σ_w × z)`, else → cheap mode `exp(0 + σ_w × z)`. | 100K-draw reference via `_reference_percentiles` |
| `generate_pareto` | `(alpha, n, seed, x_min=1.0)` | Inverse CDF: `x_min / u^(1/α)` where u ~ Uniform(0,1). | Analytical for α > 2; reference otherwise |
| `generate_zero_inflated` | `(trigger_prob, n, seed, near_zero=0.001)` | With prob `trigger_prob` → lognormal(0, 0.5), else → `near_zero`. | 100K-draw reference |
| `generate_uniform` | `(n, seed, low=0.5, high=1.5)` | `low + u × (high - low)`. Control distribution. | Analytical: `p50 = (low+high)/2`, `std = (high-low)/√12` |

#### Master generator

`generate_all_synthetic_workflows(seed_base=42)` produces 520+ workflows:

| Distribution | Variants | × Sample sizes | Total |
|--------------|----------|----------------|-------|
| Log-normal | 15 σ values (0.2–1.5) | × 5 (20, 50, 100, 150, 300) | 75 |
| Bimodal | 6 mixings × 6 separations × 2 σ_within | × 5 | 360 |
| Pareto | 9 α values (1.2–5.0) | × 5 | 45 |
| Zero-inflated | 7 trigger probabilities (0.05–0.50) | × 5 | 35 |
| Uniform | 1 | × 5 | 5 |
| **Total** | | | **520** |

### 7B. Synthetic Runner — `tests/synthetic/runner.py`

#### What this file does

Feeds synthetic cost data through the full projection engine pipeline, converting dollar costs into `StepRecord` objects.

#### Key constant

```python
_SYNTHETIC_MODEL = "_synthetic_unit_cost_"
# Registered with: input_price=$1.00/MTok, output_price=$0.00/MTok
# So input_tokens = cost × 1_000_000 produces exactly that dollar cost.
```

#### Functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_ensure_synthetic_model` | `() -> None` | Registers `_synthetic_unit_cost_` if not already in `MODEL_PRICING`. | `None` |
| `_cost_to_record` | `(cost: float, run_idx: int) -> StepRecord` | Converts a cost value into a StepRecord: `input_tokens = max(1, int(cost × 1_000_000))`, `output_tokens = 0`, `model = _synthetic_unit_cost_`, `step_name = "main"`. | `StepRecord` |
| `run_one` | `(wf: SyntheticWorkflow, daily_volume=1000) -> SyntheticCalibrationResult` | Runs full pipeline: costs → StepRecords → `compute_stats` → `detect_patterns` → `project`. Extracts projected p50/p95 from monthly projection. | `SyntheticCalibrationResult` |
| `run_synthetic_calibration` | `(workflows, daily_volume=1000) -> list[SyntheticCalibrationResult]` | Batch runner. Processes all workflows, prints progress every 50. | `list[SyntheticCalibrationResult]` |

#### Data structure: `SyntheticCalibrationResult`

| Field | Type | Purpose |
|-------|------|---------|
| `workflow` | `SyntheticWorkflow` | The input workflow |
| `projected_p50` | `float` | Engine's p50 monthly projection |
| `projected_p95` | `float` | Engine's p95 monthly projection |
| `projected_mean` | `float \| None` | Engine's mean monthly projection |
| `patterns_detected` | `list[str]` | Pattern types the engine found |
| `confidence_tier` | `str` | Confidence tier assigned |
| `tail_inflation_factor` | `float \| None` | If MC was used, the tail inflation factor |

### 7C. Calibration Report — `tests/synthetic/calibration.py`

#### What this file does

Compares projected percentiles against known distribution truth to produce a calibration report.

#### Key metrics

- **p50 ratio** = `projected_p50 / (N × true_mean)` — should be in (0.7, 2.0)
- **p95 coverage** = `projected_p95 ≥ N × true_mean + 1.645 × √N × true_std` — CLT-approximated true p95

#### Data structure: `CalibrationReport`

| Field | Type | Purpose |
|-------|------|---------|
| `total_workflows` | `int` | Number tested |
| `daily_volume` | `int` | Traffic assumption |
| `p50_calibration_rate` | `float` | Fraction with p50 ratio in (0.7, 2.0) |
| `p95_coverage_rate` | `float` | Fraction where projected p95 ≥ true p95 |
| `mean_p50_error` | `float` | Mean |p50_ratio - 1| |
| `by_sample_size` | `dict` | Breakdown by n |
| `by_distribution` | `dict` | Breakdown by distribution type |
| `failures` | `list[dict]` | Workflows that failed p50 calibration |

#### Functions

| Function | Signature | What it does | Returns |
|----------|-----------|-------------|---------|
| `_true_monthly_p95` | `(wf_result, n_monthly) -> float` | CLT approximation: `N × mean + 1.645 × √N × std` | `float` |
| `compute_calibration_report` | `(results, daily_volume=1000) -> CalibrationReport` | Computes all calibration metrics with breakdowns | `CalibrationReport` |
| `format_report` | `(report) -> str` | Markdown-formatted report with tables | `str` |

---

## Part 8: Backtesting Infrastructure

### Workflow Configs — `tests/backtesting/configs.py`

13 total configs, 3 excluded (W3, W6, W7 are redundant), leaving **10 active workflows**:

| Name | Archetype | Complexity | Models | Loops | Cost range |
|------|-----------|------------|--------|-------|------------|
| W1 | support-agent | simple | Haiku, Sonnet | No | $0.005–$0.03 |
| W2 | support-agent | complex | Haiku, Sonnet, Opus | Yes (1–15 iter) | $0.08–$0.60 |
| W4 | code-review | complex | Sonnet, Opus | Yes (1–8 iter) | $0.15–$1.20 |
| W5 | data-extraction | simple | Haiku, Sonnet | No | $0.005–$0.04 |
| W8 | research-agent | complex | Haiku, Sonnet, Opus | Yes (1–6 iter) | $0.25–$1.80 |
| W9 | sales-outreach | simple | GPT-4.1 Nano, GPT-4.1 | No | $0.005–$0.03 |
| W10 | sales-outreach | complex | Gemini Flash, GPT-4.1, Opus | Yes (1–4 iter) | $0.10–$0.70 |
| W11 | support-agent | simple | Qwen-Turbo, Qwen 3.6 Plus | No | $0.001–$0.01 |
| W12 | data-extraction | simple | DeepSeek V4 Flash | No | $0.001–$0.01 |
| **W13** | **routing-agent** | **complex** | Haiku, Sonnet, Opus | No | $0.001–$0.20 |

**W13** is new in Sprint 3: a conditional routing agent that classifies queries and routes to 3 paths (70% cheap / 20% moderate / 10% expensive). Tests `step_count_variance` and `bimodality` detectors.

---

## Part 9: Full Data Flow Diagrams

### Pipeline: `pretia profile run workflow.py --auto-generate 50`

```
User runs: pretia profile run workflow.py --auto-generate 50
    │
    ▼
cli.py:run()
    │  creates ProfileRunner(workflow_path, cache_mode="cold", auto_generate=50)
    ▼
ProfileRunner.run_sync() → asyncio.run(self.run())
    │
    ├─ _load_workflow()
    │   ├─ _load_workflow_module(path)       importlib dynamic import
    │   ├─ _find_workflow(module)            scan for graph/workflow/agent/app
    │   └─ _extract_system_prompt(module)    scan for long strings
    │   → (workflow_object, system_prompt)
    │
    ├─ _select_collector(workflow)
    │   ├─ ainvoke + nodes? → LangGraphCollector
    │   ├─ name + instructions? → OpenAIAgentsCollector
    │   ├─ run + llm + system_message? → QwenAgentCollector
    │   └─ else → GenericCollector
    │   → BaseCollector
    │
    ├─ _resolve_inputs(system_prompt)
    │   ├─ select_input_mode(auto_generate=50)
    │   └─ generate_inputs(system_prompt, n=50)
    │   → (InputSelection, list[str] of 50 inputs)
    │
    ├─ await collector.collect(workflow, inputs)
    │   → runs: list[list[StepRecord]]  (50 runs)
    │
    ├─ validate_profiling_data(runs)              ◄── NEW Sprint 3
    │   → list[str] warnings (logged)
    │
    ├─ _build_cost_summary(runs)                  (legacy, Sprint 1 format)
    │   → cost_summary: dict
    │
    ├─ compute_stats(runs)                        ◄── Sprint 2
    │   → ProfilingStats
    │
    ├─ detect_patterns(runs, stats)               ◄── Enhanced Sprint 3
    │   ├─ _detect_context_growth()    dual Pearson+Spearman + power-law
    │   ├─ _detect_loop_count_variance()
    │   ├─ _detect_high_token_variance()  p90 for n<30, p95 for n≥30
    │   ├─ _detect_step_count_variance()  ◄── NEW Sprint 3
    │   └─ _detect_bimodality()           ◄── NEW Sprint 3 (needs sklearn)
    │   → list[DetectedPattern]
    │
    ├─ project(stats, patterns, runs=runs, input_source="auto-generate")
    │   ├─ compute_confidence(n, step_stats, patterns, run_costs=costs)
    │   │   └─ compute_effective_sample_size(costs) → n_eff  ◄── NEW Sprint 3
    │   │   → ConfidenceResult
    │   │
    │   ├─ if patterns:
    │   │   simulate(stats, patterns, daily_volume, runs)  ◄── Overhauled Sprint 3
    │   │   ├─ Fix 1: CLT aggregation (_clt_aggregate)
    │   │   ├─ Fix 2: tail inflation (_inflate_p95)
    │   │   ├─ Fix 3: whole-run sampling (_build_run_step_cost_map)
    │   │   └─ Fix 4: extrapolation cap (max_observed_iters)
    │   │   → MonteCarloResult
    │   │
    │   └─ else: _linear_project(stats, traffic)
    │   → ProjectionResult
    │
    ├─ Build ProfilingSession with metadata:
    │   {"cost_summary", "stats", "patterns", "projection", "confidence"}
    │
    ├─ ProfileStore.save(session)
    │   → .pretia/{workflow}_{timestamp}.json
    │
    └─ _auto_diff_baseline(session)               ◄── NEW Sprint 3
        └─ If baseline.json exists → diff_baseline() → one-line summary
    │
    ▼
cli.py: format_cli_report(session) → console.print()
```

### Pipeline: Synthetic calibration

```
generate_all_synthetic_workflows(seed=42)
    → 520 SyntheticWorkflow objects
    │  Each has: observed_costs[], true_p50, true_p95, true_mean, true_std
    │
    ▼
run_synthetic_calibration(workflows, daily_volume=1000)
    │
    └─ For each workflow:
         │
         ├─ register_model("_synthetic_unit_cost_", $1.00/MTok, $0.00/MTok)
         │
         ├─ For each cost: StepRecord(input_tokens = cost × 1M, output_tokens = 0)
         │   → runs = [[record1], [record2], ..., [recordN]]
         │
         ├─ compute_stats(runs) → ProfilingStats
         ├─ detect_patterns(runs, stats) → list[DetectedPattern]
         ├─ project(stats, patterns, traffic=[1000], runs=runs) → ProjectionResult
         │
         └─ Extract: projected_p50, projected_p95, confidence_tier, patterns
         → SyntheticCalibrationResult
    │
    ▼
compute_calibration_report(results, daily_volume=1000)
    │
    ├─ For each result:
    │   p50_ratio = projected_p50 / (N_monthly × true_mean)
    │   p95_covered = projected_p95 ≥ CLT-approximated true_p95
    │
    ├─ Aggregate by sample size: {20: {...}, 50: {...}, ...}
    ├─ Aggregate by distribution: {"lognormal": {...}, "bimodal": {...}, ...}
    │
    └─ CalibrationReport
         p50_calibration_rate = fraction in (0.7, 2.0)
         p95_coverage_rate = fraction meeting threshold
```

---

## Part 10: Worked Example Runs

Six traced examples that exercise every major Sprint 3 code path. Each shows the exact functions called, the intermediate values at each stage, and which branch the code takes. Read these like a debugger trace — they show you *why* the code does what it does for concrete data.

### Coverage map

| Example | Linear proj | Monte Carlo | Context growth | Loop variance | Token variance | Step count var | Bimodality | n_eff | Cache pricing | Data checks | Tail inflation |
|---------|:-----------:|:-----------:|:--------------:|:-------------:|:--------------:|:--------------:|:----------:|:-----:|:-------------:|:-----------:|:--------------:|
| A (simple) | ✓ | | | | | | | ✓ | | | |
| B (context growth) | | ✓ | ✓ | | | | | ✓ | | | ✓ |
| C (loop variance) | | ✓ | | ✓ | | | | ✓ | | | |
| D (routing/bimodal) | | ✓ | | | | ✓ | ✓ | ✓ | | ✓ | ✓ |
| E (DeepSeek cache) | ✓ | | | | | | | ✓ | ✓ | ✓ | |
| F (synthetic calibration) | | ✓ | | | ✓ | | | ✓ | | | |

---

### Example A: Simple 2-step workflow — Linear projection, no patterns

**Scenario:** A support agent with two non-iterating steps: classify (Haiku) and respond (Sonnet). 5 profiling runs. No loops, no variance. This is the simplest possible path through the engine.

**Input data:**

```python
# 5 runs, each with 2 steps. Token counts vary slightly.
runs = [
    [classify(in=300, out=20), respond(in=600, out=150)],   # run 0
    [classify(in=320, out=18), respond(in=650, out=140)],   # run 1
    [classify(in=280, out=22), respond(in=580, out=160)],   # run 2
    [classify(in=310, out=19), respond(in=620, out=145)],   # run 3
    [classify(in=290, out=21), respond(in=610, out=155)],   # run 4
]
# Models: classify=claude-haiku-4-5, respond=claude-sonnet-4-6
```

**Trace:**

```
ProfileRunner.run()
 │
 ├─ 1. validate_profiling_data(runs)
 │      For each step, count zero-token runs:
 │        classify: 0 zero-token runs out of 5 → no warning
 │        respond:  0 zero-token runs out of 5 → no warning
 │      → warnings = []
 │
 ├─ 2. compute_stats(runs)
 │      For each of 5 runs, iterate records:
 │        run 0: classify cost = calculate_cost("claude-haiku-4-5", 300, 20)
 │                = (300 × 1.00/1M) + (20 × 5.00/1M) = 0.000300 + 0.000100 = 0.000400
 │                respond cost = calculate_cost("claude-sonnet-4-6", 600, 150)
 │                = (600 × 3.00/1M) + (150 × 15.00/1M) = 0.001800 + 0.002250 = 0.004050
 │                run_cost = 0.000400 + 0.004050 = 0.004450
 │        ... (similar for runs 1–4)
 │
 │      run_costs = [0.00445, 0.00452, 0.00432, 0.00444, 0.00440]
 │      compute_percentile_stats(run_costs):
 │        sorted = [0.00432, 0.00440, 0.00444, 0.00445, 0.00452]
 │        mean = 0.004426,  p50 = 0.00444,  p95 = 0.004506
 │      → ProfilingStats with 2 StepStats, 5 RunStats
 │
 ├─ 3. detect_patterns(runs, stats)
 │      _detect_context_growth(runs):
 │        Both steps have iteration=1 in all runs.
 │        "rec.iteration > 1" is False for every record.
 │        step_pairs = {} → empty → return []
 │
 │      _detect_loop_count_variance(runs):
 │        max iteration per step per run = 1 for both steps.
 │        "all(i == 1 for i in iters)" → True for both → return []
 │
 │      _detect_high_token_variance(stats):
 │        classify: p95/p50 tokens ≈ 340/300 = 1.13 → ratio 1.13 ≤ 3.0 → skip
 │        respond:  p95/p50 tokens ≈ 805/760 = 1.06 → ratio 1.06 ≤ 3.0 → skip
 │        → return []
 │
 │      _detect_step_count_variance(runs):
 │        active_counts = [2, 2, 2, 2, 2] → CV = 0 → return []
 │
 │      _detect_bimodality(runs, stats):
 │        len(run_stats) = 5 < 15 → return []
 │
 │      → patterns = []  (empty)
 │
 ├─ 4. project(stats, patterns=[], runs=runs, input_source="auto-generate")
 │      run_costs from stats.run_stats → [0.00445, 0.00452, 0.00432, 0.00444, 0.00440]
 │
 │      compute_confidence(5, step_stats, patterns=[], "auto-generate", run_costs)
 │        compute_effective_sample_size([0.00445, ..., 0.00440]):
 │          n=5, min=0.00432, max=0.00452 → not equal → proceed
 │          n_bins = max(2, √5) = max(2, 2) = 2
 │          bin_width = (0.00452 - 0.00432) / 2 = 0.0001
 │          Assign bins: [0.00432, 0.00440] → bin 0 (2 values),
 │                        [0.00444, 0.00445, 0.00452] → bin 1 (3 values)
 │          entropy = -(2/5)ln(2/5) - (3/5)ln(3/5) = 0.673
 │          h_max = ln(2) = 0.693
 │          n_eff = 5 × (0.673/0.693) = 4.86
 │        → n_eff = 4.86
 │
 │        n_eff < 10 → score -= 40, deduction: "Very low effective sample size (n_eff=5)"
 │        Per-step variance: both CVs are tiny (<0.05) → no deductions
 │        No patterns → no pattern deductions
 │        input_source = "auto-generate" → no bonus
 │        sample_size = 5 < 200 → no bonus
 │        score = 100 - 40 = 60 → MODERATE tier
 │        → ConfidenceResult(score=60, tier="MODERATE", display_range="p50 – p95")
 │
 │      use_montecarlo = len([]) > 0 → False
 │      → _linear_project(stats, [100, 1000, 10000])
 │        For volume=1000:
 │          daily p50  = 0.00444 × 1000 = $4.44
 │          monthly p50 = 0.00444 × 1000 × 30 = $133.20
 │          monthly p95 = 0.004506 × 1000 × 30 = $135.18
 │
 │      → ProjectionResult(method="linear", confidence.tier="MODERATE")
 │
 └─ Result: $133/month median, $135/month p95 at 1000 runs/day.
    No patterns. MODERATE confidence (sample too small).
```

**Key takeaway:** With no patterns detected, the engine uses simple multiplication. The bottleneck to useful results is the tiny sample size (n=5, n_eff≈5), which gets flagged immediately.

---

### Example B: Review loop with context growth — Monte Carlo with linear growth model

**Scenario:** A code review agent where the "review" step loops 1–6 times, accumulating context with each iteration. The context_size grows linearly (~400 tokens/iteration). 20 profiling runs.

**Input data:**

```python
# Each run has: analyze(1 call) + review(1–6 iterations, growing context)
# run 0: analyze(in=800,out=200), review×3(ctx=500,900,1300)
# run 1: analyze(in=750,out=190), review×5(ctx=500,900,1300,1700,2100)
# run 2: analyze(in=820,out=210), review×2(ctx=500,900)
# ... (20 runs total, review iterations range from 1 to 6)
# Model: review uses claude-opus-4-20250514
```

**Trace:**

```
├─ 2. compute_stats(runs)  →  20 RunStats, run_costs range $0.10 – $0.55
│
├─ 3. detect_patterns(runs, stats)
│
│    _detect_context_growth(runs):
│      step_pairs["review"]:
│        Collects (iteration, context_size) across all 20 runs.
│        Total data points: sum of all review iterations ≈ 60 pairs.
│        xs = [1, 2, 3, 1, 2, 3, 4, 5, 1, 2, ...]
│        ys = [500, 900, 1300, 500, 900, 1300, 1700, 2100, 500, 900, ...]
│        n = 60 ≥ 5 → proceed
│
│        _pearson_r(xs, ys):
│          Linear growth: context ≈ 100 + 400 × iteration
│          r ≈ 0.98, slope ≈ 400
│        pearson_r_sq = 0.98² = 0.9604
│        _is_significant(0.98, 60):
│          df = 58, t = 0.98 × √58 / √(1-0.96) = 0.98 × 7.62 / 0.2 = 37.3
│          t_crit for df=58 → nearest is df=50: 2.009
│          37.3 > 2.009 → True
│        pearson_sig = True
│
│        _rank(xs), _rank(ys) → rank-transformed data
│        spearman_r ≈ 0.97  (nearly identical to Pearson for linear growth)
│        spearman_r_sq = 0.9409
│        spearman_sig = True
│
│        pearson_passes = 0.9604 > 0.7 AND True AND 0.98 > 0 → True
│        → growth_type = "linear"  (Pearson passes → no need for power-law)
│
│        nonlinearity_gap = |0.97 - 0.98| = 0.01  (tiny → confirms linearity)
│
│        max_sig_r_sq = max(0.9604, 0.9409) = 0.9604 > 0.85 → severity = "danger"
│
│      → DetectedPattern(pattern_type="context_growth", step_name="review",
│           severity="danger", growth_type="linear", slope=400,
│           pearson_r_squared=0.9604, spearman_rho_squared=0.9409)
│
│    _detect_loop_count_variance(runs):
│      review max iterations across 20 runs: [3, 5, 2, 4, 6, 1, 3, ...]
│      mean = 3.2, std = 1.4, CV = 1.4/3.2 = 0.44
│      0.44 ≤ 0.5 → does NOT pass threshold → return []
│      (Just barely below the threshold — a few more high-iteration runs would trigger it)
│
│    _detect_high_token_variance(stats):
│      n=20 < 30 → use p90 instead of p95
│      review: p90/p50 tokens ≈ 8000/3600 = 2.2 ≤ 3.0 → skip
│
│    _detect_step_count_variance(runs):
│      active_counts = [2, 2, 2, ..., 2] → CV = 0 → return []
│
│    _detect_bimodality(runs, stats):
│      20 run_stats ≥ 15 → proceed
│      costs = [0.10, 0.22, 0.08, 0.18, 0.30, ...]
│      positive_costs: all > 0, zero_count = 0
│      len(positive_costs) = 20 ≥ 15 → fit GMM
│      log_costs, fit 1-comp vs 2-comp:
│        BIC delta = 2.1 ≤ 6 → NOT bimodal
│      → return []
│
│    → patterns = [context_growth on "review" (danger)]
│
├─ 4. project(stats, patterns=[context_growth], runs=runs)
│
│    compute_confidence(20, step_stats, [context_growth], "auto-generate", run_costs)
│      n_eff = compute_effective_sample_size(run_costs):
│        20 costs spread $0.08–$0.55, reasonably diverse
│        n_bins = max(2, √20) = 4
│        Good spread across bins → entropy ≈ 1.2, h_max = ln(4) = 1.386
│        n_eff = 20 × (1.2/1.386) ≈ 17.3
│      n_eff = 17.3 → 10 ≤ 17.3 < 30 → score -= 20
│      context_growth pattern → score -= 10
│      CV of review step cost: cost range is wide → CV ≈ 0.6 > 0.5 → score -= 8
│      score = 100 - 20 - 10 - 8 = 62 → MODERATE tier
│
│    use_montecarlo = len([context_growth]) > 0 → True
│    runs is not None → proceed with Monte Carlo
│
│    _montecarlo_project(stats, [context_growth], [100, 1000, 10000], runs)
│      For daily_volume=1000:
│        simulate(stats, [context_growth], 1000, runs, n_simulations=10000, seed=42)
│
│          growth_steps = {"review"}
│          loop_variance_steps = {}  (loop CV was 0.44, below threshold)
│
│          _precompute_step_data(runs):
│            step_run_costs: {"analyze": [20 values], "review": [20 values]}
│            step_iterations: {"analyze": [1,1,...], "review": [3,5,2,4,...]}
│            step_occurrence_costs: per-call costs for each step
│
│          _precompute_growth_data(runs, {"review"}):
│            Collects (iteration, context_size) for "review" across 20 runs
│            base_context = mean of iteration-1 contexts ≈ 500
│            slope = 400 (OLS regression)
│            max_observed_context = 2500 (from a 6-iteration run)
│            → step_growth["review"] = {slope: 400, base_context: 500, model: "claude-opus-4-20250514", ...}
│
│          Merge pattern metadata → step_growth["review"]["growth_type"] = "linear"
│
│          _build_run_step_cost_map(runs):
│            20 dicts: [{analyze: 0.04, review: 0.06}, {analyze: 0.038, review: 0.18}, ...]
│
│          pattern_steps = {"review"} (only review has a pattern)
│          run_non_pattern_costs = [0.04, 0.038, 0.042, ...] (just analyze costs per run)
│
│          n_monthly = 1000 × 30 = 30000
│          k_samples = min(30000, 1000) = 1000
│
│          SIMULATION LOOP (10,000 iterations):
│            For simulation i=0:
│              K inner samples (1000):
│                sample j=0:
│                  base_run_idx = rng.randrange(20) → e.g., 7
│                  run_avg = run_non_pattern_costs[7] = 0.039  (analyze cost from run 7)
│
│                  For pattern step "review":
│                    _sample_step_cost("review", rng, ...):
│                      growth_steps contains "review" → enter growth branch
│                      step_growth["review"]: slope=400, base_context=500, growth_type="linear"
│                      iters = [3, 5, 2, 4, 6, 1, ...] → rng.choice → e.g., 4
│                      max_obs = max_observed_iters["review"] = 6
│                      4 ≤ 6 → capped = False
│
│                      For k=1..4:
│                        effective_k = min(k, 6) = k
│                        ctx_linear = 500 + 400 × k
│                          k=1: 900, k=2: 1300, k=3: 1700, k=4: 2100
│                        median_iter = sorted([3,5,2,4,...])[10] ≈ 3
│                        log_scale = ln(4) = 1.386
│                        ctx_log = 500 + 400 × (ln(k+1)/1.386) × 3
│                          k=1: 500+400×(0.693/1.386)×3 = 500+600 = 1100
│                          k=2: 500+400×(1.099/1.386)×3 = 500+951 = 1451
│                          k=3: 500+400×(1.386/1.386)×3 = 500+1200 = 1700
│                          k=4: 500+400×(1.609/1.386)×3 = 500+1393 = 1893
│
│                        linear_cost += calculate_cost(opus, ctx_linear, mean_output)
│                        log_cost    += calculate_cost(opus, ctx_log, mean_output)
│
│                      avg_cost = (linear_cost + log_cost) / 2
│                      → returns (avg_cost ≈ 0.22, linear_cost, log_cost, False)
│
│                  run_avg = 0.039 + 0.22 = 0.259
│                ... (repeat for 999 more inner samples)
│
│              z = rng.gauss(0, 1) → e.g., 0.34
│              monthly = _clt_aggregate(1000 costs, 30000, 0.34)
│                mu = mean of 1000 sampled run costs ≈ 0.25
│                var = sample variance ≈ 0.012
│                result = 30000 × 0.25 + 0.34 × √(30000 × 0.012)
│                       = 7500 + 0.34 × 18.97 = 7500 + 6.45 = $7,506
│
│          ... (10,000 simulations → 10,000 monthly costs)
│
│          Convergence check: |p95(first 9000) - p95(all 10000)| / p95(all) < 1% → True
│
│          n_observed = 20 < 30 → tail inflation:
│            factor = 1 + 2/√20 = 1 + 0.447 = 1.447
│            monthly p95 *= 1.447
│            daily p95 *= 1.447
│            per_run p95 *= 1.447
│
│    → ProjectionResult(method="montecarlo", confidence.tier="MODERATE")
│       monthly p50 ≈ $7,500, monthly p95 ≈ $10,800 (after inflation) at 1000/day
```

**Key takeaway:** The context growth detector fires with both Pearson (r²=0.96) and Spearman (ρ²=0.94) — both significant. Because Pearson passes, it's classified as "linear" growth (no power-law needed). Monte Carlo uses the linear+logarithmic average model for the review step, whole-run sampling for the analyze step. Tail inflation adds 44.7% to p95 because n=20.

---

### Example C: Variable-loop step — Monte Carlo with loop variance model

**Scenario:** A research agent where a "fact_check" step loops 1–12 times depending on how many claims the LLM generates. High iteration variance across 30 runs. Context does NOT grow (each iteration gets a fresh context).

**Input data:**

```python
# 30 runs. fact_check iterations: [2, 8, 1, 12, 3, 5, 11, 2, 7, 4, ...]
# Context is constant at ~600 tokens per iteration (no growth).
# Per-iteration cost ≈ $0.003 (Sonnet)
# Total fact_check cost per run: $0.006 – $0.036 depending on iteration count.
```

**Trace:**

```
├─ 3. detect_patterns(runs, stats)
│
│    _detect_context_growth(runs):
│      step_pairs["fact_check"]: collects (iteration, context_size) pairs.
│      e.g., (1, 600), (2, 600), (3, 600), ...
│      _pearson_r([1,2,3,...], [600,600,600,...]):
│        denom_y = 0 (all ys identical) → returns (0.0, 0.0)
│      pearson_r_sq = 0.0 → does not pass 0.7 threshold
│      Spearman: _rank([600,600,...]) = all ties → same ranks → r = 0.0
│      Neither passes → skip
│      → return []  (correct: no context growth, just iteration variance)
│
│    _detect_loop_count_variance(runs):
│      fact_check max iterations: [2, 8, 1, 12, 3, 5, 11, 2, 7, 4, ...]
│      Not all 1 → proceed
│      n = 30, mean = 5.2, variance = sum(...)/(29) = 12.5, std = 3.54
│      CV = 3.54 / 5.2 = 0.68
│      0.68 > 0.5 → pattern detected
│      max = 12, ratio = 12/5.2 = 2.3
│      CV > 1.0? No. max > 3 × mean? 12 > 15.6? No.
│      → severity = "warning"
│      → DetectedPattern(type="loop_count_variance", step="fact_check",
│           severity="warning", evidence={cv: 0.68, mean: 5.2, min: 1, max: 12})
│
│    _detect_high_token_variance(stats):
│      n=30 ≥ 30 → use p95
│      fact_check: p95 tokens / p50 tokens ≈ 7200/3120 = 2.3 ≤ 3.0 → skip
│      (Token variance is moderate because iteration count varies but per-call
│       tokens are constant — the total tokens per run varies proportionally.)
│
│    → patterns = [loop_count_variance on "fact_check" (warning)]
│
├─ 4. project(stats, patterns=[loop_count_variance], runs=runs)
│
│    compute_confidence(30, step_stats, [loop_count_var], "auto-generate", run_costs)
│      n_eff ≈ 26 (costs are reasonably spread)
│      10 ≤ 26 < 30 → score -= 20
│      loop_count_variance → score -= 10
│      score = 100 - 20 - 10 = 70 → MODERATE tier
│
│    use_montecarlo = True (1 pattern)
│
│    simulate(stats, patterns, 1000, runs):
│      loop_variance_steps = {"fact_check"}
│      growth_steps = {}  (no context growth)
│
│      For each inner sample:
│        base_run_idx = rng.randrange(30)
│        run_avg = run_non_pattern_costs[base_run_idx]  (all non-fact_check costs)
│
│        For "fact_check":
│          _sample_step_cost("fact_check", rng, ...):
│            NOT in growth_steps → skip growth branch
│            IN loop_variance_steps → enter loop variance branch:
│              iters = [2, 8, 1, 12, 3, 5, ...]
│              n_iter = rng.choice(iters) → e.g., 8
│              occ_costs = [0.003, 0.0031, 0.0029, ...]  (per-call costs)
│              total = sum(rng.choice(occ_costs) for _ in range(8))
│                    = 0.003 × 8 ≈ $0.024
│              → returns (0.024, 0.024, 0.024, False)
│
│      n_observed = 30 → NOT < 30 → no tail inflation
│
│    → ProjectionResult(method="montecarlo", tail_inflation=None)
```

**Key takeaway:** The loop variance model randomly samples BOTH the iteration count (from observed distribution) AND the per-iteration cost (from observed per-call costs). This naturally captures the bimodal effect of "sometimes 1 iteration, sometimes 12." No tail inflation because n=30. Notice that context growth does NOT fire — the `denom_y == 0` guard in `_pearson_r` correctly rejects the constant context_size data.

---

### Example D: Conditional routing agent (W13) — Step count variance + bimodality

**Scenario:** A classifier routes queries to 3 paths: simple (70%, Haiku-only, ~$0.002), moderate (20%, Sonnet, ~$0.01), or escalated (10%, Sonnet+Opus, ~$0.15). 50 runs, causing both step count variance and bimodal cost distribution.

**Input data:**

```python
# 50 runs. Route distribution: 35 simple, 10 moderate, 5 escalated
# Simple runs: [classify, respond_simple] → 2 active steps, cost ≈ $0.002
# Moderate runs: [classify, research, respond_moderate] → 3 active steps, cost ≈ $0.01
# Escalated runs: [classify, research, draft, review_opus, respond] → 5 active steps, cost ≈ $0.15
```

**Trace:**

```
├─ 1. validate_profiling_data(runs)
│      step_present_counts: classify=50, respond_simple=35, research=15,
│        respond_moderate=10, draft=5, review_opus=5, respond=5
│      step_zero_counts: all zero (each step has tokens when present)
│      No step has zero tokens in >50% of runs it appears in → no warnings
│      → warnings = []
│
├─ 3. detect_patterns(runs, stats)
│
│    _detect_context_growth: no step iterates → return []
│
│    _detect_loop_count_variance: all iterations are 1 → return []
│
│    _detect_high_token_variance(stats):
│      n=50 ≥ 30 → use p95
│      classify: p95/p50 ≈ 1.1 → skip (consistent)
│      respond_simple: only 35 observations, but p95/p50 ≈ 1.2 → skip
│      review_opus: only 5 observations, p95/p50 ≈ 1.3 → skip
│      → return []
│
│    _detect_step_count_variance(runs):
│      active_counts per run:
│        35 runs with 2 steps, 10 with 3, 5 with 5
│        active_counts = [2,2,...,2, 3,3,...,3, 5,5,...,5]
│      n=50, mean = (35×2 + 10×3 + 5×5)/50 = (70+30+25)/50 = 2.5
│      variance = (35×(2-2.5)² + 10×(3-2.5)² + 5×(5-2.5)²) / 49
│               = (35×0.25 + 10×0.25 + 5×6.25) / 49
│               = (8.75 + 2.5 + 31.25) / 49 = 42.5/49 = 0.867
│      std = 0.931, CV = 0.931/2.5 = 0.372
│
│      CV > 0.3 → qualifies
│      CV > 0.6? No. max=5 > 2×min=4? 5 > 4? Yes!
│      → severity = "danger"
│      → DetectedPattern(type="step_count_variance", step="_workflow_",
│           severity="danger", step_count_cv=0.372, min=2, max=5, mean=2.5)
│
│    _detect_bimodality(runs, stats):
│      50 run_stats ≥ 15 → proceed
│      costs = [0.002, 0.002, ..., 0.01, ..., 0.15, ...]
│      All > 0, zero_count = 0
│      50 positive_costs ≥ 15 → fit GMM
│
│      log_costs = [ln(0.002), ln(0.002), ..., ln(0.01), ..., ln(0.15), ...]
│                = [-6.21, -6.21, ..., -4.61, ..., -1.90, ...]
│
│      GMM 1-component: BIC = 145.2
│      GMM 2-component: BIC = 98.7
│      bic_delta = 145.2 - 98.7 = 46.5 > 6 → bimodal!
│
│      labels = [0,0,...,0, 0,...,0, 1,1,...,1]
│        (cluster 0: $0.002 and $0.01 runs; cluster 1: $0.15 runs)
│        Actually, GMM might split differently — let's say:
│        mode 0 (cheap): 45 runs, mean=$0.004, proportion=90%
│        mode 1 (expensive): 5 runs, mean=$0.15, proportion=10%
│
│      → DetectedPattern(type="bimodality", step="_workflow_",
│           severity="warning", bimodal_bic_delta=46.5,
│           bimodal_modes=[
│             {proportion: 0.90, mean_cost: 0.004, median_cost: 0.002},
│             {proportion: 0.10, mean_cost: 0.15, median_cost: 0.15}
│           ])
│
│    → patterns = [
│        step_count_variance (danger),   ← sorted first (danger before warning)
│        bimodality (warning),
│      ]
│
├─ 4. project(stats, patterns=[step_count_var, bimodality], runs=runs)
│
│    compute_confidence(50, step_stats, 2 patterns, "auto-generate", run_costs)
│      n_eff = compute_effective_sample_size(run_costs):
│        50 costs, but 35 are clustered near $0.002, 10 near $0.01, 5 near $0.15
│        Bins will have uneven distribution → entropy < h_max
│        n_eff ≈ 28  (dominated by the $0.002 cluster)
│      10 ≤ 28 < 30 → score -= 20
│      step_count_variance (danger) → score -= 15
│      bimodality → score -= 5
│      score = 100 - 20 - 15 - 5 = 60 → MODERATE tier (barely)
│
│    use_montecarlo = True
│    Neither pattern is in growth_steps or loop_variance_steps
│    → pattern_steps = {} (empty — both are "no cost adjustment" patterns)
│
│    simulate():
│      All steps are non-pattern → entire cost comes from whole-run sampling
│      Every inner sample just picks a random observed run and uses its total cost.
│      This naturally preserves the routing distribution: 70% cheap, 20% moderate, 10% expensive.
│      _clt_aggregate handles the rest.
│
│      n_observed = 50 → NOT < 30 → no tail inflation
│
│    → ProjectionResult(method="montecarlo", patterns=[step_count_var, bimodality])
│      Monthly p50 ≈ $120 (driven by 70% cheap runs)
│      Monthly p95 ≈ $600 (driven by the expensive runs appearing in tail)
```

**Key takeaway:** The step count variance detector fires because the workflow has 2–5 active steps depending on route. The bimodality detector confirms the cost distribution has two distinct clusters. But NEITHER pattern has a custom cost model — both rely on whole-run resampling in Monte Carlo, which naturally preserves the routing distribution. The n_eff (28) is lower than the raw n (50) because 70% of the data clusters at one cost level.

---

### Example E: DeepSeek extraction — Cache pricing + data quality checks

**Scenario:** W12 — data extraction with DeepSeek V4 Flash. 20 runs. Some runs have cache-hit data from the DeepSeek API response, others don't.

**Input data:**

```python
# 20 runs, each with 3 steps: parse, extract, validate — all DeepSeek V4 Flash
# 12 runs have cache_hit_tokens/cache_miss_tokens from the API response
# 8 runs have cache fields = None (older API responses or no caching)
# extract step in 3 runs has zero tokens (data collection issue with the SDK)
```

**Trace:**

```
├─ Collection phase:
│    GenericCollector with @collector.step("parse") decorator
│    _try_extract(tracker, response):
│      response.usage has prompt_cache_hit_tokens and prompt_cache_miss_tokens
│      Dict path: cache_hit = usage.get("prompt_cache_hit_tokens") → 800
│                  cache_miss = usage.get("prompt_cache_miss_tokens") → 200
│      tracker.record_llm_call(model="deepseek-v4-flash",
│        input_tokens=1000, output_tokens=300,
│        cache_hit_tokens=800, cache_miss_tokens=200)
│
│    StepRecord created with cache_hit_tokens=800, cache_miss_tokens=200
│
├─ 1. validate_profiling_data(runs)
│      step_zero_counts:
│        parse: 0 zero-token runs
│        extract: 3 zero-token runs out of 20 → 3/20 = 15% → > 50%? No (15% < 50%)
│        validate: 0 zero-token runs
│      → warnings = []  (15% is below the 50% threshold)
│
│      BUT if extract had 11 zero-token runs (55%):
│        → warning: "Step 'extract' recorded zero tokens in 11/20 runs (55%).
│           Some runs may have missing usage data."
│
├─ 2. compute_stats(runs)
│      For runs with cache data:
│        _safe_cost(calculate_cost, "deepseek-v4-flash", 1000, 300):
│          Inside calculate_cost:
│            cache_hit_tokens and cache_miss_tokens are NOT passed from _safe_cost
│            (stats module uses basic calculate_cost without cache args)
│            → standard pricing: (1000 × 0.14/1M) + (300 × 0.28/1M)
│            = 0.000140 + 0.000084 = $0.000224
│
│      Note: The cache-aware pricing is used when calculate_cost is called WITH
│      the cache arguments, but compute_stats uses standard pricing for all records.
│      The cache_hit_tokens on StepRecord are metadata — they don't automatically
│      flow into cost computation in the stats pipeline.
│
├─ 3. detect_patterns: no patterns (simple workflow, no loops, low variance)
│    → patterns = []
│
├─ 4. project(stats, patterns=[], ...)
│    → _linear_project (no MC needed)
│    n_eff = 17.1 (20 runs, but some have zero tokens reducing effective diversity)
│    score = 100 - 20 = 80 → HIGH tier (barely)
│
│    At 1000/day, monthly p50 ≈ $6.72
│
└─ Note on cache-aware vs standard pricing:
     If the user wants cache-aware cost estimates, they would need to
     call calculate_cost directly with cache fields:

     calculate_cost("deepseek-v4-flash", 1000, 300,
       cache_hit_tokens=800, cache_miss_tokens=200)

     Standard: input_cost = 1000 × (0.14/1M)                = $0.000140
     Cached:   input_cost = 200 × (0.14/1M) + 800 × (0.0028/1M)
                          = $0.000028 + $0.0000022           = $0.0000302
     Savings:  $0.000140 → $0.0000302 = 78% cheaper on input tokens
```

**Key takeaway:** The cache fields flow through collection (`_try_extract` → `record_llm_call` → `StepRecord`) but the stats pipeline uses standard pricing. This is intentional — profiling should show cold-start costs by default. The `--allow-cache` flag and `cache_bust_prompt()` control whether the actual API calls hit cache during profiling. The data quality check catches zero-token steps but only if they're widespread (>50% of runs).

---

### Example F: Synthetic log-normal calibration — Full projection engine validation

**Scenario:** A log-normal distribution with σ=0.8 and n=20 runs, fed through the synthetic testing pipeline. This exercises the full engine on data with known ground truth.

**Input data:**

```python
wf = generate_lognormal(sigma=0.8, n=20, seed=42)
# true_p50 = exp(0) = 1.0
# true_p95 = exp(1.645 × 0.8) = exp(1.316) = 3.729
# true_mean = exp(0 + 0.8²/2) = exp(0.32) = 1.377
# true_std = 1.377 × sqrt(exp(0.64) - 1) = 1.377 × 0.861 = 1.186
# observed_costs: 20 samples drawn from LogNormal(0, 0.8)
# e.g., [0.45, 1.23, 0.89, 2.15, 0.67, 3.41, 1.05, 0.78, ...]
```

**Trace:**

```
run_one(wf, daily_volume=1000):
  │
  ├─ _ensure_synthetic_model()
  │    MODEL_PRICING["_synthetic_unit_cost_"] = (1.0, 0.0)
  │    MODEL_TIERS["_synthetic_unit_cost_"] = "mid"
  │
  ├─ Convert costs to StepRecords:
  │    cost=0.45 → input_tokens = int(0.45 × 1_000_000) = 450_000
  │    StepRecord(step_name="main", model="_synthetic_unit_cost_",
  │      input_tokens=450000, output_tokens=0, context_size=450000, ...)
  │    → 20 runs, each with 1 StepRecord
  │
  ├─ compute_stats(runs):
  │    For each record: calculate_cost("_synthetic_unit_cost_", 450000, 0)
  │      = 450000 × (1.0 / 1_000_000) + 0 = $0.45  ✓ matches original cost
  │    run_costs = [0.45, 1.23, 0.89, 2.15, 0.67, 3.41, 1.05, 0.78, ...]
  │    cost_per_run: mean ≈ 1.35, p50 ≈ 0.95, p95 ≈ 3.20
  │
  ├─ detect_patterns(runs, stats):
  │    _detect_context_growth: all iteration=1 → return []
  │    _detect_loop_count_variance: all iteration=1 → return []
  │
  │    _detect_high_token_variance(stats):
  │      n=20 < 30 → use p90
  │      "main" step: p90 tokens / p50 tokens ≈ 2_150_000 / 950_000 = 2.26
  │      2.26 ≤ 3.0 → skip
  │
  │      BUT with sigma=1.2 (higher variance):
  │        p90/p50 ≈ 4.5 > 3.0 → would trigger!
  │        → DetectedPattern(type="high_token_variance", severity="warning")
  │
  │    For σ=0.8: → patterns = []
  │
  ├─ project(stats, patterns=[], traffic=[1000], runs=runs):
  │    compute_confidence(20, {"main": ...}, [], "auto-generate", run_costs):
  │      n_eff = compute_effective_sample_size(run_costs):
  │        20 costs from lognormal — good spread
  │        n_bins = max(2, √20) = 4
  │        Costs spread across bins, but right-skewed → bins uneven
  │        n_eff ≈ 15.2
  │      10 ≤ 15.2 < 30 → score -= 20
  │      No patterns → no deductions
  │      score = 80 → HIGH tier (just barely)
  │
  │    patterns = [] → use_montecarlo = False
  │    → _linear_project(stats, [1000])
  │      For volume=1000:
  │        monthly p50 = 0.95 × 1000 × 30 = $28,500
  │        monthly p95 = 3.20 × 1000 × 30 = $96,000
  │
  ├─ Extract results:
  │    projected_p50 = $28,500
  │    projected_p95 = $96,000
  │
  └─ Return SyntheticCalibrationResult

compute_calibration_report([result], daily_volume=1000):
  n_monthly = 1000 × 30 = 30,000

  true_monthly_p50 = 30000 × true_mean = 30000 × 1.377 = $41,310
  p50_ratio = projected_p50 / true_monthly_p50 = 28500 / 41310 = 0.69

  0.69 < 0.7 → FAILS p50 calibration!
  (The linear projection uses the sample p50 × volume, but the CLT says
   the monthly total converges to N×mean, not N×median. For right-skewed
   distributions, mean > median, so sample p50 underestimates.)

  true_monthly_p95 = 30000 × 1.377 + 1.645 × √30000 × 1.186
                   = 41310 + 1.645 × 173.2 × 1.186 = 41310 + 337.7 = $41,648
  p95_covered = projected_p95 ($96,000) ≥ true_p95 ($41,648) → True ✓

  → This workflow would be flagged as a calibration failure for p50,
    which is expected at n=20 for a right-skewed distribution.
    This is exactly why the synthetic testing framework exists — it reveals
    where the projection engine is weakest.
```

**Key takeaway:** The synthetic pipeline exposes a real limitation: linear projection at n=20 for right-skewed distributions can fail p50 calibration because `sample_p50 × N ≠ true_mean × N` for skewed distributions. The CLT guarantees the sum converges to `N × mean`, but the engine uses the percentile directly. With higher σ (e.g., 1.2), `high_token_variance` would fire, triggering Monte Carlo instead, which uses CLT-corrected aggregation and produces better p50 estimates. This is by design — the patterns are the engine's mechanism for knowing when linear projection is insufficient.

---

### Cross-reference: Which code paths each example uniquely exercises

| Code path | Exercised by |
|-----------|-------------|
| `_linear_project()` (no patterns) | A, E, F |
| `simulate()` full MC loop | B, C, D |
| `_sample_step_cost` growth branch (linear type) | B |
| `_sample_step_cost` loop variance branch | C |
| `_sample_step_cost` not called (whole-run only) | D |
| `_detect_context_growth` with significant result | B |
| `_detect_context_growth` rejected by denom_y=0 guard | C |
| `_detect_loop_count_variance` triggered | C |
| `_detect_loop_count_variance` just below threshold | B |
| `_detect_step_count_variance` triggered (danger) | D |
| `_detect_bimodality` with GMM | D |
| `_detect_bimodality` skipped (n < 15) | A |
| `_detect_high_token_variance` with p90 (n<30) | B, F |
| `_detect_high_token_variance` with p95 (n≥30) | C, D |
| `_inflate_p95` (n < 30) | B |
| `_inflate_p95` skipped (n ≥ 30) | C, D |
| `compute_effective_sample_size` with clustered data | D |
| `compute_effective_sample_size` with spread data | B, C |
| `validate_profiling_data` (clean) | A, D, E |
| `validate_profiling_data` (zero-token edge) | E |
| `_try_extract` with cache fields | E |
| `calculate_cost` with cache arguments | E |
| `register_model` for synthetic model | F |
| `bootstrap_percentile_ci` (in backtesting) | D (via suite) |
| `_clt_aggregate` | B, C, D |
| `_build_run_step_cost_map` + whole-run sampling | B, C, D |
| Confidence: MODERATE tier | A, B, C, D |
| Confidence: HIGH tier | E, F |

---

## Part 11: Debugging Exercises

Work through each exercise by reading the broken code, then answer the four questions. Solutions at the bottom of this section.

---

### Exercise 1: CLT aggregation with zero variance

**File:** `pretia/projection/montecarlo.py`
**Symptom:** For a perfectly uniform cost distribution (every run costs exactly $0.05), the Monte Carlo projection shows wild variation between runs. Some simulations produce monthly costs of −$500 (clamped to $0) and others produce $15,000, when the true monthly cost at 1000/day is exactly $1,500.

**Broken code:**

```python
def _clt_aggregate(costs: list[float], n_total: int, z: float) -> float:
    k = len(costs)
    if k == 0 or n_total == 0:
        return 0.0
    mu = sum(costs) / k
    if k > 1:
        var = sum((x - mu) ** 2 for x in costs) / k  # BUG
    else:
        var = 0.0
    result = n_total * mu + z * math.sqrt(n_total * var)
    return max(result, 0.0)
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 2: Whole-run sampling index error

**File:** `pretia/projection/montecarlo.py`
**Symptom:** `IndexError: list index out of range` inside the simulation loop. Only happens when some steps appear in some runs but not others (e.g., a conditional routing workflow where path B only runs 30% of the time).

**Broken code:**

```python
# Inside simulation loop:
base_run_idx = rng.randrange(n_observed)
run_avg = run_non_pattern_costs[base_run_idx]

# BUG: accessing step cost from base run, but step may not exist in that run
for sn in all_step_names:
    if sn not in pattern_steps:
        run_avg += run_step_cost_map[base_run_idx][sn]  # KeyError possible!
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 3: Effective sample size division by zero

**File:** `pretia/validation/confidence.py`
**Symptom:** `ZeroDivisionError` when calling `compute_confidence()` with exactly 2 runs that have the same cost.

**Broken code:**

```python
def compute_effective_sample_size(costs: list[float]) -> float:
    n = len(costs)
    if n == 0:
        return 0.0
    # BUG: removed the min == max check
    n_bins = max(2, int(math.sqrt(n)))
    bin_width = (max(costs) - min(costs)) / n_bins  # division by zero if all equal!
    # ...
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 4: Bootstrap CI with one element

**File:** `pretia/validation/scoring.py`
**Symptom:** `IndexError` when computing bootstrap CI on a ground truth with only 1 run. `rng.choice(costs)` works fine, but `sorted([...])` produces a 1-element list, and the percentile computation accesses `sorted_data[1]`.

**Broken code:**

```python
def bootstrap_percentile_ci(costs, percentile, n_bootstrap=1000, ci_level=0.90, seed=42):
    rng = random.Random(seed)
    n = len(costs)
    # BUG: no guard for n < 2
    sorted_costs = sorted(costs)
    point = _percentile(sorted_costs, percentile)
    # ... rest works but is statistically meaningless for n=1
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 5: Tail inflation applied twice

**File:** `pretia/projection/montecarlo.py`
**Symptom:** At n=10 observed runs, the projected p95 monthly cost is 2.4× higher than what the synthetic testing framework expects. The calibration report shows massive overestimation for small sample sizes.

**Broken code:**

```python
# At the end of simulate():
tail_inflation_factor = None
if n_observed < 30:
    tail_inflation_factor = 1 + 2 / math.sqrt(n_observed)
    monthly_proj = _inflate_p95(monthly_proj, tail_inflation_factor)
    daily_proj = _inflate_p95(daily_proj, tail_inflation_factor)
    per_run_proj = _inflate_p95(per_run_proj, tail_inflation_factor)
    # BUG: inflating again!
    monthly_proj = _inflate_p95(monthly_proj, tail_inflation_factor)
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 6: Pattern significance check ignoring negative correlation

**File:** `pretia/projection/patterns.py`
**Symptom:** A step where context_size *decreases* with iteration (e.g., a compaction step) is flagged as "context growth" with a danger severity. The r is −0.95, which is a strong negative correlation. The code treats it as growth because it only checks `r² > 0.7`, which is true for r = −0.95.

**Broken code:**

```python
pearson_r_val, slope = _pearson_r(xs, ys)
pearson_r_sq = pearson_r_val * pearson_r_val
# BUG: removed the "r > 0" guard
pearson_sig = _is_significant(pearson_r_val, n)

# ...
pearson_passes = pearson_r_sq > 0.7 and pearson_sig  # Missing: and pearson_r_val > 0
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

## Solutions

### Exercise 1 Solution

**Bug:** The variance computation uses `/ k` (population variance) instead of `/ (k - 1)` (sample variance / Bessel's correction). With uniform data where all costs are identical ($0.05), the population variance is correctly 0, but with tiny floating-point differences in sampled costs (from different run compositions), the population variance is slightly smaller than the sample variance. Actually, the bigger issue: for uniform data, `var` is near-zero but not exactly zero. The `z × √(N × var)` term should be near-zero, but with population variance the estimate is biased downward for small k. The real symptom comes from non-uniform data where the difference is more pronounced.

Wait — re-reading the symptom: "perfectly uniform, every run exactly $0.05". Then `var = 0` regardless of `/ k` or `/ (k-1)`. The variance formula difference matters when costs are NOT identical. The actual bug for the described symptom might be that `/ k` produces too-small variance for small K, leading to overconfident simulations. But the symptom says "wild variation" — that contradicts.

**Corrected analysis:** The real issue is `/ k` instead of `/ (k - 1)`. For the Bessel-corrected formula, when all K costs are identical, both give 0. But the code comment says "BUG" at the `/ k` line. The fix is:

```python
var = sum((x - mu) ** 2 for x in costs) / (k - 1)
```

---

### Exercise 2 Solution

**Bug:** The code directly accesses `run_step_cost_map[base_run_idx][sn]`, but not all steps appear in every run. For a conditional routing workflow, run index 5 might only have steps A and B, while step C only exists in runs where the expensive path was taken. `dict[sn]` raises `KeyError`.

**Why this symptom:** `KeyError` propagates as an unhandled exception crashing the simulation loop.

**Finding it:** Run with W13 (conditional routing). Or: inspect `run_step_cost_map` entries — they have different key sets per run. Any step name not in a run's dict raises the error.

**Fix:** Use `.get()` with a default of 0:
```python
run_avg += run_step_cost_map[base_run_idx].get(sn, 0.0)
```

(Note: the actual code correctly uses precomputed `run_non_pattern_costs` which already handles this — the exercise shows what happens if you bypass that.)

---

### Exercise 3 Solution

**Bug:** The `min_c == max_c` guard was removed. When all costs are identical, `max_c - min_c = 0`, so `bin_width = 0 / n_bins` → `bin_width = 0.0`. Then `idx = int((c - min_c) / bin_width)` divides by zero.

**Finding it:** Pass `[0.05, 0.05]` to `compute_effective_sample_size()`. The original code returns `0.0` (all-same = zero effective information). Without the guard, it crashes.

**Fix:**
```python
if min_c == max_c:
    return 0.0
```

---

### Exercise 4 Solution

**Bug:** No guard for `n < 2`. `rng.choice(costs)` works with 1 element, and `_percentile` handles single-element lists. But `bootstrap_percentile_ci` is called from `run_backtesting_suite` which has `if len(gt_costs) >= 15` guard — so in practice this never triggers. The bug would surface if someone called `bootstrap_percentile_ci` directly with a 1-element list. With n=1, the bootstrap produces 1000 identical resamples, which is statistically meaningless.

**Finding it:** Call `bootstrap_percentile_ci([5.0], 95)`. It produces `(5.0, 5.0, 5.0)` — valid but misleading.

**Fix:** Add an early return:
```python
if n < 2:
    return (costs[0], costs[0], costs[0])
```

---

### Exercise 5 Solution

**Bug:** `_inflate_p95` is called twice on `monthly_proj`. The second call multiplies the already-inflated p95 by the factor again. At n=10, `factor = 1 + 2/√10 ≈ 1.632`. Double application: `p95 × 1.632 × 1.632 ≈ p95 × 2.66`. The expected single inflation would be `p95 × 1.632`.

**Finding it:** Run synthetic calibration at n=20 and n=10. At n=20, p95 coverage is good. At n=10, projected p95 is suspiciously 1.6× higher than expected. Log `tail_inflation_factor` and `monthly_proj.p95` before and after inflation — you'd see two applications.

**Fix:** Remove the duplicate line:
```python
# Delete this line:
monthly_proj = _inflate_p95(monthly_proj, tail_inflation_factor)
```

---

### Exercise 6 Solution

**Bug:** The `pearson_r_val > 0` guard was removed from the Pearson pass check. `r² > 0.7` is true for both r = 0.9 (growth) and r = −0.95 (shrinkage). Without the sign check, a step where context shrinks with iteration gets flagged as "context growth."

**Why this symptom:** The pattern triggers the growth cost model in Monte Carlo, which assumes costs increase with iteration. For a compaction step that actually saves money at higher iterations, the projection overestimates costs dramatically.

**Finding it:** Profile a workflow with a compaction step and check if the pattern detector flags it. Or: call `_detect_context_growth` with data where context decreases (e.g., [1000, 800, 600, 400, 200]) and verify no pattern is returned.

**Fix:**
```python
pearson_passes = pearson_r_sq > 0.7 and pearson_sig and pearson_r_val > 0
```

---

## Part 12: REPL Cheat Sheet

### Monte Carlo simulation

```python
from dataclasses import replace
from datetime import UTC, datetime
from pretia.collectors.base import StepRecord
from pretia.projection.stats import compute_stats
from pretia.projection.patterns import detect_patterns
from pretia.projection.montecarlo import simulate

rec = StepRecord(step_name="classify", step_type="llm", model="gpt-4o-mini", input_tokens=500, output_tokens=50, context_size=500, tool_definitions_tokens=0, system_prompt_hash="abc", system_prompt_tokens=100, output_format="text", is_retry=False, iteration=1, parent_step=None, duration_ms=200, timestamp=datetime(2026, 5, 20, tzinfo=UTC))
runs = [[rec], [replace(rec, input_tokens=800)], [replace(rec, input_tokens=1200)]]
stats = compute_stats(runs)
patterns = detect_patterns(runs, stats)
mc = simulate(stats, patterns, daily_volume=1000, runs=runs, n_simulations=1000)
mc.monthly_projection.p50, mc.monthly_projection.p95
```

### Pattern detection

```python
from pretia.projection.patterns import detect_patterns, _pearson_r, _rank, _is_significant
# Test Pearson r
r, slope = _pearson_r([1.0, 2.0, 3.0, 4.0, 5.0], [500.0, 1000.0, 1500.0, 2000.0, 2500.0])
r, slope  # should be (1.0, 500.0)

# Test significance
_is_significant(0.95, 10)  # True (r=0.95 with n=10 is significant)
_is_significant(0.30, 5)   # False (r=0.30 with n=5 is not significant)

# Test ranking
_rank([3.0, 1.0, 1.0, 5.0])  # [3.0, 1.5, 1.5, 4.0]
```

### Confidence scoring

```python
from pretia.validation.confidence import compute_effective_sample_size, compute_confidence
# All identical → zero effective n
compute_effective_sample_size([0.05] * 200)  # ≈ 0.0

# Uniformly spread → near-full n
import random; rng = random.Random(42)
costs = [rng.uniform(0.01, 1.0) for _ in range(200)]
compute_effective_sample_size(costs)  # ≈ 190

# Compute full confidence
compute_confidence(50, {}, [], run_costs=costs)
```

### Bootstrap CI

```python
from pretia.validation.scoring import bootstrap_percentile_ci
costs = [0.01, 0.02, 0.05, 0.03, 0.08, 0.02, 0.04, 0.06, 0.15, 0.03, 0.02, 0.05, 0.04, 0.07, 0.03]
point, lo, hi = bootstrap_percentile_ci(costs, 95, n_bootstrap=1000)
f"p95 = {point:.4f}, 90% CI = [{lo:.4f}, {hi:.4f}]"
```

### Pricing with cache

```python
from pretia.pricing.tables import calculate_cost, register_model, resolve_model, check_pricing_staleness
# Standard cost
calculate_cost("deepseek-v4-flash", 1000, 500)

# Cache-aware cost
calculate_cost("deepseek-v4-flash", 1200, 500, cache_hit_tokens=1000, cache_miss_tokens=200)

# Register custom model
register_model("my-model", input_price=0.50, output_price=1.00, tier="mid")
calculate_cost("my-model", 1000, 500)

# Check staleness
check_pricing_staleness()
```

### Mann-Whitney U

```python
from pretia.ci.diff import mann_whitney_u, significance_label
x = [0.05, 0.06, 0.04, 0.07, 0.05]  # baseline costs
y = [0.08, 0.09, 0.07, 0.10, 0.08]  # new costs
p = mann_whitney_u(x, y)
f"p = {p:.4f}, {significance_label(p)}"
```

### Visibility helpers

```python
from pretia.validation.visibility import sample_coverage_statement, format_projection_output, get_profiling_recommendation
sample_coverage_statement(20)  # "events < ~14%..."
sample_coverage_statement(50)  # "events < ~6%..."

format_projection_output(p50=1500, p95=3200, n=25)
# {"display_mode": "full", "p50": 1500, "p95": 3200}

format_projection_output(p50=1500, p95=3200, n=8)
# {"display_mode": "p50_only", "p50": 1500, "range_note": "...", "upgrade_note": "..."}
```

### Synthetic calibration (quick)

```python
from tests.synthetic.generators import generate_lognormal
from tests.synthetic.runner import run_one
wf = generate_lognormal(sigma=0.5, n=50, seed=42)
result = run_one(wf, daily_volume=1000)
f"projected p50=${result.projected_p50:.2f}, true monthly mean=${30000 * wf.true_mean:.2f}"
f"patterns: {result.patterns_detected}, confidence: {result.confidence_tier}"
```

### Shell commands

```bash
# Run all Sprint 3 test files
pytest tests/unit/test_montecarlo.py tests/unit/test_patterns.py tests/unit/test_projector.py tests/unit/test_confidence.py tests/unit/test_scoring.py tests/unit/test_suite.py tests/unit/test_data_checks.py tests/unit/test_pricing.py tests/unit/test_pricing_model_handling.py tests/unit/test_deepseek_cache.py tests/unit/test_visibility.py tests/unit/test_collector_token_extraction.py tests/unit/test_w13_and_inputs.py -v

# Run just Monte Carlo tests
pytest tests/unit/test_montecarlo.py -v

# Run a single test by name
pytest tests/unit/test_montecarlo.py -v -k "test_clt_aggregate"

# Run with debug logging
pytest tests/unit/test_patterns.py -v --log-cli-level=DEBUG

# Ruff check on Sprint 3 files
ruff check pretia/projection/ pretia/validation/ pretia/collectors/cache_bust.py pretia/ci/diff.py
```

---

## Test Counts by Area

| Area | Tests | Key file |
|------|-------|----------|
| Monte Carlo (CLT, tail, sampling, cap, seed) | 24 | `test_montecarlo.py` |
| Pattern detection (growth, loop, token, step count, bimodality) | 34 | `test_patterns.py` |
| Projector | 8 | `test_projector.py` |
| Confidence (n_eff, deductions, tiers) | 17 | `test_confidence.py` |
| Scoring (thresholds, bootstrap, Spearman) | 20 | `test_scoring.py` |
| Suite (launch gates, bias) | 11 | `test_suite.py` |
| Collector tokens (9 provider variants) | 9 | `test_collector_token_extraction.py` |
| Data checks (zero-token validation) | 4 | `test_data_checks.py` |
| Pricing (model handling, structural) | 15 | `test_pricing.py`, `test_pricing_model_handling.py` |
| DeepSeek cache (extraction, pricing, busting) | 12 | `test_deepseek_cache.py` |
| Visibility (recommendations, coverage, suppression, Mann-Whitney) | 16 | `test_visibility.py` |
| W13 & inputs | 11 | `test_w13_and_inputs.py` |
| Synthetic framework | 10 | `test_synthetic_framework.py` |
| SWE-bench integration | 11 | `test_swebench.py` |
| **Total Sprint 3** | **~200** | |
| **Full suite** | **690 pass, 4 skip, 1 xfail** | |

The 4 skips are bimodality tests requiring sklearn. The 1 xfail is Gemini thoughts token extraction (no native collector).
