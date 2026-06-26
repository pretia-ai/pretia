# Pretia System Architecture Guide

Complete architecture reference for the Pretia codebase as of Sprint 3. Covers every layer from data collection through validated cost projections — how each module works, how they connect, where data transforms, and where to look when something breaks.

For sprint-specific deep dives with function-level documentation, worked examples, and debugging exercises:
- [Sprint 1 Code Guide](sprint-01-code-guide.md) — collectors, pricing, store, runner, CLI, inputs
- [Sprint 2 Code Guide](sprint-02-code-guide.md) — stats, patterns, projector, Langfuse, OpenAI/Qwen collectors, CI
- [Sprint 3 Code Guide](sprint-03-code-guide.md) — Monte Carlo fixes, new detectors, validation, synthetic testing

---

## Table of Contents

- [1. The Big Picture](#1-the-big-picture)
- [2. Directory Map](#2-directory-map)
- [3. Layer 1 — Data Collection](#3-layer-1--data-collection)
- [4. Layer 2 — Pricing Engine](#4-layer-2--pricing-engine)
- [5. Layer 3 — Statistics & Pattern Detection](#5-layer-3--statistics--pattern-detection)
- [6. Layer 4 — Projection Engine](#6-layer-4--projection-engine)
- [7. Layer 5 — Confidence & Validation](#7-layer-5--confidence--validation)
- [8. Layer 6 — CI, Baselines & Diffing](#8-layer-6--ci-baselines--diffing)
- [9. Layer 7 — Orchestration & CLI](#9-layer-7--orchestration--cli)
- [10. Layer 8 — Persistence & Reporting](#10-layer-8--persistence--reporting)
- [11. Layer 9 — Testing & Validation Infrastructure](#11-layer-9--testing--validation-infrastructure)
- [12. Cross-Cutting Concerns](#12-cross-cutting-concerns)
- [13. Full Pipeline Traces](#13-full-pipeline-traces)
- [14. Dependency Graph](#14-dependency-graph)
- [15. Stubbed Modules](#15-stubbed-modules)
- [16. Known Limitations](#16-known-limitations)

---

## 1. The Big Picture

Pretia answers one question: **"How much will this AI agent workflow cost at production scale?"**

It does this by:
1. **Profiling** the workflow N times with diverse inputs
2. **Detecting** statistical patterns (context growth, loop variance, bimodality) that make simple averages misleading
3. **Projecting** to monthly volumes using either linear scaling (stable workflows) or Monte Carlo simulation (complex workflows)
4. **Validating** the projection with confidence scoring, effective sample size, and a multi-layer calibration framework

### System overview — data flow from CLI to terminal

```
 USER RUNS:  pretia profile run workflow.py --auto-generate 50
 ──────────────────────────────────────────────────────────────────

 ┌─── LAYER 7: ORCHESTRATION ──────────────────────────────────────┐
 │                                                                  │
 │  cli.py → ProfileRunner.run()                                    │
 │     │                                                            │
 │     ├─ _load_workflow()     importlib dynamic import             │
 │     ├─ _select_collector()  auto-detect framework                │
 │     └─ _resolve_inputs()    select + generate inputs             │
 └──────────────┬──────────────────────────────────────────────────┘
                │
 ┌──────────────▼── LAYER 1: COLLECTION ───────────────────────────┐
 │                                                                  │
 │  collector.collect(workflow, inputs)                              │
 │     │                                                            │
 │     ├─ GenericCollector    @step() decorator / async with        │
 │     ├─ LangGraphCollector  BaseCallbackHandler injection         │
 │     ├─ OpenAIAgentsCollector  RunHooksBase subclass              │
 │     └─ QwenAgentCollector  _InstrumentedChatModel proxy          │
 │                                                                  │
 │  → list[list[StepRecord]]  (50 runs × N steps each)             │
 └──────────────┬──────────────────────────────────────────────────┘
                │
 ┌──────────────▼── LAYER 2: PRICING ──────────────────────────────┐
 │                                                                  │
 │  For every StepRecord:                                           │
 │    calculate_cost(model, in_tok, out_tok)                        │
 │      resolve_model() → get_model_pricing() → multiply → round   │
 │    (cache-aware pricing for DeepSeek when cache fields present)  │
 │                                                                  │
 └──────────────┬──────────────────────────────────────────────────┘
                │
 ┌──────────────▼── LAYER 3: ANALYSIS ─────────────────────────────┐
 │                                                                  │
 │  validate_profiling_data(runs)  → warnings (zero-token checks)  │
 │  compute_stats(runs)            → ProfilingStats                 │
 │  detect_patterns(runs, stats)   → list[DetectedPattern]          │
 │     ├─ context_growth      dual Pearson+Spearman, p<0.05        │
 │     ├─ loop_count_variance CV > 0.5                              │
 │     ├─ high_token_variance p90/p50 or p95/p50 > 3               │
 │     ├─ step_count_variance CV > 0.3 (routing workflows)          │
 │     └─ bimodality          GMM BIC delta > 6 (sklearn optional)  │
 │                                                                  │
 └──────────────┬──────────────────────────────────────────────────┘
                │
 ┌──────────────▼── LAYER 4: PROJECTION ───────────────────────────┐
 │                                                                  │
 │  project(stats, patterns, runs)                                  │
 │     │                                                            │
 │     ├─ No patterns → linear: cost_per_run × volume × 30         │
 │     │                                                            │
 │     └─ Patterns → Monte Carlo (10K simulations):                 │
 │           ├─ Whole-run sampling (preserves step correlation)     │
 │           ├─ Pattern-step cost models:                           │
 │           │    context_growth → linear+log / power-law avg       │
 │           │    loop_variance  → sample iters × sample costs      │
 │           ├─ CLT aggregation: Nμ̂ + z√(Nσ̂²)                     │
 │           ├─ Tail inflation (if n < 30)                          │
 │           └─ Extrapolation cap at max observed iteration         │
 │                                                                  │
 │  compute_confidence(n, steps, patterns, run_costs)               │
 │     └─ n_eff (entropy-based) → deductions → tier                │
 │                                                                  │
 │  → ProjectionResult                                              │
 └──────────────┬──────────────────────────────────────────────────┘
                │
 ┌──────────────▼── LAYER 8: OUTPUT ───────────────────────────────┐
 │                                                                  │
 │  ProfileStore.save(session)  → .pretia/workflow_YYYYMMDD.json │
 │  format_cli_report(session)  → Rich tables + panels → terminal  │
 │  _auto_diff_baseline()       → one-line diff if baseline exists  │
 │  Visibility warnings + profiling recommendations                 │
 │                                                                  │
 └─────────────────────────────────────────────────────────────────┘
```

---

## 2. Directory Map

```
pretia/
├── __init__.py              Public API: ProfileRunner, StepRecord, __version__
├── cli.py                   Click CLI: 8 commands (profile run, report, analyze, ...)
├── runner.py                ProfileRunner — full pipeline orchestrator
├── store.py                 ProfilingSession persistence as JSON
│
├── collectors/
│   ├── __init__.py          Lazy imports via __getattr__ for optional deps
│   ├── base.py              StepRecord (17 fields) + BaseCollector ABC
│   ├── generic.py           Manual instrumentation: StepTracker + _try_extract
│   ├── langgraph.py         LangGraph: PretiaCallbackHandler
│   ├── openai_agents.py     OpenAI Agents: PretiaRunHooks + fallback
│   ├── qwen_agent.py        Qwen-Agent: _InstrumentedChatModel proxy
│   └── cache_bust.py        DeepSeek cache-busting utility
│
├── inputs/
│   ├── selector.py          Input mode priority ladder
│   ├── generator.py         LLM-powered synthetic input generation
│   ├── importer.py          Langfuse trace import + conversion
│   └── schema.py            [STUB] Input schema extraction
│
├── pricing/
│   ├── __init__.py          Re-exports: calculate_cost, resolve_model, register_model, ...
│   └── tables.py            MODEL_PRICING (28 models), aliases, tiers, cache pricing
│
├── projection/
│   ├── stats.py             compute_stats() → ProfilingStats (PercentileStats per metric)
│   ├── patterns.py          detect_patterns() → 5 detectors → list[DetectedPattern]
│   ├── projector.py         project() → linear or Monte Carlo → ProjectionResult
│   └── montecarlo.py        simulate() → CLT-corrected MC with growth models
│
├── validation/
│   ├── confidence.py        compute_confidence() → n_eff + deductions → ConfidenceResult
│   ├── scoring.py           score_projection() → CalibrationScore (5 metrics)
│   ├── suite.py             run_backtesting_suite() → hard/soft gates → launch decision
│   ├── data_checks.py       validate_profiling_data() → zero-token warnings
│   ├── visibility.py        Recommendations, input stats, coverage statements
│   └── validate_cmd.py      pretia validate: 20-vs-100 comparison
│
├── ci/
│   ├── baseline.py          Baseline create/save/load (per-step + monthly snapshots)
│   ├── diff.py              diff_baseline() + Mann-Whitney U + format_diff_report
│   └── report.py            format_cli_report() → Rich renderables (tables, panels)
│
├── recommend/               [PARTIALLY STUBBED — Sprint 4]
│   ├── prompts.py           Framework-specific implementation prompts (implemented)
│   ├── verify.py            Compare old/new profiles for applied recommendations (implemented)
│   ├── heuristics.py        [STUB] Rule-based recommendations
│   ├── classifier.py        [STUB] ML-powered model swap recommendations
│   └── rules.py             [STUB] Recommendation type definitions
│
├── report/                  [STUB — Sprint 6]
│   ├── renderer.py          [STUB] Jinja2 + inline SVG → HTML report
│   ├── charts.py            [STUB] Inline SVG chart generation
│   └── graph.py             [STUB] Workflow graph rendering
│
├── graph/                   [STUB — Sprint 6]
│   ├── extractor.py         [STUB] DAG extraction from frameworks
│   ├── layout.py            [STUB] Graph node positioning
│   ├── colorizer.py         [STUB] Cost-based node coloring
│   └── transform.py         [STUB] Apply recommendations to graph
│
└── ui/                      [STUB — Sprint 6]
    ├── app.py               [STUB] FastAPI + React bundle on :7100
    └── ws.py                [STUB] WebSocket for live profiling progress

tests/
├── unit/                    ~690 tests across 30+ files
├── integration/             Real LLM calls (costs money, CI-excluded)
├── backtesting/
│   ├── configs.py           13 workflow configs (10 active, 3 excluded)
│   ├── inputs/              500 inputs per workflow + skewed variants
│   ├── workflows/           Full LangGraph implementations (W1–W13)
│   └── run_backtesting.py   Execute the full backtesting suite
├── synthetic/
│   ├── generators.py        5 distribution generators → 520+ workflows
│   ├── runner.py            Feed synthetic data through projection engine
│   ├── calibration.py       Measure calibration against known truth
│   └── swebench/            SWE-bench trajectory parsing + calibration
└── conftest.py              Shared fixtures: sample_record, etc.
```

---

## 3. Layer 1 — Data Collection

### Core data structure: StepRecord

Every collector produces `list[list[StepRecord]]` — one inner list per profiling run. StepRecord is the universal currency of the codebase.

```python
@dataclass(frozen=True, slots=True)
class StepRecord:
    step_name: str                # "classify", "generate_response", etc.
    step_type: str                # "llm" | "tool" | "retrieval"
    model: str                    # "claude-sonnet-4-6", "gpt-4o", etc.
    input_tokens: int             # prompt tokens
    output_tokens: int            # completion tokens
    context_size: int             # total context including history
    tool_definitions_tokens: int  # tokens consumed by tool schemas
    system_prompt_hash: str       # SHA-256 hash of system prompt
    system_prompt_tokens: int     # estimated system prompt tokens
    output_format: str            # "json" | "text" | "code"
    is_retry: bool                # whether this call was a retry
    iteration: int                # 1-indexed loop counter for this step
    parent_step: str | None       # parent step name (for nested workflows)
    duration_ms: int              # wall-clock duration
    timestamp: datetime           # when the call completed
    cache_hit_tokens: int | None  # DeepSeek prompt cache hits (Sprint 3)
    cache_miss_tokens: int | None # DeepSeek prompt cache misses (Sprint 3)
```

**Invariants enforced on construction** (`__post_init__`):
- `step_type` must be `"llm"`, `"tool"`, or `"retrieval"`
- `output_format` must be `"json"`, `"text"`, or `"code"`
- `input_tokens`, `output_tokens`, `context_size`, `duration_ms` must be ≥ 0
- `iteration` must be ≥ 1

**Frozen** — immutable after creation. Serializable via `to_dict()` (timestamps → ISO 8601) and `from_dict()` (re-validates on construction).

### Four collectors

All implement `BaseCollector.collect(workflow, inputs) -> list[list[StepRecord]]`.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        COLLECTOR ARCHITECTURE                                │
│                                                                              │
│  GenericCollector                                                            │
│  ────────────────                                                            │
│  User code:  async with collector.step("name") as s:                         │
│                  s.record_llm_call(model=..., input_tokens=..., ...)          │
│                                                                              │
│  OR decorator: @collector.step("name")                                       │
│                async def my_step(input):                                     │
│                    return await llm.call(...)  # _try_extract auto-extracts  │
│                                                                              │
│  Flow: step() → StepTracker.__aenter__ → record_llm_call → __aexit__        │
│        → StepRecord validated → appended to _current_run                     │
│                                                                              │
│  _try_extract handles: OpenAI (.usage.prompt_tokens), Anthropic              │
│    (.usage.input_tokens), dict-style, DeepSeek cache fields                  │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  LangGraphCollector                                                          │
│  ─────────────────                                                           │
│  Auto-detect: hasattr(workflow, "ainvoke") AND hasattr(workflow, "nodes")     │
│                                                                              │
│  Injects PretiaCallbackHandler into LangChain's callback system.          │
│  Pairs start/end events by run_id UUID:                                      │
│    on_chat_model_start(run_id) → _inflight[run_id] = {model, start_ns, ...} │
│    on_llm_end(run_id)          → pop _inflight → extract tokens → StepRecord │
│    on_tool_start/on_tool_end   → tool StepRecord (zero tokens)               │
│                                                                              │
│  Token extraction: response.llm_output["token_usage"]["prompt_tokens"]       │
│  Output format: _detect_output_format(text) → "json"/"code"/"text"           │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  OpenAIAgentsCollector                                                       │
│  ────────────────────                                                        │
│  Auto-detect: hasattr(workflow, "name") AND hasattr(workflow, "instructions") │
│                                                                              │
│  Subclasses RunHooksBase. Every hook method wrapped in try/except.           │
│  Pairs start/end by agent name (LLM) or tool name (tool).                   │
│    on_llm_start → _inflight_llm[agent_name]                                 │
│    on_llm_end   → pop → response.usage.input_tokens → StepRecord            │
│    on_tool_start/end → _inflight_tool[tool_name]                             │
│    on_handoff → tracks agent transitions                                     │
│                                                                              │
│  Fallback: _build_fallback_steps(raw_responses) when hooks capture nothing   │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│  QwenAgentCollector                                                          │
│  ─────────────────                                                           │
│  Auto-detect: hasattr(workflow, "run") AND hasattr(workflow, "llm")          │
│               AND hasattr(workflow, "system_message")                         │
│                                                                              │
│  Wraps agent.llm with _InstrumentedChatModel proxy. Intercepts chat() calls. │
│  Extracts usage from:                                                        │
│    - OpenAI format: response.usage                                           │
│    - DashScope format: Message.extra["model_service_info"]                   │
│  Falls back to len(text)//4 character-count estimation.                      │
│  Restores original .llm after each run.                                      │
└──────────────────────────────────────────────────────────────────────────────┘
```

Optional-dep collectors use **lazy imports** via `__getattr__` in `collectors/__init__.py`. Importing `pretia.collectors.LangGraphCollector` when `langgraph` isn't installed raises `ImportError` — but only when you actually access the name, not when you `import pretia`.

### Cache busting (`cache_bust.py`)

DeepSeek's server-side cache prices cache-hit tokens at 2% of standard rate (50× discount). During profiling, we want cold-start costs.

```python
needs_cache_busting("deepseek-v4-flash")  → True
cache_bust_prompt("You are a helpful agent", run_id="abc123")
  → "You are a helpful agent\n<!-- profiling-run-abc123 -->"
```

Controlled by `ProfileRunner.cache_mode` ("cold" default / "warm" with `--allow-cache`).

### Input resolution (`inputs/`)

Priority ladder in `select_input_mode()`:

```
explicit_inputs → inputs_file → single_input → from_langfuse → auto_generate → _auto_detect
                                                                                     │
                                              ┌──────────────────────────────────────┘
                                              │
                                     LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY set?
                                        │                        │
                                       YES                      NO
                                        │                        │
                                   mode="langfuse"      ANTHROPIC_API_KEY or OPENAI_API_KEY?
                                                            │                │
                                                           YES              NO
                                                            │                │
                                                    mode="auto-generate"  mode="estimate"
```

`generate_inputs()` uses a cheap LLM (default Haiku) to create N diverse inputs targeting 60% typical / 20% edge / 20% adversarial distribution. `_parse_response()` strips preamble lines and numbered prefixes.

`importer.py` fetches Langfuse traces, converts them to `list[list[StepRecord]]` for analysis without re-executing the workflow. Filters EVENT observations. Resets iteration counter per trace. Detects "retrieval" step type via `"retriev"` substring matching.

---

## 4. Layer 2 — Pricing Engine

### Files

| File | Purpose |
|------|---------|
| `pretia/pricing/tables.py` | All pricing data, cost calculation, model resolution |
| `pretia/pricing/__init__.py` | Re-exports: `calculate_cost`, `resolve_model`, `register_model`, `UnrecognizedModelError`, ... |

### Data structures

```python
MODEL_PRICING: dict[str, tuple[float, float]]    # 28 models: canonical → (input $/MTok, output $/MTok)
MODEL_ALIASES: dict[str, str]                     # 24 aliases: short name → canonical
MODEL_TIERS: dict[str, str]                       # 28 entries: canonical → "frontier"/"mid"/"fast"
MODEL_CACHE_HIT_PRICING: dict[str, float]          # 4 DeepSeek models: canonical → cache-hit input $/MTok
PRICING_LAST_UPDATED: str                          # "2026-05-30" — for staleness warning
```

**7 providers covered:** Anthropic (3 models), OpenAI (6), Google Gemini (2), Meta/Together (2), Mistral (2), DeepSeek (4), Qwen/Alibaba (8).

### Cost calculation flow

```
calculate_cost("claude-opus-4", 1000, 500)
  │
  ├─ resolve_model("claude-opus-4"):
  │    "claude-opus-4" in MODEL_PRICING? → No
  │    "claude-opus-4" in MODEL_ALIASES? → Yes → "claude-opus-4-7"
  │    → "claude-opus-4-7"
  │
  ├─ get_model_pricing("claude-opus-4-7"):
  │    MODEL_PRICING["claude-opus-4-7"] = (5.00, 25.00)
  │    → (5.00/1M, 25.00/1M) = (0.000005, 0.000025)
  │
  ├─ cache_hit/miss tokens both None → standard path:
  │    cost = 1000 × 0.000005 + 500 × 0.000025
  │         = 0.005 + 0.0125 = 0.0175
  │
  └─ round(0.0175, 6) → $0.0175
```

**Cache-aware path** (DeepSeek only):
```
calculate_cost("deepseek-v4-flash", 1200, 300, cache_hit_tokens=1000, cache_miss_tokens=200)
  input_cost = 200 × (0.14/1M) + 1000 × (0.0028/1M) = $0.0000302
  output_cost = 300 × (0.28/1M) = $0.000084
  total = $0.0001142  (vs $0.000252 without caching — 55% cheaper)
```

**Error handling:** Unknown models raise `UnrecognizedModelError` (subclass of `ValueError`) with similar model suggestions via `_find_similar_models()` and a `register_model()` usage hint.

**Runtime registration:** `register_model(name, input_price, output_price, tier)` adds to `MODEL_PRICING` and `MODEL_TIERS`. Used by synthetic testing to register `_synthetic_unit_cost_`.

**Structural invariants:** Tests enforce that every key in `MODEL_PRICING` has an entry in `MODEL_TIERS` and vice versa.

---

## 5. Layer 3 — Statistics & Pattern Detection

### ProfilingStats — the bridge between raw data and projection

```
compute_stats(runs: list[list[StepRecord]]) → ProfilingStats
    │
    ├─ For each run, for each record:
    │    calculate_cost() → dollar cost per call
    │    Accumulate per-step records, per-run totals
    │
    ├─ For each step:
    │    compute_percentile_stats() on 7 metrics:
    │      input_tokens, output_tokens, total_tokens,
    │      cost, duration_ms, context_size, iterations_per_run
    │    → StepStats (one per unique step_name)
    │
    ├─ For each run:
    │    → RunStats (total_cost, total_tokens, step_count, duration_ms)
    │
    └─ Cross-run aggregation:
         compute_percentile_stats(run_costs) → cost_per_run PercentileStats
         compute_percentile_stats(run_tokens) → tokens_per_run PercentileStats
```

`PercentileStats` holds: min, max, mean, std (Bessel-corrected), p50, p75, p90, p95, p99. Uses linear interpolation for percentiles.

### Five pattern detectors

```
detect_patterns(runs, stats) → list[DetectedPattern]  (sorted: danger first)
    │
    ├─ _detect_context_growth(runs)
    │    Collects (iteration, context_size) pairs for iterating steps (≥ 5 points).
    │    Tests: Pearson r (linear) + Spearman ρ (any monotonic). Both must pass p < 0.05.
    │    If Pearson passes → "linear" growth.
    │    If only Spearman → "nonlinear" → _log_log_regression → α < 1 → sub_linear, else super_linear.
    │    Severity: r² > 0.85 → danger, else → warning.
    │
    ├─ _detect_loop_count_variance(runs)
    │    Groups max iteration per step per run. Computes CV.
    │    CV > 0.5 → pattern. CV > 1.0 or max > 3×mean → danger.
    │
    ├─ _detect_high_token_variance(stats)
    │    Uses p90 (if n < 30) or p95 (if n ≥ 30). Ratio to p50 > 3 → pattern.
    │    Ratio > 5 → danger.
    │
    ├─ _detect_step_count_variance(runs)
    │    Counts active steps (non-zero tokens) per run. CV > 0.3 → warning.
    │    CV > 0.6 or max > 2×min → danger. step_name = "_workflow_".
    │
    └─ _detect_bimodality(runs, stats)
         Requires ≥ 15 runs. Lazy-imports sklearn.
         Special case: zero-cost + positive-cost runs → inherently bimodal.
         Otherwise: 1-comp vs 2-comp GMM on log-costs. BIC delta > 6 → bimodal.
         Degrades gracefully if sklearn not installed (returns []).
```

**Detector → Monte Carlo mapping:**

| Detector | Has custom MC cost model? | What it causes |
|----------|:------------------------:|----------------|
| context_growth | YES | linear+log / power-law models in `_sample_step_cost` |
| loop_count_variance | YES | Sample iteration count × sample per-iteration costs |
| high_token_variance | NO | Triggers MC mode; whole-run resampling handles it |
| step_count_variance | NO | Triggers MC mode; enforces whole-run sampling |
| bimodality | NO | Per-mode reporting; confidence deduction |

`DetectedPattern` carries 20 fields — 5 core (type, step, severity, evidence, description) plus 15 optional metadata fields populated only by the relevant detector.

---

## 6. Layer 4 — Projection Engine

### Mode selection

```python
project(stats, patterns, traffic=[100, 1000, 10000], runs=runs)
```

- `len(patterns) == 0` → **linear**: `cost_per_run.percentile × volume × 30`
- `len(patterns) > 0 AND runs available` → **Monte Carlo**: 10,000 simulations
- `len(patterns) > 0 AND runs is None` → **fallback**: linear with warning

### Monte Carlo simulation — how it works

```
simulate(stats, patterns, daily_volume, runs, n_simulations=10000)
  │
  ├─ PRECOMPUTE (once):
  │    _precompute_step_data(runs) → per-step cost/iteration/occurrence arrays
  │    _precompute_growth_data(runs, growth_steps) → slope, base_context per growth step
  │    _build_run_step_cost_map(runs) → per-run, per-step cost dict (Fix 3)
  │    Merge pattern metadata into step_growth (growth_type, power_law_alpha, ...)
  │    run_non_pattern_costs = precomputed non-pattern total per observed run
  │    max_observed_iters per step (Fix 4)
  │    K = min(monthly_volume, 1000)
  │
  ├─ SIMULATION LOOP (10,000×):
  │    │
  │    ├─ INNER SAMPLING (K×):
  │    │    Pick random observed run (whole-run sampling, Fix 3)
  │    │    Non-pattern steps: use that run's actual costs (preserves correlation)
  │    │    Pattern steps: _sample_step_cost():
  │    │      ├─ Context growth: iterate 1..n_iter, compute cost per iteration
  │    │      │    linear type: avg(linear_model, log_model)
  │    │      │    nonlinear sub-linear: avg(log_model, power_model)
  │    │      │    nonlinear super-linear: avg(power_model, capped_power_model)
  │    │      │    Extrapolation capped at max_observed_iter (Fix 4)
  │    │      └─ Loop variance: rng.choice(observed_iters) × rng.choice(per_call_costs)
  │    │
  │    ├─ z = rng.gauss(0, 1)
  │    └─ monthly = _clt_aggregate(K costs, N_monthly, z)  (Fix 1)
  │         = N × μ̂ + z × √(N × σ̂²)    clamped to ≥ 0
  │
  ├─ POST-PROCESSING:
  │    Convergence check: |p95(first 9K) - p95(all 10K)| / p95(all) < 1%
  │    _build_percentile_projection(sim_monthly_costs) → PercentileProjection
  │    growth_model_delta: |linear_p95 - log_p95| / log_p95 × 100
  │
  └─ TAIL INFLATION (Fix 2):
       If n_observed < 30: factor = 1 + 2/√n
       Multiply p95 of monthly, daily, per-run projections by factor
```

### The four Monte Carlo fixes (Sprint 3)

| Fix | What was wrong | Solution | Impact |
|-----|---------------|----------|--------|
| **Fix 1 (CLT)** | 1 sample × N → variance = N²σ² (548× too wide at 10K/day) | K inner samples + CLT: `Nμ̂ + z√(Nσ̂²)` | Correct variance scaling |
| **Fix 2 (tail)** | Small samples underestimate tails | p95 × (1 + 2/√n) for n < 30 | Conservative at small n |
| **Fix 3 (correlation)** | Independent step sampling breaks inter-step cost correlation | Pick random observed run, use ALL its non-pattern costs | Preserves real cost structure |
| **Fix 4 (cap)** | Growth models extrapolate unboundedly past observed data | `effective_k = min(k, max_observed_iter)` | Prevents runaway projections |

### Output

`ProjectionResult` contains: method ("linear" or "montecarlo"), projections at each traffic volume, `ConfidenceResult`, warnings, detected patterns, and the raw `MonteCarloResult` (if MC was used).

---

## 7. Layer 5 — Confidence & Validation

### Confidence scoring

```
compute_confidence(sample_size, step_stats, patterns, input_source, run_costs)
  │
  ├─ compute_effective_sample_size(run_costs):
  │    Bins costs into √n bins → Shannon entropy H → n_eff = n × (H / H_max)
  │    200 identical costs → n_eff = 0
  │    200 uniformly spread → n_eff ≈ 190
  │
  ├─ Score starts at 100. Deductions:
  │    n_eff < 10 → −40 │ n_eff < 30 → −20 │ n_eff < 100 → −10
  │    Per-step CV > 1.0 → −15 (capped at −30 total)
  │    Per-step CV > 0.5 → −8 (capped at −30 total)
  │    context_growth/loop_variance/token_variance → −10 each
  │    step_count_variance → −5 (warning) / −15 (danger)
  │    bimodality → −5
  │
  ├─ Bonuses:
  │    Langfuse import → +15
  │    sample_size ≥ 200 → +10
  │
  └─ Clamp [0, 100] → tier:
       ≥ 80 → HIGH (display p50–p90, language "projected")
       ≥ 60 → MODERATE (p50–p95, "estimated")
       ≥ 40 → LOW (p25–p99, "estimated")
       < 40 → VERY_LOW (order of magnitude, "ballpark")
```

### Calibration scoring

`score_projection(projected, ground_truth)` compares across 5 metrics:

| Metric | PASS | WARN | FAIL |
|--------|------|------|------|
| **p50 ratio** | 0.7–2.0× | 0.33–0.7× or 2.0–3.0× | <0.33× or >3.0× |
| **p95 coverage** | ≥ threshold¹ | ≥ 60% | < 60% |
| **Range ratio** (p95/p50) | < threshold² | < 10× | ≥ 10× |
| **Top step correct** | matches | co-dominant (within 30%) | mismatched |
| **Step ranking** (Spearman ρ) | > 0.7 (4+ steps) | — | < 0.7 |

¹ 85% for simple, 75% for complex workflows. ² 3× for simple, 8× for complex.

Bootstrap CIs (`bootstrap_percentile_ci`, 1000 resamples, no numpy) allow projected values falling within the CI to auto-pass even if the point-estimate ratio would fail.

### Launch gate (backtesting suite)

```
Hard gates (ALL workflows must pass):
  ✓ p50 ratio in (0.7, 2.0)
  ✓ Top cost step correct

Soft gates (≥ 80% pass rate each):
  • p95 coverage ≥ threshold
  • Range ratio < threshold
  • Step ranking ρ > 0.7

overall_passed = hard_gates AND soft_gates

Directional bias check: if ≥ 80% of workflows skew one direction → diagnostic flag
```

---

## 8. Layer 6 — CI, Baselines & Diffing

### Baseline management (`ci/baseline.py`)

`create_baseline(session, traffic)` snapshots a profiling session into a `Baseline` object:
- Per-step: model, token p50/p95, cost p50/p95/mean, iteration stats, system prompt hash, output format, pattern flags
- Workflow-level: total monthly p50/p75/p90/p95, confidence tier, assumptions list
- Version "1.0" — `from_dict` validates version prefix

`save_baseline()` writes to `.pretia/baseline.json`. `load_baseline()` reads and deserializes.

### Diffing (`ci/diff.py`)

`diff_baseline(baseline, new_session, traffic)` produces a `DiffResult`:

```
DiffResult
  ├─ total_monthly_change: {"p50": +$12.30, "p95": +$45.00}
  ├─ total_monthly_pct_change: {"p50": +8%, "p95": +12%}
  ├─ step_diffs: per-step cost/token/iteration deltas + model change flags
  ├─ new_steps / removed_steps
  ├─ model_changes: [ModelChange(step, old_model, new_model, cost_impact)]
  ├─ pattern_changes: PatternChanges(new, resolved, unchanged)
  └─ summary: "Monthly cost increased 8%: $153 → $165 at 1,000/day"
```

### Significance testing

`mann_whitney_u(x, y)` — two-tailed p-value via normal approximation. All stdlib (Abramowitz & Stegun CDF). `significance_label(p)` → "significant" / "possibly significant" / "not significant".

---

## 9. Layer 7 — Orchestration & CLI

### ProfileRunner (`runner.py`)

The full pipeline orchestrator. Coordinates every layer:

```python
class ProfileRunner:
    def __init__(self, workflow_path, collector="auto", auto_generate=None,
                 single_input=None, inputs_file=None, from_langfuse=False,
                 langfuse_last_n=10, output_dir=".pretia", cache_mode="cold"):

    async def run(self) -> ProfilingSession:
        workflow, system_prompt = self._load_workflow()         # importlib dynamic import
        collector = self._select_collector(workflow)             # auto-detect framework
        selection, inputs = await self._resolve_inputs(prompt)   # input mode + generation
        runs = await collector.collect(workflow, inputs)          # LAYER 1
        data_warnings = validate_profiling_data(runs)            # LAYER 5 (data checks)
        cost_summary = _build_cost_summary(runs)                 # LAYER 2 (legacy format)
        profiling_stats = compute_stats(runs)                    # LAYER 3
        patterns = detect_patterns(runs, profiling_stats)        # LAYER 3
        projection = project(stats, patterns, runs=runs)         # LAYER 4
        session = ProfilingSession(... metadata={stats, patterns, projection, confidence})
        ProfileStore.save(session)                               # LAYER 8
        self._auto_diff_baseline(session)                        # LAYER 6 (if baseline exists)
        return session
```

**Collector auto-detection** (priority order):
1. `--collector langgraph/openai/qwen/generic` → explicit selection
2. `ainvoke` + `nodes` attributes → `LangGraphCollector`
3. `name` + `instructions` attributes → `OpenAIAgentsCollector`
4. `run` + `llm` + `system_message` attributes → `QwenAgentCollector`
5. Fallback → `GenericCollector`

**`analyze_langfuse()`** — alternate entry point that imports Langfuse traces without executing the workflow. Same analysis pipeline (stats → patterns → projection) but on imported trace data.

### CLI (`cli.py`)

8 commands via Click:

| Command | Function | What it does |
|---------|----------|-------------|
| `pretia profile run <workflow>` | `run()` | Full profiling pipeline |
| `pretia report <profile.json>` | `report_cmd()` | Render report from saved profile |
| `pretia analyze --from-langfuse` | `analyze_cmd()` | Analyze Langfuse traces without execution |
| `pretia baseline update <profile>` | `baseline_update()` | Save profile as cost baseline |
| `pretia diff <baseline> <profile>` | `diff_cmd()` | Compare baseline to new profile |
| `pretia validate <workflow>` | `validate_cmd()` | 20-vs-100 projection quality check |
| `pretia update-pricing` | `update_pricing_cmd()` | Pricing update instructions |
| `pretia ui` | (Sprint 6) | Launch local web UI on :7100 |

Key flags: `--auto-generate N`, `--input "..."`, `--inputs file.jsonl`, `--from-langfuse`, `--last N`, `--allow-cache`, `--traffic N`, `--threshold N`, `-v/--verbose`.

---

## 10. Layer 8 — Persistence & Reporting

### ProfilingSession & ProfileStore (`store.py`)

```python
@dataclass
class ProfilingSession:
    workflow_name: str           # "my_agent.py"
    workflow_hash: str           # SHA-256 first 12 chars
    profiled_at: datetime
    sample_size: int
    input_mode: str              # "auto-generate", "single", "langfuse", etc.
    runs: list[list[StepRecord]] # all raw data
    metadata: dict[str, Any]     # cost_summary, stats, patterns, projection, confidence
```

`ProfileStore.save()` writes to `.pretia/{stem}_{YYYYMMDD_HHMMSS}.json`. `_safe_name()` strips path to filename stem (`agents/v2/bot.py` → `"bot"`).

`ProfileStore.load()` deserializes via `ProfilingSession.from_dict()` → `StepRecord.from_dict()` for each record. Re-validates all invariants.

`ProfileStore.list_sessions()` globs `*.json`, sorted by mtime descending. `latest()` loads the newest.

### CLI Report (`ci/report.py`)

`format_cli_report(session, cost_summary=None, traffic=None)` produces Rich renderables:

```
┌─ Header Panel ──────────────────────────────────────────────────┐
│  workflow: my_agent.py  │  runs: 50  │  mode: auto-generate     │
└─────────────────────────────────────────────────────────────────┘

┌─ Cost Per Run ──────────────────────────────────────────────────┐
│  Mean   │  Median  │  p95    │  p99    │  Min    │  Max    │ Std│
│  $0.048 │  $0.044  │  $0.092 │  $0.120 │  $0.020 │  $0.150 │..│
└─────────────────────────────────────────────────────────────────┘

┌─ Step Breakdown (sorted by cost) ───────────────────────────────┐
│  Step     │ Model     │ Tier │ Mean   │ p95    │ Tokens │ Calls │
│  respond  │ claude-s… │ mid  │ $0.043 │ $0.085 │ 590/200│ 1.0  │
│  classify │ claude-h… │ fast │ $0.005 │ $0.006 │ 310/40 │ 1.0  │
└─────────────────────────────────────────────────────────────────┘

┌─ Monthly Projection ────────────────────────────────────────────┐
│  100/day:   Mean $144 │ p95 $276                                │
│  1,000/day: Mean $1,440 │ p95 $2,760                            │
│  10,000/day: Mean $14,400 │ p95 $27,600                         │
└─────────────────────────────────────────────────────────────────┘

┌─ Patterns ──────────────────────────────────────────────────────┐
│  ● Context growth at 'review' (r²=0.96, slope=400 tok/iter)    │
└─────────────────────────────────────────────────────────────────┘
```

`format_cost()` switches precision by magnitude: sub-cent → 4 decimals, $0.01–$999 → 2 decimals, $1000+ → 0 decimals with comma.

Supports both Sprint 2+ stats-based path and Sprint 1 legacy `cost_summary` fallback. The `report` command recomputes stats from stored runs if the `stats` key is missing.

---

## 11. Layer 9 — Testing & Validation Infrastructure

### Three-layer validation strategy

```
┌───────────────────────────────────────────────────────────────────┐
│  LAYER 1: Synthetic Distributions                                 │
│  520+ workflows with KNOWN ground truth (lognormal, bimodal,      │
│  Pareto, zero-inflated, uniform). $0 cost. Tests engine math.     │
│                                                                   │
│  generators.py → runner.py → calibration.py                       │
│    generate_lognormal(σ, n, seed) → SyntheticWorkflow             │
│    generate_bimodal(mixing, separation, n, seed)                  │
│    generate_pareto(α, n, seed)                                    │
│    generate_zero_inflated(trigger_prob, n, seed)                  │
│    generate_uniform(n, seed)                                      │
│                                                                   │
│  Trick: register _synthetic_unit_cost_ ($1/MTok input, $0 output) │
│         input_tokens = int(cost × 1M) → cost round-trips exactly  │
│                                                                   │
│  Calibration: p50_ratio should be in (0.7, 2.0)                  │
│               projected_p95 ≥ CLT-approximated true_p95           │
├───────────────────────────────────────────────────────────────────┤
│  LAYER 2: SWE-bench Trajectories                                  │
│  Real cost distributions from coding agent experiments. $0.       │
│  Validates on real-world skewed distributions.                     │
│                                                                   │
│  swebench/ → parse trajectory JSONL → group by repo →             │
│            convert to SyntheticWorkflow → same calibration flow    │
├───────────────────────────────────────────────────────────────────┤
│  LAYER 3: Real Workflow Backtesting                               │
│  10 workflows across 7 archetypes. 200–300 runs each. ~$850.     │
│  Validates the complete pipeline including collection.             │
│                                                                   │
│  configs.py defines BacktestConfig per workflow.                  │
│  suite.py implements hard + soft launch gates.                    │
│  run_backtesting.py executes the full protocol.                   │
│                                                                   │
│  W1-W12 cover: support, code-review, extraction, research,       │
│  sales (OpenAI, Gemini, mixed), Qwen, DeepSeek.                  │
│  W13 (new Sprint 3): conditional routing agent — tests            │
│  step_count_variance and bimodality detectors.                    │
└───────────────────────────────────────────────────────────────────┘
```

### Test suite

```
690 passed, 4 skipped, 1 xfailed in ~110 seconds

56 test files across:
  tests/unit/           30+ files, ~690 tests
  tests/integration/    Real LLM calls (CI-excluded)
  tests/backtesting/    10 workflow configs + 500 inputs each
  tests/synthetic/      520+ generated workflows

Skipped: 4 bimodality tests (require sklearn)
Xfailed: 1 Gemini thoughts token extraction (no native collector)
```

---

## 12. Cross-Cutting Concerns

### Error handling philosophy

| Layer | Strategy | Rationale |
|-------|----------|-----------|
| **Collectors** | Silent degradation. Unknown tokens → 0. Failed extraction → debug log. | Never crash mid-profiling. Lost data is recoverable; a crashed workflow run is wasted. |
| **Pricing** | Loud failure. Unknown model → `UnrecognizedModelError` with suggestions. | Every downstream number depends on correct pricing. A wrong price is worse than no price. |
| **Validation** | Warnings only. Zero-token steps, stale profiles, uniform inputs → log warnings. | Informative but non-blocking. The user decides whether to act. |
| **Projection** | Graceful degradation. Missing runs for MC → linear fallback. Small n → suppress p95. | Always produce *some* output. Annotate uncertainty instead of refusing to project. |
| **Hooks (OpenAI)** | Every method wrapped in `try/except Exception`. | A bug in Pretia must never crash the user's workflow execution. |

### No external dependencies for math

The entire projection engine, pattern detection, Monte Carlo simulation, bootstrap CIs, Mann-Whitney U test, Spearman/Pearson correlation, power-law regression, entropy computation, and CLT aggregation use only `math`, `random`, and `collections` from stdlib. **No numpy, scipy, or sklearn in the core pipeline.** sklearn is optional (bimodality detector only, lazy-imported, degrades gracefully).

### Backward compatibility

- All dataclass field additions use defaults (`None`, `field(default_factory=...)`) so existing constructor calls work
- `to_dict()` always includes new fields
- `from_dict()` uses `.get()` with defaults for missing keys in old JSON
- `format_cli_report` supports both Sprint 2+ stats and Sprint 1 legacy `cost_summary`
- `report` command recomputes stats from stored runs if `stats` key is missing

### Dependency management

```
Core (always installed):
  click, rich, jinja2

Optional extras:
  pip install pretia[langgraph]     # LangGraph + LangChain
  pip install pretia[openai]        # OpenAI Agents SDK
  pip install pretia[qwen]          # Qwen-Agent
  pip install pretia[ui]            # FastAPI + React bundle
  pip install pretia[validation]    # scipy + sklearn
  pip install pretia[backtesting]   # langchain-anthropic + full test deps

Dev:
  pip install pretia[dev]           # pytest, ruff, pyright, build
```

Lazy imports via `__getattr__` prevent `ImportError` when optional packages aren't installed. The error surfaces only when the specific collector is needed.

---

## 13. Full Pipeline Traces

### Trace A: `pretia profile run workflow.py --auto-generate 50`

```
cli.py:run(workflow_path, auto_generate=50)
  │
  ├─ ProfileRunner(workflow_path, auto_generate=50, cache_mode="cold")
  │
  └─ runner.run_sync() → asyncio.run(self.run())
       │
       ├─ _load_workflow()
       │    _load_workflow_module(path)           # importlib.util dynamic import
       │    _find_workflow(module)                 # scan graph/workflow/agent/app
       │    _extract_system_prompt(module)         # regex scan for "you are"/"your role"
       │    → (workflow_object, system_prompt)
       │
       ├─ _select_collector(workflow)
       │    Auto-detect: LangGraph? OpenAI? Qwen? → else GenericCollector
       │    → BaseCollector
       │
       ├─ _resolve_inputs(system_prompt)
       │    select_input_mode(auto_generate=50)   → InputSelection(mode="auto-generate")
       │    generate_inputs(system_prompt, n=50)
       │      _resolve_provider()                 → pick Anthropic/OpenAI SDK
       │      _call_anthropic/openai()            → raw LLM response
       │      _parse_response()                   → 50 clean input strings
       │    → (InputSelection, list of 50 inputs)
       │
       ├─ await collector.collect(workflow, inputs)
       │    50 runs × N steps each → list[list[StepRecord]]
       │    (Each StepRecord validated in __post_init__)
       │
       ├─ validate_profiling_data(runs)
       │    Check zero-token steps → log warnings
       │
       ├─ _build_cost_summary(runs)               # legacy Sprint 1 format
       │    calculate_cost() per record, aggregate per-step + per-run
       │
       ├─ compute_stats(runs)                      # Sprint 2 format
       │    → ProfilingStats (PercentileStats per metric per step)
       │
       ├─ detect_patterns(runs, stats)
       │    5 detectors → sorted list[DetectedPattern]
       │
       ├─ project(stats, patterns, runs=runs, input_source="auto-generate")
       │    compute_confidence(n, steps, patterns, run_costs)
       │      compute_effective_sample_size(costs) → n_eff
       │      deductions/bonuses → score → tier
       │    if patterns: simulate() → MonteCarloResult
       │    else: _linear_project()
       │    → ProjectionResult
       │
       ├─ ProfilingSession(metadata={cost_summary, stats, patterns, projection, confidence})
       │
       ├─ ProfileStore.save(session)
       │    → .pretia/workflow_20260601_143022.json
       │
       ├─ _auto_diff_baseline(session)
       │    If .pretia/baseline.json exists → diff_baseline() → summary
       │
       └─ Return to CLI → format_cli_report(session) → console.print()
```

### Trace B: `pretia analyze --from-langfuse --last 20`

```
cli.py:analyze_cmd(from_langfuse=True, last_n=20)
  │
  ├─ Validate LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY
  │
  ├─ create_langfuse_client()                      # env vars → LangfuseAPI
  │
  ├─ fetch_traces(client, last_n=20)
  │    client.trace.list(limit=20) → 20 trace summaries
  │    For each: client.trace.get(trace_id) → full trace with observations
  │    _parse_observation(obs) → LangfuseObservation per observation
  │    → list[LangfuseTrace] (each with list[LangfuseObservation])
  │
  ├─ traces_to_step_records(traces)
  │    For each trace:
  │      Reset iteration_counts
  │      Filter EVENT observations (_SKIP_TYPES)
  │      Classify: GENERATION → "llm", "retriev" in name → "retrieval", SPAN/TOOL → "tool"
  │      Build StepRecord per observation
  │    → list[list[StepRecord]]
  │
  ├─ compute_stats(runs) → ProfilingStats
  ├─ detect_patterns(runs, stats) → list[DetectedPattern]
  │
  ├─ ProfilingSession(input_mode="langfuse-analyze", metadata={stats, patterns, ...})
  ├─ ProfileStore.save(session)
  │
  └─ format_cli_report(session) → console.print()
```

### Trace C: `pretia diff baseline.json latest`

```
cli.py:diff_cmd(baseline_path, profile_path="latest")
  │
  ├─ load_baseline(baseline_path)
  │    Baseline.from_dict(json) → validates version "1.x"
  │
  ├─ ProfileStore.load(latest session)
  │    If no "stats" → recompute from runs
  │
  ├─ diff_baseline(baseline, new_session, traffic)
  │    create_baseline(new_session) → new_baseline for comparison
  │    Compute: total_monthly_change, step_diffs, model_changes, pattern_changes
  │    Generate summary string
  │    → DiffResult
  │
  ├─ format_diff_report(result) → terminal output
  │
  └─ If --threshold N: check p50_pct > N → sys.exit(1) for CI gate
```

### Trace D: `pretia validate workflow.py`

```
cli.py:validate_cmd(workflow_path, budget=10, small_n=20, large_n=100)
  │
  ├─ Confirm with user: "This will profile 120 times. ~$10. Proceed?"
  │
  ├─ run_validation(workflow_path, small_n=20, large_n=100):
  │    │
  │    ├─ ProfileRunner(auto_generate=20).run_sync()
  │    │    → small_session (20 runs)
  │    ├─ compute_stats(small_session.runs) → small_stats
  │    │
  │    ├─ ProfileRunner(auto_generate=100).run_sync()
  │    │    → large_session (100 runs)
  │    ├─ compute_stats(large_session.runs) → large_stats
  │    │
  │    ├─ score_projection(small_stats, large_stats)
  │    │    → CalibrationScore (p50 ratio, p95 coverage, ...)
  │    │
  │    ├─ convergence = |small_p50 - large_p50| / large_p50 × 100%
  │    │    ≤ 30% → "20 samples sufficient"
  │    │    > 30% → "High variance, use 50+ samples"
  │    │
  │    └─ → ValidateResult
  │
  └─ format_validate_report(result) → terminal output
       If verdict == "FAIL" → sys.exit(1)
```

---

## 14. Dependency Graph

### Import hierarchy (no circular dependencies)

```
pretia/__init__.py
  └── runner.py
        ├── collectors/base.py          (StepRecord, BaseCollector)
        ├── collectors/generic.py       (GenericCollector, StepTracker)
        ├── collectors/langgraph.py     [lazy] (LangGraphCollector)
        ├── collectors/openai_agents.py [lazy] (OpenAIAgentsCollector)
        ├── collectors/qwen_agent.py    [lazy] (QwenAgentCollector)
        ├── inputs/selector.py          (select_input_mode)
        ├── inputs/generator.py         (generate_inputs)
        ├── inputs/importer.py          [lazy] (Langfuse client + import)
        ├── pricing/tables.py           (calculate_cost, resolve_model)
        ├── projection/stats.py         (compute_stats → ProfilingStats)
        ├── projection/patterns.py      (detect_patterns → DetectedPattern)
        │     └── projection/stats.py   (compute_stats if stats not provided)
        ├── projection/projector.py     (project → ProjectionResult)
        │     ├── projection/montecarlo.py  (simulate)
        │     └── validation/confidence.py  (compute_confidence)
        ├── validation/data_checks.py   (validate_profiling_data)
        ├── store.py                    (ProfileStore, ProfilingSession)
        └── ci/baseline.py             [lazy] (auto-diff)

cli.py
  ├── runner.py                        (ProfileRunner)
  ├── ci/report.py                     (format_cli_report)
  ├── ci/baseline.py                   (create_baseline, load_baseline)
  ├── ci/diff.py                       (diff_baseline)
  ├── projection/stats.py              (compute_stats — for report recompute)
  ├── projection/patterns.py           (detect_patterns — for report recompute)
  ├── validation/validate_cmd.py       (run_validation)
  └── store.py                         (ProfileStore)
```

### What depends on what

```
StepRecord ← everything (collectors, stats, patterns, store, runner, report)
  No dependencies except stdlib

calculate_cost ← stats.py, montecarlo.py, importer.py, runner.py, baseline.py
  No dependencies except resolve_model → MODEL_PRICING/ALIASES

compute_stats ← runner.py, projector.py, cli.py, suite.py, validate_cmd.py
  Depends on: StepRecord, calculate_cost

detect_patterns ← runner.py, cli.py, suite.py
  Depends on: StepRecord, compute_stats (if stats not provided)

project ← runner.py
  Depends on: simulate, compute_confidence, ProfilingStats

simulate ← projector.py
  Depends on: StepRecord, calculate_cost, DetectedPattern, ProfilingStats
```

---

## 15. Stubbed Modules

These exist as docstring placeholders. They define intended scope but contain no implementation.

| Module | Sprint | Purpose |
|--------|--------|---------|
| `recommend/heuristics.py` | 4 | Rule-based recommendations: model swap, context compaction, iteration cap |
| `recommend/classifier.py` | 4 | ML-powered model swap (logistic regression on RouterBench) |
| `recommend/rules.py` | 4 | Recommendation type definitions, savings calculation |
| `inputs/schema.py` | 4 | Extract input schemas from type hints for better input generation |
| `report/renderer.py` | 6 | Jinja2 + inline CSS/SVG → self-contained HTML report |
| `report/charts.py` | 6 | Inline SVG: cost waterfall, context growth sparklines, score ring |
| `report/graph.py` | 6 | Render workflow DAG as SVG with cost overlays |
| `graph/extractor.py` | 6 | DAG extraction from LangGraph, OpenAI Agents, CrewAI |
| `graph/layout.py` | 6 | Position graph nodes (top-to-bottom layout) |
| `graph/colorizer.py` | 6 | Map per-step cost shares to node colors (white → amber → red) |
| `graph/transform.py` | 6 | Apply recommendations to graph for "after" view |
| `ui/app.py` | 6 | FastAPI + pre-built React bundle on localhost:7100 |
| `ui/ws.py` | 6 | WebSocket for live profiling progress |

**Already implemented** in `recommend/`: `prompts.py` (framework-specific implementation prompts, Tier 1/2/3) and `verify.py` (compare old/new profiles for applied recommendations).

---

## 16. Known Limitations

Seven documented limitations with magnitudes and mitigations. Full details in [known-limitations.md](known-limitations.md).

| # | Limitation | Magnitude | Mitigation |
|---|-----------|-----------|------------|
| 1 | **Session context accumulation** — profiles single-turn only | 3–5× underestimate for multi-turn | Profile with mid-session context prepended |
| 2 | **Input distribution mismatch** — profiling inputs ≠ production | 2–3× if input lengths differ | Compare input stats vs production; use Langfuse import |
| 3 | **Tool response size mismatch** — test stubs return short responses | Compounds through downstream steps | Profile against production data sources |
| 4 | **Provider-side caching** — DeepSeek cache 50× cheaper | Profiled cost ≪ cold-start cost if cache not busted | Default `--cache-mode cold`; `--allow-cache` for warm workflows |
| 5 | **Tiered pricing boundaries** — standard tier only | 2–4× underestimate above 200K tokens | Check p95 tokens vs tier boundaries |
| 6 | **Model drift** — silent model updates change cost patterns | 5–25% shift without warning | Re-profile periodically; staleness warning at 30+ days |
| 7 | **Batch vs real-time pricing** — real-time API rates only | ~2× overestimate for batch deployments | Apply 0.5× multiplier for batch |
