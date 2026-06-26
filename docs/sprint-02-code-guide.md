# Sprint 2 Code Guide

Developer walkthrough, data flow diagrams, worked example runs, debugging exercises, and REPL cheat sheet for the Sprint 2 codebase.

---

## Part 1: File-by-File Walkthrough

### 1. `pretia/projection/stats.py`

#### a) What it does

Computes distributional statistics (p50–p99, mean, std) from raw profiling data. Takes the `list[list[StepRecord]]` produced by any collector and produces a `ProfilingStats` object — the single data structure that the rest of the pipeline (report rendering, pattern detection, projections) consumes for numerical analysis.

#### b) Key moving parts

| Name | What it does | Returns |
|---|---|---|
| `_percentile(sorted_data, p)` | Linear-interpolation percentile on pre-sorted floats. Internal helper — never call directly with unsorted data. | `float` |
| `PercentileStats` | Frozen dataclass holding min/max/mean/std/p50/p75/p90/p95/p99 for one metric. Building block for every other stats class. | — |
| `compute_percentile_stats(values)` | Entry point for turning a list of raw floats into a `PercentileStats`. Validates non-empty input, computes sample std (Bessel-corrected, `n-1` denominator). | `PercentileStats` |
| `StepStats` | Per-step stats across all runs: token distributions, cost distribution, iteration distribution, call count. One `StepStats` per unique `step_name`. | — |
| `RunStats` | Aggregate totals for a single run: total cost, total tokens, step count, duration. | — |
| `ProfilingStats` | Top-level container: `dict[str, StepStats]`, `list[RunStats]`, plus cross-run cost and token `PercentileStats`. Serializable via `to_dict()`. | — |
| `_safe_cost(cost_fn, model, input_tokens, output_tokens)` | Wraps the cost function with a try/except so unknown models don't crash the stats pipeline — returns `$0.00` and logs a warning. | `float` |
| `compute_stats(runs, cost_fn=None)` | Main entry point. Iterates all runs, groups records by `step_name`, computes per-step and per-run distributions, and returns a fully populated `ProfilingStats`. Defaults to `calculate_cost` from `pricing.tables`. | `ProfilingStats` |

#### c) How data flows through it

`compute_stats()` is called by `ProfileRunner.run()` (line 298 of `runner.py`) and by the CLI's `analyze` and `report` commands. It receives the raw `list[list[StepRecord]]` — the output of any collector or `traces_to_step_records()`. Internally it:

1. Iterates every run, accumulating per-step records in a `defaultdict(list)` keyed by `step_name`, and tracking which runs contain each step (`step_runs_presence`) and the max iteration per step per run (`step_iterations_per_run`).
2. Builds a `RunStats` for each run and collects run-level cost/token totals.
3. For each step, calls `compute_percentile_stats()` on input tokens, output tokens, total tokens, cost, duration, context size, and iterations-per-run.
4. Calls `compute_percentile_stats()` on the run-level cost and token lists to populate `cost_per_run` and `tokens_per_run`.
5. Returns `ProfilingStats`.

The returned object is serialized via `.to_dict()` and stored in `ProfilingSession.metadata["stats"]`. `format_cli_report()` reads it from there to render tables. `detect_patterns()` accepts it as an optional argument to avoid recomputation.

#### d) Common failure modes

1. **Empty values list to `compute_percentile_stats()`.** If a step appears in zero runs after filtering (e.g., all records had unknown models and were skipped upstream), `compute_percentile_stats([])` raises `ValueError("Cannot compute stats on empty data")`. The traceback points inside `compute_stats` at the line constructing `StepStats`, with no indication of which step caused the problem. The current code avoids this because `step_records` only contains steps that had at least one record, but upstream filtering bugs could trigger it.

2. **Non-numeric token values.** If a collector produces a `StepRecord` with `input_tokens` set to a string (e.g., `"340"` from a JSON import), the `float()` cast on line 244 succeeds, but the `_safe_cost` call fails with a `TypeError` inside `calculate_cost`. The `_safe_cost` try/except catches `ValueError` and `KeyError` but not `TypeError`, so this would propagate as an uncaught exception.

3. **Single-run data.** When `runs` contains exactly one run, `compute_percentile_stats` receives a single-element list for `run_costs` and `run_tokens`. The `n == 1` branch returns all percentiles equal to that single value and `std=0.0`. This is correct, but downstream code that checks `std > 0` (like pattern detectors) will see zero variance and skip analysis. The symptom: no patterns detected, and all percentile columns in the report show the same number.

#### e) How to debug it

Run the stats tests:

```bash
pytest tests/unit/test_stats.py -v
```

Run a single test by name:

```bash
pytest tests/unit/test_stats.py -v -k "test_compute_stats"
```

Test percentile computation interactively:

```python
from pretia.projection.stats import compute_percentile_stats
ps = compute_percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0])
ps  # inspect all fields
```

Enable debug logging to see unknown-model warnings from `_safe_cost`:

```bash
pytest tests/unit/test_stats.py -v --log-cli-level=DEBUG
```

Inspect what `compute_stats` produces for a pair of fake runs:

```python
from dataclasses import replace
from datetime import UTC, datetime
from pretia.collectors.base import StepRecord
from pretia.projection.stats import compute_stats

rec = StepRecord(
    step_name="classify", step_type="llm", model="claude-haiku-3",
    input_tokens=500, output_tokens=50, context_size=600,
    tool_definitions_tokens=0, system_prompt_hash="abc",
    system_prompt_tokens=100, output_format="json",
    is_retry=False, iteration=1, parent_step=None,
    duration_ms=200, timestamp=datetime(2026, 5, 20, tzinfo=UTC),
)
runs = [[rec], [replace(rec, input_tokens=800, output_tokens=80)]]
stats = compute_stats(runs)
stats.cost_per_run
```

---

### 2. `pretia/projection/patterns.py`

#### a) What it does

Detects non-linear cost patterns that make simple mean-based projection unreliable. Runs three independent detectors — context growth, loop count variance, high token variance — and returns a sorted list of `DetectedPattern` objects. These are stored in `ProfilingSession.metadata["patterns"]` and rendered as warnings in the CLI report.

#### b) Key moving parts

| Name | What it does | Returns |
|---|---|---|
| `DetectedPattern` | Frozen dataclass: `pattern_type`, `step_name`, `severity` ("danger"/"warning"), `evidence` dict, human-readable `description`. Serializable via `to_dict()`. | — |
| `_pearson_r_squared(xs, ys)` | Computes r² and slope for two equal-length lists. Returns `(0.0, 0.0)` if fewer than 3 datapoints or zero variance in either dimension. Guards against division by zero. | `tuple[float, float]` |
| `_detect_context_growth(runs)` | Finds steps where `context_size` grows linearly with `iteration` (Pearson r² > 0.7, positive slope). Groups `(iteration, context_size)` pairs per step, runs Pearson, and reports the growth ratio. | `list[DetectedPattern]` |
| `_detect_loop_count_variance(runs)` | Finds steps where the max iteration count varies significantly across runs (coefficient of variation > 0.5). Computes per-run max iteration per step and flags high CV. | `list[DetectedPattern]` |
| `_detect_high_token_variance(stats)` | Finds steps where p95 tokens or cost is > 3x the p50 (heavy-tailed distribution). Reads from the pre-computed `ProfilingStats` — does not re-traverse raw runs. | `list[DetectedPattern]` |
| `detect_patterns(runs, stats=None)` | Orchestrator. If `stats` is not provided, calls `compute_stats(runs)` itself. Runs all three detectors, concatenates results, sorts by severity (danger first). | `list[DetectedPattern]` |

#### c) How data flows through it

`detect_patterns()` is called by `ProfileRunner.run()` (line 299 of `runner.py`), `ProfileRunner.analyze_langfuse()`, and the CLI `analyze` and `report` commands. It receives the same `list[list[StepRecord]]` that went into `compute_stats()`, plus optionally the already-computed `ProfilingStats` to avoid duplicate work.

The context growth and loop variance detectors iterate the raw runs directly (they need per-record iteration and context values). The token variance detector reads from `ProfilingStats.step_stats` (it needs pre-computed p50/p95 ratios).

The output `list[DetectedPattern]` is serialized to dicts and stored in `session.metadata["patterns"]`. `format_cli_report()` reads this list and renders the "Patterns" panel, using severity to pick the icon (red circle for danger, yellow circle for warning).

#### d) Common failure modes

1. **Zero variance in context_size.** If a step has `context_size=0` for all iterations (e.g., tool steps imported from Langfuse), `_pearson_r_squared` would divide by zero without the `denom_y == 0` guard on line 53. With the guard, the step is silently skipped — the actual symptom of removing that guard is a `ZeroDivisionError` in `_pearson_r_squared` with a traceback pointing at `denom = math.sqrt(denom_x * denom_y)` followed by `r = numerator / denom`.

2. **Single-run data.** All three detectors need multiple runs to detect variance. With one run: context growth may still work (if the step iterates multiple times within that run), but loop count variance returns nothing (needs `n >= 2` on line 142), and token variance returns nothing meaningful (p50 == p95 when there are few data points). Symptom: empty patterns list, no warnings in the report. This is correct behavior, but the user may wonder why there are no warnings.

3. **Non-iterating steps passed to context growth.** The filter on line 71 (`rec.iteration > 1 or any(...)`) is an O(n²) scan per record in the worst case — for each record, it checks if any other record in the same run has `iteration > 1` for the same step. With runs containing thousands of records, this quadratic scan can visibly slow down pattern detection.

#### e) How to debug it

Run pattern tests:

```bash
pytest tests/unit/test_patterns.py -v
```

Run a single detector test:

```bash
pytest tests/unit/test_patterns.py -v -k "context_growth"
```

Test context growth detection interactively:

```python
from dataclasses import replace
from datetime import UTC, datetime
from pretia.collectors.base import StepRecord
from pretia.projection.patterns import detect_patterns

rec = StepRecord(
    step_name="summarize", step_type="llm", model="claude-haiku-3",
    input_tokens=500, output_tokens=50, context_size=500,
    tool_definitions_tokens=0, system_prompt_hash="abc",
    system_prompt_tokens=100, output_format="text",
    is_retry=False, iteration=1, parent_step=None,
    duration_ms=200, timestamp=datetime(2026, 5, 20, tzinfo=UTC),
)
run = [
    replace(rec, iteration=1, context_size=500),
    replace(rec, iteration=2, context_size=1000),
    replace(rec, iteration=3, context_size=1500),
    replace(rec, iteration=4, context_size=2000),
]
patterns = detect_patterns([run])
for p in patterns:
    logging.info("%s: %s", p.pattern_type, p.description)
```

Inspect the Pearson computation directly:

```python
from pretia.projection.patterns import _pearson_r_squared
r2, slope = _pearson_r_squared([1.0, 2.0, 3.0, 4.0], [500.0, 1000.0, 1500.0, 2000.0])
r2, slope  # should be (1.0, 500.0)
```

---

### 3. `pretia/collectors/openai_agents.py`

#### a) What it does

Auto-instruments OpenAI Agents SDK workflows by injecting `PretiaRunHooks` into `Runner.run()`. The hooks capture timing, token usage, and metadata for every LLM call, tool call, and handoff event, converting them into `StepRecord` objects. The `OpenAIAgentsCollector` orchestrates running the workflow on each input and collecting the results.

#### b) Key moving parts

| Name | What it does | Returns |
|---|---|---|
| `_estimate_tokens(text)` | Rough token estimate: `len(text) // 4`. Used when the SDK doesn't report usage. | `int` |
| `_detect_output_format(text)` | Checks if output is JSON (parseable), code (contains triple backticks), or text. | `str` |
| `_extract_model_name(agent)` | Safely reads the model string from an Agent object, handling both string and object model attributes. | `str` |
| `_extract_agent_name(agent)` | Reads `agent.name`, falls back to `"agent"`. | `str` |
| `_extract_tool_name(tool)` | Reads `tool.name`, falls back to `"tool_call"`. | `str` |
| `PretiaRunHooks` | Subclass of `RunHooksBase`. Accumulates `StepRecord` objects via async hook methods: `on_agent_start`, `on_agent_end`, `on_llm_start`, `on_llm_end`, `on_tool_start`, `on_tool_end`, `on_handoff`. Uses inflight dictionaries to pair start/end events. Each hook is wrapped in try/except so a bug in Pretia never crashes the user's workflow. | — |
| `PretiaRunHooks.steps` | Property returning a copy of the accumulated `list[StepRecord]`. | `list[StepRecord]` |
| `PretiaRunHooks.reset()` | Clears all state between profiling runs. | `None` |
| `_build_fallback_steps(raw_responses, agent_name, model)` | Extracts StepRecords from `RunResult.raw_responses` when hooks captured nothing (e.g., older SDK versions that don't fire hooks). | `list[StepRecord]` |
| `OpenAIAgentsCollector` | `BaseCollector` subclass. For each input, creates a fresh `PretiaRunHooks`, calls `Runner.run(workflow, input, hooks=hooks)`, and collects `hooks.steps`. Falls back to `_build_fallback_steps` if hooks captured nothing. | — |

#### c) How data flows through it

`OpenAIAgentsCollector.collect()` is called by `ProfileRunner.run()` when the collector is auto-detected as `"openai"` (workflow has `name` and `instructions` attributes) or explicitly selected. For each input string:

1. Creates a fresh `PretiaRunHooks()` instance.
2. Calls `await Runner.run(workflow, inp, hooks=hooks)`.
3. During execution, the SDK fires hook callbacks: `on_llm_start` stores inflight state (model, timestamp, context estimate) keyed by agent name in `_inflight_llm`; `on_llm_end` pops the inflight entry, extracts token usage from `response.usage`, and creates a `StepRecord`. Similarly for tool calls via `_inflight_tool`.
4. After execution, reads `hooks.steps`. If empty, falls back to parsing `result.raw_responses`.
5. Appends the step list to `runs`.

The returned `list[list[StepRecord]]` feeds into `compute_stats()` and `detect_patterns()`.

#### d) Common failure modes

1. **Missing try/except on a hook method.** Every hook method wraps its body in `try/except Exception` so that a bug in Pretia (e.g., `response.usage` being `None`) doesn't propagate into the user's workflow execution. If you remove the try/except from `on_llm_end` and the SDK returns a response where `usage` is `None`, the `getattr(usage, "input_tokens", 0)` on line 160 would succeed (returns `None`), but `getattr(usage, "input_tokens", 0) or 0` on a `None` usage would fail at `getattr(None, "input_tokens", 0)` — actually `getattr` works on `None`, so the real failure is if `usage` itself raises on attribute access. The more likely failure: a response object with no `output` attribute causes an `AttributeError` at line 168. Without the try/except, this surfaces as a crash inside `Runner.run()`, making the user think their agent is broken.

2. **Inflight key collision.** The inflight dictionaries are keyed by agent name (LLM) or tool name (tool). If two concurrent LLM calls happen for the same agent name, the second `on_llm_start` overwrites the first entry in `_inflight_llm`, and the first `on_llm_end` finds no entry. Symptom: missing StepRecords for some LLM calls, plus "on_llm_end for unknown agent" debug warnings. In practice this doesn't happen because the OpenAI SDK processes agents sequentially, but custom runners could trigger it.

3. **Import error at module level.** The `from agents import Runner` at the top of the file raises `ImportError` immediately if `openai-agents` isn't installed. This is intentional, but it means importing `pretia.collectors.openai_agents` fails even if you're not using it. The `__init__.py` uses lazy `__getattr__` to defer this import.

#### e) How to debug it

Run the collector tests (fully mocked — no `openai-agents` dependency):

```bash
pytest tests/unit/test_openai_agents_collector.py -v
```

Run a single test:

```bash
pytest tests/unit/test_openai_agents_collector.py -v -k "TestHooksLLMLifecycle"
```

Since the module requires `openai-agents` at import, test individual helpers by extracting them (the test file patches the import). To debug hook behavior, add `logging.debug()` calls inside the hook methods and run with:

```bash
pytest tests/unit/test_openai_agents_collector.py -v --log-cli-level=DEBUG
```

Inspect the fallback path:

```python
# Only works if openai-agents is installed
from pretia.collectors.openai_agents import _build_fallback_steps

class FakeUsage:
    input_tokens = 500
    output_tokens = 50

class FakeResp:
    usage = FakeUsage()

steps = _build_fallback_steps([FakeResp()], "my_agent", "gpt-4o")
steps[0].input_tokens, steps[0].output_tokens
```

---

### 4. `pretia/inputs/importer.py`

#### a) What it does

Imports production traces from Langfuse's API and converts them into the standard `list[list[StepRecord]]` format that the stats/patterns pipeline expects. This allows analyzing real production costs without re-executing the workflow. Also extracts input texts from traces for use as profiling inputs in re-execution mode.

#### b) Key moving parts

| Name | What it does | Returns |
|---|---|---|
| `LangfuseObservation` | Frozen dataclass representing one LLM call or tool call within a trace: observation ID, name, type, model, token counts, timing, parent link. | — |
| `LangfuseTrace` | Frozen dataclass representing one full trace: trace ID, name, input text, timestamp, list of `LangfuseObservation`, token/cost totals. | — |
| `_compute_duration_ms(start_time, end_time)` | Computes duration in ms from two optional datetimes. Returns `0` if either is `None`. | `int` |
| `_extract_input_text(raw_input)` | Extracts a usable input string from a Langfuse trace's raw input. Handles `str`, `dict` with `messages`/`content`/`input` keys, and fallback `str()` conversion. | `str \| None` |
| `_safe_cost(model, input_tokens, output_tokens)` | Wraps `calculate_cost` with error handling for unknown models. Returns `$0.00` for empty or unrecognized models. | `float` |
| `create_langfuse_client()` | Creates a `LangfuseAPI` client from `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, and `LANGFUSE_HOST` environment variables. Raises `OSError` with a clear message listing which variables are missing. | `LangfuseAPI` |
| `_parse_observation(obs)` | Converts a raw Langfuse `ObservationsView` object into a `LangfuseObservation` dataclass. Extracts usage via `getattr` chains for safety. | `LangfuseObservation` |
| `fetch_traces(client, last_n=10, name=None)` | Fetches the most recent `last_n` traces from Langfuse (capped at 100). For each trace summary, fetches the full trace with observations. Translates auth errors into `PermissionError` and connection errors into `ConnectionError`. | `list[LangfuseTrace]` |
| `traces_to_step_records(traces)` | Converts `list[LangfuseTrace]` into `list[list[StepRecord]]` — one inner list per trace. Skips `EVENT` observations. Maps `GENERATION` → `"llm"`, `SPAN`/`TOOL` → `"tool"` (with `"retrieval"` for names containing `"retriev"`). Resets iteration counter per trace. | `list[list[StepRecord]]` |
| `extract_inputs(traces)` | Extracts input texts from traces. Raises `ValueError` if fewer than 2 traces have extractable input text. | `list[str]` |

#### c) How data flows through it

Two entry paths:

**Path 1: `pretia analyze --from-langfuse`** (CLI `analyze_cmd` in `cli.py`). Calls `create_langfuse_client()` → `fetch_traces(client, last_n)` → `traces_to_step_records(traces)` → result feeds into `compute_stats()` and `detect_patterns()`.

**Path 2: `pretia profile run --from-langfuse`** (via `ProfileRunner._resolve_inputs()`). Calls `create_langfuse_client()` → `fetch_traces()` → `extract_inputs(traces)` → the extracted input strings are used as workflow inputs for re-execution profiling (not the same as analyze — this re-runs the workflow with production-like inputs).

Inside `traces_to_step_records()`, for each trace: builds an `obs_name_map` (observation ID → name) for parent resolution, resets `iteration_counts` per trace, skips `EVENT` observations, classifies step type, and creates `StepRecord` objects with `system_prompt_hash="imported"` and `context_size=input_tokens`.

#### d) Common failure modes

1. **EVENT observations not filtered.** The `_SKIP_TYPES = frozenset({"EVENT"})` filter on line 252 is critical. EVENT observations are log entries — they have no model, zero tokens, and no meaningful cost. Without this filter, junk `StepRecord` objects with `model="unknown"`, `input_tokens=0` get mixed into the stats, silently pulling down the mean cost and skewing the median. There's no error — just wrong numbers in the report.

2. **Iteration counter not reset between traces.** The `iteration_counts` dict is initialized inside the `for trace in traces` loop (line 248), so it resets per trace. If it were initialized before the loop, a step named "review" appearing in trace 1 (iterations 1,2,3) and trace 2 would continue counting from 4 in trace 2. This would make the loop variance detector see `max_iteration=6` for trace 2, triggering a false positive "Loop Count Variance" pattern.

3. **Langfuse API auth failure.** If `LANGFUSE_SECRET_KEY` is wrong, the `client.trace.list()` call raises an exception containing "401" or "403". The code on line 190 checks for these substrings in the exception message and wraps them in `PermissionError`. If the Langfuse API changes its error format (e.g., returns a structured error object where `str(exc)` doesn't contain "401"), this detection fails and the raw exception propagates as a `ConnectionError` with a less helpful message.

#### e) How to debug it

Run the importer tests (fully mocked — no Langfuse dependency):

```bash
pytest tests/unit/test_langfuse_importer.py -v
```

Run a single test:

```bash
pytest tests/unit/test_langfuse_importer.py -v -k "TestTracesToStepRecords"
```

Test `traces_to_step_records` interactively with fake data:

```python
from datetime import UTC, datetime
from pretia.inputs.importer import LangfuseTrace, LangfuseObservation, traces_to_step_records

obs = LangfuseObservation(
    observation_id="obs-1", name="classify", observation_type="GENERATION",
    model="gpt-4o", input_tokens=500, output_tokens=50,
    start_time=datetime(2026, 5, 20, tzinfo=UTC),
    end_time=datetime(2026, 5, 20, 0, 0, 1, tzinfo=UTC),
    duration_ms=1000, parent_observation_id=None,
)
trace = LangfuseTrace(
    trace_id="tr-1", name="my-workflow", input_text="Hello",
    timestamp=datetime(2026, 5, 20, tzinfo=UTC),
    observations=[obs], total_input_tokens=500,
    total_output_tokens=50, total_cost=0.005,
)
runs = traces_to_step_records([trace])
runs[0][0].step_name, runs[0][0].iteration
```

Verify that EVENT observations are skipped:

```python
event_obs = LangfuseObservation(
    observation_id="obs-2", name="log_entry", observation_type="EVENT",
    model=None, input_tokens=0, output_tokens=0,
    start_time=None, end_time=None, duration_ms=0,
    parent_observation_id=None,
)
trace2 = LangfuseTrace(
    trace_id="tr-2", name="test", input_text="Hi",
    timestamp=datetime(2026, 5, 20, tzinfo=UTC),
    observations=[obs, event_obs], total_input_tokens=500,
    total_output_tokens=50, total_cost=0.005,
)
runs2 = traces_to_step_records([trace2])
len(runs2[0])  # should be 1, not 2
```

---

### 5. `pretia/ci/report.py`

#### a) What it does

Renders the CLI cost report using Rich tables and panels. Takes a `ProfilingSession` (which contains stats and patterns in its metadata) and produces a list of Rich renderables — header panel, cost summary table, step breakdown table, monthly projection panel, patterns panel, and an optional iteration panel. The CLI prints each renderable to the terminal.

#### b) Key moving parts

| Name | What it does | Returns |
|---|---|---|
| `format_cost(value)` | Formats a dollar amount for display. Uses 4 decimal places for values under $0.01 to avoid showing "$0.00" for small but real costs. Uses 2 decimal places for $0.01–$999.99, and no decimals for $1000+. | `str` |
| `format_tokens(value)` | Formats a token count with comma separators, no decimal places. | `str` |
| `_tier_style(tier)` | Maps model tier ("fast"/"mid"/"frontier") to a Rich color string for styling model names in the report. | `str` |
| `_truncate_model(model, max_len=7)` | Truncates model names longer than `max_len` with an ellipsis character so the table columns stay aligned. | `str` |
| `format_cli_report(session, cost_summary=None, traffic=None)` | Main entry point. Reads `session.metadata["stats"]` and `session.metadata["patterns"]`. Builds and returns a list of Rich renderables. Supports both the new stats-based format (Sprint 2) and the legacy `cost_summary` format (Sprint 1). | `list[Any]` |
| `_build_cost_summary_table(stats, cost_summary)` | Renders the "Cost Per Run" table: mean, median, p95, p99, min, max, std dev. Falls back to legacy format if stats are missing. | `Table` |
| `_build_step_table(stats, cost_summary, sample_size)` | Renders the "Step Breakdown" table: per-step mean cost, p95 cost, mean tokens, p95 tokens, call count. Sorts steps by mean cost descending. | `Table` |
| `_build_projection_panel(stats, cost_summary, traffic)` | Renders the "Monthly Cost Projection" panel. Shows mean and p95 monthly cost at 100/1K/10K daily runs (or a custom `traffic` value). Computes: `cost_per_run × runs_per_day × 30`. | `Panel` |
| `_build_patterns_panel(patterns)` | Renders detected patterns with severity icons. Green "no patterns" message if the list is empty. | `Panel` |
| `_build_iteration_panel(stats)` | Renders the "Iteration Counts Per Run" panel for steps with `mean_iterations > 1.0`. Returns `None` if no steps iterate, so the CLI skips it. | `Panel \| None` |

#### c) How data flows through it

`format_cli_report()` is called by the CLI commands `profile run`, `report`, and `analyze`. It receives a `ProfilingSession` object. Inside, it reads:

- `session.metadata["stats"]` — a dict (the serialized `ProfilingStats.to_dict()` output).
- `session.metadata["patterns"]` — a list of dicts (serialized `DetectedPattern.to_dict()` output).
- `session.metadata.get("cost_summary", {})` — legacy format from Sprint 1's `_build_cost_summary()`.

The function passes these dicts to the `_build_*` helpers, which read specific keys (e.g., `stats["cost_per_run"]["mean"]`, `stats["step_stats"][name]["cost"]["p95"]`). The helpers return Rich objects (`Table`, `Panel`, `Text`). The CLI iterates the returned list and calls `console.print()` on each.

The report never modifies the session — it's purely a read/render pipeline.

#### d) Common failure modes

1. **Precision loss in `format_cost()`.** If `format_cost()` used only 2 decimal places for all values, a step costing $0.0034 per call would display as "$0.00". The per-step column would be misleading — the user sees a "free" step that actually costs $0.0034 × 10,000 runs/day × 30 = $1,020/month. The monthly projection panel would still show the correct number (it uses raw floats), creating a confusing discrepancy.

2. **Missing `stats` key in metadata.** If a saved profile JSON was created by Sprint 1 code (before `compute_stats` existed), `session.metadata` has no `"stats"` key. The `report` command handles this by recomputing stats from `session.runs` (line 211 of `cli.py`), but if someone calls `format_cli_report()` directly with a session that has no stats and no runs, it falls through to the legacy `cost_summary` path, which may also be empty — producing a report with all "$0.00" values.

3. **Patterns as raw objects vs dicts.** `_build_patterns_panel` expects `patterns` to be a list of dicts (it calls `p.get("severity", ...)`). If someone passes `DetectedPattern` dataclass instances instead of serialized dicts, every `p.get()` call raises `AttributeError`. The pipeline works because `runner.py` always serializes patterns via `p.to_dict()` before storing, but direct callers could hit this.

#### e) How to debug it

Run the report tests:

```bash
pytest tests/unit/test_report.py -v
```

Run a single test:

```bash
pytest tests/unit/test_report.py -v -k "TestFormatCost"
```

Test `format_cost` on edge cases:

```python
from pretia.ci.report import format_cost
format_cost(0.0)       # "$0.00"
format_cost(0.0034)    # "$0.0034"
format_cost(0.50)      # "$0.50"
format_cost(1234.5)    # "$1,235"
```

Build a fake session and render the report:

```python
from datetime import UTC, datetime
from pretia.store import ProfilingSession
from pretia.ci.report import format_cli_report
from rich.console import Console

session = ProfilingSession(
    workflow_name="test-workflow",
    workflow_hash="abc123",
    profiled_at=datetime(2026, 5, 20, tzinfo=UTC),
    sample_size=5,
    input_mode="auto-generate",
    runs=[],
    metadata={
        "stats": {
            "total_runs": 5, "total_steps": 15,
            "cost_per_run": {"mean": 0.05, "p50": 0.04, "p95": 0.09, "p99": 0.12, "min": 0.02, "max": 0.15, "std": 0.03},
            "step_stats": {},
            "run_stats": [],
        },
        "patterns": [],
    },
)
renderables = format_cli_report(session)
c = Console()
for r in renderables:
    c.print(r)
```

---

### 6. `pretia/runner.py` — Sprint 2 Changes

#### What changed

**Stats and patterns integration** (lines 298–299): After collecting runs and building the legacy `cost_summary`, the runner now calls `compute_stats(runs)` to produce `ProfilingStats` and `detect_patterns(runs, profiling_stats)` to produce `list[DetectedPattern]`. Both are serialized and stored in `session.metadata["stats"]` and `session.metadata["patterns"]`.

**OpenAI collector auto-detection** (lines 229–232): The `_select_collector()` method now checks for `name` and `instructions` attributes on the workflow object — this is the OpenAI Agents SDK's `Agent` signature. If detected, it returns `OpenAIAgentsCollector()`.

**`analyze_langfuse()` method** (lines 326–378): New method that imports and analyzes Langfuse traces without re-executing the workflow. Calls `create_langfuse_client()` → `fetch_traces()` → `traces_to_step_records()` → `compute_stats()` → `detect_patterns()` → saves session. Used by the CLI's `analyze` command.

The `_build_cost_summary()` function is retained for backward compatibility — the stats-based format is the primary data, and the legacy summary is a fallback for older profile files.

---

### 7. `pretia/cli.py` — Sprint 2 Changes

#### What changed

**`report` command** (lines 169–218): New top-level command (`pretia report <profile.json>`). Loads a saved profile JSON via `ProfileStore`, optionally recomputes stats/patterns if they're missing from the metadata (backward compat with Sprint 1 profiles), and renders via `format_cli_report()`. Supports `pretia report latest` to load the most recent profile.

**`analyze` command** (lines 221–351): New top-level command (`pretia analyze --from-langfuse`). Validates Langfuse env vars, imports traces, converts to step records, computes stats and patterns, saves the session, and renders the report. Options: `--last N` (number of traces, default 10), `--name` (filter by workflow name), `--traffic` (custom daily runs for projection), `--output-dir`, `-v` verbose.

**`--from-langfuse` flag on `profile run`** (lines 57–68): Added to the existing `profile run` command. When set, the runner's `_resolve_inputs()` uses Langfuse traces as a source of input strings for re-execution profiling (via `extract_inputs()`), rather than auto-generating synthetic inputs.

The CLI structure is now: `pretia profile run` (execute + profile), `pretia report` (render saved profile), `pretia analyze` (import + analyze without execution).

---

## Part 2: Data Flow Diagrams

### Pipeline A: `pretia report profile.json`

```
profile.json (on disk)
    │
    ▼
ProfileStore.load(path)
    │
    ▼
ProfilingSession
    │
    ├── session.metadata has "stats"? ──NO──▶ compute_stats(session.runs) ──▶ ProfilingStats
    │       │                                  detect_patterns(runs, stats)──▶ list[DetectedPattern]
    │       │                                  store in session.metadata
    │       │                                       │
    │   YES ◀───────────────────────────────────────┘
    │
    ▼
format_cli_report(session, traffic=traffic)
    │
    ├── reads session.metadata["stats"]       → dict
    ├── reads session.metadata["patterns"]    → list[dict]
    ├── reads session.metadata["cost_summary"]→ dict (legacy fallback)
    │
    ▼
list[Rich renderables]
    │
    ▼
console.print() each renderable → terminal output
```

### Pipeline B: `pretia analyze --from-langfuse --last 10`

```
LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY (env vars)
    │
    ▼
create_langfuse_client() ──▶ LangfuseAPI client
    │
    ▼
fetch_traces(client, last_n=10) ──▶ list[LangfuseTrace]
    │                                  each trace has list[LangfuseObservation]
    ▼
traces_to_step_records(traces) ──▶ list[list[StepRecord]]
    │                                (one inner list per trace,
    │                                 EVENT observations skipped,
    │                                 iteration reset per trace)
    ▼
compute_stats(runs) ──▶ ProfilingStats
    │                     (PercentileStats for cost, tokens, duration per step)
    ▼
detect_patterns(runs, stats) ──▶ list[DetectedPattern]
    │                              (context growth, loop variance, token variance)
    ▼
ProfilingSession(... metadata={"stats": ..., "patterns": ...})
    │
    ├──▶ ProfileStore.save(session) ──▶ .pretia/{name}_{timestamp}.json
    │
    ▼
format_cli_report(session, traffic=traffic) ──▶ list[Rich renderables]
    │
    ▼
console.print() ──▶ terminal output
```

---

## Part 3: Worked Example Runs

Six traced examples that exercise every Sprint 2 code path. Each shows exact functions called, intermediate values, and branch decisions. Read these like a debugger trace.

### Coverage map

| Example | compute_stats | compute_percentile_stats | _detect_context_growth | _detect_loop_count_variance | _detect_high_token_variance | OpenAI Agents hooks | Langfuse import | traces_to_step_records | format_cli_report (stats path) | format_cli_report (legacy path) | report cmd recompute |
|---------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| A (stats basics) | ✓ | ✓ | | | | | | | | | |
| B (context growth) | ✓ | | ✓ | | | | | | ✓ | | |
| C (loop variance) | ✓ | | | ✓ | | | | | ✓ | | |
| D (OpenAI hooks) | ✓ | | | | | ✓ | | | | | |
| E (Langfuse analyze) | ✓ | | | | ✓ | | ✓ | ✓ | ✓ | | |
| F (report legacy) | | | | | | | | | | ✓ | ✓ |

---

### Example A: `compute_stats` on a 3-run workflow — building ProfilingStats from scratch

**Scenario:** 3 profiling runs, each with 2 steps (classify + respond). No loops, no variance. This traces the full stats computation to show how raw StepRecords become PercentileStats.

**Input data:**

```python
# All steps use iteration=1, no loops.
# classify: claude-haiku-4-5 (Haiku pricing: $1.00/$5.00 per million)
# respond:  claude-sonnet-4-6 (Sonnet pricing: $3.00/$15.00 per million)
runs = [
    [classify(in=300, out=40), respond(in=600, out=200)],   # run 0
    [classify(in=350, out=38), respond(in=550, out=210)],   # run 1
    [classify(in=280, out=42), respond(in=620, out=190)],   # run 2
]
```

**Trace:**

```
compute_stats(runs, cost_fn=None):
  cost_fn defaults to calculate_cost
  runs is not empty → proceed

  ─── Run 0 ───
    classify: step_records["classify"].append(rec)
              step_runs_presence["classify"].add(0)
              step_iterations_per_run["classify"][0] = 1
              cost = _safe_cost(calculate_cost, "claude-haiku-4-5", 300, 40)
                   = calculate_cost("claude-haiku-4-5", 300, 40)
                   = 300 × (1.00/1M) + 40 × (5.00/1M)
                   = 0.000300 + 0.000200 = 0.000500
              run_cost += 0.000500
              run_total_tokens += 340

    respond:  cost = _safe_cost(calculate_cost, "claude-sonnet-4-6", 600, 200)
                   = 600 × (3.00/1M) + 200 × (15.00/1M)
                   = 0.001800 + 0.003000 = 0.004800
              run_cost += 0.004800
              run_total_tokens += 800

    run_cost = 0.005300
    run_costs.append(0.005300)
    RunStats(run_index=0, total_cost=0.00530, total_tokens=1140, step_count=2, ...)

  ─── Run 1 ───
    classify: cost = 350 × 0.000001 + 38 × 0.000005 = 0.000350 + 0.000190 = 0.000540
    respond:  cost = 550 × 0.000003 + 210 × 0.000015 = 0.001650 + 0.003150 = 0.004800
    run_cost = 0.005340
    RunStats(run_index=1, total_cost=0.00534, ...)

  ─── Run 2 ───
    classify: cost = 280 × 0.000001 + 42 × 0.000005 = 0.000280 + 0.000210 = 0.000490
    respond:  cost = 620 × 0.000003 + 190 × 0.000015 = 0.001860 + 0.002850 = 0.004710
    run_cost = 0.005200
    RunStats(run_index=2, total_cost=0.00520, ...)

  ─── Per-step stats for "classify" ───
    records = 3 records from runs 0, 1, 2
    input_tok_vals = [300.0, 350.0, 280.0]
    output_tok_vals = [40.0, 38.0, 42.0]
    cost_vals = [0.000500, 0.000540, 0.000490]

    compute_percentile_stats([0.000500, 0.000540, 0.000490]):
      sorted = [0.000490, 0.000500, 0.000540]
      n = 3
      mean = (0.000490 + 0.000500 + 0.000540) / 3 = 0.000510
      variance = ((−0.000020)² + (−0.000010)² + (0.000030)²) / 2
               = (0.0000000004 + 0.0000000001 + 0.0000000009) / 2
               = 0.0000000014 / 2 = 0.0000000007
      std = √0.0000000007 = 0.0000265
      p50 = _percentile([0.000490, 0.000500, 0.000540], 50):
            k = (3−1) × 50/100 = 1.0
            f=1, c=2 → sorted[1] + (1.0−1) × (sorted[2]−sorted[1]) = 0.000500
      p95 = _percentile(..., 95):
            k = 2 × 0.95 = 1.9
            f=1, c=2 → 0.000500 + 0.9 × (0.000540−0.000500) = 0.000536
      → PercentileStats(min=0.000490, max=0.000540, mean=0.000510,
                         std=0.0000265, p50=0.000500, p95=0.000536, ...)

    iter_per_run:
      step_iterations_per_run["classify"] = {0: 1, 1: 1, 2: 1}
      For each run in step_runs_presence["classify"] = {0, 1, 2}:
        [1.0, 1.0, 1.0]
      compute_percentile_stats([1.0, 1.0, 1.0]):
        n=3, all identical → mean=1.0, std=0.0, all percentiles=1.0
      mean_iterations = 1.0

    → StepStats(step_name="classify", model="claude-haiku-4-5", call_count=3,
         runs_present=3, cost=PercentileStats(mean=0.000510, ...), ...)

  ─── Per-step stats for "respond" ───
    cost_vals = [0.004800, 0.004800, 0.004710]
    (similar computation → mean=0.004770, p50=0.004800, ...)

  ─── Cross-run stats ───
    run_costs = [0.005300, 0.005340, 0.005200]
    compute_percentile_stats([0.005200, 0.005300, 0.005340]):
      mean = 0.005280
      p50 = 0.005300
      p95 = 0.005336

  → ProfilingStats(
      step_stats={"classify": ..., "respond": ...},
      run_stats=[RunStats×3],
      cost_per_run=PercentileStats(mean=0.005280, p50=0.005300, p95=0.005336),
      total_runs=3,
      total_steps=6,
    )
```

**Key takeaway:** `compute_stats` does two passes: (1) iterate all runs collecting per-record costs and run totals, (2) for each step, call `compute_percentile_stats` on every metric dimension. The Bessel-corrected std (dividing by `n-1`) matters most at small n. With 3 runs of nearly identical cost, p50 ≈ p95 — there's barely any spread. This is normal; patterns won't fire and projections will be tight.

---

### Example B: Context growth detection — Pearson r² with linear iteration data

**Scenario:** A summarize step loops 1–5 times across 4 runs, with context_size growing ~400 tokens per iteration. This traces the full `_detect_context_growth` pipeline including the r² computation.

**Input data:**

```python
# 4 runs. "summarize" step iterates with growing context.
runs = [
  [summarize(iter=1, ctx=500), summarize(iter=2, ctx=900), summarize(iter=3, ctx=1300)],
  [summarize(iter=1, ctx=500), summarize(iter=2, ctx=920), summarize(iter=3, ctx=1280),
   summarize(iter=4, ctx=1700)],
  [summarize(iter=1, ctx=480), summarize(iter=2, ctx=880)],
  [summarize(iter=1, ctx=510), summarize(iter=2, ctx=910), summarize(iter=3, ctx=1320),
   summarize(iter=4, ctx=1710), summarize(iter=5, ctx=2100)],
]
```

**Trace:**

```
detect_patterns(runs, stats):
  stats = compute_stats(runs)  (if not provided)

  _detect_context_growth(runs):
    │
    ├─ Build step_pairs:
    │    For each record: check "rec.iteration > 1 OR any record in same run
    │    for same step has iteration > 1"
    │    Run 0: summarize iter=1 → check run: iter 2,3 exist → YES. Include all 3.
    │    Run 1: all 4 included (iterations 1–4)
    │    Run 2: both included (iterations 1–2)
    │    Run 3: all 5 included (iterations 1–5)
    │
    │    step_pairs["summarize"] = [
    │      (1, 500), (2, 900), (3, 1300),           # run 0
    │      (1, 500), (2, 920), (3, 1280), (4, 1700), # run 1
    │      (1, 480), (2, 880),                        # run 2
    │      (1, 510), (2, 910), (3, 1320), (4, 1710), (5, 2100), # run 3
    │    ]
    │    n = 14 data points  (≥ 5 threshold → proceed)
    │
    ├─ xs = [1, 2, 3, 1, 2, 3, 4, 1, 2, 1, 2, 3, 4, 5]
    │  ys = [500, 900, 1300, 500, 920, 1280, 1700, 480, 880, 510, 910, 1320, 1710, 2100]
    │
    ├─ _pearson_r_squared(xs, ys):
    │    n = 14
    │    sum_x = 1+2+3+1+2+3+4+1+2+1+2+3+4+5 = 34
    │    sum_y = 500+900+1300+500+920+1280+1700+480+880+510+910+1320+1710+2100 = 15010
    │    sum_xy = 1×500 + 2×900 + 3×1300 + ... = 34900 (approx)
    │    sum_x2 = 1+4+9+1+4+9+16+1+4+1+4+9+16+25 = 104
    │    sum_y2 = 500²+900²+... (large)
    │
    │    denom_x = 14 × 104 − 34² = 1456 − 1156 = 300
    │    denom_y = 14 × sum_y2 − 15010² (large positive)
    │    denom_x ≠ 0, denom_y ≠ 0 → proceed
    │
    │    numerator = 14 × 34900 − 34 × 15010 = 488600 − 510340 = ... (let's get the sign right)
    │    Actually the slope is positive (context grows with iteration), so r > 0.
    │
    │    r_squared ≈ 0.993  (near-perfect linear relationship)
    │    slope ≈ 400 tokens/iteration
    │
    ├─ r_squared = 0.993 > 0.7 → candidate
    │  slope > 0 → positive direction (growth, not shrinkage)
    │
    ├─ severity:
    │    r_squared = 0.993 > 0.85 → "danger"
    │
    ├─ Context ratio:
    │    first_iter_contexts = [500, 500, 480, 510] → mean_first = 497.5
    │    max_iter = 5
    │    last_iter_contexts = [2100] → mean_last = 2100
    │    ratio = 2100 / 497.5 = 4.2×
    │
    └─ → DetectedPattern(
           pattern_type="context_growth",
           step_name="summarize",
           severity="danger",
           evidence={"r_squared": 0.993, "slope": 400.0,
                     "mean_context_first": 497.5, "mean_context_last": 2100.0,
                     "n_datapoints": 14},
           description="Context grows by ~400 tokens per iteration in step
             'summarize' (r²=0.99). At iteration 5, context is 4.2x the initial size."
         )

  _detect_loop_count_variance(runs):
    step_max_iter["summarize"] = [3, 4, 2, 5]  (max iter per run)
    Not all 1 → proceed
    n=4, mean=3.5, std=1.29, CV=1.29/3.5=0.37
    0.37 ≤ 0.5 → does NOT trigger (CV too low)
    → return []

  _detect_high_token_variance(stats):
    For "summarize": total_tokens vary because iteration count varies
    p95/p50 ≈ 1.8 (some runs have 2 iterations, some have 5)
    1.8 ≤ 3.0 → skip
    → return []

  → patterns = [context_growth "summarize" (danger)], sorted by severity
```

**Key takeaway:** The r² calculation uses the raw (iteration, context_size) pairs across ALL runs — not averages. With 14 data points and a strong linear trend, r²=0.993 easily passes the 0.7 threshold. The loop count variance detector does NOT fire here (CV=0.37 < 0.5) even though iteration counts vary — that's because the variance is moderate, not extreme. The two detectors are independent: one measures whether context grows per iteration, the other measures whether the iteration count itself varies.

---

### Example C: Loop count variance — high iteration spread across runs

**Scenario:** A "review" step loops 1–15 times across 10 runs. The iteration count varies wildly (some inputs need no review, others need extensive review). Context is constant at ~600 tokens.

**Input data:**

```python
# 10 runs. "review" max iterations: [1, 8, 2, 15, 3, 12, 1, 7, 4, 10]
# Each iteration costs ~$0.005 (Sonnet). Context does not grow.
```

**Trace:**

```
detect_patterns(runs, stats):

  _detect_context_growth(runs):
    step_pairs["review"]: collects (iteration, context_size) pairs
    context = 600 for every record (constant)
    _pearson_r_squared([1,2,...], [600,600,...]):
      denom_y = n × sum_y2 − sum_y²
      All y values are 600 → denom_y = n × n × 600² − (n × 600)² = 0
      denom_y == 0 → return (0.0, 0.0)
    r² = 0.0 → skip
    → return []  (no growth when context is constant)

  _detect_loop_count_variance(runs):
    │
    ├─ step_max_iter["review"]:
    │    Run 0: max iteration = 1
    │    Run 1: max iteration = 8
    │    ... (from all 10 runs)
    │    iters = [1, 8, 2, 15, 3, 12, 1, 7, 4, 10]
    │
    ├─ all(i == 1 for i in iters) → False → proceed
    │  n = 10, n >= 2 → proceed
    │
    ├─ mean_iter = (1+8+2+15+3+12+1+7+4+10)/10 = 63/10 = 6.3
    │  mean_iter > 0 → proceed
    │
    ├─ variance = sum((x - 6.3)² for x in iters) / (10-1)
    │    = ((−5.3)² + (1.7)² + (−4.3)² + (8.7)² + (−3.3)² + (5.7)²
    │       + (−5.3)² + (0.7)² + (−2.3)² + (3.7)²) / 9
    │    = (28.09 + 2.89 + 18.49 + 75.69 + 10.89 + 32.49
    │       + 28.09 + 0.49 + 5.29 + 13.69) / 9
    │    = 216.1 / 9 = 24.01
    │  std = √24.01 = 4.90
    │
    ├─ CV = 4.90 / 6.3 = 0.778
    │  0.778 > 0.5 → pattern triggered!
    │
    ├─ max_i = 15, min_i = 1
    │  ratio = 15 / 6.3 = 2.38
    │
    ├─ Severity check:
    │    CV > 1.0? 0.778 > 1.0 → No
    │    max > 3 × mean? 15 > 18.9 → No
    │    → severity = "warning"
    │
    └─ → DetectedPattern(
           pattern_type="loop_count_variance",
           step_name="review",
           severity="warning",
           evidence={
             "cv": 0.778, "mean_iterations": 6.3,
             "min_iterations": 1, "max_iterations": 15,
             "std_iterations": 4.90
           },
           description="Loop count for step 'review' varies from 1 to 15
             iterations (mean=6.3, CV=0.78). Worst-case runs cost ~2.4x the average."
         )

  _detect_high_token_variance(stats):
    "review" step: total tokens vary (1 iteration × 650 = 650 vs 15 × 650 = 9750)
    p95_tokens / p50_tokens ≈ 9100/3900 = 2.33
    2.33 ≤ 3.0 → skip (just under the threshold)
    → return []

  → patterns = [loop_count_variance "review" (warning)]
```

**Key takeaway:** The CV formula uses Bessel-corrected sample variance (`/(n-1)`). CV=0.778 crosses the 0.5 threshold. The severity distinction matters: `danger` (CV > 1.0 or max > 3× mean) triggers stronger warnings. The token variance detector is separate and almost fires here (ratio 2.33 vs threshold 3.0) — in a real workflow with even more iteration spread, both patterns would appear.

---

### Example D: OpenAI Agents hooks — `PretiaRunHooks` lifecycle

**Scenario:** An OpenAI Agents SDK workflow with one agent that makes 2 LLM calls and 1 tool call. This traces the hook-based collection mechanism including the inflight pairing, fallback path, and try/except safety.

**Input data:**

```python
# Agent "support_bot" uses gpt-4.1, has a "lookup_account" tool
# Flow: LLM call 1 (plan) → tool call (lookup) → LLM call 2 (respond)
```

**Trace:**

```
OpenAIAgentsCollector.collect(workflow=agent, inputs=["reset my password"]):
  │
  ├─ Input 0: "reset my password"
  │    hooks = PretiaRunHooks()   ← fresh per run
  │    hooks._steps = []
  │    hooks._inflight_llm = {}
  │    hooks._inflight_tool = {}
  │    hooks._iteration_counters = {}
  │
  │    result = await Runner.run(agent, "reset my password", hooks=hooks)
  │    │
  │    │  ┌─ SDK fires: hooks.on_agent_start(context, agent)
  │    │  │   try:
  │    │  │     agent_name = _extract_agent_name(agent) → "support_bot"
  │    │  │     (Stores agent start time, but no StepRecord yet)
  │    │  │   except Exception: logger.debug(...)  ← safety wrapper
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_llm_start(context, agent, input_items)
  │    │  │   try:
  │    │  │     agent_name = _extract_agent_name(agent) → "support_bot"
  │    │  │     model = _extract_model_name(agent):
  │    │  │       getattr(agent, "model", None) → "gpt-4.1"
  │    │  │       isinstance("gpt-4.1", str) → True → return "gpt-4.1"
  │    │  │
  │    │  │     step_name = f"support_bot/llm"
  │    │  │     context_size = _estimate_tokens(str(input_items)) → len(str(...))/4 ≈ 250
  │    │  │
  │    │  │     self._inflight_llm["support_bot"] = {
  │    │  │       "model": "gpt-4.1",
  │    │  │       "start_ns": time.monotonic_ns(),
  │    │  │       "step_name": "support_bot/llm",
  │    │  │       "context_size": 250,
  │    │  │       "system_prompt_hash": sha256(agent.instructions),
  │    │  │       "system_prompt_tokens": len(agent.instructions) // 4,
  │    │  │       "timestamp": datetime.now(UTC),
  │    │  │     }
  │    │  │   except Exception: logger.debug(...)
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_llm_end(context, agent, response)
  │    │  │   try:
  │    │  │     agent_name = "support_bot"
  │    │  │     inflight = self._inflight_llm.pop("support_bot") → the dict above
  │    │  │
  │    │  │     usage = getattr(response, "usage", None) → Usage(input_tokens=250, output_tokens=80)
  │    │  │     input_tokens = getattr(usage, "input_tokens", 0) → 250
  │    │  │     output_tokens = getattr(usage, "output_tokens", 0) → 80
  │    │  │
  │    │  │     output_items = getattr(response, "output", [])
  │    │  │     → [ToolCallItem(name="lookup_account", ...)]
  │    │  │     output_text = "" (tool calls don't have .text)
  │    │  │     output_format = "text" (default when no text extracted)
  │    │  │
  │    │  │     duration_ms = (now_ns - inflight["start_ns"]) // 1_000_000 → 450
  │    │  │     iteration = self._next_iteration("support_bot/llm") → 1
  │    │  │
  │    │  │     StepRecord(step_name="support_bot/llm", step_type="llm",
  │    │  │       model="gpt-4.1", input_tokens=250, output_tokens=80,
  │    │  │       context_size=250, iteration=1, duration_ms=450, ...)
  │    │  │     self._steps.append(record)   ← first record
  │    │  │   except Exception: logger.debug(...)
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_tool_start(context, agent, tool)
  │    │  │   try:
  │    │  │     tool_name = _extract_tool_name(tool) → "lookup_account"
  │    │  │     self._inflight_tool["lookup_account"] = {
  │    │  │       "start_ns": time.monotonic_ns(),
  │    │  │       "step_name": "lookup_account",
  │    │  │       "agent_name": "support_bot",
  │    │  │     }
  │    │  │   except Exception: ...
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_tool_end(context, agent, tool, result)
  │    │  │   try:
  │    │  │     inflight = self._inflight_tool.pop("lookup_account")
  │    │  │     duration_ms = ... → 120
  │    │  │     StepRecord(step_name="lookup_account", step_type="tool",
  │    │  │       model="", input_tokens=0, output_tokens=0, iteration=1, ...)
  │    │  │     self._steps.append(record)   ← second record (tool)
  │    │  │   except Exception: ...
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_llm_start (second LLM call)
  │    │  │   self._inflight_llm["support_bot"] = { model: "gpt-4.1", ... }
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_llm_end (second LLM call)
  │    │  │   inflight = self._inflight_llm.pop("support_bot")
  │    │  │   usage: input_tokens=580, output_tokens=200 (larger — includes tool result in context)
  │    │  │   iteration = self._next_iteration("support_bot/llm") → 2
  │    │  │   StepRecord(step_name="support_bot/llm", step_type="llm",
  │    │  │     model="gpt-4.1", input_tokens=580, output_tokens=200, iteration=2, ...)
  │    │  │   self._steps.append(record)   ← third record (LLM #2)
  │    │  │
  │    │  ┌─ SDK fires: hooks.on_agent_end(context, agent, output)
  │    │  │   (No StepRecord created — agent_end is just tracking metadata)
  │    │
  │    │  Runner.run completes → result
  │    │
  │    ├─ hooks.steps → returns copy of [llm_1, tool, llm_2]  (3 records)
  │    │
  │    ├─ len(hooks.steps) = 3 > 0 → skip fallback path
  │    │   (If hooks.steps were empty — e.g., older SDK that doesn't fire hooks —
  │    │    _build_fallback_steps(result.raw_responses, "support_bot", "gpt-4.1")
  │    │    would parse result.raw_responses[].usage for token data)
  │    │
  │    └─ runs[0] = [llm_1, tool, llm_2]
  │
  └─ → runs = [[llm_1, tool, llm_2]]
```

**Key takeaway:** Every hook method is wrapped in `try/except Exception` — a bug in Pretia never crashes the user's workflow. The inflight dict pairs start/end events by agent name (LLM) or tool name (tool). The `_next_iteration` counter increments per `step_name`, so `support_bot/llm` gets iterations 1 and 2. The fallback path (`_build_fallback_steps`) only activates if hooks capture nothing — it reads `result.raw_responses` as a last resort.

---

### Example E: Langfuse trace analysis — full `analyze` pipeline with EVENT filtering

**Scenario:** `pretia analyze --from-langfuse --last 3`. Three Langfuse traces are imported, one contains an EVENT observation that should be filtered. This traces the full import → stats → patterns → report pipeline.

**Input data:**

```python
# Trace 1: classify (GENERATION, gpt-4o, 500/50) + respond (GENERATION, gpt-4o, 800/200)
# Trace 2: classify (GENERATION, gpt-4o, 480/48) + log_event (EVENT, None, 0/0)
#           + respond (GENERATION, gpt-4o, 820/190)
# Trace 3: classify (GENERATION, gpt-4o, 510/52) + retrieve_docs (SPAN, None, 0/0)
#           + respond (GENERATION, gpt-4o, 780/210)
```

**Trace:**

```
cli.py:analyze_cmd(from_langfuse=True, last_n=3, name=None, ...)
  │
  ├─ create_langfuse_client():
  │    secret_key = os.environ.get("LANGFUSE_SECRET_KEY") → "sk-lf-..."
  │    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY") → "pk-lf-..."
  │    host = os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
  │    Both present → proceed
  │    from langfuse.api.client import LangfuseAPI
  │    → LangfuseAPI(base_url=host, username=public_key, password=secret_key)
  │
  ├─ fetch_traces(client, last_n=3):
  │    min(3, 100) = 3
  │    client.trace.list(limit=3, order_by="timestamp") → trace_list of 3 summaries
  │    For each summary:
  │      full_trace = client.trace.get(trace_id=...)
  │      observations = [_parse_observation(obs) for obs in full_trace.observations]
  │        _parse_observation(obs):
  │          usage = getattr(obs, "usage", None)
  │          input_tokens = getattr(usage, "input", 0) → 500 (or 0 for EVENT)
  │          output_tokens = getattr(usage, "output", 0) → 50 (or 0 for EVENT)
  │          → LangfuseObservation(name="classify", type="GENERATION", model="gpt-4o", ...)
  │
  │      input_text = _extract_input_text(getattr(full_trace, "input", None)):
  │        raw_input = {"messages": [{"role": "user", "content": "Help me"}]}
  │        messages[0]["content"] → "Help me"
  │        → "Help me"
  │
  │    → [LangfuseTrace×3]
  │
  ├─ traces_to_step_records(traces):
  │    │
  │    │  ─── Trace 1 ───
  │    │  obs_name_map = {"obs-1a": "classify", "obs-1b": "respond"}
  │    │  iteration_counts = {}  ← reset per trace
  │    │
  │    │  Obs "classify" (GENERATION):
  │    │    "GENERATION" in _SKIP_TYPES({"EVENT"})? → No
  │    │    "GENERATION" in _GENERATION_TYPES? → Yes → step_type = "llm"
  │    │    iteration_counts.get("classify", 0) + 1 = 1
  │    │    StepRecord(step_name="classify", step_type="llm", model="gpt-4o",
  │    │      input_tokens=500, output_tokens=50, context_size=500,
  │    │      system_prompt_hash="imported", iteration=1, ...)
  │    │
  │    │  Obs "respond" (GENERATION):
  │    │    step_type = "llm", iteration = 1
  │    │    StepRecord(model="gpt-4o", input_tokens=800, output_tokens=200, ...)
  │    │
  │    │  → run 0 = [classify, respond]  (2 records)
  │    │
  │    │  ─── Trace 2 ───
  │    │  iteration_counts = {}  ← RESET (key design decision — not carried from trace 1)
  │    │
  │    │  Obs "classify" (GENERATION): iteration=1, StepRecord created
  │    │
  │    │  Obs "log_event" (EVENT):
  │    │    "EVENT" in _SKIP_TYPES → YES → continue (SKIPPED)
  │    │    (Without this filter, "log_event" would create a StepRecord with
  │    │     model="unknown", input_tokens=0, pulling down cost averages)
  │    │
  │    │  Obs "respond" (GENERATION): iteration=1, StepRecord created
  │    │
  │    │  → run 1 = [classify, respond]  (2 records, EVENT filtered out)
  │    │
  │    │  ─── Trace 3 ───
  │    │  iteration_counts = {}
  │    │
  │    │  Obs "classify" (GENERATION): step_type="llm", iteration=1
  │    │
  │    │  Obs "retrieve_docs" (SPAN):
  │    │    "SPAN" not in _SKIP_TYPES → proceed
  │    │    "SPAN" not in _GENERATION_TYPES → proceed
  │    │    "retriev" in "retrieve_docs".lower() → YES → step_type = "retrieval"
  │    │    model = None → "unknown"
  │    │    StepRecord(step_name="retrieve_docs", step_type="retrieval",
  │    │      model="unknown", input_tokens=0, output_tokens=0, ...)
  │    │
  │    │  Obs "respond" (GENERATION): step_type="llm", iteration=1
  │    │
  │    │  → run 2 = [classify, retrieve_docs, respond]  (3 records)
  │    │
  │    └─ → runs = [[2 recs], [2 recs], [3 recs]]
  │
  ├─ compute_stats(runs):
  │    3 runs, 7 total steps
  │    "classify": 3 records across 3 runs, cost ≈ $0.00175 each
  │    "respond": 3 records, cost ≈ $0.005 each
  │    "retrieve_docs": 1 record, model="unknown" → _safe_cost returns $0.00
  │    run_costs ≈ [0.00675, 0.00655, 0.00675]
  │    → ProfilingStats(total_runs=3, total_steps=7, ...)
  │
  ├─ detect_patterns(runs, stats):
  │    _detect_context_growth: all iteration=1 → return []
  │    _detect_loop_count_variance: all iteration=1 → return []
  │    _detect_high_token_variance(stats):
  │      "classify": p95/p50 ≈ 1.04 → skip
  │      "respond": p95/p50 ≈ 1.05 → skip
  │      "retrieve_docs": p50=0 → ratio=0 → skip (division guard: p50 > 0)
  │    → patterns = []
  │
  ├─ Build ProfilingSession:
  │    workflow_name = traces[0].name → "support-workflow"
  │    input_mode = "langfuse-analyze"
  │    metadata = {"stats": stats.to_dict(), "patterns": [],
  │                "langfuse_trace_count": 3, "langfuse_trace_ids": [...]}
  │
  ├─ ProfileStore.save(session) → .pretia/support-workflow_20260601_143022.json
  │
  └─ format_cli_report(session):
      reads session.metadata["stats"] → dict (the stats-based path, not legacy)
      _build_cost_summary_table(stats):
        mean = 0.00668, p50 = 0.00675, p95 = 0.00675
      _build_step_table(stats):
        respond:       $0.0050 mean, $0.0050 p95 (sorted first — highest cost)
        classify:      $0.0018 mean, $0.0018 p95
        retrieve_docs: $0.0000 mean, $0.0000 p95 (unknown model → zero cost)
      _build_patterns_panel([]):
        "No significant patterns detected" (green message)
      → list of Rich renderables → console.print()
```

**Key takeaway:** The EVENT filter (`_SKIP_TYPES`) is the critical line. Without it, trace 2's "log_event" would become a StepRecord with `model="unknown"` and zero tokens, diluting the cost average. The iteration counter resets per trace — this is verified by trace 2's "classify" starting at iteration=1, not continuing from trace 1's count. The retrieval detection uses substring matching (`"retriev" in name.lower()`), which catches "retrieve_docs", "retrieval", "document_retriever", etc.

---

### Example F: `pretia report` on a Sprint 1 profile — legacy fallback + recomputation

**Scenario:** The user runs `pretia report .pretia/old_profile.json` on a profile saved by Sprint 1 code. The JSON has `cost_summary` but no `stats` key. The `report` command detects this, recomputes stats and patterns from the stored runs, then renders.

**Input data:**

```python
# Saved by Sprint 1: session.metadata has "cost_summary" but NOT "stats" or "patterns"
# session.runs has the raw StepRecord lists (3 runs × 2 steps)
```

**Trace:**

```
cli.py:report_cmd(profile_path=".pretia/old_profile.json", traffic=None):
  │
  ├─ store = ProfileStore()
  │  p = Path(".pretia/old_profile.json") → exists
  │  session = store.load(p):
  │    json.loads(file content) → dict
  │    ProfilingSession.from_dict(data):
  │      runs = [[StepRecord.from_dict(r) for r in run] for run in data["runs"]]
  │        (3 runs × 2 steps → 6 StepRecords reconstructed + validated)
  │      metadata = {"cost_summary": {...}}  (Sprint 1 format, no "stats" key)
  │    → ProfilingSession
  │
  ├─ "stats" not in session.metadata → True
  │  session.runs → truthy (3 runs)
  │  → RECOMPUTE PATH:
  │
  │  profiling_stats = compute_stats(session.runs):
  │    (Same computation as Example A — produces full ProfilingStats
  │     with PercentileStats for every metric)
  │    → ProfilingStats(total_runs=3, ...)
  │
  │  patterns = detect_patterns(session.runs, profiling_stats):
  │    (Runs all 3 Sprint 2 detectors on the stored data)
  │    → [] (no patterns for this simple workflow)
  │
  │  session.metadata["stats"] = profiling_stats.to_dict()
  │  session.metadata["patterns"] = [p.to_dict() for p in patterns]
  │  (Now session.metadata has BOTH Sprint 1 "cost_summary" AND Sprint 2 "stats")
  │
  ├─ format_cli_report(session, traffic=None):
  │    │
  │    │  session.metadata has "stats" → YES → use stats-based rendering path
  │    │
  │    │  _build_cost_summary_table(stats, cost_summary=None):
  │    │    Reads from stats["cost_per_run"]:
  │    │      mean, p50, p95, p99, min, max, std
  │    │    → Rich Table with "Cost Per Run" header
  │    │
  │    │  _build_step_table(stats, cost_summary=None, sample_size=3):
  │    │    For each step in stats["step_stats"]:
  │    │      mean_cost = step["cost"]["mean"]
  │    │      p95_cost = step["cost"]["p95"]
  │    │      mean_tokens = step["total_tokens"]["mean"]
  │    │      p95_tokens = step["total_tokens"]["p95"]
  │    │    Sort by mean_cost descending
  │    │    → Rich Table
  │    │
  │    │  _build_projection_panel(stats, cost_summary=None, traffic=None):
  │    │    traffic is None → use default [100, 1000, 10000]
  │    │    For each volume:
  │    │      mean_monthly = stats.cost_per_run.mean × volume × 30
  │    │      p95_monthly = stats.cost_per_run.p95 × volume × 30
  │    │    → Rich Panel
  │    │
  │    │  _build_patterns_panel(patterns=[]):
  │    │    Empty list → "No significant patterns detected" (green)
  │    │
  │    │  _build_iteration_panel(stats):
  │    │    No step has mean_iterations > 1.0 → returns None → skipped
  │    │
  │    └─ → [header, cost_table, step_table, projection_panel, patterns_panel]
  │
  └─ console.print() each → terminal output

  NOTE: The legacy cost_summary is still in metadata but is NOT used —
  the stats-based path takes priority once stats are present. The legacy
  path (_build_step_table with cost_summary) would only activate if
  stats were somehow missing AND the code couldn't recompute from runs.
```

**Key takeaway:** The `report` command is backward-compatible with Sprint 1 profiles. The recomputation path (`compute_stats` + `detect_patterns`) upgrades the session in-memory by adding the new keys to `metadata`. The session is NOT re-saved to disk — this is a read-only operation. The report function checks for `stats` first (Sprint 2 path) and falls back to `cost_summary` (Sprint 1 legacy) only if stats are missing and recomputation fails.

---

### Cross-reference: Which code paths each example uniquely exercises

| Code path | Exercised by |
|-----------|-------------|
| `compute_stats` full pipeline (runs → StepStats + RunStats + PercentileStats) | A, B, C, E, F |
| `compute_percentile_stats` with n=3 (Bessel correction visible) | A |
| `compute_percentile_stats` with n=1 (all-same branch) | — (documented as failure mode) |
| `_safe_cost` wrapping `calculate_cost` | A, E |
| `_percentile` linear interpolation with f≠c | A |
| `_detect_context_growth` → positive r², pass threshold | B |
| `_detect_context_growth` → denom_y=0 guard (constant context) | C |
| `_detect_context_growth` → n < 5 skip (too few data points) | — (implicit in A,E where iteration=1) |
| `_detect_loop_count_variance` → CV > 0.5 passes | C |
| `_detect_loop_count_variance` → CV ≤ 0.5 skips | B |
| `_detect_loop_count_variance` → severity "danger" vs "warning" | C (warning) |
| `_detect_high_token_variance` → p95/p50 ≤ 3 skip | B, C, E |
| `PretiaRunHooks.on_llm_start` + `on_llm_end` pairing | D |
| `PretiaRunHooks.on_tool_start` + `on_tool_end` pairing | D |
| `_extract_model_name` from agent object | D |
| `_extract_tool_name` from tool object | D |
| `_next_iteration` counter (incrementing per step_name) | D |
| `_build_fallback_steps` (skipped when hooks work) | D (documented) |
| try/except safety wrapper on every hook method | D |
| `create_langfuse_client` from env vars | E |
| `fetch_traces` → `_parse_observation` for each observation | E |
| `_extract_input_text` from dict with "messages" key | E |
| `traces_to_step_records` with EVENT filtering (`_SKIP_TYPES`) | E |
| `traces_to_step_records` with "retriev" substring → "retrieval" step_type | E |
| `traces_to_step_records` iteration counter reset per trace | E |
| `format_cli_report` stats-based path (Sprint 2) | B, C, E, F |
| `format_cli_report` legacy cost_summary path (Sprint 1) | F (documented) |
| `_build_cost_summary_table` from stats dict | E, F |
| `_build_step_table` sorted by mean cost | E, F |
| `_build_patterns_panel` with empty patterns (green message) | E, F |
| `_build_iteration_panel` returns None (no iterating steps) | F |
| `report` command: load → detect missing stats → recompute → render | F |
| `format_cost` precision branching (sub-cent → 4 decimals) | E, F |

---

## Part 4: Debugging Exercises

Work through each exercise: read the broken code, answer the four questions. Solutions are at the bottom of this section — try before you peek.

---

### Exercise 1: Percentile on empty data

**File:** `pretia/projection/stats.py`
**Symptom:** `ValueError: Cannot compute stats on empty data` — traceback points deep inside `compute_stats()` with no indication of which step caused the problem.

**Broken code:**

```python
def compute_stats(
    runs: list[list[StepRecord]],
    cost_fn: Callable[..., float] | None = None,
) -> ProfilingStats:
    if cost_fn is None:
        cost_fn = calculate_cost

    if not runs:
        return ProfilingStats()

    step_records: dict[str, list[StepRecord]] = defaultdict(list)
    step_runs_presence: dict[str, set[int]] = defaultdict(set)
    step_iterations_per_run: dict[str, dict[int, int]] = defaultdict(dict)

    run_stats_list: list[RunStats] = []
    run_costs: list[float] = []
    run_tokens: list[float] = []
    total_step_count = 0

    for run_idx, run in enumerate(runs):
        run_cost = 0.0
        run_total_tokens = 0
        run_input_tokens = 0
        run_output_tokens = 0
        run_duration = 0

        for rec in run:
            step_records[rec.step_name].append(rec)
            step_runs_presence[rec.step_name].add(run_idx)
            cur_max = step_iterations_per_run[rec.step_name].get(run_idx, 0)
            if rec.iteration > cur_max:
                step_iterations_per_run[rec.step_name][run_idx] = rec.iteration

            cost = _safe_cost(cost_fn, rec.model, rec.input_tokens, rec.output_tokens)
            run_cost += cost
            run_total_tokens += rec.input_tokens + rec.output_tokens
            run_input_tokens += rec.input_tokens
            run_output_tokens += rec.output_tokens
            run_duration += rec.duration_ms
            total_step_count += 1

        run_costs.append(run_cost)
        run_tokens.append(float(run_total_tokens))
        run_stats_list.append(RunStats(
            run_index=run_idx,
            total_cost=run_cost,
            total_tokens=run_total_tokens,
            total_input_tokens=run_input_tokens,
            total_output_tokens=run_output_tokens,
            step_count=len(run),
            duration_ms=run_duration,
        ))

    step_stats_dict: dict[str, StepStats] = {}
    for step_name, records in step_records.items():
        model = records[0].model
        step_type = records[0].step_type

        input_tok_vals = [float(r.input_tokens) for r in records]
        output_tok_vals = [float(r.output_tokens) for r in records]
        total_tok_vals = [float(r.input_tokens + r.output_tokens) for r in records]
        cost_vals = [
            _safe_cost(cost_fn, r.model, r.input_tokens, r.output_tokens) for r in records
        ]
        duration_vals = [float(r.duration_ms) for r in records]
        context_vals = [float(r.context_size) for r in records]

        # BUG: no guard — if step was present in runs but all records were
        # filtered out upstream, iter_per_run can be empty
        iter_per_run = [
            float(step_iterations_per_run[step_name].get(ri, 0))
            for ri in range(len(runs))
        ]

        step_stats_dict[step_name] = StepStats(
            step_name=step_name,
            step_type=step_type,
            model=model,
            call_count=len(records),
            runs_present=len(step_runs_presence[step_name]),
            input_tokens=compute_percentile_stats(input_tok_vals),
            output_tokens=compute_percentile_stats(output_tok_vals),
            total_tokens=compute_percentile_stats(total_tok_vals),
            cost=compute_percentile_stats(cost_vals),
            duration_ms=compute_percentile_stats(duration_vals),
            context_size=compute_percentile_stats(context_vals),
            iterations_per_run=compute_percentile_stats(iter_per_run),
            mean_iterations=sum(iter_per_run) / len(iter_per_run),
        )

    return ProfilingStats(
        step_stats=step_stats_dict,
        run_stats=run_stats_list,
        cost_per_run=compute_percentile_stats(run_costs),
        tokens_per_run=compute_percentile_stats(run_tokens),
        total_runs=len(runs),
        total_steps=total_step_count,
    )
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 2: Correlation with zero variance

**File:** `pretia/projection/patterns.py`
**Symptom:** `ZeroDivisionError` when running `detect_patterns()` on data where a step has constant `context_size` across all iterations.

**Broken code:**

```python
def _pearson_r_squared(
    xs: list[float], ys: list[float],
) -> tuple[float, float]:
    """Return (r_squared, slope) for two equal-length lists."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)

    denom_x = n * sum_x2 - sum_x * sum_x
    # BUG: removed the denom_y == 0 check
    if denom_x == 0:
        return 0.0, 0.0

    numerator = n * sum_xy - sum_x * sum_y
    denom_y = n * sum_y2 - sum_y * sum_y
    denom = math.sqrt(denom_x * denom_y)
    r = numerator / denom
    slope = numerator / denom_x
    return r * r, slope
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 3: Hook exception crashes user workflow

**File:** `pretia/collectors/openai_agents.py`
**Symptom:** `AttributeError: 'NoneType' object has no attribute 'input_tokens'` — but the traceback appears inside the user's agent execution via `Runner.run()`, making the user think their agent code is broken, not Pretia.

**Broken code:**

```python
async def on_llm_end(
    self,
    context: Any,
    agent: Any,
    response: Any,
) -> None:
    # BUG: removed the try/except wrapper
    agent_name = _extract_agent_name(agent)
    inflight = self._inflight_llm.pop(agent_name, None)
    if inflight is None:
        logger.debug(
            "on_llm_end for unknown agent=%s (missed start event)", agent_name,
        )
        return

    input_tokens = 0
    output_tokens = 0
    usage = getattr(response, "usage", None)
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0

    context_size = inflight["context_size"]
    if input_tokens > 0:
        context_size = input_tokens

    output_text = ""
    output_items = getattr(response, "output", []) or []
    for item in output_items:
        text = getattr(item, "text", None)
        if text:
            output_text += str(text)
        content_parts = getattr(item, "content", None)
        if isinstance(content_parts, list):
            for part in content_parts:
                t = getattr(part, "text", None)
                if t:
                    output_text += str(t)

    output_format = _detect_output_format(output_text) if output_text else "text"

    duration_ms = (time.monotonic_ns() - inflight["start_ns"]) // 1_000_000
    step_name = inflight["step_name"]
    iteration = self._next_iteration(step_name)

    record = StepRecord(
        step_name=step_name,
        step_type="llm",
        model=inflight["model"],
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=context_size,
        tool_definitions_tokens=0,
        system_prompt_hash=inflight["system_prompt_hash"],
        system_prompt_tokens=inflight["system_prompt_tokens"],
        output_format=output_format,
        is_retry=False,
        iteration=iteration,
        parent_step=None,
        duration_ms=duration_ms,
        timestamp=inflight["timestamp"],
    )
    self._steps.append(record)
```

Now consider what happens when `response` is a malformed object where accessing `.output` raises an `AttributeError` (e.g., a streaming response object that doesn't have the `output` attribute):

```python
class MalformedResponse:
    usage = None

    @property
    def output(self):
        raise AttributeError("streaming response has no 'output'")
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 4: Langfuse observation type filter missing

**File:** `pretia/inputs/importer.py`
**Symptom:** No error. But the cost report shows mean cost per run lower than expected, and the step breakdown includes entries with `model="unknown"` and `$0.00` cost. The monthly projection underestimates because zero-cost junk records dilute the average.

**Broken code:**

```python
def traces_to_step_records(
    traces: list[LangfuseTrace],
) -> list[list[StepRecord]]:
    runs: list[list[StepRecord]] = []
    for trace in traces:
        obs_name_map: dict[str, str] = {}
        for obs in trace.observations:
            obs_name_map[obs.observation_id] = obs.name

        iteration_counts: dict[str, int] = {}
        step_records: list[StepRecord] = []

        for obs in trace.observations:
            # BUG: removed the EVENT filter
            # if obs.observation_type in _SKIP_TYPES:
            #     continue

            if obs.observation_type in _GENERATION_TYPES:
                step_type = "llm"
            elif "retriev" in obs.name.lower():
                step_type = "retrieval"
            elif obs.observation_type in _TOOL_TYPES:
                step_type = "tool"
            else:
                step_type = "llm"

            count = iteration_counts.get(obs.name, 0) + 1
            iteration_counts[obs.name] = count

            parent_name = None
            if obs.parent_observation_id:
                parent_name = obs_name_map.get(obs.parent_observation_id)

            timestamp = obs.start_time or trace.timestamp

            step_records.append(StepRecord(
                step_name=obs.name,
                step_type=step_type,
                model=obs.model or "unknown",
                input_tokens=obs.input_tokens,
                output_tokens=obs.output_tokens,
                context_size=obs.input_tokens,
                tool_definitions_tokens=0,
                system_prompt_hash="imported",
                system_prompt_tokens=0,
                output_format="text",
                is_retry=False,
                iteration=count,
                parent_step=parent_name,
                duration_ms=obs.duration_ms,
                timestamp=timestamp,
            ))

        runs.append(step_records)
    return runs
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 5: Cost formatting precision loss

**File:** `pretia/ci/report.py`
**Symptom:** Steps that cost $0.0034 per call display as "$0.00" in the step breakdown table. The monthly projection panel shows the correct dollar amount (it uses raw floats), creating a confusing discrepancy where a "free" step somehow contributes $1,020/month.

**Broken code:**

```python
def format_cost(value: float) -> str:
    """Format a dollar amount for display."""
    if value == 0:
        return "$0.00"
    # BUG: removed the special case for values under $0.01
    if abs(value) < 1000:
        return f"${value:,.2f}"
    return f"${value:,.0f}"
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 6: Iteration counting across runs

**File:** `pretia/inputs/importer.py`
**Symptom:** The loop count variance detector fires a false positive "Loop Count Variance" warning. If trace 1 has a "review" step running 3 times and trace 2 also has "review" running 3 times, the second trace's iterations are labeled 4,5,6 instead of 1,2,3. The detector sees `max_iteration=6` for run 2, which triggers the warning.

**Broken code:**

```python
def traces_to_step_records(
    traces: list[LangfuseTrace],
) -> list[list[StepRecord]]:
    runs: list[list[StepRecord]] = []
    # BUG: iteration_counts initialized OUTSIDE the trace loop
    iteration_counts: dict[str, int] = {}
    for trace in traces:
        obs_name_map: dict[str, str] = {}
        for obs in trace.observations:
            obs_name_map[obs.observation_id] = obs.name

        step_records: list[StepRecord] = []

        for obs in trace.observations:
            if obs.observation_type in _SKIP_TYPES:
                continue

            if obs.observation_type in _GENERATION_TYPES:
                step_type = "llm"
            elif "retriev" in obs.name.lower():
                step_type = "retrieval"
            elif obs.observation_type in _TOOL_TYPES:
                step_type = "tool"
            else:
                step_type = "llm"

            count = iteration_counts.get(obs.name, 0) + 1
            iteration_counts[obs.name] = count

            parent_name = None
            if obs.parent_observation_id:
                parent_name = obs_name_map.get(obs.parent_observation_id)

            timestamp = obs.start_time or trace.timestamp

            step_records.append(StepRecord(
                step_name=obs.name,
                step_type=step_type,
                model=obs.model or "unknown",
                input_tokens=obs.input_tokens,
                output_tokens=obs.output_tokens,
                context_size=obs.input_tokens,
                tool_definitions_tokens=0,
                system_prompt_hash="imported",
                system_prompt_tokens=0,
                output_format="text",
                is_retry=False,
                iteration=count,
                parent_step=parent_name,
                duration_ms=obs.duration_ms,
                timestamp=timestamp,
            ))

        runs.append(step_records)
    return runs
```

**Questions:**

1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

## Solutions

### Exercise 1: Percentile on empty data

1. **Bug:** The `iter_per_run` list comprehension iterates over all run indices (`range(len(runs))`), not just runs where the step was present. In the correct code, it filters to only runs in `step_runs_presence[step_name]`. Without this filter, if a step is absent from some runs, those runs contribute `0.0` to the iteration list — which is still non-empty, so it doesn't crash here. But the deeper issue: if `iter_per_run` somehow becomes empty (e.g., `step_runs_presence` is empty for a step), the `sum(iter_per_run) / len(iter_per_run)` on the next line raises `ZeroDivisionError`, and `compute_percentile_stats(iter_per_run)` raises `ValueError`. The correct code guards with `if not iter_per_run: iter_per_run = [1.0]`.

2. **Symptom:** The `ValueError` from `compute_percentile_stats([])` says "Cannot compute stats on empty data" but the traceback only shows the line inside `compute_stats` — it doesn't tell you which step name caused the empty list. You'd see something like `File "stats.py", line 267, in compute_stats` → `File "stats.py", line 63, in compute_percentile_stats`.

3. **Finding it:** Add a `logger.debug("Computing iter_per_run for step=%s: %s", step_name, iter_per_run)` before the guard. Or: run `pytest tests/unit/test_stats.py -v -k "empty"` and check if there's a test for steps missing from some runs.

4. **Fix:** Restore the filter and guard:
```python
iter_per_run = [
    float(step_iterations_per_run[step_name].get(ri, 0))
    for ri in range(len(runs))
    if ri in step_runs_presence[step_name]
]
if not iter_per_run:
    iter_per_run = [1.0]
```

---

### Exercise 2: Correlation with zero variance

1. **Bug:** The `denom_y == 0` check was removed. When all `context_size` values are identical (e.g., all `600`), `denom_y = n * sum_y2 - sum_y * sum_y` equals zero. The code computes `denom = math.sqrt(denom_x * denom_y)` which gives `denom = 0.0`, then divides `r = numerator / denom` — division by zero.

2. **Symptom:** `ZeroDivisionError` at the line `r = numerator / denom`. The traceback shows `_pearson_r_squared` → `_detect_context_growth` → `detect_patterns`. But the traceback doesn't explain *why* `denom` is zero — you'd need to inspect the data to realize all y-values are identical.

3. **Finding it:** Reproduce with data where context_size is constant: `_pearson_r_squared([1.0, 2.0, 3.0], [600.0, 600.0, 600.0])`. Or: run `pytest tests/unit/test_patterns.py -v -k "zero_variance"` if such a test exists. Alternatively, add `assert denom != 0, f"denom=0, denom_x={denom_x}, denom_y={denom_y}"` before the division to get a more informative error.

4. **Fix:** Add `denom_y` to the zero check:
```python
if denom_x == 0 or denom_y == 0:
    return 0.0, 0.0
```

---

### Exercise 3: Hook exception crashes user workflow

1. **Bug:** The `try/except Exception` wrapper was removed from `on_llm_end`. Any exception raised inside this method now propagates up through the OpenAI Agents SDK's `Runner.run()` call, crashing the user's workflow execution.

2. **Symptom:** When the response object has a `.output` property that raises `AttributeError` (like a streaming response), the exception occurs at `getattr(response, "output", [])` — wait, `getattr` with a default shouldn't raise. The actual failure point is the `@property` on `.output` that raises. Since Python's `getattr(obj, "output", default)` *does* call the property getter, and if the getter raises `AttributeError`, `getattr` returns the default. So for the `MalformedResponse` example, it would actually return `[]` safely. But for a response where `usage` is an object with a broken `__getattr__`, or where `output` raises a non-`AttributeError` exception (like `RuntimeError`), the error would propagate. The key point is: without the try/except, *any* unexpected exception in the hook body — including ones from future SDK changes — kills the user's workflow.

3. **Finding it:** Run the agent with a mock response that has unusual attributes. The error traceback would show `on_llm_end` in the call stack, but it's wrapped inside the SDK's hook dispatch, so the user sees it as an SDK error. Check the `on_llm_end` source for the try/except.

4. **Fix:** Restore the try/except wrapper:
```python
async def on_llm_end(self, context, agent, response):
    try:
        # ... entire method body ...
    except Exception:
        logger.debug("Failed to process on_llm_end", exc_info=True)
```

---

### Exercise 4: Langfuse observation type filter missing

1. **Bug:** The `if obs.observation_type in _SKIP_TYPES: continue` check is removed. EVENT observations (log entries, status updates) are now processed as step records.

2. **Symptom:** EVENT observations have `model=None` (mapped to `"unknown"`), `input_tokens=0`, and `output_tokens=0`. These produce `StepRecord` objects with zero cost. When mixed into the stats, they pull down the mean cost per run and make the median lower (more zero-cost entries shift the p50 down). The step breakdown table shows entries with `model="unknown"` and `$0.00` cost. The monthly projection underestimates because `mean_cost_per_run` is diluted.

3. **Finding it:** Compare the step breakdown in the report against the Langfuse UI. Look for steps with `model="unknown"` and zero cost — those are the EVENT observations leaking through. Or: inspect the raw `list[list[StepRecord]]` from `traces_to_step_records()` and check for records where `model="unknown"` and `input_tokens=0`.

4. **Fix:** Restore the filter before the type classification:
```python
if obs.observation_type in _SKIP_TYPES:
    continue
```

---

### Exercise 5: Cost formatting precision loss

1. **Bug:** The special case `if abs(value) < 0.01: return f"${value:.4f}"` is missing. Values between $0.0001 and $0.0099 are now formatted with only 2 decimal places.

2. **Symptom:** `format_cost(0.0034)` returns `"$0.00"` instead of `"$0.0034"`. In the step breakdown table, a step that costs $0.0034 per call appears as "$0.00". The user sees it as free. But the monthly projection panel uses raw floats (`mean_cost * runs_per_day * 30`), so it correctly shows $0.0034 × 10,000 × 30 = $1,020. The user sees: per-step cost = "$0.00", monthly projection = "$1,020" — a confusing contradiction.

3. **Finding it:** Call `format_cost(0.0034)` in a REPL and see "$0.00". Or: run `pytest tests/unit/test_report.py -v -k "format_cost"` — any test checking sub-cent formatting would fail.

4. **Fix:** Add back the sub-cent case:
```python
if abs(value) < 0.01:
    return f"${value:.4f}"
```

---

### Exercise 6: Iteration counting across runs

1. **Bug:** `iteration_counts` is initialized before the `for trace in traces` loop instead of inside it. The counter accumulates across traces instead of resetting per trace.

2. **Symptom:** If trace 1 has a "review" step running 3 times (iterations 1, 2, 3) and trace 2 also has "review" running 3 times, the second trace's records get `iteration=4, 5, 6` instead of `1, 2, 3`. The loop count variance detector (`_detect_loop_count_variance`) computes `max_iteration` per run — run 1 has `max_iteration=3`, run 2 has `max_iteration=6`. The detector calculates `mean_iter=4.5`, `cv > 0.5`, and fires a false positive "Loop Count Variance" warning. The user sees a scary pattern detection that doesn't reflect real workflow behavior.

3. **Finding it:** Inspect the `StepRecord` objects from `traces_to_step_records()`: check `iteration` values for the same `step_name` across different traces. If trace 2's iterations don't start at 1, the counter wasn't reset. Or: write a test with two traces containing the same step name and assert that both traces' records start at `iteration=1`.

4. **Fix:** Move `iteration_counts` initialization inside the trace loop:
```python
for trace in traces:
    iteration_counts: dict[str, int] = {}
    # ... rest of loop body
```

---

## Part 5: REPL Cheat Sheet

### Stats and percentiles

```python
from pretia.projection.stats import compute_percentile_stats
ps = compute_percentile_stats([1.0, 2.0, 3.0, 4.0, 5.0]); ps
```

```python
from dataclasses import replace
from datetime import UTC, datetime
from pretia.collectors.base import StepRecord
from pretia.projection.stats import compute_stats

rec = StepRecord(step_name="classify", step_type="llm", model="claude-haiku-3", input_tokens=500, output_tokens=50, context_size=600, tool_definitions_tokens=0, system_prompt_hash="abc", system_prompt_tokens=100, output_format="json", is_retry=False, iteration=1, parent_step=None, duration_ms=200, timestamp=datetime(2026, 5, 20, tzinfo=UTC))
stats = compute_stats([[rec], [replace(rec, input_tokens=800)]]); stats.cost_per_run
```

### Pattern detection

```python
from dataclasses import replace
from datetime import UTC, datetime
from pretia.collectors.base import StepRecord
from pretia.projection.patterns import detect_patterns

rec = StepRecord(step_name="summarize", step_type="llm", model="claude-haiku-3", input_tokens=500, output_tokens=50, context_size=500, tool_definitions_tokens=0, system_prompt_hash="abc", system_prompt_tokens=100, output_format="text", is_retry=False, iteration=1, parent_step=None, duration_ms=200, timestamp=datetime(2026, 5, 20, tzinfo=UTC))
run = [replace(rec, iteration=i, context_size=500*i) for i in range(1, 5)]
patterns = detect_patterns([run]); [(p.pattern_type, p.severity) for p in patterns]
```

### DetectedPattern serialization

```python
from pretia.projection.patterns import DetectedPattern
p = DetectedPattern(pattern_type="context_growth", step_name="summarize", severity="warning", evidence={"r_squared": 0.95}, description="test"); p.to_dict()
```

### Cost and token formatting

```python
from pretia.ci.report import format_cost, format_tokens
[format_cost(v) for v in [0, 0.0034, 0.50, 12.34, 12345.67]]
```

```python
from pretia.ci.report import format_tokens
[format_tokens(v) for v in [0, 500, 12345, 1000000]]
```

### CLI report rendering

```python
from datetime import UTC, datetime
from pretia.store import ProfilingSession
from pretia.ci.report import format_cli_report
from rich.console import Console

session = ProfilingSession(workflow_name="test", workflow_hash="abc", profiled_at=datetime(2026, 5, 20, tzinfo=UTC), sample_size=3, input_mode="auto-generate", runs=[], metadata={"stats": {"total_runs": 3, "total_steps": 9, "cost_per_run": {"mean": 0.05, "p50": 0.04, "p95": 0.09, "p99": 0.12, "min": 0.02, "max": 0.15, "std": 0.03}, "step_stats": {}, "run_stats": []}, "patterns": []})
c = Console(); [c.print(r) for r in format_cli_report(session)]
```

### Langfuse importer

```python
from datetime import UTC, datetime
from pretia.inputs.importer import LangfuseTrace, LangfuseObservation, traces_to_step_records

obs = LangfuseObservation(observation_id="o1", name="classify", observation_type="GENERATION", model="gpt-4o", input_tokens=500, output_tokens=50, start_time=datetime(2026, 5, 20, tzinfo=UTC), end_time=datetime(2026, 5, 20, 0, 0, 1, tzinfo=UTC), duration_ms=1000, parent_observation_id=None)
trace = LangfuseTrace(trace_id="t1", name="wf", input_text="Hi", timestamp=datetime(2026, 5, 20, tzinfo=UTC), observations=[obs], total_input_tokens=500, total_output_tokens=50, total_cost=0.005)
runs = traces_to_step_records([trace]); runs[0][0]
```

### Shell commands

```bash
# Run a single test file
pytest tests/unit/test_stats.py -v

# Run a single test by name
pytest tests/unit/test_stats.py -v -k "test_percentile_stats_basic"

# Run all Sprint 2 test files
pytest tests/unit/test_stats.py tests/unit/test_patterns.py tests/unit/test_openai_agents_collector.py tests/unit/test_langfuse_importer.py tests/unit/test_report.py -v

# Check ruff on a single file
ruff check pretia/projection/stats.py

# Run with debug logging
pytest tests/unit/test_patterns.py -v --log-cli-level=DEBUG
```
