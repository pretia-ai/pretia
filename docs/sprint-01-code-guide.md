# Sprint 1 Code Guide

Developer reference for every file shipped in Sprint 1. Read Part 1 top-to-bottom to understand the data flow, then attempt the exercises in Part 3 before checking the solutions.

---

## Part 1: File-by-File Walkthrough

### 1. `agentcost/collectors/base.py`

#### a) What it does

Defines the two foundational types in AgentCost: `StepRecord` (a frozen dataclass that represents one LLM call or tool invocation) and `BaseCollector` (the abstract base class that every framework adapter must implement). Everything else in the codebase either produces, consumes, or persists `StepRecord` instances.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `StepRecord` (dataclass) | Immutable snapshot of one step in a workflow run. 15 fields covering identity, tokens, timing, and metadata. Validates all invariants on construction via `__post_init__`. | N/A (data container) |
| `StepRecord.__post_init__()` | Guards against invalid data at creation time: `step_type` must be `llm`/`tool`/`retrieval`, `output_format` must be `json`/`text`/`code`, token counts must be non-negative, `iteration` must be >= 1. | `None` (raises `ValueError` on bad input) |
| `StepRecord.total_tokens` (property) | Sum of `input_tokens + output_tokens` for quick reference. | `int` |
| `StepRecord.cost(pricing)` | Computes dollar cost using a per-token pricing dict. This is the *instance-level* cost method — it expects pre-divided per-token prices, not per-million prices. | `float` |
| `StepRecord.to_dict()` | Serializes to a JSON-safe dict, converting `timestamp` to ISO 8601 string. | `dict[str, Any]` |
| `StepRecord.from_dict(data)` | Classmethod that deserializes from a dict, parsing the ISO timestamp back to `datetime`. | `StepRecord` |
| `BaseCollector` (ABC) | Abstract interface with one required method. Framework adapters subclass this. | N/A |
| `BaseCollector.collect(workflow, inputs)` | Abstract async method. Takes a workflow object + list of input strings, returns one list of `StepRecord`s per input run. | `list[list[StepRecord]]` |
| `BaseCollector.collect_sync(workflow, inputs)` | Wraps `collect()` in `asyncio.run()` for CLI/REPL use. | `list[list[StepRecord]]` |

#### c) How data flows through it

Framework collectors (`GenericCollector`, `LangGraphCollector`) subclass `BaseCollector` and implement `collect()`. Inside `collect()`, they run the workflow on each input string and build `StepRecord` instances from the framework's native token/usage data. The return value (`list[list[StepRecord]]`) flows to `ProfileRunner._build_cost_summary()` for cost calculation, and into `ProfilingSession.runs` for persistence.

`StepRecord.to_dict()` and `from_dict()` are called by `ProfilingSession.to_dict()` and `ProfilingSession.from_dict()` during JSON serialization in `ProfileStore`.

`StepRecord.cost()` is *not* used by the main pipeline — `ProfileRunner` calls `calculate_cost()` from `pricing/tables.py` instead. The `cost()` method exists for ad-hoc REPL use when you have a per-token pricing dict handy.

#### d) Common failure modes

1. **Unknown `step_type` or `output_format`.** If a collector passes a typo like `"LLM"` instead of `"llm"`, `__post_init__` raises `ValueError: step_type must be one of ['llm', 'retrieval', 'tool'], got 'LLM'`. This surfaces immediately during collection, before any data is persisted.

2. **Negative token count.** If a framework returns `-1` for a missing token field, `__post_init__` raises `ValueError: input_tokens must be >= 0, got -1`. Symptom: the collector crashes mid-run and no session is saved.

3. **Stale timestamp format in stored JSON.** If you manually edit a saved session and use a non-ISO timestamp string, `StepRecord.from_dict()` raises `ValueError` from `datetime.fromisoformat()`. Symptom: `ProfileStore.load()` fails with a datetime parse error.

#### e) How to debug it

Run just the StepRecord tests:

```bash
pytest tests/unit/test_step_record.py -v
```

Test a specific validation check:

```bash
pytest tests/unit/test_step_record.py -v -k "test_invalid_step_type"
```

Quick REPL check for serialization round-trip:

```python
from agentcost.collectors.base import StepRecord
from datetime import datetime, UTC
r = StepRecord(step_name="test", step_type="llm", model="gpt-4o", input_tokens=100, output_tokens=50, context_size=100, tool_definitions_tokens=0, system_prompt_hash="abc", system_prompt_tokens=20, output_format="text", is_retry=False, iteration=1, parent_step=None, duration_ms=500, timestamp=datetime.now(UTC))
d = r.to_dict()
r2 = StepRecord.from_dict(d)
assert r == r2
```

To see validation in action:

```python
from agentcost.collectors.base import StepRecord
from datetime import datetime, UTC
try:
    StepRecord(step_name="x", step_type="WRONG", model="m", input_tokens=0, output_tokens=0, context_size=0, tool_definitions_tokens=0, system_prompt_hash="", system_prompt_tokens=0, output_format="text", is_retry=False, iteration=1, parent_step=None, duration_ms=0, timestamp=datetime.now(UTC))
except ValueError as e:
    logging.debug(e)
```

---

### 2. `agentcost/pricing/tables.py`

#### a) What it does

Maps LLM model names to per-million-token pricing and capability tiers. Provides functions to resolve aliases (e.g., `"claude-opus-4"` to `"claude-opus-4-20250514"`), calculate the dollar cost of a single call, and look up a model's tier (`frontier`/`mid`/`fast`). This is the cost engine — every dollar figure in AgentCost originates here.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `MODEL_PRICING` (dict) | Canonical model name to `(input_price_per_M, output_price_per_M)` in USD. The source of truth for cost math. | `dict[str, tuple[float, float]]` |
| `MODEL_ALIASES` (dict) | Short/version-stamped names to canonical names. E.g., `"claude-opus-4"` maps to `"claude-opus-4-20250514"`. | `dict[str, str]` |
| `MODEL_TIERS` (dict) | Canonical names to capability tiers. Must contain every key in `MODEL_PRICING`. | `dict[str, str]` |
| `resolve_model(model)` | Resolves a model name to its canonical form. Checks `MODEL_PRICING` first (already canonical?), then `MODEL_ALIASES`. Raises `ValueError` if unknown. | `str` |
| `get_model_pricing(model)` | Calls `resolve_model()`, then divides the per-million prices by 1,000,000 to return per-token prices. | `tuple[float, float]` |
| `calculate_cost(model, input_tokens, output_tokens)` | The main entry point for cost math. Resolves the model, multiplies tokens by per-token prices, rounds to 6 decimal places. | `float` |
| `list_models()` | Returns sorted canonical model names. Useful for CLI help text. | `list[str]` |
| `model_tier(model)` | Returns the capability tier after resolving aliases. | `str` (`"frontier"`, `"mid"`, or `"fast"`) |

#### c) How data flows through it

`ProfileRunner._build_cost_summary()` calls `calculate_cost(rec.model, rec.input_tokens, rec.output_tokens)` for every `StepRecord` in every run. The returned float is accumulated into per-step and per-run cost totals. `ProfileRunner.run()` also calls `model_tier(model)` to annotate each step with its tier for the report.

`resolve_model()` is the choke point — every cost and tier lookup passes through it. If a model name isn't found, the `ValueError` propagates up to `ProfileRunner`, which catches it and falls back to `cost = 0.0`.

#### d) Common failure modes

1. **Unknown model name.** If a `StepRecord` contains a model name like `"gpt-4o-2024-11-20"` that isn't in `MODEL_PRICING` or `MODEL_ALIASES`, `resolve_model()` raises `ValueError: Unknown model 'gpt-4o-2024-11-20'. Available models: [...]`. In the runner pipeline, this is caught and the step gets `cost = 0.0`, which silently under-reports total cost.

2. **Dicts out of sync.** If you add a model to `MODEL_PRICING` but forget `MODEL_TIERS`, `model_tier()` hits a `KeyError` on `MODEL_TIERS[canonical]`. The structural invariant tests in `test_pricing.py` catch this, but only if you run them.

3. **Stale prices.** Pricing changes don't auto-update. If Anthropic drops Haiku pricing by 50%, every AgentCost report overstates Haiku costs until someone updates `MODEL_PRICING`.

#### e) How to debug it

Run all pricing tests:

```bash
pytest tests/unit/test_pricing.py -v
```

Run just the structural invariant tests (the ones that catch dict sync issues):

```bash
pytest tests/unit/test_pricing.py -v -k "TestStructuralInvariants"
```

Quick REPL check for a specific model:

```python
from agentcost.pricing.tables import resolve_model, get_model_pricing, calculate_cost, model_tier
resolve_model("claude-opus-4")          # "claude-opus-4-20250514"
get_model_pricing("claude-opus-4")      # (1.5e-05, 7.5e-05)
calculate_cost("gpt-4o", 1000, 500)     # dollar cost
model_tier("gpt-4o-mini")              # "fast"
```

To check if a model name is recognized:

```python
from agentcost.pricing.tables import resolve_model
try:
    resolve_model("my-custom-model")
except ValueError as e:
    pass  # inspect e
```

---

### 3. `agentcost/store.py`

#### a) What it does

Persists and loads `ProfilingSession` objects as JSON files in the `.agentcost/` directory. A `ProfilingSession` bundles workflow metadata (name, hash, timestamp, input mode) with all the `StepRecord` lists from N profiling runs. `ProfileStore` handles the filesystem operations: save, load, list, and retrieve the latest session for a workflow.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `ProfilingSession` (dataclass) | Holds one profiling session: workflow name, hash, timestamp, sample size, input mode, all runs (each a `list[StepRecord]`), and a free-form `metadata` dict. | N/A (data container) |
| `ProfilingSession.to_dict()` | Serializes the session including all nested `StepRecord.to_dict()` calls. | `dict[str, Any]` |
| `ProfilingSession.from_dict(data)` | Deserializes, reconstructing all `StepRecord` instances from their dicts. | `ProfilingSession` |
| `ProfileStore.__init__(storage_dir)` | Sets the storage directory. Defaults to `.agentcost/` relative to the current working directory. | N/A |
| `ProfileStore.save(session)` | Creates the directory if needed, writes JSON with filename `{workflow_stem}_{YYYYMMDD_HHMMSS}.json`. | `Path` (the saved file) |
| `ProfileStore.load(path)` | Reads a JSON file and deserializes to `ProfilingSession`. | `ProfilingSession` |
| `ProfileStore.list_sessions(workflow_name)` | Lists saved JSON files, newest first (by mtime). Optionally filters by workflow name prefix. | `list[Path]` |
| `ProfileStore.latest(workflow_name)` | Loads the most recent session for a workflow. Returns `None` if no sessions exist. | `ProfilingSession \| None` |
| `ProfileStore._safe_name(workflow_name)` | Strips path and extension to get a stable filename prefix. `"agents/my_agent.py"` becomes `"my_agent"`. | `str` |

#### c) How data flows through it

At the end of `ProfileRunner.run()`, the runner creates a `ProfilingSession` from the collected runs + cost summary metadata, then calls `ProfileStore(storage_dir).save(session)`. This writes the JSON file and returns the path, which is stored in `session.metadata["saved_path"]` for the CLI to display.

For future features (baseline comparison, reports from saved profiles), `ProfileStore.load()` and `ProfileStore.latest()` reconstruct the full session including all `StepRecord` objects.

#### d) Common failure modes

1. **Permission denied on `.agentcost/`.** If the process doesn't have write access to the working directory, `save()` raises `PermissionError` when calling `storage_dir.mkdir()`. Symptom: profiling completes but crashes at the very end when trying to save.

2. **Corrupt JSON file.** If a saved file is manually edited and has invalid JSON, `load()` raises `json.JSONDecodeError`. Symptom: `latest()` or any load call fails.

3. **Workflow name collision.** Two workflows with the same filename stem (`agents/v1/bot.py` and `agents/v2/bot.py`) both map to `"bot"` in `_safe_name()`. Their session files intermingle in `list_sessions()`.

#### e) How to debug it

Run the store tests:

```bash
pytest tests/unit/test_profile_store.py -v
```

REPL check for save/load round-trip:

```python
from agentcost.store import ProfileStore, ProfilingSession
from datetime import datetime, UTC
session = ProfilingSession(workflow_name="test.py", workflow_hash="abc123", profiled_at=datetime.now(UTC), sample_size=0, input_mode="manual", runs=[], metadata={})
store = ProfileStore(storage_dir=__import__("pathlib").Path("/tmp/agentcost_test"))
path = store.save(session)
loaded = store.load(path)
assert loaded.workflow_name == session.workflow_name
```

Inspect what's on disk:

```bash
ls -lt .agentcost/*.json
python -c "import json, sys; d = json.load(open(sys.argv[1])); [__import__('pprint').pprint((k, type(v).__name__)) for k, v in d.items()]" .agentcost/some_session.json
```

---

### 4. `agentcost/collectors/generic.py`

#### a) What it does

Provides manual instrumentation for workflows that don't use a supported framework. Users annotate their code with `@collector.step("name")` decorators or `async with collector.step("name") as s` context managers. The `StepTracker` class captures timing and token data inside each annotated block and builds `StepRecord` instances on exit. `GenericCollector` manages the lifecycle of runs and delegates to `StepTracker` for per-step capture.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `StepTracker` | Async context manager + decorator factory. Captures one step's data. Created by `GenericCollector.step()`. | N/A |
| `StepTracker.record_llm_call(model, input_tokens, ...)` | Stores token data in `self._recorded`. Must be called *inside* the `async with` block — on `__aexit__`, the data is turned into a `StepRecord`. | `None` |
| `StepTracker.__aenter__()` | Starts timing and increments the iteration counter for this step name. | `StepTracker` |
| `StepTracker.__aexit__(...)` | If `record_llm_call()` was called, builds a `StepRecord` from the recorded data + computed duration, and appends it to the collector's current run. If not called, exits silently (no record). | `None` |
| `StepTracker.__call__(fn)` | Decorator mode: wraps an async function so that `_try_extract()` auto-extracts tokens from the return value if `record_llm_call()` wasn't called explicitly. | `wrapped async function` |
| `_try_extract(tracker, result)` | Attempts to pull `usage` data from an OpenAI or Anthropic response object (attribute or dict). Falls back silently on failure. | `None` |
| `GenericCollector` | Manages runs. Holds `_current_run` (accumulating `StepRecord` list) and `_iteration_counters` (per-step iteration numbers). | N/A |
| `GenericCollector.step(name, step_type, parent_step)` | Factory for `StepTracker`. The main user-facing API. | `StepTracker` |
| `GenericCollector.new_run()` | Clears the current run buffer and iteration counters. Call before each workflow execution. | `None` |
| `GenericCollector.end_run()` | Finalizes the current run, appends it to `all_runs`, resets buffers. | `list[StepRecord]` |
| `GenericCollector.collect(workflow, inputs)` | The `BaseCollector` implementation. Iterates inputs, calls `new_run()`, `await workflow(inp)`, collects the current run. Assumes the workflow uses `self.step()` internally. | `list[list[StepRecord]]` |

#### c) How data flows through it

Two usage patterns:

**Pattern A — Context manager (explicit instrumentation):**

The user writes `async with collector.step("classify") as s:` inside their workflow function. Inside the block, they call `s.record_llm_call(model=..., input_tokens=..., ...)`. On `__aexit__`, `StepTracker` builds a `StepRecord` and appends it to `collector._current_run`. After the workflow function returns, `ProfileRunner` (or the user) calls `end_run()` to finalize.

**Pattern B — Decorator (auto-extraction):**

The user decorates their step function with `@collector.step("classify")`. The decorator wraps the function in an `async with self:` block. If the function returns an OpenAI/Anthropic response object, `_try_extract()` auto-extracts the token data. No explicit `record_llm_call()` needed.

Both patterns flow into `collector._current_run`, which is harvested by `collect()` into the final `list[list[StepRecord]]`.

#### d) Common failure modes

1. **`record_llm_call()` called outside the `async with` block.** If you call it *after* the context manager exits, `self._recorded` is set but `__aexit__` already ran. The data is lost — no error, no record, just an empty run.

2. **Forgetting `new_run()` between inputs.** If you manually use `GenericCollector` without calling `new_run()`, records from different inputs accumulate in the same run, mixing data. The `collect()` method handles this automatically, but direct usage can miss it.

3. **Sync function passed to decorator.** If you decorate a regular (non-async) function with `@collector.step("name")`, the wrapper calls `await fn(...)`, which raises `TypeError: object NoneType can't be used in 'await' expression`. The error message is confusing because it doesn't mention the sync/async mismatch.

#### e) How to debug it

Run the generic collector tests:

```bash
pytest tests/unit/test_generic_collector.py -v
```

REPL check for the context manager flow:

```python
import asyncio
from agentcost.collectors.generic import GenericCollector
collector = GenericCollector()
async def demo():
    collector.new_run()
    async with collector.step("test_step") as s:
        s.record_llm_call(model="gpt-4o", input_tokens=100, output_tokens=50)
    run = collector.end_run()
    return run
records = asyncio.run(demo())
assert len(records) == 1
assert records[0].step_name == "test_step"
```

To see `_try_extract` behavior, enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# Then run the collector — _try_extract logs failures at DEBUG level
```

---

### 5. `agentcost/collectors/langgraph.py`

#### a) What it does

Auto-instruments LangGraph (LangChain) workflows by injecting a callback handler into the graph's execution config. `AgentCostCallbackHandler` intercepts LangChain's `on_chat_model_start`, `on_llm_end`, `on_tool_start`, and `on_tool_end` events and converts them into `StepRecord` instances. `LangGraphCollector` wraps this into the `BaseCollector.collect()` interface.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `AgentCostCallbackHandler` | LangChain `BaseCallbackHandler` subclass. Stores in-flight call metadata keyed by `run_id`, then pairs start+end events to build `StepRecord`s. | N/A |
| `AgentCostCallbackHandler.on_chat_model_start(...)` | Captures model name, system prompt, tool definitions, context size estimate, and start timestamp. Stores them in `self._inflight[run_id]`. | `None` |
| `AgentCostCallbackHandler.on_llm_end(response, ...)` | Pairs with the matching `on_chat_model_start`. Extracts input/output tokens from the `LLMResult`, computes duration, detects output format, builds a `StepRecord`, appends to `self.records`. | `None` |
| `AgentCostCallbackHandler.on_tool_start(...)` | Stores tool name and start timestamp in `self._inflight[run_id]`. | `None` |
| `AgentCostCallbackHandler.on_tool_end(output, ...)` | Pairs with `on_tool_start`, builds a tool-type `StepRecord` with zero tokens (tools don't consume LLM tokens). | `None` |
| `AgentCostCallbackHandler._extract_tokens(response)` | Static method. Hunts for token counts in two LangChain locations: `response.llm_output["token_usage"]` and `response.generations[0][0].generation_info["usage"]`. | `tuple[int, int]` |
| `AgentCostCallbackHandler._extract_output_text(response)` | Pulls the LLM's text output for output format detection. | `str` |
| `_estimate_tokens(text)` | Quick heuristic: `len(text) // 4`. Used as fallback when real counts aren't available. | `int` |
| `_detect_output_format(text)` | Classifies output as `"json"` (parseable JSON), `"code"` (contains triple backticks), or `"text"`. | `str` |
| `LangGraphCollector` | `BaseCollector` subclass. Iterates inputs, injects a fresh `AgentCostCallbackHandler` per run, calls `ainvoke` (or `invoke` via thread). | N/A |
| `LangGraphCollector.collect(workflow, inputs)` | Creates one handler per input, runs the graph, collects `handler.records`. Falls back to `asyncio.to_thread(workflow.invoke, ...)` if only sync `invoke` is available. | `list[list[StepRecord]]` |

#### c) How data flows through it

`ProfileRunner._select_collector()` detects LangGraph workflows (objects with both `ainvoke` and `nodes` attributes) and instantiates `LangGraphCollector`. When `collect()` runs:

1. For each input string, a fresh `AgentCostCallbackHandler` is created.
2. The handler is injected via `config={"callbacks": [handler]}`.
3. `workflow.ainvoke({"input": inp}, config=config)` runs the graph.
4. During execution, LangChain fires `on_chat_model_start` → `on_llm_end` pairs (and `on_tool_start` → `on_tool_end` for tool calls).
5. Each pair produces one `StepRecord` appended to `handler.records`.
6. After the run, `handler.records` is copied into the results list.

The handler uses a `_inflight` dict keyed by `UUID` (LangChain's `run_id`) to match start and end events. Each LLM call gets its own unique `run_id`.

#### d) Common failure modes

1. **`langchain-core` not installed.** The module raises `ImportError` at import time (not at runtime): `"LangGraph support requires langchain-core."` This happens as soon as you import the module or select the LangGraph collector, even before any workflow runs.

2. **Token counts are zero.** If the underlying LLM provider doesn't populate `token_usage` in the LangChain response, `_extract_tokens()` returns `(0, 0)`. The `StepRecord` is still created (zero tokens are valid), but cost calculations show $0.00 for that step. Symptom: the report says the workflow is free, which it clearly isn't.

3. **Orphaned start events.** If `on_llm_end` is never called (e.g., the LLM call times out or raises an exception), the entry stays in `_inflight` forever. No `StepRecord` is created for that call. The profiling session under-counts steps for that run.

#### e) How to debug it

Run LangGraph collector tests:

```bash
pytest tests/unit/test_langgraph_collector.py -v
```

To debug callback event flow, enable debug logging before running:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
# All callback errors are caught and logged at DEBUG level
```

Test the token extraction in isolation:

```python
from agentcost.collectors.langgraph import AgentCostCallbackHandler
handler = AgentCostCallbackHandler()
# Simulate an LLMResult with token data
class FakeUsage:
    prompt_tokens = 100
    completion_tokens = 50
class FakeGeneration:
    text = "Hello world"
    generation_info = {"usage": {"prompt_tokens": 100, "completion_tokens": 50}}
class FakeLLMResult:
    llm_output = {"token_usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    generations = [[FakeGeneration()]]
inp, out = handler._extract_tokens(FakeLLMResult())
assert (inp, out) == (100, 50)
```

---

### 6. `agentcost/inputs/selector.py`

#### a) What it does

Determines which input mode to use based on CLI flags and environment state. Implements the priority ladder: explicit inputs > file > single input > Langfuse > auto-generate > static estimate. Returns an `InputSelection` dataclass that tells the runner which mode was selected, what inputs are already available, and a human-readable message.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `InputSelection` (dataclass) | Holds the selected `mode` (string), `inputs` (list of strings, possibly empty if generation is needed), and `message` (for CLI display). | N/A (data container) |
| `read_inputs_file(path)` | Reads a text or JSONL file into a list of input strings. Plain text: one input per non-blank line. JSONL: parses each line, keeps strings as-is, serializes dicts/lists back to JSON strings. | `list[str]` |
| `select_input_mode(...)` | The main decision function. Checks arguments in priority order and returns the first match. Falls back to `_auto_detect()` if no explicit mode is specified. | `InputSelection` |
| `_auto_detect(system_prompt)` | Checks environment for Langfuse credentials (`LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY`), then for any API key (`ANTHROPIC_API_KEY` or `OPENAI_API_KEY`). Falls back to static estimate if nothing is available. | `InputSelection` |

#### c) How data flows through it

`ProfileRunner._resolve_inputs()` calls `select_input_mode()` with the CLI flags. The returned `InputSelection` tells the runner what to do next:

- `mode="auto-generate"` and `inputs=[]` → runner calls `generate_inputs()` to produce the inputs.
- `mode="single"` or `"file"` → `selection.inputs` already contains the data.
- `mode="langfuse"` or `"estimate"` → runner raises `NotImplementedError` (not yet implemented).

The `selection.message` is not currently displayed by the CLI but is available in the session metadata.

#### d) Common failure modes

1. **Inputs file not found.** `read_inputs_file()` raises `FileNotFoundError` if the path doesn't exist. The CLI catches this and shows a clean error.

2. **JSONL parse error.** If a `.jsonl` file has a malformed line, `json.loads()` raises `json.JSONDecodeError`. This is *not* caught by `read_inputs_file()` — it propagates up as an unhandled exception. The CLI catches it in its general `Exception` handler.

3. **Langfuse creds partially set.** `_auto_detect()` requires *both* `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY`. If only one is set, it falls through to auto-generate instead of Langfuse mode — no warning.

#### e) How to debug it

Run the selector tests:

```bash
pytest tests/unit/test_input_selector.py -v
```

Test mode selection logic in the REPL:

```python
from agentcost.inputs.selector import select_input_mode
r = select_input_mode(single_input="test question")
assert r.mode == "single" and r.inputs == ["test question"]

r = select_input_mode(auto_generate=10)
assert r.mode == "auto-generate" and r.inputs == []

r = select_input_mode()  # depends on env vars
r.mode  # "auto-generate" or "estimate" depending on env
```

Test file reading:

```python
from agentcost.inputs.selector import read_inputs_file
# Create a temp file first, then:
inputs = read_inputs_file("/tmp/test_inputs.txt")
```

---

### 7. `agentcost/inputs/generator.py`

#### a) What it does

Generates diverse synthetic test inputs for a workflow by calling a cheap LLM (default: Haiku). Takes the workflow's system prompt, builds a generation prompt asking for varied inputs (typical, edge case, adversarial), calls the LLM, and parses the response into clean strings. Handles both Anthropic and OpenAI SDKs, auto-detecting which to use based on the model name and available API keys.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `generate_inputs(system_prompt, n, model, api_key, additional_context)` | Async. Resolves the provider, builds the prompt, calls the LLM, parses the response. Default: 20 inputs via Haiku. | `list[str]` |
| `generate_inputs_sync(...)` | Wraps `generate_inputs()` in `asyncio.run()`. | `list[str]` |
| `_resolve_provider(model, api_key)` | Determines which SDK to use. Checks model name prefix (`claude-` → Anthropic, `gpt-`/`o1`/`o3` → OpenAI), then checks for SDK installation and API keys. | `tuple[str, str, module]` |
| `_call_anthropic(sdk, api_key, model, prompt)` | Calls Anthropic's `AsyncAnthropic.messages.create()`. | `str` (response text) |
| `_call_openai(sdk, api_key, model, prompt)` | Calls OpenAI's `AsyncOpenAI.chat.completions.create()`. | `str` (response text) |
| `_parse_response(text, n)` | Cleans LLM output: strips preamble lines ("Here are...", "Sure!"), removes numbered prefixes ("1. ", "2) "), drops blank lines, truncates to N. | `list[str]` |
| `_try_import(name)` | Tries to `__import__()` a module; returns `None` on `ImportError`. | `module \| None` |
| `_GENERATION_PROMPT_TEMPLATE` | The prompt template. Instructs the LLM to generate diverse inputs covering typical usage (60%), edge cases (20%), and adversarial inputs (20%). | `str` (constant) |

#### c) How data flows through it

`ProfileRunner._resolve_inputs()` calls `await generate_inputs(system_prompt, n=N)` when the input mode is `"auto-generate"`. The returned `list[str]` becomes the inputs passed to `collector.collect(workflow, inputs)`.

Inside `generate_inputs()`:
1. `_resolve_provider()` picks Anthropic or OpenAI based on the model name and env vars.
2. The system prompt (truncated to 2000 chars) is interpolated into `_GENERATION_PROMPT_TEMPLATE`.
3. The appropriate `_call_*` function makes the async API call.
4. `_parse_response()` cleans the output and returns up to N inputs.

If the LLM returns fewer inputs than requested, `_parse_response()` logs a warning but returns what it got.

#### d) Common failure modes

1. **No SDK installed.** If neither `anthropic` nor `openai` is importable, `_resolve_provider()` raises `ImportError: "Input generation requires either the anthropic or openai package."` This surfaces when the user runs `agentcost profile run` with auto-generate mode.

2. **No API key.** If `ANTHROPIC_API_KEY` and `OPENAI_API_KEY` are both unset (and no key is passed), `_resolve_provider()` raises `ValueError: "No API key found."` The CLI shows this as a clean error.

3. **LLM returns junk.** If the LLM doesn't follow instructions (returns JSON, markdown tables, etc.), `_parse_response()` strips what it can but may return fewer inputs or inputs with artifacts. The warning `"Requested 20 inputs but LLM returned 5"` appears in the logs.

#### e) How to debug it

Run the generator tests (these mock the LLM calls):

```bash
pytest tests/unit/test_input_generator.py -v
```

Test the parser in isolation:

```python
from agentcost.inputs.generator import _parse_response
text = "Here are the inputs:\n1. How do I reset my password?\n2. What's my balance?\n3. Help"
result = _parse_response(text, n=3)
assert result == ["How do I reset my password?", "What's my balance?", "Help"]
```

Test provider resolution (without making API calls):

```python
import os
os.environ["ANTHROPIC_API_KEY"] = "test"
from agentcost.inputs.generator import _resolve_provider
provider, key, sdk = _resolve_provider("claude-haiku-3-5-20241022", None)
assert provider == "anthropic"
```

---

### 8. `agentcost/runner.py`

#### a) What it does

Orchestrates the full profiling pipeline from start to finish. Loads the workflow module from a Python file, auto-detects the workflow object and system prompt, selects the collector and input mode, runs the collector, computes cost summaries, builds a `ProfilingSession`, saves it to disk, and returns the session. This is the engine behind `agentcost profile run`.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `ProfileRunner.__init__(workflow_path, collector, ...)` | Stores all CLI options. No work done here. | N/A |
| `ProfileRunner.run()` | Async. The main entry point — executes the full pipeline. | `ProfilingSession` |
| `ProfileRunner.run_sync()` | Wraps `run()` in `asyncio.run()`. Called by the CLI. | `ProfilingSession` |
| `ProfileRunner._load_workflow()` | Calls `_load_workflow_module()` to import the file, then `_find_workflow()` to locate the graph/agent object and `_extract_system_prompt()` to find a system prompt string. | `tuple[workflow, system_prompt]` |
| `ProfileRunner._select_collector(workflow)` | Picks a collector based on `self.collector_name` or auto-detection (LangGraph if `ainvoke + nodes` attributes exist). | `BaseCollector` |
| `ProfileRunner._resolve_inputs(system_prompt)` | Calls `select_input_mode()` then generates inputs if needed. | `tuple[InputSelection, list[str]]` |
| `_load_workflow_module(path)` | Uses `importlib.util` to dynamically import a `.py` file. | `module` |
| `_find_workflow(module)` | Searches for module-level attributes named `graph`, `workflow`, `agent`, or `app`, then falls back to any object with `ainvoke`/`invoke`. | `object \| None` |
| `_extract_system_prompt(module)` | Scans module attributes for a string > 50 chars matching system prompt patterns. | `str` |
| `_build_cost_summary(runs)` | Computes per-step and per-run cost statistics: mean, min, max, p50, p95, totals, and monthly projections at 100/1000/10000 runs per day. | `dict[str, Any]` |
| `_percentile(values, pct)` | Linear interpolation percentile calculation. | `float` |
| `_get_step_model(runs, step_name)` | Finds the model name used by a specific step (scans runs). | `str` |
| `_get_step_type(runs, step_name)` | Finds the step type for a specific step name. | `str` |

#### c) How data flows through it

`ProfileRunner.run()` is the full pipeline:

```
_load_workflow()
    → (workflow_object, system_prompt)
        ↓
_select_collector(workflow_object)
    → BaseCollector instance
        ↓
_resolve_inputs(system_prompt)
    → (InputSelection, list[str])
        ↓
await collector.collect(workflow, inputs)
    → list[list[StepRecord]]
        ↓
_build_cost_summary(runs)
    → dict with per-step stats, run totals, projections
        ↓
ProfilingSession(... runs=runs, metadata={"cost_summary": cost_summary})
    → session
        ↓
ProfileStore.save(session)
    → Path (saved JSON file)
```

`_build_cost_summary()` iterates all runs and all records. For each `StepRecord`, it calls `calculate_cost(rec.model, rec.input_tokens, rec.output_tokens)` from `pricing/tables.py`. If the model isn't recognized, it catches `ValueError` and uses `cost = 0.0`.

#### d) Common failure modes

1. **Workflow file has no recognizable workflow object.** If the module doesn't have `graph`, `workflow`, `agent`, `app`, or anything with `ainvoke`/`invoke`, `_find_workflow()` returns `None` and the runner raises `click.UsageError` with a helpful message listing expected names.

2. **Dynamic import side effects.** `_load_workflow_module()` calls `spec.loader.exec_module(module)`, which *executes* the workflow file. If that file has top-level code that makes API calls or starts servers, it runs during import. Symptom: unexpected network calls or error messages before profiling even starts.

3. **Empty system prompt.** If `_extract_system_prompt()` doesn't find a string matching the heuristic patterns, it returns `""`. The input generator receives an empty prompt and produces generic, low-quality inputs. Symptom: profiling succeeds but the generated inputs are generic ("Hello", "Help me", etc.) and don't exercise the agent's real capabilities.

#### e) How to debug it

Run the runner tests:

```bash
pytest tests/unit/test_runner.py -v
```

Test workflow loading in isolation:

```python
from agentcost.runner import _load_workflow_module, _find_workflow, _extract_system_prompt
module = _load_workflow_module("path/to/your/workflow.py")
workflow = _find_workflow(module)
prompt = _extract_system_prompt(module)
```

Test cost summary with known data:

```python
from agentcost.runner import _build_cost_summary
from agentcost.collectors.base import StepRecord
from datetime import datetime, UTC
rec = StepRecord(step_name="test", step_type="llm", model="gpt-4o", input_tokens=1000, output_tokens=500, context_size=1000, tool_definitions_tokens=0, system_prompt_hash="x", system_prompt_tokens=100, output_format="text", is_retry=False, iteration=1, parent_step=None, duration_ms=200, timestamp=datetime.now(UTC))
summary = _build_cost_summary([[rec]])
summary["mean_cost_per_run"]
```

---

### 9. `agentcost/cli.py`

#### a) What it does

Defines the Click-based command-line interface. Currently exposes one command: `agentcost profile run <workflow_path>`. Wires CLI options to `ProfileRunner`, runs the profiler, formats the output using `format_cli_report()`, and displays it via `rich.Console`. Handles errors with user-friendly messages.

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `cli()` | The top-level Click group. Entry point for `agentcost` CLI. Decorated with `@click.version_option()`. | N/A |
| `profile()` | Click subgroup for `agentcost profile`. Currently only has `run`. | N/A |
| `run(workflow_path, collector, ...)` | The `agentcost profile run` command. Creates a `ProfileRunner`, calls `runner.run_sync()`, formats the report, handles errors. | `None` (prints to terminal) |
| `console` (module-level) | `rich.Console()` instance used for all terminal output. | N/A |

CLI options:

| Option | What it controls |
|--------|-----------------|
| `--collector` | Force a specific collector: `auto`, `langgraph`, `openai`, `generic` |
| `--auto-generate N` | Generate N synthetic inputs |
| `--input "..."` | Single input string |
| `--inputs file.jsonl` | Path to inputs file |
| `--from-langfuse` | Import from Langfuse traces |
| `--output-dir` | Where to save profile JSON (default: `.agentcost`) |
| `-v / --verbose` | Enable debug logging |

#### c) How data flows through it

User runs `agentcost profile run workflow.py --auto-generate 10`. Click parses the arguments and calls `run()`. Inside `run()`:

1. Logging is configured (DEBUG if `-v`, WARNING otherwise).
2. `ProfileRunner` is constructed with all the CLI options.
3. `runner.run_sync()` executes the full pipeline (see runner.py above).
4. The returned `ProfilingSession` is unpacked: `cost_summary` and `saved_path` come from `session.metadata`.
5. `format_cli_report(session, cost_summary)` returns a list of Rich renderables.
6. Each renderable is printed via `console.print()`.

Error handling: `ImportError` → "Missing dependency", `ValueError`/`FileNotFoundError` → "Error", `NotImplementedError` → "Not yet implemented", `click.UsageError` → re-raised for Click's formatting. All others → generic error with optional traceback via `-v`.

#### d) Common failure modes

1. **Workflow file doesn't exist.** Click's `type=click.Path(exists=True)` catches this *before* the command body runs. The user sees Click's own error: `"Error: Invalid value for 'WORKFLOW_PATH': Path '...' does not exist."`

2. **Missing API key for auto-generate.** The runner tries to generate inputs, which calls `_resolve_provider()`, which raises `ValueError`. The CLI catches it and shows `"Error: No API key found."` without a traceback.

3. **Verbose mode forgotten during debugging.** If a crash happens without `-v`, the traceback is hidden. The user sees only the exception message, which may not be enough. Always use `-v` when debugging.

#### e) How to debug it

Run CLI tests:

```bash
pytest tests/unit/test_cli.py -v
```

Test the CLI without actually profiling (just check help and option parsing):

```bash
agentcost --help
agentcost profile --help
agentcost profile run --help
```

Test with verbose mode to see everything:

```bash
agentcost profile run workflow.py --input "test" -v
```

---

### 10. `agentcost/ci/report.py`

#### a) What it does

Formats a profiling session's cost data into Rich renderables for terminal output. Produces four sections: a header panel (workflow name, run count, input mode, date), a step breakdown table (sorted by cost, color-coded by tier), a run summary table (mean/min/max/p95 cost), and a monthly projection panel. Also detects warning flags (high iteration counts, high variance).

#### b) Key moving parts

| Name | What it does | Returns |
|------|-------------|---------|
| `format_cli_report(session, cost_summary)` | Builds and returns a list of Rich renderables. This is the only public function. | `list[Any]` (Rich Panel, Table, etc.) |
| `_fmt_cost(value)` | Formats a dollar amount: `$0.0012` for small values, `$1,234.56` for large. | `str` |
| `_fmt_tokens(value)` | Formats a number with commas: `1,234`. | `str` |
| `_tier_style(tier)` | Maps tier names to Rich color names: `"fast"` → green, `"mid"` → yellow, `"frontier"` → red. | `str` |
| `_detect_flags(cost_summary)` | Scans per-step stats for warning conditions: iteration count > 3 (potential loop cost), p95/mean ratio > 3 (high variance). | `list[str]` |

#### c) How data flows through it

The CLI's `run()` function calls `format_cli_report(session, cost_summary)` after profiling is complete. The function reads:

- `session.workflow_name`, `session.sample_size`, `session.input_mode`, `session.profiled_at` — for the header.
- `cost_summary["per_step"]` — a dict of step names to stat dicts (cost_mean, cost_p95, input_tokens_mean, model, tier, step_type, count, etc.).
- `cost_summary["mean_cost_per_run"]`, `cost_summary["p95_cost_per_run"]`, etc. — for the summary table.
- `cost_summary["projection_100_day"]`, etc. — for the projection panel.

The returned list of renderables is iterated and printed by the CLI. Tool-type steps get a simplified row (dashes instead of token/cost columns).

#### d) Common failure modes

1. **Empty `cost_summary`.** If the profiling run collected zero steps (e.g., the workflow returned immediately), `cost_summary["per_step"]` is empty. The report renders with no rows in the step table and all zeros in the summary — technically correct but confusing.

2. **Missing keys in `cost_summary`.** The function uses `.get()` with defaults everywhere, so missing keys don't crash. But if `per_step` entries lack `"model"` or `"tier"`, the table shows empty strings instead of "unknown". Not a crash, but misleading.

3. **Division by zero in calls/run.** `stats["count"] / max(session.sample_size, 1)` is safe because of `max(..., 1)`. But if `sample_size` is 0, the "calls/run" column shows the raw count, which is wrong (it should be undefined).

#### e) How to debug it

Run the report tests:

```bash
pytest tests/unit/test_report.py -v
```

Test the formatter in the REPL with fake data:

```python
from agentcost.ci.report import format_cli_report
from agentcost.store import ProfilingSession
from datetime import datetime, UTC
session = ProfilingSession(workflow_name="test.py", workflow_hash="abc", profiled_at=datetime.now(UTC), sample_size=5, input_mode="manual", runs=[], metadata={})
cost_summary = {"per_step": {"classify": {"cost_mean": 0.001, "cost_p95": 0.003, "input_tokens_mean": 340, "output_tokens_mean": 45, "model": "gpt-4o-mini", "tier": "fast", "step_type": "llm", "count": 5, "max_iteration": 1}}, "mean_cost_per_run": 0.001, "min_cost_per_run": 0.0005, "max_cost_per_run": 0.002, "p95_cost_per_run": 0.0018, "total_session_cost": 0.005, "projection_100_day": 3.0, "projection_1000_day": 30.0, "projection_10000_day": 300.0}
renderables = format_cli_report(session, cost_summary)
from rich.console import Console
c = Console()
for r in renderables:
    c.print(r)
```

---

## Part 2: Data Flow Diagram

```
agentcost profile run workflow.py --auto-generate 10
    |
    v
cli.py:run()
    |  constructs ProfileRunner with CLI options
    v
runner.py:ProfileRunner.run_sync()
    |  calls asyncio.run(self.run())
    v
runner.py:ProfileRunner.run()
    |
    |--- _load_workflow()
    |       |  _load_workflow_module(path)      # importlib dynamic import
    |       |  _find_workflow(module)            # scan for graph/workflow/agent/app
    |       |  _extract_system_prompt(module)    # scan for long strings matching patterns
    |       v
    |    (workflow: Any, system_prompt: str)
    |
    |--- _select_collector(workflow)
    |       |  auto: ainvoke + nodes? -> LangGraphCollector
    |       |  else -> GenericCollector
    |       v
    |    collector: BaseCollector
    |
    |--- _resolve_inputs(system_prompt)
    |       |  selector.py:select_input_mode()   # priority: explicit > file > single
    |       |                                     #   > langfuse > auto-generate > estimate
    |       |  if auto-generate:
    |       |      generator.py:generate_inputs()
    |       |          |  _resolve_provider()     # pick anthropic/openai SDK + key
    |       |          |  _call_anthropic() or _call_openai()
    |       |          |  _parse_response()       # strip preamble, numbers, blanks
    |       |          v
    |       |      list[str]  (generated inputs)
    |       v
    |    (InputSelection, inputs: list[str])
    |
    |--- await collector.collect(workflow, inputs)
    |       |  For each input:
    |       |    GenericCollector: new_run() -> await workflow(inp) -> end_run()
    |       |      StepTracker.__aenter__  -> record_llm_call() -> __aexit__
    |       |        -> builds StepRecord -> appends to _current_run
    |       |    LangGraphCollector: handler = AgentCostCallbackHandler()
    |       |      workflow.ainvoke({"input": inp}, config={"callbacks": [handler]})
    |       |        on_chat_model_start -> _inflight[run_id] = {...}
    |       |        on_llm_end -> StepRecord -> handler.records
    |       v
    |    runs: list[list[StepRecord]]
    |
    |--- _build_cost_summary(runs)
    |       |  For each StepRecord:
    |       |    pricing/tables.py:calculate_cost(model, in_tok, out_tok)
    |       |      resolve_model() -> get_model_pricing() -> multiply -> round
    |       |  Aggregate: per-step means/p50/p95, run totals, projections
    |       v
    |    cost_summary: dict
    |
    |--- ProfilingSession(... runs=runs, metadata={"cost_summary": ...})
    |       v
    |    session: ProfilingSession
    |
    |--- store.py:ProfileStore.save(session)
    |       |  session.to_dict() -> StepRecord.to_dict() for each record
    |       |  json.dumps() -> write to .agentcost/{workflow}_{timestamp}.json
    |       v
    |    saved_path: Path
    |
    v  (returns session to CLI)
cli.py:run()
    |
    |--- ci/report.py:format_cli_report(session, cost_summary)
    |       |  Build Rich renderables: header panel, step table, summary, projections
    |       |  _detect_flags(): check for high iterations, high variance
    |       v
    |    list[Rich renderables]
    |
    |--- console.print() for each renderable
    v
Terminal output
```

---

## Part 3: Debugging Exercises

Work through each exercise by reading the broken code, then answer the four questions. Solutions are at the bottom of this file.

---

### Exercise 1: StepRecord validation bypass

**File:** `agentcost/collectors/base.py`
**Symptom:** A `StepRecord` is created with `iteration=0`, which is invalid. No error on construction. Later, `ProfileRunner._build_cost_summary()` computes costs normally, but the step appears as "iteration 0" in the report, confusing the user. Worse, any downstream code that assumes `iteration >= 1` (like loop detection) silently produces wrong results.

**Broken code:**

```python
_VALID_STEP_TYPES = frozenset({"llm", "tool", "retrieval"})
_VALID_OUTPUT_FORMATS = frozenset({"json", "text", "code"})
_NON_NEGATIVE_FIELDS = ("input_tokens", "output_tokens", "context_size", "duration_ms")


@dataclass(frozen=True, slots=True)
class StepRecord:
    step_name: str
    step_type: str
    model: str
    input_tokens: int
    output_tokens: int
    context_size: int
    tool_definitions_tokens: int
    system_prompt_hash: str
    system_prompt_tokens: int
    output_format: str
    is_retry: bool
    iteration: int
    parent_step: str | None
    duration_ms: int
    timestamp: datetime

    def __post_init__(self) -> None:
        if self.step_type not in _VALID_STEP_TYPES:
            raise ValueError(
                f"step_type must be one of {sorted(_VALID_STEP_TYPES)}, got {self.step_type!r}"
            )
        if self.output_format not in _VALID_OUTPUT_FORMATS:
            raise ValueError(
                f"output_format must be one of {sorted(_VALID_OUTPUT_FORMATS)}, "
                f"got {self.output_format!r}"
            )
        for name in _NON_NEGATIVE_FIELDS:
            value: int = getattr(self, name)
            if value < 0:
                raise ValueError(f"{name} must be >= 0, got {value}")
        if self.iteration < 0:
            raise ValueError(f"iteration must be >= 1, got {self.iteration}")
```

**Questions:**
1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 2: Frozen dataclass mutation attempt

**File:** `agentcost/runner.py`
**Symptom:** During cost summary computation, the code crashes with `dataclasses.FrozenInstanceError: cannot assign to field 'model'`. The profiling session is lost — the JSON file is never saved.

**Broken code:**

```python
def _build_cost_summary(
    runs: list[list[StepRecord]],
) -> dict[str, Any]:
    step_costs: dict[str, list[dict[str, Any]]] = {}
    run_totals: list[float] = []

    for run in runs:
        run_cost = 0.0
        for rec in run:
            # Normalize model name before cost lookup
            from agentcost.pricing.tables import resolve_model
            try:
                rec.model = resolve_model(rec.model)
            except ValueError:
                pass

            try:
                cost = calculate_cost(
                    rec.model, rec.input_tokens, rec.output_tokens,
                )
            except ValueError:
                cost = 0.0

            entry = {
                "cost": cost,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "duration_ms": rec.duration_ms,
                "iteration": rec.iteration,
            }
            step_costs.setdefault(rec.step_name, []).append(entry)
            run_cost += cost
        run_totals.append(run_cost)

    # ... rest of function
```

**Questions:**
1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 3: Async/sync confusion

**File:** `agentcost/runner.py`
**Symptom:** `ProfileRunner.run()` completes without errors, but the `runs` variable is a `list` containing a coroutine object instead of a `list[list[StepRecord]]`. The next line, `_build_cost_summary(runs)`, iterates over the runs and gets zero steps. The profiling session is saved with an empty cost summary and no step data.

**Broken code:**

```python
class ProfileRunner:
    async def run(self) -> ProfilingSession:
        workflow, system_prompt = self._load_workflow()
        collector = self._select_collector(workflow)
        selection, inputs = await self._resolve_inputs(system_prompt)
        runs = collector.collect(workflow, inputs)

        cost_summary = _build_cost_summary(runs)

        for step_name in cost_summary["per_step"]:
            model = _get_step_model(runs, step_name)
            # ... rest of method
```

**Questions:**
1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 4: Pricing lookup failure

**File:** `agentcost/pricing/tables.py`
**Symptom:** When profiling a workflow that uses `"claude-sonnet-4"`, cost calculation crashes with `KeyError: 'claude-sonnet-4'` deep inside `calculate_cost()`. The error is confusing because `"claude-sonnet-4"` is a valid alias — it should resolve to `"claude-sonnet-4-20250514"`.

**Broken code:**

```python
def resolve_model(model: str) -> str:
    """Return the canonical model name, resolving aliases."""
    if model in MODEL_PRICING:
        return model
    if model in MODEL_ALIASES:
        return model  # Bug: returns the alias, not the canonical name
    raise ValueError(f"Unknown model {model!r}. Available models: {sorted(MODEL_PRICING)}")


def get_model_pricing(model: str) -> tuple[float, float]:
    """Return (input_price_per_token, output_price_per_token) for the model."""
    canonical = resolve_model(model)
    per_m_input, per_m_output = MODEL_PRICING[canonical]
    return per_m_input / _PER_MILLION, per_m_output / _PER_MILLION
```

**Questions:**
1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 5: Context manager not recording

**File:** User code using `agentcost/collectors/generic.py`
**Symptom:** The user instruments their workflow with `GenericCollector`, but after profiling, the session shows zero steps per run. No errors, no warnings. The report shows all zeros.

**Broken code:**

```python
import asyncio
from agentcost.collectors.generic import GenericCollector

collector = GenericCollector()

async def my_workflow(user_input: str) -> str:
    collector.new_run()

    async with collector.step("classify_intent") as s:
        # ... LLM call happens here ...
        response = await call_llm(model="gpt-4o", prompt=user_input)

    # Record the token usage after the context manager exits
    s.record_llm_call(
        model="gpt-4o",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )

    run = collector.end_run()
    return response.text
```

**Questions:**
1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

### Exercise 6: Input generator response parsing

**File:** `agentcost/inputs/generator.py`
**Symptom:** Generated inputs contain numbered prefixes like `"1. How do I reset my password?"`. These get sent verbatim to the agent workflow as test inputs. The agent may treat the `"1. "` as part of the user's question, producing different (worse) profiling results than real user inputs would.

**Broken code:**

```python
_PREAMBLE_PATTERNS = re.compile(
    r"^(here are|below are|these (are|inputs)|the following|sure[,!]|"
    r"of course|certainly|i'?ll generate)",
    re.IGNORECASE,
)


def _parse_response(text: str, n: int) -> list[str]:
    """Extract clean inputs from an LLM response."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _PREAMBLE_PATTERNS.search(line):
            continue
        # Forgot to strip numbered prefixes!
        if line:
            lines.append(line)

    if len(lines) < n:
        logger.warning(
            "Requested %d inputs but LLM returned %d", n, len(lines),
        )
    return lines[:n]
```

**Questions:**
1. What is the bug?
2. Why does it cause this specific symptom?
3. How would you find this bug if you didn't know it was there?
4. Write the fix (one or two lines).

---

## Solutions

### Exercise 1 Solution

**Bug:** The iteration check uses `< 0` instead of `< 1`. This means `iteration=0` passes validation, even though the field is documented as 1-indexed and `iteration >= 1` is the intended invariant.

**Why this symptom:** `iteration=0` is technically non-negative, so it slips through. But the rest of the codebase (loop detection, iteration counting in `StepTracker._next_iteration()`, report flags) assumes iteration starts at 1. An iteration of 0 means "this step never ran," which is contradictory for a record that exists.

**How to find it:** Run the test `pytest tests/unit/test_step_record.py -v -k "test_iteration"`. A test that constructs a `StepRecord(iteration=0)` and expects `ValueError` would fail. You could also catch it by reading `__post_init__` and noticing the asymmetry: the error message says "must be >= 1" but the check is `< 0`.

**Fix:**

```python
# Change this line:
if self.iteration < 0:
# To:
if self.iteration < 1:
```

---

### Exercise 2 Solution

**Bug:** The code tries to assign to `rec.model` on a frozen dataclass (`StepRecord` is `@dataclass(frozen=True)`). Frozen dataclasses prohibit field mutation after construction.

**Why this symptom:** Python raises `FrozenInstanceError` the moment you try `rec.model = ...`. This happens inside `_build_cost_summary()`, which is called after collection is complete but before the session is saved. The crash kills the entire run — collected data is lost.

**How to find it:** The traceback points directly at `rec.model = resolve_model(rec.model)` with `FrozenInstanceError`. If you know `StepRecord` is frozen, the bug is obvious on sight. If you don't, search for `frozen=True` in `base.py`.

**Fix:** Don't mutate the record. Resolve the model name in a local variable:

```python
model = resolve_model(rec.model) if rec.model else rec.model
# ... then use model instead of rec.model in calculate_cost:
cost = calculate_cost(model, rec.input_tokens, rec.output_tokens)
```

Or, since `calculate_cost()` already calls `resolve_model()` internally, just remove the normalization entirely — it's redundant.

---

### Exercise 3 Solution

**Bug:** `collector.collect(workflow, inputs)` is an `async` method, but the call is missing `await`. Without `await`, the expression returns a coroutine object instead of executing the method.

**Why this symptom:** The coroutine object is assigned to `runs`. When `_build_cost_summary(runs)` tries to iterate over it, it doesn't crash (a coroutine is iterable in the sense that `for run in runs` produces zero iterations for a non-list iterable, or raises on the first attempt). The result is an empty cost summary. No error because `_build_cost_summary` handles empty input gracefully.

**How to find it:** Add `logging.debug("runs type: %s, length: %s", type(runs), len(runs) if hasattr(runs, '__len__') else 'N/A')` after the `collect()` call. You'd see `type: <class 'coroutine'>`. Alternatively, Python 3.12+ may emit a `RuntimeWarning: coroutine 'collect' was never awaited`, visible with `-v`.

**Fix:**

```python
# Change:
runs = collector.collect(workflow, inputs)
# To:
runs = await collector.collect(workflow, inputs)
```

---

### Exercise 4 Solution

**Bug:** In `resolve_model()`, the alias branch returns `model` (the alias itself) instead of `MODEL_ALIASES[model]` (the canonical name). So `resolve_model("claude-sonnet-4")` returns `"claude-sonnet-4"` — which isn't in `MODEL_PRICING`.

**Why this symptom:** `get_model_pricing()` calls `resolve_model()`, gets back `"claude-sonnet-4"`, then does `MODEL_PRICING["claude-sonnet-4"]`, which raises `KeyError`. The `ValueError` with the helpful "Unknown model" message is never triggered because `resolve_model()` *thinks* it resolved the alias successfully. The user sees a raw `KeyError` instead of a descriptive error.

**How to find it:** Test with any alias: `resolve_model("claude-opus-4")` should return `"claude-opus-4-20250514"`, but returns `"claude-opus-4"`. A simple assert in a REPL or unit test catches this immediately.

**Fix:**

```python
# Change:
if model in MODEL_ALIASES:
    return model
# To:
if model in MODEL_ALIASES:
    return MODEL_ALIASES[model]
```

---

### Exercise 5 Solution

**Bug:** `s.record_llm_call()` is called *after* the `async with` block has exited. `StepTracker.__aexit__()` already ran and checked `self._recorded`, which was `None` at that point. So `__aexit__` returned without creating a `StepRecord`. The subsequent `record_llm_call()` sets `self._recorded`, but nobody ever reads it — `__aexit__` is done.

**Why this symptom:** `__aexit__` has this guard: `if self._recorded is None: return`. When `record_llm_call()` hasn't been called yet (inside the block), `_recorded` is `None`, so `__aexit__` skips record creation. The call *after* the block sets `_recorded` too late. No error, no warning — just zero records.

**How to find it:** Add `logging.debug("record_llm_call at %s, _recorded=%s", time.monotonic(), self._recorded)` in `record_llm_call` and `logging.debug("__aexit__ _recorded=%s", self._recorded)` in `__aexit__`. You'd see `__aexit__` fires with `_recorded=None`, then `record_llm_call` fires afterward. Also: inspect `collector.end_run()` — if it returns an empty list but LLM calls happened, the recording is misplaced.

**Fix:** Move `record_llm_call()` inside the `async with` block:

```python
async with collector.step("classify_intent") as s:
    response = await call_llm(model="gpt-4o", prompt=user_input)
    s.record_llm_call(
        model="gpt-4o",
        input_tokens=response.usage.prompt_tokens,
        output_tokens=response.usage.completion_tokens,
    )
```

---

### Exercise 6 Solution

**Bug:** The `_NUMBERED_PREFIX` regex substitution line is missing. The original code has `line = _NUMBERED_PREFIX.sub("", line).strip()`, but the broken version omits this step. LLM responses frequently number their outputs ("1. ...", "2) ..."), and without stripping, those prefixes become part of the test inputs.

**Why this symptom:** The LLM follows its training pattern and numbers the outputs despite the prompt saying "No numbering." The `_parse_response()` function passes these numbered lines through unchanged. The agent workflow receives inputs like `"1. How do I reset my password?"`, which may confuse it or change its behavior compared to clean inputs, producing misleading profiling data.

**How to find it:** Inspect the generated inputs: `logging.debug("Generated inputs: %s", inputs)` after `_parse_response()` returns. If you see `"1. ..."` prefixes, the stripping step is broken or missing. You could also write a unit test for `_parse_response()` with numbered input and assert the output is clean.

**Fix:** Add the numbered prefix stripping line back:

```python
if _PREAMBLE_PATTERNS.search(line):
    continue
line = _NUMBERED_PREFIX.sub("", line).strip()
if line:
    lines.append(line)
```

---

## Part 4: REPL Cheat Sheet

### Create a StepRecord and inspect it

```python
from agentcost.collectors.base import StepRecord; from datetime import datetime, UTC
r = StepRecord(step_name="classify", step_type="llm", model="gpt-4o", input_tokens=500, output_tokens=120, context_size=800, tool_definitions_tokens=0, system_prompt_hash="abc123", system_prompt_tokens=200, output_format="json", is_retry=False, iteration=1, parent_step=None, duration_ms=340, timestamp=datetime.now(UTC))
r.step_name, r.total_tokens, r.model
```

### Serialize and deserialize a StepRecord

```python
d = r.to_dict(); r2 = StepRecord.from_dict(d); assert r == r2; d
```

### Look up pricing for a model

```python
from agentcost.pricing.tables import resolve_model, get_model_pricing, model_tier
resolve_model("claude-opus-4"), get_model_pricing("claude-opus-4"), model_tier("claude-opus-4")
```

### Calculate cost for known tokens

```python
from agentcost.pricing.tables import calculate_cost
calculate_cost("gpt-4o", input_tokens=5000, output_tokens=1500)
```

### Create a ProfilingSession with fake data and save it

```python
from agentcost.store import ProfileStore, ProfilingSession; from pathlib import Path; from datetime import datetime, UTC
session = ProfilingSession(workflow_name="demo.py", workflow_hash="abc", profiled_at=datetime.now(UTC), sample_size=1, input_mode="manual", runs=[[r]], metadata={})
path = ProfileStore(storage_dir=Path("/tmp/agentcost_test")).save(session); path
```

### Load a saved session and inspect its runs

```python
loaded = ProfileStore(storage_dir=Path("/tmp/agentcost_test")).load(path)
loaded.workflow_name, len(loaded.runs), len(loaded.runs[0])
```

### Call the input selector with different flags

```python
from agentcost.inputs.selector import select_input_mode
select_input_mode(single_input="hello").mode
select_input_mode(auto_generate=10).mode
select_input_mode(inputs_file="data.jsonl").mode
select_input_mode().mode
```

### Run a single test file in verbose mode

```bash
pytest tests/unit/test_step_record.py -v
```

### Run a single test function by name

```bash
pytest tests/unit/test_step_record.py -v -k "test_cost_calculation"
```

### Check ruff on a single file

```bash
ruff check agentcost/collectors/base.py
```
