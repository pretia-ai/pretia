# Changelog

## 1.2.2 (2026-07-22)

- Add `--traffic` flag to `pretia profile run` for custom daily volume projections
- Update tutorial to v1.2 (new SDK collectors, `pretia doctor`, `pretia update-pricing`, removed `StepRecord.cost()`)
- Update README with new CLI commands and 6 framework adapters

## 1.2.1 (2026-07-21)

### Critical fixes

- Fix: P95 projections now use empirical bootstrap resampling instead of linear scaling
- Fix: `float("inf")` in pattern detection no longer crashes JSON serialization (capped to `1e10`)
- Fix: failed runs are filtered from statistics instead of corrupting aggregates
- Fix: `_maybe_wrap_sync` uses `asyncio.to_thread` instead of blocking the event loop
- Fix: LangGraph collector now catches per-run errors instead of aborting the batch
- Fix: `robust_cv` falls back to mean-based CV when median is zero
- Fix: ChatAnthropic token extraction uses `usage_metadata` as primary source
- Fix: input generator retries with exponential backoff on transient API errors

### Medium fixes

- Fix: `asyncio.gather` race condition in SDK collectors — stream wrappers now acquire the same lock used by create wrappers
- Fix: OpenAI streaming responses no longer default to `iteration=1` — iteration value passed through stream capture wrappers
- Fix: deduplicated post-collection pipeline — `run()` and `analyze_langfuse()` share `_post_collect()`
- Fix: all Monte Carlo results stored in `ProjectionResult` (previously only first traffic volume kept)
- Fix: consolidated four `_percentile` implementations into single canonical `percentile()` in `pretia.projection.stats`
- Removed unused `StepRecord.cost()` method (use `calculate_cost()` instead)

### CI/CD

- Added pyright type checking to CI pipeline
- Removed stale exit-code-5 tolerance from pytest step
- Added trusted OIDC publish workflow for PyPI releases

## 1.1.3 (2026-07-01)

- Fix: tiered output token utilization rates (10% for classification, 40% for short gen, 30% for medium, 20% for long) replace flat 60%
- Fix: reduced default user input token estimate from 150 to 50 (matches actual 15-40 token averages)
- Detect `.bind_tools()` chains in AST and estimate 30 output tokens for tool-calling steps
- Estimate now shows cost as a range (low-high) instead of a single point estimate
- Note when models have no `max_tokens` set, explaining the default 500 output tokens

## 1.1.1 (2026-07-01)

- Fix: static estimate no longer deduplicates model calls by name, preserving per-node `max_tokens` (0.9x accuracy for multi-node workflows, was 3.0x)
- Fix: removed incorrect cost scaling multiplier that doubled costs for same-model workflows

## 1.1.0 (2026-07-01)

- Graph-aware static estimate: parses LangGraph `add_node`/`add_edge`/`add_conditional_edges` from AST, weights conditional branches at 1/N
- OpenAI Agents estimate: only prices entry agent unless `handoffs=` detected
- `max_tokens` as output ceiling: `max_tokens=32` estimates 25 output tokens instead of 500
- System prompt variable resolution: resolves module-level string constants (e.g. `system=SYSTEM_PROMPT`)
- Added `"system"` to prompt extraction kwargs (Anthropic SDK compatibility)
- Input token formula: `sp_tokens + 150` instead of `max(sp_tokens, 700)`
- Warm projection: estimates cache-hit cost discount for Anthropic/DeepSeek models with stable system prompts
- Step scaling: same model in multiple graph nodes now correctly multiplies cost

## 1.0.9 (2026-06-29)

- Fix: model swap recommendations now fire for classification steps (expanded keyword stems, lowered threshold from $10 to $1)

## 1.0.8 (2026-06-29)

- Fix: LangGraph node names now resolved from `kwargs`/`metadata` (LangGraph passes `serialized=None`)
- Verified against real LangGraph workflow: steps show "classifier"/"responder" instead of "ChatOpenAI"

## 1.0.7 (2026-06-29)

- Fix: LangGraph auto-detect now picks compiled graph (`app`) over builder (`graph`) when both exist
- Fix: LangGraph steps now labeled by graph node name through intermediate RunnableSequence layers
- Fix: model swap recommendations now fire for LangGraph workflows with classification nodes
- Fix: `create_langfuse_client()` shows friendly install message instead of raw `ModuleNotFoundError`
- Fix: tool steps no longer produce noisy "Unknown model ''" warnings in stats/projection

## 1.0.6 (2026-06-29)

- Fix: LangGraph steps now labeled by graph node name (e.g. "classifier") instead of LLM class name ("ChatOpenAI")
- Fix: model swap recommendations now fire for LangGraph workflows with classification nodes

## 1.0.5 (2026-06-29)

- Fix: `pretia.inputs` subpackage was missing from the published wheel (v1.0.0-1.0.4 broken on PyPI)
- Fix: LangGraph compiled graphs no longer wrapped in async shim, fixing auto-detection of `LangGraphCollector`
- Fix: profiling now errors with a clear message when 0 steps are captured instead of reporting a perfect 100/100 score
- Fix: `--auto-generate 0` now rejected by Click validation instead of hanging
- Fix: `list_sessions()` excludes `baseline.json` so `pretia report latest` doesn't crash on baseline files
- Fix: `--input` can now be passed multiple times for multiple runs
- Fix: Langfuse no longer auto-selected when env vars happen to be set; requires explicit `--from-langfuse`
- Fix: Langfuse profiles now include `pretia_version` and `framework` metadata
- Fix: tool steps with empty model names no longer produce noisy "Unknown model" warnings
- Fix: LICENSE references updated from "AgentCost" to "Pretia"
- Fix: README corrected to say 50 default runs (was 20)
- Add `pretia[langfuse]` optional dependency with friendly error when langfuse is missing
- Add `pretia/__main__.py` for `python -m pretia` support
- Add pricing staleness warning when model pricing data is >30 days old
- Add `anthropic` and `openai` framework detection in `pretia estimate`
- Exclude unimplemented stub modules from the wheel (`pretia/ui/`, graph stubs, `report/graph.py`)

## 1.0.4 (2026-06-29)

- Smarter workflow discovery: finds `run`, `call`, `process`, `execute`, `handle`, `main`, or any solo async callable — no longer limited to `graph`/`workflow`/`agent`/`app`
- New `--entry-point` flag to specify which variable to profile when a file has multiple candidates
- Clear error messages listing available candidates when discovery fails or finds ambiguity
- Sync workflows auto-wrapped for async profiling (no more `TypeError` on plain functions)
- Graceful error recovery: workflow crashes during profiling log the error and continue instead of aborting
- Better error messages for syntax errors in workflow files, corrupted profile JSON, malformed JSONL inputs, binary files passed to `estimate`
- Guard against `None`/empty model names in pricing resolution
- `StepRecord.cost()` now accounts for cache tokens (consistent with `calculate_cost()`)
- Warn when conflicting input flags are provided or system prompt is truncated
- `run_sync()` gives clear error when called from async context (e.g. Jupyter)

## 1.0.3 (2026-06-29)

- Dark theme HTML report redesign (score ring glow, heat-map projection table, vivid cost breakdown bars)
- Conditional hero layout: collapses to "No waste detected" when savings are zero
- Human-readable projection labels (Expected / Likely high / Bad month / Worst case)
- Hide single-step cost breakdown (no value in a 100% bar)
- Remove statistical jargon from pattern descriptions
- Inter font, tighter number spacing, sentence-case labels
- Repo cleanup: removed internal backtesting infrastructure from public repo

## 1.0.2 (2026-06-26)

- Better error messages for missing workflow dependencies (`pip install` suggestion)
- Warn about syntax errors in `pretia estimate` (was silently swallowed)
- Warn about unrecognized models in `pretia estimate` (`register_model()` hint)

## 1.0.1 (2026-06-26)

- Fix: GenericCollector now auto-extracts token usage from raw LLM response objects
- Fix: Dated model versions (e.g. `claude-haiku-4-5-20251001`) resolved correctly
- Fix: Generator model fallback raises clear error instead of sending wrong model to wrong provider
- Fix: Langfuse importer works with SDK v4 (removed invalid `order_by` parameter)
- Fix: Docker action image includes `curl` for PR comment posting
- Fix: RuntimeWarning in CI github module (lazy imports)

## 1.0.0 (2026-06-26)

- Initial release
- Five-stage profiling pipeline: Collector, StepRecord, ProfileStore, Projection, Recommendation
- Four framework collectors: LangGraph, OpenAI Agents SDK, Qwen-Agent, Generic
- Five input modes: static estimate, single input, auto-generate, Langfuse import, file-based
- Distributional projections (p50-p99) with linear and Monte Carlo methods
- Eight pattern detectors (context growth, loop variance, bimodality, etc.)
- Self-contained HTML report with score ring, projection table, cost breakdown, recommendations
- GitHub Action for PR cost comments with threshold enforcement
- CLI commands: estimate, profile run, report, recommend, analyze, baseline, diff, validate
