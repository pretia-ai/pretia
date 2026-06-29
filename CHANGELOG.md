# Changelog

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
