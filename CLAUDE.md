# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AgentCost?

AgentCost is a Python SDK for pre-deployment cost analysis of AI agent workflows. It profiles agent workflows, projects costs at scale using distributional statistics (p50-p99, not averages), detects cost time-bombs (context growth, stuck loops), and produces dollar-denominated optimization recommendations.

Status: early development (v0.1.0). Core pipeline is functional: collectors (Generic, LangGraph, OpenAI Agents, Qwen-Agent), input generation/selection/Langfuse import, projection with pattern detection and confidence scoring, validation/backtesting suite, and CLI report rendering are implemented. The `recommend/` module is partially stubbed — `prompts.py` and `verify.py` have implementations, but `heuristics.py`, `classifier.py`, and `rules.py` are stubs. Baseline diffing (`ci/baseline.py`, `ci/diff.py`) is also stubbed.

## Dev Commands

```bash
# Install in dev mode
uv pip install -e ".[dev]"

# Run all unit tests
pytest tests/unit/ -v

# Run a single test file
pytest tests/unit/test_pricing.py -v

# Run a single test by name
pytest tests/unit/test_step_record.py -v -k "test_cost_calculation"

# Integration tests (real LLM calls, costs money, skip in CI)
pytest tests/integration/ -v -m integration

# Lint + format
ruff check agentcost/ tests/
ruff format agentcost/ tests/

# Type check
pyright agentcost/

# Build
python -m build

# Launch local web UI (localhost:7100)
agentcost ui
```

## Architecture

Five-stage pipeline: **Collector → StepRecord → ProfileStore → Projection → Recommendation**.

**ProfileRunner** (`agentcost/runner.py`) orchestrates the full pipeline: load workflow module → auto-detect or select collector → resolve inputs → collect StepRecords → compute cost summary with distributional stats → run projection stats and pattern detection → persist session. The CLI delegates to `ProfileRunner.run_sync()`. Also provides `analyze_langfuse()` for trace-only analysis without re-executing workflows.

**StepRecord** (`agentcost/collectors/base.py`) is the central data structure. Frozen dataclass capturing one LLM call or tool invocation: model, token counts (input, output, context, system prompt, tool definitions), step metadata (type, iteration, parent), and timing. Validates on construction (step_type must be `llm`/`tool`/`retrieval`, output_format must be `json`/`text`/`code`, non-negative token counts, iteration >= 1). Serializes to/from dict for JSON persistence.

**BaseCollector** (`agentcost/collectors/base.py`) is the ABC that framework adapters implement. Single method: `async collect(workflow, inputs) -> list[list[StepRecord]]`. `collect_sync()` wraps for CLI use.

**Collectors** (`agentcost/collectors/`):
- `GenericCollector` — manual instrumentation via `collector.step("name")` returning a `StepTracker` (decorator + async context manager). `_try_extract()` auto-extracts usage from OpenAI/Anthropic response objects.
- `LangGraphCollector` — auto-instruments via `AgentCostCallbackHandler` (LangChain `BaseCallbackHandler`). Tracks inflight calls by UUID, extracts tokens from `llm_output["token_usage"]`.
- `OpenAIAgentsCollector` — auto-instruments via `AgentCostRunHooks` (subclass of `RunHooksBase`). Tracks inflight LLM/tool calls, handles handoffs, includes a `_build_fallback_steps()` path for when hooks capture nothing but `raw_responses` has usage data.
- `QwenAgentCollector` — instruments via `_InstrumentedChatModel`, a transparent proxy that wraps the agent's `.llm` client, intercepts `chat()` calls, and extracts token usage from both OpenAI-format `response.usage` and DashScope `Message.extra["model_service_info"]`. Restores the original LLM client after each run. Falls back to character-count estimation (`len(text) // 4`) when the framework returns no usage metadata.

Optional-dep collectors (`langgraph`, `openai`, `qwen`) use lazy imports via `__getattr__` in `collectors/__init__.py` to avoid import errors when the framework isn't installed.

**Input Resolution** (`agentcost/inputs/`): `select_input_mode()` auto-detects the best input source (explicit → file → single → langfuse → auto-generate → estimate). `generate_inputs()` uses a cheap LLM call to create N diverse synthetic inputs from the workflow's system prompt, targeting 60% typical / 20% edge / 20% adversarial distribution.

**Langfuse Import** (`agentcost/inputs/importer.py`): `create_langfuse_client()` reads credentials from env vars (`LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST`). `fetch_traces()` retrieves recent traces, `extract_inputs()` pulls input strings, and `traces_to_step_records()` converts traces into `list[list[StepRecord]]` for direct analysis without workflow re-execution. Used by both `ProfileRunner.analyze_langfuse()` and the `agentcost analyze` CLI command.

**Projection** (`agentcost/projection/`): `stats.py` computes `ProfilingStats` containing per-step `StepStats` (each with `PercentileStats` for tokens, cost, duration, context, iterations) and per-run `RunStats`. `patterns.py` runs three detectors: context growth (Pearson r² > 0.7 with positive slope), loop count variance (CV > 0.5), and high token variance (p95/p50 > 3). Each returns `DetectedPattern` with severity (`danger`/`warning`) and structured evidence. `projector.py` is the unified entry point — computes `TrafficProjection` for default volumes (100/1K/10K daily), delegates to linear or Monte Carlo based on detected non-linearity, and attaches `ConfidenceResult` from validation. `montecarlo.py` provides `simulate()` and `PercentileProjection`.

**Validation** (`agentcost/validation/`): `confidence.py` computes projection confidence tiers (HIGH/MODERATE/LOW/VERY LOW) from sample size, variance, and detected patterns — deductions for high variance, bonuses for large sample size. `scoring.py` scores projections against actual costs via `CalibrationScore`. `suite.py` runs a full backtesting suite across `BacktestConfig` workflow archetypes, producing `BacktestSuiteResult`. `validate_cmd.py` implements the `agentcost validate` CLI subcommand.

**Backtesting** (`tests/backtesting/`): 10 workflow archetypes (simple/complex support agents, RAG, code generation, etc.) defined as `BacktestConfig` in `configs.py`. Each config specifies expected models, cost ranges, and loop behavior. `run_backtesting.py` executes the full suite. This is a launch gate for v1.

**Graph** (`agentcost/graph/`): Workflow graph extraction (`extractor.py` — DAG structure from LangGraph, OpenAI Agents SDK, CrewAI), layout computation (`layout.py`), color coding (`colorizer.py`), and graph transformation (`transform.py`). Used for visual report output.

**Report** (`agentcost/report/`): HTML report generation. `renderer.py` produces self-contained HTML via Jinja2 + inline SVG. `charts.py` generates chart data. `graph.py` renders workflow graph visualizations.

**ProfileStore** (`agentcost/store.py`) persists `ProfilingSession` objects as JSON files in `.agentcost/`. Files named `{workflow}_{YYYYMMDD_HHMMSS}.json`.

**Pricing** (`agentcost/pricing/tables.py`): Three lookup dicts that must stay in sync: `MODEL_PRICING` (canonical → (input, output) per-million rates), `MODEL_ALIASES` (short names → canonical), `MODEL_TIERS` (canonical → `frontier`/`mid`/`fast`). Structural invariant tests enforce sync.

**CI Report** (`agentcost/ci/report.py`): `format_cli_report()` produces rich-formatted output — header panel, step breakdown table (tier-colored model names), run summary, monthly projections at 100/1K/10K daily calls, and flags for detected issues (loops with max_iteration > 3, high variance where p95 > 3× mean).

**Recommendations** (`agentcost/recommend/`): `prompts.py` generates framework-specific implementation prompts for recommendations. `verify.py` compares old/new profiles to verify which recommendations were applied. `heuristics.py`, `classifier.py`, and `rules.py` are stubs awaiting implementation.

## Public API

`agentcost` exports: `ProfileRunner`, `StepRecord`, `__version__`. The CLI entry point is `agentcost.cli:cli` (registered as the `agentcost` console script).

## Key Design Decisions

- HTML report is Jinja2 + inline SVG — no JS framework dependency at render time. Local web UI is FastAPI + pre-built React bundle. The React source lives in `ui-frontend/` and is compiled to a static bundle shipped inside the pip package. Users never need Node.js or npm.
- Optional framework deps are extras: `pip install agentcost[langgraph]`, `agentcost[openai]`, `agentcost[qwen]`, `agentcost[ui]`, `agentcost[validation]`. Core package depends only on `click`, `rich`, `jinja2`.
- Backtesting is a launch gate, not a nice-to-have. The projection engine is validated against 10 real-world workflow archetypes before v1 ships. If calibration fails, the engine is debugged until it passes. Every projection includes a confidence tier (HIGH/MODERATE/LOW/VERY LOW) derived from sample size, variance, and detected patterns.
- Implementation prompts are tiered by complexity. Model swaps are mechanical (Tier 1). Iteration caps need framework-specific patterns (Tier 2). Architecture changes like compaction insertion need full graph context (Tier 3). All prompts close the optimization loop with a re-profile command.

## Key Conventions

- Python 3.11+. `from __future__ import annotations` in every file.
- `@dataclass` with `slots=True` where possible. No Pydantic.
- Async-first. Sync wrappers via `asyncio.run()` for CLI.
- `ruff` for lint and format (line-length 99, config in `pyproject.toml`). Ruff ignores `S101` (assert in tests) and `ANN401` (Any in collector interfaces). Tests exempt from `ANN` and `S` entirely.
- `click` for CLI. `rich` for user-facing output. `logging` for debug/info. No `print()`.
- Test naming: `test_<module>_<behavior>.py`. Shared fixture `sample_record` in `tests/conftest.py` — use `dataclasses.replace()` to vary fields.
- Docstrings start with a verb. No filler phrases ("This module provides..."). Comments explain WHY, not WHAT.
- Always use most stable or most recent versions of libraries.

## Git Rules

Never commit. Do not run `git add`, `git commit`, `git push`, or any command that writes to git history. Read-only git commands (`git status`, `git diff`, `git log`) are fine.

## Adding a New Model to Pricing

Update three dicts in `agentcost/pricing/tables.py` — they must stay in sync:
1. `MODEL_PRICING`: canonical name → `(input_per_M, output_per_M)` in USD
2. `MODEL_TIERS`: canonical name → `"frontier"` / `"mid"` / `"fast"`
3. `MODEL_ALIASES` (if the model has common short names)

Tests in `tests/unit/test_pricing.py::TestStructuralInvariants` will fail if the dicts drift.

## Adding a New Framework Collector

Create a new file in `agentcost/collectors/` that subclasses `BaseCollector` and implements `async collect()`. Map the framework's native callback/hook system to `StepRecord` fields. Add the framework as an optional dependency group in `pyproject.toml`. Register the lazy import in `collectors/__init__.py` via `__getattr__` and add the name to `__all__`. Add auto-detection heuristics in `ProfileRunner._select_collector()`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Python 3.11 + 3.12 (ubuntu-latest). Steps: `ruff check`, `ruff format --check`, `pytest tests/unit/` (tolerates exit code 5 for uncollected tests during scaffold phase). Integration tests are excluded from CI.
