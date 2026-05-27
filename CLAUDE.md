# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AgentCost?

AgentCost is a Python SDK for pre-deployment cost analysis of AI agent workflows. It profiles agent workflows, projects costs at scale using distributional statistics (p50-p99, not averages), detects cost time-bombs (context growth, stuck loops), and produces dollar-denominated optimization recommendations.

Status: early development (v0.1.0). Core pipeline is functional: `StepRecord`, `ProfilingSession`, `ProfileStore`, pricing tables, `ProfileRunner` (orchestrator), collectors (Generic, LangGraph), input generation/selection, and CLI report rendering are implemented. Projection, recommendation, and baseline diffing modules are stubs.

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

**ProfileRunner** (`agentcost/runner.py`) orchestrates the full pipeline: load workflow module → auto-detect or select collector → resolve inputs (auto-generate, single, file, langfuse) → collect StepRecords → build cost summary with distributional stats (mean, min, max, p50, p95) → persist session. The cost summary includes per-step breakdowns and monthly projections at 100/1000/10000 daily calls. This is the main entry point — the CLI delegates to `ProfileRunner.run_sync()`.

**StepRecord** (`agentcost/collectors/base.py`) is the central data structure. Frozen dataclass capturing one LLM call or tool invocation: model, token counts (input, output, context, system prompt, tool definitions), step metadata (type, iteration, parent), and timing. Validates on construction (step_type must be `llm`/`tool`/`retrieval`, output_format must be `json`/`text`/`code`, non-negative token counts, iteration >= 1). Serializes to/from dict for JSON persistence.

**BaseCollector** (`agentcost/collectors/base.py`) is the ABC that framework adapters implement. Single method: `async collect(workflow, inputs) -> list[list[StepRecord]]`. Each framework adapter (LangGraph, OpenAI Agents, Generic) maps native callbacks/hooks to StepRecords. `collect_sync()` wraps for CLI use.

**GenericCollector** (`agentcost/collectors/generic.py`) provides manual instrumentation via `collector.step("name")` which returns a `StepTracker` that works as both a decorator and async context manager. `StepTracker.record_llm_call()` captures token usage, and `_try_extract()` auto-extracts usage metadata from OpenAI/Anthropic response objects.

**LangGraphCollector** (`agentcost/collectors/langgraph.py`) auto-instruments LangGraph workflows by injecting an `AgentCostCallbackHandler` (subclass of LangChain's `BaseCallbackHandler`). Tracks inflight LLM/tool calls by UUID, extracts tokens from `llm_output["token_usage"]`, and detects output format (json/code/text) via regex.

**Input Resolution** (`agentcost/inputs/`): `select_input_mode()` auto-detects the best input source (explicit → file → single → langfuse → auto-generate → estimate). `generate_inputs()` uses a cheap LLM call (Claude Haiku or GPT-4o-mini) to create N diverse synthetic inputs from the workflow's system prompt, targeting 60% typical / 20% edge / 20% adversarial distribution.

**ProfileStore** (`agentcost/store.py`) persists `ProfilingSession` objects as JSON files in `.agentcost/`. Sessions contain workflow metadata + all StepRecords from N runs. Files named `{workflow}_{YYYYMMDD_HHMMSS}.json`.

**Pricing** (`agentcost/pricing/tables.py`) maps model names to per-million-token costs. Three lookup layers: `MODEL_PRICING` (canonical names → (input, output) per-million rates), `MODEL_ALIASES` (short names → canonical), `MODEL_TIERS` (canonical → `frontier`/`mid`/`fast`). All three dicts must stay in sync — structural invariant tests enforce this.

**CI Report** (`agentcost/ci/report.py`): `format_cli_report()` produces rich-formatted output — header panel, step breakdown table (with tier-colored model names), run summary, monthly projections, and flags for detected issues (loops with max_iteration > 3, high variance where p95 > 3× mean).

## Public API

`agentcost` exports: `ProfileRunner`, `StepRecord`, `__version__`. The CLI entry point is `agentcost.cli:cli` (registered as the `agentcost` console script).

## Key Design Decisions

- HTML report is Jinja2 + inline SVG — no JS framework dependency at render time. Local web UI is FastAPI + pre-built React bundle. The React source lives in `ui-frontend/` and is compiled to a static bundle shipped inside the pip package. Users never need Node.js or npm.

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

Create a new file in `agentcost/collectors/` that subclasses `BaseCollector` and implements `async collect()`. Map the framework's native callback/hook system to `StepRecord` fields. Each collector is isolated — framework API changes affect only one file.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Python 3.11 + 3.12 (ubuntu-latest). Steps: `ruff check`, `ruff format --check`, `pytest tests/unit/` (tolerates exit code 5 for uncollected tests during scaffold phase). Integration tests are excluded from CI.
