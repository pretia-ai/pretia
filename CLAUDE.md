# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AgentCost?

AgentCost is a Python SDK for pre-deployment cost analysis of AI agent workflows. It profiles agent workflows, projects costs at scale using distributional statistics (p50-p99, not averages), detects cost time-bombs (context growth, stuck loops), and produces dollar-denominated optimization recommendations.

Status: early development (v0.1.0). Core pipeline is functional: collectors (Generic, LangGraph, OpenAI Agents, Qwen-Agent), input generation/selection/Langfuse import, projection with pattern detection and confidence scoring, validation/backtesting suite, baseline diffing (`ci/baseline.py`, `ci/diff.py`), and CLI report rendering are implemented. The `recommend/` module is entirely stubbed — `heuristics.py`, `classifier.py`, `rules.py`, `prompts.py`, and `verify.py` are all stubs awaiting implementation. The web UI module (`agentcost/ui/`, `ui-frontend/`) exists but is not yet wired into the CLI.

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
```

### CLI Commands

```bash
agentcost profile run <workflow_path>   # Profile a workflow (--collector, --auto-generate, --input, --from-langfuse, --allow-cache)
agentcost report <profile_path>         # Generate HTML report from a saved profile
agentcost analyze --from-langfuse       # Analyze Langfuse traces without re-executing (--last N, --name, --traffic)
agentcost baseline update <profile>     # Save a cost baseline for CI diffing
agentcost diff <baseline> <profile>     # Compare baseline vs new profile, show cost deltas
agentcost validate <workflow_path>      # Run projection quality check (small-n vs large-n)
agentcost update-pricing                # Placeholder — manual edit of pricing/tables.py
```

## Architecture

Five-stage pipeline: **Collector → StepRecord → ProfileStore → Projection → Recommendation**.

**ProfileRunner** (`agentcost/runner.py`) orchestrates the full pipeline. The CLI delegates to `ProfileRunner.run_sync()`. Also provides `analyze_langfuse()` for trace-only analysis without re-executing workflows.

**StepRecord** (`agentcost/collectors/base.py`) is the central data structure — frozen dataclass capturing one LLM call or tool invocation. v2 added optional fields for cache-aware costing (`cache_hit_tokens`, `cache_miss_tokens`), tool metadata (`tool_name`, `tool_input_tokens`, `tool_output_tokens`, `tool_success`, `tool_retry_count`), and model config (`model_version`, `temperature`, `max_tokens_setting`). All v2 fields default to `None` for backward compatibility. Cache token fields feed into `calculate_cost()` for differentiated cache-hit pricing.

**BaseCollector** (`agentcost/collectors/base.py`): ABC with `async collect(workflow, inputs) -> list[list[StepRecord]]`. Four implementations: Generic (manual instrumentation), LangGraph (callback handler), OpenAI Agents (RunHooks), Qwen-Agent (LLM proxy). Optional-dep collectors use lazy imports via `__getattr__` in `collectors/__init__.py`.

**Input Resolution** (`agentcost/inputs/`): `select_input_mode()` auto-detects the best input source (explicit → file → single → langfuse → auto-generate → estimate). `generate_inputs()` targets 60% typical / 20% edge / 20% adversarial distribution. Langfuse import reads `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` from env vars.

**Projection** (`agentcost/projection/`): `stats.py` computes distributional statistics (per-step and per-run). `patterns.py` runs five detectors: context growth (dual Pearson+Spearman), loop count variance (robust CV), high token variance (p95/p50), step count variance (routing variability), and bimodality (GMM, requires sklearn). `projector.py` computes `TrafficProjection` for default volumes (100/1K/10K daily), delegates to linear or Monte Carlo based on detected non-linearity.

**Validation** (`agentcost/validation/`): Confidence tiers (HIGH/MODERATE/LOW/VERY LOW) from sample size, variance, and patterns. Backtesting suite in `tests/backtesting/` with 10 workflow archetypes — this is a launch gate for v1.

**Cache Busting** (`agentcost/collectors/cache_bust.py`): Default profiling mode (`--allow-cache` off) appends a unique suffix to prompts to defeat server-side caching, ensuring cold-start cost measurements.

**Pricing** (`agentcost/pricing/tables.py`): Four lookup dicts that must stay in sync: `MODEL_PRICING` (canonical → (input, output) per-million rates), `MODEL_ALIASES` (short names → canonical), `MODEL_TIERS` (canonical → `frontier`/`mid`/`fast`), `MODEL_CACHE_HIT_PRICING` (canonical → input cache-hit rate per M, currently DeepSeek only). `calculate_cost()` uses cache-hit pricing when both `cache_hit_tokens` and `cache_miss_tokens` are provided.

**Graph** (`agentcost/graph/`): Workflow graph extraction and visualization. `extractor.py` builds a graph representation from StepRecords. `layout.py` positions nodes. `colorizer.py` maps cost/token metrics to visual properties. `transform.py` converts between graph formats.

**Report** (`agentcost/report/`): HTML report rendering pipeline. `renderer.py` produces the Jinja2-based HTML report. `charts.py` generates inline SVG charts. `graph.py` renders the workflow graph visualization for inclusion in reports.

**GitHub Action** (`action/`): `action.yml` + `entrypoint.sh` — comments on PRs touching agent workflow code with projected cost changes.

**Recommendations** (`agentcost/recommend/`): Entirely stubbed — all five files are single-line stubs awaiting implementation.

### Backtesting Infrastructure

The backtesting suite validates the projection engine against real-world workflow archetypes. Three directories work together:

- `agents/workflows/` — executable workflow archetypes (`w01.py`–`w19.py`). Each file defines a LangGraph/LangChain workflow used for backtesting.
- `agents/patterns/` — reusable workflow patterns (`single_step.py`, `router.py`, `rag_pipeline.py`, `map_reduce.py`, `multi_turn.py`, etc.) composed into workflow archetypes.
- `prompts/` — prompt templates per workflow archetype (`w01_support_simple/`, `w13_routing_agent/`, etc.). Each subdirectory contains the system prompts and input templates for that archetype.
- `pdfs/` — PDF corpus generation for document-processing archetypes. `pdfs/generators/` creates synthetic PDFs (insurance docs, clinical notes, policy documents, etc.). `pdfs/validation/` verifies generated PDFs meet content coverage requirements.

### Test Organization

- `tests/unit/` — fast, no external deps, runs in CI
- `tests/integration/` — real LLM calls, marked `@pytest.mark.integration`, excluded from CI
- `tests/backtesting/` — projection engine calibration against 10 workflow archetypes
- `tests/synthetic/` — synthetic calibration including SWE-bench data
- `tests/conftest.py` — `sample_record` fixture, use `dataclasses.replace()` to vary fields

## Public API

`agentcost` exports: `ProfileRunner`, `StepRecord`, `__version__`. The CLI entry point is `agentcost.cli:cli` (registered as the `agentcost` console script).

## Key Design Decisions

- HTML report is Jinja2 + inline SVG — no JS framework dependency at render time. Local web UI is FastAPI + pre-built React bundle. The React source lives in `ui-frontend/` and is compiled to a static bundle shipped inside the pip package. Users never need Node.js or npm.
- Optional framework deps are extras: `pip install agentcost[langgraph]`, `agentcost[openai]`, `agentcost[qwen]`, `agentcost[ui]`, `agentcost[validation]`, `agentcost[agents]` (litellm for backtesting execution), `agentcost[backtesting]` (langchain providers + langgraph), `agentcost[pdf-generation]` (reportlab, matplotlib, etc. for PDF corpus). Core package depends only on `click`, `rich`, `jinja2`.
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
- `pytest-asyncio` mode is `auto` (set in `pyproject.toml`) — async test functions are collected automatically without `@pytest.mark.asyncio`.
- Cache busting is on by default during profiling (`--allow-cache` to disable). Profiling measures cold-start costs unless explicitly opted out.
- `PLANNING.md` and `agentcost-technical-spec-v5.md` contain the product spec and detailed technical spec. Consult these for feature design context.

## Environment Variables

- `LANGFUSE_SECRET_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_HOST` — required for Langfuse trace import (`--from-langfuse`)
- LLM provider API keys (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, etc.) — required for integration tests and backtesting workflows

## Git Rules

Never commit. Do not run `git add`, `git commit`, `git push`, or any command that writes to git history. Read-only git commands (`git status`, `git diff`, `git log`) are fine.

## Adding a New Model to Pricing

Update dicts in `agentcost/pricing/tables.py` — they must stay in sync:
1. `MODEL_PRICING`: canonical name → `(input_per_M, output_per_M)` in USD
2. `MODEL_TIERS`: canonical name → `"frontier"` / `"mid"` / `"fast"`
3. `MODEL_ALIASES` (if the model has common short names)
4. `MODEL_CACHE_HIT_PRICING` (if the model has differentiated cache-hit input pricing)

Tests in `tests/unit/test_pricing.py::TestStructuralInvariants` will fail if the dicts drift.

For runtime use without editing tables.py: `register_model(name, input_price, output_price, tier="mid")`. Unknown models at cost-calculation time raise `UnrecognizedModelError` with "did you mean?" suggestions.

## Adding a New Framework Collector

Create a new file in `agentcost/collectors/` that subclasses `BaseCollector` and implements `async collect()`. Map the framework's native callback/hook system to `StepRecord` fields. Add the framework as an optional dependency group in `pyproject.toml`. Register the lazy import in `collectors/__init__.py` via `__getattr__` and add the name to `__all__`. Add auto-detection heuristics in `ProfileRunner._select_collector()`.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs on Python 3.11 + 3.12 (ubuntu-latest). Steps: `ruff check`, `ruff format --check`, `pytest tests/unit/` (tolerates exit code 5 for uncollected tests during scaffold phase). Integration tests are excluded from CI.
