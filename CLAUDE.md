# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AgentCost?

AgentCost is an open-source Python SDK that tells developers how much their AI agent workflows will cost before deployment, and how to make them cheaper. Point it at your agent — it auto-generates diverse test inputs, profiles the workflow, projects costs at scale, and recommends optimizations with dollar-denominated savings.

## Architecture

Five-stage pipeline:

1. **Collector** — Framework adapters (LangGraph, OpenAI Agents SDK, CrewAI, Generic) instrument agent workflows and emit unified StepRecords.
2. **StepRecord** — Normalized data structure capturing per-step tokens, model, cost, context size, iteration, timing, and system prompt metadata.
3. **ProfileStore** — Persists profiling sessions as JSON (v1) or SQLite (v2). Feeds all downstream analysis.
4. **Projection** — Distributional scaling (p50–p99) for stable workflows. Monte Carlo simulation for non-linear cases (context growth, loop variance).
5. **Recommendation** — Heuristic rules (v1) → ML classifier (v1.5). Three types: model swap, architecture, workflow. Each recommendation carries estimated savings in $/month.

## Dev Commands

```bash
# Install in dev mode (use uv)
uv pip install -e ".[dev]"

# Run all unit tests
pytest tests/unit/ -v

# Run integration tests (real LLM calls — costs money, skip in CI)
pytest tests/integration/ -v -m integration

# Lint + format
ruff check agentcost/ tests/
ruff format agentcost/ tests/

# Type check
pyright agentcost/

# Build
python -m build
```

## Code Conventions

- Python 3.11+. Use modern syntax: `X | Y` unions, `match` statements where clear.
- Type hints everywhere. Every function signature, every variable where non-obvious. Use `from __future__ import annotations` in every file.
- Dataclasses for data structures. Not Pydantic (too heavy for an SDK). Plain `@dataclass` with `slots=True` where possible.
- Async-first. Agent frameworks are async. Collectors and the profiling engine must be async. Provide sync wrappers where needed for CLI convenience.
- ruff for formatting and linting. Config in `pyproject.toml`.
- pytest + pytest-asyncio for tests. Use `@pytest.mark.asyncio` for async tests.
- Click for CLI. Group commands under `agentcost`.
- Every public function has a docstring. One-line summary + Args/Returns for non-trivial functions.
- No `print()`. Use `logging` for debug/info or `rich` for user-facing CLI output.
- Imports: standard library → third-party → local, separated by blank lines. Use absolute imports.

## Project Structure

```text
agentcost/
├── agentcost/
│   ├── __init__.py
│   ├── collectors/          # Framework adapters → StepRecords
│   │   ├── base.py          # StepRecord + BaseCollector ABC
│   │   ├── langgraph.py     # LangGraph via LangChain callbacks
│   │   ├── openai_agents.py # OpenAI Agents SDK via RunHooks
│   │   └── generic.py       # Manual instrumentation (decorator + ctx mgr)
│   ├── projection/          # Cost projection engine
│   │   ├── stats.py         # Distributional calculations
│   │   ├── montecarlo.py    # Monte Carlo for non-linear cases
│   │   └── patterns.py      # Pattern detection (context growth, loops)
│   ├── recommend/           # Optimization recommendations
│   │   ├── heuristics.py    # v1 rule-based
│   │   ├── classifier.py    # v1.5 ML-powered (loads .pkl)
│   │   └── rules.py         # Recommendation types + formatting
│   ├── ci/                  # CI/CD integration
│   │   ├── baseline.py      # Baseline management
│   │   ├── diff.py          # Baseline comparison
│   │   └── report.py        # PR comment formatting
│   ├── inputs/              # Input generation + import
│   │   ├── generator.py     # LLM-powered synthetic input generation
│   │   ├── importer.py      # Langfuse/Braintrust trace import
│   │   ├── schema.py        # Extract input schema from workflow
│   │   └── selector.py      # Auto-detect best input mode
│   ├── models/              # Shipped ML models (.pkl)
│   ├── pricing/
│   │   └── tables.py        # Static model pricing lookup
│   ├── store.py             # ProfileStore
│   └── cli.py               # Click CLI
├── action/                  # GitHub Action
├── tests/
│   ├── unit/
│   ├── integration/         # Real LLM calls (mark with @pytest.mark.integration)
│   └── fixtures/
├── examples/                # Demo agent workflows
├── pyproject.toml
└── README.md
```

## Key Design Decisions

1. **Collector pattern.** Each framework gets an adapter that maps its native callbacks/hooks to unified StepRecords. Adding a new framework = one new file, implementing BaseCollector.
2. **Input generation.** Auto-generate diverse test inputs from the workflow's system prompt via a cheap LLM call (Haiku, ~$0.02). Default mode — zero config required.
3. **Projection.** Distributional (p50–p99), not just averages. Monte Carlo simulation triggered only when non-linear patterns are detected (context growth, loop variance). 10K simulations, ~5 seconds.
4. **Recommendations.** v1 is pure heuristics (keyword scan, ratio analysis, pattern detection). v1.5 adds an ML classifier trained on 800K+ public examples. Heuristics remain as fallback when classifier confidence < 0.85.
5. **ML models ship as .pkl** inside the pip package (~2MB). No API calls, no cloud inference. Everything runs locally.

## Testing Patterns

- Unit tests: `tests/unit/` — fast, no network, no LLM calls. Mock all external dependencies.
- Integration tests: `tests/integration/` — real LLM calls, real frameworks. Mark with `@pytest.mark.integration`. Skipped in CI (costs money). Run locally before releases.
- Fixtures: `tests/fixtures/` — sample StepRecords, profile JSONs, mock callback data.
- Naming: `test_<module>_<behavior>.py`. Example: `test_stats_percentile_calculation.py`.
- Coverage target: 90%+ for unit tests on core modules (collectors, projection, recommend).

## Writing Style — No AI Slop

Everything in this repo — code, comments, docstrings, README, CLI help text, error messages, commit messages — must read like it was written by a careful human developer, not generated by an AI.

Concrete rules:

- **No filler phrases:** never write "This module provides…", "This function is responsible for…", "This is a comprehensive…", "Leveraging the power of…". Just say what it does.
- **Docstrings:** start with a verb. "Calculate per-step cost distributions." not "This function calculates the per-step cost distributions."
- **Comments:** only when the code isn't self-explanatory. No `# Import libraries` above imports. No `# Initialize variables` above variable declarations. Comments explain WHY, not WHAT.
- **Error messages:** be specific and helpful. "No LangGraph callback data found — is the workflow using `astream` with a callback config?" not "An error occurred while processing the data."
- **README/docs:** direct, second-person, no preamble. "Run `agentcost profile run workflow.py` to profile your agent." not "AgentCost provides a comprehensive profiling solution that enables developers to…"
- **Variable names:** descriptive but not verbose. `cost_per_run` not `the_calculated_cost_value_for_each_individual_run`. `steps` not `step_records_list`.
- **No emoji in code or docstrings.** Emoji in CLI output is fine (sparingly — progress indicators, status flags).
- **Commit messages:** imperative mood, specific. "Add Monte Carlo projection for looping workflows" not "Updated projection module with new features".
- **Don't hedge excessively.** "This projection assumes stable traffic" not "It's important to note that this projection may not be entirely accurate in all cases as it assumes relatively stable traffic patterns."

## What NOT To Do

- No web dashboard. CLI + markdown + GitHub PR comments.
- No proxy/gateway. We sit above gateways (LiteLLM, Cloudflare), not replace them.
- No TypeScript. Python only.
- No `print()`. Use `rich` for CLI output, `logging` for everything else.
- No heavy dependencies without clear justification. Every dep must earn its place.
- No Pydantic. Use dataclasses.
