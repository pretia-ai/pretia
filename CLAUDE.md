# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What is AgentCost?

AgentCost is an open-source Python SDK that answers two questions before you deploy an AI agent: "How much will this cost at scale?" and "How do I make it cheaper?"

You point it at your agent workflow, it auto-generates diverse test inputs, profiles the workflow, projects costs at 10x/100x/1000x traffic with distributional statistics (not just averages), detects cost time-bombs like context growth and stuck loops, and produces dollar-denominated optimization recommendations.

### The Default Experience

Two commands from install to first report:

```bash
pip install agentcost
agentcost profile run my_agent.py
```

No config files, no JSONL datasets, no setup. The SDK reads the workflow's system prompt, generates 20 diverse synthetic inputs via a cheap LLM call (~$0.02), runs the workflow on each, captures per-step token usage, computes cost distributions, detects patterns, and produces a full report with recommendations. Cost: ~$2. Time: ~3 minutes.

### Five Input Modes

AgentCost provides a friction ladder — five ways to feed inputs, from zero-effort to maximum precision:

- **Level 0** — `agentcost estimate workflow.py`: Static code analysis only. No execution, instant, free. Wide confidence intervals. Uses archetype-based priors in v1, ML cost predictor in v2.
- **Level 1** — `--input "How do I reset my password?"`: One example input. One run + priors for variance estimation.
- **Level 2** — `--auto-generate N` (DEFAULT): LLM generates diverse inputs from the system prompt + type hints. ~$0.02 for generation. Running `agentcost profile run workflow.py` with no flags defaults to `--auto-generate 20`.
- **Level 3** — `--from-langfuse --last 100`: Pull real inputs from Langfuse/Braintrust production traces. Zero cost if analyzing traces directly without re-execution (`agentcost analyze --from-langfuse`).
- **Level 4** — `--inputs samples.jsonl`: User-curated test dataset. Maximum precision, maximum friction.

The CLI auto-detects the best mode: if Langfuse credentials are in the environment, it suggests trace import. Otherwise defaults to auto-generate.

### CLI Commands

Full command surface:

- `agentcost profile run workflow.py` — Profile a workflow. Defaults to `--auto-generate 20`.
- `agentcost estimate workflow.py` — Instant cost estimate from code structure (no execution).
- `agentcost report profile.json --traffic 1000/day` — Generate cost report from a saved profile.
- `agentcost recommend profile.json` — Generate optimization recommendations.
- `agentcost analyze --from-langfuse --last 100` — Analyze existing Langfuse traces (no execution).
- `agentcost baseline update` — Save current profile as baseline to `.agentcost/baseline.json`.
- `agentcost diff baseline.json new_profile.json` — Compare two profiles, show per-step deltas.

### Recommendations

Three types of optimization recommendations, each with estimated savings in $/month:

**Model swap:** Detects steps using a higher-tier model than the task requires. v1 uses keyword heuristics on the system prompt + output/input token ratio to classify task type (classification, extraction, generation, code). If the step uses a model above the estimated minimum tier, recommend downshift with savings estimate. v1.5 replaces heuristics with an ML classifier (logistic regression on 800K+ RouterBench examples, 384-dim MiniLM embeddings + 9 numerical features). Confidence threshold: only recommend when confidence > 0.85; below that, fall back to heuristics.

**Architecture:** Detects context growth (Δcontext > 500 tokens/iteration consistently → recommend compaction), re-sent context (same content hash in prompts of successive steps → recommend caching), oversized tool definitions (tool_definitions_tokens > 30% of total prompt → recommend tool filtering per step).

**Workflow:** Detects excessive loop iterations (cost marginal per iteration + distribution of iteration counts → recommend cap). Stuck loop detection (runs with >2x mean iterations → flag as outliers, compute cost share, recommend circuit breaker).

### GitHub Action

AgentCost ships a GitHub Action that comments on every PR touching agent workflow code. The PR comment shows: projected monthly cost change (before → after), per-step breakdown with flags, and optimization recommendations.

Four CI modes:

- `static` (default, free): Parse git diff, detect model changes / step additions / loop modifications, estimate impact from baseline.
- `auto-generate` (~$1-3): Generate synthetic inputs, run workflow, compare to baseline.
- `sample` (~$2-5): Run workflow on user-curated inputs from a JSONL file.
- `import` (free): Pull latest Langfuse traces, analyze without re-execution.

Configurable threshold: block merge if cost increase exceeds X%. Config lives in `.github/workflows/agentcost.yml`.

### Projection Engine

Distributional scaling, not averages. For each target traffic volume:

- Collect per-step stats: p50, p75, p90, p95, p99 of tokens and cost.
- Detect non-linear patterns: context growth (context_size correlated with iteration, r² > 0.7), loop count variance (coefficient of variation), high variance (p95/p50 ratio > 3).
- Stable case: `monthly_cost = mean_cost_per_run × daily_volume × 30` with distributional output.
- Non-linear case: Monte Carlo — simulate 10,000 runs by sampling per-step distributions, apply context growth curves, sample loop counts. Triggered ONLY when heuristics detect non-linearity. ~5 seconds runtime.

### What We Don't Build

AgentCost sits above the entire LLM tooling stack. No proxy (use LiteLLM). No model routing (use Martian). No tracing (use Langfuse). No context compaction (use Morph). No quality evals (use Braintrust). No web dashboard in v1 — CLI + markdown + GitHub PR comments. We detect when compaction or caching is needed; we don't do it ourselves.

### Roadmap

- **v1.0** (weeks 1-12): Core SDK. Collectors (LangGraph, OpenAI Agents SDK, Generic), auto input generation, projection engine, heuristic recommendations, GitHub Action. No ML.
- **v1.5** (weeks 13-16): Task Complexity Classifier. Logistic regression on 800K+ RouterBench/LLMRouterBench examples + 1K synthetic agent prompts. MiniLM embeddings (384d) + 9 numerical features = 393 features. Trains in <30s on CPU. Ships as ~2MB .pkl in pip package. ML-powered model swap recommendations.
- **v2.0** (weeks 17-26): Live monitoring (`agentcost monitor` daemon) + Cost Prediction Model. XGBoost on 20 workflow-level features. Predicts cost from code structure alone — no profiling needed. `agentcost estimate workflow.py` becomes ML-powered.
- **v3.0** (weeks 27-40): Spend Governance (budget circuit breakers, intelligent degradation) + Workflow Benchmarking. HDBSCAN clustering on accumulated workflow data. "Your support agent costs 2.3x the median for similar workflows." Requires ~500 workflows minimum.

## Architecture

> AS OF TODAY, CAN BE CHANGED.

The data flows through a five-stage pipeline:

1. **Collector** — Framework adapters (LangGraph, OpenAI Agents SDK, CrewAI, Generic) instrument agent workflows and emit unified StepRecords.
2. **StepRecord** — Normalized data structure capturing everything about one LLM call or tool invocation within a workflow run.
3. **ProfileStore** — Persists profiling sessions as JSON (v1) or SQLite (v2). Each session = one workflow × N input runs.
4. **Projection** — Distributional scaling (p50–p99) for stable workflows. Monte Carlo simulation for non-linear cases (context growth, loop variance).
5. **Recommendation** — Heuristic rules (v1) → ML classifier (v1.5). Three types: model swap, architecture, workflow. Each recommendation carries estimated savings in $/month and a confidence level.

### StepRecord Fields

StepRecord is the central data structure. Every collector must produce these fields:

- `step_name: str` — Human-readable name (graph node name, agent name, or user-provided label).
- `step_type: str` — One of: `"llm"`, `"tool"`, `"retrieval"`.
- `model: str` — Model identifier (e.g. `"claude-sonnet-4-20250514"`, `"gpt-4o"`).
- `input_tokens: int` — Prompt tokens for this call.
- `output_tokens: int` — Completion tokens.
- `context_size: int` — Total prompt tokens including system prompt + conversation history. This is the number that grows in looping agents.
- `tool_definitions_tokens: int` — Tokens consumed by tool/function definitions in the prompt.
- `system_prompt_hash: str` — SHA-256 of the system prompt. Used for ML features in v1.5 and for detecting re-sent context.
- `system_prompt_tokens: int` — Token count of the system prompt alone.
- `output_format: str` — Auto-detected: `"json"`, `"text"`, or `"code"`. Used as ML feature.
- `is_retry: bool` — Whether this call is a retry of a previous failed call.
- `iteration: int` — Loop iteration number (1-indexed). 1 for non-looping steps.
- `parent_step: str | None` — Parent step name if this is a sub-step (e.g. inside a loop node).
- `duration_ms: int` — Wall-clock time for this call.
- `timestamp: datetime` — When the call started.

Design note: `system_prompt_hash`, `system_prompt_tokens`, and `output_format` are captured from day one even though v1 heuristics don't use them. They're the features the Task Complexity Classifier needs in v1.5. Carrying unused fields costs ~zero.

### Collector Contract

`BaseCollector` is an abstract base class. Each framework adapter implements it. The contract:

- `collect(workflow, inputs) -> list[list[StepRecord]]` — Run the workflow on each input, return a list of runs, where each run is a list of StepRecords.
- Collectors are async-first. The base method signature is `async def collect(...)`.
- LangGraphCollector hooks into LangChain's `UsageMetadataCallbackHandler`. It maps `on_llm_start` (capture model, prompt tokens), `on_llm_end` (capture output tokens), and `on_tool_start/end` (capture tool calls) to StepRecords.
- OpenAIAgentsCollector hooks into the `RunHooks` lifecycle (`on_agent_start/end`, `on_tool_start/end`) and reads `context_wrapper.usage.request_usage_entries` for per-request token breakdown.
- GenericCollector provides a `@collector.step("name")` decorator and a `with collector.step("name") as s` context manager. Auto-extracts tokens from OpenAI/Anthropic response objects.

### Baseline Format

A baseline is a JSON file (`.agentcost/baseline.json`) that stores the cost profile of a workflow at a point in time. Structure:

- Top level: `version`, `workflow` (file path), `profiled_at` (ISO timestamp), `sample_size`, `traffic_assumption`.
- `steps`: dict keyed by step name. Each step has: `model`, `tokens` (input/output with p50 and p95), `cost_per_run` (p50 and p95), `iterations` (mean and max), `system_prompt_hash`, `system_prompt_tokens`, `output_format`, `task_complexity_tier` (null in v1, filled by classifier in v1.5), and optional `flags` and `context_growth_rate`.
- `total_monthly`: dict with p50, p75, p90, p95 of projected monthly cost.

## Dev Commands

> AS OF TODAY, CAN BE CHANGED.

```bash
# Install in dev mode
uv pip install -e ".[dev]"

# Unit tests
pytest tests/unit/ -v

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

## Git Rules

Never commit. Do not run `git add`, `git commit`, `git push`, or any command that writes to git history. You can run `git init`, create `.gitignore`, and use read-only git commands (`git status`, `git diff`, `git log`). All commits are made manually so the author's name and GitHub profile appear on every commit.

## Code Conventions

- Python 3.11+. Use modern syntax: `X | Y` unions, `match` statements where clear.
- Type hints everywhere. Every function signature, every variable where non-obvious. `from __future__ import annotations` in every file.
- Dataclasses for data structures. Not Pydantic (too heavy for an SDK). `@dataclass` with `slots=True` where possible.
- Async-first. Agent frameworks are async. Collectors and profiling engine are async. Provide sync wrappers for CLI convenience.
- ruff for formatting and linting. Config in `pyproject.toml`.
- pytest + pytest-asyncio for tests. `@pytest.mark.asyncio` for async tests.
- Click for CLI. Group commands under `agentcost`.
- Every public function has a docstring. One-line summary + Args/Returns for non-trivial ones.
- No `print()`. Use `logging` for debug/info, `rich` for user-facing CLI output.
- Imports: standard library → third-party → local, separated by blank lines. Absolute imports only.

## Project Structure

```text
agentcost/
├── agentcost/
│   ├── __init__.py
│   ├── collectors/          # Framework adapters → StepRecords
│   │   ├── base.py          # StepRecord dataclass + BaseCollector ABC
│   │   ├── langgraph.py     # LangGraph via LangChain callbacks
│   │   ├── openai_agents.py # OpenAI Agents SDK via RunHooks
│   │   └── generic.py       # Decorator + context manager for manual instrumentation
│   ├── projection/          # Cost projection engine
│   │   ├── stats.py         # Distributional calculations (p50-p99)
│   │   ├── montecarlo.py    # Monte Carlo simulation for non-linear projection
│   │   └── patterns.py      # Detect context growth, loop variance, high variance
│   ├── recommend/           # Optimization recommendations
│   │   ├── heuristics.py    # v1 rule-based (keyword scan, ratio analysis)
│   │   ├── classifier.py    # v1.5 ML classifier (loads .pkl)
│   │   └── rules.py         # Recommendation types, formatting, savings calc
│   ├── ci/                  # CI/CD integration
│   │   ├── baseline.py      # Save/load .agentcost/baseline.json
│   │   ├── diff.py          # Compare two profiles, compute per-step deltas
│   │   └── report.py        # Format PR comments and CLI reports
│   ├── inputs/              # Input generation + import
│   │   ├── generator.py     # Generate synthetic inputs via cheap LLM call
│   │   ├── importer.py      # Pull traces from Langfuse/Braintrust
│   │   ├── schema.py        # Extract input schema from workflow code
│   │   └── selector.py      # Auto-detect best input mode
│   ├── models/              # Shipped ML models (.pkl files, v1.5+)
│   ├── pricing/
│   │   └── tables.py        # Static model pricing lookup (per-token costs)
│   ├── store.py             # ProfileStore — persist profiling sessions as JSON
│   └── cli.py               # Click CLI entry point
├── action/                  # GitHub Action (action.yml + entrypoint.sh)
├── tests/
│   ├── unit/                # Fast, no network, mock everything
│   ├── integration/         # Real LLM calls (@pytest.mark.integration)
│   └── fixtures/            # Sample StepRecords, profiles, mock data
├── examples/                # Demo agent workflows for testing + docs
├── pyproject.toml
└── README.md
```

## Key Design Decisions

1. **Collector pattern.** Each framework gets an adapter that maps its native callbacks/hooks to unified StepRecords. Adding a new framework = one new file implementing BaseCollector. Framework API changes are isolated to one file.
2. **Input generation as default.** Auto-generate diverse test inputs from the workflow's system prompt via a cheap LLM call (Haiku/mini, ~$0.02). Zero config required. Even mediocre synthetic inputs capture step structure, model usage, loop behavior, and context growth patterns.
3. **Distributional projection.** p50–p99, not averages. Costs follow a log-normal distribution — many cheap runs, few very expensive ones. The mean is misleading. Monte Carlo simulation triggered only when non-linear patterns are detected.
4. **Heuristics first, ML later.** v1 recommendations are pure rules. v1.5 adds an ML classifier. Heuristics remain as fallback when classifier confidence < 0.85. The product works without ML.
5. **ML models ship in the pip package.** ~2MB .pkl file. No API calls, no cloud inference. `agentcost recommend` runs the classifier locally in milliseconds.
6. **Future-proof data capture.** StepRecord captures `system_prompt_hash`, `system_prompt_tokens`, and `output_format` from day one even though v1 doesn't use them. These are the features the v1.5 classifier needs. Cost of carrying extra fields: ~zero.

## Testing Patterns

- **Unit tests:** `tests/unit/` — fast, no network, no LLM calls. Mock all external dependencies.
- **Integration tests:** `tests/integration/` — real LLM calls, real frameworks. Mark with `@pytest.mark.integration`. Skipped in CI. Run locally before releases.
- **Fixtures:** `tests/fixtures/` — sample StepRecords, profile JSONs, mock callback data.
- **Naming:** `test_<module>_<behavior>.py`. Example: `test_stats_percentile_calculation.py`.
- **Coverage:** 90%+ for unit tests on core modules (collectors, projection, recommend).

## Writing Style

Everything in this repo — code, comments, docstrings, README, CLI help text, error messages — must read like it was written by a careful human developer, not generated by an AI.

Rules:

- **No filler phrases.** Never write "This module provides…", "This function is responsible for…", "This is a comprehensive…", "Leveraging the power of…". Say what it does.
- **Docstrings start with a verb.** "Calculate per-step cost distributions." not "This function calculates the per-step cost distributions."
- **Comments explain WHY, not WHAT.** No `# Import libraries` above imports. No `# Initialize variables`. Comments are for non-obvious decisions.
- **Error messages are specific.** "No LangGraph callback data found — is the workflow using `astream` with a callback config?" not "An error occurred while processing."
- **README/docs are direct.** Second person, no preamble. "Run `agentcost profile run workflow.py` to profile your agent." not "AgentCost provides a comprehensive solution enabling developers to…"
- **Variable names:** descriptive, not verbose. `cost_per_run` not `the_calculated_cost_value_for_each_individual_run`.
- **No emoji in code or docstrings.** Emoji in CLI output is fine sparingly (status flags, progress).
- **Don't hedge.** "This projection assumes stable traffic." not "It's important to note that this projection may not be entirely accurate in all cases."

## What NOT To Do

- No web dashboard. CLI + markdown + GitHub PR comments.
- No proxy/gateway. We sit above gateways, not replace them.
- No TypeScript. Python only.
- No `print()`. Use `rich` for CLI output, `logging` for everything else.
- No heavy dependencies without justification. Every dep must earn its place.
- No Pydantic. Dataclasses only.
- No git commits. See Git Rules above.
