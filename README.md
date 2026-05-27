# AgentCost

Pre-deployment cost intelligence for AI agent workflows.

> Under active development — not yet on PyPI.

## Quick start

```bash
pip install agentcost
agentcost profile run my_agent.py
```

No config files, no JSONL datasets, no setup. AgentCost reads the workflow's system prompt, generates 20 diverse synthetic inputs via a cheap LLM call (~$0.02), runs the workflow on each, captures per-step token usage, computes cost distributions, detects patterns, and produces a full report with recommendations. Cost: ~$2. Time: ~3 minutes.

## Five Input Modes

A friction ladder — from zero-effort to maximum precision:

| Level | Command | What happens | Cost |
|-------|---------|-------------|------|
| 0 | `agentcost estimate workflow.py` | Static code analysis only. No execution, instant. Wide confidence intervals. | Free |
| 1 | `--input "How do I reset my password?"` | One run + priors for variance estimation. | ~$0.10 |
| 2 | `--auto-generate N` **(default)** | LLM generates diverse inputs from system prompt + type hints. | ~$0.02 generation + execution |
| 3 | `--from-langfuse --last 100` | Pull real inputs from Langfuse/Braintrust production traces. | Free if analyzing traces without re-execution |
| 4 | `--inputs samples.jsonl` | User-curated test dataset. Maximum precision. | Execution only |

The CLI auto-detects the best mode: if Langfuse credentials are in the environment, it suggests trace import. Otherwise defaults to `--auto-generate 20`.

## CLI Commands

```bash
agentcost profile run workflow.py          # Profile a workflow (defaults to --auto-generate 20)
agentcost estimate workflow.py             # Instant cost estimate from code structure (no execution)
agentcost report profile.json --traffic 1000/day  # Generate cost report from a saved profile
agentcost recommend profile.json           # Generate optimization recommendations
agentcost analyze --from-langfuse --last 100      # Analyze existing Langfuse traces (no execution)
agentcost baseline update                  # Save current profile as baseline
agentcost diff baseline.json new.json      # Compare two profiles, show per-step deltas
```

## Recommendations

Three types of optimization recommendations, each with estimated savings in $/month:

### Model swap

Detects steps using a higher-tier model than the task requires. v1 uses keyword heuristics on the system prompt + output/input token ratio to classify task type (classification, extraction, generation, code). If the step uses a model above the estimated minimum tier, recommend downshift with savings estimate.

v1.5 replaces heuristics with an ML classifier (logistic regression on 800K+ RouterBench examples, 384-dim MiniLM embeddings + 9 numerical features). Confidence threshold: only recommend when confidence > 0.85; below that, fall back to heuristics.

### Architecture

- **Context growth** — Δcontext > 500 tokens/iteration consistently → recommend compaction.
- **Re-sent context** — Same content hash in prompts of successive steps → recommend caching.
- **Oversized tool definitions** — tool_definitions_tokens > 30% of total prompt → recommend tool filtering per step.

### Workflow

- **Excessive loop iterations** — Cost marginal per iteration + distribution of iteration counts → recommend cap.
- **Stuck loop detection** — Runs with >2x mean iterations → flag as outliers, compute cost share, recommend circuit breaker.

## Projection Engine

Distributional scaling, not averages. For each target traffic volume:

- Collect per-step stats: p50, p75, p90, p95, p99 of tokens and cost.
- Detect non-linear patterns: context growth (context_size correlated with iteration, r² > 0.7), loop count variance (coefficient of variation), high variance (p95/p50 ratio > 3).
- **Stable case:** `monthly_cost = mean_cost_per_run × daily_volume × 30` with distributional output.
- **Non-linear case:** Monte Carlo — simulate 10,000 runs by sampling per-step distributions, apply context growth curves, sample loop counts. Triggered only when heuristics detect non-linearity. ~5 seconds runtime.

## GitHub Action

AgentCost ships a GitHub Action that comments on every PR touching agent workflow code. The PR comment shows: projected monthly cost change (before → after), per-step breakdown with flags, and optimization recommendations.

Four CI modes:

| Mode | Cost | How it works |
|------|------|-------------|
| `static` (default) | Free | Parse git diff, detect model changes / step additions / loop modifications, estimate impact from baseline. |
| `auto-generate` | ~$1-3 | Generate synthetic inputs, run workflow, compare to baseline. |
| `sample` | ~$2-5 | Run workflow on user-curated inputs from a JSONL file. |
| `import` | Free | Pull latest Langfuse traces, analyze without re-execution. |

Configurable threshold: block merge if cost increase exceeds X%. Config lives in `.github/workflows/agentcost.yml`.

## Supported Frameworks

- **LangGraph** — via LangChain's `UsageMetadataCallbackHandler`
- **OpenAI Agents SDK** — via `RunHooks` lifecycle
- **Generic** — `@collector.step("name")` decorator and `with collector.step("name") as s` context manager for manual instrumentation

## Architecture

Data flows through a five-stage pipeline:

1. **Collector** — Framework adapters instrument agent workflows and emit unified StepRecords.
2. **StepRecord** — Normalized dataclass capturing everything about one LLM call or tool invocation.
3. **ProfileStore** — Persists profiling sessions as JSON. Each session = one workflow × N input runs.
4. **Projection** — Distributional scaling (p50–p99) for stable workflows. Monte Carlo for non-linear cases.
5. **Recommendation** — Heuristic rules (v1) → ML classifier (v1.5). Each recommendation carries estimated savings in $/month and a confidence level.

### StepRecord Fields

Every collector must produce these fields:

| Field | Type | Description |
|-------|------|-------------|
| `step_name` | `str` | Human-readable name (graph node, agent name, or label) |
| `step_type` | `str` | `"llm"`, `"tool"`, or `"retrieval"` |
| `model` | `str` | Model identifier (e.g. `"claude-sonnet-4-20250514"`) |
| `input_tokens` | `int` | Prompt tokens for this call |
| `output_tokens` | `int` | Completion tokens |
| `context_size` | `int` | Total prompt tokens including system prompt + history |
| `tool_definitions_tokens` | `int` | Tokens consumed by tool/function definitions |
| `system_prompt_hash` | `str` | SHA-256 of the system prompt |
| `system_prompt_tokens` | `int` | Token count of the system prompt alone |
| `output_format` | `str` | `"json"`, `"text"`, or `"code"` |
| `is_retry` | `bool` | Whether this is a retry of a failed call |
| `iteration` | `int` | Loop iteration number (1-indexed) |
| `parent_step` | `str \| None` | Parent step name if this is a sub-step |
| `duration_ms` | `int` | Wall-clock time for this call |
| `timestamp` | `datetime` | When the call started |

`system_prompt_hash`, `system_prompt_tokens`, and `output_format` are captured from day one even though v1 heuristics don't use them — they're the features the v1.5 Task Complexity Classifier needs.

## What AgentCost Doesn't Do

AgentCost sits above the entire LLM tooling stack. It detects when these tools are needed; it doesn't replace them:

- No proxy — use LiteLLM.
- No model routing — use Martian.
- No tracing — use Langfuse.
- No context compaction — use Morph.
- No quality evals — use Braintrust.
- No web dashboard in v1 — CLI + markdown + GitHub PR comments.

## Roadmap

- **v1.0** (weeks 1-12): Core SDK. Collectors (LangGraph, OpenAI Agents SDK, Generic), auto input generation, projection engine, heuristic recommendations, GitHub Action.
- **v1.5** (weeks 13-16): Task Complexity Classifier. ML-powered model swap recommendations. Logistic regression on 800K+ RouterBench examples. Ships as ~2MB .pkl in pip package.
- **v2.0** (weeks 17-26): Live monitoring (`agentcost monitor` daemon) + ML cost prediction from code structure alone.
- **v3.0** (weeks 27-40): Spend governance (budget circuit breakers, intelligent degradation) + workflow benchmarking across users.

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/unit/ -v
ruff check agentcost/ tests/
ruff format agentcost/ tests/
pyright agentcost/
```

See [CLAUDE.md](CLAUDE.md) for architecture details and coding conventions.

## Contributing

Issues and PRs welcome. Run `pytest tests/unit/` and `ruff check agentcost/ tests/` before opening a PR.

## License

MIT
