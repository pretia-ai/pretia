# Pretia

**Know what your agent will cost before you deploy.**

Pre-deployment cost intelligence for AI agent workflows. Two commands, zero config, about $2. You get distributional cost projections (p50 through p99), automatic detection of cost risks, and a clear breakdown of where the money goes.

## Install

```bash
pip install pretia
```

Requires Python 3.11+.

Profiling runs your workflow with real API calls, so your provider key must be set in the environment (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.). `pretia estimate` is the only command that needs no key.

## Quick Start

**Zero-cost estimate** (static analysis, no execution):

```bash
pretia estimate my_agent.py
```

This gives you a conservative upper bound. Static analysis reads your code but can't know runtime behavior (which branches run, actual output lengths, caching effects). Expect estimates 2-5x higher than real costs. Good for getting a ballpark before committing to a full profile.

**Full profile** (runs your workflow, about $2, a few minutes):

```bash
pretia profile run my_agent.py
```

This runs your workflow with real API calls and produces accurate distributional projections (typically within 10% of production costs). No config files, no JSONL datasets, no setup. Pretia reads your workflow, generates diverse synthetic inputs, runs profiling passes in parallel, detects patterns, and opens an HTML report with projections and recommendations. Profiles are saved as JSON in `.pretia/` so you can re-render reports or diff them later.

### What Pretia expects from your file

Pretia imports your file, finds your workflow, and calls it as `entrypoint(input)` with a single input string. A minimal LangGraph example:

```python
# my_agent.py
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph

llm = init_chat_model("openai:gpt-4o-mini")

def respond(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("respond", respond)
builder.add_edge(START, "respond")
builder.add_edge("respond", END)
app = builder.compile()  # Pretia finds `app` automatically
```

```bash
pretia profile run my_agent.py
```

Any module-level compiled graph, agent object with `.invoke()`/`.ainvoke()`, or function that takes one input string works. Message-based graph states are handled automatically (the input string is wrapped in a `HumanMessage`). If Pretia picks the wrong entrypoint or can't find one, point it at the right object with `--entry-point <name>`. The error messages will tell you exactly what it looked at and why it was rejected.

## Features

### Distributional Projections

Cost projections at p50, p75, p90, p95, and p99. Not averages. For workflows with non-linear behavior (context growth, variable loop counts), Pretia runs Monte Carlo simulation (10,000 iterations) instead of linear scaling.

### Automatic Pattern Detection

Pretia scans your profiling data for cost risks:

- Context windows that grow with each iteration
- Unpredictable retry loops
- Wide variance between typical and worst-case runs
- Routing branches that change cost profiles
- Bimodal distributions where a cheap path and an expensive path create two distinct cost clusters

If something will surprise you at scale, the report flags it.

### Optimization Recommendations

Each recommendation comes with estimated monthly savings in dollars. Pretia identifies where you're overspending and suggests specific changes. All estimates are conservative; actual savings are typically higher.

### Optimization Score

A 0-100 score measuring workflow cost efficiency. The score factors in both recoverable savings and detected cost patterns. Three zones: red (0-30, needs optimization), amber (31-70, room to improve), green (71-100, well optimized).

### Five Input Modes

A friction ladder from zero effort to maximum precision:

| Level | Command | What happens | Cost |
|-------|---------|-------------|------|
| 0 | `pretia estimate workflow.py` | Static code analysis only. No execution. | Free |
| 1 | `--input "How do I reset my password?"` | One run + priors for variance estimation. | ~$0.10 |
| 2 | `--auto-generate N` **(default)** | LLM generates diverse inputs from system prompt. | ~$2 |
| 3 | `--from-langfuse --last 100` | Re-run your workflow on real production inputs pulled from Langfuse. | Execution only |
| 4 | `--inputs samples.jsonl` | User-curated test dataset. Maximum precision. | Execution only |

Already have production traces? `pretia analyze --from-langfuse` projects costs directly from your existing Langfuse traces with **zero execution and zero cost**. See [Analyze production traces](#analyze-production-traces-langfuse).

## Analyze Production Traces (Langfuse)

If your agent is already in production with Langfuse tracing, Pretia can project costs from your existing traces without executing anything:

```bash
pip install pretia[langfuse]

export LANGFUSE_SECRET_KEY=sk-lf-...
export LANGFUSE_PUBLIC_KEY=pk-lf-...
export LANGFUSE_HOST=https://cloud.langfuse.com  # default; set for self-hosted

pretia analyze --from-langfuse --last 100 --traffic 5000
```

What happens: Pretia fetches your most recent traces via the Langfuse API (`--last N`, default 10, max 100; filter by workflow with `--name`), converts each trace's observations into step records, and runs the full projection pipeline: distributional stats, pattern detection, Monte Carlo when needed, HTML report. No code, no re-execution, no API spend.

One requirement: your traces need `GENERATION` observations with usage data (model + token counts). If your instrumentation only logs spans, token counts read as zero and Pretia will warn you.

## Add to Your CI in 2 Minutes

Pretia ships a GitHub Action that comments on every PR with cost analysis.

**Diff-only mode** (free, the default): static analysis in seconds.

```yaml
# .github/workflows/pretia.yml
name: Pretia
on: [pull_request]

permissions:
  contents: read
  pull-requests: write

jobs:
  cost-check:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: pretia-ai/pretia/action@v1
        with:
          workflow_path: src/agent.py
          cost_threshold: "20"  # fail if cost increases >20%
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
```

**Full profile mode** (opt-in, about $2): real profiling with recommendations.

```yaml
      - uses: pretia-ai/pretia/action@v1
        with:
          workflow_path: src/agent.py
          mode: profile
          cost_threshold: "20"
        env:
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}  # or your provider key
```

The PR comment shows the optimization score, projected monthly cost, cost delta vs. baseline, and recommendations.

## CLI Commands

```bash
pretia estimate workflow.py             # Instant cost estimate (no execution)
pretia profile run workflow.py          # Full profiling (default: --auto-generate 50)
pretia report profile.json              # Generate HTML report from saved profile
pretia recommend profile.json           # Generate optimization recommendations
pretia analyze --from-langfuse          # Analyze Langfuse traces (no execution)
pretia baseline update profile.json     # Save baseline for CI diffing
pretia diff baseline.json new.json      # Compare profiles, show per-step deltas
pretia doctor                           # Environment and dependency health check
pretia doctor workflow.py               # Also verify workflow loads correctly
pretia update-pricing                   # Refresh model pricing from community data (LiteLLM)
pretia update-pricing --file p.json     # Or load custom pricing from a local JSON file
```

### Useful flags

```bash
pretia profile run workflow.py --traffic 5000      # Project costs at 5K runs/day
pretia profile run workflow.py --concurrency 10    # Limit parallel profiling runs
pretia profile run workflow.py --allow-cache       # Measure with prompt caching on
pretia profile run workflow.py --input "test"      # Single specific input
pretia profile run workflow.py --entry-point app   # Choose the workflow object explicitly
```

## Supported Frameworks

| Framework | Collection method | Install |
|-----------|------------------|---------|
| **LangGraph** | Callback handler | `pip install pretia[langgraph]` |
| **OpenAI Agents SDK** | RunHooks lifecycle | `pip install pretia[openai]` |
| **Anthropic SDK** | Messages.create monkey-patch | `pip install pretia` + `anthropic` |
| **OpenAI SDK** | Completions.create monkey-patch | `pip install pretia` + `openai` |
| **Qwen-Agent** | LLM proxy | `pip install pretia[qwen]` |
| **Generic** | `@collector.step()` decorator | `pip install pretia` |

Auto-detection picks the right collector for your workflow. You can also specify one explicitly with `--collector`.

## How It Works

Data flows through a five-stage pipeline:

1. **Collector**: framework adapters instrument your workflow and emit unified StepRecords
2. **StepRecord**: frozen dataclass capturing one LLM call (model, tokens, cost, timing, tool usage)
3. **ProfileStore**: persists profiling sessions as JSON (one workflow x N input runs)
4. **Projection**: distributional scaling (p50-p99) for stable workflows, Monte Carlo for non-linear cases
5. **Recommendation**: rule-based generators produce dollar-denominated optimization suggestions

The projection engine is validated against 13 real-world workflow archetypes (12/13 within 10% projection error).

## Positioning

**Langfuse** tells you what you spent. **Pretia** tells you what you'll spend. Use both.

Pretia sits above the LLM tooling stack. It detects when other tools are needed. No proxy (use LiteLLM), no routing (use Martian), no tracing (use Langfuse), no evals (use Braintrust).

## Development

```bash
uv pip install -e ".[dev]"
pytest tests/unit/ -v
ruff check pretia/ tests/
ruff format pretia/ tests/
pyright pretia/
```

## Contributing

Issues and PRs welcome. Run `pytest tests/unit/` and `ruff check pretia/ tests/` before opening a PR.

## License

[BSL 1.1](LICENSE) (Business Source License). Free for all use except offering Pretia as a commercial hosted service. Converts to Apache 2.0 on 2030-06-13.
