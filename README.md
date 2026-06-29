# Pretia

**Know what your agent will cost before you deploy.**

<!-- Terminal GIF: add after recording with VHS or asciinema -->

Pre-deployment cost intelligence for AI agent workflows. Two commands, zero config, ~$2. Get distributional cost projections (p50-p99), detect cost risks, and see exactly where the money goes.

## Install

```bash
pip install pretia
```

## Quick Start

**Zero-cost estimate** (static analysis, no execution):

```bash
pretia estimate my_agent.py
```

**Full profile** (runs your workflow, ~$2, ~3 minutes):

```bash
pretia profile run my_agent.py
```

No config files, no JSONL datasets, no setup. Pretia reads your workflow, generates diverse synthetic inputs, runs 20 profiling runs, detects patterns, and opens an HTML report with projections and recommendations.

<!-- Report screenshot: add after rendering -->

## Features

### Distributional Projections

Cost projections at p50, p75, p90, p95, and p99. Not averages. For workflows with non-linear behavior (context growth, variable loop counts), Pretia uses Monte Carlo simulation (10,000 runs) instead of linear scaling.

### Automatic Pattern Detection

Pretia scans your profiling data for cost risks: context windows that grow with each iteration, unpredictable retry loops, wide variance between typical and worst-case runs, routing branches that change cost profiles, and bimodal distributions where a cheap path and an expensive path create two distinct cost clusters. If something will surprise you at scale, the report flags it.

### Optimization Recommendations

Each recommendation comes with estimated monthly savings in dollars. Pretia identifies where you're overspending and suggests specific changes to bring costs down.

### Optimization Score

A 0-100 score measuring workflow cost efficiency. Three zones: red (0-40, needs optimization), amber (41-70, room to improve), green (71-100, well optimized).

### Five Input Modes

A friction ladder from zero-effort to maximum precision:

| Level | Command | What happens | Cost |
|-------|---------|-------------|------|
| 0 | `pretia estimate workflow.py` | Static code analysis only. No execution. | Free |
| 1 | `--input "How do I reset my password?"` | One run + priors for variance estimation. | ~$0.10 |
| 2 | `--auto-generate N` **(default)** | LLM generates diverse inputs from system prompt. | ~$2 |
| 3 | `--from-langfuse --last 100` | Pull real inputs from Langfuse production traces. | Free |
| 4 | `--inputs samples.jsonl` | User-curated test dataset. Maximum precision. | Execution only |

## Add to Your CI in 2 Minutes

Pretia ships a GitHub Action that comments on every PR with cost analysis.

**Diff-only mode** (free, default): static analysis in seconds.

```yaml
# .github/workflows/pretia.yml
name: Pretia
on: [pull_request]

permissions:
  contents: read
  pull-requests: write  # required for PR comments

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

**Full profile mode** (opt-in, ~$2): real profiling with recommendations.

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

The PR comment shows the optimization score, projected monthly cost, cost delta vs. baseline, and recommendations in a collapsible section.

## CLI Commands

```bash
pretia estimate workflow.py             # Instant cost estimate (no execution)
pretia profile run workflow.py          # Full profiling (default: --auto-generate 20)
pretia report profile.json              # Generate HTML report from saved profile
pretia recommend profile.json           # Generate optimization recommendations
pretia analyze --from-langfuse          # Analyze Langfuse traces (no execution)
pretia baseline update profile.json     # Save baseline for CI diffing
pretia diff baseline.json new.json      # Compare profiles, show per-step deltas
```

## Supported Frameworks

| Framework | Collection method | Install |
|-----------|------------------|---------|
| **LangGraph** | Callback handler | `pip install pretia[langgraph]` |
| **OpenAI Agents SDK** | RunHooks lifecycle | `pip install pretia[openai]` |
| **Qwen-Agent** | LLM proxy | `pip install pretia[qwen]` |
| **Generic** | `@collector.step()` decorator | `pip install pretia` |

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

See [CLAUDE.md](CLAUDE.md) for architecture details and coding conventions.

## Contributing

Issues and PRs welcome. Run `pytest tests/unit/` and `ruff check pretia/ tests/` before opening a PR.

## License

[BSL 1.1](LICENSE) (Business Source License). Free for all use except offering Pretia as a commercial hosted service. Converts to Apache 2.0 on 2030-06-13.
