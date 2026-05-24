# AgentCost

Pre-deployment cost intelligence for AI agent workflows.

> 🚧 Under active development — not yet on PyPI.

## Quick start

```bash
pip install agentcost
agentcost profile run my_agent.py
```

That's it. AgentCost generates diverse synthetic inputs, runs your workflow, projects costs at 10x/100x/1000x traffic, and produces optimization recommendations with dollar-denominated savings.

## How it works

Point AgentCost at your agent workflow. It captures per-step token usage, computes cost distributions (p50–p99, not just averages), detects cost time-bombs like context growth and stuck loops, and outputs a report with concrete recommendations.

Five input modes, from zero-effort to maximum precision:

- **Static analysis** — `agentcost estimate workflow.py`. Free, instant.
- **Single example** — `--input "..."`. One run plus priors.
- **Auto-generated (default)** — `--auto-generate 20`. ~$0.02 in generation cost.
- **From traces** — `--from-langfuse --last 100`. Free if analyzing without re-running.
- **Curated dataset** — `--inputs samples.jsonl`. Maximum precision.

## Supported frameworks

- LangGraph
- OpenAI Agents SDK
- Generic (decorator + context manager for manual instrumentation)
- More coming

## Documentation

See [CLAUDE.md](CLAUDE.md) for architecture, design decisions, and contribution guidelines.

## Contributing

Issues and PRs welcome. Run `pytest tests/unit/` before opening a PR.

## License

MIT
