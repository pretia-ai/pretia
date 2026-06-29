# Contributing to Pretia

Thanks for your interest in contributing. Here's how to get started.

## Setup

```bash
git clone https://github.com/pretia-ai/pretia.git
cd pretia
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run tests

```bash
pytest tests/unit/ -v          # fast, no external deps
ruff check pretia/ tests/      # lint
ruff format pretia/ tests/     # format
pyright pretia/                # type check
```

All four must pass before submitting a PR. CI runs the same checks on Python 3.11 and 3.12.

## Project structure

```
pretia/              # the package
  collectors/        # framework adapters (LangGraph, OpenAI Agents, Qwen, Generic)
  inputs/            # input generation and Langfuse import
  pricing/           # model pricing tables and cost calculation
  projection/        # stats, pattern detection, Monte Carlo simulation
  recommend/         # optimization recommendations (stubbed, v1.5)
  report/            # HTML report rendering (Jinja2 + inline SVG)
  ci/                # baselines, diffs, GitHub Action support
  cli.py             # Click CLI entry point
  runner.py          # orchestrates the profiling pipeline
  store.py           # JSON persistence for profiling sessions
tests/unit/          # unit tests (no external deps, runs in CI)
tests/integration/   # real LLM calls (excluded from CI)
action/              # GitHub Action (Dockerfile + entrypoint)
examples/            # example agents and demo profiles
```

See [docs/system-architecture-guide.md](docs/system-architecture-guide.md) for the full pipeline walkthrough.

## Code style

- Python 3.11+. `from __future__ import annotations` in every file.
- `ruff` for lint and format (config in `pyproject.toml`).
- `@dataclass` with `slots=True` where possible. No Pydantic.
- Async-first. Sync wrappers via `asyncio.run()` for CLI.
- `click` for CLI, `rich` for terminal output, `logging` for debug/info.
- Docstrings start with a verb. No filler phrases.
- Test naming: `test_<module>_<behavior>.py`.

## Adding a new model

Update four dicts in `pretia/pricing/tables.py` (they must stay in sync):

1. `MODEL_PRICING` — canonical name, (input $/M, output $/M)
2. `MODEL_TIERS` — canonical name, tier
3. `MODEL_ALIASES` — short names (if any)
4. `MODEL_CACHE_HIT_PRICING` — cache-hit input rate (if applicable)

Tests in `tests/unit/test_pricing.py::TestStructuralInvariants` will fail if the dicts drift.

## Adding a framework collector

1. Create `pretia/collectors/your_framework.py` subclassing `BaseCollector`
2. Implement `async collect(workflow, inputs) -> list[list[StepRecord]]`
3. Add the framework as an optional dependency in `pyproject.toml`
4. Register the lazy import in `collectors/__init__.py`
5. Add auto-detection in `ProfileRunner._select_collector()`

## Submitting a PR

1. Fork the repo and create a branch
2. Make your changes
3. Run the full test suite: `pytest tests/unit/ -v`
4. Run lint: `ruff check pretia/ tests/ && ruff format --check pretia/ tests/`
5. Open a PR with a clear description of what and why

## License

By contributing, you agree that your contributions will be licensed under the [BSL 1.1](LICENSE).
