# Pretia Backtesting Suite

Validates the projection engine against 13 real-world workflow archetypes. Backtesting is complete: 12/13 workflows project within 10% error, W5 (multimodal) recovers with `--traffic-mix` reweighting. Total backtest cost: $89.66.

## Philosophy

The goal is not exact accuracy — it's being usefully right within stated bounds. A projection of "$4,100–$11,700/mo (moderate confidence)" that lands at $9,200 in production is a success. A projection of "$11,700/mo" that lands at $45,000 is a product-killing failure.

Every projection must satisfy three properties:
1. **Calibration** — p50 exceeded ~50% of the time (ratio: 0.5–2.0x)
2. **Directional accuracy** — most expensive step identified correctly
3. **Useful range** — p95/p50 < 5x (tight enough to be actionable)

## The 13 Active Workflows

| # | Archetype | Complexity | Loops | Provider(s) |
|---|-----------|-----------|-------|-------------|
| W1 | Support agent | Simple | No | Anthropic |
| W2 | Support agent | Complex | Yes | Anthropic |
| W5 | Multimodal extraction | Simple | No | Anthropic |
| W9 | Sales outreach (OpenAI) | Simple | No | OpenAI |
| W11 | Support agent (Qwen) | Simple | No | Qwen |
| W12 | Code review (DeepSeek) | Simple | No | DeepSeek |
| W13 | Routing agent | Complex | No | Anthropic + OpenAI |
| W14 | RAG pipeline (insurance) | Complex | No | Anthropic + OpenAI |
| W15 | RAG pipeline (clinical) | Complex | No | Anthropic + OpenAI |
| W16 | Map-reduce summarization | Complex | No | Anthropic |
| W17 | Document processing | Complex | No | Anthropic + OpenAI |
| W18 | Self-assessment loop | Complex | Yes | Anthropic |
| W19 | Multi-turn conversation | Complex | Yes | Anthropic |

**Excluded:** W3, W4, W6, W7, W8, W10 (dropped during archetype refinement).

W5 requires `--traffic-mix` for accurate multimodal workflow projection. W11 tests Qwen provider with parallelism capped at 10 (Dashscope 240 RPM). W13 tests cross-provider routing.

## 3-Comparison Protocol (A/B/C)

Each workflow is validated via three comparisons:

- **Comparison A** — 50 profiling runs projected against 200 ground-truth runs. Tests projection accuracy at the standard profiling sample size.
- **Comparison B** — Same 50 profiling runs projected against 500 ground-truth runs. Tests whether accuracy holds at higher ground-truth volume.
- **Comparison C** — 10 profiling runs projected against 200 ground-truth runs. Tests degradation at minimal profiling sample size.

## Running

### Prerequisites

```bash
pip install pretia[backtesting]

export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
export DASHSCOPE_API_KEY="sk-..."       # W11 (Qwen)
export DEEPSEEK_API_KEY="sk-..."        # W12 (DeepSeek)
```

### Pilot → Backtest Flow

```bash
# 1. Pre-flight validation (no API calls)
python scripts/verify_backtest_readiness.py

# 2. Pilot run (10 runs per workflow, ~$2)
python tests/backtesting/run_pilot.py --all

# 3. Full backtest (A/B/C comparisons, ~$90)
python tests/backtesting/run_backtest.py --all

# Single workflow
python tests/backtesting/run_backtest.py --workflow W1

# Dry run (no API calls)
python tests/backtesting/run_backtest.py --workflow W1 --dry-run
```

### Budget Tracking

`budget_tracker.py` tracks cumulative spend during the backtest. If 5 or more workflows fail Comparison A, execution halts (systemic engine problem detected).

## Interpreting Results

Results are written to `tests/backtesting/results/backtest/` (gitignored).

**Metrics per comparison:**
- **p50 ratio**: projected/actual p50 cost. Pass: 0.5–2.0x.
- **p95 coverage**: fraction of ground truth runs below projected p95. Pass: ≥80%.
- **Range ratio**: projected p95/p50 spread. Pass: <5x.
- **Top step**: most expensive step matches. Pass: yes.
- **Rank correlation**: Spearman r of step cost rankings. Pass: >0.8.

## File Structure

```
tests/backtesting/
├── configs.py           # BacktestConfig for 13 active workflows
├── run_backtest.py      # Main backtest runner (A/B/C comparisons)
├── run_pilot.py         # 10-run pilot for smoke testing
├── pilot_checks.py      # Runtime checks (cache-bust, step counts, etc.)
├── budget_tracker.py    # Cumulative spend tracking + halt gates
├── concurrency.py       # Per-provider parallelism limits
├── dataset.py           # Dataset recording for reproducibility
├── attribute_failure.py # Failure attribution analysis
├── detector_validation.py # Pattern detector validation
├── generate_report.py   # Report generation from results
├── results/             # Output directory (gitignored)
└── README.md            # This file

bt_agents/
├── workflows/           # 13 active workflow agents (w01-w19)
├── patterns/            # Reusable workflow patterns
├── harness/             # Execution infrastructure
└── providers/           # LLM + embedding providers
```
