# AgentCost Backtesting Suite

Validates the projection engine against 10 real-world workflow archetypes before v1 ships.

## Philosophy

The goal is not exact accuracy — it's being usefully right within stated bounds. A projection of "$4,100–$11,700/mo (moderate confidence)" that lands at $9,200 in production is a success. A projection of "$11,700/mo" that lands at $45,000 is a product-killing failure.

Every projection must satisfy three properties:
1. **Calibration** — p50 exceeded ~50% of the time (ratio: 0.5–2.0x)
2. **Directional accuracy** — most expensive step identified correctly
3. **Useful range** — p95/p50 < 5x (tight enough to be actionable)

## The 10 Workflows

| # | Archetype | Complexity | Loops | Models | Expected $/run |
|---|-----------|-----------|-------|--------|----------------|
| W1 | Support agent | Simple | No | Haiku 4.5, Sonnet 4.6 | $0.005–0.03 |
| W2 | Support agent | Complex | Yes (1-15) | Haiku 4.5, Sonnet 4.6, Opus 4.7 | $0.08–0.60 |
| W3 | Code review | Simple | No | Sonnet 4.6 | $0.02–0.08 |
| W4 | Code review | Complex | Yes (1-8) | Sonnet 4.6, Opus 4.7 | $0.15–1.20 |
| W5 | Data extraction | Simple | No | Haiku 4.5, Sonnet 4.6 | $0.005–0.04 |
| W6 | Data extraction | Complex | Yes (1-5) | Sonnet 4.6, Opus 4.7 | $0.08–0.50 |
| W7 | Research agent | Simple | No | Haiku 4.5, Sonnet 4.6 | $0.02–0.10 |
| W8 | Research agent | Complex | Yes (1-6) | Sonnet 4.6, Opus 4.7 | $0.25–1.80 |
| W9 | Sales (OpenAI) | Simple | No | GPT-4.1 Nano, GPT-4.1 | $0.005–0.03 |
| W10 | Sales (mixed) | Complex | Yes (1-4) | Gemini Flash, GPT-4.1, Opus 4.7 | $0.10–0.70 |

W9 uses OpenAI exclusively to test cross-provider pricing accuracy. W10 mixes three providers (Google, OpenAI, Anthropic) to test multi-provider routing — a common production pattern.

## Running the Suite

### API Keys Required

```bash
export ANTHROPIC_API_KEY="sk-ant-..."   # W1-W8, W10
export OPENAI_API_KEY="sk-..."          # W9, W10
export GOOGLE_API_KEY="..."             # W10
```

### Three Phases

```bash
# Phase 1: Synthetic profiles (20 + 100 runs × 10 workflows, ~$8)
python tests/backtesting/run_backtesting.py --phase 1

# Phase 2: Ground truth (500 runs × 10 workflows, ~$750-950)
python tests/backtesting/run_backtesting.py --phase 2

# Phase 3: Score and report (no API calls, free)
python tests/backtesting/run_backtesting.py --phase 3

# All three phases
python tests/backtesting/run_backtesting.py --all

# Single workflow
python tests/backtesting/run_backtesting.py --workflow W1 --phase 1

# Resume interrupted run
python tests/backtesting/run_backtesting.py --phase 2 --resume
```

### Install Dependencies

```bash
pip install agentcost[backtesting]
```

## Interpreting Results

Phase 3 produces a calibration report:

```
Workflow         p50     p95 cov   Range    Top   Rank r   Verdict
W1 support-sim   1.1x ✓  92% ✓   2.3x ✓    ✓    0.95 ✓    PASS
W2 support-cplx  1.8x ✓  78% ⚠   4.1x ✓    ✓    0.82 ✓    WARN
...
Overall: 8 PASS, 2 WARN, 0 FAIL — LAUNCH GATE: ✅ PASSED
```

**Metrics:**
- **p50 ratio**: projected/actual p50 cost. Pass: 0.5–2.0x.
- **p95 coverage**: fraction of ground truth runs below projected p95. Pass: ≥80%.
- **Range ratio**: projected p95/p50 spread. Pass: <5x.
- **Top step**: most expensive step matches. Pass: yes (co-dominant within 20% both count).
- **Rank correlation**: Spearman r of step cost rankings. Pass: >0.8.
- **Verdict**: FAIL if any metric fails; WARN if any warns; PASS otherwise.

**Launch gate**: all 10 workflows must PASS or WARN on Synthetic-20.

## Fixing Calibration Failures

If a workflow fails:
1. Check which metric failed (p50 ratio, top step, etc.)
2. If p50 off → the projection engine may have wrong distributional assumptions. Check if Monte Carlo is triggering when it should.
3. If top step wrong → check model pricing. A wrong price per token propagates.
4. If p95 coverage low → projection is overconfident. The confidence tier should be lower.
5. If range too wide → too much variance in the workflow. Consider flagging for users.

## Opus 4.7 Tokenizer Note

Opus 4.7 ships with a new tokenizer that generates up to 35% more tokens for the same input text compared to Opus 4.6. This means effective per-request cost can be significantly higher than the rate card suggests. W2, W4, W6, W8, and W10 all use Opus 4.7 in their review loops — the backtesting suite captures this real-world tokenizer effect.

## Budget

Ground truth profiling (Phase 2) costs ~$750-950 across all 10 workflows at 500 samples each. This is the most important pre-launch investment — a projection engine that gives wrong numbers is worse than no projection engine.

## File Structure

```
tests/backtesting/
├── workflows/          # 10 LangGraph workflow files
│   ├── _shared.py      # Model helpers, canned data
│   └── w01_..w10_*.py
├── inputs/             # 500 JSONL inputs per workflow
│   └── w01_..w10_realistic.jsonl
├── results/            # Profiling results (gitignored except .gitkeep)
├── configs.py          # BacktestConfig for all 10 workflows
├── run_backtesting.py  # CLI runner
└── README.md           # This file
```
