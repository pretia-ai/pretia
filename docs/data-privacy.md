# Data Privacy Policy

Project policy for what Pretia stores, and what it must never store.

## What NOT To Store

| Prohibited data | Store instead |
|----------------|---------------|
| Raw input/output text | Token count + complexity tier label |
| Raw system prompts (by default) | SHA-256 hash + token count. Raw text is opt-in via `--include-prompts` |
| Full conversation histories | Turn count + accumulated context size per turn + growth rate |
| API keys or credentials | Never stored. Not in profiles, baselines, or logs |
| PII from user inputs | Token counts only. No content passes through to stored profiles |

## What We Store

- **Token counts**: input, output, context size, tool definition tokens per step
- **Model identifiers**: model name string (e.g., `"claude-sonnet-4-6"`)
- **Cost data**: computed from token counts × pricing table
- **Timing**: duration in milliseconds, timestamps (UTC)
- **Structural metadata**: step names, step types, iteration counts, parent relationships
- **System prompt hash**: SHA-256 for deduplication and change detection, not the content
- **Detected patterns**: context growth rates, loop variance, token distribution stats

## Opt-In Raw Data

When `--include-prompts` is passed (future feature), raw system prompts are stored in the profile JSON. This is useful for the recommendation engine (Tier 3 prompts need full context). The flag is off by default. Raw user inputs and LLM outputs are never stored regardless of flags.

## Baseline Files

Baselines (`.pretia/baseline.json`) contain aggregated statistics only: percentile distributions, model names, pattern flags, and monthly projections. No raw text. Baselines are safe to commit to version control.

## Profiling Sessions

Session files (`.pretia/{workflow}_{timestamp}.json`) contain StepRecords with token counts and metadata. No raw text content. Safe to share within a team for cost analysis.
