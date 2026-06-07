# AgentCost Backtest Analysis Prompt

Use this prompt when sending `analytics.json` and `analytics.pdf` to an LLM for analysis.

---

## The Prompt

```
You are a cost optimization engineer reviewing the backtesting results for an AI agent cost projection engine. You have two files:

1. **analytics.json** — structured data with validation scores, detector results, variance metrics, cost attribution, pain points, and optimization opportunities with estimated savings
2. **analytics.pdf** — the same data visualized as plots and tables (validation heatmaps, KDE overlays, detector matrices, cost breakdowns)

The projection engine profiles AI agent workflows with n=50 runs and projects monthly costs at scale. The backtesting validates these projections against ground truth (200-500 runs) under three conditions:
- **Comparison A (no drift):** profiling and ground truth drawn from the same distribution. Tests engine accuracy.
- **Comparison B (drifted):** ground truth has shifted tier weights, messier inputs, and longer tokens. Tests robustness.
- **Comparison C (reweighted):** same as B, but engine applies --traffic-mix weights. Tests whether reweighting recovers accuracy.

Accuracy targets:
- No-drift (A): mean error <10%, CI coverage ≥85%, CVaR error <25%
- Drifted (B/C): mean error <20%, CI coverage ≥75%, CVaR error <40%

Failure buckets:
- Bucket 1: Comparison A fails → engine/infrastructure problem
- Bucket 2: A passes, B fails, C recovers ≥50% → drift sensitivity, reweighting helps
- Bucket 3: A passes, B fails, C doesn't recover → structural drift, must re-profile

There are 5 pattern detectors that should fire on specific workflows:
- context_growth: W2, W4, W15, W19
- loop_count_variance: W2, W4, W15
- high_token_variance: W5, W16, W18
- step_count_variance: W2, W13, W15, W16, W17
- bimodality: W13, W15, W17

A false negative (expected but not fired) indicates a bug or insufficient input diversity.

---

Analyze these results and produce a report with these sections:

## 1. Launch Gate Assessment
- Is the engine ready for production? Which workflows block the gate?
- For each failing workflow: what failed, which comparison, and is it fixable before launch?
- What's the overall confidence level in the projection engine?

## 2. Projection Accuracy Analysis
- Which workflows have the tightest projections (lowest error)? Which are worst?
- Look at the PDF plots (B1: KDE overlays, B7: CI coverage bands). Do the projected distributions match ground truth visually?
- Are there systematic biases (consistently over- or under-estimating)?

## 3. Drift Resilience
- How much accuracy degrades from A to B? Is it uniform or concentrated?
- Does reweighting (C) recover accuracy? For which workflows does it help most/least?
- What drift dimensions cause the most damage: tier weights, input style, or token length?

## 4. Detector Reliability
- Check the detector activation matrix. What's the TP and FN rate?
- Are there false negatives that indicate the engine will miss real cost patterns in production?
- Are there unexpected false positives that reveal cost behaviors the design didn't anticipate?

## 5. Cost Optimization Recommendations
- Review the opportunities table. Rank the top 5 by impact (savings × confidence).
- For each, explain: what's the root cause, what specific change to make, and what's the risk of the change.
- Which recommendation types (model downshift, caching, iteration cap, etc.) offer the most total savings across all workflows?
- Flag any recommendations that seem wrong or where the savings estimate is likely off.

## 6. Risk Assessment
- Which workflows have the highest cost variance (check variance_risk table)?
- Which steps are unpredictable (p95 > 3× p50)?
- If production traffic doubles, which workflows will see disproportionate cost increases?

## 7. Action Plan
Produce a prioritized list of actions, ordered by:
1. Blockers (must fix before launch)
2. Quick wins (high savings, high confidence, low effort)
3. Investigations (uncertain but potentially high-impact)
4. Deferred (low priority or low confidence)

For each action, specify: workflow, step, what to change, estimated savings/month, and effort level (hours/days).
```

---

## How to Use

### With Claude (text + PDF)
```
# Upload analytics.pdf as an attachment, paste analytics.json inline
[paste prompt above]

[paste contents of analytics.json]
```

### With Claude Code
```bash
# From the AgentCost repo root:
cat visualization/llm_analysis_prompt.md reports/demo/analytics.json | pbcopy
# Then paste into Claude with the PDF attached
```

### With any LLM API (JSON only, no images)
```python
with open("reports/demo/analytics.json") as f:
    data = f.read()

prompt = open("visualization/llm_analysis_prompt.md").read()
# Send prompt + data as the user message
```

### With the AgentCost CLI (future)
```bash
agentcost analyze-report reports/demo/analytics.json --with-pdf reports/demo/analytics.pdf
```
