# Directions: Backtest Visualization, Reporting, and Pre-Calibration Integration

**Purpose:** This file specifies two things: (1) the visualization and reporting system that turns raw backtest results into understandable plots, dashboards, and narratives, and (2) the integration with the existing engine's pre-calibration validation layers that must pass before the pilot runs.

**Context for Claude Code:** You have the full Pretia codebase (including whatever validation, schema checks, and pre-flight infrastructure already exists), `projection-engine-recommendation-addition-2.md`, `cross-cutting-robustness.md`, `pretia-v2-additions-prompts.md` (the backtest protocol), and the technical spec. Explore the codebase to find existing validation layers, reporting patterns, and plotting utilities before building anything new.

---

## Part 1: Pre-Calibration Integration

### What Already Exists

The Pretia engine has validation steps that run before profiling begins. These may include some or all of: schema validation of inputs, model availability checks, pricing table verification, collector configuration validation, token counting sanity checks, and connectivity tests. **Your first task is to audit the codebase and catalog every existing validation or pre-flight check.**

### What the Backtest Needs Before the Pilot

The pilot calibration protocol (in `pretia-v2-additions-prompts.md`) assumes certain pre-conditions are met. These pre-conditions map to the engine's existing validation layers plus backtest-specific checks. Wire them together into a single `pre_calibration.py` that must pass before `run_pilot.py` is allowed to execute.

**Pre-calibration checklist (find or build each check):**

#### Group A: Engine Readiness (likely already in codebase)

1. **Model availability:** For each of the 11 models across 5 providers, make a trivial API call (e.g., "Say hello" with max_tokens=5) and verify a 200 response. Log model version strings from the response headers. If any model is unavailable, block the pilot for workflows that use that model.

2. **Pricing table consistency:** Load the pricing table used by the projection engine. Verify every model used by the 14 workflows has an entry. Cross-check against LiteLLM's `completion_cost()` for the same models — flag any discrepancy > 5%.

3. **Collector schema compatibility:** Load the StepRecord schema expected by the collector. Verify a sample StepRecord (from the smoke test) is accepted without validation errors.

4. **Engine configuration:** Verify the projection engine's Monte Carlo settings (number of simulations, confidence level, random seed) are set to the values specified in `projection-engine-recommendation-addition-2.md`. If settings are overridden by environment variables or config files, log the actual values.

#### Group B: Backtest-Specific Readiness (build these)

5. **Prompt inventory:** Verify every expected prompt file exists in `prompts/` per the manifest. Verify each file is non-empty and within its specified token budget range.

6. **Input inventory:** Verify both profiling and ground truth input sets exist for all 14 workflows. Check tier distributions match targets (±2 inputs). Check total counts (profiling=50, ground truth=per-workflow from v2 additions table).

7. **PDF corpus inventory:** Verify all expected PDFs exist for W14, W15, W16, W17, W18. Verify content manifests exist for W14/W15. Verify chunk embeddings exist (pre-computed vectors for retrieval simulation).

8. **Provider rate limit headroom:** Estimate the total API calls for the pilot (10 runs × 14 workflows, accounting for multi-step workflows). Check current rate limit usage against provider dashboards (if accessible via API) and warn if remaining quota is <2× the estimated pilot demand.

9. **Workspace structure:** Verify output directories exist for results, reports, and plots. Verify write permissions.

### Pre-Calibration Report

`pre_calibration.py` produces a report:

```json
{
  "timestamp": "2026-06-XX",
  "checks": {
    "model_availability": {"status": "PASS", "details": {"anthropic/claude-haiku-4.5": "available", ...}},
    "pricing_consistency": {"status": "WARN", "details": {"deepseek-v4-flash": "LiteLLM price $0.14/M vs engine $0.15/M"}},
    "collector_schema": {"status": "PASS"},
    "engine_config": {"status": "PASS", "mc_simulations": 10000, "confidence_level": 0.90},
    "prompt_inventory": {"status": "PASS", "prompt_count": 30},
    "input_inventory": {"status": "PASS", "profiling_total": 700, "ground_truth_total": 5620},
    "pdf_inventory": {"status": "PASS", "pdf_count": 1243, "embeddings_present": true},
    "rate_limit_headroom": {"status": "WARN", "anthropic_remaining": "85%"},
    "workspace": {"status": "PASS"}
  },
  "blocking_failures": [],
  "warnings": ["pricing_consistency", "rate_limit_headroom"],
  "proceed_to_pilot": true
}
```

Any Group A failure blocks the pilot. Group B failures block the pilot. Warnings are logged but don't block.

---

## Part 2: Visualization System

### Design Principles

- **Matplotlib + Seaborn** for static plots. Publication-quality, saved as PNG and PDF.
- **Plotly** for interactive HTML dashboards (hoverable, zoomable). Saved as self-contained `.html` files.
- **Consistent color palette** across all plots. Use workflow groups for color coding: green for linear/baseline, purple for loops, yellow for routing, blue for RAG/PDF.
- Every plot must have a title, axis labels, and a one-sentence caption explaining what to look for. No unlabeled plots.
- All plot functions take a results directory as input and auto-discover the data files. No hardcoded paths.

### Visualization Catalog

Organize into three categories: pilot visuals, backtest visuals, and summary dashboard.

---

#### Category 1: Pilot Visuals

Generated by `visualize_pilot.py` after the pilot completes.

**P1 — Infrastructure Check Matrix**

A heatmap (14 workflows × 7 checks). Green = pass, red = fail, gray = not applicable. At a glance, shows which workflows have infrastructure problems.

**P2 — Per-Workflow Cost Distribution (10 runs)**

One subplot per workflow (4×4 grid). Each subplot is a strip plot or swarm plot of the 10 run costs, colored by tier. Shows whether tiers separate in cost space. Annotate the min/max ratio on each subplot.

**P3 — Routing Sanity (W1, W2, W13, W17 only)**

Stacked bar chart showing the actual routing distribution vs. the designed distribution. For W13: how many went to Path A vs B vs C. For W17: how many short-circuited vs full pipeline vs routed. Overlay the target percentages as horizontal lines.

**P4 — Loop Iteration Distribution (W2, W4 only)**

Histogram of iteration counts across the 10 pilot runs. Annotate the range. Shows whether the self-assessment loop is producing variance or converging to a fixed count.

**P5 — Context Growth Verification (W2, W4, W19)**

Line plot: input tokens (y-axis) vs. step/turn number (x-axis). One line per pilot run, colored by tier. W19 should show clear linear growth. W2 should show growth that stops when the loop terminates. Annotate the Pearson correlation for each run.

**P6 — Cost Plausibility Scatter**

Scatter plot: expected per-run cost (x-axis, from v2 additions table) vs. actual per-run cost (y-axis, from pilot). One dot per workflow. Draw the 0.5× and 5× boundary lines. Dots outside the boundaries are labeled with the workflow ID.

---

#### Category 2: Backtest Visuals

Generated by `visualize_backtest.py` after each comparison completes.

**B1 — Projected vs. Actual Cost Distribution**

One subplot per workflow. Overlay two distributions:
- The projected distribution (from the n=50 profiling, rendered as a smooth KDE curve)
- The actual ground truth distribution (histogram)

For Comparison B, also overlay the reweighted projection (Comparison C) as a dashed curve. This is the single most important visualization — it shows at a glance whether the projection matches reality.

**B2 — Accuracy Metrics Heatmap**

A heatmap (14 workflows × 5 metrics × 3 comparisons). Color scale: green (<target), yellow (near target), red (>target). Immediately shows which workflows fail which metrics in which comparison.

**B3 — Drift Impact Chart**

Grouped bar chart: for each workflow, three bars showing mean error % for Comparison A, B, and C. A is the baseline, B shows drift degradation, C shows reweighting recovery. The gap between A and B is the drift impact. The gap between B and C is the reweighting recovery.

**B4 — Reweighting Recovery Scatter**

Scatter plot: drift impact (B accuracy - A accuracy) on x-axis vs. reweighting recovery (B accuracy - C accuracy) on y-axis. Workflows in the top-right are sensitive to drift but recoverable via reweighting. Workflows in the bottom-right are sensitive and not recoverable. Draw quadrant lines.

**B5 — Detector Activation Matrix**

A heatmap (14 workflows × 6 detectors). Four colors: true positive (green), true negative (light gray), false negative (red), false positive (yellow). Annotate with detector statistics (TP rate, FN rate across all workflows).

**B6 — Bimodality Visualization (W13, W17, W15)**

For workflows with expected bimodality: histogram of per-run costs with the BIC-estimated GMM overlay (two Gaussians). Annotate ΔBIC value. Show the separation between modes.

**B7 — Confidence Interval Coverage**

Per workflow: the projected 90% CI as a shaded band, with ground truth run costs plotted as dots. Dots inside the band are green, outside are red. Annotate the coverage percentage. Target: ≥85% for no-drift, ≥75% for drifted.

**B8 — Tail Risk (CVaR) Comparison**

Bar chart: projected CVaR95 vs. actual CVaR95 per workflow, for each comparison. Shows whether the engine correctly estimates the cost of the most expensive 5% of runs.

**B9 — Per-Step Cost Breakdown**

Stacked bar chart per workflow: each segment is a step's mean cost contribution. Shows which steps dominate cost. For multi-step workflows (W2, W16, W17), this reveals where the projection engine is accurate (correctly models the expensive step) and where it's off (underestimates a cheap step that accumulates).

**B10 — Token Distribution Comparison (Profiling vs. Ground Truth)**

Per workflow: overlaid histograms of input token counts for profiling (blue) vs. ground truth (orange). Shows the token length stretch dimension of drift. Annotate the mean shift.

---

#### Category 3: Summary Dashboard

Generated by `generate_dashboard.py`. A single self-contained HTML file using Plotly.

**D1 — Executive Summary**

Top of the dashboard. Three large numbers:
- Workflows passing all comparisons (green)
- Workflows needing reweighting (yellow)  
- Workflows with unresolved issues (red)

Plus total backtest cost, total runs, and date.

**D2 — Interactive Results Table**

Sortable/filterable table with one row per workflow. Columns: workflow ID, pattern type, providers, Comparison A pass/fail, Comparison B pass/fail, Comparison C pass/fail (if applicable), failure bucket, mean error %, reweighting recovery %. Click a row to expand and show the per-step breakdown and cost distribution plot.

**D3 — Interactive Projected vs. Actual Explorer**

Dropdown to select a workflow. Shows B1 (projected vs. actual distribution) interactively with hover data. Toggle between Comparison A, B, and C with radio buttons.

**D4 — Detector Dashboard**

Interactive version of B5. Click a cell to see the raw detector output (test statistic, threshold, p-value) and the underlying data pattern (e.g., the context growth curve that triggered or failed to trigger the detector).

**D5 — Budget Tracker**

Running total of API spend across all pilot and backtest runs, broken down by provider and workflow. Shows remaining budget vs. planned budget.

---

## Part 3: Narrative Report Generator

Beyond plots, generate a human-readable markdown report that interprets the results. This is the document you hand to someone who needs to understand the backtesting outcome without looking at 25 plots.

**`generate_narrative.py`** takes the backtest results and produces a markdown file with:

1. **Executive summary** (3–5 sentences): How many workflows passed, which ones failed, whether reweighting helps, overall engine reliability assessment.

2. **Per-workflow sections** (only for workflows that didn't pass cleanly): What failed, which comparison, the likely cause (from failure attribution), the recommended action.

3. **Drift analysis**: Which drift dimensions had the most impact? Was it the tier weight shift (compensable) or the style/length shift (structural)? This comes from comparing the accuracy degradation patterns — if all workflows degrade similarly, it's the tier shift; if only loop-based workflows degrade, it's the style shift affecting iteration counts.

4. **Detector reliability assessment**: TP rate and FN rate across all workflows. Any false positives that revealed unexpected cost patterns.

5. **Recommendations**: Whether `--traffic-mix` should be recommended to all users, specific workflows that need additional profiling, known limitations to document.

---

## Part 4: File Manifest

```
visualization/
  colors.py                   # Shared color palette, workflow group colors, consistent styling
  utils.py                    # Plot helpers: load results, format labels, save figures
  
  pilot/
    visualize_pilot.py        # Generates P1–P6
    
  backtest/
    visualize_backtest.py     # Generates B1–B10
    
  dashboard/
    generate_dashboard.py     # Generates the Plotly HTML dashboard (D1–D5)
    
  narrative/
    generate_narrative.py     # Generates the markdown narrative report
    templates/                # Jinja2 templates for the narrative sections
    
pre_calibration/
  pre_calibration.py          # Runs all pre-calibration checks, produces report JSON
  checks/
    model_availability.py     # Group A check 1
    pricing_consistency.py    # Group A check 2
    schema_compatibility.py   # Group A check 3
    engine_config.py          # Group A check 4
    prompt_inventory.py       # Group B check 5
    input_inventory.py        # Group B check 6
    pdf_inventory.py          # Group B check 7
    rate_limit_check.py       # Group B check 8
    workspace_check.py        # Group B check 9
```

Each visualization script is runnable standalone:
```
python visualize_pilot.py --results-dir results/pilot/ --output-dir reports/pilot/plots/
python visualize_backtest.py --results-dir results/backtest/ --comparison B --output-dir reports/backtest/plots/
python generate_dashboard.py --results-dir results/ --output reports/dashboard.html
python generate_narrative.py --results-dir results/ --output reports/narrative.md
python pre_calibration.py --prompts-dir prompts/ --inputs-dir inputs/generated/ --pdfs-dir pdfs/generated/ --output reports/pre_calibration.json
```
