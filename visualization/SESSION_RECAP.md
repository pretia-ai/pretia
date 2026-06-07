# Visualization & Reporting System — Build Recap

Built in one session, June 2026. This document provides context for future sessions that depend on this work.

---

## What Was Built

A complete visualization, pre-calibration, and reporting system for the AgentCost backtesting suite. Everything lives outside the `agentcost/` package (at repo root) except the data model extensions.

### Phase 0: Data Model Extension

**Files modified:**
- `agentcost/validation/scoring.py` — added `ComparisonScore` dataclass (5 accuracy metrics: mean_error, p75_error, ci_coverage, monthly_error, cvar95_error) with comparison-specific targets (`_COMPARISON_TARGETS`: no-drift vs drifted) and `score_comparison()` function
- `agentcost/validation/suite.py` — added `FailureAttribution` dataclass (bucket 1/2/3 classification), `ComparisonResult` dataclass, `attribute_failure()` implementing the failure attribution flowchart, `_compute_recovery()` helper
- `agentcost/validation/__init__.py` — exports updated with new types
- `pyproject.toml` — added `visualization` optional dep group (matplotlib, seaborn, plotly)

**Key design decision:** `ComparisonScore` exists alongside the original `CalibrationScore` — no breaking changes to existing backtesting code.

### Phase 1: Shared Visualization Infrastructure

**Files created:**
- `visualization/__init__.py`
- `visualization/colors.py` — `WORKFLOW_GROUPS` (14 workflows → 4 groups: linear/loops/routing/rag_pdf), `GROUP_COLORS`, `COMPARISON_COLORS`, `DETECTOR_MATRIX_COLORS`, `VERDICT_COLORS`, `EXPECTED_DETECTORS` (14×5 expected detector activation matrix from cross-cutting-robustness.md), `workflow_color()`, `classify_detector_result()` (returns TP/TN/FP/FN)
- `visualization/utils.py` — `discover_results()` (auto-discovers JSON by workflow/comparison pattern), `save_figure()` (PNG+PDF), `add_caption()`, `format_workflow_label()`, `ensure_output_dir()`
- Empty `__init__.py` in `visualization/pilot/`, `visualization/backtest/`, `visualization/dashboard/`, `visualization/narrative/`

### Phase 2: Pre-Calibration System (9 checks)

**Files created:**
- `pre_calibration/__init__.py`
- `pre_calibration/pre_calibration.py` — `CheckResult` and `PreCalibrationReport` dataclasses, `run_pre_calibration()` async orchestrator, Click CLI
- `pre_calibration/checks/` — 9 modules: `model_availability.py`, `pricing_consistency.py`, `schema_compatibility.py`, `engine_config.py`, `prompt_inventory.py`, `input_inventory.py`, `pdf_inventory.py`, `rate_limit_check.py`, `workspace_check.py`

**Key behavior:** Group A + B failures block the pilot (`proceed_to_pilot=False`). Warnings don't block. litellm-dependent checks degrade gracefully to WARN when litellm is not installed.

### Phase 3: Pilot Visualizations (P1-P6)

**File created:** `visualization/pilot/visualize_pilot.py`

| Plot | Function | What it shows |
|------|----------|--------------|
| P1 | `p1_infrastructure_check_matrix()` | Heatmap of workflows × checks (green/yellow/red) |
| P2 | `p2_per_workflow_cost_distribution()` | 4×N grid of scatter plots, 10 runs each, max/min ratio |
| P3 | `p3_routing_sanity_bars()` | Stacked bars for W1/W2/W13/W17 only |
| P4 | `p4_loop_iteration_histogram()` | Histograms for W2/W4 only |
| P5 | `p5_context_growth_lines()` | Scatter + trend line for W2/W4/W19, Pearson r annotated |
| P6 | `p6_cost_plausibility_scatter()` | Log-scale scatter with 0.5×/5× boundary lines |

All functions: take `results_dir` + `output_dir`, return `list[Path]`, gracefully skip if data missing.

### Phase 4: Backtest Visualizations (B1-B10)

**File created:** `visualization/backtest/visualize_backtest.py`

| Plot | Function | Per-comparison? |
|------|----------|----------------|
| B1 | `b1_projected_vs_actual_kde()` | No |
| B2 | `b2_accuracy_metrics_heatmap()` | Yes (A/B/C) |
| B3 | `b3_drift_impact_grouped_bars()` | No |
| B4 | `b4_reweighting_recovery_scatter()` | No |
| B5 | `b5_detector_activation_matrix()` | No |
| B6 | `b6_bimodality_gmm_overlay()` | Per-workflow (W13/W15/W17) |
| B7 | `b7_ci_coverage_bands()` | Yes (A/B/C) |
| B8 | `b8_cvar_comparison_bars()` | Yes (A/B/C) |
| B9 | `b9_per_step_cost_breakdown()` | No |
| B10 | `b10_token_distribution_comparison()` | Yes (A/B/C) |

Custom `_gaussian_kde_1d()` implementation avoids scipy dependency. Each accepts `comparison` param ("A", "B", "C", or "all").

### Phase 5: Interactive Dashboard + Narrative Report

**Dashboard:** `visualization/dashboard/generate_dashboard.py`

Self-contained HTML (plotly.js inlined, PNG plots base64-embedded). 8 tabs:
- **Summary** — 3 cards (passing/reweighting/unresolved) with workflow names, launch gate verdict, key findings
- **Pilot Plots** — P1-P6 embedded as collapsible `<details>` (expanded by default)
- **Backtest Plots** — B1-B10 embedded as collapsible `<details>` (collapsed by default)
- **Results** — Plotly table with color-coded pass/fail and bucket columns
- **Analytics** — 5 subsections (see below)
- **Explorer** — Dropdown to compare projected vs actual per-step costs
- **Detectors** — Interactive TP/TN/FP/FN heatmap with hover
- **Budget** — Spend by workflow, colored by group

**Analytics section (within dashboard):**
- A. Variance & Risk table (sorted by cost CV)
- B. Cost Attribution treemap (sized by cost, colored by model tier)
- C. Pain Points table (context overhead, loop amplifiers, retries, unpredictable steps)
- D. Optimization Opportunities table (sorted by monthly savings, with totals)
- E. Per-Step Deep Dive (dropdown → grouped bar with error bars)

**15 optimization detectors in `_detect_opportunities()`:**

| # | Detector | Signal |
|---|----------|--------|
| 1 | model_downshift | Frontier/mid model on low out/in ratio or JSON task |
| 2 | context_compaction | context_growth pattern + mean_iterations > 2 |
| 3 | iteration_cap | loop_count_variance + iterations_max > 1.5× mean |
| 4 | caching | system_prompt > 40% of input tokens |
| 5 | prompt_optimization | system_prompt > 500 tokens or tool_defs > 30% of input |
| 6 | architecture_restructuring | context p95 > 3× output mean |
| 7 | high_variance_step | cost CV > 0.8 or p95/p50 > 3× |
| 8 | token_waste | output p95 > 3× p50 |
| 9 | tool_filtering | tool_definitions > 30% of input |
| 10 | output_truncation | output_truncated_pct > 10% |
| 11 | retry_cascade | retry_pct > 10% or tool_retry > 1 |
| 12 | temperature_optimization | temp > 0 on deterministic task (JSON or low out/in) |
| 13 | tool_output_bloat | tool_output > 2× tool_input |
| 14 | redundant_context | two steps with context_size within 15% and > 500 tokens |
| 15 | batching | call_count > 2× runs_present with mean_iterations ≈ 1 |

**Narrative report:** `visualization/narrative/generate_narrative.py` + 6 Jinja2 templates in `visualization/narrative/templates/`. Outputs markdown with 5 sections: executive summary, per-workflow failures (bucketed), drift analysis, detector assessment, recommendations.

### Phase 6: Integration Wiring

**File modified:** `tests/backtesting/run_backtesting.py`
- Added Phase 4 (`_run_phase_4()`) — calls all visualization generators after scoring
- Added `_run_pre_calibration()` — gates pilot with `--pre-calibrate` flag
- `--all` now runs phases 1-4
- `--phase 4` runs visualization standalone

### LLM Export

**File created:** `visualization/export.py`
- `export_analytics_json()` — 52KB consolidated JSON with: launch gate, validation (A/B/C scores per workflow), detector matrix + rates, drift analysis, variance risk, cost attribution, pain points, 63 opportunities with savings, plot summaries in natural language
- `export_analytics_pdf()` — 8MB PDF using reportlab with: title page, validation table (color-coded), detector matrix, opportunities table (top 25), pain points, and all 26 embedded PNG plots

**File created:** `visualization/llm_analysis_prompt.md` — prompt template for sending analytics.json + analytics.pdf to an LLM. Requests 7 analysis sections: launch gate, accuracy, drift, detectors, recommendations, risk, action plan.

### Demo Script

**File created:** `visualization/demo.py`
- Generates synthetic mock data for all 14 workflows with deliberate pain points
- Runs all 7 generation steps: mock data → pilot plots → backtest plots → narrative → dashboard → JSON export → PDF export
- Auto-opens dashboard in browser
- CLI: `python3.12 visualization/demo.py`

Mock data includes realistic `step_stats` per workflow with: model, tier, cost distributions (p10-p95), token distributions, system prompt tokens, tool definition tokens, iteration ranges, retry rates, temperature, tool I/O tokens, truncation rates.

### Bug Fixes

1. **PIL/Pillow import collision** — `tests/unit/test_pdf_chart_scan.py` was mocking PIL at module level (`sys.modules.setdefault("PIL", MagicMock())`), breaking matplotlib when pytest collected all tests together. Fixed by removing the mock (Pillow is now installed).
2. **`agents` module mock collision** — `tests/unit/test_openai_agents_collector.py` used `sys.modules.setdefault("agents", ...)` which didn't work when the real `agents/` package was loaded first. Fixed with unconditional `sys.modules["agents"] = ...`.
3. **Dashboard CDN test** — plotly.js bundle itself contains internal `cdn.plot.ly` references. Fixed test to check for `src=` attributes instead of string presence.
4. **Pattern data handling** — narrative and dashboard tried to put pattern dicts in a set. Fixed to extract `pattern_type` strings first.

---

## Test Coverage

**88 new tests** across 8 test files:

| File | Tests | What it covers |
|------|-------|---------------|
| `test_comparison_scoring.py` | 13 | ComparisonScore, score_comparison(), failure attribution buckets 1/2/3 |
| `test_viz_colors.py` | 9 | Workflow groups, color palette, detector TP/TN/FP/FN classification |
| `test_viz_utils.py` | 10 | Results discovery, figure saving, captions, label formatting |
| `test_pre_calibration.py` | 15 | CheckResult blocking, orchestrator, individual checks, JSON output |
| `test_viz_pilot.py` | 13 | P1-P6 generation + graceful skip when data missing |
| `test_viz_backtest.py` | 14 | B1-B10 generation, correct targets per comparison, detector classification |
| `test_viz_dashboard.py` | 6 | HTML generation, self-contained check, executive summary |
| `test_viz_narrative.py` | 8 | 5 sections, bucket attribution, drift analysis, detector rates |

Run: `python3.12 -m pytest tests/unit/test_comparison_scoring.py tests/unit/test_viz_colors.py tests/unit/test_viz_utils.py tests/unit/test_pre_calibration.py tests/unit/test_viz_pilot.py tests/unit/test_viz_backtest.py tests/unit/test_viz_dashboard.py tests/unit/test_viz_narrative.py -v`

Full suite: `python3.12 -m pytest tests/unit/ -q` — 1201 passed, 5 skipped, 5 pre-existing failures in `test_run_workflow.py`.

---

## File Inventory

```
# Data model extensions (inside agentcost/)
agentcost/validation/scoring.py          # ComparisonScore, score_comparison(), _COMPARISON_TARGETS
agentcost/validation/suite.py            # FailureAttribution, ComparisonResult, attribute_failure()
agentcost/validation/__init__.py         # Updated exports

# Visualization system (repo root)
visualization/__init__.py
visualization/colors.py                  # Palettes, EXPECTED_DETECTORS, classify_detector_result()
visualization/utils.py                   # discover_results(), save_figure(), add_caption()
visualization/pilot/visualize_pilot.py   # P1-P6
visualization/backtest/visualize_backtest.py  # B1-B10
visualization/dashboard/generate_dashboard.py  # HTML dashboard + analytics section + 15 detectors
visualization/narrative/generate_narrative.py   # Markdown narrative
visualization/narrative/templates/*.md.j2       # 6 Jinja2 templates
visualization/export.py                  # JSON + PDF export for LLM analysis
visualization/demo.py                   # Mock data + full generation pipeline
visualization/llm_analysis_prompt.md     # Prompt template for LLM analysis

# Pre-calibration (repo root)
pre_calibration/__init__.py
pre_calibration/pre_calibration.py       # Orchestrator + CheckResult/PreCalibrationReport
pre_calibration/checks/*.py              # 9 check modules

# Integration
tests/backtesting/run_backtesting.py     # Phase 4 + --pre-calibrate flag

# Tests
tests/unit/test_comparison_scoring.py
tests/unit/test_viz_colors.py
tests/unit/test_viz_utils.py
tests/unit/test_pre_calibration.py
tests/unit/test_viz_pilot.py
tests/unit/test_viz_backtest.py
tests/unit/test_viz_dashboard.py
tests/unit/test_viz_narrative.py

# Generated demo output (not committed)
reports/demo/                            # All demo output
```

---

## Dependencies Added

`pyproject.toml` `[project.optional-dependencies]`:
```
visualization = ["matplotlib>=3.8", "seaborn>=0.13", "plotly>=5.18"]
```

Also installed at runtime: `Pillow` (for matplotlib PNG support), `plotly` (for dashboard). `reportlab` was already in `pdf-generation` extras.

---

## How to Regenerate the Demo

```bash
python3.12 visualization/demo.py
```

Produces in `reports/demo/`:
- 12 pilot plot files (PNG+PDF)
- 40 backtest plot files (PNG+PDF)
- `narrative.md` — markdown report
- `dashboard.html` — 14MB self-contained interactive dashboard
- `analytics.json` — 52KB LLM-readable structured data
- `analytics.pdf` — 8MB PDF with plots + tables

---

## Key Architectural Decisions

1. **Visualization code lives outside `agentcost/`** — not part of the pip package, it's a dev/analysis tool
2. **Pre-calibration also outside `agentcost/`** — operational tool for running backtests
3. **No new dataclasses in `agentcost/` except** `ComparisonScore`, `FailureAttribution`, `ComparisonResult`
4. **Dashboard is one self-contained HTML** — plotly.js inlined, plots base64-embedded, no CDN
5. **15 detectors are pure functions** — no ML, just threshold-based rules with calculable savings
6. **JSON export is the LLM interface** — compact, self-documenting, includes plot summaries in natural language so LLMs get the visual story without images
