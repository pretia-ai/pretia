#!/usr/bin/env python3.12
"""Generate demo visualizations with synthetic mock data.

Run: python3.12 visualization/demo.py

Produces all plots (P1-P6 pilot, B1-B10 backtest), narrative report,
and interactive HTML dashboard in reports/demo/.
"""

from __future__ import annotations

import json
import random
import webbrowser
from pathlib import Path

DEMO_DIR = Path("reports/demo")
RESULTS_DIR = DEMO_DIR / "results"

WORKFLOWS = ["W1", "W2", "W4", "W5", "W9", "W11", "W12", "W13",
             "W14", "W15", "W16", "W17", "W18", "W19"]


def _generate_pre_calibration_report() -> None:
    """Write a mock pre-calibration report."""
    report = {
        "timestamp": "2026-06-07T12:00:00+00:00",
        "checks": {
            "model_availability": {"status": "PASS", "details": {
                "claude-haiku-4.5": "available", "claude-sonnet-4.6": "available",
                "deepseek-v4": "available", "gpt-5.4-nano": "available",
                "qwen-3.6-plus": "available",
            }},
            "pricing_consistency": {"status": "WARN", "details": {
                "deepseek-v4-flash": "LiteLLM price $0.14/M vs engine $0.15/M"
            }},
            "collector_schema": {"status": "PASS", "details": {"fields_verified": 14}},
            "engine_config": {"status": "PASS", "details": {
                "mc_simulations": 10000, "confidence_level": 0.90
            }},
            "prompt_inventory": {"status": "PASS", "details": {"prompt_count": 30}},
            "input_inventory": {"status": "PASS", "details": {
                "profiling_total": 700, "ground_truth_total": 5620
            }},
            "pdf_inventory": {"status": "PASS", "details": {
                "pdf_count": 1243, "embeddings_present": True
            }},
            "rate_limit_headroom": {"status": "WARN", "details": {
                "anthropic_remaining": "85%"
            }},
            "workspace": {"status": "PASS", "details": {}},
        },
        "blocking_failures": [],
        "warnings": ["pricing_consistency", "rate_limit_headroom"],
        "proceed_to_pilot": True,
    }
    (RESULTS_DIR / "pre_calibration.json").write_text(json.dumps(report, indent=2))


def _generate_pilot_data() -> None:
    """Write mock pilot data (10 runs per workflow)."""
    rng = random.Random(42)
    cost_ranges = {
        "W1": (0.01, 0.06), "W2": (0.08, 1.80), "W4": (0.005, 0.15),
        "W5": (0.02, 0.40), "W9": (0.005, 0.10), "W11": (0.001, 0.025),
        "W12": (0.002, 0.05), "W13": (0.003, 0.35), "W14": (0.03, 0.65),
        "W15": (0.01, 0.55), "W16": (0.02, 0.30), "W17": (0.005, 0.25),
        "W18": (0.005, 0.09), "W19": (0.05, 0.65),
    }

    for wf in WORKFLOWS:
        lo, hi = cost_ranges.get(wf, (0.01, 0.10))
        costs = [rng.uniform(lo, hi) for _ in range(10)]
        data = {
            "metadata": {
                "stats": {
                    "run_stats": [
                        {"total_cost": c, "run_index": i, "total_tokens": int(c * 50000),
                         "total_input_tokens": int(c * 35000), "total_output_tokens": int(c * 15000),
                         "step_count": rng.randint(1, 5), "duration_ms": int(c * 10000)}
                        for i, c in enumerate(costs)
                    ],
                }
            }
        }
        (RESULTS_DIR / f"{wf}_pilot.json").write_text(json.dumps(data, indent=2))


_MODEL_PRICES = {
    "claude-opus-4.7": 15.0, "claude-sonnet-4.6": 3.0, "claude-haiku-4.5": 0.25,
    "gpt-5.4": 5.0, "gpt-5.4-nano": 0.10, "deepseek-v4": 0.14,
    "deepseek-v4-flash": 0.07, "qwen-3.6-plus": 0.50, "qwen-turbo": 0.05,
    "gemini-2.5-flash": 0.15, "text-embedding-3-small": 0.02,
}

_WORKFLOW_STEPS: dict[str, list[dict]] = {
    "W1": [
        {"name": "classify", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 280, "tool_def_tok": 0,
         "in_tok": 340, "out_tok": 45, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "respond", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 450, "tool_def_tok": 0,
         "in_tok": 800, "out_tok": 350, "iters": (1, 1), "retry_pct": 2.0},
    ],
    "W2": [
        {"name": "classify", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 280, "tool_def_tok": 0,
         "in_tok": 340, "out_tok": 45, "iters": (1, 1), "retry_pct": 0.0,
         "temp": 0.3},
        {"name": "generate", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 600, "tool_def_tok": 0,
         "in_tok": 1200, "out_tok": 500, "iters": (3, 8), "retry_pct": 5.0},
        {"name": "review", "model": "claude-opus-4.7", "tier": "frontier",
         "fmt": "text", "sys_prompt_tok": 800, "tool_def_tok": 0,
         "in_tok": 3000, "out_tok": 400, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W4": [
        {"name": "draft", "model": "deepseek-v4", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 500, "tool_def_tok": 0,
         "in_tok": 2000, "out_tok": 1500, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "critique", "model": "qwen-3.6-plus", "tier": "mid",
         "fmt": "json", "sys_prompt_tok": 400, "tool_def_tok": 0,
         "in_tok": 3500, "out_tok": 300, "iters": (2, 6), "retry_pct": 3.0},
    ],
    "W5": [
        {"name": "extract", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "json", "sys_prompt_tok": 350, "tool_def_tok": 200,
         "in_tok": 2500, "out_tok": 180, "iters": (1, 1), "retry_pct": 8.0,
         "trunc_pct": 12.0, "tool_retry": 2.0, "temp": 0.3},
    ],
    "W9": [
        {"name": "draft", "model": "gpt-5.4-nano", "tier": "fast",
         "fmt": "text", "sys_prompt_tok": 300, "tool_def_tok": 0,
         "in_tok": 400, "out_tok": 200, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "polish", "model": "gpt-5.4", "tier": "frontier",
         "fmt": "text", "sys_prompt_tok": 350, "tool_def_tok": 350,
         "in_tok": 700, "out_tok": 250, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W11": [
        {"name": "respond", "model": "qwen-turbo", "tier": "fast",
         "fmt": "text", "sys_prompt_tok": 280, "tool_def_tok": 0,
         "in_tok": 320, "out_tok": 120, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W12": [
        {"name": "extract", "model": "deepseek-v4-flash", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 200, "tool_def_tok": 150,
         "in_tok": 500, "out_tok": 80, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W13": [
        {"name": "classify", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 350, "tool_def_tok": 0,
         "in_tok": 400, "out_tok": 30, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "route_simple", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "text", "sys_prompt_tok": 300, "tool_def_tok": 0,
         "in_tok": 500, "out_tok": 150, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "route_complex", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 600, "tool_def_tok": 0,
         "in_tok": 1500, "out_tok": 600, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W14": [
        {"name": "embed", "model": "text-embedding-3-small", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 0, "tool_def_tok": 0,
         "in_tok": 800, "out_tok": 0, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "generate", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "json", "sys_prompt_tok": 500, "tool_def_tok": 0,
         "in_tok": 3000, "out_tok": 400, "iters": (1, 1), "retry_pct": 2.0},
    ],
    "W15": [
        {"name": "embed", "model": "text-embedding-3-small", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 0, "tool_def_tok": 0,
         "in_tok": 800, "out_tok": 0, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "assess", "model": "gemini-2.5-flash", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 300, "tool_def_tok": 0,
         "in_tok": 2000, "out_tok": 100, "iters": (1, 4), "retry_pct": 0.0},
        {"name": "generate", "model": "deepseek-v4", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 600, "tool_def_tok": 0,
         "in_tok": 5000, "out_tok": 800, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W16": [
        {"name": "split", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 200, "tool_def_tok": 0,
         "in_tok": 600, "out_tok": 50, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "process", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "text", "sys_prompt_tok": 300, "tool_def_tok": 0,
         "in_tok": 1200, "out_tok": 300, "iters": (3, 20), "retry_pct": 0.0},
        {"name": "aggregate", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 400, "tool_def_tok": 0,
         "in_tok": 4000, "out_tok": 600, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W17": [
        {"name": "intake", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 800, "tool_def_tok": 400,
         "in_tok": 1500, "out_tok": 100, "iters": (1, 1), "retry_pct": 0.0},
        {"name": "retrieve", "model": "text-embedding-3-small", "tier": "fast",
         "fmt": "json", "sys_prompt_tok": 0, "tool_def_tok": 0,
         "in_tok": 600, "out_tok": 0, "iters": (1, 1), "retry_pct": 0.0,
         "tool_in": 600, "tool_out": 3000},
        {"name": "evaluate", "model": "claude-sonnet-4.6", "tier": "mid",
         "fmt": "json", "sys_prompt_tok": 1200, "tool_def_tok": 500,
         "in_tok": 4000, "out_tok": 300, "iters": (1, 1), "retry_pct": 3.0},
    ],
    "W18": [
        {"name": "process", "model": "deepseek-v4", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 400, "tool_def_tok": 0,
         "in_tok": 60000, "out_tok": 2000, "iters": (1, 1), "retry_pct": 0.0},
    ],
    "W19": [
        {"name": "respond", "model": "deepseek-v4", "tier": "mid",
         "fmt": "text", "sys_prompt_tok": 500, "tool_def_tok": 0,
         "in_tok": 1500, "out_tok": 200, "iters": (1, 8), "retry_pct": 0.0},
    ],
}


def _build_step_stats(wf: str, base_cost: float, rng: random.Random) -> dict:
    """Build realistic step_stats with deliberate pain points per workflow."""
    step_defs = _WORKFLOW_STEPS.get(wf, [
        {"name": "step1", "model": "claude-haiku-4.5", "tier": "fast",
         "fmt": "text", "sys_prompt_tok": 280, "tool_def_tok": 0,
         "in_tok": 400, "out_tok": 100, "iters": (1, 1), "retry_pct": 0.0},
    ])

    stats: dict = {}
    for sd in step_defs:
        in_mean = sd["in_tok"]
        out_mean = sd["out_tok"]
        in_std = int(in_mean * rng.uniform(0.15, 0.35))
        out_std = int(out_mean * rng.uniform(0.2, 0.5)) if out_mean > 0 else 0
        iter_lo, iter_hi = sd["iters"]
        mean_iters = (iter_lo + iter_hi) / 2
        price = _MODEL_PRICES.get(sd["model"], 1.0)
        step_cost_mean = (in_mean + out_mean) / 1_000_000 * price * mean_iters
        cost_std = step_cost_mean * rng.uniform(0.15, 0.5)
        cost_cv = cost_std / step_cost_mean if step_cost_mean > 0 else 0

        ctx_mean = in_mean + sd["sys_prompt_tok"] + sd["tool_def_tok"]
        ctx_std = int(ctx_mean * 0.2)

        stats[sd["name"]] = {
            "model": sd["model"],
            "model_tier": sd["tier"],
            "cost": {
                "mean": round(step_cost_mean, 6),
                "std": round(cost_std, 6),
                "p50": round(step_cost_mean * 0.95, 6),
                "p95": round(step_cost_mean * (1.5 + cost_cv), 6),
                "p10": round(step_cost_mean * 0.6, 6),
                "p90": round(step_cost_mean * 1.4, 6),
                "min": round(step_cost_mean * 0.3, 6),
                "max": round(step_cost_mean * (2.0 + cost_cv * 2), 6),
            },
            "input_tokens": {
                "mean": in_mean, "std": in_std,
                "p50": int(in_mean * 0.95), "p95": int(in_mean * 1.5),
            },
            "output_tokens": {
                "mean": out_mean, "std": out_std,
                "p50": int(out_mean * 0.9), "p95": int(out_mean * (1.8 if out_mean > 0 else 0)),
            },
            "context_size": {
                "mean": ctx_mean, "std": ctx_std,
                "p50": int(ctx_mean * 0.95), "p95": int(ctx_mean * 1.4),
            },
            "system_prompt_tokens": sd["sys_prompt_tok"],
            "tool_definitions_tokens": sd["tool_def_tok"],
            "mean_iterations": mean_iters,
            "iterations_min": iter_lo,
            "iterations_max": iter_hi,
            "runs_present": 50,
            "call_count": int(50 * mean_iters),
            "output_format": sd["fmt"],
            "is_retry_pct": sd["retry_pct"],
            "temperature": sd.get("temp"),
            "tool_input_tokens": sd.get("tool_in", 0),
            "tool_output_tokens": sd.get("tool_out", 0),
            "max_tokens_setting": sd.get("max_tok", 4096),
            "output_truncated_pct": sd.get("trunc_pct", 0.0),
            "tool_retry_count_mean": sd.get("tool_retry", 0.0),
        }

    return stats


def _generate_backtest_data() -> None:
    """Write mock backtest data for all 14 workflows with 3 comparisons."""
    rng = random.Random(42)
    cost_ranges = {
        "W1": 0.02, "W2": 0.36, "W4": 0.03, "W5": 0.08, "W9": 0.02,
        "W11": 0.005, "W12": 0.01, "W13": 0.07, "W14": 0.13, "W15": 0.11,
        "W16": 0.06, "W17": 0.05, "W18": 0.018, "W19": 0.13,
    }

    # Workflows that should show specific patterns
    bimodal_workflows = {"W13", "W17", "W15"}
    loop_workflows = {"W2", "W4", "W15", "W19"}
    linear_workflows = {"W1", "W9", "W11", "W12", "W18"}

    for wf in WORKFLOWS:
        base_cost = cost_ranges.get(wf, 0.05)

        def _make_costs(n: int, drift: float = 0.0, bimodal: bool = False) -> list[float]:
            costs = []
            for _ in range(n):
                if bimodal and rng.random() < 0.3:
                    c = base_cost * rng.uniform(3.0, 8.0) * (1 + drift)
                else:
                    c = base_cost * rng.uniform(0.5, 2.0) * (1 + drift)
                costs.append(c)
            return costs

        is_bimodal = wf in bimodal_workflows
        prof_costs = _make_costs(50)
        gt_a_costs = _make_costs(200)
        gt_b_costs = _make_costs(200, drift=0.3, bimodal=is_bimodal)

        prof_mean = sum(prof_costs) / len(prof_costs)
        gt_a_mean = sum(gt_a_costs) / len(gt_a_costs)
        gt_b_mean = sum(gt_b_costs) / len(gt_b_costs)

        mean_err_a = abs(prof_mean - gt_a_mean) / gt_a_mean * 100 if gt_a_mean > 0 else 0
        mean_err_b = abs(prof_mean - gt_b_mean) / gt_b_mean * 100 if gt_b_mean > 0 else 0
        mean_err_c = mean_err_b * 0.4 if wf not in {"W19"} else mean_err_b * 0.85

        passes_a = mean_err_a < 10
        passes_b = mean_err_b < 20
        passes_c = mean_err_c < 20

        detected_patterns = []
        if wf in loop_workflows:
            detected_patterns.append({
                "pattern_type": "context_growth", "step_name": "generate",
                "severity": "DANGER", "evidence": {"pearson_r": 0.85},
                "description": f"Context grows linearly across iterations in {wf}",
            })
            detected_patterns.append({
                "pattern_type": "loop_count_variance", "step_name": "generate",
                "severity": "WARNING", "evidence": {"cv": 0.6},
                "description": f"Loop iterations vary significantly in {wf}",
            })
        if is_bimodal:
            detected_patterns.append({
                "pattern_type": "bimodality", "step_name": "workflow",
                "severity": "DANGER",
                "evidence": {"bic_delta": 12.5},
                "description": f"Cost distribution is bimodal in {wf}",
                "gmm_means": [base_cost, base_cost * 5],
                "gmm_stds": [base_cost * 0.3, base_cost * 1.5],
                "gmm_weights": [0.7, 0.3],
                "bimodal_bic_delta": 12.5,
            })
        if wf in {"W5", "W18", "W16"}:
            detected_patterns.append({
                "pattern_type": "high_token_variance", "step_name": "process",
                "severity": "WARNING", "evidence": {"p95_p50_ratio": 4.2},
                "description": f"High token variance in {wf}",
            })
        if wf in {"W2", "W13", "W15", "W16", "W17"}:
            detected_patterns.append({
                "pattern_type": "step_count_variance", "step_name": "workflow",
                "severity": "WARNING", "evidence": {"cv": 0.45},
                "description": f"Step count varies in {wf}",
            })

        steps = {"classify": base_cost * 0.1, "generate": base_cost * 0.7,
                 "format": base_cost * 0.2}

        step_stats = _build_step_stats(wf, base_cost, rng)

        data = {
            "workflow_name": wf,
            "comparisons": {
                "A": {
                    "score": {
                        "workflow_name": wf, "comparison": "A",
                        "mean_error_pct": round(mean_err_a, 1),
                        "p75_error_pct": round(mean_err_a * 1.4, 1),
                        "ci_coverage_pct": round(90 - mean_err_a * 0.5, 1),
                        "monthly_error_pct": round(mean_err_a * 0.9, 1),
                        "cvar95_error_pct": round(mean_err_a * 2.2, 1),
                        "passes": passes_a, "failures": [],
                    },
                    "profiling_run_costs": prof_costs,
                    "ground_truth_run_costs": gt_a_costs,
                    "projected_ci": [min(prof_costs) * 0.8, max(prof_costs) * 1.2],
                    "projected_cvar95": sorted(prof_costs)[int(len(prof_costs) * 0.95)],
                    "actual_cvar95": sorted(gt_a_costs)[int(len(gt_a_costs) * 0.95)],
                },
                "B": {
                    "score": {
                        "workflow_name": wf, "comparison": "B",
                        "mean_error_pct": round(mean_err_b, 1),
                        "p75_error_pct": round(mean_err_b * 1.3, 1),
                        "ci_coverage_pct": round(85 - mean_err_b * 0.4, 1),
                        "monthly_error_pct": round(mean_err_b * 0.95, 1),
                        "cvar95_error_pct": round(mean_err_b * 2.0, 1),
                        "passes": passes_b,
                        "failures": [] if passes_b else [f"Mean error {mean_err_b:.1f}% exceeds 20% target"],
                    },
                    "profiling_run_costs": prof_costs,
                    "ground_truth_run_costs": gt_b_costs,
                    "projected_ci": [min(prof_costs) * 0.7, max(prof_costs) * 1.3],
                    "projected_cvar95": sorted(prof_costs)[int(len(prof_costs) * 0.95)],
                    "actual_cvar95": sorted(gt_b_costs)[int(len(gt_b_costs) * 0.95)],
                },
                "C": {
                    "score": {
                        "workflow_name": wf, "comparison": "C",
                        "mean_error_pct": round(mean_err_c, 1),
                        "p75_error_pct": round(mean_err_c * 1.3, 1),
                        "ci_coverage_pct": round(88 - mean_err_c * 0.3, 1),
                        "monthly_error_pct": round(mean_err_c * 0.9, 1),
                        "cvar95_error_pct": round(mean_err_c * 2.0, 1),
                        "passes": passes_c,
                        "failures": [] if passes_c else [f"Mean error {mean_err_c:.1f}% exceeds 20% target"],
                    },
                    "profiling_run_costs": prof_costs,
                    "ground_truth_run_costs": gt_b_costs,
                    "projected_ci": [min(prof_costs) * 0.75, max(prof_costs) * 1.25],
                    "projected_cvar95": sorted(prof_costs)[int(len(prof_costs) * 0.95)],
                    "actual_cvar95": sorted(gt_b_costs)[int(len(gt_b_costs) * 0.95)],
                },
            },
            "detected_patterns": detected_patterns,
            "step_costs": steps,
            "step_stats": step_stats,
        }
        (RESULTS_DIR / f"{wf}_backtest.json").write_text(json.dumps(data, indent=2))


def main() -> None:
    print("=== Pretia Visualization Demo ===\n")

    # Create output dirs
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Generate mock data
    print("1. Generating mock data...")
    _generate_pre_calibration_report()
    _generate_pilot_data()
    _generate_backtest_data()
    print(f"   Data written to {RESULTS_DIR}/\n")

    # Generate pilot visuals (P1-P6)
    print("2. Generating pilot visualizations (P1-P6)...")
    from visualization.pilot.visualize_pilot import generate_all_pilot_visuals
    pilot_paths = generate_all_pilot_visuals(RESULTS_DIR, DEMO_DIR / "pilot")
    print(f"   {len(pilot_paths)} files generated\n")

    # Generate backtest visuals (B1-B10)
    print("3. Generating backtest visualizations (B1-B10)...")
    from visualization.backtest.visualize_backtest import generate_all_backtest_visuals
    backtest_paths = generate_all_backtest_visuals(RESULTS_DIR, DEMO_DIR / "backtest")
    print(f"   {len(backtest_paths)} files generated\n")

    # Generate narrative report
    print("4. Generating narrative report...")
    from visualization.narrative.generate_narrative import generate_narrative
    narrative_path = generate_narrative(RESULTS_DIR, DEMO_DIR / "narrative.md")
    print(f"   {narrative_path}\n")

    # Generate interactive dashboard
    print("5. Generating interactive HTML dashboard...")
    dashboard_path = DEMO_DIR / "dashboard.html"
    try:
        from visualization.dashboard.generate_dashboard import generate_dashboard
        result = generate_dashboard(
            RESULTS_DIR, dashboard_path,
            pilot_plots_dir=DEMO_DIR / "pilot",
            backtest_plots_dir=DEMO_DIR / "backtest",
        )
        if result:
            print(f"   {result}\n")
        else:
            print("   Skipped (plotly not installed)\n")
            dashboard_path = None
    except Exception as e:
        print(f"   Failed: {e}\n")
        dashboard_path = None

    # Export analytics JSON
    print("6. Exporting analytics JSON...")
    from visualization.export import export_analytics_json
    json_path = export_analytics_json(RESULTS_DIR, DEMO_DIR / "analytics.json")
    print(f"   {json_path}\n")

    # Export analytics PDF
    print("7. Exporting analytics PDF...")
    try:
        from visualization.export import export_analytics_pdf
        pdf_path = export_analytics_pdf(
            RESULTS_DIR,
            DEMO_DIR / "pilot",
            DEMO_DIR / "backtest",
            DEMO_DIR / "analytics.pdf",
        )
        print(f"   {pdf_path}\n")
    except Exception as e:
        print(f"   Failed: {e}\n")
        pdf_path = None

    # Summary
    print("=" * 50)
    print(f"All outputs in: {DEMO_DIR.resolve()}/")
    print()
    print("Files generated:")
    print(f"  Pilot plots:    {DEMO_DIR / 'pilot'}/")
    print(f"  Backtest plots: {DEMO_DIR / 'backtest'}/")
    print(f"  Narrative:      {DEMO_DIR / 'narrative.md'}")
    if dashboard_path and dashboard_path.exists():
        print(f"  Dashboard:      {dashboard_path}")
    print(f"  Analytics JSON: {DEMO_DIR / 'analytics.json'}")
    if pdf_path:
        print(f"  Analytics PDF:  {pdf_path}")
    print()

    # Open dashboard in browser
    if dashboard_path and dashboard_path.exists():
        url = f"file://{dashboard_path.resolve()}"
        print(f"Opening dashboard in browser: {url}")
        webbrowser.open(url)


if __name__ == "__main__":
    main()
