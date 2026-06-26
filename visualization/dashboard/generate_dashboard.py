"""Generate a self-contained interactive HTML dashboard from backtest results."""

from __future__ import annotations

import base64
import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DETECTORS = [
    "context_growth",
    "loop_count_variance",
    "high_token_variance",
    "step_count_variance",
    "bimodality",
]


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load per-workflow JSON files from results directory."""
    results: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return results
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            if "workflow_name" in data:
                results.append(data)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", f, exc)
    return results


def _compute_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute executive summary counts and workflow lists per bucket."""
    passing: list[str] = []
    reweight: list[str] = []
    unresolved: list[str] = []
    for r in results:
        wf = r.get("workflow_name", "?")
        comps = r.get("comparisons", {})
        scores = {
            k: v.get("score") if isinstance(v, dict) else None for k, v in comps.items()
        }
        a_pass = scores.get("A", {}).get("passes", False) if scores.get("A") else False
        b_pass = scores.get("B", {}).get("passes", False) if scores.get("B") else False
        c_pass = scores.get("C", {}).get("passes", False) if scores.get("C") else False
        if a_pass and b_pass:
            passing.append(wf)
        elif a_pass and not b_pass and c_pass:
            reweight.append(wf)
        else:
            unresolved.append(wf)
    return {
        "passing": len(passing),
        "reweight": len(reweight),
        "unresolved": len(unresolved),
        "passing_wfs": passing,
        "reweight_wfs": reweight,
        "unresolved_wfs": unresolved,
    }


def _build_key_findings(
    results: list[dict[str, Any]], summary: dict[str, Any],
) -> str:
    """Build the Key Findings HTML section from backtest results."""
    total = len(results)
    passing = summary["passing"]
    reweight = summary["reweight"]
    unresolved = summary["unresolved"]

    # Determine drift pattern
    b_failed = []
    for r in results:
        comps = r.get("comparisons", {})
        sb = _get_score(comps.get("B"))
        if sb and not sb.get("passes"):
            b_failed.append(r["workflow_name"])

    if not b_failed:
        drift_sentence = "No drift sensitivity detected — projections are robust to input distribution changes."
    elif len(b_failed) >= total * 0.5:
        drift_sentence = (
            f"Widespread drift sensitivity: {len(b_failed)} of {total} workflows degrade under drifted inputs. "
            "This suggests tier weight shift is the dominant drift vector."
        )
    else:
        drift_sentence = (
            f"{len(b_failed)} workflow(s) show drift sensitivity ({', '.join(b_failed)}). "
            "The impact is concentrated, suggesting style or structural drift in specific patterns."
        )

    # Build recommendations
    recs: list[str] = []
    if reweight > 0:
        recs.append(
            f"Use <code>--traffic-mix</code> for {reweight} workflow(s) that recover accuracy with reweighting "
            f"({', '.join(summary['reweight_wfs'])})."
        )
    if unresolved > 0:
        recs.append(
            f"Re-profile {unresolved} workflow(s) with production-representative inputs "
            f"({', '.join(summary['unresolved_wfs'])})."
        )
    if passing == total:
        recs.append("All workflows pass — the projection engine is ready for production use.")
    if not recs:
        recs.append("Review per-workflow results below for detailed accuracy metrics.")

    recs_html = "\n".join(f"  <li>{r}</li>" for r in recs[:4])

    return f"""<div class="findings">
<h2>Key Findings</h2>
<p style="margin:8px 0; font-size:14px; color:#495057;">
  {passing} of {total} workflows pass all comparisons.
  {drift_sentence}
</p>
<ul>
{recs_html}
</ul>
</div>"""


_MODEL_PRICES = {
    "claude-opus-4.7": 15.0, "claude-sonnet-4.6": 3.0, "claude-haiku-4.5": 0.25,
    "gpt-5.4": 5.0, "gpt-5.4-nano": 0.10, "deepseek-v4": 0.14,
    "deepseek-v4-flash": 0.07, "qwen-3.6-plus": 0.50, "qwen-turbo": 0.05,
    "gemini-2.5-flash": 0.15, "text-embedding-3-small": 0.02,
}

_TIER_COLORS = {"fast": "#27ae60", "mid": "#f39c12", "frontier": "#e74c3c"}

# Downshift targets: tier -> (target_tier, candidate_models)
_DOWNSHIFT_TARGETS: dict[str, dict[str, str]] = {
    "claude-opus-4.7": "claude-sonnet-4.6",
    "gpt-5.4": "gpt-5.4-nano",
    "claude-sonnet-4.6": "claude-haiku-4.5",
    "qwen-3.6-plus": "qwen-turbo",
    "deepseek-v4": "deepseek-v4-flash",
}


def _detect_opportunities(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Detect cost-optimization opportunities from step_stats across all workflows.

    Return a list of dicts with keys: workflow, step, rec_type, rec_label,
    issue, metric, action, monthly_savings (at 1K daily volume), confidence.
    """
    opps: list[dict[str, Any]] = []
    daily_volume = 1000

    for r in results:
        wf = r.get("workflow_name", "?")
        step_stats = r.get("step_stats", {})
        detected_patterns = r.get("detected_patterns", [])
        pattern_types = set()
        pattern_steps: dict[str, list[str]] = {}
        for p in detected_patterns:
            pt = p["pattern_type"] if isinstance(p, dict) else p
            pattern_types.add(pt)
            step_name = p.get("step_name", "") if isinstance(p, dict) else ""
            pattern_steps.setdefault(pt, []).append(step_name)

        for step_name, ss in step_stats.items():
            model = ss.get("model", "")
            tier = ss.get("model_tier", "")
            cost_stats = ss.get("cost", {})
            cost_mean = cost_stats.get("mean", 0)
            cost_std = cost_stats.get("std", 0)
            cost_p50 = cost_stats.get("p50", cost_mean)
            cost_p95 = cost_stats.get("p95", cost_mean)
            cost_p10 = cost_stats.get("p10", cost_mean * 0.6)
            cost_p90 = cost_stats.get("p90", cost_mean * 1.4)

            in_stats = ss.get("input_tokens", {})
            out_stats = ss.get("output_tokens", {})
            ctx_stats = ss.get("context_size", {})
            in_mean = in_stats.get("mean", 0)
            out_mean = out_stats.get("mean", 0)
            out_p50 = out_stats.get("p50", out_mean)
            out_p95 = out_stats.get("p95", out_mean)
            ctx_p95 = ctx_stats.get("p95", 0)

            sys_prompt_tok = ss.get("system_prompt_tokens", 0)
            tool_def_tok = ss.get("tool_definitions_tokens", 0)
            mean_iters = ss.get("mean_iterations", 1.0)
            iter_max = ss.get("iterations_max", 1)
            output_format = ss.get("output_format", "text")
            retry_pct = ss.get("is_retry_pct", 0.0)

            cost_cv = cost_std / cost_mean if cost_mean > 0 else 0
            out_in_ratio = out_mean / in_mean if in_mean > 0 else 0
            monthly = daily_volume * 30

            # 1. Model downshift
            if tier == "frontier" and model in _DOWNSHIFT_TARGETS:
                if out_in_ratio < 0.3 or output_format == "json":
                    target = _DOWNSHIFT_TARGETS[model]
                    current_price = _MODEL_PRICES.get(model, 1.0)
                    target_price = _MODEL_PRICES.get(target, current_price)
                    savings = (1 - target_price / current_price) * cost_mean * monthly
                    opps.append({
                        "workflow": wf, "step": step_name,
                        "rec_type": "model_downshift", "rec_label": "Model Downshift",
                        "issue": f"Frontier model ({model}) used for "
                                 f"{'structured output' if output_format == 'json' else 'low-complexity task'}",
                        "metric": f"out/in={out_in_ratio:.2f}, fmt={output_format}",
                        "action": f"Switch to {target}",
                        "monthly_savings": round(savings, 2),
                        "confidence": "HIGH",
                    })
            elif tier == "mid" and model in _DOWNSHIFT_TARGETS:
                if out_in_ratio < 0.3 or output_format == "json":
                    target = _DOWNSHIFT_TARGETS[model]
                    current_price = _MODEL_PRICES.get(model, 1.0)
                    target_price = _MODEL_PRICES.get(target, current_price)
                    savings = (1 - target_price / current_price) * cost_mean * monthly
                    if savings > 0:
                        opps.append({
                            "workflow": wf, "step": step_name,
                            "rec_type": "model_downshift", "rec_label": "Model Downshift",
                            "issue": f"Mid-tier model ({model}) used for "
                                     f"{'structured output' if output_format == 'json' else 'low-complexity task'}",
                            "metric": f"out/in={out_in_ratio:.2f}, fmt={output_format}",
                            "action": f"Switch to {target}",
                            "monthly_savings": round(savings, 2),
                            "confidence": "MODERATE",
                        })

            # 2. Context compaction
            if "context_growth" in pattern_types and mean_iters > 2:
                ctx_steps = pattern_steps.get("context_growth", [])
                if step_name in ctx_steps or "generate" in ctx_steps:
                    ctx_growth_savings = cost_mean * 0.3 * monthly
                    opps.append({
                        "workflow": wf, "step": step_name,
                        "rec_type": "context_compaction", "rec_label": "Context Compaction",
                        "issue": "Context grows linearly across iterations",
                        "metric": f"mean_iters={mean_iters:.1f}",
                        "action": "Insert compaction/summarization between iterations",
                        "monthly_savings": round(ctx_growth_savings, 2),
                        "confidence": "MODERATE",
                    })

            # 3. Iteration cap
            if "loop_count_variance" in pattern_types and iter_max > mean_iters * 1.5:
                suggested_cap = int(mean_iters * 1.5)
                cost_per_iter = cost_mean / mean_iters if mean_iters > 0 else cost_mean
                excess_iters = mean_iters * 0.2  # estimated savings from capping tail
                savings = excess_iters * cost_per_iter * monthly
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "iteration_cap", "rec_label": "Iteration Cap",
                    "issue": f"Loop iterations vary widely (max={iter_max}, mean={mean_iters:.1f})",
                    "metric": f"max/mean={iter_max / mean_iters:.1f}x",
                    "action": f"Set iteration cap to {suggested_cap}",
                    "monthly_savings": round(savings, 2),
                    "confidence": "MODERATE",
                })

            # 4. Caching opportunity
            if in_mean > 0 and sys_prompt_tok / in_mean > 0.4:
                sys_pct = sys_prompt_tok / in_mean
                cache_discount = 0.9
                input_cost_frac = cost_mean * (in_mean / (in_mean + out_mean)) if (in_mean + out_mean) > 0 else cost_mean * 0.5
                savings = sys_pct * input_cost_frac * cache_discount * monthly
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "caching", "rec_label": "Prompt Caching",
                    "issue": f"System prompt is {sys_pct:.0%} of input tokens",
                    "metric": f"sys_prompt={sys_prompt_tok}, input_mean={in_mean}",
                    "action": "Enable prompt caching for system prompt",
                    "monthly_savings": round(savings, 2),
                    "confidence": "HIGH",
                })

            # 5. Prompt optimization
            if sys_prompt_tok > 500 or (in_mean > 0 and tool_def_tok / in_mean > 0.3):
                trim_pct = 0.3
                input_cost_frac = cost_mean * (in_mean / (in_mean + out_mean)) if (in_mean + out_mean) > 0 else cost_mean * 0.5
                overhead = (sys_prompt_tok + tool_def_tok) / in_mean if in_mean > 0 else 0
                savings = overhead * trim_pct * input_cost_frac * monthly
                issue_parts = []
                if sys_prompt_tok > 500:
                    issue_parts.append(f"system prompt={sys_prompt_tok} tokens")
                if in_mean > 0 and tool_def_tok / in_mean > 0.3:
                    issue_parts.append(f"tool defs={tool_def_tok} tokens ({tool_def_tok / in_mean:.0%} of input)")
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "prompt_optimization", "rec_label": "Prompt Optimization",
                    "issue": "Prompt overhead: " + ", ".join(issue_parts),
                    "metric": f"overhead={overhead:.0%} of input",
                    "action": "Trim system prompt and tool definitions (~30% reduction target)",
                    "monthly_savings": round(savings, 2),
                    "confidence": "LOW",
                })

            # 6. Architecture restructuring
            if ctx_p95 > 3 * out_mean and out_mean > 0:
                savings = cost_mean * 0.15 * monthly
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "architecture_restructuring",
                    "rec_label": "Architecture Restructuring",
                    "issue": f"Context window (p95={ctx_p95}) is {ctx_p95 / out_mean:.0f}x output",
                    "metric": f"ctx_p95={ctx_p95}, out_mean={out_mean}",
                    "action": "Reduce redundant context; consider retrieval or summarization",
                    "monthly_savings": round(savings, 2),
                    "confidence": "LOW",
                })

            # 7. High variance step
            if cost_cv > 0.8 or (cost_p95 > 3 * cost_p50 and cost_p50 > 0):
                ratio = cost_p95 / cost_p50 if cost_p50 > 0 else 0
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "high_variance_step", "rec_label": "High Variance Step",
                    "issue": f"Cost variance is very high (CV={cost_cv:.2f}, p95/p50={ratio:.1f}x)",
                    "metric": f"CV={cost_cv:.2f}, range=${cost_p10:.4f}-${cost_p90:.4f}",
                    "action": "Investigate root cause of cost variance",
                    "monthly_savings": round((cost_p95 - cost_p50) * monthly * 0.1, 2),
                    "confidence": "LOW",
                })

            # 8. Token waste
            if out_p95 > 3 * out_p50 and out_p50 > 0:
                ratio = out_p95 / out_p50
                waste_frac = (out_p95 - out_p50) / out_p95 if out_p95 > 0 else 0
                output_cost_frac = cost_mean * (out_mean / (in_mean + out_mean)) if (in_mean + out_mean) > 0 else cost_mean * 0.5
                savings = waste_frac * 0.5 * output_cost_frac * monthly
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "token_waste", "rec_label": "Token Waste",
                    "issue": f"Output tokens p95 ({out_p95}) is {ratio:.1f}x p50 ({out_p50})",
                    "metric": f"p95={out_p95}, p50={out_p50}",
                    "action": "Add max_tokens constraint or structured output format",
                    "monthly_savings": round(savings, 2),
                    "confidence": "MODERATE",
                })

            # 9. Tool schema filtering
            if in_mean > 0 and tool_def_tok / in_mean > 0.3:
                tool_pct = tool_def_tok / in_mean
                input_price = _MODEL_PRICES.get(model, 1.0) / 1_000_000
                savings = tool_def_tok * 0.5 * input_price * monthly * mean_iters
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "tool_filtering", "rec_label": "Tool Schema Filtering",
                    "issue": f"Tool definitions are {tool_pct:.0%} of input ({tool_def_tok} tokens)",
                    "metric": f"tool_def={tool_def_tok}, input={in_mean}",
                    "action": "Expose only relevant tools to this step",
                    "monthly_savings": round(savings, 2),
                    "confidence": "HIGH",
                })

            # 10. Output truncation
            trunc_pct = ss.get("output_truncated_pct", 0.0)
            if trunc_pct > 10:
                savings = (trunc_pct / 100) * cost_mean * monthly
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "output_truncation", "rec_label": "Output Truncation",
                    "issue": f"{trunc_pct:.0f}% of outputs truncated (hitting max_tokens)",
                    "metric": f"truncation_rate={trunc_pct:.0f}%",
                    "action": "Increase max_tokens or split task into smaller subtasks",
                    "monthly_savings": round(savings, 2),
                    "confidence": "HIGH",
                })

            # 11. Retry cascade
            tool_retry = ss.get("tool_retry_count_mean", 0.0)
            if retry_pct > 10 or tool_retry > 1:
                retry_cost = (retry_pct / 100) * cost_mean * monthly
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "retry_cascade", "rec_label": "Retry Cascade",
                    "issue": f"Retry rate {retry_pct:.0f}%, tool retries avg {tool_retry:.1f}",
                    "metric": f"retry_pct={retry_pct:.0f}%, tool_retries={tool_retry:.1f}",
                    "action": "Fix tool reliability or add fallback logic",
                    "monthly_savings": round(retry_cost, 2),
                    "confidence": "HIGH",
                })

            # 12. Temperature on deterministic tasks
            temp = ss.get("temperature")
            if temp is not None and temp > 0:
                if output_format == "json" or out_in_ratio < 0.2:
                    sys_pct = sys_prompt_tok / in_mean if in_mean > 0 else 0
                    input_price = _MODEL_PRICES.get(model, 1.0) / 1_000_000
                    cache_savings = sys_pct * 0.9 * in_mean * input_price * monthly
                    opps.append({
                        "workflow": wf, "step": step_name,
                        "rec_type": "temperature_optimization",
                        "rec_label": "Temperature Optimization",
                        "issue": f"temperature={temp} on deterministic task ({output_format})",
                        "metric": f"temp={temp}, fmt={output_format}, out/in={out_in_ratio:.2f}",
                        "action": "Set temperature=0 to enable provider caching",
                        "monthly_savings": round(cache_savings, 2),
                        "confidence": "MODERATE",
                    })

            # 13. Tool output bloat
            tool_out = ss.get("tool_output_tokens", 0)
            tool_in = ss.get("tool_input_tokens", 0)
            if tool_out > 0 and tool_in > 0 and tool_out > 2 * tool_in:
                excess = tool_out - tool_in
                input_price = _MODEL_PRICES.get(model, 1.0) / 1_000_000
                savings = excess * input_price * monthly * mean_iters
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "tool_output_bloat", "rec_label": "Tool Output Bloat",
                    "issue": f"Tool returns {tool_out} tokens for {tool_in} input ({tool_out / tool_in:.1f}x)",
                    "metric": f"tool_out={tool_out}, tool_in={tool_in}",
                    "action": "Limit retrieval chunk count or filter tool output",
                    "monthly_savings": round(savings, 2),
                    "confidence": "MODERATE",
                })

        # Cross-step detectors (compare steps within this workflow)
        step_items = list(step_stats.items())

        # 14. Redundant context across steps
        for i, (name_a, ss_a) in enumerate(step_items):
            ctx_a = ss_a.get("context_size", {}).get("mean", 0)
            if ctx_a < 500:
                continue
            for name_b, ss_b in step_items[i + 1:]:
                ctx_b = ss_b.get("context_size", {}).get("mean", 0)
                if ctx_b < 500:
                    continue
                ratio = min(ctx_a, ctx_b) / max(ctx_a, ctx_b) if max(ctx_a, ctx_b) > 0 else 0
                if ratio > 0.85:
                    price_a = _MODEL_PRICES.get(ss_a.get("model", ""), 1.0) / 1_000_000
                    savings = min(ctx_a, ctx_b) * price_a * daily_volume * 30
                    opps.append({
                        "workflow": wf, "step": f"{name_a} + {name_b}",
                        "rec_type": "redundant_context",
                        "rec_label": "Redundant Context",
                        "issue": f"Steps share similar context ({ctx_a:.0f} vs {ctx_b:.0f} tokens)",
                        "metric": f"ctx_overlap={ratio:.0%}",
                        "action": "Extract shared context, pass summary between steps",
                        "monthly_savings": round(savings, 2),
                        "confidence": "LOW",
                    })

        # 15. Batching opportunity
        for step_name, ss in step_stats.items():
            call_count = ss.get("call_count", 0)
            runs_present = ss.get("runs_present", 1)
            mean_iters_check = ss.get("mean_iterations", 1.0)
            if runs_present > 0 and call_count > runs_present * 2 and mean_iters_check <= 1.5:
                calls_per_run = call_count / runs_present
                opps.append({
                    "workflow": wf, "step": step_name,
                    "rec_type": "batching", "rec_label": "Batching Opportunity",
                    "issue": f"Step called {calls_per_run:.1f}x per run (fan-out pattern)",
                    "metric": f"calls={call_count}, runs={runs_present}",
                    "action": "Batch API calls to reduce per-request overhead",
                    "monthly_savings": 0.0,
                    "confidence": "LOW",
                })

    # Sort by monthly savings descending
    opps.sort(key=lambda x: x["monthly_savings"], reverse=True)
    return opps


def _build_analytics_section(results: list[dict[str, Any]]) -> str:
    """Build the Analytics & Recommendations HTML section with 5 subsections."""
    import plotly.graph_objects as go

    fragments: list[str] = []

    # ---- A. Variance & Risk Table ----
    var_rows: list[dict[str, Any]] = []
    for r in results:
        wf = r.get("workflow_name", "?")
        step_stats = r.get("step_stats", {})
        if not step_stats:
            continue

        # Compute workflow-level cost stats by summing step costs
        total_mean = sum(ss.get("cost", {}).get("mean", 0) for ss in step_stats.values())
        total_std = (sum(ss.get("cost", {}).get("std", 0) ** 2 for ss in step_stats.values())) ** 0.5
        total_p50 = sum(ss.get("cost", {}).get("p50", 0) for ss in step_stats.values())
        total_p95 = sum(ss.get("cost", {}).get("p95", 0) for ss in step_stats.values())
        total_p10 = sum(ss.get("cost", {}).get("p10", 0) for ss in step_stats.values())
        total_p90 = sum(ss.get("cost", {}).get("p90", 0) for ss in step_stats.values())
        cost_cv = total_std / total_mean if total_mean > 0 else 0
        p95_p50 = total_p95 / total_p50 if total_p50 > 0 else 0

        # Find highest variance step
        max_cv_step = ""
        max_cv = 0
        for sname, ss in step_stats.items():
            sc = ss.get("cost", {})
            s_mean = sc.get("mean", 0)
            s_std = sc.get("std", 0)
            scv = s_std / s_mean if s_mean > 0 else 0
            if scv > max_cv:
                max_cv = scv
                max_cv_step = sname

        risk = "Low" if cost_cv < 0.3 else ("Medium" if cost_cv < 0.8 else "High")
        var_rows.append({
            "wf": wf, "cv": cost_cv, "p95_p50": p95_p50,
            "range": f"${total_p10:.4f} - ${total_p90:.4f}",
            "max_cv_step": f"{max_cv_step} (CV={max_cv:.2f})",
            "risk": risk,
        })

    var_rows.sort(key=lambda x: x["cv"], reverse=True)

    def _cv_color(cv: float) -> str:
        if cv < 0.3:
            return "#d4edda"
        if cv < 0.8:
            return "#fff3cd"
        return "#f8d7da"

    risk_colors = {"Low": "#d4edda", "Medium": "#fff3cd", "High": "#f8d7da"}

    if var_rows:
        var_fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Workflow", "Cost CV", "P95/P50", "Cost Range (p10-p90)",
                         "Highest Variance Step", "Risk Level"],
                fill_color="#343a40", font=dict(color="white", size=12), align="left",
            ),
            cells=dict(
                values=[
                    [r["wf"] for r in var_rows],
                    [f"{r['cv']:.2f}" for r in var_rows],
                    [f"{r['p95_p50']:.2f}x" for r in var_rows],
                    [r["range"] for r in var_rows],
                    [r["max_cv_step"] for r in var_rows],
                    [r["risk"] for r in var_rows],
                ],
                fill_color=[
                    ["white"] * len(var_rows),
                    [_cv_color(r["cv"]) for r in var_rows],
                    ["white"] * len(var_rows),
                    ["white"] * len(var_rows),
                    ["white"] * len(var_rows),
                    [risk_colors.get(r["risk"], "white") for r in var_rows],
                ],
                align="left", font=dict(size=11),
            ),
        )])
        var_fig.update_layout(title="Variance & Risk Overview", margin=dict(t=40, b=10, l=10, r=10))
        fragments.append(
            '<h3 style="margin:16px 0 4px; color:#343a40;">A. Variance & Risk</h3>'
            '<p class="section-desc">Workflow-level cost variability. '
            'Green = low (CV&lt;0.3), yellow = medium (0.3-0.8), red = high (&gt;0.8).</p>'
            + var_fig.to_html(full_html=False, include_plotlyjs=False)
        )

    # ---- B. Cost Attribution Treemap ----
    treemap_labels: list[str] = []
    treemap_parents: list[str] = []
    treemap_values: list[float] = []
    treemap_colors: list[str] = []
    treemap_hover: list[str] = []

    for r in results:
        wf = r.get("workflow_name", "?")
        step_stats = r.get("step_stats", {})
        if not step_stats:
            continue
        wf_total = sum(ss.get("cost", {}).get("mean", 0) for ss in step_stats.values())

        # Add workflow root (parent="")
        treemap_labels.append(wf)
        treemap_parents.append("")
        treemap_values.append(0)  # branch node
        treemap_colors.append("#dee2e6")
        treemap_hover.append(f"{wf}: ${wf_total:.4f} total")

        for sname, ss in step_stats.items():
            cost_mean = ss.get("cost", {}).get("mean", 0)
            model = ss.get("model", "?")
            tier = ss.get("model_tier", "mid")
            scv = ss.get("cost", {}).get("std", 0) / cost_mean if cost_mean > 0 else 0
            in_mean = ss.get("input_tokens", {}).get("mean", 0)
            out_mean = ss.get("output_tokens", {}).get("mean", 0)
            out_in = out_mean / in_mean if in_mean > 0 else 0
            pct = cost_mean / wf_total * 100 if wf_total > 0 else 0

            treemap_labels.append(f"{wf}/{sname}")
            treemap_parents.append(wf)
            treemap_values.append(cost_mean)
            treemap_colors.append(_TIER_COLORS.get(tier, "#888"))
            treemap_hover.append(
                f"<b>{sname}</b><br>Model: {model} ({tier})<br>"
                f"Cost: ${cost_mean:.5f} ({pct:.0f}% of {wf})<br>"
                f"CV: {scv:.2f}<br>Out/In: {out_in:.2f}"
            )

    if treemap_labels:
        tree_fig = go.Figure(go.Treemap(
            labels=treemap_labels,
            parents=treemap_parents,
            values=treemap_values,
            marker=dict(colors=treemap_colors),
            hovertext=treemap_hover,
            hoverinfo="text",
            branchvalues="total",
            textinfo="label+value",
            texttemplate="%{label}<br>$%{value:.5f}",
        ))
        tree_fig.update_layout(
            title="Cost Attribution by Workflow and Step",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        fragments.append(
            '<h3 style="margin:24px 0 4px; color:#343a40;">B. Cost Attribution Treemap</h3>'
            '<p class="section-desc">Step costs grouped by workflow. '
            'Colors: <span style="color:#27ae60">green=fast</span>, '
            '<span style="color:#f39c12">yellow=mid</span>, '
            '<span style="color:#e74c3c">red=frontier</span>.</p>'
            + tree_fig.to_html(full_html=False, include_plotlyjs=False)
        )

    # ---- C. Pain Points Table ----
    pain_rows: list[dict[str, str]] = []
    for r in results:
        wf = r.get("workflow_name", "?")
        step_stats = r.get("step_stats", {})
        for sname, ss in step_stats.items():
            cost_stats = ss.get("cost", {})
            cost_mean = cost_stats.get("mean", 0)
            cost_p50 = cost_stats.get("p50", cost_mean)
            cost_p95 = cost_stats.get("p95", cost_mean)
            in_mean = ss.get("input_tokens", {}).get("mean", 0)
            sys_prompt_tok = ss.get("system_prompt_tokens", 0)
            mean_iters = ss.get("mean_iterations", 1.0)
            retry_pct = ss.get("is_retry_pct", 0.0)

            # Context overhead
            if in_mean > 0 and sys_prompt_tok / in_mean > 0.5:
                pain_rows.append({
                    "wf": wf, "step": sname, "pain": "Context Overhead",
                    "metric": f"sys_prompt/input = {sys_prompt_tok / in_mean:.0%}",
                    "impact": "Medium" if sys_prompt_tok / in_mean < 0.7 else "High",
                })

            # Loop amplifier
            cost_per_iter = cost_mean / mean_iters if mean_iters > 0 else cost_mean
            if mean_iters > 2 and cost_per_iter > 0.005:
                pain_rows.append({
                    "wf": wf, "step": sname, "pain": "Loop Amplifier",
                    "metric": f"mean_iters={mean_iters:.1f}, cost/iter=${cost_per_iter:.4f}",
                    "impact": "High" if mean_iters > 5 else "Medium",
                })

            # Retry cost
            if retry_pct > 5:
                pain_rows.append({
                    "wf": wf, "step": sname, "pain": "Retry Cost",
                    "metric": f"retry_pct={retry_pct:.1f}%",
                    "impact": "Medium" if retry_pct < 10 else "High",
                })

            # Unpredictable
            if cost_p50 > 0 and cost_p95 > 3 * cost_p50:
                pain_rows.append({
                    "wf": wf, "step": sname, "pain": "Unpredictable",
                    "metric": f"p95/p50 = {cost_p95 / cost_p50:.1f}x",
                    "impact": "High",
                })

    if pain_rows:
        impact_colors = {"Low": "#d4edda", "Medium": "#fff3cd", "High": "#f8d7da"}
        pain_fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Workflow", "Step", "Pain Point", "Metric", "Impact Level"],
                fill_color="#343a40", font=dict(color="white", size=12), align="left",
            ),
            cells=dict(
                values=[
                    [r["wf"] for r in pain_rows],
                    [r["step"] for r in pain_rows],
                    [r["pain"] for r in pain_rows],
                    [r["metric"] for r in pain_rows],
                    [r["impact"] for r in pain_rows],
                ],
                fill_color=[
                    ["white"] * len(pain_rows),
                    ["white"] * len(pain_rows),
                    ["white"] * len(pain_rows),
                    ["white"] * len(pain_rows),
                    [impact_colors.get(r["impact"], "white") for r in pain_rows],
                ],
                align="left", font=dict(size=11),
            ),
        )])
        pain_fig.update_layout(title="Pain Points", margin=dict(t=40, b=10, l=10, r=10))
        fragments.append(
            '<h3 style="margin:24px 0 4px; color:#343a40;">C. Pain Points</h3>'
            '<p class="section-desc">Steps with structural cost issues that amplify spend at scale.</p>'
            + pain_fig.to_html(full_html=False, include_plotlyjs=False)
        )

    # ---- D. Optimization Opportunities Table ----
    opps = _detect_opportunities(results)
    if opps:
        rec_type_colors = {
            "model_downshift": "#d5f5e3", "context_compaction": "#d6eaf8",
            "iteration_cap": "#fdebd0", "caching": "#d5f5e3",
            "prompt_optimization": "#fef9e7", "architecture_restructuring": "#f5eef8",
            "high_variance_step": "#fadbd8", "token_waste": "#fef9e7",
        }
        total_savings = sum(o["monthly_savings"] for o in opps)

        # Add total row
        display_opps = opps + [{
            "rec_type": "", "rec_label": "TOTAL", "workflow": "", "step": "",
            "issue": "", "metric": "", "action": "",
            "monthly_savings": total_savings, "confidence": "",
        }]

        opp_fig = go.Figure(data=[go.Table(
            header=dict(
                values=["Rec Type", "Workflow", "Step", "Issue", "Key Metric",
                         "Action", "Est. Savings/mo", "Confidence"],
                fill_color="#343a40", font=dict(color="white", size=12), align="left",
            ),
            cells=dict(
                values=[
                    [o["rec_label"] for o in display_opps],
                    [o["workflow"] for o in display_opps],
                    [o["step"] for o in display_opps],
                    [o["issue"] for o in display_opps],
                    [o["metric"] for o in display_opps],
                    [o["action"] for o in display_opps],
                    [f"${o['monthly_savings']:.2f}" for o in display_opps],
                    [o["confidence"] for o in display_opps],
                ],
                fill_color=[
                    [rec_type_colors.get(o["rec_type"], "#f8f9fa") for o in display_opps],
                    ["white"] * len(display_opps),
                    ["white"] * len(display_opps),
                    ["white"] * len(display_opps),
                    ["white"] * len(display_opps),
                    ["white"] * len(display_opps),
                    ["white"] * len(display_opps),
                    ["white"] * len(display_opps),
                ],
                align="left", font=dict(size=11),
                font_color=[
                    ["black"] * len(display_opps),
                    ["black"] * len(display_opps),
                    ["black"] * len(display_opps),
                    ["black"] * len(display_opps),
                    ["black"] * len(display_opps),
                    ["black"] * len(display_opps),
                    ["black"] * (len(display_opps) - 1) + ["#e74c3c"],
                    ["black"] * len(display_opps),
                ],
            ),
        )])
        opp_fig.update_layout(
            title="Optimization Opportunities (sorted by savings)",
            margin=dict(t=40, b=10, l=10, r=10),
        )
        fragments.append(
            '<h3 style="margin:24px 0 4px; color:#343a40;">D. Optimization Opportunities</h3>'
            '<p class="section-desc">Actionable recommendations sorted by estimated monthly savings '
            f'at 1K daily volume. Total potential savings: <b>${total_savings:.2f}/mo</b>.</p>'
            + opp_fig.to_html(full_html=False, include_plotlyjs=False)
        )

    # ---- E. Per-Step Deep Dive (grouped bars with dropdown) ----
    # Build traces for each workflow, only first visible by default
    dd_data: list[Any] = []
    wf_names_dd: list[str] = []
    traces_per_wf = 4  # input_tok, output_tok, sys_prompt_tok, tool_def_tok

    for r in results:
        wf = r.get("workflow_name", "?")
        step_stats = r.get("step_stats", {})
        if not step_stats:
            continue
        wf_names_dd.append(wf)

    for idx, r in enumerate(results):
        wf = r.get("workflow_name", "?")
        step_stats = r.get("step_stats", {})
        if not step_stats:
            continue

        steps = list(step_stats.keys())
        visible = idx == 0

        in_means = [step_stats[s].get("input_tokens", {}).get("mean", 0) for s in steps]
        out_means = [step_stats[s].get("output_tokens", {}).get("mean", 0) for s in steps]
        sys_toks = [step_stats[s].get("system_prompt_tokens", 0) for s in steps]
        tool_toks = [step_stats[s].get("tool_definitions_tokens", 0) for s in steps]

        # Error bars: range from p50 to p95 for input and output
        in_p50 = [step_stats[s].get("input_tokens", {}).get("p50", 0) for s in steps]
        in_p95 = [step_stats[s].get("input_tokens", {}).get("p95", 0) for s in steps]
        out_p50 = [step_stats[s].get("output_tokens", {}).get("p50", 0) for s in steps]
        out_p95 = [step_stats[s].get("output_tokens", {}).get("p95", 0) for s in steps]

        dd_data.append(go.Bar(
            name="Input Tokens", x=steps, y=in_means,
            marker_color="#3498db", visible=visible,
            error_y=dict(type="data", symmetric=False,
                         array=[p95 - m for p95, m in zip(in_p95, in_means)],
                         arrayminus=[m - p50 for m, p50 in zip(in_means, in_p50)]),
            showlegend=visible,
        ))
        dd_data.append(go.Bar(
            name="Output Tokens", x=steps, y=out_means,
            marker_color="#e74c3c", visible=visible,
            error_y=dict(type="data", symmetric=False,
                         array=[p95 - m for p95, m in zip(out_p95, out_means)],
                         arrayminus=[m - p50 for m, p50 in zip(out_means, out_p50)]),
            showlegend=visible,
        ))
        dd_data.append(go.Bar(
            name="System Prompt", x=steps, y=sys_toks,
            marker_color="#f39c12", visible=visible,
            showlegend=visible,
        ))
        dd_data.append(go.Bar(
            name="Tool Definitions", x=steps, y=tool_toks,
            marker_color="#9b59b6", visible=visible,
            showlegend=visible,
        ))

    # Build dropdown buttons
    dd_buttons = []
    total_dd_traces = len(dd_data)
    for i, wf in enumerate(wf_names_dd):
        visibility = [False] * total_dd_traces
        base = i * traces_per_wf
        for j in range(traces_per_wf):
            if base + j < total_dd_traces:
                visibility[base + j] = True
        dd_buttons.append(dict(
            label=wf, method="update",
            args=[{"visible": visibility}],
        ))

    if dd_data:
        dd_fig = go.Figure(data=dd_data)
        dd_fig.update_layout(
            title="Per-Step Token Breakdown",
            barmode="group",
            updatemenus=[dict(
                buttons=dd_buttons[:30],
                direction="down", showactive=True,
                x=0.0, xanchor="left", y=1.15, yanchor="top",
            )],
            margin=dict(t=80, b=40, l=60, r=20),
            yaxis_title="Tokens",
            xaxis_title="Step",
        )
        fragments.append(
            '<h3 style="margin:24px 0 4px; color:#343a40;">E. Per-Step Deep Dive</h3>'
            '<p class="section-desc">Token breakdown per step with p50-p95 error bars. '
            'Select workflow from dropdown.</p>'
            + dd_fig.to_html(full_html=False, include_plotlyjs=False)
        )

    if not fragments:
        return '<p style="color:#6c757d;">No step_stats data available for analytics.</p>'

    return "\n".join(fragments)


_PLOT_TITLES = {
    "p1_infrastructure_check_matrix": "P1: Infrastructure Check Matrix",
    "p2_cost_distribution": "P2: Per-Workflow Cost Distribution",
    "p3_routing_sanity": "P3: Routing Sanity Check",
    "p4_loop_iterations": "P4: Loop Iteration Distribution",
    "p5_context_growth": "P5: Context Growth Verification",
    "p6_cost_plausibility": "P6: Cost Plausibility Scatter",
    "b1_projected_vs_actual_kde": "B1: Projected vs Actual KDE",
    "b2_accuracy_heatmap": "B2: Accuracy Metrics Heatmap",
    "b3_drift_impact_bars": "B3: Drift Impact",
    "b4_reweighting_recovery_scatter": "B4: Reweighting Recovery",
    "b5_detector_activation_matrix": "B5: Detector Activation Matrix",
    "b6_gmm_overlay": "B6: Bimodality GMM Overlay",
    "b7_ci_coverage_bands": "B7: Confidence Interval Coverage",
    "b8_cvar_comparison": "B8: CVaR Comparison",
    "b9_per_step_cost_breakdown": "B9: Per-Step Cost Breakdown",
    "b10_token_distribution": "B10: Token Distribution",
}


def _embed_plots(plot_dir: Path) -> str:
    """Embed all PNGs from a directory as base64 images in HTML."""
    if not plot_dir.is_dir():
        return ""
    pngs = sorted(plot_dir.glob("*.png"))
    if not pngs:
        return "<p style='color:#6c757d;'>No plots found.</p>"
    fragments: list[str] = []
    for png in pngs:
        b64 = base64.b64encode(png.read_bytes()).decode("ascii")
        stem = png.stem
        # Match against known titles, strip comparison suffix for lookup
        base = stem.rstrip("_ABC").rsplit("_", 1)[0] if stem[-1] in "ABC" and "_" in stem else stem
        title = _PLOT_TITLES.get(base, _PLOT_TITLES.get(stem, stem.replace("_", " ").title()))
        if stem != base:
            suffix = stem[len(base):].lstrip("_")
            if suffix in ("A", "B", "C"):
                title += f" (Comparison {suffix})"
        fragments.append(
            f'<div style="margin-bottom:24px;">'
            f'<h3 style="margin:0 0 8px; font-size:15px; color:#495057;">{title}</h3>'
            f'<img src="data:image/png;base64,{b64}" '
            f'style="max-width:100%; border:1px solid #dee2e6; border-radius:6px;" '
            f'alt="{title}">'
            f'</div>'
        )
    return "\n".join(fragments)


def _get_score(comp_data: dict | None) -> dict[str, Any] | None:
    """Extract score dict from comparison data, handling None and missing keys."""
    if comp_data is None:
        return None
    if isinstance(comp_data, dict):
        return comp_data.get("score")
    return None


def generate_dashboard(
    results_dir: Path,
    output: Path,
    pilot_plots_dir: Path | None = None,
    backtest_plots_dir: Path | None = None,
) -> Path | None:
    """Generate a self-contained HTML dashboard. Returns output path or None if plotly missing."""
    try:
        import plotly.graph_objects as go  # noqa: F401
    except ImportError:
        logger.warning(
            "plotly is not installed. Install with: pip install pretia[visualization]"
        )
        print(  # noqa: T201
            "Warning: plotly is not installed. Dashboard generation skipped. "
            "Install with: pip install pretia[visualization]"
        )
        return None

    from plotly.subplots import make_subplots

    results = _load_results(results_dir)
    if not results:
        logger.warning("No results found in %s", results_dir)
        return None

    summary = _compute_summary(results)

    # Import color helpers
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from visualization.colors import (
        COMPARISON_COLORS,
        DETECTOR_MATRIX_COLORS,
        EXPECTED_DETECTORS,
        WORKFLOW_GROUPS,
        classify_detector_result,
        workflow_color,
    )

    # ---- Build individual figures ----

    # D2: Results Table
    wf_names = []
    pattern_types = []
    a_results = []
    b_results = []
    c_results = []
    failure_buckets = []
    mean_errors = []
    recovery_pcts = []

    for r in results:
        wf = r["workflow_name"]
        wf_names.append(wf)
        pattern_types.append(WORKFLOW_GROUPS.get(wf, "unknown"))
        comps = r.get("comparisons", {})
        sa = _get_score(comps.get("A"))
        sb = _get_score(comps.get("B"))
        sc = _get_score(comps.get("C"))
        a_results.append("PASS" if sa and sa.get("passes") else "FAIL")
        b_results.append("PASS" if sb and sb.get("passes") else ("FAIL" if sb else "N/A"))
        c_results.append("PASS" if sc and sc.get("passes") else ("FAIL" if sc else "N/A"))

        # Compute failure bucket via attribution logic
        if sa and sa.get("passes") and sb and sb.get("passes"):
            failure_buckets.append("-")
        elif not sa or not sa.get("passes"):
            failure_buckets.append("1: Engine")
        elif sc and sc.get("passes"):
            failure_buckets.append("2: Drift")
        else:
            failure_buckets.append("3: Structural")

        mean_errors.append(f"{sa.get('mean_error_pct', 0):.1f}%" if sa else "N/A")

        # Recovery: (B_err - C_err) / (B_err - A_err) * 100
        if sa and sb and sc:
            a_err = sa.get("mean_error_pct", 0)
            b_err = sb.get("mean_error_pct", 0)
            c_err = sc.get("mean_error_pct", 0)
            drift = b_err - a_err
            if drift > 0:
                rec = (b_err - c_err) / drift * 100
                recovery_pcts.append(f"{rec:.0f}%")
            else:
                recovery_pcts.append("100%")
        else:
            recovery_pcts.append("N/A")

    # Color cells based on pass/fail
    def _pf_colors(vals: list[str]) -> list[str]:
        return [
            "#d4edda" if v == "PASS" else ("#f8d7da" if v == "FAIL" else "#e2e3e5")
            for v in vals
        ]

    table_fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=[
                        "Workflow",
                        "Pattern",
                        "Comp A",
                        "Comp B",
                        "Comp C",
                        "Bucket",
                        "Mean Err%",
                        "Recovery%",
                    ],
                    fill_color="#343a40",
                    font=dict(color="white", size=12),
                    align="left",
                ),
                cells=dict(
                    values=[
                        wf_names,
                        pattern_types,
                        a_results,
                        b_results,
                        c_results,
                        failure_buckets,
                        mean_errors,
                        recovery_pcts,
                    ],
                    fill_color=[
                        ["white"] * len(wf_names),
                        ["white"] * len(wf_names),
                        _pf_colors(a_results),
                        _pf_colors(b_results),
                        _pf_colors(c_results),
                        [
                            "#f8d7da" if b.startswith("1") else
                            "#fff3cd" if b.startswith("2") else
                            "#fde2e2" if b.startswith("3") else "#e2e3e5"
                            for b in failure_buckets
                        ],
                        ["white"] * len(wf_names),
                        ["white"] * len(wf_names),
                    ],
                    align="left",
                    font=dict(size=11),
                ),
            )
        ]
    )
    table_fig.update_layout(title="D2: Results Table", margin=dict(t=40, b=10, l=10, r=10))

    # D3: Projected vs Actual Explorer (first workflow as default)
    d3_buttons = []
    d3_data = []
    for i, r in enumerate(results):
        wf = r["workflow_name"]
        step_costs = r.get("step_costs", {})
        comps = r.get("comparisons", {})
        steps = sorted(step_costs.keys())
        if not steps:
            steps = ["(no steps)"]

        actual_vals = [step_costs.get(s, 0) for s in steps]

        # For each comparison, compute projected costs (use mean_error_pct to approximate)
        for comp_key in ["A", "B", "C"]:
            score = _get_score(comps.get(comp_key))
            if score:
                err = score.get("mean_error_pct", 0) / 100
                projected_vals = [v * (1 + err) for v in actual_vals]
            else:
                projected_vals = [0] * len(steps)

            visible = i == 0 and comp_key == "A"
            d3_data.append(
                go.Bar(
                    name=f"{wf} Actual ({comp_key})",
                    x=steps,
                    y=actual_vals,
                    marker_color=COMPARISON_COLORS.get(comp_key, "#888"),
                    opacity=0.5,
                    visible=visible,
                    showlegend=visible,
                )
            )
            d3_data.append(
                go.Bar(
                    name=f"{wf} Projected ({comp_key})",
                    x=steps,
                    y=projected_vals,
                    marker_color=COMPARISON_COLORS.get(comp_key, "#888"),
                    visible=visible,
                    showlegend=visible,
                )
            )

    # Build dropdown menus for D3
    n_traces_per_wf = 6  # 3 comparisons x 2 bars (actual+projected)
    total_traces = len(d3_data)
    d3_buttons = []
    for i, r in enumerate(results):
        wf = r["workflow_name"]
        for comp_idx, comp_key in enumerate(["A", "B", "C"]):
            visibility = [False] * total_traces
            base = i * n_traces_per_wf + comp_idx * 2
            if base < total_traces:
                visibility[base] = True
            if base + 1 < total_traces:
                visibility[base + 1] = True
            d3_buttons.append(
                dict(
                    label=f"{wf} ({comp_key})",
                    method="update",
                    args=[{"visible": visibility}],
                )
            )

    d3_fig = go.Figure(data=d3_data)
    d3_fig.update_layout(
        title="D3: Projected vs Actual Cost Explorer",
        barmode="group",
        updatemenus=[
            dict(
                buttons=d3_buttons[:50],  # Cap to avoid overly large menus
                direction="down",
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top",
            )
        ],
        margin=dict(t=80, b=40, l=60, r=20),
        yaxis_title="Cost ($)",
    )

    # D4: Detector Heatmap
    d4_workflows = sorted(
        [r["workflow_name"] for r in results],
        key=lambda w: int(w[1:]) if w[1:].isdigit() else 0,
    )
    d4_z = []
    d4_text = []
    classification_map = {"TP": 3, "FP": 2, "FN": 1, "TN": 0}
    label_map = {0: "TN", 1: "FN", 2: "FP", 3: "TP"}

    for wf in d4_workflows:
        row_z = []
        row_text = []
        wf_data = next((r for r in results if r["workflow_name"] == wf), None)
        raw_patterns = wf_data.get("detected_patterns", []) if wf_data else []
        detected = set(
            p["pattern_type"] if isinstance(p, dict) else p for p in raw_patterns
        )
        for det in _DETECTORS:
            fired = det in detected
            classification = classify_detector_result(wf, det, fired)
            row_z.append(classification_map[classification])
            row_text.append(classification)
        d4_z.append(row_z)
        d4_text.append(row_text)

    colorscale = [
        [0.0, DETECTOR_MATRIX_COLORS["TN"]],
        [0.33, DETECTOR_MATRIX_COLORS["FN"]],
        [0.67, DETECTOR_MATRIX_COLORS["FP"]],
        [1.0, DETECTOR_MATRIX_COLORS["TP"]],
    ]

    d4_fig = go.Figure(
        data=go.Heatmap(
            z=d4_z,
            x=[d.replace("_", " ").title() for d in _DETECTORS],
            y=d4_workflows,
            text=d4_text,
            texttemplate="%{text}",
            colorscale=colorscale,
            showscale=False,
            zmin=0,
            zmax=3,
        )
    )
    d4_fig.update_layout(
        title="D4: Detector Dashboard",
        xaxis_title="Detector",
        yaxis_title="Workflow",
        margin=dict(t=40, b=40, l=60, r=20),
    )

    # D5: Budget Tracker
    d5_workflows = sorted(
        [r["workflow_name"] for r in results],
        key=lambda w: int(w[1:]) if w[1:].isdigit() else 0,
    )
    d5_costs = []
    d5_colors = []
    for wf in d5_workflows:
        wf_data = next((r for r in results if r["workflow_name"] == wf), None)
        total = sum((wf_data.get("step_costs", {}) or {}).values()) if wf_data else 0
        d5_costs.append(total)
        d5_colors.append(workflow_color(wf))

    d5_fig = go.Figure(
        data=go.Bar(
            x=d5_workflows,
            y=d5_costs,
            marker_color=d5_colors,
            text=[f"${c:.4f}" for c in d5_costs],
            textposition="auto",
        )
    )
    d5_fig.update_layout(
        title="D5: Total Spend by Workflow",
        xaxis_title="Workflow",
        yaxis_title="Total Cost ($)",
        margin=dict(t=40, b=40, l=60, r=20),
    )

    # ---- Assemble single HTML ----
    total_cost = sum(d5_costs)
    from datetime import datetime, timezone

    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # Get HTML fragments from plotly (include_plotlyjs only on first)
    table_html = table_fig.to_html(full_html=False, include_plotlyjs=False)
    analytics_html = _build_analytics_section(results)
    d3_html = d3_fig.to_html(full_html=False, include_plotlyjs=False)
    d4_html = d4_fig.to_html(full_html=False, include_plotlyjs=False)
    d5_html = d5_fig.to_html(full_html=False, include_plotlyjs=False)

    # Get plotly.js as inline script
    import plotly.offline

    plotly_js = plotly.offline.get_plotlyjs()

    passing = summary["passing"]
    reweight_count = summary["reweight"]
    unresolved = summary["unresolved"]
    total = len(results)
    passing_wfs = ", ".join(summary["passing_wfs"]) or "none"
    reweight_wfs = ", ".join(summary["reweight_wfs"]) or "none"
    unresolved_wfs = ", ".join(summary["unresolved_wfs"]) or "none"

    blocked = unresolved > 0
    gate_icon = "&#10060;" if blocked else "&#9989;"
    gate_text = f"BLOCKED &mdash; {unresolved} workflow(s) need attention" if blocked else "PASSED"
    gate_class = "red" if blocked else "green"

    # Key findings from drift analysis
    findings_html = _build_key_findings(results, summary)

    # Embed static plots
    pilot_embed = _embed_plots(pilot_plots_dir) if pilot_plots_dir else ""
    backtest_embed = _embed_plots(backtest_plots_dir) if backtest_plots_dir else ""
    has_pilot = bool(pilot_embed and pilot_plots_dir)
    has_backtest = bool(backtest_embed and backtest_plots_dir)

    pilot_tab = (
        '<a class=\'tab\' onclick="scrollToSection(\'pilot-plots\', this)">Pilot Plots</a>'
        if has_pilot
        else ""
    )
    backtest_tab = (
        '<a class=\'tab\' onclick="scrollToSection(\'backtest-plots\', this)">Backtest Plots</a>'
        if has_backtest
        else ""
    )
    pilot_section = (
        "<div id='pilot-plots' class='section'><details open>"
        '<summary style="cursor:pointer;">'
        '<h2 style="display:inline;">Pilot Visualizations (P1-P6)</h2></summary>'
        '<p class="section-desc">Static plots from the 10-run pilot calibration.'
        " Click to expand/collapse.</p>"
        + pilot_embed
        + "</details></div>"
        if has_pilot
        else ""
    )
    backtest_section = (
        "<div id='backtest-plots' class='section'><details>"
        '<summary style="cursor:pointer;">'
        '<h2 style="display:inline;">Backtest Visualizations (B1-B10)</h2></summary>'
        '<p class="section-desc">Static plots from the full backtest across all 3 comparisons.'
        " Click to expand/collapse.</p>"
        + backtest_embed
        + "</details></div>"
        if has_backtest
        else ""
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Pretia Backtest Dashboard</title>
<script>{plotly_js}</script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 0; padding: 0; background: #f8f9fa; color: #212529; }}
h1 {{ text-align: center; margin: 20px 0 5px; }}
h2 {{ margin: 0 0 4px; font-size: 18px; color: #343a40; }}
.subtitle {{ text-align: center; color: #6c757d; margin-bottom: 10px; font-size: 14px; }}
.tab-bar {{ position: sticky; top: 0; z-index: 100; background: #343a40; display: flex; justify-content: center; gap: 0; padding: 0; box-shadow: 0 2px 6px rgba(0,0,0,0.15); }}
.tab {{ padding: 12px 24px; color: #adb5bd; cursor: pointer; font-size: 14px; font-weight: 500; border-bottom: 3px solid transparent; transition: all 0.2s; text-decoration: none; }}
.tab:hover {{ color: #fff; background: #495057; }}
.tab.active {{ color: #fff; border-bottom-color: #3498db; }}
.content {{ padding: 20px 30px; max-width: 1400px; margin: 0 auto; }}
.summary-row {{ display: flex; justify-content: center; gap: 24px; margin-bottom: 16px; flex-wrap: wrap; }}
.summary-card {{ background: white; border-radius: 12px; padding: 20px 32px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1); min-width: 200px; flex: 1; max-width: 300px; }}
.summary-card .number {{ font-size: 48px; font-weight: bold; }}
.summary-card .label {{ font-size: 14px; color: #6c757d; margin-top: 4px; }}
.summary-card .wf-list {{ font-size: 12px; color: #868e96; margin-top: 8px; line-height: 1.4; }}
.gate {{ text-align: center; font-size: 18px; font-weight: 600; margin: 16px 0 24px; padding: 12px; background: white; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.green {{ color: #27ae60; }}
.yellow {{ color: #f39c12; }}
.red {{ color: #e74c3c; }}
.section {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); }}
.section-desc {{ color: #6c757d; font-size: 13px; margin: 0 0 12px; }}
.findings {{ background: white; border-radius: 8px; padding: 20px 24px; margin-bottom: 20px; box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 4px solid #3498db; }}
.findings h2 {{ color: #2c3e50; }}
.findings ul {{ margin: 8px 0; padding-left: 20px; }}
.findings li {{ margin-bottom: 6px; font-size: 14px; color: #495057; }}
.footer {{ text-align: center; color: #adb5bd; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>

<!-- Tab Navigation -->
<div class="tab-bar">
  <a class="tab active" onclick="scrollToSection('summary', this)">Summary</a>
  {pilot_tab}
  {backtest_tab}
  <a class="tab" onclick="scrollToSection('results', this)">Results</a>
  <a class="tab" onclick="scrollToSection('analytics', this)">Analytics</a>
  <a class="tab" onclick="scrollToSection('explorer', this)">Explorer</a>
  <a class="tab" onclick="scrollToSection('detectors', this)">Detectors</a>
  <a class="tab" onclick="scrollToSection('budget', this)">Budget</a>
</div>

<div class="content">

<h1>Pretia Backtest Dashboard</h1>
<p class="subtitle">Generated {date_str} &middot; Total cost: ${total_cost:,.4f} &middot; {total} workflows</p>

<!-- D1: Executive Summary -->
<div id="summary">
<div class="summary-row">
  <div class="summary-card">
    <div class="number green">{passing}</div>
    <div class="label">Passing All Comparisons</div>
    <div class="wf-list">{passing_wfs}</div>
  </div>
  <div class="summary-card">
    <div class="number yellow">{reweight_count}</div>
    <div class="label">Need Reweighting</div>
    <div class="wf-list">{reweight_wfs}</div>
  </div>
  <div class="summary-card">
    <div class="number red">{unresolved}</div>
    <div class="label">Unresolved</div>
    <div class="wf-list">{unresolved_wfs}</div>
  </div>
</div>

<div class="gate">
  {gate_icon} LAUNCH GATE: <span class="{gate_class}">{gate_text}</span>
</div>

<!-- Key Findings -->
{findings_html}
</div>

{pilot_section}

{backtest_section}

<!-- D2: Results Table -->
<div id="results" class="section">
<h2>Results Table</h2>
<p class="section-desc">Per-workflow comparison results. Green = pass, red = fail. Bucket column shows failure classification.</p>
{table_html}
</div>

<!-- Analytics & Recommendations -->
<div id="analytics" class="section">
<h2>Analytics &amp; Recommendations</h2>
<p class="section-desc">Cost variance analysis, attribution, pain points, and optimization opportunities across all workflows.</p>
{analytics_html}
</div>

<!-- D3: Projected vs Actual Explorer -->
<div id="explorer" class="section">
<h2>Projected vs Actual Cost</h2>
<p class="section-desc">Select a workflow and comparison from the dropdown to compare projected and actual per-step costs.</p>
{d3_html}
</div>

<!-- D4: Detector Dashboard -->
<div id="detectors" class="section">
<h2>Detector Activation Matrix</h2>
<p class="section-desc">Green = true positive (expected &amp; fired), red = false negative (expected but missed), gray = true negative, yellow = false positive (unexpected).</p>
{d4_html}
</div>

<!-- D5: Budget Tracker -->
<div id="budget" class="section">
<h2>Budget Tracker</h2>
<p class="section-desc">Total API spend by workflow, colored by workflow group (green = linear, purple = loops, yellow = routing, blue = RAG/PDF).</p>
{d5_html}
</div>

<div class="footer">Pretia Backtesting Dashboard &middot; Generated with mock data</div>

</div><!-- /content -->

<script>
function scrollToSection(id, el) {{
  document.getElementById(id).scrollIntoView({{ behavior: 'smooth' }});
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');
}}
// Highlight active tab on scroll
window.addEventListener('scroll', function() {{
  const sections = ['summary', 'pilot-plots', 'backtest-plots', 'results', 'analytics', 'explorer', 'detectors', 'budget'].filter(id => document.getElementById(id));
  const tabs = document.querySelectorAll('.tab');
  let current = 0;
  for (let i = 0; i < sections.length; i++) {{
    const el = document.getElementById(sections[i]);
    if (el && el.getBoundingClientRect().top <= 80) current = i;
  }}
  tabs.forEach(t => t.classList.remove('active'));
  tabs[current].classList.add('active');
}});
</script>
</body>
</html>"""

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html)
    logger.info("Dashboard written to %s", output)
    return output


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Pretia backtest dashboard")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-workflow JSON result files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/dashboard.html"),
        help="Output HTML file path",
    )
    args = parser.parse_args()

    result = generate_dashboard(args.results_dir, args.output)
    if result:
        print(f"Dashboard generated: {result}")  # noqa: T201
    else:
        print("Dashboard generation failed or skipped.")  # noqa: T201


if __name__ == "__main__":
    main()
