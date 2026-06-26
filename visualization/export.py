"""Export analytics as JSON and PDF for LLM consumption."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _load_results(results_dir: Path) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return results
    for f in sorted(results_dir.glob("*.json")):
        if "pre_calibration" in f.name or "analytics" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            if "workflow_name" in data:
                results.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return results


def _get_score(comp_data: dict | None) -> dict[str, Any] | None:
    if comp_data is None:
        return None
    return comp_data.get("score") if isinstance(comp_data, dict) else None


def _build_validation(results: list[dict[str, Any]]) -> dict[str, Any]:
    validation: dict[str, Any] = {}
    for r in results:
        wf = r["workflow_name"]
        comps = r.get("comparisons", {})
        sa = _get_score(comps.get("A"))
        sb = _get_score(comps.get("B"))
        sc = _get_score(comps.get("C"))

        a_pass = sa.get("passes", False) if sa else False
        b_pass = sb.get("passes", False) if sb else False
        c_pass = sc.get("passes", False) if sc else False

        bucket = None
        explanation = None
        action = None
        if not a_pass:
            bucket = 1
            explanation = f"Comparison A failed: {', '.join(sa.get('failures', []))}" if sa else "No Comparison A data"
            action = "Fix projection engine"
        elif not b_pass:
            if c_pass:
                bucket = 2
                explanation = "Drifted comparison failed but reweighting recovers accuracy"
                action = "Use --traffic-mix to specify production distribution"
            elif sc and not c_pass:
                b_err = sb.get("mean_error_pct", 0) if sb else 0
                a_err = sa.get("mean_error_pct", 0) if sa else 0
                c_err = sc.get("mean_error_pct", 0) if sc else 0
                drift = b_err - a_err
                recovery = (b_err - c_err) / drift * 100 if drift > 0 else 0
                if recovery >= 50:
                    bucket = 2
                    explanation = f"Reweighting recovered {recovery:.0f}% of lost accuracy"
                    action = "Use --traffic-mix to specify production distribution"
                else:
                    bucket = 3
                    explanation = f"Structural drift — reweighting recovered only {recovery:.0f}% of lost accuracy"
                    action = "Re-profile with production-representative inputs"
            else:
                bucket = 3
                explanation = "Drifted comparison failed, no reweighting data available"
                action = "Re-profile with production-representative inputs"

        entry: dict[str, Any] = {}
        for key, score in [("comparison_a", sa), ("comparison_b", sb), ("comparison_c", sc)]:
            if score:
                entry[key] = {
                    "passes": score.get("passes", False),
                    "mean_error_pct": score.get("mean_error_pct"),
                    "p75_error_pct": score.get("p75_error_pct"),
                    "ci_coverage_pct": score.get("ci_coverage_pct"),
                    "monthly_error_pct": score.get("monthly_error_pct"),
                    "cvar95_error_pct": score.get("cvar95_error_pct"),
                    "failures": score.get("failures", []),
                }
            else:
                entry[key] = None

        entry["failure_bucket"] = bucket
        entry["failure_explanation"] = explanation
        entry["recommended_action"] = action
        validation[wf] = entry

    return validation


def _build_detectors(results: list[dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from visualization.colors import EXPECTED_DETECTORS, classify_detector_result

    detectors = ["context_growth", "loop_count_variance", "high_token_variance",
                 "step_count_variance", "bimodality"]

    matrix: dict[str, dict[str, str]] = {}
    tp = fn = fp = tn = 0
    false_negatives: list[dict] = []
    false_positives: list[dict] = []

    for r in results:
        wf = r["workflow_name"]
        raw = r.get("detected_patterns", [])
        detected = set(p["pattern_type"] if isinstance(p, dict) else p for p in raw)
        row: dict[str, str] = {}
        for det in detectors:
            fired = det in detected
            cls = classify_detector_result(wf, det, fired)
            row[det] = cls
            if cls == "TP":
                tp += 1
            elif cls == "TN":
                tn += 1
            elif cls == "FP":
                fp += 1
                false_positives.append({"workflow": wf, "detector": det})
            else:
                fn += 1
                false_negatives.append({"workflow": wf, "detector": det})
        matrix[wf] = row

    total = tp + tn + fp + fn
    return {
        "matrix": matrix,
        "rates": {
            "tp_rate": round(tp / total, 2) if total > 0 else 0,
            "fn_rate": round(fn / total, 2) if total > 0 else 0,
            "fp_rate": round(fp / total, 2) if total > 0 else 0,
            "tn_rate": round(tn / total, 2) if total > 0 else 0,
        },
        "false_negatives": false_negatives,
        "false_positives": false_positives,
    }


def _build_drift_analysis(results: list[dict[str, Any]]) -> dict[str, Any]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from visualization.colors import WORKFLOW_GROUPS

    b_failed = []
    for r in results:
        sb = _get_score(r.get("comparisons", {}).get("B"))
        if sb and not sb.get("passes"):
            b_failed.append(r["workflow_name"])

    total = len(results)
    if not b_failed:
        return {"diagnosis": "no_drift", "explanation": "No drift sensitivity detected",
                "workflows_affected": [], "reweighting_effective": False}

    group_counts: dict[str, int] = {}
    for wf in b_failed:
        g = WORKFLOW_GROUPS.get(wf, "unknown")
        group_counts[g] = group_counts.get(g, 0) + 1

    if len(b_failed) >= total * 0.5:
        diagnosis = "tier_weight_shift"
        explanation = f"Widespread degradation across {len(b_failed)}/{total} workflows suggests tier weight shift"
    elif len(group_counts) == 1:
        group = next(iter(group_counts))
        diagnosis = f"style_shift_{group}"
        explanation = f"Drift concentrated in {group} workflows ({', '.join(b_failed)})"
    else:
        diagnosis = "mixed_drift"
        explanation = f"{len(b_failed)} workflows affected across groups"

    reweight_effective = False
    for r in results:
        if r["workflow_name"] in b_failed:
            sc = _get_score(r.get("comparisons", {}).get("C"))
            if sc and sc.get("passes"):
                reweight_effective = True
                break

    return {
        "diagnosis": diagnosis,
        "explanation": explanation,
        "workflows_affected": b_failed,
        "reweighting_effective": reweight_effective,
    }


def _build_variance_risk(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    risks = []
    for r in results:
        wf = r["workflow_name"]
        step_stats = r.get("step_stats", {})
        if not step_stats:
            continue

        total_cost = sum(ss.get("cost", {}).get("mean", 0) for ss in step_stats.values())
        total_std = sum(ss.get("cost", {}).get("std", 0) for ss in step_stats.values())
        cv = total_std / total_cost if total_cost > 0 else 0

        riskiest_step = ""
        max_step_cv = 0
        for sn, ss in step_stats.items():
            c = ss.get("cost", {})
            s_cv = c.get("std", 0) / c.get("mean", 1) if c.get("mean", 0) > 0 else 0
            if s_cv > max_step_cv:
                max_step_cv = s_cv
                riskiest_step = sn

        costs = [ss.get("cost", {}) for ss in step_stats.values()]
        p50_sum = sum(c.get("p50", 0) for c in costs)
        p95_sum = sum(c.get("p95", 0) for c in costs)
        p10_sum = sum(c.get("p10", 0) for c in costs)
        p90_sum = sum(c.get("p90", 0) for c in costs)
        ratio = p95_sum / p50_sum if p50_sum > 0 else 1

        risk = "LOW" if cv < 0.3 else ("MODERATE" if cv < 0.8 else "HIGH")
        risks.append({
            "workflow": wf, "cost_cv": round(cv, 2),
            "p95_p50_ratio": round(ratio, 1),
            "cost_range": f"${p10_sum:.4f}–${p90_sum:.4f}",
            "riskiest_step": riskiest_step,
            "riskiest_step_cv": round(max_step_cv, 2),
            "risk_level": risk,
        })

    risks.sort(key=lambda x: x["cost_cv"], reverse=True)
    return risks


def _build_cost_attribution(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    attrs = []
    for r in results:
        wf = r["workflow_name"]
        step_stats = r.get("step_stats", {})
        total = sum(ss.get("cost", {}).get("mean", 0) for ss in step_stats.values())
        for sn, ss in step_stats.items():
            cost_mean = ss.get("cost", {}).get("mean", 0)
            in_mean = ss.get("input_tokens", {}).get("mean", 0)
            out_mean = ss.get("output_tokens", {}).get("mean", 0)
            attrs.append({
                "workflow": wf, "step": sn,
                "model": ss.get("model", ""),
                "tier": ss.get("model_tier", ""),
                "cost_pct": round(cost_mean / total * 100, 1) if total > 0 else 0,
                "cost_mean": round(cost_mean, 6),
                "output_input_ratio": round(out_mean / in_mean, 2) if in_mean > 0 else 0,
            })
    attrs.sort(key=lambda x: x["cost_mean"], reverse=True)
    return attrs


def _build_pain_points(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points = []
    for r in results:
        wf = r["workflow_name"]
        for sn, ss in r.get("step_stats", {}).items():
            in_mean = ss.get("input_tokens", {}).get("mean", 0)
            sys_tok = ss.get("system_prompt_tokens", 0)
            if in_mean > 0 and sys_tok / in_mean > 0.5:
                points.append({"workflow": wf, "step": sn, "type": "context_overhead",
                               "metric": f"sys_prompt={sys_tok / in_mean:.0%} of input",
                               "impact": "HIGH"})

            iters = ss.get("mean_iterations", 1)
            cost_mean = ss.get("cost", {}).get("mean", 0)
            if iters > 2 and cost_mean / iters > 0.005:
                points.append({"workflow": wf, "step": sn, "type": "loop_amplifier",
                               "metric": f"{iters:.1f} iters × ${cost_mean / iters:.4f}/iter",
                               "impact": "HIGH"})

            retry = ss.get("is_retry_pct", 0)
            if retry > 5:
                points.append({"workflow": wf, "step": sn, "type": "retry_cost",
                               "metric": f"retry_rate={retry:.0f}%",
                               "impact": "MODERATE"})

            p50 = ss.get("cost", {}).get("p50", 0)
            p95 = ss.get("cost", {}).get("p95", 0)
            if p50 > 0 and p95 > 3 * p50:
                points.append({"workflow": wf, "step": sn, "type": "unpredictable",
                               "metric": f"p95/p50={p95 / p50:.1f}x",
                               "impact": "MODERATE"})
    return points


def _build_plot_summaries(results: list[dict[str, Any]], detectors_data: dict) -> dict[str, str]:
    summaries: dict[str, str] = {}

    passing_wfs = []
    bimodal_wfs = []
    drift_wfs = []
    for r in results:
        wf = r["workflow_name"]
        sa = _get_score(r.get("comparisons", {}).get("A"))
        sb = _get_score(r.get("comparisons", {}).get("B"))
        if sa and sa.get("passes") and sb and sb.get("passes"):
            passing_wfs.append(wf)
        if sb and not sb.get("passes"):
            drift_wfs.append(wf)
        pats = r.get("detected_patterns", [])
        for p in pats:
            if (isinstance(p, dict) and p.get("pattern_type") == "bimodality") or p == "bimodality":
                bimodal_wfs.append(wf)

    summaries["p1_infrastructure_check_matrix"] = (
        "Pre-calibration check results across all workflows. "
        "Look for red cells indicating infrastructure issues that must be fixed before the pilot."
    )
    summaries["p2_cost_distribution"] = (
        f"Per-run cost distribution for {len(results)} workflows. "
        "Higher max/min ratios indicate more cost variance from input diversity."
    )
    summaries["b1_projected_vs_actual_kde"] = (
        f"{len(passing_wfs)} workflows show tight projection-to-actuals overlap "
        f"({', '.join(passing_wfs[:5])}). "
        + (f"{', '.join(bimodal_wfs)} show bimodal distributions. " if bimodal_wfs else "")
        + (f"{', '.join(drift_wfs)} show systematic mismatch under drift." if drift_wfs else "")
    )

    rates = detectors_data.get("rates", {})
    fn_list = detectors_data.get("false_negatives", [])
    fp_list = detectors_data.get("false_positives", [])
    fn_desc = ", ".join(f"{fn['workflow']}/{fn['detector']}" for fn in fn_list[:3]) if fn_list else ""
    fp_desc = ", ".join(f"{fp['workflow']}/{fp['detector']}" for fp in fp_list[:3]) if fp_list else ""
    summaries["b5_detector_activation_matrix"] = (
        f"TP rate: {rates.get('tp_rate', 0):.0%}. "
        f"FN rate: {rates.get('fn_rate', 0):.0%}. "
        + (f"False negatives: {fn_desc}. " if fn_list else "No false negatives. ")
        + (f"False positives: {fp_desc}." if fp_list else "No false positives.")
    )

    summaries["b3_drift_impact"] = (
        f"{len(drift_wfs)} workflows degrade under drifted inputs. "
        + ("Reweighting recovers accuracy for most." if len(drift_wfs) < len(results) * 0.5 else
           "Widespread — suggests tier weight shift is the dominant drift vector.")
    )

    return summaries


def export_analytics_json(results_dir: Path, output: Path) -> Path:
    """Export consolidated analytics as a single LLM-readable JSON."""
    results = _load_results(results_dir)
    if not results:
        logger.warning("No results to export")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("{}")
        return output

    from visualization.dashboard.generate_dashboard import _detect_opportunities

    validation = _build_validation(results)
    detectors_data = _build_detectors(results)
    drift = _build_drift_analysis(results)
    variance = _build_variance_risk(results)
    attribution = _build_cost_attribution(results)
    pain_points = _build_pain_points(results)
    opportunities = _detect_opportunities(results)
    plot_summaries = _build_plot_summaries(results, detectors_data)

    passing = [wf for wf, v in validation.items() if v["failure_bucket"] is None]
    reweight = [wf for wf, v in validation.items() if v["failure_bucket"] == 2]
    unresolved = [wf for wf, v in validation.items() if v["failure_bucket"] in (1, 3)]
    gate = "PASSED" if not unresolved else "BLOCKED"

    total_savings = sum(o.get("monthly_savings", 0) for o in opportunities)
    total_cost = sum(
        sum(ss.get("cost", {}).get("mean", 0) for ss in r.get("step_stats", {}).values())
        for r in results
    )

    export = {
        "meta": {
            "generated": datetime.now(timezone.utc).isoformat(),
            "total_workflows": len(results),
            "total_backtest_cost": round(total_cost, 4),
            "format_version": "1.0",
        },
        "launch_gate": {
            "status": gate,
            "passing": passing,
            "need_reweighting": reweight,
            "unresolved": unresolved,
        },
        "validation": validation,
        "detectors": detectors_data,
        "drift_analysis": drift,
        "variance_risk": variance,
        "cost_attribution": attribution[:30],
        "pain_points": pain_points,
        "opportunities": opportunities,
        "total_monthly_savings": round(total_savings, 2),
        "plot_summaries": plot_summaries,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2))
    logger.info("Analytics JSON exported to %s", output)
    return output


def export_analytics_pdf(
    results_dir: Path,
    pilot_plots_dir: Path | None,
    backtest_plots_dir: Path | None,
    output: Path,
) -> Path:
    """Export analytics as a PDF with embedded plots, tables, and narrative."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    results = _load_results(results_dir)
    from visualization.dashboard.generate_dashboard import _detect_opportunities

    validation = _build_validation(results)
    detectors_data = _build_detectors(results)
    drift = _build_drift_analysis(results)
    opportunities = _detect_opportunities(results)
    pain_points = _build_pain_points(results)

    passing = [wf for wf, v in validation.items() if v["failure_bucket"] is None]
    reweight = [wf for wf, v in validation.items() if v["failure_bucket"] == 2]
    unresolved = [wf for wf, v in validation.items() if v["failure_bucket"] in (1, 3)]
    gate = "PASSED" if not unresolved else "BLOCKED"
    total_savings = sum(o.get("monthly_savings", 0) for o in opportunities)

    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=letter,
                            leftMargin=0.75 * inch, rightMargin=0.75 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.75 * inch)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title2", parent=styles["Title"], fontSize=20, spaceAfter=12)
    h1 = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=16, spaceAfter=8, spaceBefore=16)
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceAfter=6, spaceBefore=12)
    body = styles["BodyText"]
    small = ParagraphStyle("Small", parent=body, fontSize=8, leading=10)

    elements: list = []

    # Title page
    elements.append(Paragraph("Pretia Backtest Analytics", title_style))
    elements.append(Paragraph(
        f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} | "
        f"{len(results)} workflows",
        body,
    ))
    elements.append(Spacer(1, 12))

    gate_color = "green" if gate == "PASSED" else "red"
    elements.append(Paragraph(
        f'<b>LAUNCH GATE: <font color="{gate_color}">{gate}</font></b>',
        ParagraphStyle("Gate", parent=body, fontSize=14),
    ))
    elements.append(Spacer(1, 6))
    elements.append(Paragraph(f"<b>Passing:</b> {', '.join(passing) or 'none'}", body))
    elements.append(Paragraph(f"<b>Need reweighting:</b> {', '.join(reweight) or 'none'}", body))
    elements.append(Paragraph(f"<b>Unresolved:</b> {', '.join(unresolved) or 'none'}", body))
    elements.append(Paragraph(
        f"<b>Total potential savings:</b> ${total_savings:,.2f}/month", body,
    ))
    elements.append(Spacer(1, 12))

    # Drift analysis
    elements.append(Paragraph(
        f"<b>Drift analysis:</b> {drift['explanation']}", body,
    ))
    elements.append(PageBreak())

    # Validation table
    elements.append(Paragraph("Validation Results", h1))
    val_data = [["Workflow", "Comp A", "Comp B", "Comp C", "Bucket", "Action"]]
    for wf in sorted(validation.keys(), key=lambda w: int(w[1:]) if w[1:].isdigit() else 0):
        v = validation[wf]
        a = "PASS" if v.get("comparison_a", {}) and v["comparison_a"].get("passes") else "FAIL"
        b = "PASS" if v.get("comparison_b", {}) and v["comparison_b"].get("passes") else ("FAIL" if v.get("comparison_b") else "N/A")
        c = "PASS" if v.get("comparison_c", {}) and v["comparison_c"].get("passes") else ("FAIL" if v.get("comparison_c") else "N/A")
        bucket = str(v["failure_bucket"]) if v["failure_bucket"] else "-"
        action = (v["recommended_action"] or "-")[:50]
        val_data.append([wf, a, b, c, bucket, action])

    val_table = Table(val_data, colWidths=[50, 45, 45, 45, 45, 220])
    val_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    for i, row in enumerate(val_data[1:], 1):
        for j, val in enumerate(row[1:4], 1):
            if val == "PASS":
                val_table.setStyle(TableStyle([
                    ("BACKGROUND", (j, i), (j, i), rl_colors.HexColor("#d4edda")),
                ]))
            elif val == "FAIL":
                val_table.setStyle(TableStyle([
                    ("BACKGROUND", (j, i), (j, i), rl_colors.HexColor("#f8d7da")),
                ]))
    elements.append(val_table)
    elements.append(Spacer(1, 12))

    # Detector matrix
    elements.append(Paragraph("Detector Activation Matrix", h2))
    det_names = ["context_growth", "loop_count_var", "token_var", "step_count_var", "bimodality"]
    det_data = [["Workflow"] + det_names]
    matrix = detectors_data.get("matrix", {})
    for wf in sorted(matrix.keys(), key=lambda w: int(w[1:]) if w[1:].isdigit() else 0):
        row = [wf]
        for det in ["context_growth", "loop_count_variance", "high_token_variance",
                     "step_count_variance", "bimodality"]:
            row.append(matrix[wf].get(det, "?"))
        det_data.append(row)

    det_table = Table(det_data, colWidths=[50] + [75] * 5)
    det_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("ALIGN", (1, 1), (-1, -1), "CENTER"),
    ]))
    cls_colors = {"TP": "#d4edda", "TN": "#e2e3e5", "FP": "#fff3cd", "FN": "#f8d7da"}
    for i, row in enumerate(det_data[1:], 1):
        for j, val in enumerate(row[1:], 1):
            if val in cls_colors:
                det_table.setStyle(TableStyle([
                    ("BACKGROUND", (j, i), (j, i), rl_colors.HexColor(cls_colors[val])),
                ]))
    elements.append(det_table)
    elements.append(Spacer(1, 6))
    rates = detectors_data.get("rates", {})
    elements.append(Paragraph(
        f"TP rate: {rates.get('tp_rate', 0):.0%} | "
        f"FN rate: {rates.get('fn_rate', 0):.0%} | "
        f"FP rate: {rates.get('fp_rate', 0):.0%}",
        small,
    ))
    elements.append(PageBreak())

    # Opportunities table
    elements.append(Paragraph("Optimization Opportunities", h1))
    elements.append(Paragraph(
        f"Total potential savings: <b>${total_savings:,.2f}/month</b> at 1K daily volume", body,
    ))
    elements.append(Spacer(1, 6))

    opp_data = [["Type", "Workflow", "Step", "Issue", "Savings/mo"]]
    for o in opportunities[:25]:
        opp_data.append([
            o["rec_label"][:18],
            o["workflow"],
            o["step"][:12],
            o["issue"][:45],
            f"${o['monthly_savings']:,.2f}",
        ])
    opp_table = Table(opp_data, colWidths=[90, 40, 60, 220, 65])
    opp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#343a40")),
        ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 7),
        ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    elements.append(opp_table)
    elements.append(PageBreak())

    # Pain points
    elements.append(Paragraph("Pain Points", h1))
    if pain_points:
        pp_data = [["Workflow", "Step", "Type", "Metric", "Impact"]]
        for pp in pain_points[:20]:
            pp_data.append([pp["workflow"], pp["step"], pp["type"],
                           pp["metric"][:40], pp["impact"]])
        pp_table = Table(pp_data, colWidths=[50, 60, 90, 200, 55])
        pp_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#343a40")),
            ("TEXTCOLOR", (0, 0), (-1, 0), rl_colors.white),
            ("FONTSIZE", (0, 0), (-1, -1), 7),
            ("GRID", (0, 0), (-1, -1), 0.5, rl_colors.grey),
        ]))
        elements.append(pp_table)
    elements.append(PageBreak())

    # Embedded plots
    def _add_plots(plot_dir: Path | None, section_title: str) -> None:
        if not plot_dir or not plot_dir.is_dir():
            return
        pngs = sorted(plot_dir.glob("*.png"))
        if not pngs:
            return
        elements.append(Paragraph(section_title, h1))
        for png in pngs:
            try:
                img = Image(str(png), width=6.5 * inch, height=4 * inch)
                img.hAlign = "CENTER"
                elements.append(Paragraph(png.stem.replace("_", " ").title(), h2))
                elements.append(img)
                elements.append(Spacer(1, 8))
            except Exception as e:
                elements.append(Paragraph(f"[Could not embed {png.name}: {e}]", small))

    _add_plots(pilot_plots_dir, "Pilot Visualizations (P1-P6)")
    _add_plots(backtest_plots_dir, "Backtest Visualizations (B1-B10)")

    doc.build(elements)
    logger.info("Analytics PDF exported to %s", output)
    return output
