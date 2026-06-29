"""Render profiling data to a self-contained HTML report."""

from __future__ import annotations

import json
import logging
import webbrowser
from pathlib import Path
from typing import Any

import jinja2

from pretia.ci.report import format_cost
from pretia.report.charts import StepCostEntry, render_cost_waterfall, render_score_ring
from pretia.store import ProfilingSession

logger = logging.getLogger(__name__)

_TEMPLATE_NAME = "report.html.j2"
_RAW_JSON_MAX_BYTES = 50_000
_MAX_RUNS_IN_RAW = 3


def _load_template() -> jinja2.Template:
    """Load the Jinja2 report template from package data."""
    template_path = Path(__file__).parent / "templates" / _TEMPLATE_NAME
    if template_path.exists():
        text = template_path.read_text(encoding="utf-8")
    else:
        from importlib.resources import files

        text = (files("pretia.report").joinpath("templates").joinpath(_TEMPLATE_NAME)).read_text(
            encoding="utf-8"
        )

    env = jinja2.Environment(autoescape=True)
    env.filters["format_cost"] = format_cost

    def format_volume(v: int) -> str:
        if v >= 1_000_000:
            return f"{v // 1_000_000}M"
        if v >= 1_000:
            return f"{v // 1_000}K"
        return str(v)

    env.filters["format_volume"] = format_volume
    return env.from_string(text)


def _prepare_context(session: ProfilingSession) -> dict[str, Any]:
    """Extract all template variables from a profiling session."""
    meta = session.metadata or {}
    stats = meta.get("stats")
    score_data = meta.get("score")
    recommendations = meta.get("recommendations", [])
    patterns = meta.get("patterns", [])
    projection = meta.get("projection")

    total_runs = stats["total_runs"] if stats else session.sample_size
    total_steps = stats.get("total_steps", 0) if stats else 0

    score_ring_svg = ""
    if score_data:
        score_ring_svg = render_score_ring(
            score=score_data.get("score", 0),
            zone_color=score_data.get("zone_color", "#A0AEC0"),
            zone_label=score_data.get("zone_label", ""),
        )

    step_entries = _build_step_entries(stats)
    waterfall_svg = render_cost_waterfall(step_entries)

    cost_per_run = _extract_cost_per_run(stats, meta.get("cost_summary", {}))

    projection_rows, traffic_volumes, has_cvar, cvar_values, confidence, method = (
        _extract_projection(projection, cost_per_run)
    )

    # Hero projected cost — pick the middle traffic volume (prefer 1K)
    hero_projected_cost = ""
    if traffic_volumes:
        hero_vol = 1_000 if 1_000 in traffic_volumes else traffic_volumes[0]
        hero_p50 = projection_rows.get(hero_vol, {}).get("p50", 0)
        if hero_vol >= 1_000_000:
            vol_label = f"{hero_vol // 1_000_000}M"
        elif hero_vol >= 1_000:
            vol_label = f"{hero_vol // 1_000}K"
        else:
            vol_label = str(hero_vol)
        hero_projected_cost = f"{format_cost(hero_p50)}/mo at {vol_label} daily runs"

    projection_labels = {
        "p50": "Expected",
        "p90": "Likely high",
        "p95": "Bad month",
        "p99": "Worst case",
    }

    projection_display_pcts = ["p50", "p90", "p95", "p99"]

    raw_dict = _truncated_session_dict(session)
    raw_json = json.dumps(raw_dict, indent=2, default=str)

    return {
        "workflow_name": session.workflow_name,
        "profiled_at_display": session.profiled_at.strftime("%Y-%m-%d %H:%M"),
        "total_runs": total_runs,
        "total_steps": total_steps,
        "input_mode": session.input_mode,
        "framework": session.framework,
        "pretia_version": session.pretia_version,
        "profiling_cost": session.profiling_cost,
        "score": score_data,
        "score_ring_svg": score_ring_svg,
        "cost_per_run": cost_per_run,
        "steps": [
            {
                "name": e.step_name,
                "mean_cost": e.mean_cost,
                "share_pct": e.share_pct,
            }
            for e in step_entries
        ],
        "waterfall_svg": waterfall_svg,
        "projection_rows": projection_rows,
        "traffic_volumes": traffic_volumes,
        "has_cvar": has_cvar,
        "cvar_values": cvar_values,
        "confidence": confidence,
        "projection_method": method,
        "hero_projected_cost": hero_projected_cost,
        "projection_labels": projection_labels,
        "projection_display_pcts": projection_display_pcts,
        "patterns": patterns,
        "has_patterns": bool(patterns),
        "recommendations": recommendations,
        "has_recommendations": bool(recommendations),
        "raw_json": raw_json,
    }


def _build_step_entries(stats: dict[str, Any] | None) -> list[StepCostEntry]:
    """Build sorted step cost entries from stats."""
    if not stats or "step_stats" not in stats:
        return []

    entries: list[StepCostEntry] = []
    total_cost = 0.0
    step_costs: list[tuple[str, float]] = []

    for step_name, ss in stats["step_stats"].items():
        mean_cost = ss.get("cost", {}).get("mean", 0.0)
        step_costs.append((step_name, mean_cost))
        total_cost += mean_cost

    for step_name, mean_cost in step_costs:
        share = (mean_cost / total_cost * 100) if total_cost > 0 else 0.0
        entries.append(StepCostEntry(step_name=step_name, mean_cost=mean_cost, share_pct=share))

    entries.sort(key=lambda e: e.mean_cost, reverse=True)
    return entries


def _extract_cost_per_run(
    stats: dict[str, Any] | None,
    cost_summary: dict[str, Any],
) -> dict[str, float]:
    """Extract cost-per-run percentile data."""
    if stats and "cost_per_run" in stats and stats["cost_per_run"]:
        cpr = stats["cost_per_run"]
        return {
            "mean": cpr.get("mean", 0),
            "p50": cpr.get("p50", 0),
            "p75": cpr.get("p75", 0),
            "p90": cpr.get("p90", 0),
            "p95": cpr.get("p95", 0),
            "p99": cpr.get("p99", 0),
            "min": cpr.get("min", 0),
            "max": cpr.get("max", 0),
            "std": cpr.get("std", 0),
        }

    return {
        "mean": cost_summary.get("mean_cost_per_run", 0),
        "p50": cost_summary.get("mean_cost_per_run", 0),
        "p75": 0,
        "p90": 0,
        "p95": cost_summary.get("p95_cost_per_run", 0),
        "p99": 0,
        "min": cost_summary.get("min_cost_per_run", 0),
        "max": cost_summary.get("max_cost_per_run", 0),
        "std": 0,
    }


def _extract_projection(
    projection: dict[str, Any] | None,
    cost_per_run: dict[str, float],
) -> tuple[
    dict[int, dict[str, float]],
    list[int],
    bool,
    dict[int, float],
    dict[str, Any] | None,
    str,
]:
    """Extract projection table data.

    Returns (projection_rows, traffic_volumes, has_cvar, cvar_values, confidence, method).
    """
    if not projection:
        volumes = [100, 1_000, 10_000]
        rows: dict[int, dict[str, float]] = {}
        mean = cost_per_run.get("mean", 0)
        p95 = cost_per_run.get("p95", 0)
        for v in volumes:
            rows[v] = {
                "p50": mean * v * 30,
                "p75": mean * v * 30,
                "p90": p95 * v * 30 * 0.95,
                "p95": p95 * v * 30,
                "p99": p95 * v * 30 * 1.1,
            }
        # Filter out traffic volumes where ALL percentile values round to less than $0.01
        filtered_volumes = [
            v
            for v in volumes
            if any(rows[v][pct] >= 0.01 for pct in ["p50", "p75", "p90", "p95", "p99"])
        ]
        if not filtered_volumes:
            filtered_volumes = volumes  # keep all if everything is near-zero
        return rows, filtered_volumes, False, {}, None, "linear (estimated)"

    method = projection.get("method", "linear")
    confidence = projection.get("confidence")
    volumes = projection.get("traffic_volumes", [100, 1_000, 10_000])
    projs = projection.get("projections", {})
    mc_result = projection.get("montecarlo_result")

    method_label = "Linear"
    if method == "montecarlo":
        n_sims = mc_result.get("n_simulations", 10_000) if mc_result else 10_000
        method_label = f"Monte Carlo ({n_sims:,} sims)"

    rows = {}
    for v in volumes:
        vol_data = projs.get(str(v), projs.get(v, {}))
        monthly = vol_data.get("monthly_cost", {})
        rows[v] = {
            "p50": monthly.get("p50", 0),
            "p75": monthly.get("p75", 0),
            "p90": monthly.get("p90", 0),
            "p95": monthly.get("p95", 0),
            "p99": monthly.get("p99", 0),
        }

    # Filter out traffic volumes where ALL percentile values round to less than $0.01
    filtered_volumes = [
        v
        for v in volumes
        if any(rows[v][pct] >= 0.01 for pct in ["p50", "p75", "p90", "p95", "p99"])
    ]
    if not filtered_volumes:
        filtered_volumes = volumes  # keep all if everything is near-zero

    has_cvar = mc_result is not None and "cvar_95" in (mc_result or {})
    cvar_values: dict[int, float] = {}
    if has_cvar and mc_result:
        cvar_base = mc_result.get("cvar_95", 0)
        for v in filtered_volumes:
            cvar_values[v] = cvar_base * v * 30

    return rows, filtered_volumes, has_cvar, cvar_values, confidence, method_label


def _truncated_session_dict(session: ProfilingSession) -> dict[str, Any]:
    """Serialize session to dict, truncating runs if the result is too large."""
    d = session.to_dict()
    raw = json.dumps(d, default=str)
    too_large = len(raw.encode("utf-8")) > _RAW_JSON_MAX_BYTES
    if too_large and len(d.get("runs", [])) > _MAX_RUNS_IN_RAW:
        d["runs"] = d["runs"][:_MAX_RUNS_IN_RAW]
        d["_runs_truncated"] = True
        d["_runs_truncated_note"] = (
            f"Showing first {_MAX_RUNS_IN_RAW} of {session.sample_size} runs. "
            "Load the full profile JSON for complete data."
        )
    return d


def render_html_report(session: ProfilingSession) -> str:
    """Render a ProfilingSession to a self-contained HTML string."""
    template = _load_template()
    context = _prepare_context(session)
    return template.render(**context)


def render_and_save(
    session: ProfilingSession,
    output_path: Path | None = None,
    open_browser: bool = True,
) -> Path:
    """Render to HTML, write to disk, and optionally open in the browser.

    Returns the path of the written HTML file.
    """
    html = render_html_report(session)

    if output_path is None:
        stamp = session.profiled_at.strftime("%Y%m%d_%H%M%S")
        safe_name = Path(session.workflow_name).stem.replace(" ", "_") or "workflow"
        report_dir = Path(".pretia") / "reports"
        output_path = report_dir / f"{safe_name}_{stamp}.html"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")

    if open_browser:
        try:
            webbrowser.open(f"file://{output_path.resolve()}")
        except Exception:
            logger.debug("Could not open browser", exc_info=True)

    return output_path
