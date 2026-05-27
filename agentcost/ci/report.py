"""Format cost reports for CLI output and GitHub PR comments."""

from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentcost.store import ProfilingSession


def _fmt_cost(value: float) -> str:
    if value == 0:
        return "$0.00"
    if abs(value) < 1.0:
        return f"${value:.4f}"
    return f"${value:,.2f}"


def _fmt_tokens(value: float) -> str:
    return f"{value:,.0f}"


def _tier_style(tier: str) -> str:
    if tier == "fast":
        return "green"
    if tier == "mid":
        return "yellow"
    if tier == "frontier":
        return "red"
    return "dim"


def format_cli_report(
    session: ProfilingSession,
    cost_summary: dict[str, Any],
) -> list[Any]:
    """Build a list of rich renderables for terminal output."""
    renderables: list[Any] = []

    header = Text.assemble(
        ("Workflow: ", "bold"),
        (session.workflow_name, ""),
        (" | ", "dim"),
        (f"{session.sample_size} runs", ""),
        (" | ", "dim"),
        (session.input_mode, "cyan"),
        (" | ", "dim"),
        (session.profiled_at.strftime("%Y-%m-%d %H:%M"), "dim"),
    )
    renderables.append(Panel(header, title="AgentCost Report"))

    step_table = Table(
        title="Step Breakdown",
        show_lines=False,
        pad_edge=False,
    )
    step_table.add_column("Step", style="bold")
    step_table.add_column("Model")
    step_table.add_column("Tier")
    step_table.add_column("Avg In", justify="right")
    step_table.add_column("Avg Out", justify="right")
    step_table.add_column("Avg Cost", justify="right")
    step_table.add_column("p95 Cost", justify="right")
    step_table.add_column("Calls/Run", justify="right")

    per_step = cost_summary.get("per_step", {})
    sorted_steps = sorted(
        per_step.items(),
        key=lambda kv: kv[1].get("cost_mean", 0),
        reverse=True,
    )

    for step_name, stats in sorted_steps:
        tier = stats.get("tier", "unknown")
        style = _tier_style(tier)
        step_type = stats.get("step_type", "llm")

        if step_type == "tool":
            step_table.add_row(
                step_name,
                Text("—", style="dim"),
                Text("tool", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
                Text("—", style="dim"),
                _fmt_tokens(
                    stats["count"] / max(session.sample_size, 1),
                ),
            )
        else:
            step_table.add_row(
                step_name,
                stats.get("model", ""),
                Text(tier, style=style),
                _fmt_tokens(stats.get("input_tokens_mean", 0)),
                _fmt_tokens(stats.get("output_tokens_mean", 0)),
                Text(_fmt_cost(stats.get("cost_mean", 0)), style=style),
                _fmt_cost(stats.get("cost_p95", 0)),
                _fmt_tokens(
                    stats["count"] / max(session.sample_size, 1),
                ),
            )

    renderables.append(step_table)

    summary_table = Table(
        title="Run Summary",
        show_lines=False,
        pad_edge=False,
    )
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row(
        "Mean cost/run",
        _fmt_cost(cost_summary.get("mean_cost_per_run", 0)),
    )
    summary_table.add_row(
        "Min cost/run",
        _fmt_cost(cost_summary.get("min_cost_per_run", 0)),
    )
    summary_table.add_row(
        "Max cost/run",
        _fmt_cost(cost_summary.get("max_cost_per_run", 0)),
    )
    summary_table.add_row(
        "p95 cost/run",
        _fmt_cost(cost_summary.get("p95_cost_per_run", 0)),
    )
    summary_table.add_row(
        "Total session cost",
        _fmt_cost(cost_summary.get("total_session_cost", 0)),
    )
    renderables.append(summary_table)

    proj_text = Text()
    for label, key in [
        ("    100/day", "projection_100_day"),
        ("  1,000/day", "projection_1000_day"),
        (" 10,000/day", "projection_10000_day"),
    ]:
        val = cost_summary.get(key, 0)
        proj_text.append(f"  {label}  →  ", style="dim")
        proj_text.append(f"{_fmt_cost(val)}/mo\n")
    renderables.append(
        Panel(proj_text, title="Monthly Projection", expand=False),
    )

    flags = _detect_flags(cost_summary)
    if flags:
        flag_text = Text()
        for flag in flags:
            flag_text.append("  ⚠ ", style="yellow")
            flag_text.append(flag + "\n")
        renderables.append(
            Panel(flag_text, title="Flags", expand=False),
        )

    return renderables


def _detect_flags(cost_summary: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for step_name, stats in cost_summary.get("per_step", {}).items():
        max_iter = stats.get("max_iteration", 1)
        if max_iter > 3:
            flags.append(
                f"{step_name}: max {max_iter} iterations in a "
                f"single run — potential loop cost."
            )

        mean_cost = stats.get("cost_mean", 0)
        p95_cost = stats.get("cost_p95", 0)
        if mean_cost > 0 and p95_cost > 3 * mean_cost:
            flags.append(
                f"{step_name}: p95 cost ({_fmt_cost(p95_cost)}) "
                f"is >3x mean ({_fmt_cost(mean_cost)}) — "
                f"high variance."
            )
    return flags
