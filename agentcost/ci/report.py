"""Format cost reports for CLI output and GitHub PR comments."""

from __future__ import annotations

from typing import Any

from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentcost.store import ProfilingSession


def format_cost(value: float) -> str:
    """Format a dollar amount for display."""
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.4f}"
    if abs(value) < 1000:
        return f"${value:,.2f}"
    return f"${value:,.0f}"


def format_tokens(value: float) -> str:
    """Format a token count with commas."""
    return f"{value:,.0f}"


def _tier_style(tier: str) -> str:
    if tier == "fast":
        return "green"
    if tier == "mid":
        return "yellow"
    if tier == "frontier":
        return "red"
    return "dim"


def _truncate_model(model: str, max_len: int = 7) -> str:
    if len(model) <= max_len:
        return model
    return model[: max_len - 1] + "…"


def format_cli_report(
    session: ProfilingSession,
    cost_summary: dict[str, Any] | None = None,
    traffic: int | None = None,
) -> list[Any]:
    """Build a list of rich renderables for terminal output.

    Supports both the new stats-based metadata format and the legacy
    cost_summary format from earlier prompts.
    """
    meta = session.metadata or {}
    stats = meta.get("stats")
    patterns = meta.get("patterns", [])
    if cost_summary is None:
        cost_summary = meta.get("cost_summary", {})

    renderables: list[Any] = []

    total_runs = stats["total_runs"] if stats else session.sample_size
    total_steps = stats.get("total_steps", 0) if stats else ""
    header = Text.assemble(
        ("AgentCost Profile Report\n", "bold"),
        ("Workflow: ", "dim"),
        (session.workflow_name, ""),
        ("\n", ""),
        ("Runs: ", "dim"),
        (str(total_runs), ""),
        (" | ", "dim"),
        ("Steps: ", "dim"),
        (str(total_steps), ""),
        (" | ", "dim"),
        ("Generated: ", "dim"),
        (session.profiled_at.strftime("%Y-%m-%d %H:%M"), ""),
    )
    renderables.append(Panel(header, title="AgentCost Report"))

    renderables.append(_build_cost_summary_table(stats, cost_summary))

    renderables.append(_build_step_table(stats, cost_summary, session.sample_size))

    renderables.append(_build_projection_panel(stats, cost_summary, traffic))

    renderables.append(_build_patterns_panel(patterns))

    iter_panel = _build_iteration_panel(stats)
    if iter_panel is not None:
        renderables.append(iter_panel)

    saved_path = meta.get("saved_path", "")
    footer_parts = []
    if saved_path:
        footer_parts.append(f"Source: {saved_path}")
    footer_parts.append(f"Input mode: {session.input_mode}")
    renderables.append(Text("\n".join(footer_parts), style="dim"))

    return renderables


def _build_cost_summary_table(
    stats: dict[str, Any] | None,
    cost_summary: dict[str, Any],
) -> Table:
    table = Table(title="Cost Per Run", show_lines=True, pad_edge=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    if stats and "cost_per_run" in stats and stats["cost_per_run"]:
        cpr = stats["cost_per_run"]
        table.add_row("Mean", format_cost(cpr.get("mean", 0)))
        table.add_row("Median", format_cost(cpr.get("p50", 0)))
        table.add_row("p95", format_cost(cpr.get("p95", 0)))
        table.add_row("p99", format_cost(cpr.get("p99", 0)))
        table.add_row("Min", format_cost(cpr.get("min", 0)))
        table.add_row("Max", format_cost(cpr.get("max", 0)))
        table.add_row("Std Dev", format_cost(cpr.get("std", 0)))
    else:
        table.add_row("Mean", format_cost(cost_summary.get("mean_cost_per_run", 0)))
        table.add_row("Min", format_cost(cost_summary.get("min_cost_per_run", 0)))
        table.add_row("Max", format_cost(cost_summary.get("max_cost_per_run", 0)))
        table.add_row("p95", format_cost(cost_summary.get("p95_cost_per_run", 0)))
        table.add_row(
            "Total session",
            format_cost(cost_summary.get("total_session_cost", 0)),
        )

    return table


def _build_step_table(
    stats: dict[str, Any] | None,
    cost_summary: dict[str, Any],
    sample_size: int,
) -> Table:
    table = Table(title="Step Breakdown", show_lines=False, pad_edge=False)
    table.add_column("Step", style="bold")
    table.add_column("Model")
    table.add_column("Mean Cost", justify="right")
    table.add_column("p95 Cost", justify="right")
    table.add_column("Mean Tokens", justify="right")
    table.add_column("p95 Tokens", justify="right")
    table.add_column("Calls", justify="right")

    if stats and "step_stats" in stats:
        step_stats = stats["step_stats"]
        sorted_steps = sorted(
            step_stats.items(),
            key=lambda kv: kv[1].get("cost", {}).get("mean", 0),
            reverse=True,
        )
        for step_name, ss in sorted_steps:
            model = ss.get("model", "")
            cost_data = ss.get("cost", {})
            tok_data = ss.get("total_tokens", {})
            table.add_row(
                step_name,
                _truncate_model(model),
                format_cost(cost_data.get("mean", 0)),
                format_cost(cost_data.get("p95", 0)),
                format_tokens(tok_data.get("mean", 0)),
                format_tokens(tok_data.get("p95", 0)),
                str(ss.get("call_count", 0)),
            )
    else:
        per_step = cost_summary.get("per_step", {})
        sorted_steps = sorted(
            per_step.items(),
            key=lambda kv: kv[1].get("cost_mean", 0),
            reverse=True,
        )
        for step_name, ss in sorted_steps:
            step_type = ss.get("step_type", "llm")
            if step_type == "tool":
                table.add_row(
                    step_name,
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    Text("—", style="dim"),
                    str(ss.get("count", 0)),
                )
            else:
                in_mean = ss.get("input_tokens_mean", 0)
                out_mean = ss.get("output_tokens_mean", 0)
                table.add_row(
                    step_name,
                    _truncate_model(ss.get("model", "")),
                    format_cost(ss.get("cost_mean", 0)),
                    format_cost(ss.get("cost_p95", 0)),
                    format_tokens(in_mean + out_mean),
                    Text("—", style="dim"),
                    str(ss.get("count", 0)),
                )

    return table


def _build_projection_panel(
    stats: dict[str, Any] | None,
    cost_summary: dict[str, Any],
    traffic: int | None,
) -> Panel:
    if traffic is not None:
        levels = [traffic]
    else:
        levels = [100, 1000, 10000]

    mean_cost = 0.0
    p95_cost = 0.0
    if stats and "cost_per_run" in stats and stats["cost_per_run"]:
        cpr = stats["cost_per_run"]
        mean_cost = cpr.get("mean", 0)
        p95_cost = cpr.get("p95", 0)
    else:
        mean_cost = cost_summary.get("mean_cost_per_run", 0)
        p95_cost = cost_summary.get("p95_cost_per_run", 0)

    table = Table(title="Monthly Cost Projection", show_lines=True, pad_edge=True)
    table.add_column("Runs/Day", style="bold")
    for lvl in levels:
        table.add_column(f"{lvl:,}", justify="right")

    mean_row = ["Mean/month"]
    p95_row = ["p95/month"]
    for lvl in levels:
        mean_row.append(format_cost(mean_cost * lvl * 30))
        p95_row.append(format_cost(p95_cost * lvl * 30))

    table.add_row(*mean_row)
    table.add_row(*p95_row)

    return Panel(table, expand=False)


def _build_patterns_panel(patterns: list[dict[str, Any]]) -> Panel:
    if not patterns:
        return Panel(
            Text("No non-linear cost patterns detected. "
                 "Linear projection is reliable.", style="green"),
            title="Patterns",
            expand=False,
        )

    text = Text()
    text.append("PATTERNS DETECTED\n\n", style="bold yellow")
    for p in patterns:
        severity = p.get("severity", "warning")
        icon = "🔴" if severity == "danger" else "🟡"
        ptype = p.get("pattern_type", "unknown").replace("_", " ").title()
        step = p.get("step_name", "")
        desc = p.get("description", "")
        text.append(f"{icon} {ptype}", style="bold")
        text.append(f" — Step '{step}': ", style="")
        text.append(f"{desc}\n\n", style="")

    return Panel(text, title="Patterns", expand=False)


def _build_iteration_panel(stats: dict[str, Any] | None) -> Panel | None:
    if not stats or "step_stats" not in stats:
        return None

    iterating_steps = []
    for step_name, ss in stats["step_stats"].items():
        mean_iter = ss.get("mean_iterations", 1.0)
        if mean_iter > 1.0:
            iterating_steps.append((step_name, ss))

    if not iterating_steps:
        return None

    table = Table(
        title="Iteration Counts Per Run", show_lines=True, pad_edge=True,
    )
    table.add_column("Step", style="bold")
    table.add_column("Mean", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("p95", justify="right")

    for step_name, ss in iterating_steps:
        ipr = ss.get("iterations_per_run", {})
        table.add_row(
            step_name,
            f"{ipr.get('mean', 0):.1f}",
            f"{ipr.get('min', 0):.0f}",
            f"{ipr.get('max', 0):.0f}",
            f"{ipr.get('p95', 0):.1f}",
        )

    return Panel(table, expand=False)
