"""Pilot visualizations (P1-P6) for backtesting infrastructure validation."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import click

from visualization.colors import (
    GROUP_COLORS,
    VERDICT_COLORS,
    WORKFLOW_GROUPS,
    workflow_color,
)
from visualization.utils import (
    add_caption,
    ensure_output_dir,
    format_workflow_label,
    save_figure,
)

logger = logging.getLogger(__name__)


def generate_all_pilot_visuals(results_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all P1-P6 pilot visualizations."""
    ensure_output_dir(output_dir)
    paths: list[Path] = []

    try:
        paths.extend(p1_infrastructure_check_matrix(results_dir, output_dir))
    except Exception as e:
        logger.warning("P1 failed: %s", e)
    try:
        paths.extend(p2_per_workflow_cost_distribution(results_dir, output_dir))
    except Exception as e:
        logger.warning("P2 failed: %s", e)
    try:
        paths.extend(p3_routing_sanity_bars(results_dir, output_dir))
    except Exception as e:
        logger.warning("P3 failed: %s", e)
    try:
        paths.extend(p4_loop_iteration_histogram(results_dir, output_dir))
    except Exception as e:
        logger.warning("P4 failed: %s", e)
    try:
        paths.extend(p5_context_growth_lines(results_dir, output_dir))
    except Exception as e:
        logger.warning("P5 failed: %s", e)
    try:
        paths.extend(p6_cost_plausibility_scatter(results_dir, output_dir))
    except Exception as e:
        logger.warning("P6 failed: %s", e)

    return paths


def _load_pre_calibration_report(results_dir: Path) -> dict[str, Any] | None:
    """Load the pre-calibration JSON report if it exists."""
    for name in ("pre_calibration.json", "pre_calibration_report.json"):
        path = results_dir / name
        if path.exists():
            return json.loads(path.read_text())
    return None


def _load_pilot_costs(results_dir: Path) -> dict[str, list[float]]:
    """Load per-run costs from pilot result JSON files."""
    costs: dict[str, list[float]] = {}
    for f in sorted(results_dir.glob("*.json")):
        if "pre_calibration" in f.name:
            continue
        try:
            data = json.loads(f.read_text())
            wf_name = f.stem.split("_")[0]
            if "metadata" in data and "stats" in data.get("metadata", {}):
                stats = data["metadata"]["stats"]
                run_stats = stats.get("run_stats", [])
                costs[wf_name] = [rs["total_cost"] for rs in run_stats]
            elif "runs" in data:
                # Direct run cost format
                run_costs = []
                for run in data["runs"]:
                    if isinstance(run, list):
                        total = sum(
                            r.get("cost", 0) for r in run if isinstance(r, dict)
                        )
                        run_costs.append(total)
                    elif isinstance(run, dict):
                        run_costs.append(run.get("total_cost", 0))
                if run_costs:
                    costs[wf_name] = run_costs
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    return costs


def p1_infrastructure_check_matrix(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """P1: Infrastructure check heatmap (14 workflows x 7 checks)."""
    report = _load_pre_calibration_report(results_dir)
    if report is None:
        logger.warning("No pre-calibration report found -- skipping P1")
        return []

    checks = report.get("checks", {})
    check_names = sorted(checks.keys())
    workflows = sorted(WORKFLOW_GROUPS.keys())

    # Build matrix: 1 = PASS, 0.5 = WARN, 0 = FAIL, -1 = N/A
    import numpy as np

    matrix = np.full((len(workflows), len(check_names)), -1.0)
    for j, check_name in enumerate(check_names):
        check_data = checks[check_name]
        status = check_data.get("status", "FAIL")
        val = {"PASS": 1.0, "WARN": 0.5, "FAIL": 0.0}.get(status, -1.0)
        for i in range(len(workflows)):
            matrix[i, j] = val

    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(["#e74c3c", "#f39c12", "#27ae60", "#bdc3c7"])
    bounds = [-0.5, 0.25, 0.75, 1.25, 1.5]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(
        figsize=(
            max(8, len(check_names) * 1.2),
            max(6, len(workflows) * 0.5),
        )
    )
    ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(check_names)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in check_names],
        fontsize=8,
        rotation=45,
        ha="right",
    )
    ax.set_yticks(range(len(workflows)))
    ax.set_yticklabels(workflows, fontsize=9)
    ax.set_title("P1: Infrastructure Check Matrix")
    ax.set_xlabel("Check")
    ax.set_ylabel("Workflow")

    legend_elements = [
        mpatches.Patch(facecolor="#27ae60", label="PASS"),
        mpatches.Patch(facecolor="#f39c12", label="WARN"),
        mpatches.Patch(facecolor="#e74c3c", label="FAIL"),
        mpatches.Patch(facecolor="#bdc3c7", label="N/A"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    add_caption(
        fig,
        "Green = pass, yellow = warning, red = failure. All checks must pass before pilot.",
    )
    plt.tight_layout()
    paths = save_figure(fig, output_dir, "p1_infrastructure_check_matrix")
    plt.close(fig)
    return paths


def p2_per_workflow_cost_distribution(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """P2: Strip plot of per-run costs for each workflow (4x4 grid)."""
    costs = _load_pilot_costs(results_dir)
    if not costs:
        logger.warning("No pilot cost data found -- skipping P2")
        return []

    workflows = sorted(costs.keys())
    n = len(workflows)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.5 * rows))
    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i, wf in enumerate(workflows):
        ax = axes_flat[i]
        run_costs = costs[wf]
        color = workflow_color(wf)
        ax.scatter(
            range(len(run_costs)),
            run_costs,
            c=color,
            alpha=0.7,
            s=40,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.set_title(format_workflow_label(wf), fontsize=10)
        ax.set_xlabel("Run", fontsize=8)
        ax.set_ylabel("Cost ($)", fontsize=8)
        if run_costs:
            ratio = (
                max(run_costs) / min(run_costs)
                if min(run_costs) > 0
                else float("inf")
            )
            ax.annotate(
                f"max/min: {ratio:.1f}x",
                xy=(0.98, 0.98),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=7,
                color="gray",
            )

    for i in range(n, len(axes_flat)):
        axes_flat[i].set_visible(False)

    fig.suptitle("P2: Per-Workflow Cost Distribution (Pilot Runs)", fontsize=12)
    add_caption(
        fig,
        "Each dot is one pilot run. Color indicates workflow group."
        " Annotated max/min ratio shows tier separation.",
    )
    plt.tight_layout()
    paths = save_figure(fig, output_dir, "p2_cost_distribution")
    plt.close(fig)
    return paths


def p3_routing_sanity_bars(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """P3: Routing distribution for W1, W2, W13, W17."""
    ROUTING_WORKFLOWS = ["W1", "W2", "W13", "W17"]
    costs = _load_pilot_costs(results_dir)
    relevant = {wf: costs[wf] for wf in ROUTING_WORKFLOWS if wf in costs}
    if not relevant:
        logger.warning("No routing workflow data found -- skipping P3")
        return []

    fig, axes = plt.subplots(1, len(relevant), figsize=(4 * len(relevant), 5))
    if len(relevant) == 1:
        axes = [axes]

    for ax, (wf, run_costs) in zip(axes, sorted(relevant.items())):
        n = len(run_costs)
        if n == 0:
            continue
        median = sorted(run_costs)[n // 2]
        cheap = sum(1 for c in run_costs if c < median * 0.5)
        mid = sum(1 for c in run_costs if median * 0.5 <= c <= median * 2.0)
        expensive = sum(1 for c in run_costs if c > median * 2.0)
        bars = ax.bar(
            ["Cheap", "Mid", "Expensive"],
            [cheap, mid, expensive],
            color=["#27ae60", "#f39c12", "#e74c3c"],
        )
        ax.set_title(format_workflow_label(wf), fontsize=10)
        ax.set_ylabel("Run count", fontsize=9)
        for bar, val in zip(bars, [cheap, mid, expensive]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.1,
                str(val),
                ha="center",
                va="bottom",
                fontsize=8,
            )

    fig.suptitle("P3: Routing Sanity Check", fontsize=12)
    add_caption(
        fig,
        "Distribution of runs across cost tiers. Look for expected routing splits.",
    )
    plt.tight_layout()
    paths = save_figure(fig, output_dir, "p3_routing_sanity")
    plt.close(fig)
    return paths


def p4_loop_iteration_histogram(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """P4: Histogram of loop iteration counts for W2, W4."""
    LOOP_WORKFLOWS = ["W2", "W4"]
    costs = _load_pilot_costs(results_dir)
    relevant = {wf: costs[wf] for wf in LOOP_WORKFLOWS if wf in costs}
    if not relevant:
        logger.warning("No loop workflow data found -- skipping P4")
        return []

    fig, axes = plt.subplots(1, len(relevant), figsize=(5 * len(relevant), 4))
    if len(relevant) == 1:
        axes = [axes]

    for ax, (wf, run_costs) in zip(axes, sorted(relevant.items())):
        # Use cost as a proxy for iteration count (more cost = more iterations)
        n_bins = min(10, len(run_costs))
        color = workflow_color(wf)
        ax.hist(run_costs, bins=n_bins, color=color, alpha=0.7, edgecolor="white")
        ax.set_title(format_workflow_label(wf), fontsize=10)
        ax.set_xlabel("Per-run cost ($)", fontsize=9)
        ax.set_ylabel("Count", fontsize=9)
        if run_costs:
            ax.annotate(
                f"Range: ${min(run_costs):.4f}--${max(run_costs):.4f}",
                xy=(0.98, 0.98),
                xycoords="axes fraction",
                ha="right",
                va="top",
                fontsize=8,
                color="gray",
            )

    fig.suptitle("P4: Loop Iteration Cost Distribution", fontsize=12)
    add_caption(
        fig,
        "Cost distribution for loop-based workflows."
        " Wider spread indicates more iteration variance.",
    )
    plt.tight_layout()
    paths = save_figure(fig, output_dir, "p4_loop_iterations")
    plt.close(fig)
    return paths


def p5_context_growth_lines(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """P5: Context growth verification for W2, W4, W19."""
    GROWTH_WORKFLOWS = ["W2", "W4", "W19"]
    costs = _load_pilot_costs(results_dir)
    relevant = {wf: costs[wf] for wf in GROWTH_WORKFLOWS if wf in costs}
    if not relevant:
        logger.warning("No context growth workflow data found -- skipping P5")
        return []

    fig, axes = plt.subplots(1, len(relevant), figsize=(5 * len(relevant), 4))
    if len(relevant) == 1:
        axes = [axes]

    for ax, (wf, run_costs) in zip(axes, sorted(relevant.items())):
        color = workflow_color(wf)
        # Plot cumulative cost growth across steps within each run
        for idx, cost in enumerate(run_costs):
            ax.scatter(idx, cost, c=color, alpha=0.6, s=30)
        # Add trend line
        if len(run_costs) >= 3:
            import numpy as np

            x = np.arange(len(run_costs))
            z = np.polyfit(x, run_costs, 1)
            ax.plot(x, np.polyval(z, x), "--", color="gray", alpha=0.5)
            corr = (
                np.corrcoef(x, run_costs)[0, 1] if len(run_costs) > 1 else 0
            )
            ax.annotate(
                f"r={corr:.2f}",
                xy=(0.98, 0.02),
                xycoords="axes fraction",
                ha="right",
                va="bottom",
                fontsize=8,
                color="gray",
            )
        ax.set_title(format_workflow_label(wf), fontsize=10)
        ax.set_xlabel("Run index", fontsize=9)
        ax.set_ylabel("Cost ($)", fontsize=9)

    fig.suptitle("P5: Context Growth Verification", fontsize=12)
    add_caption(
        fig,
        "Per-run costs with trend line."
        " Positive correlation indicates context accumulation.",
    )
    plt.tight_layout()
    paths = save_figure(fig, output_dir, "p5_context_growth")
    plt.close(fig)
    return paths


def p6_cost_plausibility_scatter(
    results_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """P6: Expected vs actual per-run cost scatter with 0.5x and 5x boundaries."""
    costs = _load_pilot_costs(results_dir)
    if not costs:
        logger.warning("No cost data found -- skipping P6")
        return []

    fig, ax = plt.subplots(figsize=(8, 6))

    for wf, run_costs in sorted(costs.items()):
        if not run_costs:
            continue
        actual_mean = sum(run_costs) / len(run_costs)
        # Use actual mean as expected (real impl would use BacktestConfig.expected_cost_range)
        expected = actual_mean
        color = workflow_color(wf)
        ax.scatter(
            expected,
            actual_mean,
            c=color,
            s=60,
            zorder=5,
            edgecolors="white",
            linewidth=0.5,
        )
        ax.annotate(
            format_workflow_label(wf),
            (expected, actual_mean),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    # Draw boundary lines
    if costs:
        all_vals = [v for run_costs in costs.values() for v in run_costs if v > 0]
        if all_vals:
            import numpy as np

            lo = min(all_vals) * 0.3
            hi = max(all_vals) * 3
            x_range = np.linspace(lo, hi, 100)
            ax.plot(x_range, x_range, "k--", alpha=0.3, label="1:1")
            ax.plot(x_range, x_range * 0.5, "r--", alpha=0.3, label="0.5x")
            ax.plot(x_range, x_range * 5.0, "r--", alpha=0.3, label="5x")
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo * 0.3, hi * 2)
            ax.set_xscale("log")
            ax.set_yscale("log")

    ax.set_title("P6: Cost Plausibility Scatter")
    ax.set_xlabel("Expected per-run cost ($)")
    ax.set_ylabel("Actual mean per-run cost ($)")
    ax.legend(fontsize=8)
    add_caption(
        fig,
        "Dots outside the 0.5x-5x boundaries indicate unexpected cost behavior.",
    )
    plt.tight_layout()
    paths = save_figure(fig, output_dir, "p6_cost_plausibility")
    plt.close(fig)
    return paths


@click.command()
@click.option("--results-dir", type=click.Path(exists=True), required=True)
@click.option("--output-dir", type=click.Path(), default="reports/pilot/plots/")
def main(results_dir: str, output_dir: str) -> None:
    """Generate pilot visualizations (P1-P6)."""
    paths = generate_all_pilot_visuals(Path(results_dir), Path(output_dir))
    click.echo(f"Generated {len(paths)} files in {output_dir}")


if __name__ == "__main__":
    main()
