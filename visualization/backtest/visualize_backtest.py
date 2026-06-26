"""Generate 10 backtest visualizations (B1-B10) from backtest result JSON files.

Standalone CLI script. Each function accepts a comparison param ("A", "B", "C", or "all")
and returns list[Path] of saved files.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from pretia.validation.scoring import _COMPARISON_TARGETS  # noqa: E402
from visualization.colors import (  # noqa: E402
    COMPARISON_COLORS,
    DETECTOR_MATRIX_COLORS,
    EXPECTED_DETECTORS,
    WORKFLOW_GROUPS,
    classify_detector_result,
    workflow_color,
)
from visualization.utils import add_caption, ensure_output_dir, save_figure  # noqa: E402


def _flatten_axes(axes: Any) -> list[Any]:
    """Flatten axes from plt.subplots into a flat list regardless of shape."""
    arr = np.array(axes)
    return list(arr.flatten())


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

_DETECTORS = [
    "context_growth",
    "loop_count_variance",
    "high_token_variance",
    "step_count_variance",
    "bimodality",
]

_METRICS = [
    "mean_error_pct",
    "p75_error_pct",
    "ci_coverage_pct",
    "monthly_error_pct",
    "cvar95_error_pct",
]


def _load_workflow_data(path: Path) -> dict[str, Any]:
    """Load a per-workflow backtest JSON file."""
    with open(path) as f:
        return json.load(f)


def _load_all_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every workflow JSON in results_dir."""
    results: list[dict[str, Any]] = []
    if not results_dir.is_dir():
        return results
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = _load_workflow_data(f)
            results.append(data)
        except (json.JSONDecodeError, KeyError):
            continue
    return results


def _filter_comparisons(
    data: dict[str, Any],
    comparison: str,
) -> dict[str, Any]:
    """Return filtered comparisons dict based on the comparison param."""
    comps = data.get("comparisons", {})
    if comparison == "all":
        return comps
    return {k: v for k, v in comps.items() if k == comparison}


def _gaussian_kde_1d(data: list[float], x_grid: np.ndarray) -> np.ndarray:
    """Simple Gaussian KDE without scipy dependency.

    Use Silverman's rule of thumb for bandwidth selection.
    """
    arr = np.array(data)
    n = len(arr)
    if n < 2:
        return np.zeros_like(x_grid)
    std = np.std(arr, ddof=1)
    iqr = np.percentile(arr, 75) - np.percentile(arr, 25)
    # Silverman bandwidth
    h = 0.9 * min(std, iqr / 1.34) * n ** (-0.2)
    if h <= 0:
        h = std * 0.5 if std > 0 else 1.0
    kde = np.zeros_like(x_grid, dtype=float)
    for xi in arr:
        kde += np.exp(-0.5 * ((x_grid - xi) / h) ** 2) / (h * math.sqrt(2 * math.pi))
    kde /= n
    return kde


# ---------------------------------------------------------------------------
# B1: Projected vs Actual KDE
# ---------------------------------------------------------------------------


def b1_projected_vs_actual_kde(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """KDE of projected costs + histogram of actual costs per workflow.

    Subplots in a 4x4 grid. If comparison="all", overlay A/B/C with different colors.
    """
    ensure_output_dir(output_dir)
    n_wf = len(results)
    ncols = 4
    nrows = max(1, math.ceil(n_wf / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_flat = _flatten_axes(axes)

    for idx, wf_data in enumerate(results):
        ax = axes_flat[idx]
        wf_name = wf_data.get("workflow_name", f"W{idx + 1}")
        comps = _filter_comparisons(wf_data, comparison)

        for comp_key, comp_data in sorted(comps.items()):
            profiling_costs = comp_data.get("profiling_run_costs", [])
            gt_costs = comp_data.get("ground_truth_run_costs", [])
            color = COMPARISON_COLORS.get(comp_key, "#333333")

            if gt_costs:
                ax.hist(
                    gt_costs,
                    bins=min(20, max(5, len(gt_costs) // 3)),
                    alpha=0.3,
                    color=color,
                    label=f"{comp_key} actual",
                    density=True,
                )

            if profiling_costs:
                all_costs = profiling_costs + gt_costs
                margin = (max(all_costs) - min(all_costs)) * 0.2 if all_costs else 0.01
                lo = min(all_costs) - margin
                hi = max(all_costs) + margin
                x_grid = np.linspace(lo, hi, 200)
                kde = _gaussian_kde_1d(profiling_costs, x_grid)
                ax.plot(x_grid, kde, color=color, linewidth=1.5, label=f"{comp_key} projected")

        ax.set_title(wf_name, fontsize=10)
        ax.set_xlabel("Cost ($)")
        ax.set_ylabel("Density")
        ax.legend(fontsize=7)

    # Hide unused axes
    for idx in range(n_wf, len(axes_flat)):
        axes_flat[idx].set_visible(False)

    fig.suptitle("B1: Projected vs Actual Cost Distributions", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    add_caption(fig, "KDE of profiling costs overlaid on histogram of ground truth costs.")
    return save_figure(fig, output_dir, "b1_projected_vs_actual_kde")


# ---------------------------------------------------------------------------
# B2: Accuracy Metrics Heatmap
# ---------------------------------------------------------------------------


def b2_accuracy_metrics_heatmap(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "A",
) -> list[Path]:
    """Heatmap of workflows x metrics. Color: green (<target), yellow (near), red (>target)."""
    ensure_output_dir(output_dir)
    comp_keys = ["A", "B", "C"] if comparison == "all" else [comparison]
    all_paths: list[Path] = []

    for comp_key in comp_keys:
        targets = _COMPARISON_TARGETS["no_drift" if comp_key == "A" else "drifted"]
        target_list = [
            targets["mean_error"],
            targets["p75_error"],
            targets["ci_coverage"],
            targets["monthly_error"],
            targets["cvar95_error"],
        ]

        wf_names: list[str] = []
        matrix: list[list[float]] = []
        color_matrix: list[list[str]] = []

        for wf_data in results:
            wf_name = wf_data.get("workflow_name", "?")
            comp_data = wf_data.get("comparisons", {}).get(comp_key)
            if comp_data is None:
                continue
            score = comp_data.get("score", {})
            wf_names.append(wf_name)
            row: list[float] = []
            crow: list[str] = []
            for m_idx, metric in enumerate(_METRICS):
                val = score.get(metric, 0.0)
                row.append(val)
                tgt = target_list[m_idx]
                if metric == "ci_coverage_pct":
                    # Higher is better for CI coverage
                    if val >= tgt:
                        crow.append("#27ae60")  # green
                    elif val >= tgt * 0.9:
                        crow.append("#f1c40f")  # yellow
                    else:
                        crow.append("#e74c3c")  # red
                else:
                    # Lower is better for error metrics
                    if val <= tgt:
                        crow.append("#27ae60")
                    elif val <= tgt * 1.5:
                        crow.append("#f1c40f")
                    else:
                        crow.append("#e74c3c")
            matrix.append(row)
            color_matrix.append(crow)

        if not matrix:
            continue

        n_wf = len(wf_names)
        n_met = len(_METRICS)
        fig, ax = plt.subplots(figsize=(max(8, n_met * 1.5), max(5, n_wf * 0.5)))

        for i in range(n_wf):
            for j in range(n_met):
                ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color_matrix[i][j], alpha=0.7))
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{matrix[i][j]:.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

        ax.set_xlim(0, n_met)
        ax.set_ylim(0, n_wf)
        ax.set_xticks([x + 0.5 for x in range(n_met)])
        ax.set_xticklabels([m.replace("_pct", "%").replace("_", " ") for m in _METRICS], fontsize=8, rotation=30, ha="right")
        ax.set_yticks([y + 0.5 for y in range(n_wf)])
        ax.set_yticklabels(wf_names, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(f"B2: Accuracy Metrics Heatmap (Comparison {comp_key})", fontsize=12)
        fig.tight_layout()
        add_caption(fig, f"Green = passes target, yellow = near target, red = exceeds target. Targets: {'no_drift' if comp_key == 'A' else 'drifted'}.")
        paths = save_figure(fig, output_dir, f"b2_accuracy_heatmap_{comp_key}")
        all_paths.extend(paths)
        plt.close(fig)

    return all_paths


# ---------------------------------------------------------------------------
# B3: Drift Impact Grouped Bars
# ---------------------------------------------------------------------------


def b3_drift_impact_grouped_bars(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """Grouped bar chart: mean_error_pct for A, B, C per workflow."""
    ensure_output_dir(output_dir)
    comp_keys = ["A", "B", "C"] if comparison == "all" else [comparison]

    wf_names: list[str] = []
    bars_data: dict[str, list[float]] = {k: [] for k in comp_keys}

    for wf_data in results:
        wf_name = wf_data.get("workflow_name", "?")
        comps = wf_data.get("comparisons", {})
        has_any = False
        for ck in comp_keys:
            if ck in comps:
                has_any = True
                score = comps[ck].get("score", {})
                bars_data[ck].append(score.get("mean_error_pct", 0.0))
            else:
                bars_data[ck].append(0.0)
        if has_any:
            wf_names.append(wf_name)
        else:
            for ck in comp_keys:
                bars_data[ck].pop()

    if not wf_names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return save_figure(fig, output_dir, "b3_drift_impact_bars")

    n = len(wf_names)
    n_bars = len(comp_keys)
    bar_width = 0.8 / n_bars
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))
    for i, ck in enumerate(comp_keys):
        offset = (i - n_bars / 2 + 0.5) * bar_width
        ax.bar(
            x + offset,
            bars_data[ck],
            bar_width,
            label=f"Comparison {ck}",
            color=COMPARISON_COLORS.get(ck, "#999999"),
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(wf_names, fontsize=9, rotation=45, ha="right")
    ax.set_ylabel("Mean Error %")
    ax.set_title("B3: Drift Impact on Mean Error", fontsize=12)
    ax.legend()
    fig.tight_layout()
    add_caption(fig, "Grouped bars show mean error % for each comparison per workflow.")
    paths = save_figure(fig, output_dir, "b3_drift_impact_bars")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# B4: Reweighting Recovery Scatter
# ---------------------------------------------------------------------------


def b4_reweighting_recovery_scatter(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """Scatter: x=drift impact (B-A error), y=recovery (B-C error).

    Draw quadrant lines at 0. Color by workflow group.
    """
    ensure_output_dir(output_dir)
    xs: list[float] = []
    ys: list[float] = []
    colors: list[str] = []
    labels: list[str] = []

    for wf_data in results:
        wf_name = wf_data.get("workflow_name", "?")
        comps = wf_data.get("comparisons", {})
        a_score = comps.get("A", {}).get("score", {})
        b_score = comps.get("B", {}).get("score", {})
        c_score = comps.get("C", {}).get("score", {})
        a_err = a_score.get("mean_error_pct")
        b_err = b_score.get("mean_error_pct")
        c_err = c_score.get("mean_error_pct")
        if a_err is None or b_err is None or c_err is None:
            continue
        drift_impact = b_err - a_err
        recovery = b_err - c_err
        xs.append(drift_impact)
        ys.append(recovery)
        colors.append(workflow_color(wf_name))
        labels.append(wf_name)

    fig, ax = plt.subplots(figsize=(7, 6))
    if xs:
        ax.scatter(xs, ys, c=colors, s=80, zorder=3, edgecolors="black", linewidths=0.5)
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (xs[i], ys[i]), fontsize=7, textcoords="offset points", xytext=(5, 5))

    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Drift Impact (B - A mean error %)")
    ax.set_ylabel("Reweighting Recovery (B - C mean error %)")
    ax.set_title("B4: Reweighting Recovery Scatter", fontsize=12)
    fig.tight_layout()
    add_caption(fig, "Top-right quadrant = drift hurts but reweighting helps.")
    paths = save_figure(fig, output_dir, "b4_reweighting_recovery_scatter")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# B5: Detector Activation Matrix
# ---------------------------------------------------------------------------


def b5_detector_activation_matrix(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """14 workflows x 5 detectors heatmap with TP/TN/FP/FN classification."""
    ensure_output_dir(output_dir)
    wf_names: list[str] = []
    classifications: list[list[str]] = []

    # Count TP, TN, FP, FN globally
    counts: dict[str, int] = {"TP": 0, "TN": 0, "FP": 0, "FN": 0}

    for wf_data in results:
        wf_name = wf_data.get("workflow_name", "?")
        wf_names.append(wf_name)
        detected = wf_data.get("detected_patterns", [])
        fired_types = {p.get("pattern_type", "") for p in detected}
        row: list[str] = []
        for det in _DETECTORS:
            fired = det in fired_types
            cls = classify_detector_result(wf_name, det, fired)
            row.append(cls)
            counts[cls] += 1
        classifications.append(row)

    n_wf = len(wf_names)
    n_det = len(_DETECTORS)
    fig, ax = plt.subplots(figsize=(max(8, n_det * 1.8), max(5, n_wf * 0.5)))

    for i in range(n_wf):
        for j in range(n_det):
            cls = classifications[i][j]
            color = DETECTOR_MATRIX_COLORS.get(cls, "#cccccc")
            ax.add_patch(plt.Rectangle((j, i), 1, 1, color=color, alpha=0.8))
            ax.text(j + 0.5, i + 0.5, cls, ha="center", va="center", fontsize=8, fontweight="bold")

    ax.set_xlim(0, n_det)
    ax.set_ylim(0, n_wf)
    ax.set_xticks([x + 0.5 for x in range(n_det)])
    ax.set_xticklabels([d.replace("_", "\n") for d in _DETECTORS], fontsize=8)
    ax.set_yticks([y + 0.5 for y in range(n_wf)])
    ax.set_yticklabels(wf_names, fontsize=9)
    ax.invert_yaxis()

    # Annotate TP/FN rates
    total_expected = counts["TP"] + counts["FN"]
    tp_rate = counts["TP"] / total_expected * 100 if total_expected > 0 else 0
    fn_rate = counts["FN"] / total_expected * 100 if total_expected > 0 else 0
    ax.set_title(
        f"B5: Detector Activation Matrix  (TP rate: {tp_rate:.0f}%, FN rate: {fn_rate:.0f}%)",
        fontsize=12,
    )
    fig.tight_layout()
    add_caption(fig, "Green=TP, Gray=TN, Yellow=FP, Red=FN. Rates computed across all workflows.")
    paths = save_figure(fig, output_dir, "b5_detector_activation_matrix")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# B6: Bimodality GMM Overlay
# ---------------------------------------------------------------------------


def b6_bimodality_gmm_overlay(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """Histogram + 2-component Gaussian overlay for bimodal workflows."""
    ensure_output_dir(output_dir)
    all_paths: list[Path] = []

    for wf_data in results:
        wf_name = wf_data.get("workflow_name", "?")
        detected = wf_data.get("detected_patterns", [])
        bimodal_patterns = [p for p in detected if p.get("pattern_type") == "bimodality"]
        if not bimodal_patterns:
            continue

        pattern = bimodal_patterns[0]
        gmm_means = pattern.get("gmm_means")
        gmm_stds = pattern.get("gmm_stds")
        gmm_weights = pattern.get("gmm_weights")
        if not gmm_means or not gmm_stds or not gmm_weights:
            continue

        # Get cost data from any available comparison
        comps = _filter_comparisons(wf_data, comparison)
        all_costs: list[float] = []
        for comp_data in comps.values():
            all_costs.extend(comp_data.get("profiling_run_costs", []))

        if not all_costs:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        # Work in log-space since GMM was fit on log-costs
        positive_costs = [c for c in all_costs if c > 0]
        if not positive_costs:
            plt.close(fig)
            continue

        log_costs = [math.log(c) for c in positive_costs]
        ax.hist(log_costs, bins=min(30, max(5, len(log_costs) // 3)), density=True, alpha=0.5, color="#3498db", label="Observed (log)")

        x_grid = np.linspace(min(log_costs) - 1, max(log_costs) + 1, 300)
        total_pdf = np.zeros_like(x_grid)
        for k in range(len(gmm_means)):
            mu = gmm_means[k]
            sigma = gmm_stds[k]
            w = gmm_weights[k]
            if sigma <= 0:
                sigma = 0.01
            component = w * np.exp(-0.5 * ((x_grid - mu) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
            total_pdf += component
            ax.plot(x_grid, component, linestyle="--", alpha=0.6, label=f"Component {k + 1} (w={w:.2f})")

        ax.plot(x_grid, total_pdf, color="black", linewidth=2, label="GMM total")
        ax.set_xlabel("Log Cost")
        ax.set_ylabel("Density")
        ax.set_title(f"B6: Bimodality GMM Overlay - {wf_name}", fontsize=12)
        ax.legend(fontsize=8)
        fig.tight_layout()
        bic_delta = pattern.get("bimodal_bic_delta", 0)
        add_caption(fig, f"2-component GMM fit on log-costs. BIC delta = {bic_delta:.1f}.")
        paths = save_figure(fig, output_dir, f"b6_gmm_overlay_{wf_name}")
        all_paths.extend(paths)
        plt.close(fig)

    return all_paths


# ---------------------------------------------------------------------------
# B7: CI Coverage Bands
# ---------------------------------------------------------------------------


def b7_ci_coverage_bands(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "A",
) -> list[Path]:
    """Per workflow: shaded CI band + ground truth dots (green=inside, red=outside)."""
    ensure_output_dir(output_dir)
    comp_keys = ["A", "B", "C"] if comparison == "all" else [comparison]
    all_paths: list[Path] = []

    for comp_key in comp_keys:
        n_wf = len(results)
        ncols = 4
        nrows = max(1, math.ceil(n_wf / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = _flatten_axes(axes)

        for idx, wf_data in enumerate(results):
            ax = axes_flat[idx]
            wf_name = wf_data.get("workflow_name", f"W{idx + 1}")
            comp_data = wf_data.get("comparisons", {}).get(comp_key, {})
            ci = comp_data.get("projected_ci")
            gt_costs = comp_data.get("ground_truth_run_costs", [])

            if ci and len(ci) == 2:
                ci_lo, ci_hi = ci
                ax.axhspan(ci_lo, ci_hi, alpha=0.2, color="#3498db", label="Projected CI")
                ax.axhline(ci_lo, color="#3498db", linewidth=0.5, linestyle="--")
                ax.axhline(ci_hi, color="#3498db", linewidth=0.5, linestyle="--")

                inside = 0
                total = len(gt_costs)
                for i, c in enumerate(gt_costs):
                    if ci_lo <= c <= ci_hi:
                        ax.scatter(i, c, color="#27ae60", s=15, zorder=3)
                        inside += 1
                    else:
                        ax.scatter(i, c, color="#e74c3c", s=15, zorder=3)

                coverage = inside / total * 100 if total > 0 else 0
                ax.set_title(f"{wf_name} ({coverage:.0f}%)", fontsize=10)
            else:
                for i, c in enumerate(gt_costs):
                    ax.scatter(i, c, color="#999999", s=15)
                ax.set_title(wf_name, fontsize=10)

            ax.set_xlabel("Run")
            ax.set_ylabel("Cost ($)")

        for idx in range(n_wf, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"B7: CI Coverage Bands (Comparison {comp_key})", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        add_caption(fig, "Green dots = inside CI, red dots = outside CI. Percentage = coverage.")
        paths = save_figure(fig, output_dir, f"b7_ci_coverage_bands_{comp_key}")
        all_paths.extend(paths)
        plt.close(fig)

    return all_paths


# ---------------------------------------------------------------------------
# B8: CVaR Comparison Bars
# ---------------------------------------------------------------------------


def b8_cvar_comparison_bars(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "A",
) -> list[Path]:
    """Grouped bars: projected vs actual CVaR95 per workflow."""
    ensure_output_dir(output_dir)
    comp_keys = ["A", "B", "C"] if comparison == "all" else [comparison]
    all_paths: list[Path] = []

    for comp_key in comp_keys:
        wf_names: list[str] = []
        projected_cvars: list[float] = []
        actual_cvars: list[float] = []

        for wf_data in results:
            wf_name = wf_data.get("workflow_name", "?")
            comp_data = wf_data.get("comparisons", {}).get(comp_key, {})
            p_cvar = comp_data.get("projected_cvar95")
            a_cvar = comp_data.get("actual_cvar95")
            if p_cvar is None or a_cvar is None:
                continue
            wf_names.append(wf_name)
            projected_cvars.append(p_cvar)
            actual_cvars.append(a_cvar)

        if not wf_names:
            continue

        n = len(wf_names)
        x = np.arange(n)
        bar_width = 0.35

        fig, ax = plt.subplots(figsize=(max(8, n * 0.8), 5))
        ax.bar(x - bar_width / 2, projected_cvars, bar_width, label="Projected CVaR95", color="#3498db", alpha=0.85)
        ax.bar(x + bar_width / 2, actual_cvars, bar_width, label="Actual CVaR95", color="#e74c3c", alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(wf_names, fontsize=9, rotation=45, ha="right")
        ax.set_ylabel("CVaR95 ($)")
        ax.set_title(f"B8: CVaR95 Comparison (Comparison {comp_key})", fontsize=12)
        ax.legend()
        fig.tight_layout()
        add_caption(fig, "Projected vs actual conditional value-at-risk at the 95th percentile.")
        paths = save_figure(fig, output_dir, f"b8_cvar_comparison_{comp_key}")
        all_paths.extend(paths)
        plt.close(fig)

    return all_paths


# ---------------------------------------------------------------------------
# B9: Per-Step Cost Breakdown
# ---------------------------------------------------------------------------


def b9_per_step_cost_breakdown(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """Stacked horizontal bars of step cost contributions."""
    ensure_output_dir(output_dir)
    wf_names: list[str] = []
    all_step_costs: list[dict[str, float]] = []

    for wf_data in results:
        wf_name = wf_data.get("workflow_name", "?")
        step_costs = wf_data.get("step_costs", {})
        if not step_costs:
            continue
        wf_names.append(wf_name)
        all_step_costs.append(step_costs)

    if not wf_names:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No step cost data", ha="center", va="center")
        return save_figure(fig, output_dir, "b9_per_step_cost_breakdown")

    # Collect all step names
    all_steps: list[str] = []
    seen: set[str] = set()
    for sc in all_step_costs:
        for s in sc:
            if s not in seen:
                all_steps.append(s)
                seen.add(s)

    # Assign colors to steps
    cmap = matplotlib.colormaps.get_cmap("tab20").resampled(max(1, len(all_steps)))
    step_colors = {s: cmap(i) for i, s in enumerate(all_steps)}

    n = len(wf_names)
    fig, ax = plt.subplots(figsize=(10, max(4, n * 0.5)))
    y = np.arange(n)

    lefts = np.zeros(n)
    for step_name in all_steps:
        widths = [sc.get(step_name, 0.0) for sc in all_step_costs]
        ax.barh(y, widths, left=lefts, height=0.6, label=step_name, color=step_colors[step_name])
        lefts += np.array(widths)

    ax.set_yticks(y)
    ax.set_yticklabels(wf_names, fontsize=9)
    ax.set_xlabel("Cost ($)")
    ax.set_title("B9: Per-Step Cost Breakdown", fontsize=12)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    fig.tight_layout()
    add_caption(fig, "Stacked horizontal bars show the cost contribution of each step.")
    paths = save_figure(fig, output_dir, "b9_per_step_cost_breakdown")
    plt.close(fig)
    return paths


# ---------------------------------------------------------------------------
# B10: Token Distribution Comparison
# ---------------------------------------------------------------------------


def b10_token_distribution_comparison(
    results: list[dict[str, Any]],
    output_dir: Path,
    comparison: str = "A",
) -> list[Path]:
    """Overlaid histograms: profiling (blue) vs ground truth (orange) input tokens."""
    ensure_output_dir(output_dir)
    comp_keys = ["A", "B", "C"] if comparison == "all" else [comparison]
    all_paths: list[Path] = []

    for comp_key in comp_keys:
        n_wf = len(results)
        ncols = 4
        nrows = max(1, math.ceil(n_wf / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes_flat = _flatten_axes(axes)

        for idx, wf_data in enumerate(results):
            ax = axes_flat[idx]
            wf_name = wf_data.get("workflow_name", f"W{idx + 1}")
            comp_data = wf_data.get("comparisons", {}).get(comp_key, {})
            profiling_costs = comp_data.get("profiling_run_costs", [])
            gt_costs = comp_data.get("ground_truth_run_costs", [])

            if profiling_costs:
                ax.hist(
                    profiling_costs,
                    bins=min(20, max(5, len(profiling_costs) // 3)),
                    alpha=0.5,
                    color="#3498db",
                    label="Profiling",
                    density=True,
                )
            if gt_costs:
                ax.hist(
                    gt_costs,
                    bins=min(20, max(5, len(gt_costs) // 3)),
                    alpha=0.5,
                    color="#e67e22",
                    label="Ground Truth",
                    density=True,
                )

            ax.set_title(wf_name, fontsize=10)
            ax.set_xlabel("Cost ($)")
            ax.set_ylabel("Density")
            ax.legend(fontsize=7)

        for idx in range(n_wf, len(axes_flat)):
            axes_flat[idx].set_visible(False)

        fig.suptitle(f"B10: Token Distribution Comparison (Comparison {comp_key})", fontsize=14)
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        add_caption(fig, "Profiling (blue) vs ground truth (orange) cost distributions.")
        paths = save_figure(fig, output_dir, f"b10_token_distribution_{comp_key}")
        all_paths.extend(paths)
        plt.close(fig)

    return all_paths


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def generate_all_backtest_visuals(
    results_dir: Path,
    output_dir: Path,
    comparison: str = "all",
) -> list[Path]:
    """Generate all 10 backtest visualizations. Returns all saved file paths."""
    results = _load_all_results(results_dir)
    if not results:
        return []

    all_paths: list[Path] = []
    all_paths.extend(b1_projected_vs_actual_kde(results, output_dir, comparison))
    all_paths.extend(b2_accuracy_metrics_heatmap(results, output_dir, comparison))
    all_paths.extend(b3_drift_impact_grouped_bars(results, output_dir, comparison))
    all_paths.extend(b4_reweighting_recovery_scatter(results, output_dir, comparison))
    all_paths.extend(b5_detector_activation_matrix(results, output_dir, comparison))
    all_paths.extend(b6_bimodality_gmm_overlay(results, output_dir, comparison))
    all_paths.extend(b7_ci_coverage_bands(results, output_dir, comparison))
    all_paths.extend(b8_cvar_comparison_bars(results, output_dir, comparison))
    all_paths.extend(b9_per_step_cost_breakdown(results, output_dir, comparison))
    all_paths.extend(b10_token_distribution_comparison(results, output_dir, comparison))

    plt.close("all")
    return all_paths


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("Usage: python visualize_backtest.py <results_dir> <output_dir> [comparison]")
        sys.exit(1)

    results_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    comparison = sys.argv[3] if len(sys.argv) > 3 else "all"
    paths = generate_all_backtest_visuals(results_dir, output_dir, comparison)
    print(f"Generated {len(paths)} files in {output_dir}")
    for p in paths:
        print(f"  {p}")
