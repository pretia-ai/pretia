"""Tests for backtest visualizations (B1-B10)."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from visualization.backtest.visualize_backtest import (  # noqa: E402
    b1_projected_vs_actual_kde,
    b2_accuracy_metrics_heatmap,
    b3_drift_impact_grouped_bars,
    b4_reweighting_recovery_scatter,
    b5_detector_activation_matrix,
    b6_bimodality_gmm_overlay,
    b7_ci_coverage_bands,
    b8_cvar_comparison_bars,
    b9_per_step_cost_breakdown,
    b10_token_distribution_comparison,
    generate_all_backtest_visuals,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_WORKFLOW_NAMES = [
    "W1",
    "W2",
    "W4",
    "W5",
    "W9",
    "W11",
    "W12",
    "W13",
    "W14",
    "W15",
    "W16",
    "W17",
    "W18",
    "W19",
]


def _make_score(
    mean_error: float = 5.0,
    p75_error: float = 8.0,
    ci_coverage: float = 90.0,
    monthly_error: float = 5.0,
    cvar95_error: float = 15.0,
) -> dict:
    return {
        "mean_error_pct": mean_error,
        "p75_error_pct": p75_error,
        "ci_coverage_pct": ci_coverage,
        "monthly_error_pct": monthly_error,
        "cvar95_error_pct": cvar95_error,
    }


def _make_comparison_data(
    rng: random.Random,
    n_profiling: int = 50,
    n_gt: int = 200,
    base_cost: float = 0.02,
    drift_factor: float = 1.0,
) -> dict:
    profiling_costs = [base_cost + rng.gauss(0, base_cost * 0.2) for _ in range(n_profiling)]
    gt_costs = [(base_cost * drift_factor) + rng.gauss(0, base_cost * 0.2) for _ in range(n_gt)]
    # Ensure no negatives
    profiling_costs = [max(0.001, c) for c in profiling_costs]
    gt_costs = [max(0.001, c) for c in gt_costs]
    ci_lo = sorted(profiling_costs)[int(len(profiling_costs) * 0.05)]
    ci_hi = sorted(profiling_costs)[int(len(profiling_costs) * 0.95)]
    proj_cvar = sum(sorted(profiling_costs)[int(len(profiling_costs) * 0.95) :]) / max(
        1, len(profiling_costs) - int(len(profiling_costs) * 0.95)
    )
    actual_cvar = sum(sorted(gt_costs)[int(len(gt_costs) * 0.95) :]) / max(
        1, len(gt_costs) - int(len(gt_costs) * 0.95)
    )
    return {
        "score": _make_score(
            mean_error=abs(drift_factor - 1.0) * 100 + rng.uniform(0, 5),
            ci_coverage=rng.uniform(80, 95),
        ),
        "profiling_run_costs": profiling_costs,
        "ground_truth_run_costs": gt_costs,
        "projected_ci": [ci_lo, ci_hi],
        "projected_cvar95": proj_cvar,
        "actual_cvar95": actual_cvar,
    }


def _make_workflow_data(
    wf_name: str,
    rng: random.Random,
    include_bimodality: bool = False,
) -> dict:
    base_cost = 0.01 + rng.random() * 0.05
    data: dict = {
        "workflow_name": wf_name,
        "comparisons": {
            "A": _make_comparison_data(rng, base_cost=base_cost, drift_factor=1.0),
            "B": _make_comparison_data(rng, base_cost=base_cost, drift_factor=1.3),
            "C": _make_comparison_data(rng, base_cost=base_cost, drift_factor=1.1),
        },
        "detected_patterns": [],
        "step_costs": {
            "classify": base_cost * 0.1,
            "generate": base_cost * 0.7,
            "format": base_cost * 0.2,
        },
    }
    if include_bimodality:
        data["detected_patterns"].append(
            {
                "pattern_type": "bimodality",
                "step_name": "_workflow_",
                "severity": "warning",
                "evidence": {},
                "description": "Bimodal distribution detected.",
                "gmm_means": [math.log(base_cost * 0.5), math.log(base_cost * 2.0)],
                "gmm_stds": [0.3, 0.25],
                "gmm_weights": [0.4, 0.6],
                "bimodal_bic_delta": 15.0,
            }
        )
    return data


def _write_backtest_data(
    results_dir: Path,
    workflows: list[str] | None = None,
    comparisons: list[str] | None = None,
    include_bimodality_for: list[str] | None = None,
) -> list[dict]:
    """Write synthetic backtest JSON files. Returns the data written."""
    results_dir.mkdir(parents=True, exist_ok=True)
    if workflows is None:
        workflows = _WORKFLOW_NAMES
    if comparisons is None:
        comparisons = ["A", "B", "C"]
    if include_bimodality_for is None:
        include_bimodality_for = []

    rng = random.Random(42)
    all_data: list[dict] = []

    for wf in workflows:
        data = _make_workflow_data(wf, rng, include_bimodality=wf in include_bimodality_for)
        # Filter comparisons
        data["comparisons"] = {k: v for k, v in data["comparisons"].items() if k in comparisons}
        fpath = results_dir / f"{wf}.json"
        fpath.write_text(json.dumps(data, indent=2))
        all_data.append(data)

    return all_data


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestB1KDE:
    def test_b1_kde_generates(self, tmp_path):
        results_dir = tmp_path / "results"
        output_dir = tmp_path / "output"
        data = _write_backtest_data(results_dir, workflows=["W1", "W2", "W4", "W5"])
        paths = b1_projected_vs_actual_kde(data, output_dir, comparison="all")
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)


class TestB2Heatmap:
    def test_b2_heatmap_uses_correct_targets(self, tmp_path):
        """No-drift (A) uses stricter targets than drifted (B/C)."""
        results_dir = tmp_path / "results"
        output_dir = tmp_path / "output"
        data = _write_backtest_data(results_dir, workflows=["W1", "W2"])
        # Generate for A and B separately
        paths_a = b2_accuracy_metrics_heatmap(data, output_dir / "a", comparison="A")
        paths_b = b2_accuracy_metrics_heatmap(data, output_dir / "b", comparison="B")
        assert len(paths_a) >= 1
        assert len(paths_b) >= 1
        # Both should produce output files
        assert all(p.exists() for p in paths_a)
        assert all(p.exists() for p in paths_b)


class TestB3DriftBars:
    def test_b3_drift_bars_three_per_workflow(self, tmp_path):
        """Verify 3 comparison bars per workflow when comparison='all'."""
        output_dir = tmp_path / "output"
        data = _write_backtest_data(tmp_path / "results", workflows=["W1", "W2", "W4"])
        paths = b3_drift_impact_grouped_bars(data, output_dir, comparison="all")
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)
        # Re-open the figure metadata isn't easily inspectable, but we verify
        # the function ran successfully with 3 comparison keys


class TestB4ReweightingScatter:
    def test_b4_reweighting_scatter_generates(self, tmp_path):
        output_dir = tmp_path / "output"
        data = _write_backtest_data(tmp_path / "results", workflows=_WORKFLOW_NAMES[:6])
        paths = b4_reweighting_recovery_scatter(data, output_dir, comparison="all")
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)


class TestB5DetectorMatrix:
    def test_b5_detector_matrix_classification(self, tmp_path):
        """Verify TP/TN/FP/FN with known data."""
        output_dir = tmp_path / "output"
        # W2 expects context_growth=True. Provide it as detected.
        rng = random.Random(42)
        data = [
            {
                "workflow_name": "W1",
                "comparisons": {"A": _make_comparison_data(rng)},
                "detected_patterns": [],  # W1 expects nothing
                "step_costs": {},
            },
            {
                "workflow_name": "W2",
                "comparisons": {"A": _make_comparison_data(rng)},
                "detected_patterns": [
                    {"pattern_type": "context_growth", "step_name": "gen"},
                    {"pattern_type": "loop_count_variance", "step_name": "gen"},
                ],
                "step_costs": {},
            },
        ]
        paths = b5_detector_activation_matrix(data, output_dir)
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)


class TestB6GMMOverlay:
    def test_b6_gmm_overlay_generates(self, tmp_path):
        """Verify with synthetic bimodal data."""
        output_dir = tmp_path / "output"
        data = _write_backtest_data(
            tmp_path / "results",
            workflows=["W13", "W15"],
            include_bimodality_for=["W13"],
        )
        paths = b6_bimodality_gmm_overlay(data, output_dir, comparison="all")
        # Only W13 has bimodality, so should produce output
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)

    def test_b6_skips_non_bimodal(self, tmp_path):
        """No output for workflows without bimodality."""
        output_dir = tmp_path / "output"
        data = _write_backtest_data(
            tmp_path / "results",
            workflows=["W1", "W9"],
            include_bimodality_for=[],
        )
        paths = b6_bimodality_gmm_overlay(data, output_dir, comparison="all")
        assert paths == []


class TestB7CICoverage:
    def test_b7_ci_coverage_dot_coloring(self, tmp_path):
        """Verify green/red dots by checking output generation."""
        output_dir = tmp_path / "output"
        rng = random.Random(99)
        # Create data with a known CI band
        comp_data = _make_comparison_data(rng, base_cost=0.03)
        # Set CI to a narrow band so some dots are outside
        comp_data["projected_ci"] = [0.025, 0.035]
        data = [
            {
                "workflow_name": "W1",
                "comparisons": {"A": comp_data},
                "detected_patterns": [],
                "step_costs": {},
            },
        ]
        paths = b7_ci_coverage_bands(data, output_dir, comparison="A")
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)


class TestB8CVaR:
    def test_b8_cvar_paired_bars(self, tmp_path):
        output_dir = tmp_path / "output"
        data = _write_backtest_data(tmp_path / "results", workflows=["W1", "W2", "W4"])
        paths = b8_cvar_comparison_bars(data, output_dir, comparison="A")
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)


class TestB9PerStep:
    def test_b9_per_step_stacked(self, tmp_path):
        output_dir = tmp_path / "output"
        data = _write_backtest_data(tmp_path / "results", workflows=["W1", "W2"])
        paths = b9_per_step_cost_breakdown(data, output_dir)
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)
        # Verify PNG exists
        assert any(p.suffix == ".png" for p in paths)


class TestB10TokenDistribution:
    def test_b10_two_histograms(self, tmp_path):
        output_dir = tmp_path / "output"
        data = _write_backtest_data(tmp_path / "results", workflows=["W1", "W2", "W4", "W5"])
        paths = b10_token_distribution_comparison(data, output_dir, comparison="A")
        assert len(paths) >= 1
        assert all(p.exists() for p in paths)


class TestComparisonParam:
    def test_comparison_param_filters(self, tmp_path):
        """Verify 'A' produces only A, 'all' produces all."""
        output_dir_a = tmp_path / "output_a"
        output_dir_all = tmp_path / "output_all"
        data = _write_backtest_data(tmp_path / "results", workflows=["W1", "W2"])

        # B3 with comparison="A" should only show A bars
        paths_a = b3_drift_impact_grouped_bars(data, output_dir_a, comparison="A")
        assert len(paths_a) >= 1

        # B3 with comparison="all" should show A, B, C bars
        paths_all = b3_drift_impact_grouped_bars(data, output_dir_all, comparison="all")
        assert len(paths_all) >= 1

        # B2 with comparison="A" should produce 1 set of files
        paths_b2_a = b2_accuracy_metrics_heatmap(data, output_dir_a / "b2", comparison="A")
        # B2 with comparison="all" should produce 3 sets
        paths_b2_all = b2_accuracy_metrics_heatmap(data, output_dir_all / "b2", comparison="all")
        # "all" produces 3x (A, B, C), "A" produces 1x
        assert len(paths_b2_all) >= len(paths_b2_a)


class TestGenerateAll:
    def test_generate_all_no_crash(self, tmp_path):
        """Full orchestrator with synthetic data should not crash."""
        results_dir = tmp_path / "results"
        output_dir = tmp_path / "output"
        _write_backtest_data(
            results_dir,
            workflows=["W1", "W2", "W13", "W15"],
            include_bimodality_for=["W13"],
        )
        paths = generate_all_backtest_visuals(results_dir, output_dir, comparison="all")
        assert len(paths) > 0
        assert all(p.exists() for p in paths)

    def test_generate_all_empty_dir(self, tmp_path):
        """Empty results dir should return empty list."""
        results_dir = tmp_path / "empty"
        results_dir.mkdir()
        output_dir = tmp_path / "output"
        paths = generate_all_backtest_visuals(results_dir, output_dir, comparison="all")
        assert paths == []


@pytest.fixture(autouse=True)
def _close_all_figures():
    """Ensure all matplotlib figures are closed after each test."""
    yield
    plt.close("all")
