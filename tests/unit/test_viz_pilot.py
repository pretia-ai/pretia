"""Tests for pilot visualizations (P1-P6)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

matplotlib = pytest.importorskip("matplotlib")

matplotlib.use("Agg")

from visualization.pilot.visualize_pilot import (  # noqa: E402
    generate_all_pilot_visuals,
    p1_infrastructure_check_matrix,
    p2_per_workflow_cost_distribution,
    p3_routing_sanity_bars,
    p4_loop_iteration_histogram,
    p5_context_growth_lines,
    p6_cost_plausibility_scatter,
)


def _write_pre_cal_report(results_dir: Path) -> None:
    report = {
        "timestamp": "2026-06-07",
        "checks": {
            "model_availability": {"status": "PASS", "details": {}},
            "pricing_consistency": {"status": "WARN", "details": {}},
            "collector_schema": {"status": "PASS", "details": {}},
            "engine_config": {"status": "PASS", "details": {}},
            "prompt_inventory": {"status": "PASS", "details": {}},
            "input_inventory": {"status": "FAIL", "details": {}},
            "pdf_inventory": {"status": "PASS", "details": {}},
        },
        "blocking_failures": ["input_inventory"],
        "warnings": ["pricing_consistency"],
        "proceed_to_pilot": False,
    }
    (results_dir / "pre_calibration.json").write_text(json.dumps(report))


def _write_pilot_data(results_dir: Path, workflows: list[str]) -> None:
    for wf in workflows:
        costs = [0.01 * (1 + i * 0.5) for i in range(10)]
        data = {
            "metadata": {
                "stats": {
                    "run_stats": [{"total_cost": c, "run_index": i} for i, c in enumerate(costs)],
                }
            }
        }
        (results_dir / f"{wf}_pilot.json").write_text(json.dumps(data))


class TestP1InfrastructureCheckMatrix:
    def test_generates_with_report(self, tmp_path: Path) -> None:
        _write_pre_cal_report(tmp_path)
        out = tmp_path / "out"
        paths = p1_infrastructure_check_matrix(tmp_path, out)
        assert len(paths) == 2  # PNG + PDF
        assert (out / "p1_infrastructure_check_matrix.png").exists()

    def test_skips_without_report(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        paths = p1_infrastructure_check_matrix(tmp_path, out)
        assert paths == []


class TestP2CostDistribution:
    def test_generates_subplots(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W1", "W2", "W13"])
        out = tmp_path / "out"
        paths = p2_per_workflow_cost_distribution(tmp_path, out)
        assert len(paths) == 2
        assert (out / "p2_cost_distribution.png").exists()

    def test_skips_without_data(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        paths = p2_per_workflow_cost_distribution(tmp_path, out)
        assert paths == []


class TestP3RoutingSanity:
    def test_only_routing_workflows(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W1", "W2", "W13", "W17", "W5", "W12"])
        out = tmp_path / "out"
        paths = p3_routing_sanity_bars(tmp_path, out)
        assert len(paths) == 2  # Generated (has W1, W2, W13, W17)

    def test_skips_when_no_routing_workflows(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W5", "W12"])
        out = tmp_path / "out"
        paths = p3_routing_sanity_bars(tmp_path, out)
        assert paths == []


class TestP4LoopHistogram:
    def test_generates_for_loop_workflows(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W2", "W4"])
        out = tmp_path / "out"
        paths = p4_loop_iteration_histogram(tmp_path, out)
        assert len(paths) == 2

    def test_skips_without_loop_data(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W1", "W9"])
        out = tmp_path / "out"
        paths = p4_loop_iteration_histogram(tmp_path, out)
        assert paths == []


class TestP5ContextGrowth:
    def test_generates_for_growth_workflows(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W2", "W4", "W19"])
        out = tmp_path / "out"
        paths = p5_context_growth_lines(tmp_path, out)
        assert len(paths) == 2

    def test_skips_without_growth_data(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W1"])
        out = tmp_path / "out"
        paths = p5_context_growth_lines(tmp_path, out)
        assert paths == []


class TestP6CostPlausibility:
    def test_generates_scatter(self, tmp_path: Path) -> None:
        _write_pilot_data(tmp_path, ["W1", "W2", "W13"])
        out = tmp_path / "out"
        paths = p6_cost_plausibility_scatter(tmp_path, out)
        assert len(paths) == 2
        assert (out / "p6_cost_plausibility.png").exists()


class TestGenerateAllPilotVisuals:
    def test_generates_all_available(self, tmp_path: Path) -> None:
        _write_pre_cal_report(tmp_path)
        _write_pilot_data(tmp_path, ["W1", "W2", "W4", "W13", "W17", "W19"])
        out = tmp_path / "out"
        paths = generate_all_pilot_visuals(tmp_path, out)
        # Should generate P1 + P2 + P3 + P4 + P5 + P6 (each 2 files)
        assert len(paths) >= 10  # At least 5 plots * 2 formats

    def test_handles_empty_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "out"
        paths = generate_all_pilot_visuals(tmp_path, out)
        assert paths == []
