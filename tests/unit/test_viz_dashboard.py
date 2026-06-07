"""Tests for the interactive dashboard generator."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def _write_backtest_results(
    results_dir: Path,
    workflows: list[dict] | None = None,
) -> None:
    """Write synthetic backtest results in the expected JSON format."""
    results_dir.mkdir(parents=True, exist_ok=True)

    if workflows is None:
        workflows = [
            {
                "workflow_name": "W1",
                "comparisons": {
                    "A": {
                        "score": {
                            "workflow_name": "W1",
                            "comparison": "A",
                            "mean_error_pct": 5.0,
                            "p75_error_pct": 8.0,
                            "ci_coverage_pct": 90.0,
                            "monthly_error_pct": 5.0,
                            "cvar95_error_pct": 15.0,
                            "passes": True,
                            "failures": [],
                        }
                    },
                    "B": {
                        "score": {
                            "workflow_name": "W1",
                            "comparison": "B",
                            "mean_error_pct": 12.0,
                            "p75_error_pct": 15.0,
                            "ci_coverage_pct": 85.0,
                            "monthly_error_pct": 12.0,
                            "cvar95_error_pct": 20.0,
                            "passes": True,
                            "failures": [],
                        }
                    },
                    "C": None,
                },
                "detected_patterns": [],
                "step_costs": {"classify": 0.001, "generate": 0.028},
            },
            {
                "workflow_name": "W2",
                "comparisons": {
                    "A": {
                        "score": {
                            "workflow_name": "W2",
                            "comparison": "A",
                            "mean_error_pct": 7.0,
                            "p75_error_pct": 10.0,
                            "ci_coverage_pct": 88.0,
                            "monthly_error_pct": 7.0,
                            "cvar95_error_pct": 18.0,
                            "passes": True,
                            "failures": [],
                        }
                    },
                    "B": {
                        "score": {
                            "workflow_name": "W2",
                            "comparison": "B",
                            "mean_error_pct": 25.0,
                            "p75_error_pct": 30.0,
                            "ci_coverage_pct": 70.0,
                            "monthly_error_pct": 25.0,
                            "cvar95_error_pct": 45.0,
                            "passes": False,
                            "failures": ["Mean error 25.0% exceeds 20.0% target"],
                        }
                    },
                    "C": {
                        "score": {
                            "workflow_name": "W2",
                            "comparison": "C",
                            "mean_error_pct": 10.0,
                            "p75_error_pct": 13.0,
                            "ci_coverage_pct": 86.0,
                            "monthly_error_pct": 10.0,
                            "cvar95_error_pct": 22.0,
                            "passes": True,
                            "failures": [],
                        }
                    },
                },
                "detected_patterns": ["context_growth", "loop_count_variance"],
                "step_costs": {"loop_step": 0.015, "summarize": 0.008},
            },
            {
                "workflow_name": "W13",
                "comparisons": {
                    "A": {
                        "score": {
                            "workflow_name": "W13",
                            "comparison": "A",
                            "mean_error_pct": 30.0,
                            "p75_error_pct": 35.0,
                            "ci_coverage_pct": 60.0,
                            "monthly_error_pct": 30.0,
                            "cvar95_error_pct": 50.0,
                            "passes": False,
                            "failures": ["Mean error 30.0% exceeds 10.0% target"],
                        }
                    },
                    "B": None,
                    "C": None,
                },
                "detected_patterns": ["step_count_variance", "bimodality"],
                "step_costs": {"route": 0.002, "handle_a": 0.012, "handle_b": 0.009},
            },
        ]

    for wf in workflows:
        name = wf["workflow_name"]
        path = results_dir / f"{name}.json"
        path.write_text(json.dumps(wf, indent=2))


class TestDashboard:
    def test_generates_html(self, tmp_path: Path) -> None:
        """Dashboard generates an HTML file with key markers."""
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")

        from visualization.dashboard.generate_dashboard import generate_dashboard

        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "dashboard.html"

        result = generate_dashboard(results_dir, output)

        assert result is not None
        assert output.exists()
        content = output.read_text()
        assert "AgentCost Backtest Dashboard" in content
        assert "W1" in content
        assert "W2" in content

    def test_self_contained_no_cdn(self, tmp_path: Path) -> None:
        """Dashboard has no external CDN src= attributes."""
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")

        from visualization.dashboard.generate_dashboard import generate_dashboard

        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "dashboard.html"

        generate_dashboard(results_dir, output)
        content = output.read_text()

        # Should not load plotly.js from external CDN via <script src="...">
        # Note: the bundled plotly.js itself contains internal "cdn.plot.ly" references
        assert 'src="https://cdn.plot.ly' not in content
        assert 'src="http://cdn.plot.ly' not in content

    def test_d1_three_numbers(self, tmp_path: Path) -> None:
        """Executive summary has 3 numeric count cards."""
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")

        from visualization.dashboard.generate_dashboard import generate_dashboard

        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "dashboard.html"

        generate_dashboard(results_dir, output)
        content = output.read_text()

        # Three summary cards with class="summary-card"
        assert content.count("summary-card") >= 3
        # Should have "Passing All Comparisons", "Need Reweighting", "Unresolved"
        assert "Passing All Comparisons" in content
        assert "Need Reweighting" in content
        assert "Unresolved" in content

    def test_skips_without_plotly(self, tmp_path: Path) -> None:
        """Dashboard generation returns None when plotly is not installed."""
        from visualization.dashboard.generate_dashboard import generate_dashboard

        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "dashboard.html"

        # Patch the import inside generate_dashboard to simulate plotly missing
        import builtins

        original_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "plotly.graph_objects" or name == "plotly":
                raise ImportError("No module named 'plotly'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = generate_dashboard(results_dir, output)

        assert result is None
        assert not output.exists()

    def test_empty_results_dir(self, tmp_path: Path) -> None:
        """Dashboard returns None for empty results directory."""
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")

        from visualization.dashboard.generate_dashboard import generate_dashboard

        results_dir = tmp_path / "results"
        results_dir.mkdir()
        output = tmp_path / "dashboard.html"

        result = generate_dashboard(results_dir, output)
        assert result is None

    def test_nonexistent_results_dir(self, tmp_path: Path) -> None:
        """Dashboard returns None for nonexistent results directory."""
        try:
            import plotly  # noqa: F401
        except ImportError:
            pytest.skip("plotly not installed")

        from visualization.dashboard.generate_dashboard import generate_dashboard

        results_dir = tmp_path / "nonexistent"
        output = tmp_path / "dashboard.html"

        result = generate_dashboard(results_dir, output)
        assert result is None
