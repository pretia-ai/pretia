"""Tests for the narrative report generator."""

from __future__ import annotations

import json
from pathlib import Path

from visualization.narrative.generate_narrative import (
    _compute_detector_assessment,
    _compute_drift_analysis,
    _load_workflow_scores,
    generate_narrative,
)


def _write_backtest_results(
    results_dir: Path,
    workflows: list[dict] | None = None,
) -> None:
    """Write synthetic backtest results in the expected JSON format."""
    results_dir.mkdir(parents=True, exist_ok=True)

    if workflows is None:
        workflows = _default_workflows()

    for wf in workflows:
        name = wf["workflow_name"]
        path = results_dir / f"{name}.json"
        path.write_text(json.dumps(wf, indent=2))


def _default_workflows() -> list[dict]:
    """Return a default set of 5 workflow results covering all buckets."""
    return [
        # W1: passes all (A pass, B pass)
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
        # W2: bucket 2 (A pass, B fail, C pass) -> drift_sensitivity
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
        # W13: bucket 1 (A fail) -> engine_problem
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
        # W4: bucket 3 (A pass, B fail, C fail) -> structural_drift
        {
            "workflow_name": "W4",
            "comparisons": {
                "A": {
                    "score": {
                        "workflow_name": "W4",
                        "comparison": "A",
                        "mean_error_pct": 8.0,
                        "p75_error_pct": 11.0,
                        "ci_coverage_pct": 87.0,
                        "monthly_error_pct": 8.0,
                        "cvar95_error_pct": 20.0,
                        "passes": True,
                        "failures": [],
                    }
                },
                "B": {
                    "score": {
                        "workflow_name": "W4",
                        "comparison": "B",
                        "mean_error_pct": 35.0,
                        "p75_error_pct": 40.0,
                        "ci_coverage_pct": 55.0,
                        "monthly_error_pct": 35.0,
                        "cvar95_error_pct": 55.0,
                        "passes": False,
                        "failures": ["Mean error 35.0% exceeds 20.0% target"],
                    }
                },
                "C": {
                    "score": {
                        "workflow_name": "W4",
                        "comparison": "C",
                        "mean_error_pct": 30.0,
                        "p75_error_pct": 35.0,
                        "ci_coverage_pct": 60.0,
                        "monthly_error_pct": 30.0,
                        "cvar95_error_pct": 48.0,
                        "passes": False,
                        "failures": ["Mean error 30.0% exceeds 20.0% target"],
                    }
                },
            },
            "detected_patterns": ["context_growth", "loop_count_variance"],
            "step_costs": {"iterate": 0.020, "check": 0.003},
        },
        # W5: passes all
        {
            "workflow_name": "W5",
            "comparisons": {
                "A": {
                    "score": {
                        "workflow_name": "W5",
                        "comparison": "A",
                        "mean_error_pct": 3.0,
                        "p75_error_pct": 5.0,
                        "ci_coverage_pct": 95.0,
                        "monthly_error_pct": 3.0,
                        "cvar95_error_pct": 10.0,
                        "passes": True,
                        "failures": [],
                    }
                },
                "B": {
                    "score": {
                        "workflow_name": "W5",
                        "comparison": "B",
                        "mean_error_pct": 8.0,
                        "p75_error_pct": 12.0,
                        "ci_coverage_pct": 88.0,
                        "monthly_error_pct": 8.0,
                        "cvar95_error_pct": 16.0,
                        "passes": True,
                        "failures": [],
                    }
                },
                "C": None,
            },
            "detected_patterns": ["high_token_variance"],
            "step_costs": {"analyze": 0.005, "respond": 0.018},
        },
    ]


class TestNarrative:
    def test_has_all_five_sections(self, tmp_path: Path) -> None:
        """Generated report has all 5 section headers."""
        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "report.md"

        generate_narrative(results_dir, output)

        assert output.exists()
        content = output.read_text()
        assert "## Executive Summary" in content
        assert "## Per-Workflow Failure Analysis" in content
        assert "## Drift Analysis" in content
        assert "## Detector Assessment" in content
        assert "## Recommendations" in content

    def test_bucket_1_in_engine_section(self, tmp_path: Path) -> None:
        """Workflow with bucket 1 appears under engine problem analysis."""
        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "report.md"

        generate_narrative(results_dir, output)

        content = output.read_text()
        # W13 is bucket 1 (engine_problem)
        assert "W13" in content
        assert "engine_problem" in content
        assert "Engine Problems" in content

    def test_bucket_2_recommends_traffic_mix(self, tmp_path: Path) -> None:
        """Narrative mentions --traffic-mix for bucket 2 workflows."""
        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "report.md"

        generate_narrative(results_dir, output)

        content = output.read_text()
        assert "--traffic-mix" in content
        # W2 is bucket 2
        assert "W2" in content
        assert "drift_sensitivity" in content

    def test_detector_assessment_rates(self, tmp_path: Path) -> None:
        """TP/FN rates are computed correctly."""
        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)

        assessment = _compute_detector_assessment(results_dir)

        # Should have counts
        assert assessment["tp"] + assessment["fn"] + assessment["fp"] + assessment["tn"] > 0
        # Rates should be between 0 and 1
        assert 0 <= assessment["tp_rate"] <= 1
        assert 0 <= assessment["fn_rate"] <= 1
        assert 0 <= assessment["fp_rate"] <= 1
        # tp_rate + fn_rate should equal 1.0
        total_pos = assessment["tp"] + assessment["fn"]
        if total_pos > 0:
            assert abs(assessment["tp_rate"] + assessment["fn_rate"] - 1.0) < 1e-9

    def test_drift_analysis_identifies_tier_shift(self, tmp_path: Path) -> None:
        """When all workflows degrade uniformly, identifies tier weight shift."""
        # Create workflows where all fail B with similar degradation amounts
        workflows = []
        for wf_name in ["W1", "W5", "W9", "W11", "W12", "W18"]:
            workflows.append(
                {
                    "workflow_name": wf_name,
                    "comparisons": {
                        "A": {
                            "score": {
                                "workflow_name": wf_name,
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
                                "workflow_name": wf_name,
                                "comparison": "B",
                                "mean_error_pct": 25.0,
                                "p75_error_pct": 30.0,
                                "ci_coverage_pct": 70.0,
                                "monthly_error_pct": 25.0,
                                "cvar95_error_pct": 45.0,
                                "passes": False,
                                "failures": ["Mean error exceeds target"],
                            }
                        },
                        "C": None,
                    },
                    "detected_patterns": [],
                    "step_costs": {"step_a": 0.01},
                }
            )

        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir, workflows)

        scores = _load_workflow_scores(results_dir)
        drift = _compute_drift_analysis(scores)

        assert drift["diagnosis"] == "tier_weight_shift"
        assert drift["uniform_degradation"] is True

    def test_all_passing_no_failures_section(self, tmp_path: Path) -> None:
        """When all workflows pass, failure section says so."""
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
                            "mean_error_pct": 8.0,
                            "p75_error_pct": 12.0,
                            "ci_coverage_pct": 88.0,
                            "monthly_error_pct": 8.0,
                            "cvar95_error_pct": 16.0,
                            "passes": True,
                            "failures": [],
                        }
                    },
                    "C": None,
                },
                "detected_patterns": [],
                "step_costs": {"step": 0.01},
            },
        ]

        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir, workflows)
        output = tmp_path / "report.md"

        generate_narrative(results_dir, output)

        content = output.read_text()
        assert "No workflow failures to report" in content

    def test_structural_drift_workflow(self, tmp_path: Path) -> None:
        """Bucket 3 (structural drift) workflow gets correct recommendation."""
        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)
        output = tmp_path / "report.md"

        generate_narrative(results_dir, output)

        content = output.read_text()
        # W4 is bucket 3 (structural_drift)
        assert "W4" in content
        assert "structural_drift" in content
        assert "Re-profile" in content or "re-profile" in content

    def test_load_workflow_scores(self, tmp_path: Path) -> None:
        """Score loading correctly parses ComparisonScore from JSON."""
        results_dir = tmp_path / "results"
        _write_backtest_results(results_dir)

        scores = _load_workflow_scores(results_dir)

        assert "W1" in scores
        assert "W2" in scores
        assert scores["W1"]["A"] is not None
        assert scores["W1"]["A"].passes is True
        assert scores["W1"]["A"].mean_error_pct == 5.0
        assert scores["W1"]["C"] is None  # W1 has no C comparison
