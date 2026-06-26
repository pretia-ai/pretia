"""Tests for CLI recommend command and recommendation wiring."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from pretia.cli import cli
from pretia.store import ProfilingSession

runner = CliRunner()


def _make_session_with_metadata(tmp_path, **extra_meta):
    """Create a real ProfilingSession, write it to disk, return the path."""
    meta = {
        "cost_summary": {
            "per_step": {},
            "run_totals": [],
            "mean_cost_per_run": 0.01,
            "min_cost_per_run": 0,
            "max_cost_per_run": 0.02,
            "p95_cost_per_run": 0.015,
            "total_session_cost": 0.05,
        },
        "stats": {
            "total_runs": 5,
            "total_steps": 5,
            "step_stats": {},
            "run_stats": [],
            "cost_per_run": {
                "mean": 0.01,
                "p50": 0.009,
                "p75": 0.012,
                "p90": 0.014,
                "p95": 0.016,
                "p99": 0.02,
                "min": 0.005,
                "max": 0.025,
                "std": 0.004,
            },
            "tokens_per_run": None,
        },
        "patterns": [],
        "projection": {
            "method": "linear",
            "traffic_volumes": [100, 1000, 10000],
            "projections": {
                "10000": {
                    "daily_volume": 10000,
                    "monthly_cost": {
                        "p50": 2700.0,
                        "p75": 3600.0,
                        "p90": 4200.0,
                        "p95": 4800.0,
                        "p99": 6000.0,
                        "mean": 3000.0,
                    },
                    "daily_cost": {
                        "p50": 90.0,
                        "p75": 120.0,
                        "p90": 140.0,
                        "p95": 160.0,
                        "p99": 200.0,
                        "mean": 100.0,
                    },
                    "cost_per_run": {
                        "p50": 0.009,
                        "p75": 0.012,
                        "p90": 0.014,
                        "p95": 0.016,
                        "p99": 0.02,
                        "mean": 0.01,
                    },
                },
            },
            "confidence": {
                "score": 72,
                "tier": "MODERATE",
                "display_range": "p50 - p95",
                "language": "estimated",
                "deductions": [],
                "bonuses": [],
            },
            "warnings": [],
            "patterns_detected": [],
        },
        "confidence": {
            "score": 72,
            "tier": "MODERATE",
            "display_range": "p50 - p95",
            "language": "estimated",
            "deductions": [],
            "bonuses": [],
        },
    }
    meta.update(extra_meta)

    session = ProfilingSession(
        workflow_name="test_agent.py",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=5,
        input_mode="auto",
        runs=[],
        metadata=meta,
    )
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(session.to_dict(), indent=2))
    return path


class TestRecommendCommand:
    def test_help(self) -> None:
        result = runner.invoke(cli, ["recommend", "--help"])
        assert result.exit_code == 0
        assert "PROFILE_PATH" in result.output

    def test_file_not_found(self) -> None:
        result = runner.invoke(cli, ["recommend", "nonexistent.json"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "invalid" in result.output.lower()

    def test_loads_profile_and_renders(self, tmp_path) -> None:
        profile_path = _make_session_with_metadata(tmp_path)
        result = runner.invoke(cli, ["recommend", str(profile_path)])
        assert result.exit_code == 0, result.output
        assert "Optimization Score" in result.output

    def test_latest_no_profiles(self) -> None:
        with patch(
            "pretia.store.ProfileStore.list_sessions",
            return_value=[],
        ):
            result = runner.invoke(cli, ["recommend", "latest"])
        assert result.exit_code == 1
        assert "No saved profiles" in result.output

    def test_renders_score_and_recommendations(self, tmp_path) -> None:
        profile_path = _make_session_with_metadata(tmp_path)
        result = runner.invoke(cli, ["recommend", str(profile_path)])
        assert result.exit_code == 0, result.output
        assert "/ 100" in result.output
        assert "Recommendations" in result.output


class TestReportWithRecommendations:
    def test_report_includes_recommendations(self, tmp_path) -> None:
        """Report command enriches session with recommendations."""
        profile_path = _make_session_with_metadata(tmp_path)
        result = runner.invoke(cli, ["report", str(profile_path)])
        assert result.exit_code == 0, result.output
        assert "Optimization Score" in result.output

    def test_report_with_pre_existing_recommendations(self, tmp_path) -> None:
        """Report renders pre-existing recommendations in metadata."""
        profile_path = _make_session_with_metadata(
            tmp_path,
            recommendations=[
                {
                    "id": "test-rec",
                    "type": "model_swap",
                    "title": "Test swap",
                    "description": "Test description.",
                    "monthly_savings": 500.0,
                    "confidence": "HIGH",
                    "affected_steps": ["step_a"],
                    "evidence": {},
                    "priority": 500,
                }
            ],
            score={
                "score": 85,
                "zone": "green",
                "zone_label": "well optimized",
                "zone_color": "#38A169",
                "total_savings": 500.0,
                "waste_pct": 0.15,
                "recommendation_count": 1,
                "scope_note": "",
            },
        )
        result = runner.invoke(cli, ["report", str(profile_path)])
        assert result.exit_code == 0, result.output
        assert "Test swap" in result.output


class TestProfileRunWithRecommendations:
    def test_profile_run_includes_recommendations(self, tmp_path) -> None:
        mock_session = MagicMock(spec=ProfilingSession)
        mock_session.workflow_name = "test.py"
        mock_session.workflow_hash = "abc"
        mock_session.sample_size = 1
        mock_session.input_mode = "single"
        mock_session.runs = []
        profiled_at = MagicMock()
        profiled_at.strftime.return_value = "2026-05-25 12:00"
        mock_session.profiled_at = profiled_at
        mock_session.metadata = {
            "cost_summary": {
                "per_step": {},
                "run_totals": [],
                "mean_cost_per_run": 0,
                "min_cost_per_run": 0,
                "max_cost_per_run": 0,
                "p95_cost_per_run": 0,
                "total_session_cost": 0,
            },
            "saved_path": "/tmp/test.json",
            "patterns": [],
            "projection": {},
        }

        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        with (
            patch(
                "pretia.runner.ProfileRunner.run_sync",
                return_value=mock_session,
            ),
            patch(
                "pretia.runner.ProfileRunner.__init__",
                return_value=None,
            ),
        ):
            result = runner.invoke(
                cli,
                ["profile", "run", str(wf), "--input", "hello", "-y"],
            )

        assert result.exit_code == 0, result.output
        assert "recommendations" in mock_session.metadata
        assert "score" in mock_session.metadata
