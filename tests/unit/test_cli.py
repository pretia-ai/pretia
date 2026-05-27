"""Tests for CLI commands using Click's test runner."""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agentcost.cli import cli
from agentcost.store import ProfilingSession

runner = CliRunner()


def _make_mock_session(**kwargs):
    mock = MagicMock(spec=ProfilingSession)
    mock.workflow_name = kwargs.get("workflow_name", "test_agent.py")
    mock.workflow_hash = "abc123"
    mock.sample_size = kwargs.get("sample_size", 1)
    mock.input_mode = kwargs.get("input_mode", "single")
    mock.runs = []
    profiled_at = MagicMock()
    profiled_at.strftime.return_value = "2026-05-25 12:00"
    mock.profiled_at = profiled_at
    mock.metadata = kwargs.get("metadata", {
        "cost_summary": {
            "per_step": {},
            "run_totals": [],
            "mean_cost_per_run": 0,
            "min_cost_per_run": 0,
            "max_cost_per_run": 0,
            "p95_cost_per_run": 0,
            "total_session_cost": 0,
            "projection_100_day": 0,
            "projection_1000_day": 0,
            "projection_10000_day": 0,
        },
        "saved_path": "/tmp/test.json",
    })
    return mock


# ---------------------------------------------------------------------------
# Top-level
# ---------------------------------------------------------------------------

class TestTopLevel:
    def test_help(self):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "cost intelligence" in result.output

    def test_version(self):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


# ---------------------------------------------------------------------------
# profile group
# ---------------------------------------------------------------------------

class TestProfileGroup:
    def test_help(self):
        result = runner.invoke(cli, ["profile", "--help"])
        assert result.exit_code == 0
        assert "Profile agent workflows" in result.output


# ---------------------------------------------------------------------------
# profile run
# ---------------------------------------------------------------------------

class TestProfileRun:
    def test_help(self):
        result = runner.invoke(cli, ["profile", "run", "--help"])
        assert result.exit_code == 0
        assert "WORKFLOW_PATH" in result.output
        assert "--collector" in result.output
        assert "--auto-generate" in result.output
        assert "--input" in result.output

    def test_nonexistent_file(self):
        result = runner.invoke(
            cli, ["profile", "run", "nonexistent.py"],
        )
        assert result.exit_code != 0

    def test_invokes_runner(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        mock_session = _make_mock_session()

        with patch(
            "agentcost.runner.ProfileRunner.run_sync",
            return_value=mock_session,
        ) as mock_run, patch(
            "agentcost.runner.ProfileRunner.__init__",
            return_value=None,
        ) as mock_init:
            result = runner.invoke(
                cli,
                ["profile", "run", str(wf), "--input", "hello"],
            )

        assert result.exit_code == 0
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["workflow_path"] == str(wf)
        assert call_kwargs["single_input"] == "hello"
        mock_run.assert_called_once()


# ---------------------------------------------------------------------------
# report command
# ---------------------------------------------------------------------------

class TestReportCommand:
    def test_help(self):
        result = runner.invoke(cli, ["report", "--help"])
        assert result.exit_code == 0
        assert "PROFILE_PATH" in result.output

    def test_file_not_found(self):
        result = runner.invoke(cli, ["report", "nonexistent.json"])
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "invalid" in result.output.lower()

    def test_latest_no_profiles(self, tmp_path):
        with patch(
            "agentcost.store.ProfileStore.list_sessions",
            return_value=[],
        ):
            result = runner.invoke(cli, ["report", "latest"])
        assert result.exit_code == 1
        assert "No saved profiles" in result.output

    def test_latest_loads_profile(self, tmp_path):
        session = ProfilingSession(
            workflow_name="test.py",
            workflow_hash="abc",
            profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
            sample_size=3,
            input_mode="auto-generate",
            runs=[],
            metadata={"cost_summary": {
                "per_step": {},
                "run_totals": [],
                "mean_cost_per_run": 0,
                "min_cost_per_run": 0,
                "max_cost_per_run": 0,
                "p95_cost_per_run": 0,
                "total_session_cost": 0,
            }},
        )
        profile_path = tmp_path / "test.json"
        profile_path.write_text(json.dumps(session.to_dict(), indent=2))

        with patch(
            "agentcost.store.ProfileStore.list_sessions",
            return_value=[profile_path],
        ):
            result = runner.invoke(cli, ["report", "latest"])
        assert result.exit_code == 0
        assert "test.py" in result.output

    def test_load_specific_file(self, tmp_path):
        session = ProfilingSession(
            workflow_name="specific_agent.py",
            workflow_hash="def",
            profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
            sample_size=5,
            input_mode="single",
            runs=[],
            metadata={"cost_summary": {
                "per_step": {},
                "run_totals": [],
                "mean_cost_per_run": 0.01,
                "min_cost_per_run": 0,
                "max_cost_per_run": 0.02,
                "p95_cost_per_run": 0.015,
                "total_session_cost": 0.05,
            }},
        )
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(session.to_dict(), indent=2))

        result = runner.invoke(cli, ["report", str(profile_path)])
        assert result.exit_code == 0
        assert "specific_agent.py" in result.output


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------

class TestAnalyzeCommand:
    def test_help(self):
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "--from-langfuse" in result.output

    def test_missing_credentials(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        result = runner.invoke(cli, ["analyze", "--from-langfuse"])
        assert result.exit_code == 1
        assert "credentials" in result.output.lower()

    def test_from_langfuse(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")

        mock_obs = MagicMock()
        mock_obs.observation_id = "obs-1"
        mock_obs.name = "classify"
        mock_obs.observation_type = "GENERATION"
        mock_obs.model = "gpt-4o"
        mock_obs.input_tokens = 100
        mock_obs.output_tokens = 50
        mock_obs.start_time = datetime(2026, 1, 1, tzinfo=UTC)
        mock_obs.end_time = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)
        mock_obs.duration_ms = 1000
        mock_obs.parent_observation_id = None

        mock_trace = MagicMock()
        mock_trace.trace_id = "t-1"
        mock_trace.name = "test_workflow"
        mock_trace.input_text = "hello"
        mock_trace.timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        mock_trace.observations = [mock_obs]
        mock_trace.total_input_tokens = 100
        mock_trace.total_output_tokens = 50
        mock_trace.total_cost = 0.001

        mock_records = [MagicMock()]
        mock_records[0].step_name = "classify"
        mock_records[0].step_type = "llm"
        mock_records[0].model = "gpt-4o"
        mock_records[0].input_tokens = 100
        mock_records[0].output_tokens = 50
        mock_records[0].context_size = 100
        mock_records[0].iteration = 1
        mock_records[0].duration_ms = 1000

        with (
            patch(
                "agentcost.inputs.importer.create_langfuse_client",
                return_value=MagicMock(),
            ),
            patch(
                "agentcost.inputs.importer.fetch_traces",
                return_value=[mock_trace, mock_trace, mock_trace],
            ),
            patch(
                "agentcost.inputs.importer.traces_to_step_records",
                return_value=[[], [], []],
            ),
            patch(
                "agentcost.store.ProfileStore.save",
                return_value=tmp_path / "out.json",
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "analyze", "--from-langfuse",
                    "--last", "3",
                    "--output-dir", str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "test_workflow" in result.output or "Profile saved" in result.output
