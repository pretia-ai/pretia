"""Tests for CLI commands using Click's test runner."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from agentcost.cli import cli

runner = CliRunner()


class TestTopLevel:
    def test_help(self):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "cost intelligence" in result.output

    def test_version(self):
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0


class TestProfileGroup:
    def test_help(self):
        result = runner.invoke(cli, ["profile", "--help"])
        assert result.exit_code == 0
        assert "Profile agent workflows" in result.output


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

        mock_session = MagicMock()
        mock_session.metadata = {
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
            "saved_path": str(tmp_path / "out.json"),
        }
        mock_session.workflow_name = str(wf)
        mock_session.sample_size = 1
        mock_session.input_mode = "single"
        mock_session.profiled_at = MagicMock()
        mock_session.profiled_at.strftime.return_value = "2026-05-25 12:00"

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
