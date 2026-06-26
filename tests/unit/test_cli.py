"""Tests for CLI commands using Click's test runner."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from pretia.cli import cli
from pretia.store import ProfilingSession

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
    mock.metadata = kwargs.get(
        "metadata",
        {
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
        },
    )
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


class TestRagDetection:
    def test_detects_chromadb(self, tmp_path):
        wf = tmp_path / "rag_agent.py"
        wf.write_text("import chromadb\nclient = chromadb.Client()\n")
        from pretia.cli import _detect_rag_imports

        assert _detect_rag_imports(str(wf)) is True

    def test_detects_langchain_vectorstores(self, tmp_path):
        wf = tmp_path / "rag_agent.py"
        wf.write_text("from langchain.vectorstores import Chroma\n")
        from pretia.cli import _detect_rag_imports

        assert _detect_rag_imports(str(wf)) is True

    def test_no_match(self, tmp_path):
        wf = tmp_path / "simple.py"
        wf.write_text("import os\nx = 42\n")
        from pretia.cli import _detect_rag_imports

        assert _detect_rag_imports(str(wf)) is False

    def test_nonexistent_file(self):
        from pretia.cli import _detect_rag_imports

        assert _detect_rag_imports("/nonexistent/path.py") is False


class TestProfileRun:
    def test_help(self):
        result = runner.invoke(cli, ["profile", "run", "--help"])
        assert result.exit_code == 0
        assert "WORKFLOW_PATH" in result.output
        assert "--collector" in result.output
        assert "--auto-generate" in result.output
        assert "--input" in result.output
        assert "--generator-model" in result.output

    def test_nonexistent_file(self):
        result = runner.invoke(
            cli,
            ["profile", "run", "nonexistent.py"],
        )
        assert result.exit_code != 0

    def test_invokes_runner(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        mock_session = _make_mock_session()

        with (
            patch(
                "pretia.runner.ProfileRunner.run_sync",
                return_value=mock_session,
            ) as mock_run,
            patch(
                "pretia.runner.ProfileRunner.__init__",
                return_value=None,
            ) as mock_init,
        ):
            result = runner.invoke(
                cli,
                ["profile", "run", str(wf), "--input", "hello", "-y"],
            )

        assert result.exit_code == 0
        mock_init.assert_called_once()
        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["workflow_path"] == str(wf)
        assert call_kwargs["single_input"] == "hello"
        mock_run.assert_called_once()

    def test_generator_model_passed_to_runner(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        mock_session = _make_mock_session()

        with (
            patch(
                "pretia.runner.ProfileRunner.run_sync",
                return_value=mock_session,
            ),
            patch(
                "pretia.runner.ProfileRunner.__init__",
                return_value=None,
            ) as mock_init,
        ):
            runner.invoke(
                cli,
                [
                    "profile",
                    "run",
                    str(wf),
                    "--input",
                    "hello",
                    "-y",
                    "--generator-model",
                    "gpt-4o-mini",
                ],
            )

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["generator_model"] == "gpt-4o-mini"

    def test_generator_model_defaults_to_deepseek(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        mock_session = _make_mock_session()

        with (
            patch(
                "pretia.runner.ProfileRunner.run_sync",
                return_value=mock_session,
            ),
            patch(
                "pretia.runner.ProfileRunner.__init__",
                return_value=None,
            ) as mock_init,
        ):
            runner.invoke(
                cli,
                ["profile", "run", str(wf), "--input", "hello", "-y"],
            )

        call_kwargs = mock_init.call_args[1]
        assert call_kwargs["generator_model"] == "deepseek-v4-flash"


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
            "pretia.store.ProfileStore.list_sessions",
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
            metadata={
                "cost_summary": {
                    "per_step": {},
                    "run_totals": [],
                    "mean_cost_per_run": 0,
                    "min_cost_per_run": 0,
                    "max_cost_per_run": 0,
                    "p95_cost_per_run": 0,
                    "total_session_cost": 0,
                }
            },
        )
        profile_path = tmp_path / "test.json"
        profile_path.write_text(json.dumps(session.to_dict(), indent=2))

        with patch(
            "pretia.store.ProfileStore.list_sessions",
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
            metadata={
                "cost_summary": {
                    "per_step": {},
                    "run_totals": [],
                    "mean_cost_per_run": 0.01,
                    "min_cost_per_run": 0,
                    "max_cost_per_run": 0.02,
                    "p95_cost_per_run": 0.015,
                    "total_session_cost": 0.05,
                }
            },
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
                "pretia.inputs.importer.create_langfuse_client",
                return_value=MagicMock(),
            ),
            patch(
                "pretia.inputs.importer.fetch_traces",
                return_value=[mock_trace, mock_trace, mock_trace],
            ),
            patch(
                "pretia.inputs.importer.traces_to_step_records",
                return_value=[[], [], []],
            ),
            patch(
                "pretia.store.ProfileStore.save",
                return_value=tmp_path / "out.json",
            ),
        ):
            result = runner.invoke(
                cli,
                [
                    "analyze",
                    "--from-langfuse",
                    "--last",
                    "3",
                    "--output-dir",
                    str(tmp_path),
                ],
            )

        assert result.exit_code == 0, result.output
        assert "test_workflow" in result.output or "Profile saved" in result.output


# ---------------------------------------------------------------------------
# baseline commands
# ---------------------------------------------------------------------------


class TestBaselineCommand:
    def test_help(self):
        result = runner.invoke(cli, ["baseline", "--help"])
        assert result.exit_code == 0
        assert "Manage cost baselines" in result.output

    def test_update_help(self):
        result = runner.invoke(cli, ["baseline", "update", "--help"])
        assert result.exit_code == 0
        assert "PROFILE_PATH" in result.output

    def test_update_latest(self, tmp_path):
        session = ProfilingSession(
            workflow_name="test.py",
            workflow_hash="abc",
            profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
            sample_size=10,
            input_mode="auto-generate",
            runs=[],
            metadata={
                "stats": {
                    "total_runs": 10,
                    "total_steps": 20,
                    "cost_per_run": {
                        "mean": 0.03,
                        "p50": 0.028,
                        "p75": 0.035,
                        "p90": 0.042,
                        "p95": 0.048,
                        "p99": 0.06,
                        "min": 0.015,
                        "max": 0.08,
                        "std": 0.012,
                    },
                    "tokens_per_run": {
                        "mean": 2000.0,
                        "p50": 1800.0,
                        "p75": 2400.0,
                        "p90": 2800.0,
                        "p95": 3200.0,
                        "p99": 3800.0,
                        "min": 1000.0,
                        "max": 4500.0,
                        "std": 600.0,
                    },
                    "step_stats": {
                        "step_a": {
                            "step_name": "step_a",
                            "step_type": "llm",
                            "model": "gpt-4o-mini",
                            "call_count": 10,
                            "runs_present": 10,
                            "input_tokens": {
                                "mean": 200.0,
                                "p50": 180.0,
                                "p75": 220.0,
                                "p90": 250.0,
                                "p95": 280.0,
                                "p99": 300.0,
                                "min": 100.0,
                                "max": 350.0,
                                "std": 40.0,
                            },
                            "output_tokens": {
                                "mean": 60.0,
                                "p50": 55.0,
                                "p75": 70.0,
                                "p90": 80.0,
                                "p95": 85.0,
                                "p99": 95.0,
                                "min": 30.0,
                                "max": 110.0,
                                "std": 12.0,
                            },
                            "total_tokens": {
                                "mean": 260.0,
                                "p50": 235.0,
                                "p75": 290.0,
                                "p90": 330.0,
                                "p95": 365.0,
                                "p99": 395.0,
                                "min": 130.0,
                                "max": 460.0,
                                "std": 52.0,
                            },
                            "cost": {
                                "mean": 0.001,
                                "p50": 0.0009,
                                "p75": 0.0012,
                                "p90": 0.0014,
                                "p95": 0.0016,
                                "p99": 0.002,
                                "min": 0.0004,
                                "max": 0.003,
                                "std": 0.0003,
                            },
                            "duration_ms": {
                                "mean": 180.0,
                                "p50": 160.0,
                                "p75": 200.0,
                                "p90": 230.0,
                                "p95": 250.0,
                                "p99": 280.0,
                                "min": 80.0,
                                "max": 350.0,
                                "std": 45.0,
                            },
                            "context_size": {
                                "mean": 200.0,
                                "p50": 180.0,
                                "p75": 220.0,
                                "p90": 250.0,
                                "p95": 280.0,
                                "p99": 300.0,
                                "min": 100.0,
                                "max": 350.0,
                                "std": 40.0,
                            },
                            "iterations_per_run": {
                                "mean": 1.0,
                                "p50": 1.0,
                                "p75": 1.0,
                                "p90": 1.0,
                                "p95": 1.0,
                                "p99": 1.0,
                                "min": 1.0,
                                "max": 1.0,
                                "std": 0.0,
                            },
                            "mean_iterations": 1.0,
                        },
                    },
                    "run_stats": [],
                },
                "patterns": [],
                "projection": {
                    "method": "linear",
                    "traffic_volumes": [1000],
                    "projections": {
                        "1000": {
                            "daily_volume": 1000,
                            "monthly_cost": {
                                "p50": 840.0,
                                "p75": 1050.0,
                                "p90": 1260.0,
                                "p95": 1440.0,
                                "p99": 1800.0,
                                "mean": 900.0,
                            },
                            "daily_cost": {
                                "p50": 28.0,
                                "p75": 35.0,
                                "p90": 42.0,
                                "p95": 48.0,
                                "p99": 60.0,
                                "mean": 30.0,
                            },
                            "cost_per_run": {
                                "p50": 0.028,
                                "p75": 0.035,
                                "p90": 0.042,
                                "p95": 0.048,
                                "p99": 0.06,
                                "mean": 0.03,
                            },
                        },
                    },
                    "confidence": {
                        "score": 72,
                        "tier": "MODERATE",
                        "display_range": "p50 – p95",
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
                    "display_range": "p50 – p95",
                    "language": "estimated",
                    "deductions": [],
                    "bonuses": [],
                },
            },
        )
        profile_path = tmp_path / "test.json"
        profile_path.write_text(json.dumps(session.to_dict(), indent=2))

        with patch(
            "pretia.store.ProfileStore.list_sessions",
            return_value=[profile_path],
        ):
            result = runner.invoke(
                cli,
                ["baseline", "update", "latest", "--output-dir", str(tmp_path)],
            )

        assert result.exit_code == 0, result.output
        assert "Baseline saved" in result.output
        assert (tmp_path / "baseline.json").exists()


# ---------------------------------------------------------------------------
# diff command
# ---------------------------------------------------------------------------


class TestDiffCommand:
    def test_help(self):
        result = runner.invoke(cli, ["diff", "--help"])
        assert result.exit_code == 0

    def test_baseline_not_found(self):
        result = runner.invoke(
            cli,
            ["diff", "nonexistent.json", "also_nonexistent.json"],
        )
        assert result.exit_code == 1
        assert "not found" in result.output.lower() or "error" in result.output.lower()


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------


class TestValidateCommand:
    def test_help(self):
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "WORKFLOW_PATH" in result.output
        assert "--budget" in result.output
