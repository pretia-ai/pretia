"""Tests for SDK collectors: step naming, multi-SDK, streaming, LangGraph input detection."""

from __future__ import annotations

from unittest.mock import MagicMock

from pretia.collectors._utils import get_caller_name

# ---------------------------------------------------------------------------
# Step naming from call stack
# ---------------------------------------------------------------------------


class TestCallerName:
    def test_caller_name_from_direct_call(self):
        def classify():
            return get_caller_name()

        assert classify() == "classify"

    def test_caller_name_from_nested_call(self):
        def inner():
            return get_caller_name()

        def triage():
            return inner()

        assert triage() == "inner"

    def test_default_when_no_user_frame(self):
        name = get_caller_name(default="fallback")
        assert isinstance(name, str)


# ---------------------------------------------------------------------------
# LangGraph input key detection
# ---------------------------------------------------------------------------


class TestGraphInputDetection:
    def test_detects_key_from_schema_annotations(self):
        from pretia.runner import _detect_graph_input_key

        class FakeState:
            __annotations__ = {"post": str, "result": str}

        graph = MagicMock()
        graph.builder.schema = FakeState
        assert _detect_graph_input_key(graph) == "post"

    def test_falls_back_to_input(self):
        from pretia.runner import _detect_graph_input_key

        graph = MagicMock()
        graph.builder = None
        graph.channels = None
        assert _detect_graph_input_key(graph) == "input"

    def test_detects_from_channels_dict(self):
        from pretia.runner import _detect_graph_input_key

        graph = MagicMock()
        graph.builder = None
        graph.channels = {"query": MagicMock(), "context": MagicMock()}
        assert _detect_graph_input_key(graph) == "query"


# ---------------------------------------------------------------------------
# Doctor command
# ---------------------------------------------------------------------------


class TestDoctorCommand:
    def test_doctor_no_args(self):
        from click.testing import CliRunner

        from pretia.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "Python version" in result.output
        assert "API key" in result.output

    def test_doctor_with_valid_workflow(self, tmp_path):
        from click.testing import CliRunner

        from pretia.cli import cli

        wf = tmp_path / "simple.py"
        wf.write_text("async def workflow(inp: str) -> str:\n    return 'hello'\n")

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", str(wf)])
        assert result.exit_code == 0
        assert "Workflow file" in result.output

    def test_doctor_missing_workflow(self):
        from click.testing import CliRunner

        from pretia.cli import cli

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor", "/nonexistent/workflow.py"])
        assert result.exit_code == 0
        assert "not found" in result.output

    def test_doctor_missing_keys(self, monkeypatch):
        from click.testing import CliRunner

        from pretia.cli import cli

        for var in (
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "DASHSCOPE_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)

        runner = CliRunner()
        result = runner.invoke(cli, ["doctor"])
        assert result.exit_code == 0
        assert "not set" in result.output


# ---------------------------------------------------------------------------
# Anthropic collector record building
# ---------------------------------------------------------------------------


class TestAnthropicRecord:
    def test_record_from_response(self):
        from pretia.collectors.anthropic_sdk import _record_from_response

        response = MagicMock()
        response.model = "claude-haiku-4-5"
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = None
        response.usage.cache_creation_input_tokens = None

        captured = []
        _record_from_response(response, 0, captured, "classify")
        assert len(captured) == 1
        assert captured[0].step_name == "classify"
        assert captured[0].model == "claude-haiku-4-5"
        assert captured[0].input_tokens == 100
        assert captured[0].output_tokens == 50


# ---------------------------------------------------------------------------
# OpenAI collector record building
# ---------------------------------------------------------------------------


class TestOpenAIRecord:
    def test_record_from_response(self):
        from pretia.collectors.openai_sdk import _record_from_response

        response = MagicMock()
        response.model = "gpt-4o-mini"
        response.usage.prompt_tokens = 200
        response.usage.completion_tokens = 30
        response.usage.prompt_tokens_details = None

        captured = []
        _record_from_response(response, 0, captured, "triage")
        assert len(captured) == 1
        assert captured[0].step_name == "triage"
        assert captured[0].model == "gpt-4o-mini"
        assert captured[0].input_tokens == 200
        assert captured[0].output_tokens == 30

    def test_record_from_stream_chunk(self):
        from pretia.collectors.openai_sdk import _record_from_chunk

        chunk = MagicMock()
        chunk.model = "gpt-4o"
        chunk.usage.prompt_tokens = 150
        chunk.usage.completion_tokens = 80

        captured = []
        _record_from_chunk(chunk, 0, captured, "review")
        assert len(captured) == 1
        assert captured[0].step_name == "review"
        assert captured[0].input_tokens == 150

    def test_stream_no_usage_warns(self, caplog):
        import logging

        from pretia.collectors.openai_sdk import _record_from_chunk

        chunk = MagicMock()
        chunk.usage = None

        captured = []
        with caplog.at_level(logging.WARNING):
            _record_from_chunk(chunk, 0, captured, "test")
        assert len(captured) == 0
        assert "without usage data" in caplog.text
