"""Tests for input selector: mode resolution, file reading, priority order."""

from __future__ import annotations

import json

import pytest

from pretia.inputs.generator import resolve_generator_model
from pretia.inputs.selector import read_inputs_file, select_input_mode

# ---------------------------------------------------------------------------
# Explicit inputs
# ---------------------------------------------------------------------------


class TestExplicitInputs:
    def test_manual_mode(self):
        sel = select_input_mode(explicit_inputs=["a", "b", "c"])
        assert sel.mode == "manual"
        assert sel.inputs == ["a", "b", "c"]
        assert "3" in sel.message


# ---------------------------------------------------------------------------
# File input
# ---------------------------------------------------------------------------


class TestFileInput:
    def test_plain_text(self, tmp_path):
        f = tmp_path / "inputs.txt"
        f.write_text("line one\nline two\n\nline three\n")
        sel = select_input_mode(inputs_file=str(f))
        assert sel.mode == "file"
        assert sel.inputs == ["line one", "line two", "line three"]

    def test_jsonl(self, tmp_path):
        f = tmp_path / "inputs.jsonl"
        lines = [
            json.dumps("How do I reset?"),
            json.dumps({"query": "billing", "lang": "en"}),
        ]
        f.write_text("\n".join(lines))
        sel = select_input_mode(inputs_file=str(f))
        assert sel.mode == "file"
        assert sel.inputs[0] == "How do I reset?"
        assert json.loads(sel.inputs[1]) == {
            "query": "billing",
            "lang": "en",
        }

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError, match="nope.txt"):
            select_input_mode(inputs_file="nope.txt")


# ---------------------------------------------------------------------------
# Single input
# ---------------------------------------------------------------------------


class TestSingleInput:
    def test_single_mode(self):
        sel = select_input_mode(single_input="test query")
        assert sel.mode == "single"
        assert sel.inputs == ["test query"]


# ---------------------------------------------------------------------------
# Langfuse flag
# ---------------------------------------------------------------------------


class TestLangfuseFlag:
    def test_langfuse_mode(self):
        sel = select_input_mode(from_langfuse=True)
        assert sel.mode == "langfuse"
        assert sel.inputs == []


# ---------------------------------------------------------------------------
# Auto-generate explicit
# ---------------------------------------------------------------------------


class TestAutoGenerate:
    def test_auto_generate_mode(self):
        sel = select_input_mode(auto_generate=30)
        assert sel.mode == "auto-generate"
        assert "30" in sel.message


# ---------------------------------------------------------------------------
# Auto-detect
# ---------------------------------------------------------------------------


class TestAutoDetect:
    def test_langfuse_credentials_no_auto_detect(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        sel = select_input_mode()
        assert sel.mode != "langfuse"

    def test_langfuse_explicit_flag(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        sel = select_input_mode(from_langfuse=True)
        assert sel.mode == "langfuse"

    def test_api_key_with_prompt(self, monkeypatch):
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        sel = select_input_mode(
            system_prompt="You are a support agent",
        )
        assert sel.mode == "auto-generate"

    def test_nothing_available(self, monkeypatch):
        for var in (
            "LANGFUSE_PUBLIC_KEY",
            "LANGFUSE_SECRET_KEY",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DEEPSEEK_API_KEY",
            "DASHSCOPE_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        sel = select_input_mode()
        assert sel.mode == "estimate"

    def test_deepseek_key_triggers_auto_generate(self, monkeypatch):
        for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        sel = select_input_mode()
        assert sel.mode == "auto-generate"


# ---------------------------------------------------------------------------
# resolve_generator_model
# ---------------------------------------------------------------------------


class TestResolveGeneratorModel:
    def test_explicit_model_returned(self):
        assert resolve_generator_model("gpt-4o") == "gpt-4o"

    def test_deepseek_default(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
        assert resolve_generator_model(None) == "deepseek-v4-flash"

    def test_falls_back_to_openai(self, monkeypatch):
        for var in ("DEEPSEEK_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        result = resolve_generator_model(None)
        assert result == "gpt-4o-mini"

    def test_falls_back_to_anthropic(self, monkeypatch):
        for var in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
            monkeypatch.delenv(var, raising=False)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        result = resolve_generator_model(None)
        assert result == "claude-haiku-4-5"

    def test_no_keys_returns_default(self, monkeypatch):
        for var in (
            "DEEPSEEK_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DASHSCOPE_API_KEY",
        ):
            monkeypatch.delenv(var, raising=False)
        assert resolve_generator_model(None) == "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# Priority order
# ---------------------------------------------------------------------------


class TestPriorityOrder:
    def test_explicit_wins_over_auto_generate(self):
        sel = select_input_mode(
            explicit_inputs=["a"],
            auto_generate=20,
        )
        assert sel.mode == "manual"

    def test_file_wins_over_single(self, tmp_path):
        f = tmp_path / "inputs.txt"
        f.write_text("from file\n")
        sel = select_input_mode(
            inputs_file=str(f),
            single_input="query",
        )
        assert sel.mode == "file"

    def test_single_wins_over_langfuse(self):
        sel = select_input_mode(
            single_input="query",
            from_langfuse=True,
        )
        assert sel.mode == "single"

    def test_langfuse_wins_over_auto_generate(self):
        sel = select_input_mode(
            from_langfuse=True,
            auto_generate=20,
        )
        assert sel.mode == "langfuse"


# ---------------------------------------------------------------------------
# read_inputs_file standalone
# ---------------------------------------------------------------------------


class TestReadInputsFile:
    def test_skips_blank_lines(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("a\n\n\nb\n")
        assert read_inputs_file(str(f)) == ["a", "b"]

    def test_jsonl_string_values(self, tmp_path):
        f = tmp_path / "test.jsonl"
        f.write_text(json.dumps("hello") + "\n" + json.dumps("world"))
        assert read_inputs_file(str(f)) == ["hello", "world"]

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            read_inputs_file("/nonexistent/path.txt")
