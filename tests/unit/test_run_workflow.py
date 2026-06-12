"""Tests for agents/harness/run_workflow.py — harness utilities."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("litellm")
from click.exceptions import UsageError

from bt_agents.harness.run_workflow import load_agent, load_inputs, load_prompts

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "prompts")


class TestLoadAgent:
    def test_w1_returns_agent_with_execute(self):
        agent = load_agent("W1")
        assert hasattr(agent, "execute")

    def test_lowercase_works(self):
        agent = load_agent("w1")
        assert hasattr(agent, "execute")

    def test_w13_returns_agent(self):
        agent = load_agent("W13")
        assert hasattr(agent, "execute")

    def test_all_14_workflows_valid(self):
        for wid in [
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
        ]:
            agent = load_agent(wid)
            assert hasattr(agent, "execute"), f"{wid} agent missing execute method"

    def test_invalid_id_raises_usage_error(self):
        with pytest.raises(UsageError, match="Unknown workflow"):
            load_agent("INVALID")


class TestLoadPrompts:
    def test_w1_has_classify_respond(self):
        prompts = load_prompts("W1", _PROMPTS_DIR)
        assert "classify_respond" in prompts
        assert len(prompts["classify_respond"]) > 0

    def test_w2_has_three_prompts(self):
        prompts = load_prompts("W2", _PROMPTS_DIR)
        assert "intake_classify" in prompts
        assert "research_draft_loop" in prompts
        assert "final_review" in prompts

    def test_missing_manifest_raises(self):
        with pytest.raises(UsageError, match="Manifest not found"):
            load_prompts("W1", "/nonexistent/dir")

    def test_zero_padded_ids_match(self):
        prompts = load_prompts("W1", _PROMPTS_DIR)
        assert len(prompts) > 0


class TestLoadInputs:
    def test_json_objects_parsed(self, tmp_path):
        f = tmp_path / "inputs.jsonl"
        f.write_text('{"input": "hello"}\n{"input": "world"}\n')
        inputs = load_inputs(str(f))
        assert len(inputs) == 2
        assert inputs[0]["input"] == "hello"

    def test_plain_strings_wrapped(self, tmp_path):
        f = tmp_path / "inputs.jsonl"
        f.write_text('"just a string"\n')
        inputs = load_inputs(str(f))
        assert inputs[0]["input"] == "just a string"

    def test_empty_lines_skipped(self, tmp_path):
        f = tmp_path / "inputs.jsonl"
        f.write_text('{"input": "a"}\n\n\n{"input": "b"}\n')
        inputs = load_inputs(str(f))
        assert len(inputs) == 2

    def test_n_truncates(self, tmp_path):
        f = tmp_path / "inputs.jsonl"
        f.write_text('{"input": "a"}\n{"input": "b"}\n{"input": "c"}\n')
        inputs = load_inputs(str(f), n=2)
        assert len(inputs) == 2

    def test_seed_shuffles_deterministically(self, tmp_path):
        f = tmp_path / "inputs.jsonl"
        f.write_text('{"input": "a"}\n{"input": "b"}\n{"input": "c"}\n{"input": "d"}\n')
        inputs1 = load_inputs(str(f), seed=42)
        inputs2 = load_inputs(str(f), seed=42)
        assert [i["input"] for i in inputs1] == [i["input"] for i in inputs2]

    def test_missing_file_raises(self):
        with pytest.raises(UsageError, match="not found"):
            load_inputs("/nonexistent/inputs.jsonl")
