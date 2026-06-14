"""Tests for graph/extractor.py: extracting step names from workflow objects."""

from __future__ import annotations

import types

from agentcost.graph.extractor import extract_step_names


class TestExtractStepNamesLangGraph:
    def test_dict_nodes(self):
        workflow = types.SimpleNamespace(nodes={"classify": None, "respond": None, "route": None})
        result = extract_step_names(workflow)
        assert result == ["classify", "respond", "route"]

    def test_empty_dict_nodes(self):
        workflow = types.SimpleNamespace(nodes={})
        result = extract_step_names(workflow)
        assert result == []


class TestExtractStepNamesOpenAIAgents:
    def test_handoffs_with_names(self):
        agent_a = types.SimpleNamespace(name="summarizer")
        agent_b = types.SimpleNamespace(name="reviewer")
        workflow = types.SimpleNamespace(handoffs=[agent_a, agent_b])
        result = extract_step_names(workflow)
        assert result == ["reviewer", "summarizer"]

    def test_empty_handoffs(self):
        workflow = types.SimpleNamespace(handoffs=[])
        result = extract_step_names(workflow)
        assert result is None

    def test_handoffs_without_name_attr(self):
        h1 = types.SimpleNamespace()
        workflow = types.SimpleNamespace(handoffs=[h1])
        result = extract_step_names(workflow)
        assert result is None


class TestExtractStepNamesUnsupported:
    def test_plain_object(self):
        workflow = types.SimpleNamespace(foo="bar")
        result = extract_step_names(workflow)
        assert result is None

    def test_none_input(self):
        result = extract_step_names(None)
        assert result is None


class TestExtractStepNamesSorted:
    def test_results_are_sorted(self):
        workflow = types.SimpleNamespace(nodes={"z_step": None, "a_step": None, "m_step": None})
        result = extract_step_names(workflow)
        assert result == ["a_step", "m_step", "z_step"]
