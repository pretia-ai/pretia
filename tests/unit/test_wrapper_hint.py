"""Test the deterministic wrapper snippet builder."""

from __future__ import annotations

import types

from pretia.wrapper_hint import build_wrapper_snippet


def _make_module(**attrs: object) -> types.ModuleType:
    mod = types.ModuleType("_test_module")
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


class TestBuildWrapperSnippet:
    def test_openai_symbol_import(self):
        """Module with `from openai import OpenAI` + SYSTEM_PROMPT + run_agent."""
        from unittest.mock import MagicMock

        openai_client = MagicMock()
        type(openai_client).__name__ = "OpenAI"
        type(openai_client).__module__ = "openai"

        prompt = "You are a helpful assistant. " * 5

        def run_agent(client, messages):
            pass

        mod = _make_module(
            client=openai_client,
            SYSTEM_PROMPT=prompt,
            run_agent=run_agent,
        )
        rejected = [("run_agent", run_agent)]

        snippet = build_wrapper_snippet(mod, rejected)
        assert "def workflow(user_input: str):" in snippet
        assert "OpenAI()" in snippet
        assert "SYSTEM_PROMPT" in snippet
        assert "run_agent(client, messages)" in snippet

    def test_anthropic_module_import(self):
        """Module with `import anthropic` (module-level binding)."""
        anthropic_mod = types.ModuleType("anthropic")

        mod = _make_module(anthropic=anthropic_mod)
        rejected = []

        snippet = build_wrapper_snippet(mod, rejected)
        assert "anthropic.Anthropic()" in snippet

    def test_no_provider_fallback(self):
        """Module with no detected provider produces generic stub."""
        mod = _make_module()
        snippet = build_wrapper_snippet(mod, [])
        assert "def workflow(user_input: str):" in snippet
        assert "your_agent(user_input)" in snippet

    def test_agent_loop_param_mapping(self):
        """Params named 'client' and 'messages' are mapped correctly."""

        def agent_loop(client, messages, extra):
            pass

        mod = _make_module()
        rejected = [("agent_loop", agent_loop)]

        snippet = build_wrapper_snippet(mod, rejected)
        assert "agent_loop(client, messages, ...)" in snippet

    def test_system_prompt_variable_detected(self):
        """A string > 50 chars matching the system prompt regex is used by name."""
        from unittest.mock import MagicMock

        openai_client = MagicMock()
        type(openai_client).__name__ = "OpenAI"
        type(openai_client).__module__ = "openai"

        prompt = "You are an AI assistant that helps users with tasks. Be concise and helpful."
        mod = _make_module(MY_PROMPT=prompt, client=openai_client)
        snippet = build_wrapper_snippet(mod, [])
        assert "MY_PROMPT" in snippet

    def test_snippet_ends_with_adjust_note(self):
        mod = _make_module()
        snippet = build_wrapper_snippet(mod, [])
        assert "Adjust to your code" in snippet
