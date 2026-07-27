"""Tests for _module_uses_sdk and its integration with _select_collector."""

from __future__ import annotations

import types

from pretia.runner import _module_uses_sdk


class TestModuleUsesSDK:
    def test_detects_module_object_binding(self):
        """A module-level `anthropic = <module 'anthropic'>` binding is detected."""
        mod = types.ModuleType("fake_workflow")
        mod.anthropic = types.ModuleType("anthropic")  # type: ignore[attr-defined]
        assert _module_uses_sdk(mod, "anthropic") is True

    def test_detects_symbol_import(self):
        """An imported class whose __module__ starts with the SDK package is detected."""
        mod = types.ModuleType("fake_workflow")

        class FakeClient:
            __module__ = "anthropic.resources"

        mod.Client = FakeClient  # type: ignore[attr-defined]
        assert _module_uses_sdk(mod, "anthropic") is True

    def test_dotted_prefix_no_false_positive(self):
        """A module named 'openai_agents' must NOT match sdk_name='openai'."""
        mod = types.ModuleType("fake_workflow")
        mod.agents = types.ModuleType("openai_agents")  # type: ignore[attr-defined]
        assert _module_uses_sdk(mod, "openai") is False

    def test_clean_module_returns_false(self):
        """An empty module with no SDK imports returns False."""
        mod = types.ModuleType("empty")
        assert _module_uses_sdk(mod, "anthropic") is False

    def test_detects_langchain_openai(self):
        """An object whose __module__ is 'langchain_openai.chat_models' is detected."""
        mod = types.ModuleType("fake_workflow")

        class ChatOpenAI:
            __module__ = "langchain_openai.chat_models"

        mod.ChatOpenAI = ChatOpenAI  # type: ignore[attr-defined]
        assert _module_uses_sdk(mod, "langchain_openai") is True
