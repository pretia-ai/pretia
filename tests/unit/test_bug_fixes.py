"""Tests for specific bug fixes (BUG-1, BUG-7, BUG-16, BUG-LLM-PICKUP)."""

from __future__ import annotations

import asyncio
import types
from unittest.mock import patch

import pytest

from pretia.runner import (
    _detect_graph_input_key,
    _find_workflow,
    _is_llm_model,
)

# ---------------------------------------------------------------------------
# BUG-1: _find_workflow prefers ainvoke-capable canonical names
# ---------------------------------------------------------------------------


class TestFindWorkflowPreference:
    def test_prefers_app_with_ainvoke_over_graph_without(self):
        mod = types.ModuleType("test_mod")

        class FakeStateGraph:
            pass

        class FakeCompiledGraph:
            async def ainvoke(self, payload, config=None):
                pass

            def invoke(self, payload, config=None):
                pass

            nodes = {}

        mod.graph = FakeStateGraph()
        mod.app = FakeCompiledGraph()

        result = _find_workflow(mod, None)
        assert result is mod.app
        assert hasattr(result, "ainvoke")

    def test_returns_graph_if_it_has_ainvoke(self):
        mod = types.ModuleType("test_mod")

        class FakeCompiledGraph:
            async def ainvoke(self, payload, config=None):
                pass

        mod.graph = FakeCompiledGraph()

        result = _find_workflow(mod, None)
        assert result is mod.graph

    def test_falls_back_to_non_ainvoke_canonical_name(self):
        mod = types.ModuleType("test_mod")
        mod.workflow = lambda x: x

        result = _find_workflow(mod, None)
        assert result is mod.workflow


# ---------------------------------------------------------------------------
# BUG-7: Langfuse import produces friendly error
# ---------------------------------------------------------------------------


class TestLangfuseImportGuard:
    def test_create_langfuse_client_friendly_error(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")

        blocked = {"langfuse": None, "langfuse.api": None, "langfuse.api.client": None}
        with patch.dict("sys.modules", blocked):
            from pretia.inputs.importer import create_langfuse_client

            with pytest.raises(ImportError, match="pip install pretia\\[langfuse\\]"):
                create_langfuse_client()


# ---------------------------------------------------------------------------
# BUG-16: _safe_cost returns 0.0 silently for empty model
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-LLM-PICKUP: _find_workflow must skip LLM classes/instances and discover
# compiled graphs via build_graph() when no canonical name exists at module level.
# Regression: ChatAnthropic class has ainvoke, so old step-2 scanner returned the
# class. Calling ChatAnthropic.ainvoke(payload) without an instance meant payload
# filled self, leaving input missing → TypeError.
# ---------------------------------------------------------------------------


class _FakeCompiledGraph:
    """Minimal stand-in for a compiled LangGraph."""

    async def ainvoke(self, payload, config=None):
        return payload

    def invoke(self, payload, config=None):
        return payload

    nodes = {"agent": ..., "tools": ...}


class _FakeLLMClass:
    """Simulates a LangChain chat model class sitting in module namespace."""

    async def ainvoke(self, input, config=None):
        return input

    def invoke(self, input, config=None):
        return input


class _FakeToolObject:
    """Simulates a @tool-decorated StructuredTool (has ainvoke but is not a graph)."""

    async def ainvoke(self, input, config=None):
        return input

    def invoke(self, input, config=None):
        return input


class TestIsLlmModel:
    def test_detects_class_by_name(self):
        cls = type("ChatAnthropic", (), {})
        assert _is_llm_model(cls)

    def test_detects_instance_by_class_name(self):
        cls = type("ChatOpenAI", (), {})
        assert _is_llm_model(cls())

    def test_detects_base_chat_model(self):
        cls = type("BaseChatModel", (), {})
        assert _is_llm_model(cls())

    def test_rejects_compiled_graph(self):
        assert not _is_llm_model(_FakeCompiledGraph())

    def test_rejects_plain_callable(self):
        assert not _is_llm_model(lambda x: x)

    def test_detects_subclass_via_mro(self):
        base = type("BaseChatModel", (), {})
        sub = type("MyCustomLLM", (base,), {})
        assert _is_llm_model(sub)


class TestFindWorkflowSkipsLLMClasses:
    def test_skips_chat_anthropic_class_in_namespace(self):
        """The exact scenario from the bug report: ChatAnthropic class in module."""
        mod = types.ModuleType("customer_support")
        mod.ChatAnthropic = type("ChatAnthropic", (), {"ainvoke": _FakeLLMClass.ainvoke})

        compiled = _FakeCompiledGraph()

        def build_graph():
            return compiled

        mod.build_graph = build_graph

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_skips_chat_openai_class_in_namespace(self):
        mod = types.ModuleType("customer_support")
        mod.ChatOpenAI = type("ChatOpenAI", (), {"ainvoke": _FakeLLMClass.ainvoke})

        compiled = _FakeCompiledGraph()
        mod.build_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_skips_both_llm_classes_finds_graph_builder(self):
        """Multi-provider module with both ChatAnthropic and ChatOpenAI."""
        mod = types.ModuleType("multi_provider")
        mod.ChatAnthropic = type("ChatAnthropic", (), {"ainvoke": _FakeLLMClass.ainvoke})
        mod.ChatOpenAI = type("ChatOpenAI", (), {"ainvoke": _FakeLLMClass.ainvoke})

        compiled = _FakeCompiledGraph()
        mod.build_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_skips_llm_instance_in_namespace(self):
        mod = types.ModuleType("test_mod")
        llm_cls = type("ChatAnthropic", (), {"ainvoke": _FakeLLMClass.ainvoke})
        mod.llm = llm_cls()

        compiled = _FakeCompiledGraph()
        mod.build_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled


class TestFindWorkflowBuildGraphDiscovery:
    def test_calls_build_graph_when_no_canonical_name(self):
        mod = types.ModuleType("test_mod")
        compiled = _FakeCompiledGraph()
        mod.build_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_calls_create_graph(self):
        mod = types.ModuleType("test_mod")
        compiled = _FakeCompiledGraph()
        mod.create_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_calls_make_graph(self):
        mod = types.ModuleType("test_mod")
        compiled = _FakeCompiledGraph()
        mod.make_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_ignores_builder_that_raises(self):
        mod = types.ModuleType("test_mod")

        def bad_builder():
            raise RuntimeError("missing config")

        mod.build_graph = bad_builder

        async def my_workflow(inp):
            return inp

        mod.my_workflow = my_workflow

        result = _find_workflow(mod, None)
        assert result is mod.my_workflow

    def test_ignores_builder_returning_non_invocable(self):
        mod = types.ModuleType("test_mod")
        mod.build_graph = lambda: {"not": "a graph"}

        async def my_workflow(inp):
            return inp

        mod.my_workflow = my_workflow

        result = _find_workflow(mod, None)
        assert result is mod.my_workflow

    def test_canonical_name_takes_priority_over_builder(self):
        mod = types.ModuleType("test_mod")

        canonical = _FakeCompiledGraph()
        mod.app = canonical

        builder_result = _FakeCompiledGraph()
        mod.build_graph = lambda: builder_result

        result = _find_workflow(mod, None)
        assert result is canonical


class TestFindWorkflowToolObjectsSkipped:
    def test_prefers_build_graph_over_tool_objects(self):
        """@tool-decorated functions produce StructuredTool with ainvoke."""
        mod = types.ModuleType("test_mod")
        mod.lookup_order = _FakeToolObject()
        mod.check_return = _FakeToolObject()

        compiled = _FakeCompiledGraph()
        mod.build_graph = lambda: compiled

        result = _find_workflow(mod, None)
        assert result is compiled


class TestFindWorkflowSkipsClasses:
    """Regression: step-2 scanner returned bare classes whose ainvoke is unbound.

    Calling ToolNode.ainvoke(payload) on the class puts payload in `self` and
    raises "missing 1 required positional argument: 'input'".
    """

    def test_skips_class_with_ainvoke_in_namespace(self):
        mod = types.ModuleType("test_mod")
        mod.ToolNode = type("ToolNode", (), {"ainvoke": _FakeLLMClass.ainvoke})

        compiled = _FakeCompiledGraph()
        mod.the_graph = compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_skips_tool_instances_in_scanner(self):
        """StructuredTool instances (BaseTool subclasses) must not win step 2."""
        mod = types.ModuleType("test_mod")
        base_tool = type("BaseTool", (), {"ainvoke": _FakeLLMClass.ainvoke})
        structured_tool = type("StructuredTool", (base_tool,), {})
        mod.aaa_lookup_order = structured_tool()

        compiled = _FakeCompiledGraph()
        mod.zzz_compiled = compiled

        result = _find_workflow(mod, None)
        assert result is compiled

    def test_class_only_module_falls_through_to_async_callable(self):
        mod = types.ModuleType("test_mod")
        mod.ToolNode = type("ToolNode", (), {"ainvoke": _FakeLLMClass.ainvoke})

        async def my_workflow(inp):
            return inp

        mod.my_workflow = my_workflow

        result = _find_workflow(mod, None)
        assert result is my_workflow


class TestLoadWorkflowModuleSysModules:
    """Regression: loaded workflow modules were not registered in sys.modules.

    With `from __future__ import annotations`, get_type_hints() (called by
    LangGraph's StateGraph on TypedDict schemas) resolves string annotations
    via sys.modules[cls.__module__].__dict__ — without registration it raises
    NameError, build_graph() silently fails, and discovery picks a wrong object.
    """

    def test_module_registered_in_sys_modules(self, tmp_path):
        import sys

        from pretia.runner import _load_workflow_module

        wf = tmp_path / "my_support_agent.py"
        wf.write_text("x = 1\n")

        mod = _load_workflow_module(str(wf))
        assert sys.modules.get("my_support_agent") is mod
        sys.modules.pop("my_support_agent", None)

    def test_type_hints_resolve_with_future_annotations(self, tmp_path):
        """The user's exact failure: TypedDict + Annotated + future annotations."""
        import sys

        from pretia.runner import _load_workflow_module

        wf = tmp_path / "typed_agent.py"
        wf.write_text(
            "from __future__ import annotations\n"
            "import typing\n"
            "from typing import Annotated\n"
            "from typing_extensions import TypedDict\n"
            "\n"
            "class AgentState(TypedDict):\n"
            "    messages: Annotated[list, 'reducer']\n"
            "\n"
            "def resolve_hints():\n"
            "    return typing.get_type_hints(AgentState, include_extras=True)\n"
        )

        mod = _load_workflow_module(str(wf))
        try:
            hints = mod.resolve_hints()
            assert "messages" in hints
        finally:
            sys.modules.pop("typed_agent", None)

    def test_existing_module_name_not_shadowed(self, tmp_path):
        import sys

        from pretia.runner import _load_workflow_module

        wf = tmp_path / "json.py"
        wf.write_text("x = 1\n")

        real_json = sys.modules["json"]
        mod = _load_workflow_module(str(wf))
        try:
            assert sys.modules["json"] is real_json
            assert sys.modules.get("_pretia_workflow_json") is mod
        finally:
            sys.modules.pop("_pretia_workflow_json", None)

    def test_failed_load_cleaned_from_sys_modules(self, tmp_path):
        import sys

        import click
        import pytest

        from pretia.runner import _load_workflow_module

        wf = tmp_path / "broken_agent.py"
        wf.write_text("raise RuntimeError('boom')\n")

        with pytest.raises(click.UsageError):
            _load_workflow_module(str(wf))
        assert "broken_agent" not in sys.modules

    def test_build_graph_discovery_end_to_end_with_future_annotations(self, tmp_path):
        """Full repro of the user's script shape without framework deps."""
        import sys

        from pretia.runner import _find_workflow, _load_workflow_module

        wf = tmp_path / "support_bot.py"
        wf.write_text(
            "from __future__ import annotations\n"
            "import typing\n"
            "from typing import Annotated\n"
            "from typing_extensions import TypedDict\n"
            "\n"
            "class ToolNode:\n"  # class with ainvoke, must NOT be picked
            "    async def ainvoke(self, input, config=None):\n"
            "        return input\n"
            "\n"
            "class AgentState(TypedDict):\n"
            "    messages: Annotated[list, 'reducer']\n"
            "\n"
            "class _Compiled:\n"
            "    nodes = {}\n"
            "    async def ainvoke(self, payload, config=None):\n"
            "        return payload\n"
            "\n"
            "def build_graph():\n"
            "    typing.get_type_hints(AgentState, include_extras=True)\n"
            "    return _Compiled()\n"
        )

        mod = _load_workflow_module(str(wf))
        try:
            result = _find_workflow(mod, None)
            assert type(result).__name__ == "_Compiled"
        finally:
            sys.modules.pop("support_bot", None)


class TestDetectGraphInputKey:
    def test_detects_messages_from_annotations(self):
        class FakeSchema:
            __annotations__ = {"messages": list, "provider": str}

        class FakeBuilder:
            schema = FakeSchema

        graph = types.SimpleNamespace(builder=FakeBuilder)
        assert _detect_graph_input_key(graph) == "messages"

    def test_detects_messages_from_channels(self):
        graph = types.SimpleNamespace(
            builder=None,
            channels={"messages": ..., "provider": ...},
        )
        assert _detect_graph_input_key(graph) == "messages"

    def test_falls_back_to_first_channel_key(self):
        graph = types.SimpleNamespace(builder=None, channels={"query": ...})
        assert _detect_graph_input_key(graph) == "query"

    def test_falls_back_to_input_when_no_schema(self):
        graph = types.SimpleNamespace(builder=None, channels=None)
        assert _detect_graph_input_key(graph) == "input"

    def test_prefers_messages_over_str_field(self):
        class FakeSchema:
            __annotations__ = {"query": str, "messages": list}

        class FakeBuilder:
            schema = FakeSchema

        graph = types.SimpleNamespace(builder=FakeBuilder)
        assert _detect_graph_input_key(graph) == "messages"


class TestMaybeWrapSyncMessagePayload:
    def test_ainvoke_wrapper_creates_human_message_for_messages_key(self, monkeypatch):
        import sys

        from pretia.collectors.generic import GenericCollector
        from pretia.runner import ProfileRunner

        # Stub langchain_core.messages: not installed in CI, and other test
        # modules replace langchain_core with a MagicMock in sys.modules.
        class FakeHumanMessage:
            def __init__(self, content):
                self.content = content

        messages_mod = types.ModuleType("langchain_core.messages")
        messages_mod.HumanMessage = FakeHumanMessage
        monkeypatch.setitem(sys.modules, "langchain_core.messages", messages_mod)

        class FakeSchema:
            __annotations__ = {"messages": list}

        class FakeBuilder:
            schema = FakeSchema

        received_payloads = []

        class FakeGraph:
            builder = FakeBuilder

            async def ainvoke(self, payload, config=None):
                received_payloads.append(payload)
                return payload

        graph = FakeGraph()
        collector = GenericCollector()
        wrapped = ProfileRunner._maybe_wrap_sync(graph, collector)

        asyncio.run(wrapped("Hello, world!"))

        assert len(received_payloads) == 1
        payload = received_payloads[0]
        assert "messages" in payload
        msgs = payload["messages"]
        assert len(msgs) == 1
        assert msgs[0].content == "Hello, world!"

    def test_ainvoke_wrapper_passes_dict_input_unchanged(self):
        from pretia.collectors.generic import GenericCollector
        from pretia.runner import ProfileRunner

        received_payloads = []

        class FakeGraph:
            builder = None
            channels = {"query": ...}

            async def ainvoke(self, payload, config=None):
                received_payloads.append(payload)
                return payload

        graph = FakeGraph()
        collector = GenericCollector()
        wrapped = ProfileRunner._maybe_wrap_sync(graph, collector)

        dict_input = {"query": "test", "extra": 42}
        asyncio.run(wrapped(dict_input))

        assert received_payloads[0] is dict_input

    def test_ainvoke_wrapper_uses_detected_key_for_non_messages(self):
        from pretia.collectors.generic import GenericCollector
        from pretia.runner import ProfileRunner

        received_payloads = []

        class FakeGraph:
            builder = None
            channels = {"query": ...}

            async def ainvoke(self, payload, config=None):
                received_payloads.append(payload)
                return payload

        graph = FakeGraph()
        collector = GenericCollector()
        wrapped = ProfileRunner._maybe_wrap_sync(graph, collector)

        asyncio.run(wrapped("search term"))

        assert received_payloads[0] == {"query": "search term"}

    def test_langgraph_collector_not_wrapped(self):
        """LangGraphCollector handles invocation itself; _maybe_wrap_sync is a no-op."""
        from pretia.collectors.langgraph import LangGraphCollector
        from pretia.runner import ProfileRunner

        graph = _FakeCompiledGraph()
        collector = LangGraphCollector()
        result = ProfileRunner._maybe_wrap_sync(graph, collector)
        assert result is graph


class TestSafeCostEmptyModel:
    def test_empty_model_returns_zero_no_warning(self, caplog):
        from pretia.pricing.tables import calculate_cost
        from pretia.projection.stats import _safe_cost

        result = _safe_cost(calculate_cost, "", 100, 50)
        assert result == 0.0
        assert "Unknown model" not in caplog.text

    def test_none_model_returns_zero_no_warning(self, caplog):
        from pretia.pricing.tables import calculate_cost
        from pretia.projection.stats import _safe_cost

        result = _safe_cost(calculate_cost, None, 100, 50)
        assert result == 0.0
        assert "Unknown model" not in caplog.text

    def test_valid_model_still_works(self):
        from pretia.pricing.tables import calculate_cost
        from pretia.projection.stats import _safe_cost

        result = _safe_cost(calculate_cost, "gpt-4o", 1000, 500)
        assert result > 0
