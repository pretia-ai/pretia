"""Test entrypoint validation helpers and _find_workflow contract enforcement."""

from __future__ import annotations

import types

import click
import pytest

from pretia.runner import (
    AmbiguousEntrypointError,
    EntrypointError,
    _callable_accepts_single_input,
    _find_workflow,
    _registered_tool_names,
    _signature_str,
)

# ---------------------------------------------------------------------------
# _callable_accepts_single_input
# ---------------------------------------------------------------------------


class TestCallableAcceptsSingleInput:
    def test_one_arg_accepted(self):
        def f(x):
            pass

        assert _callable_accepts_single_input(f) is True

    def test_zero_arg_rejected(self):
        def f():
            pass

        assert _callable_accepts_single_input(f) is False

    def test_two_required_rejected(self):
        def f(a, b):
            pass

        assert _callable_accepts_single_input(f) is False

    def test_one_required_one_default_accepted(self):
        def f(a, b=None):
            pass

        assert _callable_accepts_single_input(f) is True

    def test_var_positional_accepted(self):
        def f(*args):
            pass

        assert _callable_accepts_single_input(f) is True

    def test_required_keyword_only_rejected(self):
        def f(a, *, client):
            pass

        assert _callable_accepts_single_input(f) is False

    def test_optional_keyword_only_accepted(self):
        def f(a, *, client=None):
            pass

        assert _callable_accepts_single_input(f) is True

    def test_callable_instance_accepted(self):
        class Invoker:
            def __call__(self, x):
                pass

        assert _callable_accepts_single_input(Invoker()) is True

    def test_builtin_uninspectable(self):
        # C builtins like len cannot be introspected; the function falls through
        # to return True so Pretia can attempt the call at runtime.
        assert _callable_accepts_single_input(len) is True


# ---------------------------------------------------------------------------
# _find_workflow contract (module-level attribute scanning)
# ---------------------------------------------------------------------------


def _make_module(**attrs: object) -> types.ModuleType:
    """Build a synthetic module with the given attributes."""
    mod = types.ModuleType("_test_module")
    mod.__name__ = "_test_module"
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


class TestFindWorkflowContract:
    def test_cli_script_shape_rejected(self):
        """Module with only zero-arg main and multi-arg helper raises UsageError."""

        def main():
            pass

        def run_agent(client, messages):
            pass

        mod = _make_module(main=main, run_agent=run_agent)
        # Attach __module__ so the rejection filter considers them "local"
        main.__module__ = "_test_module"
        run_agent.__module__ = "_test_module"

        with pytest.raises(click.UsageError, match="main()") as exc_info:
            _find_workflow(mod)
        msg = str(exc_info.value)
        assert "run_agent(client, messages)" in msg
        assert "takes no arguments" in msg
        assert "takes 2 required arguments" in msg

    def test_main_plus_valid_workflow_picks_workflow(self):
        """When both main() and workflow(inp) exist, the valid one wins."""

        def main():
            pass

        def workflow(inp):
            pass

        mod = _make_module(main=main, workflow=workflow)
        result = _find_workflow(mod)
        assert result is workflow

    def test_step1b_skips_string(self):
        """A string attribute named 'workflow' is not a workflow candidate."""
        mod = _make_module(workflow="gpt-4")
        result = _find_workflow(mod)
        assert result is None

    def test_step1b_skips_class(self):
        """A bare class (not an instance) is not a workflow candidate."""
        mod = _make_module(agent=type("Agent", (), {}))
        result = _find_workflow(mod)
        assert result is None

    def test_zero_arg_async_main_rejected(self):
        """An async def main() with no args is rejected."""

        async def main():
            pass

        mod = _make_module(main=main)
        main.__module__ = "_test_module"

        with pytest.raises(click.UsageError, match="main()"):
            _find_workflow(mod)

    def test_two_valid_async_still_raises_ambiguity(self):
        """Two valid single-arg async functions trigger ambiguity error."""

        async def foo(x):
            pass

        async def bar(x):
            pass

        mod = _make_module(foo=foo, bar=bar)

        with pytest.raises(click.UsageError, match="multiple async"):
            _find_workflow(mod)


class TestSyncLastResortAmbiguity:
    """Step 5 must not silently pick among multiple single-arg functions.

    Regression: a raw-SDK helpdesk script had two single-arg tool functions
    (check_system_status, lookup_policy); discovery picked the alphabetically
    first one and profiled a tool stub instead of the agent.
    """

    def test_multiple_single_arg_functions_raise_ambiguity(self):
        def check_system_status(system_name):
            pass

        def lookup_policy(topic):
            pass

        mod = _make_module(check_system_status=check_system_status, lookup_policy=lookup_policy)

        with pytest.raises(click.UsageError) as exc_info:
            _find_workflow(mod)
        msg = str(exc_info.value)
        assert "multiple single-argument callables" in msg
        assert "check_system_status" in msg
        assert "lookup_policy" in msg
        assert "--entry-point" in msg

    def test_unique_single_arg_function_still_picked(self):
        def answer_question(query):
            pass

        mod = _make_module(answer_question=answer_question)
        assert _find_workflow(mod) is answer_question

    def test_local_function_preferred_over_imported(self):
        """A function defined in the workflow file wins over an imported one."""

        def my_agent(query):
            pass

        def imported_helper(x):
            pass

        my_agent.__module__ = "_test_module"
        imported_helper.__module__ = "some_library"

        mod = _make_module(my_agent=my_agent, imported_helper=imported_helper)
        assert _find_workflow(mod) is my_agent


# ---------------------------------------------------------------------------
# Explicit --entry-point
# ---------------------------------------------------------------------------


class TestExplicitEntryPoint:
    def test_entry_point_zero_arg_rejected(self):
        """--entry-point pointing at a zero-arg function raises with signature."""

        def main():
            pass

        mod = _make_module(main=main)

        with pytest.raises(click.UsageError, match="main()"):
            _find_workflow(mod, entry_point="main")

    def test_entry_point_invoke_object_accepted(self):
        """An object with ainvoke is returned without further validation."""

        class FakeGraph:
            def ainvoke(self, x):
                return x

        obj = FakeGraph()
        mod = _make_module(graph=obj)
        result = _find_workflow(mod, entry_point="graph")
        assert result is obj

    def test_entry_point_string_attr_rejected(self):
        """A string attribute cannot be profiled."""
        mod = _make_module(app="hello")

        with pytest.raises(click.UsageError, match="cannot be profiled"):
            _find_workflow(mod, entry_point="app")

    def test_entry_point_missing_name(self):
        """A non-existent attribute raises with 'not found'."""
        mod = _make_module()

        with pytest.raises(click.UsageError, match="not found"):
            _find_workflow(mod, entry_point="nonexistent")


# ---------------------------------------------------------------------------
# _signature_str
# ---------------------------------------------------------------------------


class TestSignatureStr:
    def test_normal_function(self):
        result = _signature_str("foo", lambda x: x)
        assert "foo(" in result

    def test_uninspectable_or_introspectable(self):
        result = _signature_str("len", len)
        assert result.startswith("len(")


# ---------------------------------------------------------------------------
# _registered_tool_names
# ---------------------------------------------------------------------------


class TestRegisteredToolNames:
    def test_dict_registry_collects_callable_names(self):
        def diagnose_issue(symptoms):
            pass

        def check_status(system):
            pass

        mod = _make_module(
            TOOL_DISPATCH={"diagnose_issue": diagnose_issue, "check_status": check_status},
        )
        names = _registered_tool_names(mod)
        assert "diagnose_issue" in names
        assert "check_status" in names

    def test_openai_schema_list_collects_names(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "lookup_policy",
                    "parameters": {"type": "object"},
                },
            },
        ]
        mod = _make_module(TOOLS=tools)
        names = _registered_tool_names(mod)
        assert "lookup_policy" in names

    def test_anthropic_flat_schema_collects_names(self):
        tools = [{"name": "search_docs", "input_schema": {"type": "object"}}]
        mod = _make_module(TOOLS=tools)
        names = _registered_tool_names(mod)
        assert "search_docs" in names

    def test_non_tool_dict_ignored(self):
        mod = _make_module(CONFIG={"key": "value", "count": 42})
        names = _registered_tool_names(mod)
        assert len(names) == 0


# ---------------------------------------------------------------------------
# Tool-registry exclusion in _find_workflow
# ---------------------------------------------------------------------------


class TestToolRegistryExclusion:
    def test_helpdesk_shape_raises_entrypoint_error(self):
        """Full helpdesk shape: tool funcs in TOOL_DISPATCH + TOOLS schema +
        zero-arg main + multi-arg run_agent → EntrypointError (not ambiguity)."""

        def diagnose_issue(symptoms):
            pass

        def check_system_status(system_name):
            pass

        def lookup_policy(topic):
            pass

        def run_agent(client, messages):
            pass

        def main():
            pass

        tool_dispatch = {
            "diagnose_issue": diagnose_issue,
            "check_system_status": check_system_status,
            "lookup_policy": lookup_policy,
        }
        tools = [
            {"type": "function", "function": {"name": "diagnose_issue"}},
            {"type": "function", "function": {"name": "check_system_status"}},
            {"type": "function", "function": {"name": "lookup_policy"}},
        ]

        mod = _make_module(
            diagnose_issue=diagnose_issue,
            check_system_status=check_system_status,
            lookup_policy=lookup_policy,
            run_agent=run_agent,
            main=main,
            TOOL_DISPATCH=tool_dispatch,
            TOOLS=tools,
        )
        main.__module__ = "_test_module"
        run_agent.__module__ = "_test_module"

        with pytest.raises(EntrypointError) as exc_info:
            _find_workflow(mod)
        assert exc_info.value.wrapper_snippet
        msg = str(exc_info.value)
        assert "No usable entrypoint" in msg
        assert "main()" in msg

    def test_tool_funcs_excluded_leaves_valid_workflow(self):
        """If a real workflow(inp) exists alongside tool stubs, it still wins."""

        def check_status(system):
            pass

        def workflow(inp):
            pass

        mod = _make_module(
            check_status=check_status,
            workflow=workflow,
            TOOL_DISPATCH={"check_status": check_status},
        )
        assert _find_workflow(mod) is workflow

    def test_ambiguity_still_raised_without_tool_registry(self):
        """Two single-arg functions with no tool registry still raise ambiguity."""

        def foo(x):
            pass

        def bar(x):
            pass

        mod = _make_module(foo=foo, bar=bar)

        with pytest.raises(AmbiguousEntrypointError):
            _find_workflow(mod)


# ---------------------------------------------------------------------------
# Class-based entrypoint discovery
# ---------------------------------------------------------------------------


class TestClassBasedEntrypoint:
    def test_class_with_run_method_discovered(self):
        """A class with zero-arg init and run(self, inp) is auto-discovered."""

        class MyAgent:
            __module__ = "_test_module"

            def run(self, inp):
                return f"handled: {inp}"

        mod = _make_module(MyAgent=MyAgent)
        result = _find_workflow(mod)
        assert result is not None
        assert result("hello") == "handled: hello"

    def test_class_fresh_instance_per_call(self):
        """Each invocation of the wrapper creates a fresh instance."""
        call_count = 0

        class StatefulAgent:
            __module__ = "_test_module"

            def __init__(self):
                nonlocal call_count
                call_count += 1

            def run(self, inp):
                return inp

        mod = _make_module(StatefulAgent=StatefulAgent)
        result = _find_workflow(mod)
        call_count = 0
        result("a")
        result("b")
        assert call_count == 2

    def test_class_with_call_method_discovered(self):
        """A class with __call__(self, inp) is discovered and returned."""

        class CallableAgent:
            __module__ = "_test_module"

            def __call__(self, inp):
                return f"called: {inp}"

        mod = _make_module(CallableAgent=CallableAgent)
        result = _find_workflow(mod)
        assert result is not None
        assert result("test") == "called: test"

    def test_class_needing_args_skipped(self):
        """A class requiring constructor arguments is not auto-discovered."""

        class NeedsArgs:
            __module__ = "_test_module"

            def __init__(self, api_key):
                self.key = api_key

            def run(self, inp):
                return inp

        mod = _make_module(NeedsArgs=NeedsArgs)
        assert _find_workflow(mod) is None

    def test_class_without_agent_methods_skipped(self):
        """A bare class with no run/call/etc method is not picked."""
        mod = _make_module(agent=type("Agent", (), {}))
        result = _find_workflow(mod)
        assert result is None

    async def test_async_run_method_returns_async_wrapper(self):
        """A class with async run(self, inp) produces an async wrapper."""

        class AsyncAgent:
            __module__ = "_test_module"

            async def run(self, inp):
                return f"async: {inp}"

        mod = _make_module(AsyncAgent=AsyncAgent)
        result = _find_workflow(mod)
        assert result is not None
        import asyncio

        assert asyncio.iscoroutinefunction(result)
        assert await result("hi") == "async: hi"

    def test_explicit_entry_point_class_instantiated(self):
        """--entry-point MyAgent where MyAgent is a class with .run()."""

        class MyAgent:
            def run(self, inp):
                return f"ep: {inp}"

        mod = _make_module(MyAgent=MyAgent)
        result = _find_workflow(mod, entry_point="MyAgent")
        assert result is not None
        assert result("test") == "ep: test"

    def test_explicit_entry_point_class_needs_args(self):
        """--entry-point on a class requiring args raises with hint."""

        class NeedsKey:
            def __init__(self, key):
                pass

            def run(self, inp):
                return inp

        mod = _make_module(NeedsKey=NeedsKey)

        with pytest.raises(click.UsageError, match="constructor arguments"):
            _find_workflow(mod, entry_point="NeedsKey")

    def test_multiple_agent_classes_raise_ambiguity(self):
        """Two classes with .run() raise AmbiguousEntrypointError."""

        class AgentA:
            __module__ = "_test_module"

            def run(self, inp):
                return inp

        class AgentB:
            __module__ = "_test_module"

            def run(self, inp):
                return inp

        mod = _make_module(AgentA=AgentA, AgentB=AgentB)

        with pytest.raises(AmbiguousEntrypointError):
            _find_workflow(mod)

    def test_imported_class_skipped(self):
        """A class with __module__ != the workflow module is not instantiated."""

        class ImportedAgent:
            __module__ = "some_library"

            def run(self, inp):
                return inp

        mod = _make_module(ImportedAgent=ImportedAgent)
        assert _find_workflow(mod) is None


# ---------------------------------------------------------------------------
# Module-level instance discovery
# ---------------------------------------------------------------------------


class TestInstanceEntrypoint:
    def test_instance_at_canonical_name(self):
        """Module-level instance named 'agent' with .run() is discovered."""

        class MyAgent:
            def run(self, inp):
                return f"inst: {inp}"

        instance = MyAgent()
        mod = _make_module(agent=instance)
        result = _find_workflow(mod)
        assert result is not None
        assert callable(result)

    def test_instance_with_chat_method(self):
        """Wider method vocabulary: 'chat' is recognized."""

        class Chatbot:
            __module__ = "_test_module"

            def chat(self, msg):
                return f"chat: {msg}"

        instance = Chatbot()
        mod = _make_module(bot=instance)
        result = _find_workflow(mod)
        assert result is not None


# ---------------------------------------------------------------------------
# Callable-instance arity guard (FastAPI/Flask protection)
# ---------------------------------------------------------------------------


class TestCallableInstanceGuard:
    def test_asgi_shaped_app_not_picked(self):
        """An ASGI-shaped callable named 'app' is not picked at step 1b."""

        class FakeASGI:
            def __call__(self, scope, receive, send):
                pass

        def real_workflow(inp):
            return inp

        mod = _make_module(app=FakeASGI(), real_workflow=real_workflow)
        result = _find_workflow(mod)
        assert result is real_workflow

    def test_genuine_callable_instance_picked(self):
        """A callable instance with single-arg __call__ named 'app' works."""

        class SimpleApp:
            def __call__(self, inp):
                return f"app: {inp}"

        mod = _make_module(app=SimpleApp())
        result = _find_workflow(mod)
        assert result is not None
        assert result("hello") == "app: hello"


# ---------------------------------------------------------------------------
# Agent factory discovery
# ---------------------------------------------------------------------------


class TestAgentFactory:
    def test_create_agent_returning_instance(self):
        """create_agent() returning an instance with .run() is discovered."""

        class MyAgent:
            def run(self, inp):
                return f"factory: {inp}"

        def create_agent():
            return MyAgent()

        mod = _make_module(create_agent=create_agent)
        result = _find_workflow(mod)
        assert result is not None
        assert result("test") == "factory: test"

    def test_factory_returning_invoke_object(self):
        """Factory returning an invoke-object is handled by existing 1c logic."""

        class FakeGraph:
            def ainvoke(self, x):
                return x

        def build_graph():
            return FakeGraph()

        mod = _make_module(build_graph=build_graph)
        result = _find_workflow(mod)
        assert hasattr(result, "ainvoke")


# ---------------------------------------------------------------------------
# functools.wraps false-rejection fallback
# ---------------------------------------------------------------------------


class TestWrappedFunction:
    def test_wraps_decorated_single_arg_accepted(self):
        """A @functools.wraps wrapper around a multi-arg fn is accepted."""
        import functools

        def multi(a, b, c):
            return a

        @functools.wraps(multi)
        def workflow(inp):
            return multi(inp, None, None)

        assert _callable_accepts_single_input(workflow) is True

    def test_unwrapped_multi_arg_still_rejected(self):
        """Without __wrapped__, a multi-arg fn is still rejected."""

        def multi(a, b, c):
            return a

        assert _callable_accepts_single_input(multi) is False

    def test_wraps_around_single_arg_still_works(self):
        """@functools.wraps around a single-arg fn: both signatures agree."""
        import functools

        def original(x):
            return x

        @functools.wraps(original)
        def wrapper(x):
            return original(x)

        assert _callable_accepts_single_input(wrapper) is True


# ---------------------------------------------------------------------------
# __all__ preference in discovery
# ---------------------------------------------------------------------------


class TestAllExport:
    def test_all_resolves_ambiguity(self):
        """__all__ listing one of two single-arg functions resolves ambiguity."""

        def workflow(inp):
            return inp

        def helper(x):
            return x

        mod = _make_module(workflow=workflow, helper=helper)
        mod.__all__ = ["workflow"]
        result = _find_workflow(mod)
        assert result is workflow

    def test_all_empty_intersection_ignored(self):
        """__all__ listing only constants does not eliminate candidates."""

        def foo(x):
            return x

        def bar(x):
            return x

        mod = _make_module(foo=foo, bar=bar)
        mod.__all__ = ["VERSION"]

        with pytest.raises(AmbiguousEntrypointError):
            _find_workflow(mod)
