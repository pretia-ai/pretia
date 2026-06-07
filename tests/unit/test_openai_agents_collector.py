"""Tests for OpenAIAgentsCollector and AgentCostRunHooks (fully mocked, no openai-agents)."""

from __future__ import annotations

import hashlib
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the agents module before importing our collector
# ---------------------------------------------------------------------------

_mock_agents = MagicMock()


class _MockRunHooksBase:
    pass


_mock_lifecycle = MagicMock()
_mock_lifecycle.RunHooksBase = _MockRunHooksBase

_mock_agents.Agent = type("Agent", (), {})
_mock_agents.ModelResponse = type("ModelResponse", (), {})
_mock_agents.Runner = MagicMock()
_mock_agents.lifecycle = _mock_lifecycle
_mock_agents.lifecycle.RunHooksBase = _MockRunHooksBase

_saved_agents = sys.modules.get("agents")
_saved_agents_lifecycle = sys.modules.get("agents.lifecycle")
sys.modules["agents"] = _mock_agents
sys.modules["agents.lifecycle"] = _mock_lifecycle

from agentcost.collectors.openai_agents import (  # noqa: E402
    AgentCostRunHooks,
    OpenAIAgentsCollector,
    _build_fallback_steps,
    _detect_output_format,
    _extract_agent_name,
    _extract_model_name,
    _extract_tool_name,
)

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class MockUsage:
    def __init__(self, input_tokens=100, output_tokens=50, total_tokens=150):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.requests = 1


class MockModelResponse:
    def __init__(self, usage=None, output=None, response_id=None):
        self.usage = usage or MockUsage()
        self.output = output or []
        self.response_id = response_id


class MockOutputMessage:
    def __init__(self, text="Hello"):
        self.content = [MockTextContent(text)]


class MockTextContent:
    def __init__(self, text="Hello"):
        self.text = text


class MockAgent:
    def __init__(self, name="test_agent", model="gpt-4o", instructions="You are a helper."):
        self.name = name
        self.model = model
        self.instructions = instructions


class MockTool:
    def __init__(self, name="search_db"):
        self.name = name


class MockRunResult:
    def __init__(self, raw_responses=None, final_output="done"):
        self.raw_responses = raw_responses or []
        self.final_output = final_output


# ---------------------------------------------------------------------------
# Helper extraction functions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_extract_model_name_string(self):
        agent = MockAgent(model="gpt-4o-mini")
        assert _extract_model_name(agent) == "gpt-4o-mini"

    def test_extract_model_name_none(self):
        agent = MockAgent(model=None)
        assert _extract_model_name(agent) == "unknown"

    def test_extract_model_name_object(self):
        model_obj = MagicMock()
        model_obj.model = "gpt-4.1"
        agent = MagicMock()
        agent.model = model_obj
        assert _extract_model_name(agent) == "gpt-4.1"

    def test_extract_agent_name(self):
        agent = MockAgent(name="classifier")
        assert _extract_agent_name(agent) == "classifier"

    def test_extract_agent_name_fallback(self):
        agent = MagicMock(spec=[])
        assert _extract_agent_name(agent) == "agent"

    def test_extract_tool_name(self):
        tool = MockTool(name="web_search")
        assert _extract_tool_name(tool) == "web_search"

    def test_extract_tool_name_fallback(self):
        tool = MagicMock(spec=[])
        assert _extract_tool_name(tool) == "tool_call"

    def test_detect_output_format_json(self):
        assert _detect_output_format('{"intent": "billing"}') == "json"

    def test_detect_output_format_code(self):
        assert _detect_output_format("```python\nprint('hi')\n```") == "code"

    def test_detect_output_format_text(self):
        assert _detect_output_format("Here is your answer.") == "text"


# ---------------------------------------------------------------------------
# AgentCostRunHooks — LLM lifecycle
# ---------------------------------------------------------------------------


class TestHooksLLMLifecycle:
    @pytest.mark.asyncio
    async def test_hooks_capture_agent_step(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent(name="classifier", model="gpt-4o")
        context = MagicMock()

        await hooks.on_llm_start(
            context,
            agent,
            "You are a classifier.",
            [{"content": "Classify this"}],
        )
        resp = MockModelResponse(usage=MockUsage(input_tokens=200, output_tokens=80))
        await hooks.on_llm_end(context, agent, resp)

        assert len(hooks.steps) == 1
        rec = hooks.steps[0]
        assert rec.step_name == "classifier"
        assert rec.step_type == "llm"
        assert rec.model == "gpt-4o"
        assert rec.input_tokens == 200
        assert rec.output_tokens == 80

    @pytest.mark.asyncio
    async def test_hooks_system_prompt_hashed(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent(name="agent_a")
        prompt = "You are a support agent."

        await hooks.on_llm_start(MagicMock(), agent, prompt, [])
        await hooks.on_llm_end(MagicMock(), agent, MockModelResponse())

        rec = hooks.steps[0]
        expected_hash = hashlib.sha256(prompt.encode()).hexdigest()
        assert rec.system_prompt_hash == expected_hash
        assert rec.system_prompt_tokens == len(prompt) // 4

    @pytest.mark.asyncio
    async def test_hooks_null_system_prompt(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()

        await hooks.on_llm_start(MagicMock(), agent, None, [])
        await hooks.on_llm_end(MagicMock(), agent, MockModelResponse())

        rec = hooks.steps[0]
        expected_hash = hashlib.sha256(b"").hexdigest()
        assert rec.system_prompt_hash == expected_hash
        assert rec.system_prompt_tokens == 0

    @pytest.mark.asyncio
    async def test_hooks_context_size_from_usage(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()

        await hooks.on_llm_start(MagicMock(), agent, "prompt", [{"content": "hi"}])
        resp = MockModelResponse(usage=MockUsage(input_tokens=500, output_tokens=100))
        await hooks.on_llm_end(MagicMock(), agent, resp)

        rec = hooks.steps[0]
        assert rec.context_size == 500

    @pytest.mark.asyncio
    async def test_hooks_context_size_estimated_when_no_usage(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()

        await hooks.on_llm_start(
            MagicMock(),
            agent,
            "system prompt text",
            [{"content": "user message"}],
        )
        resp = MockModelResponse(usage=MockUsage(input_tokens=0, output_tokens=0))
        await hooks.on_llm_end(MagicMock(), agent, resp)

        rec = hooks.steps[0]
        assert rec.context_size > 0

    @pytest.mark.asyncio
    async def test_hooks_output_format_json(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()

        await hooks.on_llm_start(MagicMock(), agent, None, [])
        output_msg = MockOutputMessage(text='{"result": true}')
        resp = MockModelResponse(output=[output_msg])
        await hooks.on_llm_end(MagicMock(), agent, resp)

        assert hooks.steps[0].output_format == "json"


# ---------------------------------------------------------------------------
# AgentCostRunHooks — tool lifecycle
# ---------------------------------------------------------------------------


class TestHooksToolLifecycle:
    @pytest.mark.asyncio
    async def test_hooks_capture_tool_step(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()
        tool = MockTool(name="search_db")
        context = MagicMock()

        await hooks.on_tool_start(context, agent, tool)
        await hooks.on_tool_end(context, agent, tool, "search results")

        assert len(hooks.steps) == 1
        rec = hooks.steps[0]
        assert rec.step_type == "tool"
        assert rec.step_name == "search_db"
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.model == ""
        assert rec.duration_ms >= 0

    @pytest.mark.asyncio
    async def test_hooks_tool_timing(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()
        tool = MockTool(name="slow_tool")

        await hooks.on_tool_start(MagicMock(), agent, tool)
        time.sleep(0.02)
        await hooks.on_tool_end(MagicMock(), agent, tool, "result")

        assert hooks.steps[0].duration_ms >= 20


# ---------------------------------------------------------------------------
# AgentCostRunHooks — multi-agent / handoff
# ---------------------------------------------------------------------------


class TestHooksMultiAgent:
    @pytest.mark.asyncio
    async def test_hooks_capture_multiple_agents(self):
        hooks = AgentCostRunHooks()
        agent_1 = MockAgent(name="triage", model="gpt-4o-mini")
        agent_2 = MockAgent(name="resolver", model="gpt-4o")
        ctx = MagicMock()

        await hooks.on_llm_start(ctx, agent_1, "Triage prompt", [])
        await hooks.on_llm_end(
            ctx,
            agent_1,
            MockModelResponse(usage=MockUsage(input_tokens=100, output_tokens=30)),
        )

        await hooks.on_llm_start(ctx, agent_2, "Resolver prompt", [])
        await hooks.on_llm_end(
            ctx,
            agent_2,
            MockModelResponse(usage=MockUsage(input_tokens=300, output_tokens=150)),
        )

        assert len(hooks.steps) == 2
        assert hooks.steps[0].step_name == "triage"
        assert hooks.steps[0].model == "gpt-4o-mini"
        assert hooks.steps[1].step_name == "resolver"
        assert hooks.steps[1].model == "gpt-4o"

    @pytest.mark.asyncio
    async def test_hooks_handoff(self):
        hooks = AgentCostRunHooks()
        from_agent = MockAgent(name="triage")
        to_agent = MockAgent(name="billing_agent")

        await hooks.on_handoff(MagicMock(), from_agent, to_agent)

        assert len(hooks.steps) == 1
        rec = hooks.steps[0]
        assert rec.step_name == "handoff_billing_agent"
        assert rec.step_type == "tool"
        assert rec.input_tokens == 0


# ---------------------------------------------------------------------------
# AgentCostRunHooks — iteration counting
# ---------------------------------------------------------------------------


class TestHooksIterationCounting:
    @pytest.mark.asyncio
    async def test_hooks_iteration_counting(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent(name="loop_agent")
        ctx = MagicMock()

        for _ in range(3):
            await hooks.on_llm_start(ctx, agent, None, [])
            await hooks.on_llm_end(ctx, agent, MockModelResponse())

        assert [s.iteration for s in hooks.steps] == [1, 2, 3]

    @pytest.mark.asyncio
    async def test_hooks_different_steps_independent_iterations(self):
        hooks = AgentCostRunHooks()
        agent_a = MockAgent(name="agent_a")
        agent_b = MockAgent(name="agent_b")
        ctx = MagicMock()

        await hooks.on_llm_start(ctx, agent_a, None, [])
        await hooks.on_llm_end(ctx, agent_a, MockModelResponse())

        await hooks.on_llm_start(ctx, agent_b, None, [])
        await hooks.on_llm_end(ctx, agent_b, MockModelResponse())

        await hooks.on_llm_start(ctx, agent_a, None, [])
        await hooks.on_llm_end(ctx, agent_a, MockModelResponse())

        assert hooks.steps[0].iteration == 1  # agent_a first
        assert hooks.steps[1].iteration == 1  # agent_b first
        assert hooks.steps[2].iteration == 2  # agent_a second


# ---------------------------------------------------------------------------
# AgentCostRunHooks — reset
# ---------------------------------------------------------------------------


class TestHooksReset:
    @pytest.mark.asyncio
    async def test_hooks_reset_between_runs(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()
        ctx = MagicMock()

        await hooks.on_llm_start(ctx, agent, None, [])
        await hooks.on_llm_end(ctx, agent, MockModelResponse())
        assert len(hooks.steps) == 1

        hooks.reset()
        assert len(hooks.steps) == 0

        await hooks.on_llm_start(ctx, agent, None, [])
        await hooks.on_llm_end(ctx, agent, MockModelResponse())
        assert len(hooks.steps) == 1
        assert hooks.steps[0].iteration == 1


# ---------------------------------------------------------------------------
# AgentCostRunHooks — error resilience
# ---------------------------------------------------------------------------


class TestHooksErrorResilience:
    @pytest.mark.asyncio
    async def test_hooks_error_resilience_none_agent(self):
        hooks = AgentCostRunHooks()
        await hooks.on_llm_start(MagicMock(), None, None, [])
        assert len(hooks.steps) == 0

    @pytest.mark.asyncio
    async def test_hooks_end_without_start_no_crash(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()
        await hooks.on_llm_end(MagicMock(), agent, MockModelResponse())
        assert len(hooks.steps) == 0

    @pytest.mark.asyncio
    async def test_hooks_tool_end_without_start_no_crash(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()
        tool = MockTool()
        await hooks.on_tool_end(MagicMock(), agent, tool, "result")
        assert len(hooks.steps) == 0

    @pytest.mark.asyncio
    async def test_hooks_none_response_still_records(self):
        hooks = AgentCostRunHooks()
        agent = MockAgent()
        await hooks.on_llm_start(MagicMock(), agent, None, [])
        await hooks.on_llm_end(MagicMock(), agent, None)
        assert len(hooks.steps) == 1
        assert hooks.steps[0].input_tokens == 0
        assert hooks.steps[0].output_tokens == 0

    @pytest.mark.asyncio
    async def test_hooks_handoff_none_agents_still_records(self):
        hooks = AgentCostRunHooks()
        await hooks.on_handoff(MagicMock(), None, None)
        assert len(hooks.steps) == 1
        assert hooks.steps[0].step_name == "handoff_agent"


# ---------------------------------------------------------------------------
# Fallback token extraction from RunResult
# ---------------------------------------------------------------------------


class TestFallbackSteps:
    def test_build_fallback_from_raw_responses(self):
        responses = [
            MockModelResponse(usage=MockUsage(input_tokens=100, output_tokens=50)),
            MockModelResponse(usage=MockUsage(input_tokens=200, output_tokens=80)),
        ]
        steps = _build_fallback_steps(responses, "my_agent", "gpt-4o")

        assert len(steps) == 2
        assert steps[0].step_name == "my_agent"
        assert steps[0].model == "gpt-4o"
        assert steps[0].input_tokens == 100
        assert steps[0].iteration == 1
        assert steps[1].input_tokens == 200
        assert steps[1].iteration == 2

    def test_build_fallback_skips_zero_usage(self):
        responses = [
            MockModelResponse(usage=MockUsage(input_tokens=0, output_tokens=0)),
            MockModelResponse(usage=MockUsage(input_tokens=100, output_tokens=50)),
        ]
        steps = _build_fallback_steps(responses, "agent", "gpt-4o")

        assert len(steps) == 1
        assert steps[0].input_tokens == 100

    def test_build_fallback_empty_responses(self):
        steps = _build_fallback_steps([], "agent", "gpt-4o")
        assert steps == []


# ---------------------------------------------------------------------------
# OpenAIAgentsCollector.collect()
# ---------------------------------------------------------------------------


class TestCollect:
    @pytest.mark.asyncio
    async def test_collector_collect_basic(self):
        collector = OpenAIAgentsCollector()
        agent = MockAgent(name="helper", model="gpt-4o")

        async def mock_run(starting_agent, inp, *, hooks=None, **kwargs):
            if hooks:
                await hooks.on_llm_start(
                    MagicMock(),
                    starting_agent,
                    "You are helpful.",
                    [{"content": inp}],
                )
                await hooks.on_llm_end(
                    MagicMock(),
                    starting_agent,
                    MockModelResponse(usage=MockUsage(input_tokens=150, output_tokens=60)),
                )
            return MockRunResult()

        with patch.object(
            sys.modules["agents"].Runner,
            "run",
            side_effect=mock_run,
        ):
            runs = await collector.collect(agent, ["hello", "world"])

        assert len(runs) == 2
        assert len(runs[0]) == 1
        assert len(runs[1]) == 1
        assert runs[0][0].step_name == "helper"
        assert runs[0][0].model == "gpt-4o"
        assert runs[0][0].input_tokens == 150

    @pytest.mark.asyncio
    async def test_collector_collect_with_tools(self):
        collector = OpenAIAgentsCollector()
        agent = MockAgent(name="tool_user", model="gpt-4o")
        tool = MockTool(name="calculator")

        async def mock_run(starting_agent, inp, *, hooks=None, **kwargs):
            if hooks:
                await hooks.on_llm_start(
                    MagicMock(),
                    starting_agent,
                    None,
                    [{"content": inp}],
                )
                await hooks.on_tool_start(MagicMock(), starting_agent, tool)
                await hooks.on_tool_end(MagicMock(), starting_agent, tool, "42")
                await hooks.on_llm_end(
                    MagicMock(),
                    starting_agent,
                    MockModelResponse(usage=MockUsage(input_tokens=200, output_tokens=30)),
                )
            return MockRunResult()

        with patch.object(
            sys.modules["agents"].Runner,
            "run",
            side_effect=mock_run,
        ):
            runs = await collector.collect(agent, ["what is 6*7?"])

        assert len(runs) == 1
        steps = runs[0]
        assert len(steps) == 2
        tool_steps = [s for s in steps if s.step_type == "tool"]
        llm_steps = [s for s in steps if s.step_type == "llm"]
        assert len(tool_steps) == 1
        assert tool_steps[0].step_name == "calculator"
        assert len(llm_steps) == 1

    @pytest.mark.asyncio
    async def test_collector_fallback_to_result_usage(self):
        collector = OpenAIAgentsCollector()
        agent = MockAgent(name="fallback_agent", model="gpt-4o-mini")

        async def mock_run(starting_agent, inp, *, hooks=None, **kwargs):
            return MockRunResult(
                raw_responses=[
                    MockModelResponse(
                        usage=MockUsage(input_tokens=300, output_tokens=120),
                    ),
                ],
            )

        with patch.object(
            sys.modules["agents"].Runner,
            "run",
            side_effect=mock_run,
        ):
            runs = await collector.collect(agent, ["test"])

        assert len(runs) == 1
        assert len(runs[0]) == 1
        rec = runs[0][0]
        assert rec.step_name == "fallback_agent"
        assert rec.model == "gpt-4o-mini"
        assert rec.input_tokens == 300
        assert rec.output_tokens == 120

    @pytest.mark.asyncio
    async def test_collector_workflow_error_skips_input(self):
        collector = OpenAIAgentsCollector()
        agent = MockAgent()

        async def mock_run(starting_agent, inp, *, hooks=None, **kwargs):
            raise RuntimeError("API error")

        with patch.object(
            sys.modules["agents"].Runner,
            "run",
            side_effect=mock_run,
        ):
            runs = await collector.collect(agent, ["bad_input"])

        assert len(runs) == 1
        assert runs[0] == []

    @pytest.mark.asyncio
    async def test_collector_multiple_inputs(self):
        collector = OpenAIAgentsCollector()
        agent = MockAgent(name="multi")
        call_count = 0

        async def mock_run(starting_agent, inp, *, hooks=None, **kwargs):
            nonlocal call_count
            call_count += 1
            if hooks:
                await hooks.on_llm_start(MagicMock(), starting_agent, None, [])
                await hooks.on_llm_end(
                    MagicMock(),
                    starting_agent,
                    MockModelResponse(
                        usage=MockUsage(
                            input_tokens=call_count * 100,
                            output_tokens=call_count * 50,
                        ),
                    ),
                )
            return MockRunResult()

        with patch.object(
            sys.modules["agents"].Runner,
            "run",
            side_effect=mock_run,
        ):
            runs = await collector.collect(agent, ["a", "b", "c"])

        assert len(runs) == 3
        assert runs[0][0].input_tokens == 100
        assert runs[1][0].input_tokens == 200
        assert runs[2][0].input_tokens == 300


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------


class TestLazyImport:
    def test_lazy_import_in_collectors_package(self):
        from agentcost.collectors import OpenAIAgentsCollector

        assert OpenAIAgentsCollector is not None
        assert OpenAIAgentsCollector.__name__ == "OpenAIAgentsCollector"

    def test_import_agentcost_without_sdk(self):
        saved = {}
        for mod_name in list(sys.modules):
            if mod_name == "agents" or mod_name.startswith("agents."):
                saved[mod_name] = sys.modules.pop(mod_name)
        saved_oa = sys.modules.pop("agentcost.collectors.openai_agents", None)

        try:
            with patch.dict(
                sys.modules,
                {
                    "agents": None,
                    "agents.lifecycle": None,
                },
            ):
                with pytest.raises(ImportError, match="OpenAI Agents SDK"):
                    import importlib

                    importlib.import_module("agentcost.collectors.openai_agents")
        finally:
            sys.modules.update(saved)
            if saved_oa is not None:
                sys.modules["agentcost.collectors.openai_agents"] = saved_oa


# ---------------------------------------------------------------------------
# Runner auto-detection
# ---------------------------------------------------------------------------


class TestRunnerAutoDetection:
    def test_runner_detects_openai_agent(self):
        from agentcost.runner import ProfileRunner

        agent = MockAgent(name="detect_me", instructions="Be helpful")
        runner = ProfileRunner(workflow_path="fake.py", single_input="test")
        coll = runner._select_collector(agent)
        assert type(coll).__name__ == "OpenAIAgentsCollector"

    def test_runner_explicit_openai(self):
        from agentcost.runner import ProfileRunner

        runner = ProfileRunner(
            workflow_path="fake.py",
            single_input="test",
            collector="openai",
        )
        coll = runner._select_collector(object())
        assert type(coll).__name__ == "OpenAIAgentsCollector"

    def test_runner_generic_fallback_no_instructions(self):
        from agentcost.runner import ProfileRunner

        class _Plain:
            name = "has_name_only"

        runner = ProfileRunner(workflow_path="fake.py", single_input="test")
        coll = runner._select_collector(_Plain())
        assert type(coll).__name__ == "GenericCollector"
