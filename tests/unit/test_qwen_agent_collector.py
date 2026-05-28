"""Tests for QwenAgentCollector (fully mocked, no qwen-agent dependency required)."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock the qwen_agent modules before importing our collector
# ---------------------------------------------------------------------------

_mock_qwen_agent = MagicMock()
_mock_agent_module = MagicMock()
_mock_llm_base = MagicMock()
_mock_llm_schema = MagicMock()
_mock_tools_base = MagicMock()


class _MockQwenAgent:
    pass


class _MockBaseChatModel:
    pass


class _MockMessage:
    def __init__(self, role="assistant", content="", **kwargs):
        self.role = role
        self.content = content
        self.extra = kwargs.get("extra", {})
        self.function_call = kwargs.get("function_call")
        self.name = kwargs.get("name")


_mock_agent_module.Agent = _MockQwenAgent
_mock_llm_base.BaseChatModel = _MockBaseChatModel
_mock_llm_schema.Message = _MockMessage

sys.modules.setdefault("qwen_agent", _mock_qwen_agent)
sys.modules.setdefault("qwen_agent.agent", _mock_agent_module)
sys.modules.setdefault("qwen_agent.llm", MagicMock())
sys.modules.setdefault("qwen_agent.llm.base", _mock_llm_base)
sys.modules.setdefault("qwen_agent.llm.schema", _mock_llm_schema)
sys.modules.setdefault("qwen_agent.tools", MagicMock())
sys.modules.setdefault("qwen_agent.tools.base", _mock_tools_base)

from agentcost.collectors.qwen_agent import (  # noqa: E402, I001
    QwenAgentCollector,
    _CapturedCall,
    _InstrumentedChatModel,
    _detect_output_format,
    _estimate_tokens,
    _extract_agent_name,
    _extract_system_prompt,
    _extract_usage_from_dashscope_message,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------

class MockLLM:
    def __init__(self, model="qwen3.7-max"):
        self.model = model
        self._call_count = 0
        self._responses = []

    def set_responses(self, responses):
        self._responses = list(responses)

    def chat(self, messages=None, functions=None, stream=True,
             delta_stream=False, extra_generate_cfg=None):
        self._call_count += 1
        if self._responses:
            resp = self._responses.pop(0)
        else:
            resp = [_MockMessage(role="assistant", content="default response")]

        if stream:
            def _gen():
                yield resp
            return _gen()
        return resp


class MockAgent:
    def __init__(self, name="test_agent", model="qwen3.7-max",
                 system_message="You are a helpful assistant.",
                 function_map=None):
        self.name = name
        self.llm = MockLLM(model=model)
        self.system_message = system_message
        self.function_map = function_map or {}

    def run(self, messages):
        yield from self.llm.chat(messages=messages, stream=True)


class MockAgentWithToolCall(MockAgent):
    def run(self, messages):
        yield from self.llm.chat(messages=messages, stream=True)
        yield from self.llm.chat(messages=messages, stream=True)


class MockAgentWithLoop(MockAgent):
    def __init__(self, loop_count=3, **kwargs):
        super().__init__(**kwargs)
        self._loop_count = loop_count

    def run(self, messages):
        for _ in range(self._loop_count):
            yield from self.llm.chat(messages=messages, stream=True)


# ---------------------------------------------------------------------------
# Helper function tests
# ---------------------------------------------------------------------------

class TestHelperFunctions:
    def test_estimate_tokens(self):
        assert _estimate_tokens("hello world test") == 4

    def test_detect_output_format_json(self):
        assert _detect_output_format('{"result": true}') == "json"

    def test_detect_output_format_code(self):
        assert _detect_output_format("```python\nprint('hi')\n```") == "code"

    def test_detect_output_format_text(self):
        assert _detect_output_format("Here is your answer.") == "text"

    def test_extract_agent_name(self):
        agent = MockAgent(name="classifier")
        assert _extract_agent_name(agent) == "classifier"

    def test_extract_agent_name_fallback_to_class(self):
        agent = MagicMock(spec=[])
        agent.name = None
        assert _extract_agent_name(agent) == "MagicMock"

    def test_extract_system_prompt(self):
        agent = MockAgent(system_message="Be helpful.")
        assert _extract_system_prompt(agent) == "Be helpful."

    def test_extract_system_prompt_none(self):
        agent = MagicMock(spec=[])
        assert _extract_system_prompt(agent) == ""


# ---------------------------------------------------------------------------
# DashScope usage extraction
# ---------------------------------------------------------------------------

class TestUsageExtraction:
    def test_extract_usage_from_dashscope_dict(self):
        msg = _MockMessage(
            extra={
                "model_service_info": {
                    "usage": {"input_tokens": 200, "output_tokens": 80},
                },
            },
        )
        pt, ct = _extract_usage_from_dashscope_message(msg)
        assert pt == 200
        assert ct == 80

    def test_extract_usage_from_dashscope_prompt_tokens_key(self):
        msg = _MockMessage(
            extra={
                "model_service_info": {
                    "usage": {"prompt_tokens": 150, "completion_tokens": 60},
                },
            },
        )
        pt, ct = _extract_usage_from_dashscope_message(msg)
        assert pt == 150
        assert ct == 60

    def test_extract_usage_no_extra(self):
        msg = _MockMessage()
        pt, ct = _extract_usage_from_dashscope_message(msg)
        assert pt == 0
        assert ct == 0

    def test_extract_usage_from_dashscope_object(self):
        usage = SimpleNamespace(input_tokens=300, output_tokens=100,
                                prompt_tokens=0, completion_tokens=0)
        info = SimpleNamespace(usage=usage)
        msg = _MockMessage(extra={"model_service_info": info})
        pt, ct = _extract_usage_from_dashscope_message(msg)
        assert pt == 300
        assert ct == 100


# ---------------------------------------------------------------------------
# InstrumentedChatModel
# ---------------------------------------------------------------------------

class TestInstrumentedChatModel:
    def test_delegates_attributes(self):
        original = MockLLM(model="qwen-turbo")
        captured: list[_CapturedCall] = []
        instrumented = _InstrumentedChatModel(original, captured)
        assert instrumented.model == "qwen-turbo"

    def test_chat_non_stream_captures_call(self):
        original = MockLLM(model="qwen3.7-max")
        captured: list[_CapturedCall] = []
        instrumented = _InstrumentedChatModel(original, captured)

        instrumented.chat(messages=[], stream=False)
        assert len(captured) == 1
        assert captured[0].model == "qwen3.7-max"

    def test_chat_stream_captures_after_consumption(self):
        original = MockLLM(model="qwen3.6-plus")
        captured: list[_CapturedCall] = []
        instrumented = _InstrumentedChatModel(original, captured)

        gen = instrumented.chat(messages=[], stream=True)
        assert len(captured) == 0
        for _ in gen:
            pass
        assert len(captured) == 1
        assert captured[0].model == "qwen3.6-plus"

    def test_chat_captures_dashscope_usage(self):
        original = MockLLM(model="qwen3.7-max")
        original.set_responses([
            [_MockMessage(
                role="assistant",
                content="response text",
                extra={"model_service_info": {
                    "usage": {"input_tokens": 500, "output_tokens": 200},
                }},
            )],
        ])
        captured: list[_CapturedCall] = []
        instrumented = _InstrumentedChatModel(original, captured)

        instrumented.chat(messages=[], stream=False)
        assert captured[0].input_tokens == 500
        assert captured[0].output_tokens == 200

    def test_chat_captures_output_text(self):
        original = MockLLM()
        original.set_responses([
            [_MockMessage(role="assistant", content="hello world")],
        ])
        captured: list[_CapturedCall] = []
        instrumented = _InstrumentedChatModel(original, captured)

        instrumented.chat(messages=[], stream=False)
        assert "hello world" in captured[0].output_text

    def test_chat_detects_tool_call(self):
        fc = SimpleNamespace(name="search", arguments='{"q": "test"}')
        original = MockLLM()
        original.set_responses([
            [_MockMessage(role="assistant", content="", function_call=fc)],
        ])
        captured: list[_CapturedCall] = []
        instrumented = _InstrumentedChatModel(original, captured)

        instrumented.chat(messages=[], stream=False)
        assert captured[0].is_tool_call is True


# ---------------------------------------------------------------------------
# QwenAgentCollector.collect() — basic
# ---------------------------------------------------------------------------

class TestCollectorBasic:
    @pytest.mark.asyncio
    async def test_collector_basic(self):
        collector = QwenAgentCollector()
        agent = MockAgent(name="helper", model="qwen3.7-max")

        runs = await collector.collect(agent, ["hello", "world"])

        assert len(runs) == 2
        assert len(runs[0]) >= 1
        assert len(runs[1]) >= 1

    @pytest.mark.asyncio
    async def test_collector_captures_model_name(self):
        collector = QwenAgentCollector()
        agent = MockAgent(name="helper", model="qwen3.7-max")

        runs = await collector.collect(agent, ["test"])
        rec = runs[0][0]
        assert rec.model == "qwen3.7-max"

    @pytest.mark.asyncio
    async def test_collector_captures_step_name(self):
        collector = QwenAgentCollector()
        agent = MockAgent(name="classifier")

        runs = await collector.collect(agent, ["test"])
        assert runs[0][0].step_name == "classifier"

    @pytest.mark.asyncio
    async def test_collector_step_type_llm(self):
        collector = QwenAgentCollector()
        agent = MockAgent()

        runs = await collector.collect(agent, ["test"])
        assert runs[0][0].step_type == "llm"

    @pytest.mark.asyncio
    async def test_collector_system_prompt_hashed(self):
        collector = QwenAgentCollector()
        prompt = "You are a support agent."
        agent = MockAgent(system_message=prompt)

        runs = await collector.collect(agent, ["test"])
        rec = runs[0][0]
        expected_hash = hashlib.sha256(prompt.encode()).hexdigest()
        assert rec.system_prompt_hash == expected_hash
        assert rec.system_prompt_tokens == len(prompt) // 4

    @pytest.mark.asyncio
    async def test_collector_captures_tokens_from_dashscope(self):
        collector = QwenAgentCollector()
        agent = MockAgent(model="qwen3.6-plus")
        agent.llm.set_responses([
            [_MockMessage(
                role="assistant",
                content="classified as billing",
                extra={"model_service_info": {
                    "usage": {"input_tokens": 150, "output_tokens": 30},
                }},
            )],
        ])

        runs = await collector.collect(agent, ["test"])
        rec = runs[0][0]
        assert rec.input_tokens == 150
        assert rec.output_tokens == 30


# ---------------------------------------------------------------------------
# QwenAgentCollector — tool calls
# ---------------------------------------------------------------------------

class TestCollectorWithToolCalls:
    @pytest.mark.asyncio
    async def test_collector_with_multiple_llm_calls(self):
        collector = QwenAgentCollector()
        agent = MockAgentWithToolCall(name="tool_user", model="qwen3.7-max")

        runs = await collector.collect(agent, ["what is 6*7?"])

        assert len(runs) == 1
        assert len(runs[0]) == 2


# ---------------------------------------------------------------------------
# QwenAgentCollector — iteration counting
# ---------------------------------------------------------------------------

class TestCollectorIterationCounting:
    @pytest.mark.asyncio
    async def test_collector_iteration_counting(self):
        collector = QwenAgentCollector()
        agent = MockAgentWithLoop(
            loop_count=3, name="loop_agent", model="qwen3.7-max",
        )

        runs = await collector.collect(agent, ["test"])

        assert len(runs[0]) == 3
        assert [s.iteration for s in runs[0]] == [1, 2, 3]


# ---------------------------------------------------------------------------
# QwenAgentCollector — error resilience
# ---------------------------------------------------------------------------

class TestCollectorErrorResilience:
    @pytest.mark.asyncio
    async def test_collector_workflow_error_skips_input(self):
        collector = QwenAgentCollector()

        class FailingAgent(MockAgent):
            def run(self, messages):
                raise RuntimeError("API error")

        agent = FailingAgent()
        runs = await collector.collect(agent, ["bad_input"])

        assert len(runs) == 1
        assert runs[0] == []

    @pytest.mark.asyncio
    async def test_collector_no_llm_attribute(self):
        collector = QwenAgentCollector()

        class NoLLMAgent:
            name = "bare_agent"
            system_message = "test"

            def run(self, messages):
                yield [{"role": "assistant", "content": "response"}]

        agent = NoLLMAgent()
        runs = await collector.collect(agent, ["test"])

        assert len(runs) == 1
        assert runs[0] == []

    @pytest.mark.asyncio
    async def test_collector_restores_llm_after_error(self):
        collector = QwenAgentCollector()

        class FailingAgent(MockAgent):
            def run(self, messages):
                raise RuntimeError("fail")

        agent = FailingAgent(name="restore_test", model="qwen-turbo")
        original_llm = agent.llm

        await collector.collect(agent, ["test"])

        assert agent.llm is original_llm


# ---------------------------------------------------------------------------
# QwenAgentCollector — streaming
# ---------------------------------------------------------------------------

class TestCollectorStreaming:
    @pytest.mark.asyncio
    async def test_collector_streaming_captures_final_usage(self):
        collector = QwenAgentCollector()

        class StreamingAgent(MockAgent):
            def run(self, messages):
                yield from self.llm.chat(messages=messages, stream=True)

        agent = StreamingAgent(model="qwen3.7-max")
        agent.llm.set_responses([
            [_MockMessage(
                role="assistant",
                content="streamed response",
                extra={"model_service_info": {
                    "usage": {"input_tokens": 300, "output_tokens": 120},
                }},
            )],
        ])

        runs = await collector.collect(agent, ["test"])
        assert len(runs[0]) == 1
        rec = runs[0][0]
        assert rec.input_tokens == 300
        assert rec.output_tokens == 120


# ---------------------------------------------------------------------------
# Lazy import
# ---------------------------------------------------------------------------

class TestLazyImport:
    def test_lazy_import_in_collectors_package(self):
        from agentcost.collectors import QwenAgentCollector

        assert QwenAgentCollector is not None
        assert QwenAgentCollector.__name__ == "QwenAgentCollector"

    def test_import_agentcost_without_sdk(self):
        saved = {}
        for mod_name in list(sys.modules):
            if mod_name.startswith("qwen_agent"):
                saved[mod_name] = sys.modules.pop(mod_name)
        saved_qa = sys.modules.pop("agentcost.collectors.qwen_agent", None)

        try:
            with patch.dict(sys.modules, {
                "qwen_agent": None,
                "qwen_agent.agent": None,
                "qwen_agent.llm": None,
                "qwen_agent.llm.base": None,
                "qwen_agent.llm.schema": None,
                "qwen_agent.tools": None,
                "qwen_agent.tools.base": None,
            }):
                with pytest.raises(ImportError, match="Qwen-Agent"):
                    import importlib
                    importlib.import_module("agentcost.collectors.qwen_agent")
        finally:
            sys.modules.update(saved)
            if saved_qa is not None:
                sys.modules["agentcost.collectors.qwen_agent"] = saved_qa


# ---------------------------------------------------------------------------
# Runner auto-detection
# ---------------------------------------------------------------------------

class TestRunnerAutoDetection:
    def test_runner_detects_qwen_agent(self):
        from agentcost.runner import ProfileRunner

        agent = MockAgent(name="qwen_test", system_message="Be helpful")
        runner = ProfileRunner(workflow_path="fake.py", single_input="test")
        coll = runner._select_collector(agent)
        assert type(coll).__name__ == "QwenAgentCollector"

    def test_runner_explicit_qwen(self):
        from agentcost.runner import ProfileRunner

        runner = ProfileRunner(
            workflow_path="fake.py", single_input="test", collector="qwen",
        )
        coll = runner._select_collector(object())
        assert type(coll).__name__ == "QwenAgentCollector"

    def test_runner_openai_agent_not_confused_with_qwen(self):
        from agentcost.runner import ProfileRunner

        class OpenAIAgent:
            name = "oai"
            instructions = "Be helpful"

        runner = ProfileRunner(workflow_path="fake.py", single_input="test")
        coll = runner._select_collector(OpenAIAgent())
        assert type(coll).__name__ == "OpenAIAgentsCollector"
