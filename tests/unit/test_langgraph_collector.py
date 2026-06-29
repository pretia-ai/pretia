"""Tests for LangGraphCollector and PretiaCallbackHandler (fully mocked, no langchain)."""

from __future__ import annotations

import hashlib
import sys
import time
from unittest.mock import MagicMock, patch
from uuid import uuid4

import pytest

# Mock langchain_core before importing our module
_mock_lc = MagicMock()
_mock_lc.callbacks.BaseCallbackHandler = type(
    "BaseCallbackHandler", (), {"__init__": lambda self, *a, **kw: None}
)
_mock_lc.outputs.LLMResult = type("LLMResult", (), {})
sys.modules.setdefault("langchain_core", _mock_lc)
sys.modules.setdefault("langchain_core.callbacks", _mock_lc.callbacks)
sys.modules.setdefault("langchain_core.outputs", _mock_lc.outputs)

from pretia.collectors.langgraph import (  # noqa: E402
    LangGraphCollector,
    PretiaCallbackHandler,
)

# Reusable short aliases
_USER_MSG = [{"role": "user", "content": "hi"}]
_USAGE_10_5 = {"token_usage": {"prompt_tokens": 10, "completion_tokens": 5}}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _MockGeneration:
    def __init__(self, text="", generation_info=None):
        self.text = text
        self.generation_info = generation_info or {}


def _make_response(*, text="Hello", llm_output=None, generation_info=None):
    resp = MagicMock()
    resp.llm_output = llm_output
    resp.generations = [[_MockGeneration(text=text, generation_info=generation_info or {})]]
    return resp


def _serialized(*, model_name="gpt-4o", name="classify"):
    return {
        "name": name,
        "kwargs": {"model_name": model_name},
        "id": ["langchain", "chat_models", name],
    }


def _start_llm(handler, *, run_id, ser=None, messages=None):
    handler.on_chat_model_start(
        ser or _serialized(),
        messages or [_USER_MSG],
        run_id=run_id,
    )


# ---------------------------------------------------------------------------
# LLM call lifecycle
# ---------------------------------------------------------------------------


class TestLLMCallLifecycle:
    def test_start_and_end_produces_step_record(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()
        messages = [[{"role": "user", "content": "Hello"}]]

        _start_llm(handler, run_id=rid, messages=messages)
        resp = _make_response(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                }
            },
        )
        handler.on_llm_end(resp, run_id=rid)

        assert len(handler.records) == 1
        rec = handler.records[0]
        assert rec.step_type == "llm"
        assert rec.model == "gpt-4o"
        assert rec.input_tokens == 100
        assert rec.output_tokens == 50
        assert rec.step_name == "classify"


# ---------------------------------------------------------------------------
# Token extraction from multiple locations
# ---------------------------------------------------------------------------


class TestTokenExtraction:
    def test_tokens_from_llm_output(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()

        _start_llm(handler, run_id=rid)
        resp = _make_response(
            llm_output={
                "token_usage": {
                    "prompt_tokens": 200,
                    "completion_tokens": 80,
                }
            },
        )
        handler.on_llm_end(resp, run_id=rid)

        assert handler.records[0].input_tokens == 200
        assert handler.records[0].output_tokens == 80

    def test_tokens_from_generation_info(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()

        _start_llm(handler, run_id=rid)
        resp = _make_response(
            llm_output={},
            generation_info={
                "usage": {
                    "prompt_tokens": 300,
                    "completion_tokens": 120,
                }
            },
        )
        handler.on_llm_end(resp, run_id=rid)

        assert handler.records[0].input_tokens == 300
        assert handler.records[0].output_tokens == 120

    def test_no_usage_data_falls_back_to_zero(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()

        _start_llm(handler, run_id=rid)
        resp = _make_response(llm_output={}, generation_info={})
        handler.on_llm_end(resp, run_id=rid)

        rec = handler.records[0]
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0


# ---------------------------------------------------------------------------
# System prompt extraction
# ---------------------------------------------------------------------------


class TestSystemPrompt:
    def test_system_prompt_hashed(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()
        prompt_text = "You are a support agent."
        messages = [
            [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": "hi"},
            ]
        ]

        _start_llm(handler, run_id=rid, messages=messages)
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=rid,
        )

        rec = handler.records[0]
        expected = hashlib.sha256(prompt_text.encode()).hexdigest()
        assert rec.system_prompt_hash == expected
        assert rec.system_prompt_tokens == len(prompt_text) // 4


# ---------------------------------------------------------------------------
# Output format detection
# ---------------------------------------------------------------------------


class TestOutputFormat:
    def _run_with_text(self, text):
        handler = PretiaCallbackHandler()
        rid = uuid4()
        _start_llm(handler, run_id=rid)
        handler.on_llm_end(
            _make_response(text=text, llm_output=_USAGE_10_5),
            run_id=rid,
        )
        return handler.records[0].output_format

    def test_json_response(self):
        assert self._run_with_text('{"intent": "billing"}') == "json"

    def test_text_response(self):
        fmt = self._run_with_text("Here is your answer to the billing question.")
        assert fmt == "text"

    def test_code_response(self):
        text = "Here is the code:\n```python\nprint('hello')\n```"
        assert self._run_with_text(text) == "code"


# ---------------------------------------------------------------------------
# Tool call lifecycle
# ---------------------------------------------------------------------------


class TestToolCall:
    def test_tool_start_and_end_produces_record(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()

        handler.on_tool_start(
            {"name": "search_db"},
            "query string",
            run_id=rid,
        )
        handler.on_tool_end("result data", run_id=rid)

        assert len(handler.records) == 1
        rec = handler.records[0]
        assert rec.step_type == "tool"
        assert rec.step_name == "search_db"
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0
        assert rec.duration_ms >= 0


# ---------------------------------------------------------------------------
# Iteration counting
# ---------------------------------------------------------------------------


class TestIterationCounting:
    def test_same_step_increments(self):
        handler = PretiaCallbackHandler()
        resp = _make_response(llm_output=_USAGE_10_5)

        for _ in range(3):
            rid = uuid4()
            _start_llm(
                handler,
                run_id=rid,
                ser=_serialized(name="review"),
            )
            handler.on_llm_end(resp, run_id=rid)

        assert [r.iteration for r in handler.records] == [1, 2, 3]


# ---------------------------------------------------------------------------
# Resilience
# ---------------------------------------------------------------------------


class TestResilience:
    def test_end_without_start_no_crash(self):
        handler = PretiaCallbackHandler()
        handler.on_llm_end(_make_response(), run_id=uuid4())
        assert len(handler.records) == 0

    def test_missing_model_name_falls_back_to_unknown(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()

        handler.on_chat_model_start(
            {},
            [_USER_MSG],
            run_id=rid,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=rid,
        )

        assert handler.records[0].model == "unknown"

    def test_tool_end_without_start_no_crash(self):
        handler = PretiaCallbackHandler()
        handler.on_tool_end("output", run_id=uuid4())
        assert len(handler.records) == 0


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


class TestTiming:
    def test_duration_positive(self):
        handler = PretiaCallbackHandler()
        rid = uuid4()

        _start_llm(handler, run_id=rid)
        time.sleep(0.02)
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=rid,
        )

        assert handler.records[0].duration_ms >= 20


# ---------------------------------------------------------------------------
# LangGraphCollector.collect()
# ---------------------------------------------------------------------------


class TestCollect:
    @pytest.mark.asyncio
    async def test_collect_invokes_workflow_per_input(self):
        collector = LangGraphCollector()

        async def mock_ainvoke(payload, config=None):
            handler = config["callbacks"][0]
            rid = uuid4()
            _start_llm(
                handler,
                run_id=rid,
                ser=_serialized(model_name="gpt-4o-mini"),
            )
            handler.on_llm_end(
                _make_response(
                    llm_output={
                        "token_usage": {
                            "prompt_tokens": 50,
                            "completion_tokens": 20,
                        },
                    }
                ),
                run_id=rid,
            )

        mock_graph = MagicMock()
        mock_graph.ainvoke = mock_ainvoke

        runs = await collector.collect(mock_graph, ["input1", "input2"])

        assert len(runs) == 2
        assert len(runs[0]) == 1
        assert len(runs[1]) == 1
        assert runs[0][0].model == "gpt-4o-mini"
        assert runs[1][0].model == "gpt-4o-mini"

    @pytest.mark.asyncio
    async def test_collect_wraps_string_input_as_dict(self):
        collector = LangGraphCollector()
        received_payloads = []

        async def mock_ainvoke(payload, config=None):
            received_payloads.append(payload)

        mock_graph = MagicMock()
        mock_graph.ainvoke = mock_ainvoke

        await collector.collect(mock_graph, ["hello"])

        assert received_payloads == [{"input": "hello"}]


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# BUG-3: Chain hierarchy → node name resolution
# ---------------------------------------------------------------------------


class TestChainNodeNameResolution:
    def test_direct_parent_resolves_node_name(self):
        handler = PretiaCallbackHandler()
        node_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "classifier"}, {}, run_id=node_id)
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm_id,
            parent_run_id=node_id,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=llm_id,
        )

        assert handler.records[0].step_name == "classifier"

    def test_intermediate_wrapper_skipped(self):
        handler = PretiaCallbackHandler()
        node_id = uuid4()
        seq_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "responder"}, {}, run_id=node_id)
        handler.on_chain_start(
            {"name": "RunnableSequence"}, {}, run_id=seq_id, parent_run_id=node_id
        )
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm_id,
            parent_run_id=seq_id,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=llm_id,
        )

        assert handler.records[0].step_name == "responder"

    def test_deeply_nested_wrappers(self):
        handler = PretiaCallbackHandler()
        node_id = uuid4()
        seq_id = uuid4()
        bind_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start({"name": "extract"}, {}, run_id=node_id)
        handler.on_chain_start(
            {"name": "RunnableSequence"}, {}, run_id=seq_id, parent_run_id=node_id
        )
        handler.on_chain_start(
            {"name": "RunnableBinding"}, {}, run_id=bind_id, parent_run_id=seq_id
        )
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm_id,
            parent_run_id=bind_id,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=llm_id,
        )

        assert handler.records[0].step_name == "extract"

    def test_none_serialized_does_not_crash(self):
        handler = PretiaCallbackHandler()
        node_id = uuid4()
        internal_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(None, {}, run_id=internal_id)
        handler.on_chain_start({"name": "classifier"}, {}, run_id=node_id)
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm_id,
            parent_run_id=node_id,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=llm_id,
        )

        assert handler.records[0].step_name == "classifier"

    def test_real_langgraph_kwargs_name(self):
        """LangGraph passes serialized=None and node name via kwargs['name']."""
        handler = PretiaCallbackHandler()
        graph_id = uuid4()
        node_id = uuid4()
        llm_id = uuid4()

        handler.on_chain_start(None, {}, run_id=graph_id, name="LangGraph")
        handler.on_chain_start(
            None,
            {},
            run_id=node_id,
            parent_run_id=graph_id,
            name="classifier",
            metadata={"langgraph_node": "classifier"},
        )
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm_id,
            parent_run_id=node_id,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=llm_id,
        )

        assert handler.records[0].step_name == "classifier"

    def test_no_chain_falls_back_to_llm_class(self):
        handler = PretiaCallbackHandler()
        llm_id = uuid4()

        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm_id,
        )
        handler.on_llm_end(
            _make_response(llm_output=_USAGE_10_5),
            run_id=llm_id,
        )

        assert handler.records[0].step_name == "ChatOpenAI"

    def test_chain_end_cleans_up(self):
        handler = PretiaCallbackHandler()
        node_id = uuid4()

        handler.on_chain_start({"name": "classifier"}, {}, run_id=node_id)
        assert node_id in handler._active_chains
        assert node_id in handler._parent_chain

        handler.on_chain_end({}, run_id=node_id)
        assert node_id not in handler._active_chains
        assert node_id not in handler._parent_chain

    def test_two_nodes_get_distinct_names(self):
        handler = PretiaCallbackHandler()
        resp = _make_response(llm_output=_USAGE_10_5)

        node1_id = uuid4()
        llm1_id = uuid4()
        handler.on_chain_start({"name": "classifier"}, {}, run_id=node1_id)
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm1_id,
            parent_run_id=node1_id,
        )
        handler.on_llm_end(resp, run_id=llm1_id)
        handler.on_chain_end({}, run_id=node1_id)

        node2_id = uuid4()
        llm2_id = uuid4()
        handler.on_chain_start({"name": "responder"}, {}, run_id=node2_id)
        handler.on_chat_model_start(
            _serialized(name="ChatOpenAI"),
            [_USER_MSG],
            run_id=llm2_id,
            parent_run_id=node2_id,
        )
        handler.on_llm_end(resp, run_id=llm2_id)
        handler.on_chain_end({}, run_id=node2_id)

        assert handler.records[0].step_name == "classifier"
        assert handler.records[1].step_name == "responder"


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------


class TestImportGuard:
    def test_lazy_import_without_langchain(self):
        saved = {}
        for mod_name in list(sys.modules):
            if mod_name == "langchain_core" or mod_name.startswith("langchain_core."):
                saved[mod_name] = sys.modules.pop(mod_name)
        saved_lg = sys.modules.pop(
            "pretia.collectors.langgraph",
            None,
        )

        try:
            with patch.dict(
                sys.modules,
                {
                    "langchain_core": None,
                    "langchain_core.callbacks": None,
                    "langchain_core.outputs": None,
                },
            ):
                with pytest.raises(ImportError, match="langchain-core"):
                    import importlib

                    importlib.import_module(
                        "pretia.collectors.langgraph",
                    )
        finally:
            sys.modules.update(saved)
            if saved_lg is not None:
                sys.modules["pretia.collectors.langgraph"] = saved_lg
