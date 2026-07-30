"""Tests for RAG-specific capture: embedding calls, retriever steps, RAG detection."""

from __future__ import annotations

import asyncio
import contextvars
import uuid
from unittest.mock import MagicMock


def _embedding_response(model: str = "text-embedding-3-small", prompt_tokens: int = 42):
    response = MagicMock()
    response.model = model
    response.usage.prompt_tokens = prompt_tokens
    return response


class TestRecordFromEmbeddingResponse:
    def test_creates_retrieval_record(self):
        from pretia.collectors.openai_sdk import _record_from_embedding_response

        captured = []
        _record_from_embedding_response(
            _embedding_response(), 0, captured, "retrieve_embedding", 1
        )
        assert len(captured) == 1
        rec = captured[0]
        assert rec.step_type == "retrieval"
        assert rec.step_name == "retrieve_embedding"
        assert rec.model == "text-embedding-3-small"
        assert rec.input_tokens == 42
        assert rec.output_tokens == 0

    def test_no_usage_skips_record(self):
        from pretia.collectors.openai_sdk import _record_from_embedding_response

        response = MagicMock()
        response.usage = None
        captured = []
        _record_from_embedding_response(response, 0, captured, "x", 1)
        assert captured == []


class TestEmbeddingsWrapper:
    def test_sync_outside_context_passes_through(self):
        from pretia.collectors.openai_sdk import _make_embeddings_wrapper

        original = MagicMock(return_value="raw-response")
        wrapper = _make_embeddings_wrapper(original, is_async=False)
        assert wrapper(model="text-embedding-3-small", input="q") == "raw-response"
        original.assert_called_once()

    def test_sync_inside_context_records(self):
        from pretia.collectors import openai_sdk

        captured: list = []
        counters: dict = {}
        token = openai_sdk._run_ctx.set((captured, asyncio.Lock(), counters))
        try:
            original = MagicMock(return_value=_embedding_response(prompt_tokens=17))
            wrapper = openai_sdk._make_embeddings_wrapper(original, is_async=False)
            wrapper(model="text-embedding-3-small", input="q")
            wrapper(model="text-embedding-3-small", input="q2")
        finally:
            openai_sdk._run_ctx.reset(token)

        assert len(captured) == 2
        assert captured[0].step_type == "retrieval"
        assert captured[0].input_tokens == 17
        assert captured[0].iteration == 1
        assert captured[1].iteration == 2

    async def test_async_inside_context_records(self):
        from pretia.collectors import openai_sdk

        captured: list = []

        async def original(*args, **kwargs):
            return _embedding_response(prompt_tokens=9)

        async def run():
            token = openai_sdk._run_ctx.set((captured, asyncio.Lock(), {}))
            try:
                wrapper = openai_sdk._make_embeddings_wrapper(original, is_async=True)
                await wrapper(model="text-embedding-3-small", input="q")
            finally:
                openai_sdk._run_ctx.reset(token)

        ctx = contextvars.copy_context()
        await asyncio.get_event_loop().create_task(run(), context=ctx)
        assert len(captured) == 1
        assert captured[0].input_tokens == 9


class TestRetrieverCallbacks:
    def _handler(self):
        from pretia.collectors.langgraph import PretiaCallbackHandler

        return PretiaCallbackHandler()

    def test_start_end_produces_retrieval_record(self):
        handler = self._handler()
        run_id = uuid.uuid4()
        handler.on_retriever_start({"name": "VectorStoreRetriever"}, "query", run_id=run_id)
        handler.on_retriever_end([], run_id=run_id)

        assert len(handler.records) == 1
        rec = handler.records[0]
        assert rec.step_type == "retrieval"
        assert rec.step_name == "VectorStoreRetriever"
        assert rec.input_tokens == 0
        assert rec.output_tokens == 0

    def test_end_without_start_is_noop(self):
        handler = self._handler()
        handler.on_retriever_end([], run_id=uuid.uuid4())
        assert handler.records == []

    def test_error_clears_inflight(self):
        handler = self._handler()
        run_id = uuid.uuid4()
        handler.on_retriever_start({"name": "r"}, "query", run_id=run_id)
        handler.on_retriever_error(RuntimeError("boom"), run_id=run_id)
        assert run_id not in handler._inflight
        assert handler.records == []

    def test_fallback_step_name(self):
        handler = self._handler()
        run_id = uuid.uuid4()
        handler.on_retriever_start({}, "query", run_id=run_id)
        handler.on_retriever_end([], run_id=run_id)
        assert handler.records[0].step_name == "retriever"

    def test_iteration_increments_per_step(self):
        handler = self._handler()
        for _ in range(2):
            run_id = uuid.uuid4()
            handler.on_retriever_start({"name": "r"}, "q", run_id=run_id)
            handler.on_retriever_end([], run_id=run_id)
        assert [r.iteration for r in handler.records] == [1, 2]


class TestRagImportDetection:
    def _detect(self, source: str, tmp_path) -> bool:
        from pretia.cli import _detect_rag_imports

        wf = tmp_path / "wf.py"
        wf.write_text(source, encoding="utf-8")
        return _detect_rag_imports(str(wf))

    def test_detects_langchain_chroma(self, tmp_path):
        assert self._detect("from langchain_chroma import Chroma", tmp_path) is True

    def test_detects_weaviate(self, tmp_path):
        assert self._detect("import weaviate", tmp_path) is True

    def test_detects_pgvector(self, tmp_path):
        assert self._detect("from pgvector.sqlalchemy import Vector", tmp_path) is True

    def test_detects_legacy_chromadb(self, tmp_path):
        assert self._detect("import chromadb", tmp_path) is True

    def test_no_false_positive_on_plain_agent(self, tmp_path):
        assert self._detect("import openai\n\ndef agent(q): ...", tmp_path) is False


class TestEmbeddingCostInStats:
    def test_embedding_record_cost_counted(self):
        import dataclasses

        from pretia.collectors.openai_sdk import _record_from_embedding_response
        from pretia.projection.stats import compute_stats

        captured: list = []
        _record_from_embedding_response(
            _embedding_response(prompt_tokens=1_000_000), 0, captured, "embed", 1
        )
        rec = dataclasses.replace(captured[0])
        stats = compute_stats([[rec]])
        # text-embedding-3-small: $0.02 per M input tokens
        assert stats.cost_per_run.mean > 0
        assert abs(stats.cost_per_run.mean - 0.02) < 0.001
