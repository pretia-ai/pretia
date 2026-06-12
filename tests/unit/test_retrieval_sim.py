"""Tests for agents.harness.retrieval_sim — cosine similarity, mock corpus, and retrieval."""

from __future__ import annotations

import math
import random

import pytest

from bt_agents.harness.retrieval_sim import (
    Corpus,
    CorpusChunk,
    RetrievedChunk,
    _build_mock_corpus,
    _MOCK_EMBED_DIM,
    cosine_similarity,
    load_corpus,
    retrieve,
)
from bt_agents.providers.embeddings import _DEFAULT_EMBEDDING_DIM


# ── cosine_similarity ────────────────────────────────────────────────────


class TestCosineSimilarity:
    """Verify cosine_similarity for canonical geometric cases."""

    def test_identical_unit_vectors(self) -> None:
        """Identical unit vectors produce similarity 1.0."""
        assert cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors produce similarity 0.0."""
        assert cosine_similarity([1, 0, 0], [0, 1, 0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors produce similarity -1.0."""
        assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector_returns_zero(self) -> None:
        """A zero vector against any vector returns 0.0, not NaN."""
        result = cosine_similarity([0, 0, 0], [1, 0, 0])
        assert result == pytest.approx(0.0)
        assert not math.isnan(result)

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched vector lengths raise ValueError."""
        with pytest.raises(ValueError, match="Vector length mismatch"):
            cosine_similarity([1, 0], [1, 0, 0])

    def test_result_bounded_for_random_vectors(self) -> None:
        """Cosine similarity of random vectors stays in [-1, 1]."""
        rng = random.Random(99)
        for _ in range(50):
            a = [rng.gauss(0, 1) for _ in range(64)]
            b = [rng.gauss(0, 1) for _ in range(64)]
            sim = cosine_similarity(a, b)
            assert -1.0 - 1e-9 <= sim <= 1.0 + 1e-9


# ── _build_mock_corpus ───────────────────────────────────────────────────


class TestBuildMockCorpus:
    """Validate the canned mock corpus used for development and testing."""

    def test_mock_corpus_has_10_chunks(self) -> None:
        """Mock corpus contains exactly 10 chunks."""
        corpus = _build_mock_corpus()
        assert isinstance(corpus, Corpus)
        assert len(corpus.chunks) == 10

    def test_embedding_dimension_matches_mock_dim(self) -> None:
        """Each mock embedding is _MOCK_EMBED_DIM-dimensional."""
        corpus = _build_mock_corpus()
        for chunk in corpus.chunks:
            assert len(chunk.embedding) == _MOCK_EMBED_DIM

    def test_deterministic_across_calls(self) -> None:
        """Two calls return identical chunk_ids in the same order."""
        ids_a = [c.chunk_id for c in _build_mock_corpus().chunks]
        ids_b = [c.chunk_id for c in _build_mock_corpus().chunks]
        assert ids_a == ids_b

    def test_chunks_have_nonempty_text_and_document_name(self) -> None:
        """Every chunk has non-empty text and document_name."""
        corpus = _build_mock_corpus()
        for chunk in corpus.chunks:
            assert chunk.text.strip(), f"Chunk {chunk.chunk_id} has empty text"
            assert chunk.document_name.strip(), (
                f"Chunk {chunk.chunk_id} has empty document_name"
            )

    def test_provider_metadata_present(self) -> None:
        """All 10 mock chunks carry metadata['provider']."""
        corpus = _build_mock_corpus()
        providers = [
            c.metadata["provider"]
            for c in corpus.chunks
            if "provider" in c.metadata
        ]
        assert len(providers) == 10


# ── retrieve ─────────────────────────────────────────────────────────────


class TestRetrieve:
    """Test the retrieve function using the mock corpus."""

    @pytest.fixture()
    def corpus(self) -> Corpus:
        return _build_mock_corpus()

    @pytest.fixture()
    def query_embedding(self, corpus: Corpus) -> list[float]:
        """Use the first chunk's embedding as the query for predictable results."""
        return corpus.chunks[0].embedding

    def test_top_k_returns_exact_count(
        self, query_embedding: list[float], corpus: Corpus
    ) -> None:
        """top_k=3 returns exactly 3 results."""
        results = retrieve(query_embedding, corpus, top_k=3)
        assert len(results) == 3
        assert all(isinstance(r, RetrievedChunk) for r in results)

    def test_results_sorted_descending_by_similarity(
        self, query_embedding: list[float], corpus: Corpus
    ) -> None:
        """Results are ordered from most to least similar."""
        results = retrieve(query_embedding, corpus, top_k=5)
        similarities = [r.similarity for r in results]
        assert similarities == sorted(similarities, reverse=True)

    def test_top_k_exceeds_corpus_size(
        self, query_embedding: list[float], corpus: Corpus
    ) -> None:
        """top_k larger than the corpus returns all chunks without crashing."""
        results = retrieve(query_embedding, corpus, top_k=999)
        assert len(results) == len(corpus.chunks)

    def test_filter_fn_provider_aetna(
        self, query_embedding: list[float], corpus: Corpus
    ) -> None:
        """Filtering for provider=='aetna' returns only the 2 aetna chunks."""
        results = retrieve(
            query_embedding,
            corpus,
            top_k=10,
            filter_fn=lambda c: c.metadata.get("provider") == "aetna",
        )
        assert len(results) == 2
        for r in results:
            assert r.chunk_id in ("claims-guide-001", "claims-guide-002")

    def test_filter_fn_rejects_all_returns_empty(
        self, query_embedding: list[float], corpus: Corpus
    ) -> None:
        """A filter that rejects every chunk produces an empty result list."""
        results = retrieve(
            query_embedding,
            corpus,
            top_k=5,
            filter_fn=lambda _c: False,
        )
        assert results == []


# ── load_corpus ──────────────────────────────────────────────────────────


class TestLoadCorpus:
    """Test load_corpus fallback behaviour."""

    def test_missing_path_returns_mock_corpus(self) -> None:
        """A nonexistent path falls back to the mock corpus with 10 chunks."""
        corpus = load_corpus("/nonexistent/path.json")
        assert isinstance(corpus, Corpus)
        assert len(corpus.chunks) == 10

    def test_mock_fallback_chunks_are_valid(self) -> None:
        """Mock fallback chunks have proper embeddings and fields."""
        corpus = load_corpus("/nonexistent/path.json")
        for chunk in corpus.chunks:
            assert len(chunk.embedding) == _MOCK_EMBED_DIM
            assert chunk.chunk_id
            assert chunk.text


# ── Dimension consistency (regression guard for Bug 1) ───────────────────


class TestDimensionConsistency:
    """Prevent dimension mismatch between retrieval_sim and embeddings provider."""

    def test_mock_embed_dim_matches_default_embedding_dim(self) -> None:
        """_MOCK_EMBED_DIM must equal _DEFAULT_EMBEDDING_DIM from embeddings.py.

        Regression guard: a previous bug set _MOCK_EMBED_DIM to 384 while the
        embedding provider used 1536, causing silent shape mismatches at
        retrieval time.
        """
        assert _MOCK_EMBED_DIM == _DEFAULT_EMBEDDING_DIM, (
            f"_MOCK_EMBED_DIM ({_MOCK_EMBED_DIM}) != "
            f"_DEFAULT_EMBEDDING_DIM ({_DEFAULT_EMBEDDING_DIM}) — "
            "these must stay in sync"
        )
