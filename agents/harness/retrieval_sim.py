"""Simulate document retrieval via cosine similarity for RAG workflows.

Load pre-computed chunk embeddings from a JSON manifest and perform vector
search. Used by W14 (insurance Q&A), W15 (claims extraction), and W17
(multi-provider RAG) workflow archetypes.
"""

from __future__ import annotations

import json
import logging
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try numpy for fast dot-product; fall back to pure Python.
# ---------------------------------------------------------------------------
try:
    import numpy as np

    _HAS_NUMPY = True
except ImportError:  # pragma: no cover
    _HAS_NUMPY = False

# ---------------------------------------------------------------------------
# Embedding dimension used for mock corpus vectors.
# ---------------------------------------------------------------------------
_MOCK_EMBED_DIM = 1536


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class CorpusChunk:
    """Single chunk stored in a retrieval corpus."""

    chunk_id: str
    text: str
    embedding: list[float]
    document_name: str
    page: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """Chunk returned by a retrieval query, with similarity score."""

    chunk_id: str
    text: str
    similarity: float
    document_name: str
    page: int


@dataclass(frozen=True, slots=True)
class Corpus:
    """Collection of embedded chunks that can be searched."""

    chunks: list[CorpusChunk]


# ---------------------------------------------------------------------------
# Cosine similarity
# ---------------------------------------------------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors.

    Use numpy when available for performance; otherwise fall back to pure
    Python with ``math.sqrt`` and ``sum``.
    """
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )

    if _HAS_NUMPY:
        va = np.asarray(a, dtype=np.float64)
        vb = np.asarray(b, dtype=np.float64)
        dot = float(np.dot(va, vb))
        norm_a = float(np.linalg.norm(va))
        norm_b = float(np.linalg.norm(vb))
    else:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(
    query_embedding: list[float],
    corpus: Corpus,
    top_k: int,
    filter_fn: Callable[[CorpusChunk], bool] | None = None,
) -> list[RetrievedChunk]:
    """Retrieve the top-K most similar chunks from *corpus*.

    Compute cosine similarity between *query_embedding* and every chunk,
    optionally filtering chunks first (e.g. W17 provider-specific retrieval),
    then return the top-K results sorted by descending similarity.
    """
    candidates = corpus.chunks
    if filter_fn is not None:
        candidates = [c for c in candidates if filter_fn(c)]

    scored: list[tuple[float, CorpusChunk]] = [
        (cosine_similarity(query_embedding, chunk.embedding), chunk)
        for chunk in candidates
    ]
    scored.sort(key=lambda pair: pair[0], reverse=True)

    return [
        RetrievedChunk(
            chunk_id=chunk.chunk_id,
            text=chunk.text,
            similarity=sim,
            document_name=chunk.document_name,
            page=chunk.page,
        )
        for sim, chunk in scored[:top_k]
    ]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------
def load_corpus(corpus_path: str) -> Corpus:
    """Load a chunk corpus from a JSON manifest file.

    The manifest must be a JSON object with a ``"chunks"`` key containing a
    list of chunk objects, each with keys ``chunk_id``, ``text``,
    ``embedding``, ``document_name``, ``page``, and optionally ``metadata``.

    If *corpus_path* does not exist, return a mock corpus with ~10 canned
    insurance/healthcare/corporate document chunks and log a warning.
    """
    path = Path(corpus_path)
    if not path.exists():
        logger.warning(
            "Corpus file %s not found — returning mock corpus with canned "
            "insurance/healthcare chunks for development",
            path,
        )
        return _build_mock_corpus()

    logger.info("Loading corpus from %s", path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = [
        CorpusChunk(
            chunk_id=entry["chunk_id"],
            text=entry["text"],
            embedding=entry["embedding"],
            document_name=entry["document_name"],
            page=entry["page"],
            metadata=entry.get("metadata", {}),
        )
        for entry in raw["chunks"]
    ]
    logger.info("Loaded %d chunks from corpus manifest", len(chunks))
    return Corpus(chunks=chunks)


# ---------------------------------------------------------------------------
# Mock corpus
# ---------------------------------------------------------------------------
_MOCK_CHUNKS: list[dict[str, Any]] = [
    {
        "chunk_id": "ins-policy-001",
        "text": (
            "Section 4.2 — Covered Services. Inpatient hospital stays are "
            "covered at 80% of the allowed amount after the annual deductible "
            "has been met. Pre-authorization is required for elective "
            "admissions exceeding 48 hours."
        ),
        "document_name": "BlueCross PPO Policy 2025",
        "page": 12,
        "metadata": {"provider": "bluecross", "doc_type": "policy"},
    },
    {
        "chunk_id": "ins-policy-002",
        "text": (
            "Section 7.1 — Prescription Drug Coverage. Generic medications "
            "are subject to a $10 copay per 30-day supply. Brand-name drugs "
            "require prior authorization and carry a $45 copay. Specialty "
            "medications are covered under Tier 4 at 20% coinsurance up to "
            "a $250 per-fill maximum."
        ),
        "document_name": "BlueCross PPO Policy 2025",
        "page": 28,
        "metadata": {"provider": "bluecross", "doc_type": "policy"},
    },
    {
        "chunk_id": "claims-guide-001",
        "text": (
            "Claims must be submitted within 90 days of the date of service. "
            "Electronic submissions via the EDI 837P format are processed "
            "within 14 business days. Paper claims require an additional "
            "10 business days. Incomplete claims are returned with a "
            "Remark Code N56."
        ),
        "document_name": "Claims Processing Manual v3.1",
        "page": 5,
        "metadata": {"provider": "aetna", "doc_type": "claims_guide"},
    },
    {
        "chunk_id": "claims-guide-002",
        "text": (
            "Explanation of Benefits (EOB) codes: CO-4 indicates the "
            "procedure code is inconsistent with the modifier or bill type. "
            "CO-16 means the claim lacks information needed for adjudication. "
            "PR-1 denotes the portion of the allowed amount the patient owes "
            "as deductible."
        ),
        "document_name": "Claims Processing Manual v3.1",
        "page": 18,
        "metadata": {"provider": "aetna", "doc_type": "claims_guide"},
    },
    {
        "chunk_id": "corp-benefits-001",
        "text": (
            "Employees enrolled in the High Deductible Health Plan (HDHP) "
            "are eligible for a Health Savings Account (HSA) with an employer "
            "seed contribution of $750 for individual coverage or $1,500 for "
            "family coverage, deposited in Q1 of each plan year."
        ),
        "document_name": "Acme Corp Benefits Guide 2025",
        "page": 3,
        "metadata": {"provider": "acme_corp", "doc_type": "benefits"},
    },
    {
        "chunk_id": "corp-benefits-002",
        "text": (
            "Dental coverage under the DMO plan includes two preventive "
            "visits per calendar year at no cost. Basic restorative services "
            "(fillings, extractions) are covered at 80% after a $50 "
            "individual deductible. Major services (crowns, bridges) are "
            "covered at 50% with a $2,000 annual maximum."
        ),
        "document_name": "Acme Corp Benefits Guide 2025",
        "page": 7,
        "metadata": {"provider": "acme_corp", "doc_type": "benefits"},
    },
    {
        "chunk_id": "healthcare-reg-001",
        "text": (
            "Under 42 CFR § 422.504, Medicare Advantage organizations must "
            "maintain a compliance plan that includes written policies, a "
            "designated compliance officer, training and education, effective "
            "lines of communication, and internal monitoring and auditing."
        ),
        "document_name": "Medicare Advantage Compliance Handbook",
        "page": 44,
        "metadata": {"provider": "cms", "doc_type": "regulation"},
    },
    {
        "chunk_id": "healthcare-reg-002",
        "text": (
            "HIPAA Privacy Rule (45 CFR § 164.502) establishes the minimum "
            "necessary standard: covered entities must make reasonable efforts "
            "to limit access to protected health information (PHI) to the "
            "minimum amount necessary to accomplish the intended purpose of "
            "the use, disclosure, or request."
        ),
        "document_name": "HIPAA Compliance Reference",
        "page": 15,
        "metadata": {"provider": "hhs", "doc_type": "regulation"},
    },
    {
        "chunk_id": "appeal-process-001",
        "text": (
            "Members may file a first-level appeal within 180 days of "
            "receiving a claim denial. The appeal must include a written "
            "statement, copies of relevant medical records, and a letter of "
            "medical necessity from the treating physician. The plan will "
            "issue a determination within 30 days for pre-service appeals "
            "and 60 days for post-service appeals."
        ),
        "document_name": "UnitedHealth Appeal Procedures",
        "page": 2,
        "metadata": {"provider": "united", "doc_type": "appeals"},
    },
    {
        "chunk_id": "network-directory-001",
        "text": (
            "In-network providers have agreed to accept negotiated rates as "
            "payment in full. Out-of-network providers may bill the member "
            "for the difference between their billed charges and the plan's "
            "allowed amount (balance billing). Members in HMO plans do not "
            "have out-of-network benefits except in emergencies."
        ),
        "document_name": "Provider Network Guide",
        "page": 9,
        "metadata": {"provider": "cigna", "doc_type": "network"},
    },
]


def _build_mock_corpus() -> Corpus:
    """Build a deterministic mock corpus with realistic healthcare chunks."""
    rng = random.Random(42)  # noqa: S311 — deterministic seed for tests
    chunks: list[CorpusChunk] = []

    for entry in _MOCK_CHUNKS:
        # Generate a deterministic pseudo-random embedding vector.
        embedding = [rng.gauss(0.0, 1.0) for _ in range(_MOCK_EMBED_DIM)]
        # L2-normalize so cosine similarity is just the dot product.
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]

        chunks.append(
            CorpusChunk(
                chunk_id=entry["chunk_id"],
                text=entry["text"],
                embedding=embedding,
                document_name=entry["document_name"],
                page=entry["page"],
                metadata=entry.get("metadata", {}),
            )
        )

    return Corpus(chunks=chunks)
