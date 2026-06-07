"""Simple RAG pattern: embed query, retrieve chunks, generate answer.

Used by W14. Flow: embed → retrieve → generate.
"""

from __future__ import annotations

import logging

from agentcost.collectors.base import StepRecord

from agents.harness.retrieval_sim import load_corpus, retrieve
from agents.harness.step_builder import build_embedding_step, build_llm_step
from agents.providers.embeddings import embed_text
from agents.providers.llm import call_model

logger = logging.getLogger(__name__)


async def run_rag_pipeline(
    *,
    query: str,
    prompts: dict[str, str],
    embedding_model: str,
    generation_model: str,
    generation_prompt_key: str,
    generation_step_name: str,
    generation_output_format: str,
    generation_max_tokens: int,
    corpus_path: str,
    top_k: int,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute a single-hop RAG pipeline: embed, retrieve, generate.

    Args:
        query: User query to answer.
        prompts: Map of prompt keys to system prompt text.
        embedding_model: Model for embedding the query.
        generation_model: Model for the answer generation step.
        generation_prompt_key: Key into *prompts* for the generation system prompt.
        generation_step_name: Step name for the generation StepRecord.
        generation_output_format: Output format for the generation step.
        generation_max_tokens: Max output tokens for answer generation.
        corpus_path: Path to the corpus JSON manifest file.
        top_k: Number of chunks to retrieve.
        dry_run: If True, skip real API calls.

    Returns:
        List of StepRecords: [embed_record, generate_record].
    """
    records: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Step 1: Embed the query
    # ------------------------------------------------------------------
    embed_response = await embed_text(query, embedding_model, dry_run=dry_run)

    embed_record = build_embedding_step(
        step_name="embed_query",
        response=embed_response,
    )
    records.append(embed_record)

    # ------------------------------------------------------------------
    # Step 2: Load corpus and retrieve top-K chunks
    # ------------------------------------------------------------------
    corpus = load_corpus(corpus_path)
    retrieved = retrieve(embed_response.embedding, corpus, top_k)

    logger.info(
        "Retrieved %d chunks (top similarity: %.4f)",
        len(retrieved),
        retrieved[0].similarity if retrieved else 0.0,
    )

    # ------------------------------------------------------------------
    # Step 3: Format retrieved context and generate answer
    # ------------------------------------------------------------------
    context_block = _format_chunks(retrieved)
    user_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context provided above."
    )

    generation_prompt = prompts[generation_prompt_key]
    gen_messages = [{"role": "user", "content": user_message}]

    gen_response = await call_model(
        generation_model,
        generation_prompt,
        gen_messages,
        max_tokens=generation_max_tokens,
        dry_run=dry_run,
    )

    gen_record = build_llm_step(
        step_name=generation_step_name,
        response=gen_response,
        system_prompt=generation_prompt,
        output_format=generation_output_format,
    )
    records.append(gen_record)

    return records


def _format_chunks(retrieved: list) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for idx, chunk in enumerate(retrieved, start=1):
        parts.append(
            f"[{idx}] (Source: {chunk.document_name}, p.{chunk.page})\n{chunk.text}"
        )
    return "\n\n".join(parts)
