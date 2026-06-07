"""Multi-hop RAG pattern: iterative retrieval with sufficiency assessment.

Used by W15. Flow: (embed → retrieve → assess sufficiency) × N hops → generate.
"""

from __future__ import annotations

import json
import logging

from agentcost.collectors.base import StepRecord

from agents.harness.retrieval_sim import load_corpus, retrieve
from agents.harness.step_builder import build_embedding_step, build_llm_step
from agents.providers.embeddings import embed_text
from agents.providers.llm import call_model

logger = logging.getLogger(__name__)


async def run_multi_hop_rag(
    *,
    query: str,
    prompts: dict[str, str],
    embedding_model: str,
    sufficiency_model: str,
    sufficiency_prompt_key: str,
    sufficiency_step_name: str,
    sufficiency_max_tokens: int,
    generation_model: str,
    generation_prompt_key: str,
    generation_step_name: str,
    generation_output_format: str,
    generation_max_tokens: int,
    corpus_path: str,
    top_k: int,
    max_hops: int = 4,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute a multi-hop RAG pipeline with iterative retrieval.

    Each hop embeds a query, retrieves chunks, then asks a sufficiency model
    whether the accumulated context is enough to answer the original question.
    If not, the sufficiency model produces a refined query for the next hop.
    After the loop, a generation model synthesizes the final answer.

    Args:
        query: Original user query.
        prompts: Map of prompt keys to system prompt text.
        embedding_model: Model for embedding queries.
        sufficiency_model: Model for assessing context sufficiency.
        sufficiency_prompt_key: Key into *prompts* for the sufficiency system prompt.
        sufficiency_step_name: Step name prefix for sufficiency StepRecords.
        sufficiency_max_tokens: Max output tokens for sufficiency checks.
        generation_model: Model for the final answer generation.
        generation_prompt_key: Key into *prompts* for the generation system prompt.
        generation_step_name: Step name for the generation StepRecord.
        generation_output_format: Output format for the generation step.
        generation_max_tokens: Max output tokens for answer generation.
        corpus_path: Path to the corpus JSON manifest file.
        top_k: Number of chunks to retrieve per hop.
        max_hops: Maximum number of retrieval hops before forcing generation.
        dry_run: If True, skip real API calls.

    Returns:
        List of StepRecords from all hops plus the final generation step.
    """
    records: list[StepRecord] = []
    corpus = load_corpus(corpus_path)
    accumulated_chunks: list = []
    seen_chunk_ids: set[str] = set()
    current_query = query

    # ------------------------------------------------------------------
    # Retrieval loop: embed → retrieve → assess sufficiency
    # ------------------------------------------------------------------
    for hop in range(1, max_hops + 1):
        # Embed the current query
        embed_response = await embed_text(
            current_query, embedding_model, dry_run=dry_run
        )

        embed_record = build_embedding_step(
            step_name=f"embed_hop_{hop}",
            response=embed_response,
        )
        records.append(embed_record)

        # Retrieve top-K chunks and accumulate (deduplicate by chunk_id)
        retrieved = retrieve(embed_response.embedding, corpus, top_k)
        new_chunks = [c for c in retrieved if c.chunk_id not in seen_chunk_ids]
        seen_chunk_ids.update(c.chunk_id for c in new_chunks)
        accumulated_chunks.extend(new_chunks)

        logger.info(
            "Hop %d: retrieved %d chunks (total accumulated: %d)",
            hop,
            len(retrieved),
            len(accumulated_chunks),
        )

        # Assess sufficiency
        context_block = _format_chunks(accumulated_chunks)
        sufficiency_message = (
            f"Original question: {query}\n\n"
            f"Accumulated context:\n{context_block}\n\n"
            "Assess whether the context above is sufficient to fully answer "
            "the original question. Respond with JSON:\n"
            '{"sufficient": true/false, "refined_query": "..."}\n'
            "If sufficient is false, provide a refined_query that targets "
            "the missing information."
        )

        sufficiency_prompt = prompts[sufficiency_prompt_key]
        suff_messages = [{"role": "user", "content": sufficiency_message}]

        suff_response = await call_model(
            sufficiency_model,
            sufficiency_prompt,
            suff_messages,
            max_tokens=sufficiency_max_tokens,
            dry_run=dry_run,
        )

        suff_record = build_llm_step(
            step_name=f"{sufficiency_step_name}_hop_{hop}",
            response=suff_response,
            system_prompt=sufficiency_prompt,
            output_format="json",
            iteration=hop,
        )
        records.append(suff_record)

        # Parse sufficiency response
        if hop == max_hops:
            logger.info("Reached max hops (%d), proceeding to generation", max_hops)
            break

        try:
            parsed = json.loads(suff_response.content)
            is_sufficient = parsed.get("sufficient", False)
            if is_sufficient:
                logger.info("Sufficiency achieved at hop %d", hop)
                break
            refined = parsed.get("refined_query", "")
            if refined:
                current_query = refined
                logger.info("Hop %d: refining query to %r", hop, current_query)
            else:
                logger.warning(
                    "Hop %d: sufficiency=false but no refined_query provided, "
                    "reusing previous query",
                    hop,
                )
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.warning(
                "Failed to parse sufficiency output at hop %d: %s — "
                "continuing with current query",
                hop,
                exc,
            )

    # ------------------------------------------------------------------
    # Final generation step
    # ------------------------------------------------------------------
    context_block = _format_chunks(accumulated_chunks)
    gen_message = (
        f"Context:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer the question using only the context provided above. "
        "Synthesize information from all retrieved sources."
    )

    generation_prompt = prompts[generation_prompt_key]
    gen_messages = [{"role": "user", "content": gen_message}]

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


def _format_chunks(chunks: list) -> str:
    """Format retrieved chunks into a numbered context block."""
    parts: list[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[{idx}] (Source: {chunk.document_name}, p.{chunk.page})\n{chunk.text}"
        )
    return "\n\n".join(parts)
