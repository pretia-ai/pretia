"""Pipeline with override/short-circuit pattern for insurance claims processing.

Used by W17.
Flow: intake -> branch(short_circuit | proceed -> embed -> retrieve ->
evaluate -> conditional routing)
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agentcost.collectors.base import StepRecord
from agents.harness.retrieval_sim import load_corpus, retrieve
from agents.harness.step_builder import build_embedding_step, build_llm_step
from agents.providers.embeddings import embed_text
from agents.providers.llm import call_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class PipelineStepConfig:
    """Configuration for one step in the pipeline."""

    model: str
    prompt_key: str
    step_name: str
    output_format: str
    max_tokens: int = 4096


async def run_pipeline_overrides(
    *,
    input_data: dict[str, Any],
    prompts: dict[str, str],
    intake_step: PipelineStepConfig,
    embedding_model: str,
    corpus_path: str,
    evaluate_step: PipelineStepConfig,
    routing_step: PipelineStepConfig,
    provider_filter: Callable[[dict[str, Any]], str] | None = None,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute a pipeline with early short-circuit and conditional routing.

    The pipeline processes an insurance claim through intake classification,
    optional RAG-based policy retrieval, evaluation, and conditional routing
    for flagged claims.

    Returns:
        1 StepRecord  if short-circuited at intake.
        3 StepRecords if evaluated with no flags  (intake, embed, evaluate).
        4 StepRecords if flags trigger routing     (intake, embed, evaluate, routing).
    """
    records: list[StepRecord] = []

    # ── Step 1: Intake classification ────────────────────────────────────
    intake_system = prompts[intake_step.prompt_key]
    claim_json_str = json.dumps(input_data, indent=2)

    intake_response = await call_model(
        intake_step.model,
        intake_system,
        [{"role": "user", "content": claim_json_str}],
        max_tokens=intake_step.max_tokens,
        dry_run=dry_run,
    )

    intake_record = build_llm_step(
        step_name=intake_step.step_name,
        response=intake_response,
        system_prompt=intake_system,
        output_format=intake_step.output_format,
    )
    records.append(intake_record)

    # Parse intake output to check for short-circuit.
    try:
        intake_parsed = json.loads(intake_response.content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse intake output as JSON: %s", exc)
        intake_parsed = {}

    # ── Step 2: Short-circuit check ──────────────────────────────────────
    if intake_parsed.get("short_circuit", False):
        logger.info("Claim short-circuited at intake — returning early")
        return records

    # ── Step 3: Build retrieval query and embed ──────────────────────────
    procedure_code = input_data.get("procedure_code", "")
    diagnosis_code = input_data.get("diagnosis_code", "")
    claim_type = input_data.get("claim_type", "")
    retrieval_query = f"{procedure_code} {diagnosis_code} {claim_type}".strip()

    embed_response = await embed_text(
        retrieval_query,
        embedding_model,
        dry_run=dry_run,
    )

    embed_record = build_embedding_step(
        step_name="embed_claim_query",
        response=embed_response,
    )
    records.append(embed_record)

    # ── Step 4: Load corpus, optionally filter, retrieve top-K ───────────
    corpus = load_corpus(corpus_path)

    filter_fn = None
    if provider_filter is not None:
        target_provider = provider_filter(input_data)

        def filter_fn(chunk: Any) -> bool:  # noqa: E731
            return chunk.metadata.get("provider") == target_provider

    retrieved_chunks = retrieve(
        query_embedding=embed_response.embedding,
        corpus=corpus,
        top_k=5,
        filter_fn=filter_fn,
    )

    # ── Step 5: Evaluate claim against retrieved policy sections ─────────
    policy_sections = "\n\n---\n\n".join(
        f"[{chunk.document_name}, p.{chunk.page}]\n{chunk.text}" for chunk in retrieved_chunks
    )

    eval_system = prompts[evaluate_step.prompt_key]
    eval_user_content = (
        f"## Claim\n{claim_json_str}\n\n## Retrieved Policy Sections\n{policy_sections}"
    )

    eval_response = await call_model(
        evaluate_step.model,
        eval_system,
        [{"role": "user", "content": eval_user_content}],
        max_tokens=evaluate_step.max_tokens,
        dry_run=dry_run,
    )

    eval_record = build_llm_step(
        step_name=evaluate_step.step_name,
        response=eval_response,
        system_prompt=eval_system,
        output_format=evaluate_step.output_format,
    )
    records.append(eval_record)

    # Parse evaluate output for action and flags.
    try:
        eval_parsed = json.loads(eval_response.content)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse evaluate output as JSON: %s", exc)
        eval_parsed = {}

    flags: list[str] = eval_parsed.get("flags", [])
    trigger_flags = {"high_amount", "code_inconsistency"}

    # ── Step 6: Conditional routing for flagged claims ───────────────────
    if flags and trigger_flags.intersection(flags):
        logger.info("Claim flagged (%s) — routing for review", flags)
        routing_system = prompts[routing_step.prompt_key]

        routing_user_content = (
            f"## Claim\n{claim_json_str}\n\n## Evaluation Result\n{eval_response.content}"
        )

        routing_response = await call_model(
            routing_step.model,
            routing_system,
            [{"role": "user", "content": routing_user_content}],
            max_tokens=routing_step.max_tokens,
            dry_run=dry_run,
        )

        routing_record = build_llm_step(
            step_name=routing_step.step_name,
            response=routing_response,
            system_prompt=routing_system,
            output_format=routing_step.output_format,
        )
        records.append(routing_record)

    return records
