"""Self-assessment loop pattern: iterative refinement with termination criteria.

Used by W2 (classify -> loop(draft with self-assessment) -> conditional review)
and W4 (initial_review -> loop(critique -> revision alternation)).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from agentcost.collectors.base import StepRecord

from bt_agents.harness.step_builder import build_llm_step
from bt_agents.providers.llm import call_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LoopStepConfig:
    """Configuration for one step within a self-assessment loop."""

    model: str
    prompt_key: str
    step_name: str
    output_format: str
    max_tokens: int = 4096


def _check_termination(
    parsed: dict[str, Any],
    field: str,
    threshold: float | bool,
) -> bool:
    """Evaluate whether the termination criterion is met.

    Returns True when the loop should stop: numeric fields must reach or
    exceed the threshold, boolean fields must equal it exactly.
    """
    value = parsed.get(field)
    if value is None:
        return False
    if isinstance(threshold, bool):
        return value is threshold or value == threshold
    try:
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def _safe_parse_json(raw: str, step_name: str) -> tuple[dict[str, Any], bool]:
    """Parse JSON from LLM output, returning (parsed_dict, had_error)."""
    try:
        result = json.loads(raw)
        if isinstance(result, dict):
            return result, False
        logger.warning("Non-dict JSON from step %s, wrapping in dict", step_name)
        return {"value": result, "parse_error": True}, True
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("Failed to parse JSON from step %s: %s", step_name, exc)
        return {"raw_output": raw, "parse_error": True}, True


async def run_self_assessment_loop(
    *,
    input_data: dict[str, Any],
    prompts: dict[str, str],
    initial_step: LoopStepConfig,
    loop_step: LoopStepConfig,
    revision_step: LoopStepConfig | None = None,
    termination_field: str,
    termination_threshold: float | bool,
    max_iterations: int,
    conditional_step: LoopStepConfig | None = None,
    conditional_trigger: Callable[[dict, int], bool] | None = None,
    history_builder: Callable[[dict, list[dict], str], list[dict[str, Any]]],
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute an iterative self-assessment loop with optional conditional tail step.

    Supports two variants:
    - W2: classify -> loop(draft+self-assess) -> conditional Opus review
    - W4: initial_review -> loop(critique -> revision alternation)

    Context grows each iteration because history_builder receives the full
    accumulated history, enabling the projection engine to detect context
    growth patterns.
    """
    records: list[StepRecord] = []
    history: list[dict[str, Any]] = []

    # ── Step 1: Initial step (classify for W2, initial_review for W4) ────
    initial_system = prompts[initial_step.prompt_key]
    initial_messages = history_builder(input_data, history, "initial")

    initial_response = await call_model(
        initial_step.model,
        initial_system,
        initial_messages,
        max_tokens=initial_step.max_tokens,
        dry_run=dry_run,
    )

    initial_record = build_llm_step(
        step_name=initial_step.step_name,
        response=initial_response,
        system_prompt=initial_system,
        output_format=initial_step.output_format,
        iteration=1,
    )
    records.append(initial_record)

    initial_parsed, _ = _safe_parse_json(initial_response.content, initial_step.step_name)
    history.append(initial_parsed)

    # ── Step 2: Assessment loop ──────────────────────────────────────────
    iteration = 0
    terminated = False

    while iteration < max_iterations and not terminated:
        iteration += 1

        # 2a-2b: Build messages and call the loop step (draft/critique)
        loop_system = prompts[loop_step.prompt_key]
        loop_messages = history_builder(input_data, history, "loop")

        loop_response = await call_model(
            loop_step.model,
            loop_system,
            loop_messages,
            max_tokens=loop_step.max_tokens,
            dry_run=dry_run,
        )

        loop_record = build_llm_step(
            step_name=loop_step.step_name,
            response=loop_response,
            system_prompt=loop_system,
            output_format=loop_step.output_format,
            iteration=iteration,
        )
        records.append(loop_record)

        # 2c-2d: Parse output, extract termination field, accumulate history
        loop_parsed, parse_failed = _safe_parse_json(
            loop_response.content, loop_step.step_name
        )
        history.append(loop_parsed)

        # JSON parse failure: tokens were consumed but we cannot assess
        # termination reliably, so treat as terminated
        if parse_failed:
            logger.warning(
                "Parse failure at iteration %d of %s; treating as terminated",
                iteration,
                loop_step.step_name,
            )
            break

        # Check termination after the loop step
        if _check_termination(loop_parsed, termination_field, termination_threshold):
            terminated = True
            break

        # 2e: If revision_step provided and not yet terminated, run revision
        if revision_step is not None:
            revision_system = prompts[revision_step.prompt_key]
            revision_messages = history_builder(input_data, history, "revision")

            revision_response = await call_model(
                revision_step.model,
                revision_system,
                revision_messages,
                max_tokens=revision_step.max_tokens,
                dry_run=dry_run,
            )

            revision_record = build_llm_step(
                step_name=revision_step.step_name,
                response=revision_response,
                system_prompt=revision_system,
                output_format=revision_step.output_format,
                iteration=iteration,
            )
            records.append(revision_record)

            revision_parsed, rev_parse_failed = _safe_parse_json(
                revision_response.content, revision_step.step_name
            )
            history.append(revision_parsed)

            if rev_parse_failed:
                logger.warning(
                    "Parse failure in revision at iteration %d; treating as terminated",
                    iteration,
                )
                break

    # ── Step 3: Conditional tail step ────────────────────────────────────
    if (
        conditional_step is not None
        and conditional_trigger is not None
        and conditional_trigger(initial_parsed, iteration)
    ):
        cond_system = prompts[conditional_step.prompt_key]
        cond_messages = history_builder(input_data, history, "conditional")

        cond_response = await call_model(
            conditional_step.model,
            cond_system,
            cond_messages,
            max_tokens=conditional_step.max_tokens,
            dry_run=dry_run,
        )

        cond_record = build_llm_step(
            step_name=conditional_step.step_name,
            response=cond_response,
            system_prompt=cond_system,
            output_format=conditional_step.output_format,
            iteration=1,
        )
        records.append(cond_record)

    return records
