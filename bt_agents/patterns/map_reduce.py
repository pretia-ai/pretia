"""Map-reduce pattern: split document, process N sections in parallel, aggregate.

Used by W16.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from agentcost.collectors.base import StepRecord
from bt_agents.harness.step_builder import build_llm_step
from bt_agents.providers.llm import call_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MapReduceStepConfig:
    """Configuration for one phase of a map-reduce pipeline."""

    model: str
    prompt_key: str
    step_name: str
    output_format: str
    max_tokens: int = 4096


async def _process_section(
    *,
    section: dict[str, Any],
    section_index: int,
    process_step: MapReduceStepConfig,
    prompts: dict[str, str],
    dry_run: bool,
) -> tuple[StepRecord, str]:
    """Process a single section and return (StepRecord, raw output text)."""
    system_prompt = prompts[process_step.prompt_key]
    title = section.get("title", f"Section {section_index}")
    content = section.get("content", "")
    user_message = f"{title}\n\n{content}"

    response = await call_model(
        process_step.model,
        system_prompt,
        [{"role": "user", "content": user_message}],
        max_tokens=process_step.max_tokens,
        dry_run=dry_run,
    )

    record = build_llm_step(
        step_name=f"{process_step.step_name}_{section_index}",
        response=response,
        system_prompt=system_prompt,
        output_format=process_step.output_format,
        iteration=section_index,
    )

    return record, response.content


async def run_map_reduce(
    *,
    input_text: str,
    prompts: dict[str, str],
    split_step: MapReduceStepConfig,
    process_step: MapReduceStepConfig,
    aggregate_step: MapReduceStepConfig,
    max_sections: int = 20,
    parallel: bool = True,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute a map-reduce pipeline over a document.

    Flow: split document into sections, process each section (optionally in
    parallel via asyncio.gather), then aggregate all section outputs into a
    final summary.

    Returns:
        List of StepRecords: [split_record, *section_records, aggregate_record].
    """
    records: list[StepRecord] = []

    # ── Phase 1: Split ───────────────────────────────────────────────────
    split_system = prompts[split_step.prompt_key]
    split_response = await call_model(
        split_step.model,
        split_system,
        [{"role": "user", "content": input_text}],
        max_tokens=split_step.max_tokens,
        dry_run=dry_run,
    )

    split_record = build_llm_step(
        step_name=split_step.step_name,
        response=split_response,
        system_prompt=split_system,
        output_format=split_step.output_format,
    )
    records.append(split_record)

    # Parse sections array from split output.
    try:
        parsed = json.loads(split_response.content)
        sections: list[dict[str, Any]] = parsed.get("sections", [])
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.warning("Failed to parse split output as JSON: %s", exc)
        sections = [{"title": "full_document", "content": split_response.content}]

    # Clamp to max_sections.
    if len(sections) > max_sections:
        logger.warning(
            "Split produced %d sections, clamping to max_sections=%d",
            len(sections),
            max_sections,
        )
        sections = sections[:max_sections]

    if not sections:
        logger.warning("Split produced zero sections; skipping process and aggregate")
        return records

    # ── Phase 2: Process (map) ───────────────────────────────────────────
    section_outputs: list[str] = []

    if parallel:
        tasks = [
            _process_section(
                section=section,
                section_index=idx + 1,
                process_step=process_step,
                prompts=prompts,
                dry_run=dry_run,
            )
            for idx, section in enumerate(sections)
        ]
        results = await asyncio.gather(*tasks)
        for record, output_text in results:
            records.append(record)
            section_outputs.append(output_text)
    else:
        for idx, section in enumerate(sections):
            record, output_text = await _process_section(
                section=section,
                section_index=idx + 1,
                process_step=process_step,
                prompts=prompts,
                dry_run=dry_run,
            )
            records.append(record)
            section_outputs.append(output_text)

    # ── Phase 3: Aggregate (reduce) ─────────────────────────────────────
    concatenated = "\n\n---\n\n".join(section_outputs)
    agg_system = prompts[aggregate_step.prompt_key]

    agg_response = await call_model(
        aggregate_step.model,
        agg_system,
        [{"role": "user", "content": concatenated}],
        max_tokens=aggregate_step.max_tokens,
        dry_run=dry_run,
    )

    agg_record = build_llm_step(
        step_name=aggregate_step.step_name,
        response=agg_response,
        system_prompt=agg_system,
        output_format=aggregate_step.output_format,
    )
    records.append(agg_record)

    return records
