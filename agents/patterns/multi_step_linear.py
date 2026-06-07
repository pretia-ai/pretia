"""Multi-step linear pattern: sequential LLM calls with inter-step parsing.

Used by W9.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Callable

from agentcost.collectors.base import StepRecord

from agents.harness.step_builder import build_llm_step
from agents.providers.llm import call_model

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class LinearStepConfig:
    """Configuration for one step in a linear pipeline."""

    model: str
    prompt_key: str
    step_name: str
    output_format: str
    max_tokens: int = 4096


async def run_multi_step_linear(
    *,
    input_data: dict[str, Any],
    prompts: dict[str, str],
    steps: list[LinearStepConfig],
    message_builder: Callable[
        [dict[str, Any], dict[str, Any] | None, LinearStepConfig], list[dict[str, Any]]
    ],
    output_parser: Callable[[str], dict[str, Any]] = json.loads,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute a linear sequence of LLM calls.

    ``message_builder(input_data, previous_parsed_output, current_step_config)``
    constructs the user messages for each step.
    """
    records: list[StepRecord] = []
    prev_output: dict[str, Any] | None = None

    for step_cfg in steps:
        system_prompt = prompts[step_cfg.prompt_key]
        messages = message_builder(input_data, prev_output, step_cfg)

        response = await call_model(
            step_cfg.model,
            system_prompt,
            messages,
            max_tokens=step_cfg.max_tokens,
            dry_run=dry_run,
        )

        record = build_llm_step(
            step_name=step_cfg.step_name,
            response=response,
            system_prompt=system_prompt,
            output_format=step_cfg.output_format,
        )
        records.append(record)

        try:
            prev_output = output_parser(response.content)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Failed to parse output from step %s: %s", step_cfg.step_name, exc
            )
            prev_output = {"raw_output": response.content, "parse_error": True}

    return records
