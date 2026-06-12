"""Single-step pattern: one LLM call per input with optional routing.

Used by W1, W5, W11, W12, W18.
"""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from bt_agents.harness.step_builder import build_llm_step
from bt_agents.providers.llm import call_model


async def run_single_step(
    *,
    input_text: str,
    system_prompt: str,
    model: str,
    step_name: str,
    output_format: str,
    max_tokens: int = 4096,
    alternate_model: str | None = None,
    routing_threshold: int | None = None,
    messages: list[dict[str, Any]] | None = None,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute a single LLM call, optionally routing by input token count.

    When ``alternate_model`` and ``routing_threshold`` are provided, the input
    text length (chars / 4 as rough token estimate) is compared against the
    threshold. Below threshold uses the primary model; at or above uses the
    alternate.
    """
    selected_model = model
    if alternate_model and routing_threshold is not None:
        approx_tokens = len(input_text) // 4
        if approx_tokens >= routing_threshold:
            selected_model = alternate_model

    if messages is None:
        messages = [{"role": "user", "content": input_text}]

    response = await call_model(
        selected_model,
        system_prompt,
        messages,
        max_tokens=max_tokens,
        dry_run=dry_run,
    )

    record = build_llm_step(
        step_name=step_name,
        response=response,
        system_prompt=system_prompt,
        output_format=output_format,
    )

    return [record]
