"""Multi-turn conversation pattern: accumulating context across turns.

Used by W19 (8-turn DeepSeek conversations).
"""

from __future__ import annotations

from typing import Any

from pretia.collectors.base import StepRecord

from bt_agents.harness.step_builder import build_llm_step
from bt_agents.providers.llm import call_model


async def run_multi_turn(
    *,
    conversation_script: list[str],
    system_prompt: str,
    model: str,
    step_name: str,
    output_format: str,
    max_tokens: int = 512,
    dry_run: bool = False,
) -> list[StepRecord]:
    """Execute an N-turn conversation with accumulating history.

    Each turn gets a fresh ``{{CACHE_BUST_SUFFIX}}`` via ``call_model``.
    The conversation history grows with each turn, producing linearly
    increasing input token counts.

    Args:
        conversation_script: List of user messages (one per turn).
    """
    records: list[StepRecord] = []
    history: list[dict[str, Any]] = []

    for turn_idx, user_msg in enumerate(conversation_script, start=1):
        history.append({"role": "user", "content": user_msg})

        response = await call_model(
            model,
            system_prompt,
            list(history),
            max_tokens=max_tokens,
            dry_run=dry_run,
        )

        history.append({"role": "assistant", "content": response.content})

        record = build_llm_step(
            step_name=f"{step_name}_turn_{turn_idx}",
            response=response,
            system_prompt=system_prompt,
            output_format=output_format,
            iteration=turn_idx,
        )
        records.append(record)

    return records
