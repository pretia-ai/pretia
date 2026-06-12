"""W4 — Compliance Review: initial review → critique/revision loop.

DeepSeek V4 Pro reviews, Qwen 3.6 Plus critiques, DeepSeek V4 Pro revises.
Loop alternates critique ↔ revision until satisfied or 8 pairs.
"""

from __future__ import annotations

import json
from typing import Any

from agentcost.collectors.base import StepRecord
from bt_agents import BaseAgent
from bt_agents.patterns.self_assessment_loop import LoopStepConfig, run_self_assessment_loop


def _w4_history_builder(
    input_data: dict[str, Any],
    history: list[dict[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    """Build messages with document + accumulated findings and revisions."""
    document = input_data.get("input", "")
    parts = [f"Document to review:\n{document}"]
    for item in history:
        parts.append(f"\n--- Previous round ---\n{json.dumps(item, indent=2)}")
    return [{"role": "user", "content": "\n".join(parts)}]


class W04ComplianceReview(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_self_assessment_loop(
            input_data=input_data,
            prompts=prompts,
            initial_step=LoopStepConfig(
                model="deepseek-v4-pro",
                prompt_key="initial_review",
                step_name="initial_review",
                output_format="json",
                max_tokens=4096,
            ),
            loop_step=LoopStepConfig(
                model="qwen3.6-plus",
                prompt_key="self_critique",
                step_name="self_critique",
                output_format="json",
                max_tokens=4096,
            ),
            revision_step=LoopStepConfig(
                model="deepseek-v4-pro",
                prompt_key="revision",
                step_name="revision",
                output_format="json",
                max_tokens=4096,
            ),
            termination_field="satisfied",
            termination_threshold=True,
            max_iterations=4,
            history_builder=_w4_history_builder,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W04ComplianceReview()
