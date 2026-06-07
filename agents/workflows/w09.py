"""W9 — Sales/Outreach (OpenAI): two-step linear pipeline.

Step 1 qualifies a lead (GPT-5.4 Nano), Step 2 drafts an email (GPT-5.4).
"""

from __future__ import annotations

import json
from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.multi_step_linear import LinearStepConfig, run_multi_step_linear


def _build_messages(
    input_data: dict[str, Any],
    prev_output: dict[str, Any] | None,
    step_cfg: LinearStepConfig,
) -> list[dict[str, Any]]:
    """Build user messages for each step in the W9 pipeline."""
    if prev_output is None:
        return [{"role": "user", "content": json.dumps(input_data, indent=2)}]
    lead_profile = json.dumps(input_data, indent=2)
    qualification = json.dumps(prev_output, indent=2)
    return [
        {
            "role": "user",
            "content": (
                f"Lead Profile:\n{lead_profile}\n\n"
                f"Qualification Results:\n{qualification}"
            ),
        }
    ]


class W09SalesOutreach(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        steps = [
            LinearStepConfig(
                model="gpt-5.4-nano",
                prompt_key="qualify",
                step_name="qualify",
                output_format="json",
                max_tokens=512,
            ),
            LinearStepConfig(
                model="gpt-5.4",
                prompt_key="draft_email",
                step_name="draft_email",
                output_format="json",
                max_tokens=512,
            ),
        ]
        return await run_multi_step_linear(
            input_data=input_data,
            prompts=prompts,
            steps=steps,
            message_builder=_build_messages,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W09SalesOutreach()
