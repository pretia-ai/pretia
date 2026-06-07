"""W1 — Support Agent (Simple): single-step with Haiku/Sonnet routing.

Routes based on input token count: < 80 tokens → Haiku, ≥ 80 → Sonnet.
"""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.single_step import run_single_step


class W01SupportSimple(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_single_step(
            input_text=input_data.get("input", ""),
            system_prompt=prompts["classify_respond"],
            model="claude-haiku-4-5",
            alternate_model="claude-sonnet-4-6",
            routing_threshold=80,
            step_name="classify_respond",
            output_format="text",
            max_tokens=512,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W01SupportSimple()
