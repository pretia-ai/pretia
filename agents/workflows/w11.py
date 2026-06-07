"""W11 — Support Agent (Qwen): single-step with Qwen-Turbo/Plus routing.

Same heuristic as W1 but uses Qwen models for cross-provider comparison.
"""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.single_step import run_single_step


class W11SupportQwen(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_single_step(
            input_text=input_data.get("input", ""),
            system_prompt=prompts["classify_respond"],
            model="qwen-turbo",
            alternate_model="qwen3.6-plus",
            routing_threshold=80,
            step_name="classify_respond",
            output_format="text",
            max_tokens=512,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W11SupportQwen()
