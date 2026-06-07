"""W12 — Extraction (DeepSeek): single-step DeepSeek V4 Flash."""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.single_step import run_single_step


class W12ExtractionDeepseek(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_single_step(
            input_text=input_data.get("input", ""),
            system_prompt=prompts["extract"],
            model="deepseek-v4-flash",
            step_name="extract",
            output_format="json",
            max_tokens=1024,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W12ExtractionDeepseek()
