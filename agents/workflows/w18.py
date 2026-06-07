"""W18 — Long Document Single-Pass: DeepSeek V4 Pro on 30K-100K token docs."""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.single_step import run_single_step


class W18LongDocument(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_single_step(
            input_text=input_data.get("input", ""),
            system_prompt=prompts["process"],
            model="deepseek-v4-pro",
            step_name="process",
            output_format="json",
            max_tokens=2500,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W18LongDocument()
