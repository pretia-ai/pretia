"""W16 — Map-Reduce PDF Analysis: split → N parallel section processes → aggregate.

Sonnet splits and aggregates, Haiku processes each section in parallel.
"""

from __future__ import annotations

from typing import Any

from pretia.collectors.base import StepRecord

from bt_agents import BaseAgent
from bt_agents.patterns.map_reduce import MapReduceStepConfig, run_map_reduce


class W16MapReduce(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_map_reduce(
            input_text=input_data.get("input", ""),
            prompts=prompts,
            split_step=MapReduceStepConfig(
                model="claude-sonnet-4-6",
                prompt_key="split",
                step_name="split",
                output_format="json",
                max_tokens=2048,
            ),
            process_step=MapReduceStepConfig(
                model="claude-haiku-4-5",
                prompt_key="process_section",
                step_name="process_section",
                output_format="json",
                max_tokens=2048,
            ),
            aggregate_step=MapReduceStepConfig(
                model="claude-sonnet-4-6",
                prompt_key="aggregate",
                step_name="aggregate",
                output_format="json",
                max_tokens=2048,
            ),
            max_sections=20,
            parallel=True,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W16MapReduce()
