"""W13 — Routing Agent: classify → route to Tier 1/2/3 paths.

Tier 1 (Haiku, 70%), Tier 2 (Sonnet, 20%), Tier 3 (Sonnet + tools, 10%).
Cost bimodality: Path A ~$0.003, Path C ~$0.08.
"""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.harness.tool_sim import TOOL_SCHEMAS, simulate_tool_call
from agents.patterns.router import RouteConfig, run_router


class W13RoutingAgent(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        routes = {
            "TIER_1": RouteConfig(
                model="claude-haiku-4-5",
                prompt_key="path_a_simple",
                step_name="path_a_simple",
                output_format="text",
                max_tokens=256,
            ),
            "TIER_2": RouteConfig(
                model="claude-sonnet-4-6",
                prompt_key="path_b_moderate",
                step_name="path_b_moderate",
                output_format="text",
                max_tokens=512,
            ),
            "TIER_3": RouteConfig(
                model="claude-sonnet-4-6",
                prompt_key="path_c_complex",
                step_name="path_c_complex",
                output_format="text",
                max_tokens=1024,
                has_tools=True,
            ),
        }
        return await run_router(
            input_text=input_data.get("input", ""),
            prompts=prompts,
            classifier_model="claude-haiku-4-5",
            classifier_prompt_key="classify",
            classifier_step_name="classify",
            classifier_max_tokens=128,
            routes=routes,
            default_route="TIER_1",
            tool_schemas=TOOL_SCHEMAS,
            tool_simulator=simulate_tool_call,
            max_tool_rounds=3,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W13RoutingAgent()
