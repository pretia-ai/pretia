"""W19 — Multi-Turn Conversation: 8-turn DeepSeek V4 Pro session.

Context accumulates linearly across turns. Each turn gets a fresh cache-bust.
"""

from __future__ import annotations

from typing import Any

from pretia.collectors.base import StepRecord
from bt_agents import BaseAgent
from bt_agents.patterns.multi_turn import run_multi_turn


class W19MultiTurn(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        conversation = input_data.get("turns", [])
        if isinstance(conversation, str):
            conversation = [conversation]
        if not conversation:
            conversation = [input_data.get("input", "Hello")]

        return await run_multi_turn(
            conversation_script=conversation,
            system_prompt=prompts["respond"],
            model="deepseek-v4-pro",
            step_name="respond",
            output_format="text",
            max_tokens=4096,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W19MultiTurn()
