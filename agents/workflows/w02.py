"""W2 — Support Agent (Complex): classify → self-assessment loop → conditional Opus review.

Loop terminates when confidence >= 0.9 OR 12 iterations.
Opus fires when complexity == "complex" AND iterations >= 4.
"""

from __future__ import annotations

import json
from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.self_assessment_loop import LoopStepConfig, run_self_assessment_loop


def _w2_history_builder(
    input_data: dict[str, Any],
    history: list[dict[str, Any]],
    phase: str,
) -> list[dict[str, Any]]:
    """Build messages with accumulating conversation history."""
    user_question = input_data.get("input", "")
    parts = [f"Customer question: {user_question}"]
    for item in history:
        parts.append(f"\n--- Previous iteration ---\n{json.dumps(item, indent=2)}")
    return [{"role": "user", "content": "\n".join(parts)}]


def _w2_conditional_trigger(initial_output: dict[str, Any], iterations: int) -> bool:
    """Opus fires when complexity is 'complex' AND loop ran >= 4 iterations."""
    return initial_output.get("complexity") == "complex" and iterations >= 4


class W02SupportComplex(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_self_assessment_loop(
            input_data=input_data,
            prompts=prompts,
            initial_step=LoopStepConfig(
                model="claude-haiku-4-5",
                prompt_key="intake_classify",
                step_name="intake_classify",
                output_format="json",
                max_tokens=256,
            ),
            loop_step=LoopStepConfig(
                model="claude-sonnet-4-6",
                prompt_key="research_draft_loop",
                step_name="research_draft_loop",
                output_format="json",
                max_tokens=1024,
            ),
            termination_field="confidence",
            termination_threshold=0.9,
            max_iterations=12,
            conditional_step=LoopStepConfig(
                model="claude-opus-4-7",
                prompt_key="final_review",
                step_name="final_review",
                output_format="json",
                max_tokens=512,
            ),
            conditional_trigger=_w2_conditional_trigger,
            history_builder=_w2_history_builder,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W02SupportComplex()
