"""W17 — Insurance Claims Agent: intake with overrides → embed → evaluate → route.

The most complex workflow: conditional routing, multi-doc RAG, structured output,
function calling, and variable step topology (1-4 StepRecords per run).
"""

from __future__ import annotations

import json
from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.pipeline_overrides import PipelineStepConfig, run_pipeline_overrides

_DEFAULT_CORPUS = "pdfs/w17_corpus/embeddings.json"


def _extract_provider(claim: dict[str, Any]) -> str:
    """Extract insurance provider name from claim for corpus filtering."""
    return claim.get("provider", "").lower()


class W17ClaimsAgent(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        return await run_pipeline_overrides(
            input_data=input_data,
            prompts=prompts,
            intake_step=PipelineStepConfig(
                model="claude-haiku-4-5",
                prompt_key="intake_override",
                step_name="intake_override",
                output_format="json",
                max_tokens=512,
            ),
            embedding_model="text-embedding-3-small",
            corpus_path=input_data.get("corpus_path", _DEFAULT_CORPUS),
            evaluate_step=PipelineStepConfig(
                model="claude-sonnet-4-6",
                prompt_key="evaluate_decide",
                step_name="evaluate_decide",
                output_format="json",
                max_tokens=1024,
            ),
            routing_step=PipelineStepConfig(
                model="claude-haiku-4-5",
                prompt_key="conditional_routing",
                step_name="conditional_routing",
                output_format="json",
                max_tokens=256,
            ),
            provider_filter=_extract_provider,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W17ClaimsAgent()
