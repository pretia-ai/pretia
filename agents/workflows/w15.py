"""W15 — Agentic Multi-Hop PDF RAG: embed → retrieve → assess → loop → generate.

OpenAI embed + Gemini sufficiency + DeepSeek V4 Pro generation. Max 4 hops.
"""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.multi_hop_rag import run_multi_hop_rag

_DEFAULT_CORPUS = "pdfs/w15_corpus/embeddings.json"


class W15MultiHopRAG(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        top_k = input_data.get("expected_chunk_count", 5)
        return await run_multi_hop_rag(
            query=input_data.get("input", ""),
            prompts=prompts,
            embedding_model="text-embedding-3-small",
            sufficiency_model="gemini-2.5-flash",
            sufficiency_prompt_key="assess_sufficiency",
            sufficiency_step_name="assess_sufficiency",
            sufficiency_max_tokens=512,
            generation_model="deepseek-v4-pro",
            generation_prompt_key="generate_answer",
            generation_step_name="generate_answer",
            generation_output_format="json",
            generation_max_tokens=1024,
            corpus_path=input_data.get("corpus_path", _DEFAULT_CORPUS),
            top_k=top_k,
            max_hops=4,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W15MultiHopRAG()
