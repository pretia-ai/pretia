"""W14 — Simple PDF RAG: embed query → retrieve → generate answer.

OpenAI embeddings + Sonnet generation. Context size variance is the cost driver.
"""

from __future__ import annotations

from typing import Any

from agentcost.collectors.base import StepRecord

from agents import BaseAgent
from agents.patterns.rag_pipeline import run_rag_pipeline

_DEFAULT_CORPUS = "pdfs/w14_corpus/embeddings.json"


class W14SimpleRAG(BaseAgent):
    async def execute(
        self, input_data: dict[str, Any], prompts: dict[str, str]
    ) -> list[StepRecord]:
        top_k = input_data.get("expected_chunk_count", 5)
        return await run_rag_pipeline(
            query=input_data.get("input", ""),
            prompts=prompts,
            embedding_model="text-embedding-3-small",
            generation_model="claude-sonnet-4-6",
            generation_prompt_key="generate_answer",
            generation_step_name="generate_answer",
            generation_output_format="json",
            generation_max_tokens=1024,
            corpus_path=input_data.get("corpus_path", _DEFAULT_CORPUS),
            top_k=top_k,
            dry_run=input_data.get("_dry_run", False),
        )


agent = W14SimpleRAG()
