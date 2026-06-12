"""Test pattern logic using dry_run=True mode (no API calls)."""

from __future__ import annotations

import os

import pytest

pytest.importorskip("litellm")

from agentcost.collectors.base import StepRecord
from bt_agents.harness.run_workflow import load_prompts
from bt_agents.patterns.multi_turn import run_multi_turn
from bt_agents.patterns.rag_pipeline import run_rag_pipeline
from bt_agents.patterns.router import RouteConfig, run_router
from bt_agents.patterns.self_assessment_loop import (
    LoopStepConfig,
    run_self_assessment_loop,
)
from bt_agents.patterns.single_step import run_single_step

# ---------------------------------------------------------------------------
# Resolve prompts directory relative to project root
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PROMPTS_DIR = os.path.join(_PROJECT_ROOT, "prompts")


# ── SingleStep pattern ─────────────────────────────────────────────────────


class TestSingleStepDryRun:
    """Validate SingleStep pattern behaviour using dry_run=True."""

    @pytest.fixture()
    def w1_prompts(self) -> dict[str, str]:
        return load_prompts("W1", _PROMPTS_DIR)

    async def test_dry_run_produces_exactly_one_record(self, w1_prompts: dict[str, str]) -> None:
        records = await run_single_step(
            input_text="Hi",
            system_prompt=w1_prompts["classify_respond"],
            model="claude-haiku-4-5",
            step_name="classify_respond",
            output_format="text",
            dry_run=True,
        )
        assert len(records) == 1
        assert isinstance(records[0], StepRecord)

    async def test_without_alternate_uses_primary_model(self, w1_prompts: dict[str, str]) -> None:
        records = await run_single_step(
            input_text="Hi",
            system_prompt=w1_prompts["classify_respond"],
            model="claude-haiku-4-5",
            step_name="classify_respond",
            output_format="text",
            dry_run=True,
        )
        assert records[0].model == "claude-haiku-4-5"

    async def test_routing_short_input_uses_primary(self, w1_prompts: dict[str, str]) -> None:
        """Input 'Hi' has len 2 -> chars/4 = 0, well below threshold 80."""
        records = await run_single_step(
            input_text="Hi",
            system_prompt=w1_prompts["classify_respond"],
            model="claude-haiku-4-5",
            alternate_model="claude-sonnet-4-6",
            routing_threshold=80,
            step_name="classify_respond",
            output_format="text",
            dry_run=True,
        )
        assert records[0].model == "claude-haiku-4-5"

    async def test_routing_long_input_uses_alternate(self, w1_prompts: dict[str, str]) -> None:
        """Input of 400+ chars -> chars/4 >= 100 >= threshold 80 -> alternate model."""
        long_input = "x" * 400
        records = await run_single_step(
            input_text=long_input,
            system_prompt=w1_prompts["classify_respond"],
            model="claude-haiku-4-5",
            alternate_model="claude-sonnet-4-6",
            routing_threshold=80,
            step_name="classify_respond",
            output_format="text",
            dry_run=True,
        )
        assert records[0].model == "claude-sonnet-4-6"

    async def test_step_type_is_llm(self, w1_prompts: dict[str, str]) -> None:
        records = await run_single_step(
            input_text="test",
            system_prompt=w1_prompts["classify_respond"],
            model="claude-haiku-4-5",
            step_name="classify_respond",
            output_format="text",
            dry_run=True,
        )
        assert records[0].step_type == "llm"


# ── MultiTurn pattern ──────────────────────────────────────────────────────


class TestMultiTurnDryRun:
    """Validate MultiTurn pattern behaviour using dry_run=True."""

    @pytest.fixture()
    def w19_prompts(self) -> dict[str, str]:
        return load_prompts("W19", _PROMPTS_DIR)

    async def test_three_turn_produces_three_records(self, w19_prompts: dict[str, str]) -> None:
        records = await run_multi_turn(
            conversation_script=["Hello", "How are you?", "Goodbye"],
            system_prompt=w19_prompts["respond"],
            model="deepseek-v4-flash",
            step_name="respond",
            output_format="text",
            dry_run=True,
        )
        assert len(records) == 3

    async def test_step_names_follow_turn_pattern(self, w19_prompts: dict[str, str]) -> None:
        records = await run_multi_turn(
            conversation_script=["A", "B", "C"],
            system_prompt=w19_prompts["respond"],
            model="deepseek-v4-flash",
            step_name="respond",
            output_format="text",
            dry_run=True,
        )
        assert records[0].step_name == "respond_turn_1"
        assert records[1].step_name == "respond_turn_2"
        assert records[2].step_name == "respond_turn_3"

    async def test_iteration_field_matches_turn_number(self, w19_prompts: dict[str, str]) -> None:
        records = await run_multi_turn(
            conversation_script=["A", "B", "C"],
            system_prompt=w19_prompts["respond"],
            model="deepseek-v4-flash",
            step_name="respond",
            output_format="text",
            dry_run=True,
        )
        assert records[0].iteration == 1
        assert records[1].iteration == 2
        assert records[2].iteration == 3

    async def test_all_records_are_step_records(self, w19_prompts: dict[str, str]) -> None:
        records = await run_multi_turn(
            conversation_script=["Hello"],
            system_prompt=w19_prompts["respond"],
            model="deepseek-v4-flash",
            step_name="respond",
            output_format="text",
            dry_run=True,
        )
        for r in records:
            assert isinstance(r, StepRecord)


# ── Router pattern ─────────────────────────────────────────────────────────


class TestRouterDryRun:
    """Validate Router pattern behaviour using dry_run=True."""

    @pytest.fixture()
    def w13_prompts(self) -> dict[str, str]:
        return load_prompts("W13", _PROMPTS_DIR)

    async def test_produces_at_least_two_records(self, w13_prompts: dict[str, str]) -> None:
        routes = {
            "TIER_1": RouteConfig(
                model="claude-haiku-4-5",
                prompt_key="path_a_simple",
                step_name="path_a_simple",
                output_format="text",
                max_tokens=256,
            ),
        }
        records = await run_router(
            input_text="test input",
            prompts=w13_prompts,
            classifier_model="claude-haiku-4-5",
            classifier_prompt_key="classify",
            classifier_step_name="classify",
            classifier_max_tokens=128,
            routes=routes,
            default_route="TIER_1",
            dry_run=True,
        )
        assert len(records) >= 2

    async def test_first_record_is_classifier(self, w13_prompts: dict[str, str]) -> None:
        routes = {
            "TIER_1": RouteConfig(
                model="claude-haiku-4-5",
                prompt_key="path_a_simple",
                step_name="path_a_simple",
                output_format="text",
                max_tokens=256,
            ),
        }
        records = await run_router(
            input_text="test input",
            prompts=w13_prompts,
            classifier_model="claude-haiku-4-5",
            classifier_prompt_key="classify",
            classifier_step_name="classify",
            classifier_max_tokens=128,
            routes=routes,
            default_route="TIER_1",
            dry_run=True,
        )
        assert records[0].step_name == "classify"

    async def test_classifier_falls_back_to_default_route(
        self, w13_prompts: dict[str, str]
    ) -> None:
        """dry_run returns '{"dry_run": true}' which has no 'tier' key,
        so the router should fall back to the default route."""
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
        }
        records = await run_router(
            input_text="test input",
            prompts=w13_prompts,
            classifier_model="claude-haiku-4-5",
            classifier_prompt_key="classify",
            classifier_step_name="classify",
            classifier_max_tokens=128,
            routes=routes,
            default_route="TIER_1",
            dry_run=True,
        )
        # Second record should be the default route (TIER_1 -> path_a_simple)
        assert records[1].step_name == "path_a_simple"


# ── SelfAssessmentLoop pattern ─────────────────────────────────────────────


class TestSelfAssessmentLoopDryRun:
    """Validate SelfAssessmentLoop pattern behaviour using dry_run=True."""

    @pytest.fixture()
    def w2_prompts(self) -> dict[str, str]:
        return load_prompts("W2", _PROMPTS_DIR)

    def _history_builder(self, input_data: dict, history: list[dict], phase: str) -> list[dict]:
        """Minimal history builder for tests."""
        parts = [f"Input: {input_data.get('input', '')}"]
        for item in history:
            parts.append(str(item))
        return [{"role": "user", "content": "\n".join(parts)}]

    async def test_produces_at_least_two_records(self, w2_prompts: dict[str, str]) -> None:
        """Initial step + at least 1 loop iteration."""
        records = await run_self_assessment_loop(
            input_data={"input": "test"},
            prompts=w2_prompts,
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
            max_iterations=5,
            history_builder=self._history_builder,
            dry_run=True,
        )
        assert len(records) >= 2

    async def test_loop_terminates_on_parse_failure(self, w2_prompts: dict[str, str]) -> None:
        """dry_run returns '{"dry_run": true}' which lacks the termination
        field 'confidence'. The first loop iteration will parse successfully
        but not meet threshold, so the loop continues. However, every
        iteration returns the same content and should eventually stop at
        max_iterations. With dry_run the JSON parses fine but never has
        'confidence' >= 0.9, so it runs until max_iterations."""
        records = await run_self_assessment_loop(
            input_data={"input": "test"},
            prompts=w2_prompts,
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
            max_iterations=3,
            history_builder=self._history_builder,
            dry_run=True,
        )
        # 1 initial + 3 loop iterations = 4
        assert len(records) == 4

    async def test_first_record_is_initial_step(self, w2_prompts: dict[str, str]) -> None:
        records = await run_self_assessment_loop(
            input_data={"input": "test"},
            prompts=w2_prompts,
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
            max_iterations=3,
            history_builder=self._history_builder,
            dry_run=True,
        )
        assert records[0].step_name == "intake_classify"

    async def test_all_records_are_step_records(self, w2_prompts: dict[str, str]) -> None:
        records = await run_self_assessment_loop(
            input_data={"input": "test"},
            prompts=w2_prompts,
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
            max_iterations=2,
            history_builder=self._history_builder,
            dry_run=True,
        )
        for r in records:
            assert isinstance(r, StepRecord)
            assert r.step_type == "llm"


# ── RAG Pipeline pattern (Bug 1 regression guard) ────────────────────────


class TestRAGPipelineDryRun:
    """Validate RAG pipeline works in dry-run mode (Bug 1 fix: dimension match)."""

    @pytest.fixture()
    def w14_prompts(self) -> dict[str, str]:
        return load_prompts("W14", _PROMPTS_DIR)

    async def test_produces_exactly_two_records(self, w14_prompts: dict[str, str]) -> None:
        records = await run_rag_pipeline(
            query="What is covered under inpatient care?",
            prompts=w14_prompts,
            embedding_model="text-embedding-3-small",
            generation_model="claude-sonnet-4-6",
            generation_prompt_key="generate_answer",
            generation_step_name="generate_answer",
            generation_output_format="json",
            generation_max_tokens=1024,
            corpus_path="/nonexistent/corpus.json",
            top_k=3,
            dry_run=True,
        )
        assert len(records) == 2

    async def test_first_record_is_retrieval(self, w14_prompts: dict[str, str]) -> None:
        records = await run_rag_pipeline(
            query="test query",
            prompts=w14_prompts,
            embedding_model="text-embedding-3-small",
            generation_model="claude-sonnet-4-6",
            generation_prompt_key="generate_answer",
            generation_step_name="generate_answer",
            generation_output_format="json",
            generation_max_tokens=1024,
            corpus_path="/nonexistent/corpus.json",
            top_k=3,
            dry_run=True,
        )
        assert records[0].step_type == "retrieval"
        assert records[0].step_name == "embed_query"
        assert records[0].output_tokens == 0

    async def test_second_record_is_generation(self, w14_prompts: dict[str, str]) -> None:
        records = await run_rag_pipeline(
            query="test query",
            prompts=w14_prompts,
            embedding_model="text-embedding-3-small",
            generation_model="claude-sonnet-4-6",
            generation_prompt_key="generate_answer",
            generation_step_name="generate_answer",
            generation_output_format="json",
            generation_max_tokens=1024,
            corpus_path="/nonexistent/corpus.json",
            top_k=3,
            dry_run=True,
        )
        assert records[1].step_type == "llm"
        assert records[1].step_name == "generate_answer"
