"""Integration test: W1 with 5 dry-run inputs → projection engine.

Verifies that workflow agents produce StepRecords compatible with the
existing projection pipeline (compute_stats → detect_patterns → project).
"""

from __future__ import annotations

import asyncio

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.projection.stats import compute_stats

from bt_agents.harness.run_workflow import load_agent, load_prompts


@pytest.fixture
def w1_prompts() -> dict[str, str]:
    return load_prompts("W1", "prompts/")


@pytest.fixture
def w1_inputs() -> list[dict[str, str]]:
    return [
        {"input": "How do I reset my password?", "_dry_run": True},
        {"input": "I was charged twice this month for my subscription.", "_dry_run": True},
        {"input": "Can you help?", "_dry_run": True},
        {
            "input": (
                "I need to integrate your API with my CI/CD pipeline. "
                "Can you provide documentation on webhook events and rate limits?"
            ),
            "_dry_run": True,
        },
        {"input": "Your product is terrible and I want a refund immediately.", "_dry_run": True},
    ]


class TestW1DryRun:
    """Verify W1 produces valid StepRecords in dry-run mode."""

    def test_produces_step_records(self, w1_prompts, w1_inputs):
        agent = load_agent("W1")
        all_records: list[list[StepRecord]] = []
        for inp in w1_inputs:
            records = asyncio.run(agent.execute(inp, w1_prompts))
            all_records.append(records)

        assert len(all_records) == 5
        for run_records in all_records:
            assert len(run_records) == 1
            rec = run_records[0]
            assert isinstance(rec, StepRecord)
            assert rec.step_name == "classify_respond"
            assert rec.step_type == "llm"
            assert rec.output_format == "text"
            assert rec.model in ("claude-haiku-4-5", "claude-sonnet-4-6")

    def test_step_record_round_trips(self, w1_prompts, w1_inputs):
        agent = load_agent("W1")
        records = asyncio.run(agent.execute(w1_inputs[0], w1_prompts))
        rec = records[0]
        serialized = rec.to_dict()
        restored = StepRecord.from_dict(serialized)
        assert restored.step_name == rec.step_name
        assert restored.model == rec.model


class TestProjectionPipelineCompat:
    """Verify StepRecords flow through the projection pipeline without errors."""

    def test_compute_stats_accepts_records(self, w1_prompts, w1_inputs):
        agent = load_agent("W1")
        all_records: list[list[StepRecord]] = []
        for inp in w1_inputs:
            records = asyncio.run(agent.execute(inp, w1_prompts))
            all_records.append(records)

        stats = compute_stats(all_records)
        assert stats.total_runs == 5
        assert stats.total_steps == 5
        assert "classify_respond" in stats.step_stats
