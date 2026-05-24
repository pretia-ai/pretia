"""Shared test fixtures for AgentCost."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord


@pytest.fixture
def sample_record() -> StepRecord:
    """A valid StepRecord with sensible defaults; reuse in tests via `dataclasses.replace`."""
    return StepRecord(
        step_name="classify_intent",
        step_type="llm",
        model="claude-haiku-3",
        input_tokens=340,
        output_tokens=45,
        context_size=620,
        tool_definitions_tokens=0,
        system_prompt_hash="a3f8c2d1e5b9",
        system_prompt_tokens=280,
        output_format="json",
        is_retry=False,
        iteration=1,
        parent_step=None,
        duration_ms=230,
        timestamp=datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC),
    )
