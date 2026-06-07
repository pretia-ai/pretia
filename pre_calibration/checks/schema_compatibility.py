"""Verify StepRecord schema roundtrip."""

from __future__ import annotations

from datetime import UTC, datetime

from agentcost.collectors.base import StepRecord
from pre_calibration.pre_calibration import CheckResult


async def check() -> CheckResult:
    """Create a sample StepRecord and verify serialization roundtrip."""
    try:
        record = StepRecord(
            step_name="test_step",
            step_type="llm",
            model="gpt-4o-mini",
            input_tokens=340,
            output_tokens=45,
            context_size=620,
            tool_definitions_tokens=0,
            system_prompt_hash="abc123",
            system_prompt_tokens=280,
            output_format="json",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=230,
            timestamp=datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC),
        )
        d = record.to_dict()
        restored = StepRecord.from_dict(d)
        assert restored.step_name == record.step_name
        assert restored.input_tokens == record.input_tokens
        return CheckResult(
            name="collector_schema",
            status="PASS",
            details={"fields_verified": len(d)},
            blocking=True,
        )
    except Exception as e:
        return CheckResult(
            name="collector_schema",
            status="FAIL",
            details={"error": str(e)},
            blocking=True,
        )
