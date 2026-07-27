"""Test the preflight fail-fast behavior in ProfileRunner.run()."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from pretia.collectors.base import BaseCollector, StepRecord
from pretia.inputs.selector import InputSelection
from pretia.runner import ProfileRunner


def _make_record(**overrides: Any) -> StepRecord:
    """Build a valid StepRecord with sensible defaults."""
    defaults = {
        "step_name": "llm_call",
        "step_type": "llm",
        "model": "gpt-4o",
        "input_tokens": 100,
        "output_tokens": 50,
        "context_size": 100,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 50,
        "output_format": "text",
        "is_retry": False,
        "iteration": 1,
        "parent_step": None,
        "duration_ms": 100,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(overrides)
    return StepRecord(**defaults)


_RECORD = _make_record()
_SELECTION = InputSelection(
    mode="manual",
    inputs=["test1", "test2", "test3"],
    message="manual inputs",
)


class FakeCollector(BaseCollector):
    """A controllable collector for testing the preflight protocol."""

    def __init__(
        self,
        results_per_call: list[list[list[StepRecord]]],
        errors_per_call: list[BaseException | None] | None = None,
    ) -> None:
        super().__init__()
        self._results_per_call = results_per_call
        self._errors_per_call = errors_per_call or [None] * len(results_per_call)
        self.calls: list[dict[str, Any]] = []

    async def collect(
        self,
        workflow: Any,
        inputs: list[str],
        on_run_complete: Any = None,
        concurrency: int | None = None,
    ) -> list[list[StepRecord]]:
        call_index = len(self.calls)
        self.calls.append({"inputs": list(inputs), "concurrency": concurrency})

        self.last_error = None

        if call_index < len(self._results_per_call):
            result = self._results_per_call[call_index]
        else:
            result = []

        error = (
            self._errors_per_call[call_index] if call_index < len(self._errors_per_call) else None
        )
        if error is not None:
            self.last_error = error

        # Fire the on_run_complete callback for each run if provided
        if on_run_complete is not None:
            for i, run in enumerate(result):
                on_run_complete(i, len(inputs), run)

        return result


def _patch_runner(
    fake_collector: FakeCollector,
    inputs: list[str] | None = None,
):
    """Return a context manager that patches the runner internals."""
    if inputs is None:
        inputs = ["test1", "test2", "test3"]

    selection = InputSelection(mode="manual", inputs=inputs, message="manual inputs")

    async def fake_workflow(inp: str) -> str:
        return f"result: {inp}"

    return (
        patch.object(
            ProfileRunner,
            "_load_workflow",
            return_value=(fake_workflow, "", None),
        ),
        patch.object(
            ProfileRunner,
            "_select_collector",
            return_value=fake_collector,
        ),
        patch.object(
            ProfileRunner,
            "_resolve_inputs",
            new_callable=AsyncMock,
            return_value=(selection, inputs),
        ),
    )


class TestPreflightFailFast:
    async def test_first_run_failure_aborts(self, tmp_path):
        """When the first run returns empty steps + last_error, abort early."""
        error = TypeError("main() takes 0 positional arguments")
        collector = FakeCollector(
            results_per_call=[[[]]],
            errors_per_call=[error],
        )

        p1, p2, p3 = _patch_runner(collector)
        with p1, p2, p3:
            runner = ProfileRunner(
                workflow_path="fake.py",
                explicit_inputs=["test1", "test2", "test3"],
                output_dir=str(tmp_path),
            )
            with pytest.raises(ValueError, match="First profiling run failed") as exc_info:
                await runner.run()

        assert "TypeError" in str(exc_info.value)
        assert len(collector.calls) == 1
        assert exc_info.value.__cause__ is error

    async def test_first_run_zero_steps_no_error_aborts(self, tmp_path):
        """First run returns empty steps with no error -- still aborts."""
        collector = FakeCollector(
            results_per_call=[[[]]],
            errors_per_call=[None],
        )

        p1, p2, p3 = _patch_runner(collector)
        with p1, p2, p3:
            runner = ProfileRunner(
                workflow_path="fake.py",
                explicit_inputs=["test1", "test2", "test3"],
                output_dir=str(tmp_path),
            )
            with pytest.raises(ValueError, match="captured 0 LLM steps") as exc_info:
                await runner.run()

        msg = str(exc_info.value)
        assert "FakeCollector" in msg or "fake" in msg.lower()

    async def test_success_makes_two_collect_calls(self, tmp_path):
        """Preflight OK, then batch -- two collect calls, first gets inputs[:1]."""
        collector = FakeCollector(
            results_per_call=[
                [[_RECORD]],
                [[_RECORD], [_RECORD]],
            ],
        )

        inputs = ["test1", "test2", "test3"]
        p1, p2, p3 = _patch_runner(collector, inputs=inputs)
        with p1, p2, p3:
            runner = ProfileRunner(
                workflow_path="fake.py",
                explicit_inputs=inputs,
                output_dir=str(tmp_path),
            )
            session = await runner.run()

        assert len(collector.calls) == 2
        assert collector.calls[0]["inputs"] == inputs[:1]
        assert collector.calls[1]["inputs"] == inputs[1:]
        assert session is not None

    async def test_single_input_makes_one_call(self, tmp_path):
        """With only one input, only the preflight call is made (no batch)."""
        collector = FakeCollector(
            results_per_call=[[[_RECORD]]],
        )

        inputs = ["only_one"]
        p1, p2, p3 = _patch_runner(collector, inputs=inputs)
        with p1, p2, p3:
            runner = ProfileRunner(
                workflow_path="fake.py",
                explicit_inputs=inputs,
                output_dir=str(tmp_path),
            )
            session = await runner.run()

        assert len(collector.calls) == 1
        assert session is not None

    async def test_batch_failures_logged_but_session_returned(self, tmp_path):
        """Preflight OK, batch has failures — session is returned with warnings."""
        collector = FakeCollector(
            results_per_call=[
                [[_RECORD]],
                [[], [_RECORD]],
            ],
            errors_per_call=[None, RuntimeError("intermittent")],
        )

        inputs = ["test1", "test2", "test3"]
        p1, p2, p3 = _patch_runner(collector, inputs=inputs)
        with p1, p2, p3:
            runner = ProfileRunner(
                workflow_path="fake.py",
                explicit_inputs=inputs,
                output_dir=str(tmp_path),
            )
            session = await runner.run()

        assert session is not None
        assert len(collector.calls) == 2
