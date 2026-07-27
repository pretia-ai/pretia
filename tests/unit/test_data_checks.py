"""Tests for post-profiling data quality validation."""

from __future__ import annotations

from datetime import UTC, datetime

from pretia.collectors.base import StepRecord
from pretia.validation.data_checks import validate_profiling_data


def _make_record(
    step_name: str = "classify",
    input_tokens: int = 100,
    output_tokens: int = 50,
    step_type: str = "llm",
) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        step_type=step_type,
        model="gpt-4o-mini",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash="abc123",
        system_prompt_tokens=50,
        output_format="text",
        is_retry=False,
        iteration=1,
        parent_step=None,
        duration_ms=100,
        timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    )


class TestZeroTokenAllRunsWarning:
    def test_zero_token_all_runs(self):
        runs = []
        for _ in range(10):
            runs.append(
                [
                    _make_record("classify", 100, 50),
                    _make_record("generate", 200, 100),
                    _make_record("review", 0, 0),
                ]
            )
        warnings = validate_profiling_data(runs)
        review_warnings = [w for w in warnings if "review" in w]
        assert len(review_warnings) == 1
        assert "zero tokens across all 10 runs" in review_warnings[0]


class TestZeroTokenPartialWarning:
    def test_zero_token_partial(self):
        runs = []
        for i in range(10):
            if i < 3:
                runs.append(
                    [
                        _make_record("classify", 100, 50),
                        _make_record("review", 200, 100),
                    ]
                )
            else:
                runs.append(
                    [
                        _make_record("classify", 100, 50),
                        _make_record("review", 0, 0),
                    ]
                )
        warnings = validate_profiling_data(runs)
        review_warnings = [w for w in warnings if "review" in w]
        assert len(review_warnings) == 1
        assert "70%" in review_warnings[0]


class TestZeroTokenNoWarning:
    def test_no_warning(self):
        runs = []
        for _ in range(10):
            runs.append(
                [
                    _make_record("classify", 100, 50),
                    _make_record("generate", 200, 100),
                ]
            )
        warnings = validate_profiling_data(runs)
        assert len(warnings) == 0


class TestToolStepsNotWarned:
    def test_tool_steps_with_zero_tokens_ignored(self):
        """Tool invocations consume no LLM tokens — no warning expected."""
        runs = []
        for _ in range(10):
            runs.append(
                [
                    _make_record("agent", 500, 200),
                    _make_record("lookup_order", 0, 0, step_type="tool"),
                ]
            )
        warnings = validate_profiling_data(runs)
        assert len(warnings) == 0

    def test_llm_step_sharing_name_with_tool_still_warned(self):
        """A zero-token LLM step warns even if tool records share its name."""
        runs = []
        for _ in range(10):
            runs.append(
                [
                    _make_record("agent", 500, 200),
                    _make_record("broken_step", 0, 0),
                    _make_record("broken_step", 0, 0, step_type="tool"),
                ]
            )
        warnings = validate_profiling_data(runs)
        broken_warnings = [w for w in warnings if "broken_step" in w]
        assert len(broken_warnings) == 1


class TestZeroTokenMissingStep:
    def test_conditional_step_not_warned(self):
        runs = []
        for i in range(10):
            run = [_make_record("classify", 100, 50)]
            if i < 3:
                run.append(_make_record("conditional_branch", 150, 75))
            runs.append(run)
        warnings = validate_profiling_data(runs)
        branch_warnings = [w for w in warnings if "conditional_branch" in w]
        assert len(branch_warnings) == 0
