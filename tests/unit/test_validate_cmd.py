"""Tests for the validate command logic (all mocked, no real API calls)."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from pretia.collectors.base import StepRecord
from pretia.validation.validate_cmd import (
    format_validate_report,
    run_validation,
)


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o-mini",
    input_tokens: int = 100,
    output_tokens: int = 50,
    iteration: int = 1,
) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        step_type="llm",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash="abc123",
        system_prompt_tokens=50,
        output_format="text",
        is_retry=False,
        iteration=iteration,
        parent_step=None,
        duration_ms=100,
        timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    )


def _make_mock_session(runs):
    from pretia.store import ProfilingSession

    return ProfilingSession(
        workflow_name="test.py",
        workflow_hash="abc",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=len(runs),
        input_mode="auto-generate",
        runs=runs,
        metadata={},
    )


class TestRunValidation:
    def test_produces_result(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        small_runs = [
            [
                _make_record("classify", "gpt-4o-mini", 100, 50),
                _make_record("generate", "gpt-4o-mini", 200, 100),
            ]
            for _ in range(20)
        ]
        large_runs = [
            [
                _make_record("classify", "gpt-4o-mini", 100, 50),
                _make_record("generate", "gpt-4o-mini", 200, 100),
            ]
            for _ in range(100)
        ]
        small_session = _make_mock_session(small_runs)
        large_session = _make_mock_session(large_runs)

        with patch("pretia.runner.ProfileRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run_sync.side_effect = [small_session, large_session]

            result = run_validation(str(wf), budget=5.0, small_n=20, large_n=100)

        assert isinstance(result.score.verdict, str)
        assert result.convergence_pct >= 0
        assert len(result.recommendation) > 0


class TestRunValidationSufficientSamples:
    def test_sufficient(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        runs = [
            [
                _make_record("classify", "gpt-4o-mini", 100, 50),
                _make_record("generate", "gpt-4o-mini", 200, 100),
            ]
            for _ in range(100)
        ]
        small_session = _make_mock_session(runs[:20])
        large_session = _make_mock_session(runs)

        with patch("pretia.runner.ProfileRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run_sync.side_effect = [small_session, large_session]

            result = run_validation(str(wf), small_n=20, large_n=100)

        assert "sufficient" in result.recommendation.lower()


class TestRunValidationInsufficientSamples:
    def test_insufficient(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        small_runs = [[_make_record("classify", "gpt-4o-mini", 500, 200)] for _ in range(20)]
        large_runs = [[_make_record("classify", "gpt-4o-mini", 100, 50)] for _ in range(100)]
        small_session = _make_mock_session(small_runs)
        large_session = _make_mock_session(large_runs)

        with patch("pretia.runner.ProfileRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run_sync.side_effect = [small_session, large_session]

            result = run_validation(str(wf), small_n=20, large_n=100)

        assert result.convergence_pct > 30
        assert "50+" in result.recommendation or "warning" in result.recommendation.lower()


class TestFormatValidateReport:
    def test_readable_output(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")

        runs = [
            [
                _make_record("classify", "gpt-4o-mini", 100, 50),
                _make_record("generate", "gpt-4o-mini", 200, 100),
            ]
            for _ in range(100)
        ]
        small_session = _make_mock_session(runs[:20])
        large_session = _make_mock_session(runs)

        with patch("pretia.runner.ProfileRunner") as mock_cls:
            instance = mock_cls.return_value
            instance.run_sync.side_effect = [small_session, large_session]

            result = run_validation(str(wf), small_n=20, large_n=100)

        report = format_validate_report(result)
        assert "Validation" in report
        assert "Convergence" in report
        assert "Verdict" in report
