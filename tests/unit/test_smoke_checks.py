"""Tests for the single-run smoke check functions in scripts/_smoke_checks.py."""

from __future__ import annotations

from dataclasses import replace

from agentcost.collectors.base import StepRecord
from scripts._smoke_checks import (
    check_cache_bust,
    check_cost_plausibility,
    check_cross_provider_accounting,
    check_finish_reason,
    check_nonzero_tokens,
    check_output_schema,
    check_step_count,
    check_template_substitution,
    run_smoke_checks,
)
from scripts._validation_types import CheckStatus


class TestCheckCacheBust:
    def test_pass_no_cache_hits(self, sample_record: StepRecord) -> None:
        result = check_cache_bust("W1", [sample_record, sample_record])
        assert result.status == CheckStatus.PASS
        assert result.blocking is True

    def test_fail_cache_hit_detected(self, sample_record: StepRecord) -> None:
        cached = replace(sample_record, cache_hit_tokens=150, cache_miss_tokens=190)
        result = check_cache_bust("W1", [cached])
        assert result.status == CheckStatus.FAIL
        assert result.details["violation_count"] == 1

    def test_not_applicable_non_cache_provider(self, sample_record: StepRecord) -> None:
        gpt = replace(sample_record, model="gpt-4o")
        result = check_cache_bust("W9", [gpt])
        assert result.status == CheckStatus.PASS
        assert "not applicable" in result.details["note"]


class TestCheckTemplateSubstitution:
    def test_pass_clean_input(self, sample_record: StepRecord) -> None:
        result = check_template_substitution(
            [sample_record], {"query": "How do I reset my password?"}
        )
        assert result.status == CheckStatus.PASS

    def test_fail_unresolved_placeholder(self, sample_record: StepRecord) -> None:
        result = check_template_substitution(
            [sample_record], {"query": "Process: {{PLACEHOLDER}}"}
        )
        assert result.status == CheckStatus.FAIL
        assert result.details["placeholder_count"] >= 1

    def test_fail_placeholder_in_step_name(self, sample_record: StepRecord) -> None:
        bad = replace(sample_record, step_name="step_{{NAME}}")
        result = check_template_substitution([bad], {"query": "clean"})
        assert result.status == CheckStatus.FAIL


class TestCheckCrossProviderAccounting:
    def test_pass_single_provider(self, sample_record: StepRecord) -> None:
        result = check_cross_provider_accounting("W1", [sample_record])
        assert result.status == CheckStatus.PASS
        assert "not applicable" in result.details["note"]

    def test_pass_multi_provider_known_models(self, sample_record: StepRecord) -> None:
        r1 = replace(sample_record, model="claude-haiku-4-5")
        r2 = replace(sample_record, model="gpt-4o")
        result = check_cross_provider_accounting("W14", [r1, r2])
        assert result.status == CheckStatus.PASS

    def test_fail_unrecognized_model(self, sample_record: StepRecord) -> None:
        bad = replace(sample_record, model="unknown-model-xyz")
        result = check_cross_provider_accounting("W14", [bad])
        assert result.status == CheckStatus.FAIL
        assert "unknown-model-xyz" in result.details["unrecognized_models"]


class TestCheckFinishReason:
    def test_pass_no_truncation(self, sample_record: StepRecord) -> None:
        result = check_finish_reason([sample_record])
        assert result.status == CheckStatus.PASS

    def test_fail_any_truncation(self, sample_record: StepRecord) -> None:
        truncated = replace(sample_record, output_truncated=True)
        result = check_finish_reason([truncated, sample_record])
        assert result.status == CheckStatus.FAIL

    def test_pass_empty_records(self) -> None:
        result = check_finish_reason([])
        assert result.status == CheckStatus.PASS


class TestCheckOutputSchema:
    def test_pass_no_json_steps(self, sample_record: StepRecord) -> None:
        non_json = replace(sample_record, output_format="text")
        result = check_output_schema([non_json])
        assert result.status == CheckStatus.PASS

    def test_pass_json_no_truncation(self, sample_record: StepRecord) -> None:
        result = check_output_schema([sample_record])
        assert result.status == CheckStatus.PASS

    def test_fail_json_truncation(self, sample_record: StepRecord) -> None:
        truncated = replace(sample_record, output_truncated=True)
        result = check_output_schema([truncated])
        assert result.status == CheckStatus.FAIL


class TestCheckCostPlausibility:
    def test_pass_within_range(self, sample_record: StepRecord) -> None:
        cost_range = (0.0001, 0.01)
        result = check_cost_plausibility([sample_record], cost_range)
        assert result.status in (CheckStatus.PASS, CheckStatus.WARN)

    def test_warn_out_of_range(self, sample_record: StepRecord) -> None:
        cost_range = (100.0, 200.0)
        result = check_cost_plausibility([sample_record], cost_range)
        assert result.status == CheckStatus.WARN

    def test_warn_empty_records(self) -> None:
        result = check_cost_plausibility([], (0.01, 0.1))
        assert result.status == CheckStatus.WARN


class TestCheckNonzeroTokens:
    def test_pass_all_nonzero(self, sample_record: StepRecord) -> None:
        result = check_nonzero_tokens([sample_record])
        assert result.status == CheckStatus.PASS

    def test_fail_zero_input(self, sample_record: StepRecord) -> None:
        zero = replace(sample_record, input_tokens=0)
        result = check_nonzero_tokens([zero])
        assert result.status == CheckStatus.FAIL

    def test_fail_zero_output(self, sample_record: StepRecord) -> None:
        zero = replace(sample_record, output_tokens=0)
        result = check_nonzero_tokens([zero])
        assert result.status == CheckStatus.FAIL

    def test_pass_empty_records(self) -> None:
        result = check_nonzero_tokens([])
        assert result.status == CheckStatus.PASS


class TestCheckStepCount:
    def test_pass_within_range(self, sample_record: StepRecord) -> None:
        result = check_step_count("W1", [sample_record, sample_record], (1, 5))
        assert result.status == CheckStatus.PASS

    def test_fail_below_range(self, sample_record: StepRecord) -> None:
        result = check_step_count("W1", [sample_record], (3, 10))
        assert result.status == CheckStatus.FAIL

    def test_fail_above_range(self, sample_record: StepRecord) -> None:
        records = [sample_record] * 20
        result = check_step_count("W1", records, (1, 5))
        assert result.status == CheckStatus.FAIL


class TestRunSmokeChecks:
    def test_returns_all_checks(self, sample_record: StepRecord) -> None:
        results = run_smoke_checks(
            workflow_id="W1",
            records=[sample_record],
            input_data={"query": "test"},
            expected_cost_range=(0.0001, 0.01),
            expected_step_range=(1, 5),
        )
        assert len(results) == 8
        names = {r.name for r in results}
        assert "cache_bust" in names
        assert "template_substitution" in names
        assert "cross_provider_accounting" in names
        assert "finish_reason" in names
        assert "output_schema" in names
        assert "cost_plausibility" in names
        assert "nonzero_tokens" in names
        assert "step_count" in names

    def test_all_blocking(self, sample_record: StepRecord) -> None:
        results = run_smoke_checks(
            workflow_id="W1",
            records=[sample_record],
            input_data={"query": "test"},
            expected_cost_range=(0.0001, 0.01),
            expected_step_range=(1, 5),
        )
        assert all(r.blocking for r in results)
