"""Test the 11 pilot calibration checks from tests/backtesting/pilot_checks.py."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from agentcost.collectors.base import StepRecord
from agentcost.validation.suite import BacktestConfig
from tests.backtesting.pilot_checks import (
    PilotCheckResult,
    check_cache_bust,
    check_cost_plausibility,
    check_cross_provider_accounting,
    check_finish_reason,
    check_output_schema,
    check_pdf_validity,
    check_routing_ratio,
    check_template_substitution,
    check_tier_separation,
    check_w19_history_accumulation,
    run_pilot_checks,
)

_TEST_CONFIG = BacktestConfig(
    name="W1-test",
    archetype="test",
    complexity="simple",
    workflow_path="test.py",
    description="test",
    expected_models=["claude-haiku-3"],
    has_loops=False,
    expected_cost_range=(0.005, 0.03),
)


# ---------------------------------------------------------------------------
# check_cache_bust
# ---------------------------------------------------------------------------


class TestCheckCacheBust:
    """Verify cache bust detection for DeepSeek/Anthropic providers."""

    def test_check_cache_bust_pass(self, sample_record: StepRecord) -> None:
        """Records with no cache_hit_tokens should PASS."""
        # sample_record uses model="claude-haiku-3" (Anthropic), cache_hit_tokens=None
        records = [[sample_record, sample_record]]
        result = check_cache_bust("W1-support-simple", records)

        assert result.status == "PASS"
        assert result.name == "cache_bust"
        assert result.layer == 1
        assert result.blocking is True

    def test_check_cache_bust_fail(self, sample_record: StepRecord) -> None:
        """Record with cache_hit_tokens > 0 should FAIL."""
        cached_record = replace(sample_record, cache_hit_tokens=150, cache_miss_tokens=190)
        records = [[cached_record]]
        result = check_cache_bust("W1-support-simple", records)

        assert result.status == "FAIL"
        assert result.details["violation_count"] == 1
        assert result.details["violations"][0]["cache_hit_tokens"] == 150

    def test_check_cache_bust_not_applicable(self, sample_record: StepRecord) -> None:
        """Non-DeepSeek/Anthropic workflow should PASS with 'not applicable' note."""
        gpt_record = replace(sample_record, model="gpt-4o")
        records = [[gpt_record]]
        result = check_cache_bust("W1-support-simple", records)

        assert result.status == "PASS"
        assert "not applicable" in result.details["note"]


# ---------------------------------------------------------------------------
# check_template_substitution
# ---------------------------------------------------------------------------


class TestCheckTemplateSubstitution:
    """Verify detection of unresolved template placeholders."""

    def test_check_template_substitution_pass(self, sample_record: StepRecord) -> None:
        """Clean inputs with no placeholders should PASS."""
        records = [[sample_record]]
        inputs = [{"query": "How do I reset my password?"}]
        result = check_template_substitution(records, inputs)

        assert result.status == "PASS"
        assert result.details["inputs_checked"] == 1

    def test_check_template_substitution_fail(self, sample_record: StepRecord) -> None:
        """Input containing {{PLACEHOLDER}} should FAIL."""
        records = [[sample_record]]
        inputs = [{"query": "Process this: {{PLACEHOLDER}}"}]
        result = check_template_substitution(records, inputs)

        assert result.status == "FAIL"
        assert result.details["placeholder_count"] == 1
        assert result.details["examples"][0]["location"] == "input[0].query"


# ---------------------------------------------------------------------------
# check_finish_reason
# ---------------------------------------------------------------------------


class TestCheckFinishReason:
    """Verify detection of excessive output truncation."""

    def test_check_finish_reason_pass(self, sample_record: StepRecord) -> None:
        """No truncated records should PASS."""
        # output_truncated defaults to None (not True), so this should pass
        records = [[sample_record] * 10]
        result = check_finish_reason(records)

        assert result.status == "PASS"
        assert result.details["truncated_count"] == 0

    def test_check_finish_reason_fail(self, sample_record: StepRecord) -> None:
        """More than 5% truncated should FAIL."""
        normal = replace(sample_record, output_truncated=False)
        truncated = replace(sample_record, output_truncated=True)
        # 2 truncated out of 10 = 20% > 5%
        records = [[normal] * 8 + [truncated] * 2]
        result = check_finish_reason(records)

        assert result.status == "FAIL"
        assert result.details["truncated_count"] == 2
        assert result.details["truncated_pct"] == 20.0


# ---------------------------------------------------------------------------
# check_tier_separation
# ---------------------------------------------------------------------------


class TestCheckTierSeparation:
    """Verify max/min cost ratio meets the threshold."""

    def test_check_tier_separation_pass(self, sample_record: StepRecord) -> None:
        """Cost ratio >= threshold should PASS."""
        # W1 is a linear workflow, threshold = 2.0
        # Create two runs with different token counts to produce different costs
        cheap_record = replace(
            sample_record,
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=20,
        )
        expensive_record = replace(
            sample_record,
            model="claude-haiku-4-5",
            input_tokens=2000,
            output_tokens=500,
        )
        records = [[cheap_record], [expensive_record]]
        result = check_tier_separation("W1-support-simple", records)

        assert result.status == "PASS"
        assert result.details["ratio"] >= 2.0

    def test_check_tier_separation_warn(self, sample_record: StepRecord) -> None:
        """Cost ratio below threshold should WARN."""
        record = replace(sample_record, model="claude-haiku-4-5")
        records = [[record], [record]]
        result = check_tier_separation("W1-support-simple", records)

        assert result.status == "WARN"
        assert result.details["ratio"] < 2.0


# ---------------------------------------------------------------------------
# check_cost_plausibility
# ---------------------------------------------------------------------------


class TestCheckCostPlausibility:
    """Verify per-run costs fall within plausible range."""

    def test_check_cost_plausibility_pass(self, sample_record: StepRecord) -> None:
        """All costs in range should PASS."""
        # expected_cost_range = (0.005, 0.03), midpoint = 0.0175
        # plausibility bounds: 0.00875 to 0.0875
        # A claude-haiku-3 record with 340 input + 45 output tokens produces a small cost.
        # Create records whose cost falls within the plausibility bounds.
        record = replace(
            sample_record,
            model="claude-haiku-4-5",
            input_tokens=5000,
            output_tokens=1000,
        )
        records = [[record]]
        config = BacktestConfig(
            name="W1-test",
            archetype="test",
            complexity="simple",
            workflow_path="test.py",
            description="test",
            expected_models=["claude-haiku-4-5"],
            has_loops=False,
            expected_cost_range=(0.0001, 0.01),
        )
        result = check_cost_plausibility(records, config)

        assert result.status == "PASS"
        assert result.details["out_of_range_count"] == 0

    def test_check_cost_plausibility_warn(self, sample_record: StepRecord) -> None:
        """Costs outside range should WARN."""
        config = BacktestConfig(
            name="W1-test",
            archetype="test",
            complexity="simple",
            workflow_path="test.py",
            description="test",
            expected_models=["claude-haiku-4-5"],
            has_loops=False,
            expected_cost_range=(50.0, 100.0),
        )
        record = replace(sample_record, model="claude-haiku-4-5")
        records = [[record]]
        result = check_cost_plausibility(records, config)

        assert result.status == "WARN"
        assert result.details["out_of_range_count"] > 0


# ---------------------------------------------------------------------------
# check_output_schema
# ---------------------------------------------------------------------------


class TestCheckOutputSchema:
    """Verify JSON output truncation detection."""

    def test_check_output_schema_pass(self, sample_record: StepRecord) -> None:
        """No truncated JSON outputs should PASS."""
        # sample_record has output_format="json" and output_truncated=None
        records = [[sample_record] * 5]
        result = check_output_schema(records)

        assert result.status == "PASS"
        assert result.name == "output_schema"
        assert result.details["json_step_count"] == 5
        assert result.details["truncated_count"] == 0

    def test_check_output_schema_fail(self, sample_record: StepRecord) -> None:
        """More than 20% truncated JSON outputs should FAIL."""
        normal = replace(sample_record, output_truncated=False)
        truncated = replace(sample_record, output_truncated=True)
        # 3 truncated out of 5 = 60% > 20%
        records = [[normal, normal, truncated, truncated, truncated]]
        result = check_output_schema(records)

        assert result.status == "FAIL"
        assert result.details["truncated_count"] == 3

    def test_check_output_schema_no_json_steps(self, sample_record: StepRecord) -> None:
        """No JSON output steps should PASS with zero counts."""
        text_record = replace(sample_record, output_format="text")
        records = [[text_record]]
        result = check_output_schema(records)

        assert result.status == "PASS"
        assert result.details["json_step_count"] == 0


# ---------------------------------------------------------------------------
# check_cross_provider_accounting
# ---------------------------------------------------------------------------


class TestCheckCrossProviderAccounting:
    """Verify multi-provider model pricing validation."""

    def test_check_cross_provider_not_applicable(self, sample_record: StepRecord) -> None:
        """Single-provider workflow should PASS with 'not applicable' note."""
        records = [[sample_record]]
        result = check_cross_provider_accounting("W1-support-simple", records)

        assert result.status == "PASS"
        assert "not applicable" in result.details["note"]

    def test_check_cross_provider_pass(self, sample_record: StepRecord) -> None:
        """Multi-provider workflow with recognized models should PASS."""
        claude_record = replace(sample_record, model="claude-haiku-4-5")
        gpt_record = replace(sample_record, model="gpt-4.1")
        records = [[claude_record, gpt_record]]
        result = check_cross_provider_accounting("W4-multi-provider", records)

        assert result.status == "PASS"
        assert "models_verified" in result.details

    def test_check_cross_provider_fail(self, sample_record: StepRecord) -> None:
        """Multi-provider workflow with unrecognized model should FAIL."""
        bad_record = replace(sample_record, model="totally-fake-model-xyz-999")
        records = [[bad_record]]
        result = check_cross_provider_accounting("W4-multi-provider", records)

        assert result.status == "FAIL"
        assert "totally-fake-model-xyz-999" in result.details["unrecognized_models"]


# ---------------------------------------------------------------------------
# check_w19_history_accumulation
# ---------------------------------------------------------------------------


class TestCheckW19HistoryAccumulation:
    """Verify W19 multi-turn history growth detection."""

    def test_not_applicable_for_non_w19(self, sample_record: StepRecord) -> None:
        """Non-W19 workflow should PASS with 'not applicable' note."""
        records = [[sample_record]]
        result = check_w19_history_accumulation("W1-support-simple", records)

        assert result.status == "PASS"
        assert "not applicable" in result.details["note"]

    def test_pass_with_growing_history(self, sample_record: StepRecord) -> None:
        """W19 with last/first input ratio >= 5.0 should PASS."""
        first = replace(sample_record, input_tokens=100)
        last = replace(sample_record, input_tokens=600)
        records = [[first, last]]
        result = check_w19_history_accumulation("W19-multi-turn", records)

        assert result.status == "PASS"

    def test_fail_with_flat_history(self, sample_record: StepRecord) -> None:
        """W19 with flat input tokens (ratio < 1.3) should FAIL."""
        first = replace(sample_record, input_tokens=100)
        last = replace(sample_record, input_tokens=120)  # ratio = 1.2 < 1.3
        records = [[first, last]]
        result = check_w19_history_accumulation("W19-multi-turn", records)

        assert result.status == "FAIL"
        assert result.details["failure_count"] == 1


# ---------------------------------------------------------------------------
# check_pdf_validity
# ---------------------------------------------------------------------------


class TestCheckPdfValidity:
    """Verify PDF input validation."""

    def test_not_applicable_for_non_pdf_workflow(self, sample_record: StepRecord) -> None:
        """Non-PDF workflow should PASS with 'not applicable' note."""
        inputs: list[dict[str, Any]] = [{"query": "test"}]
        result = check_pdf_validity("W1-support-simple", inputs)

        assert result.status == "PASS"
        assert "not applicable" in result.details["note"]

    def test_pass_when_no_pdf_paths(self) -> None:
        """PDF workflow with no pdf_path fields should PASS."""
        inputs: list[dict[str, Any]] = [{"query": "test"}]
        result = check_pdf_validity("W14-rag-pdf", inputs)

        assert result.status == "PASS"
        assert "no PDF paths" in result.details["note"]


# ---------------------------------------------------------------------------
# check_routing_ratio
# ---------------------------------------------------------------------------


class TestCheckRoutingRatio:
    """Verify routing path diversity detection."""

    def test_not_applicable_for_non_routing(self, sample_record: StepRecord) -> None:
        """Non-routing workflow should PASS with 'not applicable' note."""
        records = [[sample_record]]
        result = check_routing_ratio("W9-linear", records)

        assert result.status == "PASS"
        assert "not applicable" in result.details["note"]

    def test_pass_with_diverse_paths(self, sample_record: StepRecord) -> None:
        """Routing workflow with multiple distinct paths should PASS."""
        # W13 checks for keywords: simple, research, escalate, path_a, path_b, path_c
        simple_record = replace(sample_record, step_name="simple_handler")
        research_record = replace(sample_record, step_name="research_handler")
        records = [[simple_record], [research_record]]
        result = check_routing_ratio("W13-routing-agent", records)

        assert result.status == "PASS"
        assert result.details["distinct_paths"] >= 2

    def test_warn_with_single_path(self, sample_record: StepRecord) -> None:
        """Routing workflow with only one path should WARN."""
        # Same step name in all runs -> single path
        records = [[sample_record], [sample_record], [sample_record]]
        result = check_routing_ratio("W13-routing-agent", records)

        assert result.status == "WARN"
        assert result.details["distinct_paths"] == 1


# ---------------------------------------------------------------------------
# run_pilot_checks
# ---------------------------------------------------------------------------


class TestRunPilotChecks:
    """Verify the orchestrator returns all check results."""

    def test_run_pilot_checks_returns_all(self, sample_record: StepRecord) -> None:
        """run_pilot_checks should return a list of PilotCheckResult for all 11 checks."""
        records = [[sample_record] * 3, [sample_record] * 3]
        inputs: list[dict[str, Any]] = [{"query": "test input"}]
        results = run_pilot_checks("W1-support-simple", records, inputs, _TEST_CONFIG)

        assert isinstance(results, list)
        assert all(isinstance(r, PilotCheckResult) for r in results)
        # 7 Layer 1 + 4 Layer 2 = 11 checks
        assert len(results) == 11

        check_names = {r.name for r in results}
        expected_names = {
            "cache_bust",
            "template_substitution",
            "cross_provider_accounting",
            "w19_history_accumulation",
            "pdf_validity",
            "output_schema",
            "finish_reason",
            "tier_separation",
            "routing_ratio",
            "cost_plausibility",
            "detector_preactivation",
        }
        assert check_names == expected_names

    def test_run_pilot_checks_layer_classification(self, sample_record: StepRecord) -> None:
        """Layer 1 checks should be blocking, Layer 2 should not."""
        records = [[sample_record]]
        inputs: list[dict[str, Any]] = [{"query": "test"}]
        results = run_pilot_checks("W1-support-simple", records, inputs, _TEST_CONFIG)

        layer1 = [r for r in results if r.layer == 1]
        layer2 = [r for r in results if r.layer == 2]

        assert len(layer1) == 7
        assert len(layer2) == 4
        assert all(r.blocking is True for r in layer1)
        assert all(r.blocking is False for r in layer2)
