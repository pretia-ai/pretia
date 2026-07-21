"""Tests for AUDIT.md reliability fixes."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pretia.collectors.base import StepRecord
from pretia.projection.stats import percentile, robust_cv


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    iteration: int = 1,
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": input_tokens,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 50,
        "output_format": "text",
        "is_retry": False,
        "iteration": iteration,
        "parent_step": None,
        "duration_ms": 500,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


# ---------------------------------------------------------------------------
# Fix 1: CLT projection
# ---------------------------------------------------------------------------


class TestBootstrapProjection:
    def _make_stats(self, run_costs: list[float]):
        from pretia.projection.stats import (
            ProfilingStats,
            RunStats,
            compute_percentile_stats,
        )

        cpr = compute_percentile_stats(run_costs)
        run_stats = [
            RunStats(
                run_index=i,
                total_cost=c,
                total_tokens=100,
                total_input_tokens=80,
                total_output_tokens=20,
                step_count=1,
                duration_ms=500,
            )
            for i, c in enumerate(run_costs)
        ]
        return ProfilingStats(
            step_stats={},
            run_stats=run_stats,
            cost_per_run=cpr,
            tokens_per_run=cpr,
            total_runs=len(run_costs),
            total_steps=len(run_costs),
        )

    def test_p95_less_than_naive_at_high_volume(self):
        from pretia.projection.projector import _linear_project

        run_costs = [0.005, 0.008, 0.01, 0.01, 0.012, 0.015, 0.02, 0.025, 0.03, 0.03]
        stats = self._make_stats(run_costs)
        result = _linear_project(stats, [10000])
        daily_p95 = result[10000].daily_cost.p95
        naive_p95 = max(run_costs) * 10000
        assert daily_p95 < naive_p95

    def test_bootstrap_daily_basic(self):
        from pretia.projection.projector import _bootstrap_daily

        run_costs = [0.01] * 50
        result = _bootstrap_daily(run_costs, daily_volume=1000)
        assert set(result.keys()) == {"p50", "p75", "p90", "p95", "p99", "mean"}
        assert result["mean"] == pytest.approx(10.0, rel=0.01)

    def test_bootstrap_preserves_heavy_tails(self):
        from pretia.projection.projector import _bootstrap_daily

        bimodal = [0.001] * 40 + [0.10] * 10
        result = _bootstrap_daily(bimodal, daily_volume=1000)
        mean_n = sum(bimodal) / len(bimodal) * 1000
        assert result["p95"] > mean_n * 1.05


# ---------------------------------------------------------------------------
# Fix 2: float("inf") JSON crash
# ---------------------------------------------------------------------------


class TestBimodalInfFix:
    def test_bimodal_bic_delta_serializable(self):
        from pretia.projection.patterns import DetectedPattern

        pattern = DetectedPattern(
            pattern_type="bimodality",
            step_name="_workflow_",
            severity="warning",
            evidence={},
            description="test",
            bimodal_bic_delta=1e10,
        )
        d = pattern.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)


# ---------------------------------------------------------------------------
# Fix 6: robust_cv median=0
# ---------------------------------------------------------------------------


class TestRobustCvMedianZero:
    def test_detects_variance_with_zero_median(self):
        assert robust_cv([0.0, 0.0, 0.0, 0.5, 1.0]) > 0

    def test_all_zeros_returns_zero(self):
        assert robust_cv([0.0, 0.0, 0.0, 0.0]) == 0.0


# ---------------------------------------------------------------------------
# Fix 14: Consolidated percentile
# ---------------------------------------------------------------------------


class TestCanonicalPercentile:
    def test_empty_list_returns_zero(self):
        assert percentile([], 50) == 0.0

    def test_single_value(self):
        assert percentile([42.0], 95) == 42.0

    def test_basic_median(self):
        assert percentile([1.0, 2.0, 3.0, 4.0, 5.0], 50) == 3.0

    def test_imported_in_montecarlo(self):
        from pretia.projection.montecarlo import percentile as mc_pct

        assert mc_pct is percentile

    def test_imported_in_workflow(self):
        from pretia.recommend.workflow import percentile as wf_pct

        assert wf_pct is percentile


# ---------------------------------------------------------------------------
# Fix 20: model_tier ValueError
# ---------------------------------------------------------------------------


class TestModelTierError:
    def test_raises_value_error_not_key_error(self):
        from pretia.pricing.tables import MODEL_PRICING, MODEL_TIERS, register_model

        register_model("test-audit-model", input_price=1.0, output_price=2.0, tier="mid")
        del MODEL_TIERS["test-audit-model"]

        from pretia.pricing.tables import model_tier

        with pytest.raises(ValueError, match="No tier"):
            model_tier("test-audit-model")

        del MODEL_PRICING["test-audit-model"]


# ---------------------------------------------------------------------------
# Fix 18: from_dict KeyError
# ---------------------------------------------------------------------------


class TestFromDictError:
    def test_missing_field_gives_value_error(self):
        with pytest.raises(ValueError, match="Missing required field"):
            StepRecord.from_dict({"step_name": "test"})


# ---------------------------------------------------------------------------
# Fix 27: MODEL_SWAP skips 0-output steps
# ---------------------------------------------------------------------------


class TestModelSwapZeroOutput:
    def test_skips_zero_output_steps(self):
        from pretia.recommend.model_swap import ModelSwapGenerator
        from pretia.store import ProfilingSession

        runs = [
            [_make_record(model="gpt-4o", output_tokens=0, output_format="json")] for _ in range(5)
        ]
        session = ProfilingSession(
            workflow_name="test",
            workflow_hash="abc",
            profiled_at=datetime(2026, 5, 25, tzinfo=UTC),
            sample_size=5,
            input_mode="auto",
            runs=runs,
            metadata={},
        )
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert recs == []
