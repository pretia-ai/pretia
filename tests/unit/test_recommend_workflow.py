"""Tests for pretia.recommend.workflow — LoopCapGenerator and CircuitBreakerGenerator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pretia.collectors.base import StepRecord
from pretia.recommend.base import _DEFAULT_DAILY_VOLUME, _safe_record_cost
from pretia.recommend.workflow import (
    CircuitBreakerGenerator,
    LoopCapGenerator,
    _iter_distribution,
    percentile,
)
from pretia.store import ProfilingSession


def _make_record(
    step_name: str = "research_loop",
    model: str = "gpt-4o",
    input_tokens: int = 500,
    output_tokens: int = 200,
    iteration: int = 1,
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": 700,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 100,
        "output_format": "text",
        "is_retry": False,
        "iteration": iteration,
        "parent_step": None,
        "duration_ms": 500,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


def _make_session(
    runs: list[list[StepRecord]],
    patterns: list[dict] | None = None,
) -> ProfilingSession:
    return ProfilingSession(
        workflow_name="test_workflow",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=len(runs),
        input_mode="auto",
        runs=runs,
        metadata={"patterns": patterns or []},
    )


def _loop_variance_pattern(
    step_name: str = "research_loop",
    cv: float = 0.8,
    mean_iterations: float = 5.0,
    min_iterations: int = 2,
    max_iterations: int = 12,
) -> dict:
    """Build a serialized loop_count_variance pattern dict."""
    return {
        "pattern_type": "loop_count_variance",
        "step_name": step_name,
        "severity": "warning",
        "description": f"Loop count variance detected in {step_name}",
        "evidence": {
            "cv": cv,
            "mean_iterations": mean_iterations,
            "min_iterations": min_iterations,
            "max_iterations": max_iterations,
        },
    }


def _build_loop_runs(
    step_name: str = "research_loop",
    iter_counts: list[int] | None = None,
    model: str = "gpt-4o",
    input_tokens: int = 500,
    output_tokens: int = 200,
) -> list[list[StepRecord]]:
    """Build runs where each run has iter_count iterations of the step."""
    if iter_counts is None:
        iter_counts = [3, 4, 5, 6, 8]
    runs: list[list[StepRecord]] = []
    for count in iter_counts:
        run = [
            _make_record(
                step_name=step_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                iteration=i,
            )
            for i in range(1, count + 1)
        ]
        runs.append(run)
    return runs


class TestPercentile:
    def test_median_odd(self) -> None:
        assert percentile([1.0, 2.0, 3.0], 50) == pytest.approx(2.0)

    def test_p75(self) -> None:
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert percentile(vals, 75) == pytest.approx(4.0)

    def test_single_value(self) -> None:
        assert percentile([42.0], 50) == pytest.approx(42.0)

    def test_empty(self) -> None:
        assert percentile([], 50) == 0.0


class TestIterDistribution:
    def test_basic(self) -> None:
        runs = _build_loop_runs(iter_counts=[3, 5, 7])
        dist = _iter_distribution(runs, "research_loop")
        assert dist == [3, 5, 7]

    def test_step_not_in_run(self) -> None:
        runs = _build_loop_runs(iter_counts=[3])
        dist = _iter_distribution(runs, "nonexistent_step")
        assert dist == []

    def test_sorted(self) -> None:
        runs = _build_loop_runs(iter_counts=[7, 3, 5])
        dist = _iter_distribution(runs, "research_loop")
        assert dist == [3, 5, 7]


class TestLoopCapGenerator:
    def test_fires_with_high_cv(self) -> None:
        """Loop cap fires when loop_count_variance pattern exists with CV > 0.5."""
        iter_counts = [3, 4, 5, 6, 12]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=0.8, mean_iterations=6.0, max_iterations=12)
        session = _make_session(runs, patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.type == "workflow"
        assert rec.id == "loop-cap-research_loop"
        assert rec.confidence == "MODERATE"
        assert rec.affected_steps == ["research_loop"]

    def test_cap_at_p75(self) -> None:
        """Recommended cap should be at p75 of iteration distribution."""
        iter_counts = [3, 4, 5, 6, 12]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=0.8, mean_iterations=6.0)
        session = _make_session(runs, patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].evidence["recommended_cap"] == recs[0].evidence["p75"]

    def test_savings_calculation(self) -> None:
        """Verify savings match cost of iterations above cap."""
        iter_counts = [3, 4, 5, 6, 12]
        runs = _build_loop_runs(
            iter_counts=iter_counts, model="gpt-4o", input_tokens=500, output_tokens=200
        )
        pattern = _loop_variance_pattern(cv=0.8, mean_iterations=6.0)
        session = _make_session(runs, patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        rec = recs[0]

        cap = rec.evidence["recommended_cap"]
        cost_per_record = _safe_record_cost("gpt-4o", 500, 200)

        total_excess = 0.0
        for count in iter_counts:
            excess_iters = max(0, count - cap)
            total_excess += excess_iters * cost_per_record

        avg_excess = total_excess / len(iter_counts)
        expected_monthly = round(avg_excess * _DEFAULT_DAILY_VOLUME * 30, 2)

        assert rec.monthly_savings == pytest.approx(expected_monthly, rel=1e-4)

    def test_no_recommendation_without_pattern(self) -> None:
        runs = _build_loop_runs(iter_counts=[3, 5, 7])
        session = _make_session(runs, patterns=[])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_no_recommendation_low_cv(self) -> None:
        """CV <= 0.5 → no recommendation."""
        runs = _build_loop_runs(iter_counts=[4, 5, 5, 5, 6])
        pattern = _loop_variance_pattern(cv=0.3, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_no_recommendation_when_cap_equals_max(self) -> None:
        """If all runs have the same iteration count, cap == max → skip."""
        runs = _build_loop_runs(iter_counts=[5, 5, 5, 5, 5])
        pattern = _loop_variance_pattern(cv=0.6, mean_iterations=5.0, max_iterations=5)
        session = _make_session(runs, patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_empty_runs(self) -> None:
        pattern = _loop_variance_pattern()
        session = _make_session([], patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_evidence_fields(self) -> None:
        runs = _build_loop_runs(iter_counts=[3, 4, 5, 6, 12])
        pattern = _loop_variance_pattern(cv=0.8, mean_iterations=6.0)
        session = _make_session(runs, patterns=[pattern])

        gen = LoopCapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        ev = recs[0].evidence
        assert "cv" in ev
        assert "mean_iterations" in ev
        assert "max_iterations" in ev
        assert "recommended_cap" in ev
        assert "iteration_distribution" in ev
        assert "p75" in ev
        assert "p90" in ev
        assert "daily_volume" in ev
        assert ev["daily_volume"] == _DEFAULT_DAILY_VOLUME


class TestCircuitBreakerGenerator:
    def test_fires_with_outliers_and_high_cost_share(self) -> None:
        """Circuit breaker fires when outlier runs > 2x mean AND cost share > 15%."""
        # mean ~5, threshold ~10. Run with 20 iterations is a clear outlier.
        iter_counts = [4, 5, 5, 6, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.type == "workflow"
        assert rec.id == "circuit-breaker-research_loop"
        assert rec.confidence == "HIGH"
        assert rec.affected_steps == ["research_loop"]

    def test_threshold_is_2x_mean(self) -> None:
        iter_counts = [4, 5, 5, 6, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].evidence["threshold"] == 10  # ceil(2 * 5.0)

    def test_savings_calculation(self) -> None:
        """Savings = outlier excess cost, averaged across runs, * volume * 30."""
        iter_counts = [4, 5, 5, 6, 20]
        runs = _build_loop_runs(
            iter_counts=iter_counts, model="gpt-4o", input_tokens=500, output_tokens=200
        )
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        rec = recs[0]

        threshold = rec.evidence["threshold"]  # 10
        cost_per_record = _safe_record_cost("gpt-4o", 500, 200)

        # Only the 20-iteration run is an outlier. Excess iterations: 20 - 10 = 10
        excess_cost = (20 - threshold) * cost_per_record
        avg_excess = excess_cost / len(iter_counts)
        expected_monthly = round(avg_excess * _DEFAULT_DAILY_VOLUME * 30, 2)

        assert rec.monthly_savings == pytest.approx(expected_monthly, rel=1e-4)

    def test_no_recommendation_when_no_outliers(self) -> None:
        """All runs within 2x mean → no recommendation."""
        iter_counts = [4, 5, 5, 6, 7]
        runs = _build_loop_runs(iter_counts=iter_counts)
        # mean=5.4, threshold=ceil(10.8)=11, max=7 < 11
        pattern = _loop_variance_pattern(cv=0.6, mean_iterations=5.4)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_no_recommendation_when_cost_share_low(self) -> None:
        """Outliers exist but cost share <= 15% → no recommendation."""
        # 20 normal runs + 1 slightly-outlier run. mean_iterations=4.0,
        # threshold = ceil(8) = 8. Run with 9 is an outlier.
        # total cost: 20*5 + 9 = 109 records. outlier cost = 9 records.
        # cost share = 9/109 = 8.3% < 15%
        runs = _build_loop_runs(iter_counts=[5] * 20 + [9])
        pattern = _loop_variance_pattern(cv=0.6, mean_iterations=4.0)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_no_recommendation_without_pattern(self) -> None:
        runs = _build_loop_runs(iter_counts=[4, 5, 5, 6, 20])
        session = _make_session(runs, patterns=[])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_empty_runs(self) -> None:
        pattern = _loop_variance_pattern()
        session = _make_session([], patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_evidence_fields(self) -> None:
        iter_counts = [4, 5, 5, 6, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        ev = recs[0].evidence
        assert "mean_iterations" in ev
        assert "threshold" in ev
        assert "outlier_run_count" in ev
        assert "total_run_count" in ev
        assert "outlier_cost_share" in ev
        assert ev["outlier_cost_share"] > 0.15
        assert "daily_volume" in ev

    def test_outlier_count_correct(self) -> None:
        """Verify exactly the right number of runs are flagged as outliers."""
        # mean=5, threshold=10. Two runs are outliers (15 and 20).
        iter_counts = [4, 5, 5, 6, 15, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.5, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        gen = CircuitBreakerGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].evidence["outlier_run_count"] == 2
        assert recs[0].evidence["total_run_count"] == 6


class TestBothGeneratorsTogether:
    def test_both_fire_for_same_step(self) -> None:
        """LoopCap and CircuitBreaker can both produce recommendations for the same step."""
        iter_counts = [3, 4, 5, 6, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        loop_recs = LoopCapGenerator().generate(session)
        cb_recs = CircuitBreakerGenerator().generate(session)

        assert len(loop_recs) >= 1
        assert len(cb_recs) >= 1

        all_ids = {r.id for r in loop_recs + cb_recs}
        assert "loop-cap-research_loop" in all_ids
        assert "circuit-breaker-research_loop" in all_ids

    def test_different_ids_no_conflict(self) -> None:
        """The two generators produce different recommendation ids."""
        iter_counts = [3, 4, 5, 6, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        loop_recs = LoopCapGenerator().generate(session)
        cb_recs = CircuitBreakerGenerator().generate(session)

        loop_ids = {r.id for r in loop_recs}
        cb_ids = {r.id for r in cb_recs}
        assert loop_ids.isdisjoint(cb_ids)

    def test_circuit_breaker_higher_confidence(self) -> None:
        """Circuit breaker always has HIGH confidence vs MODERATE for loop cap."""
        iter_counts = [3, 4, 5, 6, 20]
        runs = _build_loop_runs(iter_counts=iter_counts)
        pattern = _loop_variance_pattern(cv=1.2, mean_iterations=5.0)
        session = _make_session(runs, patterns=[pattern])

        loop_recs = LoopCapGenerator().generate(session)
        cb_recs = CircuitBreakerGenerator().generate(session)

        if loop_recs:
            assert loop_recs[0].confidence == "MODERATE"
        if cb_recs:
            assert cb_recs[0].confidence == "HIGH"


class TestLowConfidenceFallback:
    """When savings < $1/month but pattern is severe, emit with LOW confidence."""

    def test_loop_cap_low_confidence_severe_cv(self) -> None:
        iter_counts = [1, 1, 1, 1, 1, 1, 1, 1, 2, 5]
        runs = _build_loop_runs(
            iter_counts=iter_counts,
            model="claude-haiku-4-5",
            input_tokens=1,
            output_tokens=1,
        )
        pattern = _loop_variance_pattern(
            cv=3.0, mean_iterations=1.6, min_iterations=1, max_iterations=5
        )
        session = _make_session(runs, patterns=[pattern])
        recs = LoopCapGenerator().generate(session)
        assert len(recs) >= 1
        assert recs[0].confidence == "LOW"

    def test_loop_cap_no_rec_when_cv_low_and_savings_low(self) -> None:
        iter_counts = [1, 1, 1, 1, 1, 1, 1, 1, 2, 5]
        runs = _build_loop_runs(
            iter_counts=iter_counts,
            model="claude-haiku-4-5",
            input_tokens=1,
            output_tokens=1,
        )
        pattern = _loop_variance_pattern(cv=0.8, mean_iterations=1.6)
        session = _make_session(runs, patterns=[pattern])
        recs = LoopCapGenerator().generate(session)
        assert recs == []

    def test_circuit_breaker_low_confidence_high_cost_share(self) -> None:
        iter_counts = [1, 1, 1, 1, 1, 1, 1, 1, 2, 5]
        runs = _build_loop_runs(
            iter_counts=iter_counts,
            model="claude-haiku-4-5",
            input_tokens=1,
            output_tokens=1,
        )
        pattern = _loop_variance_pattern(
            cv=3.0, mean_iterations=1.6, min_iterations=1, max_iterations=5
        )
        session = _make_session(runs, patterns=[pattern])
        recs = CircuitBreakerGenerator().generate(session)
        assert len(recs) >= 1
        assert recs[0].confidence == "LOW"
