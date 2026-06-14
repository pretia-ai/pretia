"""Tests for agentcost.recommend.registry — generator registry and deduplication."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import patch

from agentcost.recommend.base import Recommendation, RecommendationGenerator
from agentcost.recommend.registry import _GENERATORS, generate_recommendations, register
from agentcost.store import ProfilingSession


def _make_session(
    runs: list | None = None,
    metadata: dict | None = None,
) -> ProfilingSession:
    return ProfilingSession(
        workflow_name="test_workflow",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=3,
        input_mode="auto",
        runs=runs or [],
        metadata=metadata or {},
    )


def _make_rec(
    id: str = "test-rec",
    type: str = "model_swap",
    monthly_savings: float = 100.0,
    confidence: str = "HIGH",
    priority: int = 100,
    **kwargs: object,
) -> Recommendation:
    defaults: dict[str, object] = {
        "id": id,
        "type": type,
        "title": f"Test {id}",
        "description": "Test recommendation.",
        "monthly_savings": monthly_savings,
        "confidence": confidence,
        "affected_steps": ["step_a"],
        "evidence": {},
        "priority": priority,
    }
    defaults.update(kwargs)
    return Recommendation(**defaults)


class _EmptyGenerator(RecommendationGenerator):
    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        return []


class _SingleGenerator(RecommendationGenerator):
    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        return [_make_rec(id="single-rec", priority=200, monthly_savings=200.0)]


class _DuplicateIdGenerator(RecommendationGenerator):
    """Produces a recommendation with the same id as _SingleGenerator but lower priority."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        return [_make_rec(id="single-rec", priority=50, monthly_savings=50.0)]


class _HighPriorityDuplicateGenerator(RecommendationGenerator):
    """Produces same id as _SingleGenerator but higher priority."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        return [_make_rec(id="single-rec", priority=500, monthly_savings=500.0)]


class _MultiGenerator(RecommendationGenerator):
    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        return [
            _make_rec(id="rec-a", priority=300, monthly_savings=300.0),
            _make_rec(id="rec-b", priority=100, monthly_savings=100.0),
        ]


class _FailingGenerator(RecommendationGenerator):
    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        raise RuntimeError("Generator crashed")


class TestRegisterDecorator:
    def test_register_adds_to_list(self) -> None:
        original_len = len(_GENERATORS)
        try:

            @register
            class _TestGen(RecommendationGenerator):
                def generate(self, profile: ProfilingSession) -> list[Recommendation]:
                    return []

            assert len(_GENERATORS) == original_len + 1
            assert _GENERATORS[-1] is _TestGen
        finally:
            if len(_GENERATORS) > original_len:
                _GENERATORS.pop()


class TestGenerateRecommendations:
    def test_empty_registry(self) -> None:
        session = _make_session()
        with patch.object(
            __import__("agentcost.recommend.registry", fromlist=["_GENERATORS"]),
            "_GENERATORS",
            [],
        ):
            result = generate_recommendations(session)
        assert result == []

    def test_single_generator(self) -> None:
        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS", [_SingleGenerator]
        ):
            result = generate_recommendations(session)
        assert len(result) == 1
        assert result[0].id == "single-rec"

    def test_empty_generator_returns_empty(self) -> None:
        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS", [_EmptyGenerator]
        ):
            result = generate_recommendations(session)
        assert result == []

    def test_sorted_by_priority_descending(self) -> None:
        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS", [_MultiGenerator]
        ):
            result = generate_recommendations(session)
        assert len(result) == 2
        assert result[0].priority >= result[1].priority
        assert result[0].id == "rec-a"
        assert result[1].id == "rec-b"

    def test_dedup_keeps_higher_priority(self) -> None:
        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS",
            [_SingleGenerator, _DuplicateIdGenerator],
        ):
            result = generate_recommendations(session)
        assert len(result) == 1
        assert result[0].id == "single-rec"
        assert result[0].priority == 200

    def test_dedup_keeps_higher_priority_reversed_order(self) -> None:
        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS",
            [_DuplicateIdGenerator, _HighPriorityDuplicateGenerator],
        ):
            result = generate_recommendations(session)
        assert len(result) == 1
        assert result[0].priority == 500

    def test_different_ids_same_step_both_kept(self) -> None:
        class _GenA(RecommendationGenerator):
            def generate(self, profile: ProfilingSession) -> list[Recommendation]:
                return [
                    _make_rec(
                        id="loop-cap-loop_step",
                        type="workflow",
                        affected_steps=["loop_step"],
                        priority=200,
                        monthly_savings=200.0,
                    )
                ]

        class _GenB(RecommendationGenerator):
            def generate(self, profile: ProfilingSession) -> list[Recommendation]:
                return [
                    _make_rec(
                        id="circuit-breaker-loop_step",
                        type="workflow",
                        affected_steps=["loop_step"],
                        priority=300,
                        monthly_savings=300.0,
                    )
                ]

        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS", [_GenA, _GenB]
        ):
            result = generate_recommendations(session)
        assert len(result) == 2
        ids = {r.id for r in result}
        assert ids == {"loop-cap-loop_step", "circuit-breaker-loop_step"}

    def test_failing_generator_does_not_crash(self) -> None:
        session = _make_session()
        with patch(
            "agentcost.recommend.registry._GENERATORS",
            [_FailingGenerator, _SingleGenerator],
        ):
            result = generate_recommendations(session)
        assert len(result) == 1
        assert result[0].id == "single-rec"
