"""Tests for pretia.recommend.base — Recommendation dataclass and helpers."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pretia.collectors.base import StepRecord
from pretia.recommend.base import (
    CONFIDENCE_WEIGHTS,
    Recommendation,
    _extract_pattern_dicts,
    _safe_record_cost,
    compute_priority,
)
from pretia.store import ProfilingSession


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": 100,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 50,
        "output_format": "text",
        "is_retry": False,
        "iteration": 1,
        "parent_step": None,
        "duration_ms": 500,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


def _make_session(
    runs: list[list[StepRecord]] | None = None,
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


def _make_recommendation(**kwargs: object) -> Recommendation:
    defaults: dict[str, object] = {
        "id": "model-swap-classify",
        "type": "model_swap",
        "title": "Swap classify to Haiku",
        "description": "Classification task can use a cheaper model.",
        "monthly_savings": 500.0,
        "confidence": "HIGH",
        "affected_steps": ["classify"],
        "evidence": {"ratio": 0.15},
        "priority": 500,
    }
    defaults.update(kwargs)
    return Recommendation(**defaults)


class TestRecommendation:
    def test_create_valid(self) -> None:
        rec = _make_recommendation()
        assert rec.id == "model-swap-classify"
        assert rec.type == "model_swap"
        assert rec.monthly_savings == 500.0
        assert rec.confidence == "HIGH"

    def test_frozen(self) -> None:
        rec = _make_recommendation()
        with pytest.raises(AttributeError):
            rec.title = "new title"  # type: ignore[misc]

    def test_invalid_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="type must be one of"):
            _make_recommendation(type="invalid")

    def test_invalid_confidence_rejected(self) -> None:
        with pytest.raises(ValueError, match="confidence must be one of"):
            _make_recommendation(confidence="VERY_HIGH")

    def test_all_valid_types(self) -> None:
        for t in ("model_swap", "architecture", "workflow"):
            rec = _make_recommendation(type=t)
            assert rec.type == t

    def test_all_valid_confidences(self) -> None:
        for c in ("HIGH", "MODERATE", "LOW"):
            rec = _make_recommendation(confidence=c)
            assert rec.confidence == c


class TestToDict:
    def test_serializes_all_fields(self) -> None:
        rec = _make_recommendation()
        d = rec.to_dict()
        assert d["id"] == "model-swap-classify"
        assert d["type"] == "model_swap"
        assert d["title"] == "Swap classify to Haiku"
        assert d["monthly_savings"] == 500.0
        assert d["confidence"] == "HIGH"
        assert d["affected_steps"] == ["classify"]
        assert d["evidence"] == {"ratio": 0.15}
        assert d["priority"] == 500

    def test_json_serializable(self) -> None:
        rec = _make_recommendation()
        s = json.dumps(rec.to_dict())
        assert isinstance(s, str)

    def test_affected_steps_is_copy(self) -> None:
        steps = ["step_a", "step_b"]
        rec = _make_recommendation(affected_steps=steps)
        d = rec.to_dict()
        d["affected_steps"].append("step_c")
        assert rec.affected_steps == ["step_a", "step_b"]


class TestComputePriority:
    def test_high_confidence(self) -> None:
        assert compute_priority(1000.0, "HIGH") == 1000

    def test_moderate_confidence(self) -> None:
        assert compute_priority(1000.0, "MODERATE") == 600

    def test_low_confidence(self) -> None:
        assert compute_priority(1000.0, "LOW") == 300

    def test_unknown_confidence_uses_low_weight(self) -> None:
        assert compute_priority(1000.0, "UNKNOWN") == 300

    def test_zero_savings(self) -> None:
        assert compute_priority(0.0, "HIGH") == 0

    def test_all_weights_documented(self) -> None:
        assert set(CONFIDENCE_WEIGHTS) == {"HIGH", "MODERATE", "LOW"}


class TestExtractPatternDicts:
    def test_returns_patterns_from_metadata(self) -> None:
        patterns = [{"pattern_type": "loop_count_variance", "step_name": "loop"}]
        session = _make_session(metadata={"patterns": patterns})
        result = _extract_pattern_dicts(session)
        assert result == patterns

    def test_empty_list_when_no_patterns_key(self) -> None:
        session = _make_session(metadata={})
        assert _extract_pattern_dicts(session) == []

    def test_empty_list_when_patterns_not_list(self) -> None:
        session = _make_session(metadata={"patterns": "bad"})
        assert _extract_pattern_dicts(session) == []

    def test_empty_list_when_patterns_empty(self) -> None:
        session = _make_session(metadata={"patterns": []})
        assert _extract_pattern_dicts(session) == []


class TestSafeRecordCost:
    def test_known_model(self) -> None:
        cost = _safe_record_cost("gpt-4o-mini", 1000, 500)
        assert cost > 0

    def test_unknown_model_returns_zero(self) -> None:
        assert _safe_record_cost("nonexistent-model-xyz", 1000, 500) == 0.0
