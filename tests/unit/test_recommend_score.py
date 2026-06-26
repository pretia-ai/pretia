"""Tests for pretia.recommend.score — optimization score computation."""

from __future__ import annotations

import json

import pytest

from pretia.recommend.base import Recommendation, compute_priority
from pretia.recommend.score import _classify_zone, compute_score


def _make_rec(monthly_savings: float = 100.0) -> Recommendation:
    return Recommendation(
        id=f"test-{monthly_savings}",
        type="model_swap",
        title="Test rec",
        description="Test.",
        monthly_savings=monthly_savings,
        confidence="HIGH",
        affected_steps=["step"],
        evidence={},
        priority=compute_priority(monthly_savings, "HIGH"),
    )


class TestClassifyZone:
    def test_score_0_is_red(self) -> None:
        zone, label, _ = _classify_zone(0)
        assert zone == "red"
        assert label == "needs optimization"

    def test_score_40_is_red(self) -> None:
        zone, label, _ = _classify_zone(40)
        assert zone == "red"
        assert label == "needs optimization"

    def test_score_41_is_amber(self) -> None:
        zone, label, _ = _classify_zone(41)
        assert zone == "amber"
        assert label == "room to improve"

    def test_score_70_is_amber(self) -> None:
        zone, label, _ = _classify_zone(70)
        assert zone == "amber"
        assert label == "room to improve"

    def test_score_71_is_green(self) -> None:
        zone, label, _ = _classify_zone(71)
        assert zone == "green"
        assert label == "well optimized"

    def test_score_100_is_green(self) -> None:
        zone, label, _ = _classify_zone(100)
        assert zone == "green"
        assert label == "well optimized"

    def test_zone_colors(self) -> None:
        _, _, color_red = _classify_zone(0)
        _, _, color_amber = _classify_zone(50)
        _, _, color_green = _classify_zone(100)
        assert color_red == "#E53E3E"
        assert color_amber == "#DD6B20"
        assert color_green == "#38A169"


class TestComputeScore:
    def test_no_recommendations_score_100(self) -> None:
        result = compute_score([], 1000.0)
        assert result.score == 100
        assert result.zone == "green"
        assert result.waste_pct == 0.0
        assert result.total_savings == 0.0
        assert result.recommendation_count == 0

    def test_savings_equal_cost_score_0(self) -> None:
        recs = [_make_rec(1000.0)]
        result = compute_score(recs, 1000.0)
        assert result.score == 0
        assert result.zone == "red"
        assert result.waste_pct == 1.0

    def test_savings_exceed_cost_waste_capped(self) -> None:
        recs = [_make_rec(5000.0)]
        result = compute_score(recs, 1000.0)
        assert result.score == 0
        assert result.waste_pct == 1.0

    def test_zero_projected_cost_score_100(self) -> None:
        recs = [_make_rec(500.0)]
        result = compute_score(recs, 0.0)
        assert result.score == 100
        assert result.zone == "green"
        assert result.waste_pct == 0.0

    def test_boundary_40_red(self) -> None:
        """60% waste → score 40 → red zone."""
        recs = [_make_rec(600.0)]
        result = compute_score(recs, 1000.0)
        assert result.score == 40
        assert result.zone == "red"

    def test_boundary_41_amber(self) -> None:
        """59% waste → score 41 → amber zone."""
        recs = [_make_rec(590.0)]
        result = compute_score(recs, 1000.0)
        assert result.score == 41
        assert result.zone == "amber"

    def test_boundary_70_amber(self) -> None:
        """30% waste → score 70 → amber zone."""
        recs = [_make_rec(300.0)]
        result = compute_score(recs, 1000.0)
        assert result.score == 70
        assert result.zone == "amber"

    def test_boundary_71_green(self) -> None:
        """29% waste → score 71 → green zone."""
        recs = [_make_rec(290.0)]
        result = compute_score(recs, 1000.0)
        assert result.score == 71
        assert result.zone == "green"

    def test_multiple_recommendations_summed(self) -> None:
        recs = [_make_rec(200.0), _make_rec(300.0)]
        result = compute_score(recs, 1000.0)
        assert result.total_savings == 500.0
        assert result.recommendation_count == 2
        assert result.score == 50  # 50% waste
        assert result.zone == "amber"

    def test_scope_note_present(self) -> None:
        result = compute_score([], 1000.0)
        assert "Architecture analysis" in result.scope_note

    def test_negative_projected_cost_treated_as_zero(self) -> None:
        result = compute_score([_make_rec(100.0)], -500.0)
        assert result.score == 100
        assert result.waste_pct == 0.0


class TestOptimizationScoreDataclass:
    def test_frozen(self) -> None:
        result = compute_score([], 1000.0)
        with pytest.raises(AttributeError):
            result.score = 50  # type: ignore[misc]

    def test_to_dict(self) -> None:
        result = compute_score([_make_rec(300.0)], 1000.0)
        d = result.to_dict()
        assert d["score"] == 70
        assert d["zone"] == "amber"
        assert d["zone_label"] == "room to improve"
        assert d["total_savings"] == 300.0
        assert d["recommendation_count"] == 1
        assert "scope_note" in d

    def test_to_dict_json_serializable(self) -> None:
        result = compute_score([_make_rec(500.0)], 1000.0)
        s = json.dumps(result.to_dict())
        assert isinstance(s, str)
