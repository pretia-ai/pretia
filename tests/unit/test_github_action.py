"""Tests for GitHub Action integration: PR comment generation, threshold checks, orchestration."""

from __future__ import annotations

import json

import pytest

from pretia.ci.baseline import Baseline, BaselineStep
from pretia.ci.diff import DiffResult, PatternChanges
from pretia.ci.github import (
    ActionResult,
    _score_emoji,
    check_threshold,
    format_diff_only_comment,
    format_full_profile_comment,
    format_pr_comment,
    run_diff_analysis,
)
from pretia.estimate import ModelEstimate, WorkflowEstimate
from pretia.recommend.base import Recommendation, compute_priority
from pretia.recommend.score import OptimizationScore

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_estimate(
    cost_per_run: float = 0.03,
    models: list[str] | None = None,
    steps: int | None = None,
) -> WorkflowEstimate:
    if models is None:
        models = ["gpt-4o"]
    model_objects = [
        ModelEstimate(
            model_name=m,
            canonical_name=m,
            step_name=None,
            max_tokens=None,
            input_price_per_m=2.5,
            output_price_per_m=10.0,
        )
        for m in models
    ]
    return WorkflowEstimate(
        workflow_path="agent.py",
        framework="langgraph",
        models=model_objects,
        estimated_cost_per_run=cost_per_run,
        estimated_steps=steps if steps is not None else len(model_objects) or 1,
        estimated_system_prompt_tokens=0,
    )


def _make_baseline_step(model: str = "gpt-4o") -> BaselineStep:
    return BaselineStep(
        model=model,
        tokens_input={"p50": 400, "p95": 600},
        tokens_output={"p50": 100, "p95": 200},
        cost_per_run={"p50": 0.01, "p95": 0.02, "mean": 0.012},
        iterations={"mean": 1.0, "max": 1},
        system_prompt_hash="abc123",
        system_prompt_tokens=300,
        output_format="json",
        flags=[],
        task_complexity_tier=None,
    )


def _make_baseline(
    total_monthly_p50: float = 900.0,
    step_model: str = "gpt-4o",
) -> Baseline:
    return Baseline(
        version="1.0",
        workflow="test-agent",
        profiled_at="2026-06-01T12:00:00",
        sample_size=20,
        traffic_assumption="1000/day",
        input_source="auto",
        collector_type="auto",
        confidence_tier="MODERATE",
        steps={"classify": _make_baseline_step(model=step_model)},
        total_monthly={
            "p50": total_monthly_p50,
            "p75": total_monthly_p50 * 1.2,
            "p90": total_monthly_p50 * 1.4,
            "p95": total_monthly_p50 * 1.6,
        },
        patterns=[],
        assumptions=[],
    )


def _make_score(score: int = 80, zone: str = "green") -> OptimizationScore:
    zone_labels = {
        "red": "needs optimization",
        "amber": "room to improve",
        "green": "well optimized",
    }
    zone_colors = {"red": "#E53E3E", "amber": "#DD6B20", "green": "#38A169"}
    return OptimizationScore(
        score=score,
        zone=zone,
        zone_label=zone_labels.get(zone, "well optimized"),
        zone_color=zone_colors.get(zone, "#38A169"),
        total_savings=500.0,
        waste_pct=0.15,
        recommendation_count=1,
        scope_note="Score based on model and workflow optimization.",
    )


def _make_rec(
    title: str = "Swap to Haiku",
    savings: float = 420.0,
    confidence: str = "HIGH",
) -> Recommendation:
    return Recommendation(
        id=f"test-{title.lower().replace(' ', '-')}",
        type="model_swap",
        title=title,
        description=f"Consider swapping to save {savings:.0f}/mo.",
        monthly_savings=savings,
        confidence=confidence,
        affected_steps=["classify"],
        evidence={},
        priority=compute_priority(savings, confidence),
    )


def _make_diff_result(
    delta_pct: float = 0.0,
    delta_abs: float = 0.0,
) -> DiffResult:
    return DiffResult(
        baseline_workflow="test-agent",
        baseline_date="2026-06-01T12:00:00",
        new_date="2026-06-13T12:00:00",
        total_monthly_change={"p50": delta_abs, "p95": delta_abs * 1.5},
        total_monthly_pct_change={"p50": delta_pct, "p95": delta_pct},
        step_diffs={},
        new_steps=[],
        removed_steps=[],
        model_changes=[],
        pattern_changes=PatternChanges(
            new_patterns=[], resolved_patterns=[], unchanged_patterns=[]
        ),
        exceeds_threshold=None,
        summary=f"Delta {delta_pct:.0f}%",
        traffic=1000,
    )


# ---------------------------------------------------------------------------
# Tests: check_threshold
# ---------------------------------------------------------------------------


class TestCheckThreshold:
    def test_no_threshold_never_fails(self) -> None:
        assert check_threshold(500.0, None) is False

    def test_below_threshold_passes(self) -> None:
        assert check_threshold(9.0, 10.0) is False

    def test_at_threshold_passes(self) -> None:
        assert check_threshold(10.0, 10.0) is False

    def test_above_threshold_fails(self) -> None:
        assert check_threshold(11.0, 10.0) is True

    def test_negative_delta_passes(self) -> None:
        assert check_threshold(-50.0, 10.0) is False

    def test_zero_threshold_positive_delta(self) -> None:
        assert check_threshold(0.1, 0.0) is True

    def test_zero_threshold_zero_delta(self) -> None:
        assert check_threshold(0.0, 0.0) is False


# ---------------------------------------------------------------------------
# Tests: _score_emoji
# ---------------------------------------------------------------------------


class TestScoreEmoji:
    def test_red_zone(self) -> None:
        assert _score_emoji("red") == "\U0001f534"

    def test_amber_zone(self) -> None:
        assert _score_emoji("amber") == "\U0001f7e1"

    def test_green_zone(self) -> None:
        assert _score_emoji("green") == "\U0001f7e2"

    def test_unknown_zone(self) -> None:
        assert _score_emoji("unknown") == "⚪"


# ---------------------------------------------------------------------------
# Tests: format_diff_only_comment
# ---------------------------------------------------------------------------


class TestFormatDiffOnlyComment:
    def test_hidden_marker_present(self) -> None:
        comment = format_diff_only_comment(_make_estimate(), None, 1000)
        assert "<!-- pretia-pr-comment -->" in comment

    def test_with_baseline_shows_delta(self) -> None:
        estimate = _make_estimate(cost_per_run=0.03)
        baseline = _make_baseline(total_monthly_p50=840.0)
        comment = format_diff_only_comment(estimate, baseline, 1000)
        assert "baseline" in comment.lower()
        assert "delta" in comment.lower() or "+" in comment or "-" in comment

    def test_without_baseline_shows_estimate_only(self) -> None:
        estimate = _make_estimate(cost_per_run=0.03)
        comment = format_diff_only_comment(estimate, None, 1000)
        assert "No baseline" in comment or "no baseline" in comment

    def test_without_baseline_no_delta_row(self) -> None:
        estimate = _make_estimate(cost_per_run=0.03)
        comment = format_diff_only_comment(estimate, None, 1000)
        assert "Cost delta vs baseline" not in comment

    def test_model_names_shown(self) -> None:
        estimate = _make_estimate(models=["gpt-4o-mini", "claude-haiku-3"])
        comment = format_diff_only_comment(estimate, None, 1000)
        assert "gpt-4o-mini" in comment
        assert "claude-haiku-3" in comment

    def test_zero_cost_workflow(self) -> None:
        estimate = _make_estimate(cost_per_run=0.0, models=[])
        comment = format_diff_only_comment(estimate, None, 1000)
        assert "<!-- pretia-pr-comment -->" in comment
        assert "$0.00" in comment

    def test_footer_shows_diff_mode(self) -> None:
        comment = format_diff_only_comment(_make_estimate(), None, 1000)
        assert "diff-only" in comment.lower()

    def test_footer_has_powered_by(self) -> None:
        comment = format_diff_only_comment(_make_estimate(), None, 1000)
        assert "Powered by Pretia" in comment

    def test_projected_cost_calculation(self) -> None:
        estimate = _make_estimate(cost_per_run=0.01)
        comment = format_diff_only_comment(estimate, None, 500)
        assert "$150" in comment

    def test_baseline_models_shown_in_details(self) -> None:
        estimate = _make_estimate(models=["gpt-4o-mini"])
        baseline = _make_baseline(step_model="gpt-4o")
        comment = format_diff_only_comment(estimate, baseline, 1000)
        assert "gpt-4o" in comment
        assert "gpt-4o-mini" in comment


# ---------------------------------------------------------------------------
# Tests: format_full_profile_comment
# ---------------------------------------------------------------------------


class TestFormatFullProfileComment:
    def test_hidden_marker_present(self) -> None:
        score = _make_score(72, "amber")
        comment = format_full_profile_comment(score, 840.0, [], None, None)
        assert "<!-- pretia-pr-comment -->" in comment

    def test_includes_score(self) -> None:
        score = _make_score(72, "amber")
        comment = format_full_profile_comment(score, 840.0, [], None, None)
        assert "72/100" in comment

    def test_includes_zone_label(self) -> None:
        score = _make_score(72, "amber")
        comment = format_full_profile_comment(score, 840.0, [], None, None)
        assert "room to improve" in comment

    def test_includes_emoji(self) -> None:
        score = _make_score(30, "red")
        comment = format_full_profile_comment(score, 1000.0, [], None, None)
        assert "\U0001f534" in comment

    def test_includes_recommendations_in_details(self) -> None:
        recs = [_make_rec("Swap to Haiku", 420.0)]
        score = _make_score(60, "amber")
        comment = format_full_profile_comment(score, 1000.0, recs, None, None)
        assert "Swap to Haiku" in comment
        assert "<details>" in comment
        assert "<summary>" in comment

    def test_no_recommendations_no_details_block(self) -> None:
        score = _make_score(95, "green")
        comment = format_full_profile_comment(score, 840.0, [], None, None)
        assert "<details>" not in comment

    def test_report_url_shown(self) -> None:
        score = _make_score(80, "green")
        comment = format_full_profile_comment(score, 840.0, [], None, "https://example.com/report")
        assert "https://example.com/report" in comment
        assert "View full report" in comment

    def test_no_report_url_graceful(self) -> None:
        score = _make_score(80, "green")
        comment = format_full_profile_comment(score, 840.0, [], None, None)
        assert "View full report" not in comment

    def test_baseline_diff_shows_delta(self) -> None:
        score = _make_score(80, "green")
        diff = _make_diff_result(delta_pct=15.0, delta_abs=120.0)
        comment = format_full_profile_comment(score, 960.0, [], diff, None)
        assert "+15%" in comment or "+$120" in comment

    def test_footer_shows_full_profile_mode(self) -> None:
        score = _make_score(80, "green")
        comment = format_full_profile_comment(score, 840.0, [], None, None)
        assert "full profile" in comment.lower()

    def test_potential_savings_shown(self) -> None:
        score = _make_score(60, "amber")
        comment = format_full_profile_comment(score, 1000.0, [], None, None)
        assert "savings" in comment.lower()

    def test_multiple_recommendations_numbered(self) -> None:
        recs = [_make_rec("Rec A", 500.0), _make_rec("Rec B", 200.0)]
        score = _make_score(50, "amber")
        comment = format_full_profile_comment(score, 2000.0, recs, None, None)
        assert "### 1." in comment
        assert "### 2." in comment


# ---------------------------------------------------------------------------
# Tests: format_pr_comment
# ---------------------------------------------------------------------------


class TestFormatPrComment:
    def test_dispatches_to_diff_mode(self) -> None:
        estimate = _make_estimate()
        comment = format_pr_comment("diff", estimate=estimate, baseline=None, daily_volume=1000)
        assert "diff-only" in comment.lower()

    def test_dispatches_to_profile_mode(self) -> None:
        score = _make_score(80, "green")
        comment = format_pr_comment(
            "profile",
            estimate=None,
            baseline=None,
            daily_volume=1000,
            score=score,
            recommendations=[],
        )
        assert "full profile" in comment.lower()

    def test_diff_mode_requires_estimate(self) -> None:
        with pytest.raises(ValueError):
            format_pr_comment("diff", estimate=None, daily_volume=1000)

    def test_profile_mode_requires_score(self) -> None:
        with pytest.raises(ValueError):
            format_pr_comment("profile", estimate=None, daily_volume=1000)


# ---------------------------------------------------------------------------
# Tests: ActionResult
# ---------------------------------------------------------------------------


class TestActionResult:
    def test_to_dict_serializable(self) -> None:
        result = ActionResult(
            score=80,
            projected_cost=840.0,
            cost_delta=0.0,
            delta_pct=0.0,
            rec_count=0,
            report_path="",
            comment_markdown="test",
            threshold_exceeded=False,
        )
        d = result.to_dict()
        s = json.dumps(d)
        assert isinstance(s, str)

    def test_to_dict_roundtrip(self) -> None:
        result = ActionResult(
            score=72,
            projected_cost=1200.50,
            cost_delta=-100.0,
            delta_pct=-8.33,
            rec_count=3,
            report_path="/tmp/report.html",
            comment_markdown="## Test",
            threshold_exceeded=True,
        )
        d = result.to_dict()
        assert d["score"] == 72
        assert d["projected_cost"] == 1200.50
        assert d["cost_delta"] == -100.0
        assert d["threshold_exceeded"] is True

    def test_threshold_exceeded_flag(self) -> None:
        result = ActionResult(
            score=80,
            projected_cost=840.0,
            cost_delta=100.0,
            delta_pct=12.0,
            rec_count=0,
            report_path="",
            comment_markdown="test",
            threshold_exceeded=True,
        )
        assert result.threshold_exceeded is True


# ---------------------------------------------------------------------------
# Tests: run_diff_analysis
# ---------------------------------------------------------------------------


class TestRunDiffAnalysis:
    def test_returns_action_result(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        result = run_diff_analysis(str(wf), str(tmp_path / "baseline.json"), 1000, None)
        assert isinstance(result, ActionResult)
        assert result.comment_markdown
        assert result.threshold_exceeded is False

    def test_missing_baseline_graceful(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        result = run_diff_analysis(str(wf), str(tmp_path / "nonexistent.json"), 1000, None)
        assert isinstance(result, ActionResult)
        assert result.cost_delta == 0.0
        assert result.delta_pct == 0.0

    def test_with_baseline(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        baseline = _make_baseline(total_monthly_p50=500.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(baseline.to_dict()))
        result = run_diff_analysis(str(wf), str(bl_path), 1000, None)
        assert isinstance(result, ActionResult)

    def test_threshold_exceeded(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        baseline = _make_baseline(total_monthly_p50=0.01)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(baseline.to_dict()))
        result = run_diff_analysis(str(wf), str(bl_path), 1000, cost_threshold=5.0)
        assert result.threshold_exceeded is True

    def test_threshold_not_exceeded(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        baseline = _make_baseline(total_monthly_p50=999999.0)
        bl_path = tmp_path / "baseline.json"
        bl_path.write_text(json.dumps(baseline.to_dict()))
        result = run_diff_analysis(str(wf), str(bl_path), 1000, cost_threshold=5.0)
        assert result.threshold_exceeded is False

    def test_score_is_zero_for_diff_mode(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        result = run_diff_analysis(str(wf), str(tmp_path / "baseline.json"), 1000, None)
        assert result.score == 0
        assert result.rec_count == 0

    def test_comment_has_marker(self, tmp_path) -> None:
        wf = tmp_path / "agent.py"
        wf.write_text('run(model="gpt-4o")\n')
        result = run_diff_analysis(str(wf), str(tmp_path / "baseline.json"), 1000, None)
        assert "<!-- pretia-pr-comment -->" in result.comment_markdown


# ---------------------------------------------------------------------------
# Tests: comment marker consistency
# ---------------------------------------------------------------------------


class TestCommentMarker:
    def test_diff_mode_has_marker(self) -> None:
        comment = format_diff_only_comment(_make_estimate(), None, 1000)
        assert "<!-- pretia-pr-comment -->" in comment

    def test_full_mode_has_marker(self) -> None:
        comment = format_full_profile_comment(_make_score(80, "green"), 840.0, [], None, None)
        assert "<!-- pretia-pr-comment -->" in comment

    def test_marker_is_first_line(self) -> None:
        comment = format_diff_only_comment(_make_estimate(), None, 1000)
        assert comment.startswith("<!-- pretia-pr-comment -->")

    def test_full_mode_marker_is_first_line(self) -> None:
        comment = format_full_profile_comment(_make_score(80, "green"), 840.0, [], None, None)
        assert comment.startswith("<!-- pretia-pr-comment -->")
