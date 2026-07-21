"""Tests for pretia.report — HTML report generation (charts, renderer, CLI wiring)."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from pretia.report.charts import StepCostEntry, render_cost_waterfall, render_score_ring
from pretia.report.renderer import (
    _build_step_entries,
    _extract_cost_per_run,
    _extract_projection,
    _prepare_context,
    render_and_save,
    render_html_report,
)
from pretia.store import ProfilingSession

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_session(
    runs: list | None = None,
    metadata: dict[str, Any] | None = None,
    **kwargs: object,
) -> ProfilingSession:
    defaults: dict[str, object] = {
        "workflow_name": "test_workflow",
        "workflow_hash": "abc123",
        "profiled_at": datetime(2026, 6, 10, 14, 30, 0, tzinfo=UTC),
        "sample_size": 5,
        "input_mode": "auto",
        "runs": runs or [],
        "metadata": metadata or {},
    }
    defaults.update(kwargs)
    return ProfilingSession(**defaults)


def _make_full_metadata() -> dict[str, Any]:
    """Metadata dict with stats, score, recommendations, patterns, and projection."""
    return {
        "stats": {
            "total_runs": 5,
            "total_steps": 15,
            "step_stats": {
                "classify_intent": {
                    "step_name": "classify_intent",
                    "model": "gpt-4o-mini",
                    "cost": {"mean": 0.0012, "p50": 0.0011, "p95": 0.0018},
                    "total_tokens": {"mean": 400, "p95": 600},
                    "call_count": 5,
                },
                "generate_response": {
                    "step_name": "generate_response",
                    "model": "gpt-4o",
                    "cost": {"mean": 0.045, "p50": 0.042, "p95": 0.065},
                    "total_tokens": {"mean": 2000, "p95": 3200},
                    "call_count": 5,
                },
                "summarize": {
                    "step_name": "summarize",
                    "model": "gpt-4o-mini",
                    "cost": {"mean": 0.008, "p50": 0.007, "p95": 0.012},
                    "total_tokens": {"mean": 800, "p95": 1100},
                    "call_count": 5,
                },
            },
            "cost_per_run": {
                "mean": 0.0542,
                "p50": 0.0501,
                "p75": 0.058,
                "p90": 0.068,
                "p95": 0.075,
                "p99": 0.088,
                "min": 0.035,
                "max": 0.092,
                "std": 0.015,
            },
        },
        "score": {
            "score": 62,
            "zone": "amber",
            "zone_label": "room to improve",
            "zone_color": "#DD6B20",
            "total_savings": 4200.0,
            "waste_pct": 0.38,
            "recommendation_count": 2,
            "scope_note": "Score based on model and workflow optimization.",
        },
        "recommendations": [
            {
                "id": "model-swap-classify_intent",
                "type": "model_swap",
                "title": "Swap classify_intent to Haiku",
                "description": "Classification task can use a cheaper model.",
                "monthly_savings": 3500.0,
                "confidence": "HIGH",
                "affected_steps": ["classify_intent"],
                "evidence": {"ratio": 0.15},
                "priority": 3500,
            },
            {
                "id": "cache-context-summarize",
                "type": "architecture",
                "title": "Enable caching for summarize",
                "description": "Redundant system prompts detected.",
                "monthly_savings": 700.0,
                "confidence": "MODERATE",
                "affected_steps": ["summarize"],
                "evidence": {},
                "priority": 420,
            },
        ],
        "patterns": [
            {
                "pattern_type": "context_growth",
                "step_name": "generate_response",
                "severity": "warning",
                "description": "Context growing at 120 tokens/iteration.",
            },
        ],
        "projection": {
            "method": "linear",
            "traffic_volumes": [100, 1000, 10000],
            "projections": {
                "100": {
                    "monthly_cost": {
                        "p50": 150.30,
                        "p75": 174.00,
                        "p90": 204.00,
                        "p95": 225.00,
                        "p99": 264.00,
                        "mean": 162.60,
                    },
                },
                "1000": {
                    "monthly_cost": {
                        "p50": 1503.00,
                        "p75": 1740.00,
                        "p90": 2040.00,
                        "p95": 2250.00,
                        "p99": 2640.00,
                        "mean": 1626.00,
                    },
                },
                "10000": {
                    "monthly_cost": {
                        "p50": 15030.00,
                        "p75": 17400.00,
                        "p90": 20400.00,
                        "p95": 22500.00,
                        "p99": 26400.00,
                        "mean": 16260.00,
                    },
                },
            },
            "confidence": {
                "tier": "MODERATE",
                "display_range": "p50 – p95",
            },
        },
    }


# ===========================================================================
# Charts tests
# ===========================================================================


class TestScoreRingSvg:
    def test_renders_valid_svg(self) -> None:
        svg = render_score_ring(75, "#38A169", "well optimized")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

    def test_score_zero_full_offset(self) -> None:
        svg = render_score_ring(0, "#E53E3E", "needs optimization")
        assert 'stroke-dashoffset="502.65"' in svg

    def test_score_100_zero_offset(self) -> None:
        svg = render_score_ring(100, "#38A169", "well optimized")
        assert 'stroke-dashoffset="0.00"' in svg

    def test_score_50_half_offset(self) -> None:
        svg = render_score_ring(50, "#DD6B20", "room to improve")
        assert 'stroke-dashoffset="251.33"' in svg

    def test_score_text_present(self) -> None:
        svg = render_score_ring(82, "#38A169", "well optimized")
        assert ">82</text>" in svg
        assert "of 100" in svg

    def test_zone_label_present(self) -> None:
        svg = render_score_ring(45, "#DD6B20", "room to improve")
        assert "room to improve" in svg

    def test_aria_label(self) -> None:
        svg = render_score_ring(70, "#DD6B20", "room to improve")
        assert 'aria-label="Optimization score: 70 of 100"' in svg

    def test_score_clamped_above_100(self) -> None:
        svg = render_score_ring(150, "#38A169", "well optimized")
        assert ">100</text>" in svg
        assert 'stroke-dashoffset="0.00"' in svg

    def test_score_clamped_below_0(self) -> None:
        svg = render_score_ring(-10, "#E53E3E", "needs optimization")
        assert ">0</text>" in svg

    def test_html_escaping(self) -> None:
        svg = render_score_ring(50, "#DD6B20", 'a<b>"c')
        assert "<b>" not in svg
        assert "&lt;b&gt;" in svg


class TestCostWaterfall:
    def test_renders_valid_svg(self) -> None:
        steps = [StepCostEntry("step_a", 0.05, 60.0)]
        svg = render_cost_waterfall(steps)
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")

    def test_empty_steps(self) -> None:
        svg = render_cost_waterfall([])
        assert "No step cost data" in svg
        assert "<svg" in svg

    def test_single_step(self) -> None:
        steps = [StepCostEntry("only_step", 0.10, 100.0)]
        svg = render_cost_waterfall(steps)
        assert "only_step" in svg
        assert "100%" in svg

    def test_bars_sorted_by_cost(self) -> None:
        steps = [
            StepCostEntry("cheap", 0.01, 10.0),
            StepCostEntry("expensive", 0.09, 90.0),
        ]
        svg = render_cost_waterfall(steps)
        expensive_pos = svg.index("expensive")
        cheap_pos = svg.index("cheap")
        assert expensive_pos < cheap_pos

    def test_step_names_present(self) -> None:
        steps = [
            StepCostEntry("classify", 0.05, 50.0),
            StepCostEntry("generate", 0.05, 50.0),
        ]
        svg = render_cost_waterfall(steps)
        assert "classify" in svg
        assert "generate" in svg

    def test_dollar_amounts_present(self) -> None:
        steps = [StepCostEntry("step_x", 1.50, 100.0)]
        svg = render_cost_waterfall(steps)
        assert "$1.50" in svg

    def test_html_escaping_in_names(self) -> None:
        steps = [StepCostEntry("<script>", 1.0, 100.0)]
        svg = render_cost_waterfall(steps)
        assert "<script>" not in svg
        assert "&lt;script&gt;" in svg

    def test_zero_cost_step(self) -> None:
        steps = [StepCostEntry("free_step", 0.0, 0.0)]
        svg = render_cost_waterfall(steps)
        assert "free_step" in svg


# ===========================================================================
# Renderer internals
# ===========================================================================


class TestBuildStepEntries:
    def test_with_stats(self) -> None:
        stats = {
            "step_stats": {
                "step_a": {"cost": {"mean": 0.03}},
                "step_b": {"cost": {"mean": 0.07}},
            },
        }
        entries = _build_step_entries(stats)
        assert len(entries) == 2
        assert entries[0].step_name == "step_b"
        assert entries[0].mean_cost == 0.07
        assert pytest.approx(entries[0].share_pct, abs=0.1) == 70.0

    def test_no_stats(self) -> None:
        assert _build_step_entries(None) == []

    def test_no_step_stats_key(self) -> None:
        assert _build_step_entries({"total_runs": 5}) == []

    def test_zero_total_cost(self) -> None:
        stats = {"step_stats": {"s": {"cost": {"mean": 0.0}}}}
        entries = _build_step_entries(stats)
        assert entries[0].share_pct == 0.0


class TestExtractCostPerRun:
    def test_from_stats(self) -> None:
        stats = {
            "cost_per_run": {
                "mean": 0.05,
                "p50": 0.04,
                "p75": 0.06,
                "p90": 0.07,
                "p95": 0.08,
                "p99": 0.09,
                "min": 0.03,
                "max": 0.10,
                "std": 0.02,
            },
        }
        result = _extract_cost_per_run(stats, {})
        assert result["p50"] == 0.04
        assert result["p95"] == 0.08

    def test_from_legacy_cost_summary(self) -> None:
        result = _extract_cost_per_run(None, {"mean_cost_per_run": 0.05, "p95_cost_per_run": 0.08})
        assert result["mean"] == 0.05
        assert result["p95"] == 0.08

    def test_empty_inputs(self) -> None:
        result = _extract_cost_per_run(None, {})
        assert result["mean"] == 0
        assert result["p50"] == 0


class TestExtractProjection:
    def test_with_full_projection(self) -> None:
        proj = _make_full_metadata()["projection"]
        cpr = {"mean": 0.05, "p95": 0.08}
        rows, volumes, has_cvar, _, conf, method = _extract_projection(proj, cpr)
        assert volumes == [100, 1000, 10000]
        assert rows[100]["p50"] == 150.30
        assert rows[10000]["p95"] == 22500.00
        assert has_cvar is False
        assert method == "Linear"

    def test_without_projection(self) -> None:
        cpr = {"mean": 0.05, "p95": 0.08}
        rows, volumes, has_cvar, _, conf, method = _extract_projection(None, cpr)
        assert volumes == [100, 1000, 10000]
        assert rows[100]["p50"] == pytest.approx(0.05 * 100 * 30)
        assert conf is None
        assert "estimated" in method

    def test_montecarlo_method_label(self) -> None:
        proj = {
            "method": "montecarlo",
            "traffic_volumes": [1000],
            "projections": {"1000": {"monthly_cost": {"p50": 500}}},
            "montecarlo_result": {"n_simulations": 10000, "cvar_95": 0.09},
        }
        _, _, has_cvar, cvar_values, _, method = _extract_projection(proj, {})
        assert "Monte Carlo" in method
        assert has_cvar is True
        assert 1000 in cvar_values


class TestPrepareContext:
    def test_full_context_keys(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        ctx = _prepare_context(session)
        expected_keys = {
            "workflow_name",
            "profiled_at_display",
            "total_runs",
            "total_steps",
            "score",
            "score_ring_svg",
            "cost_per_run",
            "steps",
            "waterfall_svg",
            "projection_rows",
            "traffic_volumes",
            "has_cvar",
            "cvar_values",
            "confidence",
            "projection_method",
            "patterns",
            "has_patterns",
            "recommendations",
            "has_recommendations",
            "raw_json",
        }
        assert expected_keys.issubset(ctx.keys())

    def test_has_patterns_true(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        ctx = _prepare_context(session)
        assert ctx["has_patterns"] is True

    def test_has_recommendations_true(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        ctx = _prepare_context(session)
        assert ctx["has_recommendations"] is True

    def test_missing_score(self) -> None:
        meta = _make_full_metadata()
        del meta["score"]
        session = _make_session(metadata=meta)
        ctx = _prepare_context(session)
        assert ctx["score"] is None
        assert ctx["score_ring_svg"] == ""

    def test_missing_recommendations(self) -> None:
        meta = _make_full_metadata()
        del meta["recommendations"]
        session = _make_session(metadata=meta)
        ctx = _prepare_context(session)
        assert ctx["has_recommendations"] is False

    def test_missing_patterns(self) -> None:
        meta = _make_full_metadata()
        del meta["patterns"]
        session = _make_session(metadata=meta)
        ctx = _prepare_context(session)
        assert ctx["has_patterns"] is False

    def test_steps_sorted_by_cost_desc(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        ctx = _prepare_context(session)
        costs = [s["mean_cost"] for s in ctx["steps"]]
        assert costs == sorted(costs, reverse=True)


# ===========================================================================
# Full render tests
# ===========================================================================


class TestRenderHtmlReport:
    def test_returns_html_string(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "<!DOCTYPE html>" in html
        assert "</html>" in html

    def test_contains_workflow_name(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "test_workflow" in html

    def test_contains_score_section(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "<svg" in html
        assert "62" in html
        assert "room to improve" in html

    def test_contains_projection_table(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "Monthly Cost Projection" in html
        assert "10,000 runs/day" in html

    def test_contains_recommendation_cards(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "Swap classify_intent to Haiku" in html
        assert "Enable caching for summarize" in html
        assert "Top Recommendation" in html

    def test_contains_waterfall_chart(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "Where does the money go?" in html
        assert "generate_response" in html

    def test_contains_raw_data_toggle(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "<details>" in html
        assert "<summary>" in html
        assert "Raw profile data" in html

    def test_contains_footer(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "Profiled with" in html
        assert "Pretia" in html

    def test_no_javascript(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "<script" not in html

    def test_self_contained_no_external_links(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        link_tags = re.findall(r"<link[^>]+href=[\"']https?://[^\"']+", html)
        font_links = [
            t
            for t in link_tags
            if "fonts.googleapis.com" not in t and "fonts.gstatic.com" not in t
        ]
        assert len(font_links) == 0
        style_imports = re.findall(r"@import\s+url\(", html)
        assert len(style_imports) == 0

    def test_output_size_under_100kb(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert len(html.encode("utf-8")) < 100_000

    def test_patterns_section_present(self) -> None:
        session = _make_session(metadata=_make_full_metadata())
        html = render_html_report(session)
        assert "Detected Patterns" in html
        assert "Context Growth" in html


class TestRenderHtmlEdgeCases:
    def test_zero_recommendations(self) -> None:
        meta = _make_full_metadata()
        meta["recommendations"] = []
        session = _make_session(metadata=meta)
        html = render_html_report(session)
        assert "<!DOCTYPE html>" in html
        assert "well optimized" in html.lower() or "no recommendations" in html.lower()

    def test_zero_patterns(self) -> None:
        meta = _make_full_metadata()
        meta["patterns"] = []
        session = _make_session(metadata=meta)
        html = render_html_report(session)
        assert "No cost risks detected" in html

    def test_single_run(self) -> None:
        meta = _make_full_metadata()
        meta["stats"]["total_runs"] = 1
        session = _make_session(metadata=meta, sample_size=1)
        html = render_html_report(session)
        assert "<!DOCTYPE html>" in html
        assert "1 run" in html

    def test_missing_all_optional_data(self) -> None:
        session = _make_session(metadata={})
        html = render_html_report(session)
        assert "<!DOCTYPE html>" in html
        assert "test_workflow" in html

    def test_missing_projection(self) -> None:
        meta = _make_full_metadata()
        del meta["projection"]
        session = _make_session(metadata=meta)
        html = render_html_report(session)
        assert "<!DOCTYPE html>" in html

    def test_missing_stats(self) -> None:
        meta = _make_full_metadata()
        del meta["stats"]
        session = _make_session(metadata=meta)
        html = render_html_report(session)
        assert "<!DOCTYPE html>" in html

    def test_no_score_omits_ring(self) -> None:
        meta = _make_full_metadata()
        del meta["score"]
        session = _make_session(metadata=meta)
        html = render_html_report(session)
        assert '<div class="score-hero">' not in html
        assert "Recoverable Savings" not in html

    def test_framework_and_version_in_footer(self) -> None:
        meta = _make_full_metadata()
        session = _make_session(
            metadata=meta,
            framework="langgraph",
            pretia_version="0.1.0",
            profiling_cost=1.84,
        )
        html = render_html_report(session)
        assert "langgraph" in html
        assert "v0.1.0" in html
        assert "$1.84" in html


# ===========================================================================
# render_and_save tests
# ===========================================================================


class TestRenderAndSave:
    def test_writes_file_to_disk(self, tmp_path: Path) -> None:
        session = _make_session(metadata=_make_full_metadata())
        output = tmp_path / "report.html"
        with patch("pretia.report.renderer.webbrowser.open"):
            path = render_and_save(session, output_path=output, open_browser=False)
        assert path.exists()
        content = path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_default_output_path(self, tmp_path: Path) -> None:
        session = _make_session(metadata=_make_full_metadata())
        with patch("pretia.report.renderer.webbrowser.open"):
            with patch("pretia.report.renderer.Path", wraps=Path) as mock_path:
                mock_path.side_effect = None
                path = render_and_save(
                    session,
                    output_path=tmp_path / "auto_report.html",
                    open_browser=False,
                )
        assert path.exists()
        assert path.suffix == ".html"

    def test_creates_parent_directory(self, tmp_path: Path) -> None:
        nested = tmp_path / "deep" / "nested" / "dir" / "report.html"
        session = _make_session(metadata=_make_full_metadata())
        with patch("pretia.report.renderer.webbrowser.open"):
            path = render_and_save(session, output_path=nested, open_browser=False)
        assert path.exists()

    def test_open_browser_called(self, tmp_path: Path) -> None:
        session = _make_session(metadata=_make_full_metadata())
        output = tmp_path / "report.html"
        with patch("pretia.report.renderer.webbrowser.open") as mock_open:
            render_and_save(session, output_path=output, open_browser=True)
        mock_open.assert_called_once()
        assert "file://" in mock_open.call_args[0][0]

    def test_no_browser_when_disabled(self, tmp_path: Path) -> None:
        session = _make_session(metadata=_make_full_metadata())
        output = tmp_path / "report.html"
        with patch("pretia.report.renderer.webbrowser.open") as mock_open:
            render_and_save(session, output_path=output, open_browser=False)
        mock_open.assert_not_called()

    def test_browser_error_does_not_raise(self, tmp_path: Path) -> None:
        session = _make_session(metadata=_make_full_metadata())
        output = tmp_path / "report.html"
        with patch(
            "pretia.report.renderer.webbrowser.open",
            side_effect=OSError("no browser"),
        ):
            path = render_and_save(session, output_path=output, open_browser=True)
        assert path.exists()


# ===========================================================================
# Template loading
# ===========================================================================


class TestLoadTemplate:
    def test_template_loads_successfully(self) -> None:
        from pretia.report.renderer import _load_template

        template = _load_template()
        assert template is not None

    def test_template_renders_minimal(self) -> None:
        from pretia.report.renderer import _load_template

        template = _load_template()
        html = template.render(
            workflow_name="test",
            profiled_at_display="2026-06-10",
            total_runs=1,
            total_steps=0,
            score=None,
            score_ring_svg="",
            cost_per_run=None,
            steps=[],
            waterfall_svg="",
            projection_rows=None,
            traffic_volumes=[],
            has_cvar=False,
            cvar_values={},
            confidence=None,
            projection_method="",
            patterns=[],
            has_patterns=False,
            recommendations=[],
            has_recommendations=False,
            raw_json="{}",
            input_mode="auto",
            framework=None,
            pretia_version=None,
            profiling_cost=None,
        )
        assert "<!DOCTYPE html>" in html
        assert "test" in html
