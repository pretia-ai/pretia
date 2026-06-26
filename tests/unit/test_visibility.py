"""Tests for visibility warnings, recommendations, and display helpers."""

from __future__ import annotations

from types import SimpleNamespace

from pretia.ci.diff import mann_whitney_u, significance_label
from pretia.validation.visibility import (
    format_projection_output,
    get_profiling_recommendation,
    sample_coverage_statement,
)

# ---------------------------------------------------------------------------
# Change 1: Tiered profiling recommendations
# ---------------------------------------------------------------------------


class TestRecommendationLowNeff:
    def test_low_neff(self):
        rec = get_profiling_recommendation(n_eff=5, patterns=[], current_n=200)
        assert rec is not None
        assert "effective sample size" in rec.lower()
        assert "diverse inputs" in rec.lower()


class TestRecommendationHighVarianceSmallN:
    def test_high_variance_small_n(self):
        pattern = SimpleNamespace(severity="danger", pattern_type="context_growth")
        rec = get_profiling_recommendation(n_eff=25, patterns=[pattern], current_n=20)
        assert rec is not None
        assert "100" in rec


class TestRecommendationConditionalSmallN:
    def test_conditional_small_n(self):
        pattern = SimpleNamespace(severity="warning", pattern_type="step_count_variance")
        rec = get_profiling_recommendation(n_eff=15, patterns=[pattern], current_n=20)
        assert rec is not None
        assert "50" in rec


class TestRecommendationSufficientData:
    def test_sufficient(self):
        rec = get_profiling_recommendation(n_eff=80, patterns=[], current_n=100)
        assert rec is None


class TestDefaultNChangedTo50:
    def test_default_help_text(self):
        from pretia.cli import run

        for param in run.params:
            if param.name == "auto_generate":
                assert "50" in (param.help or "")
                break


# ---------------------------------------------------------------------------
# Change 7: Sample coverage statement
# ---------------------------------------------------------------------------


class TestCoverageStatementN20:
    def test_n20(self):
        stmt = sample_coverage_statement(20)
        assert "14%" in stmt


class TestCoverageStatementN100:
    def test_n100(self):
        stmt = sample_coverage_statement(100)
        assert "3%" in stmt


# ---------------------------------------------------------------------------
# Change 8: p95 suppression
# ---------------------------------------------------------------------------


class TestP95SuppressedAtN5:
    def test_suppressed(self):
        result = format_projection_output(p50=1000, p95=3000, n=5)
        assert result["display_mode"] == "p50_only"
        assert "p95" not in result
        assert "0.5" in result.get("range_note", "")


class TestP95ShownWithWarningAtN15:
    def test_warning(self):
        result = format_projection_output(p50=1000, p95=3000, n=15)
        assert result["display_mode"] == "p50_p95_warning"
        assert result["p50"] == 1000
        assert result["p95"] == 3000
        assert "warning" in result


class TestFullOutputAtN50:
    def test_full(self):
        result = format_projection_output(p50=1000, p95=3000, n=50)
        assert result["display_mode"] == "full"
        assert result["p50"] == 1000
        assert result["p95"] == 3000


# ---------------------------------------------------------------------------
# Change 9: Mann-Whitney U
# ---------------------------------------------------------------------------


class TestMannWhitneyIdenticalSamples:
    def test_identical(self):
        x = list(range(1, 21))
        y = list(range(1, 21))
        p = mann_whitney_u(x, y)
        assert p > 0.5


class TestMannWhitneyDifferentSamples:
    def test_different(self):
        x = list(range(1, 11))
        y = list(range(11, 21))
        p = mann_whitney_u(x, y)
        assert p < 0.01


class TestMannWhitneyOverlappingSamples:
    def test_overlapping(self):
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        y = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        p = mann_whitney_u(x, y)
        assert 0.01 < p < 0.50


class TestSignificanceLabelMapping:
    def test_labels(self):
        assert significance_label(0.03) == "significant"
        assert significance_label(0.07) == "possibly significant"
        assert significance_label(0.15) == "not significant"


# ---------------------------------------------------------------------------
# Change 6: Pricing staleness
# ---------------------------------------------------------------------------


class TestPricingStalenessWarning:
    def test_stale_warning(self):
        import pretia.pricing.tables as tables

        original = tables.PRICING_LAST_UPDATED
        try:
            tables.PRICING_LAST_UPDATED = "2026-01-01"
            warning = tables.check_pricing_staleness()
            assert warning is not None
            assert "days old" in warning
        finally:
            tables.PRICING_LAST_UPDATED = original


class TestPricingFreshNoWarning:
    def test_fresh(self):
        import pretia.pricing.tables as tables

        original = tables.PRICING_LAST_UPDATED
        try:
            from datetime import date

            tables.PRICING_LAST_UPDATED = date.today().isoformat()
            warning = tables.check_pricing_staleness()
            assert warning is None
        finally:
            tables.PRICING_LAST_UPDATED = original
