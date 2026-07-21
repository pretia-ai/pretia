"""Tests for report features: --traffic anchoring, --unit label, --current-cost ROI."""

from __future__ import annotations

from datetime import UTC, datetime

from pretia.report.renderer import _prepare_context
from pretia.store import ProfilingSession


def _make_session(
    unit_label: str | None = None,
    current_cost: float | None = None,
) -> ProfilingSession:
    meta: dict = {
        "stats": {
            "total_runs": 10,
            "total_steps": 20,
            "cost_per_run": {
                "min": 0.005,
                "max": 0.02,
                "mean": 0.01,
                "std": 0.003,
                "p50": 0.01,
                "p75": 0.012,
                "p90": 0.015,
                "p95": 0.018,
                "p99": 0.02,
            },
        },
        "projection": {
            "method": "linear",
            "traffic_volumes": [100, 1000, 10000],
            "projections": {
                "100": {
                    "monthly_cost": {
                        "p50": 30.0,
                        "p75": 36.0,
                        "p90": 45.0,
                        "p95": 54.0,
                        "p99": 60.0,
                        "mean": 30.0,
                    },
                    "cost_per_run": {
                        "p50": 0.01,
                        "p75": 0.012,
                        "p90": 0.015,
                        "p95": 0.018,
                        "p99": 0.02,
                        "mean": 0.01,
                    },
                },
                "1000": {
                    "monthly_cost": {
                        "p50": 300.0,
                        "p75": 360.0,
                        "p90": 450.0,
                        "p95": 540.0,
                        "p99": 600.0,
                        "mean": 300.0,
                    },
                    "cost_per_run": {
                        "p50": 0.01,
                        "mean": 0.01,
                    },
                },
            },
            "confidence": {"tier": "MODERATE", "display_range": "p50-p95", "deductions": []},
        },
    }
    if unit_label:
        meta["unit_label"] = unit_label
    if current_cost is not None:
        meta["current_cost"] = current_cost
    return ProfilingSession(
        workflow_name="test_workflow",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=10,
        input_mode="auto",
        runs=[],
        metadata=meta,
    )


class TestTrafficAnchoring:
    def test_custom_traffic_sets_hero_volume(self):
        session = _make_session()
        ctx = _prepare_context(session, traffic=800)
        assert "800" in ctx["hero_projected_cost"]
        assert ctx["traffic_volumes"] == [800]

    def test_default_traffic_uses_1k(self):
        session = _make_session()
        ctx = _prepare_context(session)
        assert "1K" in ctx["hero_projected_cost"]


class TestUnitLabel:
    def test_custom_unit_in_hero(self):
        session = _make_session(unit_label="claim")
        ctx = _prepare_context(session)
        assert "claims" in ctx["hero_projected_cost"]
        assert ctx["unit_label"] == "claim"

    def test_default_unit_is_run(self):
        session = _make_session()
        ctx = _prepare_context(session)
        assert "runs" in ctx["hero_projected_cost"]
        assert ctx["unit_label"] == "run"


class TestROIBanner:
    def test_savings_banner(self):
        session = _make_session(current_cost=1000.0)
        ctx = _prepare_context(session)
        assert "Saves" in ctx["roi_banner"]
        assert "reduction" in ctx["roi_banner"]

    def test_increase_banner(self):
        session = _make_session(current_cost=100.0)
        ctx = _prepare_context(session)
        assert "Costs" in ctx["roi_banner"]
        assert "increase" in ctx["roi_banner"]

    def test_no_banner_without_current_cost(self):
        session = _make_session()
        ctx = _prepare_context(session)
        assert ctx["roi_banner"] == ""
