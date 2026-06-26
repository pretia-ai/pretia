"""Tests for baseline creation, serialization, and loading."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from pretia.ci.baseline import (
    Baseline,
    create_baseline,
    load_baseline,
    save_baseline,
)
from pretia.store import ProfilingSession


def _make_stats_metadata(
    sample_size: int = 20,
    patterns: list | None = None,
    input_mode: str = "auto-generate",
) -> dict:
    """Build metadata dict with stats, patterns, projection, and confidence."""
    if patterns is None:
        patterns = []
    return {
        "stats": {
            "total_runs": sample_size,
            "total_steps": sample_size * 2,
            "cost_per_run": {
                "mean": 0.03,
                "p50": 0.028,
                "p75": 0.035,
                "p90": 0.042,
                "p95": 0.048,
                "p99": 0.06,
                "min": 0.015,
                "max": 0.08,
                "std": 0.012,
            },
            "tokens_per_run": {
                "mean": 2000.0,
                "p50": 1800.0,
                "p75": 2400.0,
                "p90": 2800.0,
                "p95": 3200.0,
                "p99": 3800.0,
                "min": 1000.0,
                "max": 4500.0,
                "std": 600.0,
            },
            "step_stats": {
                "classify": {
                    "step_name": "classify",
                    "step_type": "llm",
                    "model": "gpt-4o-mini",
                    "call_count": sample_size,
                    "runs_present": sample_size,
                    "input_tokens": {
                        "mean": 250.0,
                        "p50": 230.0,
                        "p75": 270.0,
                        "p90": 300.0,
                        "p95": 320.0,
                        "p99": 350.0,
                        "min": 150.0,
                        "max": 400.0,
                        "std": 50.0,
                    },
                    "output_tokens": {
                        "mean": 70.0,
                        "p50": 65.0,
                        "p75": 80.0,
                        "p90": 90.0,
                        "p95": 95.0,
                        "p99": 100.0,
                        "min": 40.0,
                        "max": 120.0,
                        "std": 15.0,
                    },
                    "total_tokens": {
                        "mean": 320.0,
                        "p50": 295.0,
                        "p75": 350.0,
                        "p90": 390.0,
                        "p95": 415.0,
                        "p99": 450.0,
                        "min": 190.0,
                        "max": 520.0,
                        "std": 65.0,
                    },
                    "cost": {
                        "mean": 0.0012,
                        "p50": 0.001,
                        "p75": 0.0014,
                        "p90": 0.0016,
                        "p95": 0.0018,
                        "p99": 0.002,
                        "min": 0.0005,
                        "max": 0.003,
                        "std": 0.0004,
                    },
                    "duration_ms": {
                        "mean": 200.0,
                        "p50": 180.0,
                        "p75": 220.0,
                        "p90": 260.0,
                        "p95": 290.0,
                        "p99": 320.0,
                        "min": 100.0,
                        "max": 400.0,
                        "std": 50.0,
                    },
                    "context_size": {
                        "mean": 250.0,
                        "p50": 230.0,
                        "p75": 270.0,
                        "p90": 300.0,
                        "p95": 320.0,
                        "p99": 350.0,
                        "min": 150.0,
                        "max": 400.0,
                        "std": 50.0,
                    },
                    "iterations_per_run": {
                        "mean": 1.0,
                        "p50": 1.0,
                        "p75": 1.0,
                        "p90": 1.0,
                        "p95": 1.0,
                        "p99": 1.0,
                        "min": 1.0,
                        "max": 1.0,
                        "std": 0.0,
                    },
                    "mean_iterations": 1.0,
                },
                "generate": {
                    "step_name": "generate",
                    "step_type": "llm",
                    "model": "gpt-4o",
                    "call_count": sample_size,
                    "runs_present": sample_size,
                    "input_tokens": {
                        "mean": 1500.0,
                        "p50": 1400.0,
                        "p75": 1700.0,
                        "p90": 2000.0,
                        "p95": 2200.0,
                        "p99": 2500.0,
                        "min": 800.0,
                        "max": 3000.0,
                        "std": 400.0,
                    },
                    "output_tokens": {
                        "mean": 400.0,
                        "p50": 380.0,
                        "p75": 450.0,
                        "p90": 520.0,
                        "p95": 580.0,
                        "p99": 650.0,
                        "min": 200.0,
                        "max": 800.0,
                        "std": 100.0,
                    },
                    "total_tokens": {
                        "mean": 1900.0,
                        "p50": 1780.0,
                        "p75": 2150.0,
                        "p90": 2520.0,
                        "p95": 2780.0,
                        "p99": 3150.0,
                        "min": 1000.0,
                        "max": 3800.0,
                        "std": 500.0,
                    },
                    "cost": {
                        "mean": 0.0288,
                        "p50": 0.027,
                        "p75": 0.033,
                        "p90": 0.039,
                        "p95": 0.044,
                        "p99": 0.052,
                        "min": 0.014,
                        "max": 0.065,
                        "std": 0.01,
                    },
                    "duration_ms": {
                        "mean": 500.0,
                        "p50": 450.0,
                        "p75": 600.0,
                        "p90": 700.0,
                        "p95": 800.0,
                        "p99": 1000.0,
                        "min": 200.0,
                        "max": 1200.0,
                        "std": 200.0,
                    },
                    "context_size": {
                        "mean": 1500.0,
                        "p50": 1400.0,
                        "p75": 1700.0,
                        "p90": 2000.0,
                        "p95": 2200.0,
                        "p99": 2500.0,
                        "min": 800.0,
                        "max": 3000.0,
                        "std": 400.0,
                    },
                    "iterations_per_run": {
                        "mean": 1.0,
                        "p50": 1.0,
                        "p75": 1.0,
                        "p90": 1.0,
                        "p95": 1.0,
                        "p99": 1.0,
                        "min": 1.0,
                        "max": 1.0,
                        "std": 0.0,
                    },
                    "mean_iterations": 1.0,
                },
            },
            "run_stats": [],
        },
        "patterns": patterns,
        "projection": {
            "method": "linear",
            "traffic_volumes": [100, 1000, 10000],
            "projections": {
                "1000": {
                    "daily_volume": 1000,
                    "monthly_cost": {
                        "p50": 840.0,
                        "p75": 1050.0,
                        "p90": 1260.0,
                        "p95": 1440.0,
                        "p99": 1800.0,
                        "mean": 900.0,
                    },
                    "daily_cost": {
                        "p50": 28.0,
                        "p75": 35.0,
                        "p90": 42.0,
                        "p95": 48.0,
                        "p99": 60.0,
                        "mean": 30.0,
                    },
                    "cost_per_run": {
                        "p50": 0.028,
                        "p75": 0.035,
                        "p90": 0.042,
                        "p95": 0.048,
                        "p99": 0.06,
                        "mean": 0.03,
                    },
                },
            },
            "confidence": {
                "score": 72,
                "tier": "MODERATE",
                "display_range": "p50 – p95",
                "language": "estimated",
                "deductions": [],
                "bonuses": [],
            },
            "warnings": [],
            "patterns_detected": [],
        },
        "confidence": {
            "score": 72,
            "tier": "MODERATE",
            "display_range": "p50 – p95",
            "language": "estimated",
            "deductions": [],
            "bonuses": [],
        },
    }


def _make_session(
    sample_size: int = 20,
    patterns: list | None = None,
    input_mode: str = "auto-generate",
) -> ProfilingSession:
    return ProfilingSession(
        workflow_name="test_agent.py",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 14, 0, 0, tzinfo=UTC),
        sample_size=sample_size,
        input_mode=input_mode,
        runs=[],
        metadata=_make_stats_metadata(sample_size, patterns, input_mode),
    )


class TestCreateBaselineBasic:
    def test_creates_valid_baseline(self):
        session = _make_session()
        bl = create_baseline(session, traffic=1000)

        assert bl.version == "1.0"
        assert bl.sample_size == 20
        assert "classify" in bl.steps
        assert "generate" in bl.steps
        assert bl.steps["classify"].cost_per_run["mean"] > 0
        assert bl.steps["generate"].cost_per_run["mean"] > 0
        assert bl.total_monthly["p50"] > 0
        assert bl.total_monthly["p95"] > 0
        assert len(bl.assumptions) > 0


class TestCreateBaselineMissingStats:
    def test_raises_on_missing_stats(self):
        session = ProfilingSession(
            workflow_name="test.py",
            workflow_hash="abc",
            profiled_at=datetime(2026, 5, 25, tzinfo=UTC),
            sample_size=1,
            input_mode="single",
            runs=[],
            metadata={},
        )
        with pytest.raises(ValueError, match="no stats"):
            create_baseline(session)


class TestBaselineSerialization:
    def test_round_trip(self):
        session = _make_session()
        bl = create_baseline(session, traffic=1000)
        d = bl.to_dict()
        serialized = json.dumps(d, indent=2)
        deserialized = json.loads(serialized)
        bl2 = Baseline.from_dict(deserialized)

        assert bl2.version == bl.version
        assert bl2.workflow == bl.workflow
        assert bl2.sample_size == bl.sample_size
        assert bl2.traffic_assumption == bl.traffic_assumption
        assert set(bl2.steps.keys()) == set(bl.steps.keys())
        assert bl2.total_monthly["p50"] == bl.total_monthly["p50"]
        assert bl2.assumptions == bl.assumptions


class TestSaveAndLoadBaseline:
    def test_save_and_load(self, tmp_path):
        session = _make_session()
        bl = create_baseline(session, traffic=1000)
        saved_path = save_baseline(bl, output_dir=str(tmp_path))
        loaded = load_baseline(saved_path)

        assert loaded.version == bl.version
        assert loaded.workflow == bl.workflow
        assert loaded.sample_size == bl.sample_size
        assert loaded.total_monthly == bl.total_monthly


class TestLoadBaselineNotFound:
    def test_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Baseline not found"):
            load_baseline("nonexistent.json")


class TestBaselineAssumptionsContextGrowth:
    def test_includes_context_growth_note(self):
        patterns = [
            {
                "pattern_type": "context_growth",
                "step_name": "review",
                "severity": "danger",
                "evidence": {"r_squared": 0.92, "slope": 800},
                "description": "Context grows in review",
            }
        ]
        session = _make_session(patterns=patterns)
        bl = create_baseline(session)
        assert any("context growth" in a.lower() for a in bl.assumptions)


class TestBaselineAssumptionsSmallSample:
    def test_includes_small_sample_note(self):
        session = _make_session(sample_size=5)
        bl = create_baseline(session)
        assert any("small sample" in a.lower() for a in bl.assumptions)


class TestBaselineStepFlags:
    def test_step_gets_pattern_flags(self):
        patterns = [
            {
                "pattern_type": "context_growth",
                "step_name": "generate",
                "severity": "danger",
                "evidence": {},
                "description": "Context grows in generate",
            },
            {
                "pattern_type": "high_token_variance",
                "step_name": "generate",
                "severity": "warning",
                "evidence": {},
                "description": "High variance in generate",
            },
        ]
        session = _make_session(patterns=patterns)
        bl = create_baseline(session)
        assert "context_growth" in bl.steps["generate"].flags
        assert "high_token_variance" in bl.steps["generate"].flags
