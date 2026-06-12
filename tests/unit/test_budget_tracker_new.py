"""Tests for BudgetTracker accumulation, gates, and serialization."""

from __future__ import annotations

import json

from tests.backtesting.budget_tracker import BudgetTracker


class TestRecord:
    """record() updates accumulators correctly."""

    def test_record_updates_totals(self) -> None:
        """record() increments spent, per_workflow, and per_comparison."""
        tracker = BudgetTracker(limit=10.0)
        tracker.record("W1-support-simple", "comp-a", 1.50)
        tracker.record("W1-support-simple", "comp-b", 0.75)
        tracker.record("W9-sales-openai", "comp-a", 2.00)

        assert tracker.spent == 4.25
        assert tracker.per_workflow == {
            "W1-support-simple": 2.25,
            "W9-sales-openai": 2.00,
        }
        assert tracker.per_comparison == {"comp-a": 3.50, "comp-b": 0.75}

    def test_record_appends_log(self) -> None:
        """Each record() call appends one entry to the log list."""
        tracker = BudgetTracker(limit=10.0)

        assert len(tracker.log) == 0
        tracker.record("W1-support-simple", "comp-a", 0.10)
        assert len(tracker.log) == 1
        tracker.record("W9-sales-openai", "comp-b", 0.20)
        assert len(tracker.log) == 2

        entry = tracker.log[0]
        assert entry["workflow"] == "W1-support-simple"
        assert entry["comparison"] == "comp-a"
        assert entry["cost"] == 0.10
        assert "cumulative" in entry
        assert "timestamp" in entry


class TestCheckLimit:
    """check_limit() compares spent against limit."""

    def test_check_limit_false(self) -> None:
        """spent < limit returns False."""
        tracker = BudgetTracker(limit=5.0)
        tracker.record("W1-support-simple", "comp-a", 1.0)

        assert tracker.check_limit() is False

    def test_check_limit_true(self) -> None:
        """spent >= limit returns True."""
        tracker = BudgetTracker(limit=2.0)
        tracker.record("W1-support-simple", "comp-a", 2.0)

        assert tracker.check_limit() is True


class TestComparisonAGate:
    """check_comparison_a_gate() detects systemic failures."""

    def test_check_comparison_a_gate_pass(self) -> None:
        """Fewer than 3 failures returns (False, message)."""
        tracker = BudgetTracker(limit=10.0)
        scores = {
            "W1-support-simple": {"passed": True},
            "W9-sales-openai": {"passed": False},
            "W11-support-qwen": {"passed": True},
            "W12-extraction-deepseek": {"passed": True},
        }

        should_stop, msg = tracker.check_comparison_a_gate(scores)

        assert should_stop is False
        assert "1 failure(s)" in msg

    def test_check_comparison_a_gate_stop(self) -> None:
        """5 or more failures returns (True, message)."""
        tracker = BudgetTracker(limit=10.0)
        scores = {
            "W1-support-simple": {"passed": False},
            "W2-support-complex": {"passed": False},
            "W9-sales-openai": {"passed": False},
            "W11-support-qwen": {"passed": False},
            "W12-extraction-deepseek": {"passed": False},
            "W13-routing-agent": {"passed": True},
        }

        should_stop, msg = tracker.check_comparison_a_gate(scores)

        assert should_stop is True
        assert "5 workflows failed" in msg
        assert "Systemic engine problem" in msg


class TestComparisonBCheapGate:
    """check_comparison_b_cheap_gate() validates cheap/linear workflows."""

    def test_check_comparison_b_cheap_gate_pass(self) -> None:
        """All cheap workflows pass returns (False, message)."""
        tracker = BudgetTracker(limit=10.0)
        scores = {
            "W1-support-simple": {"passed": True},
            "W9-sales-openai": {"passed": True},
            "W11-support-qwen": {"passed": True},
            "W12-extraction-deepseek": {"passed": True},
        }

        should_stop, msg = tracker.check_comparison_b_cheap_gate(scores)

        assert should_stop is False
        assert "all" in msg.lower()
        assert "passed" in msg

    def test_check_comparison_b_cheap_gate_stop(self) -> None:
        """A cheap workflow failing returns (True, message)."""
        tracker = BudgetTracker(limit=10.0)
        scores = {
            "W1-support-simple": {"passed": False},
            "W9-sales-openai": {"passed": True},
        }

        should_stop, msg = tracker.check_comparison_b_cheap_gate(scores)

        assert should_stop is True
        assert "W1-support-simple" in msg
        assert "halting" in msg.lower()


class TestSummaryAndSerialization:
    """summary() and to_dict() produce correct outputs."""

    def test_summary_fields(self) -> None:
        """summary() returns dict with expected keys."""
        tracker = BudgetTracker(limit=10.0)
        tracker.record("W1-support-simple", "comp-a", 2.0)

        result = tracker.summary()

        expected_keys = {
            "total_spent",
            "per_workflow",
            "per_comparison",
            "limit",
            "remaining",
            "log_entries",
        }
        assert set(result.keys()) == expected_keys
        assert result["total_spent"] == 2.0
        assert result["remaining"] == 8.0
        assert result["log_entries"] == 1

    def test_to_dict_serializable(self) -> None:
        """to_dict() returns a JSON-serializable dict."""
        tracker = BudgetTracker(limit=5.0)
        tracker.record("W1-support-simple", "comp-a", 1.0)
        tracker.record("W9-sales-openai", "comp-b", 0.5)

        result = tracker.to_dict()

        # Must not raise
        serialized = json.dumps(result)
        roundtripped = json.loads(serialized)

        assert roundtripped["limit"] == 5.0
        assert roundtripped["spent"] == 1.5
        assert len(roundtripped["log"]) == 2
