"""Tests for W13 config, inputs, cut workflows, and skewed input sets."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

INPUTS_DIR = Path(__file__).parent.parent / "backtesting" / "inputs"


# ---------------------------------------------------------------------------
# W13 config tests
# ---------------------------------------------------------------------------


class TestW13InActiveWorkflows:
    def test_w13_active(self):
        from tests.backtesting.configs import BACKTESTING_CONFIGS

        names = [c.name for c in BACKTESTING_CONFIGS]
        assert any("W13" in n for n in names)


class TestW13Has4Steps:
    def test_4_steps(self):
        from tests.backtesting.configs import BACKTESTING_CONFIGS

        w13 = next(c for c in BACKTESTING_CONFIGS if "W13" in c.name)
        desc = w13.description.lower()
        assert "classify" in desc
        assert "respond_simple" in desc
        assert "research_and_respond" in desc
        assert "escalate_review" in desc


class TestW13ExpectedPatterns:
    def test_expected_patterns(self):
        from tests.backtesting.configs import BACKTESTING_CONFIGS

        w13 = next(c for c in BACKTESTING_CONFIGS if "W13" in c.name)
        desc = w13.description.lower()
        assert "step count variance" in desc or "bimodal" in desc


# ---------------------------------------------------------------------------
# W13 input tests
# ---------------------------------------------------------------------------


class TestW13InputCount:
    def test_count(self):
        path = INPUTS_DIR / "w13_inputs.jsonl"
        assert path.exists(), f"Missing: {path}"
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 500
        for line in lines:
            json.loads(line)


class TestW13InputDistribution:
    def test_distribution(self):
        path = INPUTS_DIR / "w13_inputs.jsonl"
        with open(path) as f:
            items = [json.loads(line) for line in f if line.strip()]
        counts = Counter(item.get("difficulty", "unknown") for item in items)
        total = len(items)
        assert abs(counts.get("easy", 0) / total - 0.35) < 0.06
        assert abs(counts.get("medium", 0) / total - 0.30) < 0.06
        assert abs(counts.get("hard", 0) / total - 0.20) < 0.06
        assert abs(counts.get("edge", 0) / total - 0.10) < 0.04
        assert abs(counts.get("adversarial", 0) / total - 0.05) < 0.03


class TestW13InputsUnique:
    def test_unique(self):
        path = INPUTS_DIR / "w13_inputs.jsonl"
        with open(path) as f:
            items = [json.loads(line) for line in f if line.strip()]
        texts = [item.get("input", "") for item in items]
        assert len(set(texts)) == len(texts), "Duplicate input texts found"


class TestW13BranchCoverage:
    def test_branch_coverage(self):
        path = INPUTS_DIR / "w13_inputs.jsonl"
        with open(path) as f:
            items = [json.loads(line) for line in f if line.strip()]
        routes = Counter(item.get("expected_route", "UNKNOWN") for item in items)
        assert routes.get("SIMPLE", 0) >= 2
        assert routes.get("MODERATE", 0) >= 2
        assert routes.get("COMPLEX", 0) >= 2


# ---------------------------------------------------------------------------
# Cut workflow tests
# ---------------------------------------------------------------------------


class TestW3W6W7Excluded:
    def test_excluded(self):
        from tests.backtesting.configs import BACKTESTING_CONFIGS

        names = {c.name for c in BACKTESTING_CONFIGS}
        assert "W3-codereview-simple" not in names
        assert "W6-extraction-complex" not in names
        assert "W7-research-simple" not in names


class TestActiveWorkflowCount:
    def test_count(self):
        from tests.backtesting.configs import BACKTESTING_CONFIGS

        assert len(BACKTESTING_CONFIGS) == 13


# ---------------------------------------------------------------------------
# Skewed input tests
# ---------------------------------------------------------------------------


class TestW2SkewedInputCount:
    def test_count(self):
        path = INPUTS_DIR / "w02_skewed.jsonl"
        assert path.exists()
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 500
        for line in lines:
            json.loads(line)


class TestW8SkewedExists:
    def test_exists(self):
        path = INPUTS_DIR / "w08_skewed.jsonl"
        assert path.exists()
        with open(path) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert len(lines) == 500
