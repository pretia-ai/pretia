"""Tests for visualization color palette and detector classification."""
from __future__ import annotations
import re
from visualization.colors import (
    WORKFLOW_GROUPS, GROUP_COLORS, COMPARISON_COLORS,
    DETECTOR_MATRIX_COLORS, EXPECTED_DETECTORS,
    workflow_color, classify_detector_result,
)


class TestWorkflowGroups:
    def test_all_14_workflows_have_group(self):
        expected = {"W1","W2","W4","W5","W9","W11","W12","W13","W14","W15","W16","W17","W18","W19"}
        assert set(WORKFLOW_GROUPS.keys()) == expected

    def test_group_colors_are_distinct(self):
        colors = list(GROUP_COLORS.values())
        assert len(colors) == len(set(colors))

    def test_workflow_color_returns_hex(self):
        for wf in WORKFLOW_GROUPS:
            color = workflow_color(wf)
            assert re.match(r"^#[0-9a-f]{6}$", color), f"{wf} got {color}"

    def test_unknown_workflow_defaults_to_linear(self):
        color = workflow_color("W999")
        assert color == GROUP_COLORS["linear"]


class TestDetectorClassification:
    def test_true_positive(self):
        assert classify_detector_result("W2", "context_growth", True) == "TP"

    def test_true_negative(self):
        assert classify_detector_result("W1", "context_growth", False) == "TN"

    def test_false_positive(self):
        assert classify_detector_result("W1", "context_growth", True) == "FP"

    def test_false_negative(self):
        assert classify_detector_result("W2", "context_growth", False) == "FN"

    def test_all_14_workflows_in_expected_matrix(self):
        assert set(EXPECTED_DETECTORS.keys()) == set(WORKFLOW_GROUPS.keys())
