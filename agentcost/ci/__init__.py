"""Integrate AgentCost with CI/CD via baselines, diffs, and PR-ready reports."""

from __future__ import annotations

from agentcost.ci.baseline import (
    Baseline,
    BaselineStep,
    create_baseline,
    load_baseline,
    save_baseline,
)
from agentcost.ci.diff import DiffResult, StepDiff, diff_baseline, format_diff_report
from agentcost.ci.github import (
    ActionResult,
    check_threshold,
    format_diff_only_comment,
    format_full_profile_comment,
    format_pr_comment,
    run_diff_analysis,
)
from agentcost.ci.report import format_cli_report

__all__ = [
    "ActionResult",
    "Baseline",
    "BaselineStep",
    "DiffResult",
    "StepDiff",
    "check_threshold",
    "create_baseline",
    "diff_baseline",
    "format_cli_report",
    "format_diff_only_comment",
    "format_diff_report",
    "format_full_profile_comment",
    "format_pr_comment",
    "load_baseline",
    "run_diff_analysis",
    "save_baseline",
]
