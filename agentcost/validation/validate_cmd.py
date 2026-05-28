"""Logic for 'agentcost validate' command: 20-vs-100 comparison test."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agentcost.validation.scoring import CalibrationScore

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ValidateResult:
    """Result of a validation run comparing small-n to large-n profiling."""

    workflow: str
    small_n: int
    large_n: int
    score: CalibrationScore
    convergence_pct: float
    recommendation: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow": self.workflow,
            "small_n": self.small_n,
            "large_n": self.large_n,
            "score": self.score.to_dict(),
            "convergence_pct": self.convergence_pct,
            "recommendation": self.recommendation,
        }


def run_validation(
    workflow_path: str,
    budget: float = 10.0,
    small_n: int = 20,
    large_n: int = 100,
) -> ValidateResult:
    """Profile a workflow at two sample sizes and compare."""
    from agentcost.projection.stats import compute_stats
    from agentcost.runner import ProfileRunner

    small_runner = ProfileRunner(
        workflow_path=workflow_path,
        auto_generate=small_n,
    )
    small_session = small_runner.run_sync()
    small_stats = compute_stats(small_session.runs)

    large_runner = ProfileRunner(
        workflow_path=workflow_path,
        auto_generate=large_n,
    )
    large_session = large_runner.run_sync()
    large_stats = compute_stats(large_session.runs)

    from agentcost.validation.scoring import score_projection

    score = score_projection(small_stats, large_stats)

    convergence_pct = 0.0
    if (
        small_stats.cost_per_run is not None
        and large_stats.cost_per_run is not None
        and large_stats.cost_per_run.p50 > 0
    ):
        convergence_pct = (
            abs(small_stats.cost_per_run.p50 - large_stats.cost_per_run.p50)
            / large_stats.cost_per_run.p50
            * 100
        )

    if convergence_pct <= 30:
        recommendation = f"{small_n} samples is sufficient for this workflow."
    else:
        next_tier = 50 if small_n <= 20 else 200
        recommendation = (
            f"Warning: {small_n}-sample projection differs by "
            f"{convergence_pct:.0f}% from {large_n}-sample. "
            f"This workflow has high variance. Use {next_tier}+ samples."
        )

    return ValidateResult(
        workflow=workflow_path,
        small_n=small_n,
        large_n=large_n,
        score=score,
        convergence_pct=convergence_pct,
        recommendation=recommendation,
    )


def format_validate_report(result: ValidateResult) -> str:
    """Format a human-readable validation report."""
    lines: list[str] = []
    lines.append(f"AgentCost Validation: {result.workflow}")
    lines.append("=" * 40)
    lines.append("")
    lines.append(
        f"Profiled with {result.small_n} samples, validated against {result.large_n} samples."
    )
    lines.append("")

    sc = result.score
    header = f"{'Metric':<14} {'Status':>12}"
    lines.append(header)
    lines.append("-" * 30)

    def _check(ok: bool) -> str:
        return "✓" if ok else "✗"

    p50_ok = 0.5 <= sc.p50_ratio <= 2.0
    lines.append(f"{'p50 ratio':<14} {sc.p50_ratio:.2f}x {_check(p50_ok):>5}")

    p95_ok = sc.p95_coverage >= 0.80
    lines.append(f"{'p95 coverage':<14} {sc.p95_coverage:.0%} {_check(p95_ok):>7}")

    range_ok = sc.range_ratio < 5.0
    lines.append(f"{'Range ratio':<14} {sc.range_ratio:.1f}x {_check(range_ok):>5}")

    lines.append(f"{'Top step':<14} {_check(sc.top_step_correct):>12}")

    rank_ok = sc.ranking_correlation > 0.8
    lines.append(f"{'Rank corr.':<14} {sc.ranking_correlation:.2f} {_check(rank_ok):>6}")

    lines.append("")
    lines.append(
        f"Convergence: {result.convergence_pct:.1f}% difference "
        f"between {result.small_n} and {result.large_n} samples."
    )

    verdict_icon = "✅" if sc.verdict in ("PASS", "WARN") else "❌"
    lines.append(f"Verdict: {verdict_icon} {result.recommendation}")

    return "\n".join(lines)
