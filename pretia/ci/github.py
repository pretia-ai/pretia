"""GitHub Action integration: PR comment generation, threshold checking, CI orchestration."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pretia.ci.baseline import Baseline, load_baseline
from pretia.ci.report import format_cost
from pretia.estimate import WorkflowEstimate, estimate_workflow
from pretia.recommend.base import Recommendation
from pretia.recommend.score import OptimizationScore

if TYPE_CHECKING:
    from pretia.ci.diff import DiffResult

logger = logging.getLogger(__name__)

_COMMENT_MARKER = "<!-- pretia-pr-comment -->"

_ZONE_EMOJIS: dict[str, str] = {
    "red": "\U0001f534",
    "amber": "\U0001f7e1",
    "green": "\U0001f7e2",
}
_DEFAULT_EMOJI = "⚪"


@dataclass(frozen=True, slots=True)
class ActionResult:
    """Result of a GitHub Action analysis run."""

    score: int
    projected_cost: float
    cost_delta: float
    delta_pct: float
    rec_count: int
    report_path: str
    comment_markdown: str
    threshold_exceeded: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "score": self.score,
            "projected_cost": self.projected_cost,
            "cost_delta": self.cost_delta,
            "delta_pct": self.delta_pct,
            "rec_count": self.rec_count,
            "report_path": self.report_path,
            "comment_markdown": self.comment_markdown,
            "threshold_exceeded": self.threshold_exceeded,
        }


def check_threshold(cost_delta_pct: float, threshold: float | None) -> bool:
    """Return True if the cost increase percentage exceeds the configured threshold."""
    if threshold is None:
        return False
    return cost_delta_pct > threshold


def _score_emoji(zone: str) -> str:
    """Map optimization score zone to a color-coded emoji."""
    return _ZONE_EMOJIS.get(zone, _DEFAULT_EMOJI)


def _pct_change(old: float, new: float) -> float:
    """Compute percentage change from old to new."""
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return ((new - old) / abs(old)) * 100


def _format_delta(delta: float, delta_pct: float) -> str:
    """Format a cost delta with percentage for display in a table."""
    if delta == 0 and delta_pct == 0:
        return "—"
    sign = "+" if delta >= 0 else ""
    return f"{sign}{format_cost(delta)} ({sign}{delta_pct:.0f}%)"


def format_diff_only_comment(
    estimate: WorkflowEstimate,
    baseline: Baseline | None,
    daily_volume: int,
) -> str:
    """Generate a concise PR comment for diff-only mode (no LLM calls)."""
    projected = estimate.estimated_cost_per_run * daily_volume * 30
    model_names = [m.model_name for m in estimate.models]

    delta = 0.0
    delta_pct = 0.0
    has_baseline = baseline is not None

    if has_baseline:
        baseline_p50 = baseline.total_monthly.get("p50", 0)
        delta = projected - baseline_p50
        delta_pct = _pct_change(baseline_p50, projected)

    lines: list[str] = [_COMMENT_MARKER]
    lines.append(f"## {_DEFAULT_EMOJI} Pretia — Cost Estimate")
    lines.append("")

    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Projected monthly cost | {format_cost(projected)} |")
    if has_baseline:
        lines.append(f"| Cost delta vs baseline | {_format_delta(delta, delta_pct)} |")
    lines.append(f"| Models detected | {len(model_names)} |")
    lines.append(f"| Steps detected | {estimate.estimated_steps} |")
    lines.append("")

    detail_lines: list[str] = []
    if has_baseline:
        baseline_models = {name: step.model for name, step in baseline.steps.items()}
        if baseline_models or model_names:
            detail_lines.append("**Models in current code:**")
            if model_names:
                for m in model_names:
                    detail_lines.append(f"- `{m}`")
            else:
                detail_lines.append("- None detected")
            detail_lines.append("")
            detail_lines.append("**Models in baseline:**")
            for step_name, model in sorted(baseline_models.items()):
                detail_lines.append(f"- `{step_name}`: `{model}`")
    else:
        if model_names:
            detail_lines.append("**Models detected:**")
            for m in model_names:
                detail_lines.append(f"- `{m}`")
        detail_lines.append("")
        detail_lines.append(
            "*No baseline found. Run `pretia baseline update` to enable delta tracking.*"
        )

    if detail_lines:
        lines.append("<details>")
        lines.append("<summary>Details</summary>")
        lines.append("")
        lines.extend(detail_lines)
        lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    lines.append(
        "<sub>Mode: diff-only (static analysis, no LLM calls) | "
        "[Powered by Pretia](https://github.com/pretia/pretia)</sub>"
    )

    return "\n".join(lines)


def format_full_profile_comment(
    score: OptimizationScore,
    projected_cost: float,
    recommendations: list[Recommendation],
    baseline_diff: DiffResult | None,
    report_url: str | None,
) -> str:
    """Generate a detailed PR comment for full profile mode."""
    emoji = _score_emoji(score.zone)

    lines: list[str] = [_COMMENT_MARKER]
    lines.append(f"## {emoji} Pretia — {score.score}/100 ({score.zone_label})")
    lines.append("")

    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Projected monthly cost (p50) | {format_cost(projected_cost)} |")
    lines.append(f"| Optimization score | {score.score}/100 |")

    if baseline_diff is not None:
        p50_delta = baseline_diff.total_monthly_change.get("p50", 0)
        p50_pct = baseline_diff.total_monthly_pct_change.get("p50", 0)
        lines.append(f"| Cost delta vs baseline | {_format_delta(p50_delta, p50_pct)} |")

    if score.total_savings > 0:
        lines.append(f"| Potential savings | {format_cost(score.total_savings)}/mo |")

    lines.append("")

    if recommendations:
        lines.append("<details>")
        lines.append(f"<summary>Recommendations ({len(recommendations)})</summary>")
        lines.append("")
        for i, rec in enumerate(recommendations, 1):
            lines.append(
                f"### {i}. {rec.title} — "
                f"Saves {format_cost(rec.monthly_savings)}/mo ({rec.confidence})"
            )
            lines.append(f"{rec.description}")
            lines.append("")
        lines.append("</details>")
        lines.append("")

    lines.append("---")
    footer_parts = ["Mode: full profile"]
    if report_url:
        footer_parts.append(f"[View full report]({report_url})")
    footer_parts.append("[Powered by Pretia](https://github.com/pretia/pretia)")
    lines.append(f"<sub>{' | '.join(footer_parts)}</sub>")

    return "\n".join(lines)


def format_pr_comment(
    mode: str,
    *,
    estimate: WorkflowEstimate | None = None,
    baseline: Baseline | None = None,
    daily_volume: int = 1000,
    score: OptimizationScore | None = None,
    recommendations: list[Recommendation] | None = None,
    baseline_diff: DiffResult | None = None,
    report_url: str | None = None,
) -> str:
    """Generate the full PR comment markdown, dispatching to mode-specific formatters."""
    if mode == "diff":
        if estimate is None:
            raise ValueError("estimate is required for diff mode")
        return format_diff_only_comment(estimate, baseline, daily_volume)

    if score is None:
        raise ValueError("score is required for profile mode")
    return format_full_profile_comment(
        score=score,
        projected_cost=_extract_projected_cost(score, estimate, baseline, daily_volume),
        recommendations=recommendations or [],
        baseline_diff=baseline_diff,
        report_url=report_url,
    )


def _extract_projected_cost(
    score: OptimizationScore | None,
    estimate: WorkflowEstimate | None,
    baseline: Baseline | None,
    daily_volume: int,
) -> float:
    """Best-effort projected monthly cost from available data."""
    if estimate is not None:
        return estimate.estimated_cost_per_run * daily_volume * 30
    if baseline is not None:
        return baseline.total_monthly.get("p50", 0)
    return 0.0


def run_diff_analysis(
    workflow_path: str,
    baseline_path: str,
    daily_volume: int,
    cost_threshold: float | None,
) -> ActionResult:
    """Execute diff-only analysis: static analysis + baseline comparison.

    Zero-cost path — no LLM calls, no profiling.
    """
    estimate = estimate_workflow(workflow_path)
    projected = estimate.estimated_cost_per_run * daily_volume * 30

    baseline: Baseline | None = None
    delta = 0.0
    delta_pct = 0.0

    try:
        baseline = load_baseline(baseline_path)
        baseline_p50 = baseline.total_monthly.get("p50", 0)
        delta = projected - baseline_p50
        delta_pct = _pct_change(baseline_p50, projected)
    except FileNotFoundError:
        logger.info("No baseline found at %s — skipping delta calculation.", baseline_path)

    comment = format_diff_only_comment(estimate, baseline, daily_volume)
    exceeded = check_threshold(delta_pct, cost_threshold)

    return ActionResult(
        score=0,
        projected_cost=round(projected, 2),
        cost_delta=round(delta, 2),
        delta_pct=round(delta_pct, 2),
        rec_count=0,
        report_path="",
        comment_markdown=comment,
        threshold_exceeded=exceeded,
    )


def run_full_profile(
    workflow_path: str,
    baseline_path: str,
    framework: str,
    daily_volume: int,
    cost_threshold: float | None,
) -> ActionResult:
    """Execute full profiling pipeline (~5 min, ~$2).

    Runs the actual agent workflow, collects StepRecords, generates
    recommendations, computes score, and produces HTML report.
    """
    from pretia.ci.diff import diff_baseline
    from pretia.recommend import compute_score, generate_recommendations
    from pretia.report.renderer import render_and_save
    from pretia.runner import ProfileRunner

    collector = framework if framework != "auto" else None
    runner = ProfileRunner(
        workflow_path=workflow_path,
        collector_name=collector,
    )
    session = runner.run_sync()

    recs = generate_recommendations(session)
    session.metadata["recommendations"] = [r.to_dict() for r in recs]

    projected_cost = 0.0
    projection = session.metadata.get("projection", {})
    projs = projection.get("projections", {})
    for vol_data in projs.values():
        monthly = vol_data.get("monthly_cost", {})
        p50 = monthly.get("p50", 0)
        if p50 > projected_cost:
            projected_cost = p50

    if projected_cost == 0.0:
        cost_summary = session.metadata.get("cost_summary", {})
        mean_cost = cost_summary.get("mean_cost_per_run", 0)
        projected_cost = mean_cost * daily_volume * 30

    score = compute_score(recs, projected_cost)
    session.metadata["score"] = score.to_dict()

    report_path = render_and_save(session, open_browser=False)

    baseline_diff = None
    delta = 0.0
    delta_pct = 0.0
    try:
        baseline = load_baseline(baseline_path)
        baseline_diff = diff_baseline(baseline, session, daily_volume)
        delta = baseline_diff.total_monthly_change.get("p50", 0)
        delta_pct = baseline_diff.total_monthly_pct_change.get("p50", 0)
    except FileNotFoundError:
        logger.info("No baseline found at %s — skipping delta.", baseline_path)
    except ValueError as exc:
        logger.warning("Could not diff against baseline: %s", exc)

    rec_objects = (
        [
            Recommendation(**{k: v for k, v in r.items()})
            for r in session.metadata.get("recommendations", [])
        ]
        if not recs
        else recs
    )

    comment = format_full_profile_comment(
        score=score,
        projected_cost=projected_cost,
        recommendations=rec_objects,
        baseline_diff=baseline_diff,
        report_url=None,
    )
    exceeded = check_threshold(delta_pct, cost_threshold)

    return ActionResult(
        score=score.score,
        projected_cost=round(projected_cost, 2),
        cost_delta=round(delta, 2),
        delta_pct=round(delta_pct, 2),
        rec_count=len(recs),
        report_path=str(report_path),
        comment_markdown=comment,
        threshold_exceeded=exceeded,
    )


def main() -> None:
    """CLI entry point for the GitHub Action. Invoked by entrypoint.sh."""
    parser = argparse.ArgumentParser(description="Pretia GitHub Action")
    parser.add_argument("--workflow-path", required=True)
    parser.add_argument("--baseline-path", default=".pretia/baseline.json")
    parser.add_argument("--mode", choices=["diff", "profile"], default="diff")
    parser.add_argument("--framework", default="auto")
    parser.add_argument("--daily-volume", type=int, default=1000)
    parser.add_argument("--cost-threshold", type=float, default=None)
    parser.add_argument("--output-file", required=True)
    args = parser.parse_args()

    if args.mode == "diff":
        result = run_diff_analysis(
            workflow_path=args.workflow_path,
            baseline_path=args.baseline_path,
            daily_volume=args.daily_volume,
            cost_threshold=args.cost_threshold,
        )
    else:
        result = run_full_profile(
            workflow_path=args.workflow_path,
            baseline_path=args.baseline_path,
            framework=args.framework,
            daily_volume=args.daily_volume,
            cost_threshold=args.cost_threshold,
        )

    Path(args.output_file).write_text(json.dumps(result.to_dict(), indent=2))

    if result.threshold_exceeded:
        sys.exit(1)


if __name__ == "__main__":
    main()
