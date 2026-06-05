"""Compute calibration metrics comparing projections to known distribution truth."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import UTC, datetime

from tests.synthetic.runner import SyntheticCalibrationResult


@dataclass
class CalibrationReport:
    """Aggregate calibration metrics."""

    total_workflows: int
    daily_volume: int
    p50_calibration_rate: float
    p95_coverage_rate: float
    mean_p50_error: float
    by_sample_size: dict[int, dict[str, float]]
    by_distribution: dict[str, dict[str, float]]
    failures: list[dict]
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())


def _true_monthly_p95(wf_result: SyntheticCalibrationResult, n_monthly: int) -> float:
    """Approximate true monthly p95 via CLT."""
    wf = wf_result.workflow
    return n_monthly * wf.true_mean + 1.645 * math.sqrt(n_monthly) * wf.true_std


def compute_calibration_report(
    results: list[SyntheticCalibrationResult],
    daily_volume: int = 1000,
) -> CalibrationReport:
    """Compute all calibration metrics from synthetic results."""
    n_monthly = daily_volume * 30
    total = len(results)

    p50_ok_count = 0
    p95_ok_count = 0
    p50_errors: list[float] = []
    failures: list[dict] = []

    by_n: dict[int, list[dict]] = {}
    by_dist: dict[str, list[dict]] = {}

    for r in results:
        true_monthly_p50 = n_monthly * r.workflow.true_mean
        if true_monthly_p50 > 0:
            p50_ratio = r.projected_p50 / true_monthly_p50
        else:
            p50_ratio = 1.0

        true_mp95 = _true_monthly_p95(r, n_monthly)
        p95_covered = r.projected_p95 >= true_mp95

        p50_in_range = 0.7 <= p50_ratio <= 2.0
        if p50_in_range:
            p50_ok_count += 1
        else:
            failures.append(
                {
                    "name": r.workflow.name,
                    "p50_ratio": round(p50_ratio, 3),
                    "type": r.workflow.distribution_type,
                    "n": r.workflow.sample_size,
                }
            )

        if p95_covered:
            p95_ok_count += 1

        p50_errors.append(abs(p50_ratio - 1.0))

        entry = {"p50_ok": p50_in_range, "p95_ok": p95_covered, "p50_error": abs(p50_ratio - 1.0)}

        by_n.setdefault(r.workflow.sample_size, []).append(entry)

        dist = r.workflow.distribution_type
        by_dist.setdefault(dist, []).append(
            {
                **entry,
                "patterns": r.patterns_detected,
            }
        )

    by_sample_size: dict[int, dict[str, float]] = {}
    for n, entries in sorted(by_n.items()):
        ct = len(entries)
        by_sample_size[n] = {
            "p50_calibration": sum(1 for e in entries if e["p50_ok"]) / ct,
            "p95_coverage": sum(1 for e in entries if e["p95_ok"]) / ct,
            "mean_p50_error": sum(e["p50_error"] for e in entries) / ct,
        }

    by_distribution: dict[str, dict[str, float]] = {}
    for dist, entries in sorted(by_dist.items()):
        ct = len(entries)
        by_distribution[dist] = {
            "p50_calibration": sum(1 for e in entries if e["p50_ok"]) / ct,
            "p95_coverage": sum(1 for e in entries if e["p95_ok"]) / ct,
            "mean_p50_error": sum(e["p50_error"] for e in entries) / ct,
        }

    return CalibrationReport(
        total_workflows=total,
        daily_volume=daily_volume,
        p50_calibration_rate=p50_ok_count / total if total else 0,
        p95_coverage_rate=p95_ok_count / total if total else 0,
        mean_p50_error=sum(p50_errors) / len(p50_errors) if p50_errors else 0,
        by_sample_size=by_sample_size,
        by_distribution=by_distribution,
        failures=failures,
    )


def format_report(report: CalibrationReport) -> str:
    """Format calibration report as markdown."""
    lines = [
        "# Synthetic Calibration Report",
        "",
        f"Date: {report.timestamp}",
        f"Workflows tested: {report.total_workflows}",
        f"Daily volume: {report.daily_volume} requests/day",
        "",
        "## Overall Calibration",
        "",
        f"- p50 within (0.7, 2.0): {report.p50_calibration_rate:.0%}",
        f"- p95 coverage >= true: {report.p95_coverage_rate:.0%}",
        f"- Mean |p50 error|: {report.mean_p50_error:.3f}",
        "",
        "## By Sample Size",
        "",
        "| n | p50 calib | p95 coverage | Mean |p50 err| |",
        "|---|-----------|-------------|-----------------|",
    ]
    for n, m in sorted(report.by_sample_size.items()):
        lines.append(
            f"| {n} | {m['p50_calibration']:.0%} "
            f"| {m['p95_coverage']:.0%} "
            f"| {m['mean_p50_error']:.3f} |"
        )

    lines.extend(["", "## By Distribution Type", ""])
    lines.append("| Type | p50 calib | p95 coverage |")
    lines.append("|------|-----------|-------------|")
    for dist, m in sorted(report.by_distribution.items()):
        lines.append(f"| {dist} | {m['p50_calibration']:.0%} | {m['p95_coverage']:.0%} |")

    if report.failures:
        lines.extend(["", f"## Failures ({len(report.failures)} workflows)", ""])
        lines.append("| Workflow | p50 ratio | Type | n |")
        lines.append("|----------|-----------|------|---|")
        for f in report.failures[:20]:
            lines.append(f"| {f['name']} | {f['p50_ratio']} | {f['type']} | {f['n']} |")
        if len(report.failures) > 20:
            lines.append(f"| ... and {len(report.failures) - 20} more | | | |")

    return "\n".join(lines)
