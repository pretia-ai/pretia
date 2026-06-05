"""Calibration scoring: p50 ratio, p95 coverage, directional accuracy, ranking correlation."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agentcost.projection.stats import ProfilingStats

if TYPE_CHECKING:
    from agentcost.projection.projector import ProjectionResult

logger = logging.getLogger(__name__)

_THRESHOLDS = {
    "simple": {
        "p50_ratio_lo": 0.8,
        "p50_ratio_hi": 1.7,
        "p95_coverage": 0.88,
        "range_ratio": 2.5,
        "co_dominant_pct": 0.25,
    },
    "complex": {
        "p50_ratio_lo": 0.8,
        "p50_ratio_hi": 1.7,
        "p95_coverage": 0.80,
        "range_ratio": 6.0,
        "co_dominant_pct": 0.25,
    },
}


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via Abramowitz & Stegun approximation."""
    if x < -8:
        return 0.0
    if x > 8:
        return 1.0
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989422804014327
    poly = t * (
        0.319381530
        + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    p = d * math.exp(-x * x / 2.0) * poly
    return 1.0 - p if x > 0 else p


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF via bisection on _normal_cdf."""
    if p <= 0.001:
        return -3.09
    if p >= 0.999:
        return 3.09
    lo, hi = -5.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _normal_cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


@dataclass(frozen=True, slots=True)
class CalibrationScore:
    """Calibration results for one workflow's projection vs ground truth."""

    workflow_name: str
    sample_size: int
    ground_truth_size: int
    p50_ratio: float
    p95_coverage: float
    range_ratio: float
    top_step_correct: bool
    ranking_correlation: float
    verdict: str
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow_name": self.workflow_name,
            "sample_size": self.sample_size,
            "ground_truth_size": self.ground_truth_size,
            "p50_ratio": self.p50_ratio,
            "p95_coverage": self.p95_coverage,
            "range_ratio": self.range_ratio,
            "top_step_correct": self.top_step_correct,
            "ranking_correlation": self.ranking_correlation,
            "verdict": self.verdict,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
        }


def _spearman_rank_correlation(
    projected_costs: dict[str, float],
    actual_costs: dict[str, float],
) -> float:
    """Compute Spearman rank correlation between two step-cost dicts."""
    common = sorted(set(projected_costs) & set(actual_costs))
    n = len(common)
    if n < 3:
        return 1.0

    def _assign_ranks(values: list[float]) -> list[float]:
        indexed = sorted(enumerate(values), key=lambda t: t[1])
        ranks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j < n - 1 and indexed[j + 1][1] == indexed[j][1]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg_rank
            i = j + 1
        return ranks

    proj_vals = [projected_costs[s] for s in common]
    actual_vals = [actual_costs[s] for s in common]
    proj_ranks = _assign_ranks(proj_vals)
    actual_ranks = _assign_ranks(actual_vals)

    d_squared_sum = sum((pr - ar) ** 2 for pr, ar in zip(proj_ranks, actual_ranks, strict=True))
    return 1.0 - (6.0 * d_squared_sum) / (n * (n * n - 1))


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) via linear interpolation."""
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def bootstrap_bca_ci(
    costs: list[float],
    stat_fn: Any = None,
    n_bootstrap: int = 2000,
    ci_level: float = 0.90,
    seed: int = 42,
    percentile: float | None = None,
) -> tuple[float, float, float]:
    """BCa bootstrap confidence interval. Returns (point_estimate, lower, upper).

    Pass either stat_fn (a callable on list[float]) or percentile (0-100).
    """
    rng = random.Random(seed)  # noqa: S311
    n = len(costs)
    if n == 0:
        return 0.0, 0.0, 0.0

    if stat_fn is None and percentile is not None:
        def stat_fn(c: list[float]) -> float:
            return _percentile(sorted(c), percentile)

    if stat_fn is None:
        def stat_fn(c: list[float]) -> float:
            return sum(c) / len(c) if c else 0.0

    theta_hat = stat_fn(costs)

    boot_thetas: list[float] = []
    for _ in range(n_bootstrap):
        sample = [costs[rng.randint(0, n - 1)] for _ in range(n)]
        boot_thetas.append(stat_fn(sample))
    boot_thetas.sort()

    # Bias correction (z0)
    count_below = sum(1 for t in boot_thetas if t < theta_hat)
    p0 = max(0.001, min(0.999, count_below / n_bootstrap))
    z0 = _normal_ppf(p0)

    # Acceleration (a) via jackknife
    jack_thetas: list[float] = []
    for i in range(n):
        loo = costs[:i] + costs[i + 1 :]
        jack_thetas.append(stat_fn(loo))
    theta_dot = sum(jack_thetas) / n

    num = sum((theta_dot - t) ** 3 for t in jack_thetas)
    den = sum((theta_dot - t) ** 2 for t in jack_thetas)
    a = num / (6 * den**1.5) if den > 0 else 0.0

    alpha_lo = (1 - ci_level) / 2
    alpha_hi = 1 - alpha_lo
    z_lo = _normal_ppf(alpha_lo)
    z_hi = _normal_ppf(alpha_hi)

    denom_lo = 1 - a * (z0 + z_lo)
    denom_hi = 1 - a * (z0 + z_hi)
    if abs(denom_lo) < 1e-10:
        denom_lo = 1e-10
    if abs(denom_hi) < 1e-10:
        denom_hi = 1e-10

    adj_lo = _normal_cdf(z0 + (z0 + z_lo) / denom_lo)
    adj_hi = _normal_cdf(z0 + (z0 + z_hi) / denom_hi)

    idx_lo = max(0, min(n_bootstrap - 1, int(adj_lo * n_bootstrap)))
    idx_hi = max(0, min(n_bootstrap - 1, int(adj_hi * n_bootstrap)))

    return theta_hat, boot_thetas[idx_lo], boot_thetas[idx_hi]


def bootstrap_percentile_ci(
    costs: list[float],
    percentile: float,
    n_bootstrap: int = 2000,
    ci_level: float = 0.90,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Backward-compatible wrapper. Delegates to BCa bootstrap."""
    return bootstrap_bca_ci(
        costs, percentile=percentile, n_bootstrap=n_bootstrap,
        ci_level=ci_level, seed=seed,
    )


def score_projection(
    projected: ProfilingStats,
    ground_truth: ProfilingStats,
    projected_projection: ProjectionResult | None = None,
    traffic: int = 1000,
    workflow_complexity: str = "simple",
    ground_truth_p50_ci: tuple[float, float] | None = None,
    ground_truth_p95_ci: tuple[float, float] | None = None,
) -> CalibrationScore:
    """Score a projection against ground truth data."""
    failures: list[str] = []
    warnings: list[str] = []
    thresholds = _THRESHOLDS.get(workflow_complexity, _THRESHOLDS["simple"])

    # --- p50 ratio ---
    gt_p50 = ground_truth.cost_per_run.p50 if ground_truth.cost_per_run else 0
    proj_p50 = projected.cost_per_run.p50 if projected.cost_per_run else 0
    if gt_p50 > 0:
        p50_ratio = proj_p50 / gt_p50
    else:
        p50_ratio = 1.0
        warnings.append("Ground truth p50 is zero — p50 ratio defaulted to 1.0.")

    p50_in_ci = False
    if ground_truth_p50_ci is not None:
        ci_lo, ci_hi = ground_truth_p50_ci
        p50_in_ci = ci_lo <= proj_p50 <= ci_hi

    p50_lo = thresholds.get("p50_ratio_lo", 0.8)
    p50_hi = thresholds.get("p50_ratio_hi", 1.7)
    if not p50_in_ci:
        if p50_ratio < 0.33 or p50_ratio > 3.0:
            failures.append(f"p50 estimate off by {p50_ratio:.1f}x — projection is unreliable")
        elif p50_ratio < p50_lo or p50_ratio > p50_hi:
            warnings.append(
                f"p50 estimate off by {p50_ratio:.1f}x "
                f"(projected ${proj_p50:.4f}, actual ${gt_p50:.4f})"
            )

    # --- p95 coverage ---
    proj_p95 = projected.cost_per_run.p95 if projected.cost_per_run else 0
    run_costs = [rs.total_cost for rs in ground_truth.run_stats]
    if run_costs:
        below = sum(1 for c in run_costs if c <= proj_p95)
        p95_coverage = below / len(run_costs)
    else:
        gt_p95 = ground_truth.cost_per_run.p95 if ground_truth.cost_per_run else 0
        p95_coverage = 1.0 if gt_p95 <= proj_p95 else 0.0

    p95_cov_threshold = thresholds["p95_coverage"]

    p95_in_ci = False
    if ground_truth_p95_ci is not None:
        ci_lo, ci_hi = ground_truth_p95_ci
        p95_in_ci = ci_lo <= proj_p95 <= ci_hi * 1.2

    if not p95_in_ci:
        if p95_coverage < 0.60:
            failures.append(
                f"p95 coverage is only {p95_coverage:.0%} — "
                "projection is dangerously overconfident"
            )
        elif p95_coverage < p95_cov_threshold:
            warnings.append(
                f"p95 coverage is {p95_coverage:.0%} — projected p95 underestimates tail costs"
            )

    # --- Range ratio ---
    if proj_p50 > 0:
        range_ratio = proj_p95 / proj_p50
    else:
        range_ratio = 1.0

    range_threshold = thresholds["range_ratio"]

    if range_ratio >= 10.0:
        failures.append(f"Projection range is {range_ratio:.1f}x — too wide to be actionable")
    elif range_ratio >= range_threshold:
        warnings.append(
            f"Wide projection range ({range_ratio:.1f}x). Consider more profiling runs."
        )

    # --- Top step correct (with co-dominant detection at 30% threshold) ---
    top_step_correct = True
    if projected.step_stats and ground_truth.step_stats:
        proj_sorted = sorted(
            projected.step_stats.values(),
            key=lambda s: s.cost.mean,
            reverse=True,
        )
        actual_sorted = sorted(
            ground_truth.step_stats.values(),
            key=lambda s: s.cost.mean,
            reverse=True,
        )
        proj_top_name = proj_sorted[0].step_name
        actual_top_name = actual_sorted[0].step_name

        if proj_top_name != actual_top_name:
            co_dominant = False
            if len(actual_sorted) >= 2:
                top_cost = actual_sorted[0].cost.mean
                second_cost = actual_sorted[1].cost.mean
                co_pct = thresholds.get("co_dominant_pct", 0.25)
                if top_cost > 0 and abs(top_cost - second_cost) / top_cost <= co_pct:
                    second_name = actual_sorted[1].step_name
                    if proj_top_name in (actual_top_name, second_name):
                        co_dominant = True
                        warnings.append(
                            f"Top two steps are co-dominant "
                            f"({actual_top_name}: ${top_cost:.4f}, "
                            f"{second_name}: ${second_cost:.4f}). "
                            "Both are acceptable as the top step."
                        )

            if not co_dominant:
                top_step_correct = False
                failures.append(
                    f"Top cost step mismatch: projected "
                    f"'{proj_top_name}', actual '{actual_top_name}'"
                )
    elif not projected.step_stats or not ground_truth.step_stats:
        warnings.append("Cannot compare top step — one profile has no step stats.")

    # --- Spearman rank correlation ---
    proj_step_costs = {name: ss.cost.mean for name, ss in projected.step_stats.items()}
    actual_step_costs = {name: ss.cost.mean for name, ss in ground_truth.step_stats.items()}
    common_steps = set(proj_step_costs) & set(actual_step_costs)

    if len(common_steps) <= 3:
        ranking_correlation = 1.0
    else:
        ranking_correlation = _spearman_rank_correlation(
            proj_step_costs,
            actual_step_costs,
        )
        if ranking_correlation < 0.7:
            failures.append(
                f"Step ranking doesn't match ground truth (Spearman r={ranking_correlation:.2f})"
            )

    # --- Overall verdict ---
    if failures:
        verdict = "FAIL"
    elif warnings:
        verdict = "WARN"
    else:
        verdict = "PASS"

    return CalibrationScore(
        workflow_name=projected.step_stats[next(iter(projected.step_stats))].step_name
        if projected.step_stats
        else "unknown",
        sample_size=projected.total_runs,
        ground_truth_size=ground_truth.total_runs,
        p50_ratio=p50_ratio,
        p95_coverage=p95_coverage,
        range_ratio=range_ratio,
        top_step_correct=top_step_correct,
        ranking_correlation=ranking_correlation,
        verdict=verdict,
        failures=failures,
        warnings=warnings,
    )


def format_calibration_report(scores: list[CalibrationScore]) -> str:
    """Format a rich terminal report for a batch of calibration scores."""
    lines: list[str] = []
    lines.append("AgentCost Backtesting Report")
    lines.append("=" * 40)
    lines.append("")

    header = (
        f"{'Workflow':<22} {'p50':>7} {'p95 cov':>9} {'Range':>8} "
        f"{'Top':>5} {'Rank r':>8} {'Verdict':>8}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    pass_count = 0
    warn_count = 0
    fail_count = 0

    for sc in scores:

        def _check(ok: bool, warn: bool = False) -> str:
            if ok and not warn:
                return "✓"
            if warn:
                return "⚠"
            return "✗"

        p50_ok = 0.7 <= sc.p50_ratio <= 2.0
        p50_warn = (0.33 <= sc.p50_ratio < 0.7) or (2.0 < sc.p50_ratio <= 3.0)
        p95_ok = sc.p95_coverage >= 0.85
        p95_warn = 0.60 <= sc.p95_coverage < 0.85
        range_ok = sc.range_ratio < 3.0
        range_warn = 3.0 <= sc.range_ratio < 10.0
        rank_ok = sc.ranking_correlation > 0.7
        rank_warn = False

        p50_str = f"{sc.p50_ratio:.1f}x {_check(p50_ok, p50_warn)}"
        p95_str = f"{sc.p95_coverage:.0%} {_check(p95_ok, p95_warn)}"
        range_str = f"{sc.range_ratio:.1f}x {_check(range_ok, range_warn)}"
        top_str = _check(sc.top_step_correct)
        rank_str = f"{sc.ranking_correlation:.2f} {_check(rank_ok, rank_warn)}"

        name = sc.workflow_name[:20]
        lines.append(
            f"{name:<22} {p50_str:>7} {p95_str:>9} {range_str:>8} "
            f"{top_str:>5} {rank_str:>8} {sc.verdict:>8}"
        )

        if sc.verdict == "PASS":
            pass_count += 1
        elif sc.verdict == "WARN":
            warn_count += 1
        else:
            fail_count += 1

    lines.append("")
    gate = "PASSED" if fail_count == 0 else "FAILED"
    gate_icon = "✅" if fail_count == 0 else "❌"
    lines.append(
        f"Overall: {pass_count} PASS, {warn_count} WARN, {fail_count} FAIL "
        f"— LAUNCH GATE: {gate_icon} {gate}"
    )

    if fail_count > 0:
        lines.append("Fix projection engine before launch.")

    return "\n".join(lines)
