"""Loop cap and circuit breaker recommendations based on iteration patterns."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pretia.recommend.base import (
    _DEFAULT_DAILY_VOLUME,
    Recommendation,
    RecommendationGenerator,
    _extract_pattern_dicts,
    _safe_record_cost,
    compute_priority,
)
from pretia.recommend.registry import register

if TYPE_CHECKING:
    from pretia.collectors.base import StepRecord
    from pretia.store import ProfilingSession


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Compute the *pct*-th percentile from a sorted list (linear interpolation)."""
    n = len(sorted_values)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_values[0]
    k = (pct / 100) * (n - 1)
    lo = int(k)
    hi = min(lo + 1, n - 1)
    frac = k - lo
    return sorted_values[lo] + frac * (sorted_values[hi] - sorted_values[lo])


def _iter_distribution(
    runs: list[list[StepRecord]], step_name: str
) -> list[int]:
    """Return per-run max iteration for *step_name*, sorted ascending."""
    counts: list[int] = []
    for run in runs:
        step_iters = [r.iteration for r in run if r.step_name == step_name]
        if step_iters:
            counts.append(max(step_iters))
    counts.sort()
    return counts


def _step_cost_per_run(
    runs: list[list[StepRecord]], step_name: str
) -> list[float]:
    """Return total cost for *step_name* in each run."""
    costs: list[float] = []
    for run in runs:
        total = sum(
            _safe_record_cost(r.model, r.input_tokens, r.output_tokens)
            for r in run
            if r.step_name == step_name
        )
        costs.append(total)
    return costs


@register
class LoopCapGenerator(RecommendationGenerator):
    """Recommend iteration caps for steps with high loop count variance."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        patterns = _extract_pattern_dicts(profile)
        loop_patterns = [
            p for p in patterns if p.get("pattern_type") == "loop_count_variance"
        ]
        if not loop_patterns or not profile.runs:
            return []

        recommendations: list[Recommendation] = []
        for pattern in loop_patterns:
            rec = self._evaluate_pattern(pattern, profile)
            if rec is not None:
                recommendations.append(rec)
        return recommendations

    def _evaluate_pattern(
        self, pattern: dict[str, Any], profile: ProfilingSession
    ) -> Recommendation | None:
        step_name = pattern.get("step_name", "")
        evidence = pattern.get("evidence", {})
        cv = evidence.get("cv", 0.0)

        if cv <= 0.5:
            return None

        dist = _iter_distribution(profile.runs, step_name)
        if len(dist) < 2:
            return None

        max_iter = dist[-1]
        p75 = int(math.ceil(_percentile(dist, 75)))
        p90 = int(math.ceil(_percentile(dist, 90)))

        cap = p75 if p75 < max_iter else p90
        if cap >= max_iter:
            return None

        total_excess_cost = 0.0
        for run in profile.runs:
            step_records = [r for r in run if r.step_name == step_name]
            for r in step_records:
                if r.iteration > cap:
                    total_excess_cost += _safe_record_cost(
                        r.model, r.input_tokens, r.output_tokens
                    )

        n_runs = len(profile.runs)
        avg_excess = total_excess_cost / n_runs if n_runs > 0 else 0.0
        monthly_savings = round(avg_excess * _DEFAULT_DAILY_VOLUME * 30, 2)

        if monthly_savings < 1.0:
            return None

        mean_iters = evidence.get("mean_iterations", sum(dist) / len(dist))

        return Recommendation(
            id=f"loop-cap-{step_name}",
            type="workflow",
            title=f"Cap {step_name} iterations at {cap}",
            description=(
                f"{step_name} averages {mean_iters:.1f} iterations per run "
                f"(CV={cv:.2f}) with a max of {max_iter}. "
                f"Capping at {cap} (p75) saves ${monthly_savings:,.0f}/month "
                f"at {_DEFAULT_DAILY_VOLUME:,} daily runs. "
                f"Iterations beyond {cap} show diminishing returns."
            ),
            monthly_savings=monthly_savings,
            confidence="MODERATE",
            affected_steps=[step_name],
            evidence={
                "cv": cv,
                "mean_iterations": mean_iters,
                "max_iterations": max_iter,
                "recommended_cap": cap,
                "iteration_distribution": dist,
                "p75": p75,
                "p90": p90,
                "avg_excess_cost_per_run": round(avg_excess, 6),
                "daily_volume": _DEFAULT_DAILY_VOLUME,
            },
            priority=compute_priority(monthly_savings, "MODERATE"),
        )


@register
class CircuitBreakerGenerator(RecommendationGenerator):
    """Recommend circuit breakers for steps with runaway outlier iterations."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        patterns = _extract_pattern_dicts(profile)
        loop_patterns = [
            p for p in patterns if p.get("pattern_type") == "loop_count_variance"
        ]
        if not loop_patterns or not profile.runs:
            return []

        recommendations: list[Recommendation] = []
        for pattern in loop_patterns:
            rec = self._evaluate_pattern(pattern, profile)
            if rec is not None:
                recommendations.append(rec)
        return recommendations

    def _evaluate_pattern(
        self, pattern: dict[str, Any], profile: ProfilingSession
    ) -> Recommendation | None:
        step_name = pattern.get("step_name", "")
        evidence = pattern.get("evidence", {})

        mean_iters = evidence.get("mean_iterations", 0.0)
        if mean_iters <= 0:
            return None

        threshold = int(math.ceil(2 * mean_iters))

        dist = _iter_distribution(profile.runs, step_name)
        if len(dist) < 2:
            return None

        outlier_run_indices: list[int] = []
        for i, run in enumerate(profile.runs):
            step_iters = [r.iteration for r in run if r.step_name == step_name]
            if step_iters and max(step_iters) > threshold:
                outlier_run_indices.append(i)

        if not outlier_run_indices:
            return None

        total_step_cost = 0.0
        outlier_step_cost = 0.0
        for i, run in enumerate(profile.runs):
            run_cost = sum(
                _safe_record_cost(r.model, r.input_tokens, r.output_tokens)
                for r in run
                if r.step_name == step_name
            )
            total_step_cost += run_cost
            if i in outlier_run_indices:
                outlier_step_cost += run_cost

        if total_step_cost <= 0:
            return None

        cost_share = outlier_step_cost / total_step_cost
        if cost_share <= 0.15:
            return None

        excess_cost = 0.0
        for i in outlier_run_indices:
            run = profile.runs[i]
            for r in run:
                if r.step_name == step_name and r.iteration > threshold:
                    excess_cost += _safe_record_cost(
                        r.model, r.input_tokens, r.output_tokens
                    )

        n_runs = len(profile.runs)
        avg_excess = excess_cost / n_runs if n_runs > 0 else 0.0
        monthly_savings = round(avg_excess * _DEFAULT_DAILY_VOLUME * 30, 2)

        if monthly_savings < 1.0:
            return None

        n_outliers = len(outlier_run_indices)

        return Recommendation(
            id=f"circuit-breaker-{step_name}",
            type="workflow",
            title=f"Add circuit breaker for {step_name} at {threshold} iterations",
            description=(
                f"{n_outliers} of {n_runs} runs exceeded {threshold} iterations "
                f"(2x the mean of {mean_iters:.1f}), consuming "
                f"{cost_share:.0%} of total step cost. "
                f"Adding a hard exit at {threshold} iterations "
                f"saves ${monthly_savings:,.0f}/month "
                f"at {_DEFAULT_DAILY_VOLUME:,} daily runs."
            ),
            monthly_savings=monthly_savings,
            confidence="HIGH",
            affected_steps=[step_name],
            evidence={
                "mean_iterations": mean_iters,
                "threshold": threshold,
                "outlier_run_count": n_outliers,
                "total_run_count": n_runs,
                "outlier_cost_share": round(cost_share, 4),
                "excess_cost_total": round(excess_cost, 6),
                "avg_excess_cost_per_run": round(avg_excess, 6),
                "daily_volume": _DEFAULT_DAILY_VOLUME,
            },
            priority=compute_priority(monthly_savings, "HIGH"),
        )
