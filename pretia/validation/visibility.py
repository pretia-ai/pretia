"""Visibility warnings and display helpers for projection output."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

from pretia.collectors.base import StepRecord


def get_profiling_recommendation(
    n_eff: float,
    patterns: list[Any],
    current_n: int,
) -> str | None:
    """Return a profiling recommendation based on current data quality."""
    has_high_variance_patterns = any(getattr(p, "severity", "") == "danger" for p in patterns)
    has_conditional_patterns = any(
        getattr(p, "pattern_type", "") in ("step_count_variance", "bimodality") for p in patterns
    )

    if current_n >= 100 and n_eff >= 50:
        return None

    if n_eff < 10:
        return (
            f"Effective sample size is {n_eff:.0f} (from {current_n} runs). "
            "Most runs produced similar costs, limiting statistical power. "
            "Profile with more diverse inputs to improve projection accuracy."
        )

    if has_high_variance_patterns and current_n < 100:
        return (
            f"High-variance patterns detected with only {current_n} runs. "
            "For workflows with context growth or high loop variance, "
            "profiling with n=100+ significantly improves tail estimates. "
            "Run: pretia profile run <workflow> --auto-generate 100"
        )

    if has_conditional_patterns and current_n < 50:
        return (
            f"Conditional branching detected with only {current_n} runs. "
            "Rare execution paths may be underrepresented. "
            "Profile with n=50+ to ensure all paths are sampled. "
            "Run: pretia profile run <workflow> --auto-generate 50"
        )

    if current_n < 50:
        return (
            f"Profiled with {current_n} runs. For production use, "
            "n=50 is recommended for stable projections. "
            "Run: pretia profile run <workflow> --auto-generate 50"
        )

    return None


def compute_input_stats(records: list[list[StepRecord]]) -> dict[str, Any]:
    """Compute input token distribution statistics across profiling runs."""
    if not records:
        return {"total_input_tokens": {}, "per_step": {}}

    run_totals: list[int] = []
    step_tokens: dict[str, list[int]] = defaultdict(list)

    for run in records:
        total = 0
        run_step_tokens: dict[str, int] = defaultdict(int)
        for rec in run:
            total += rec.input_tokens
            run_step_tokens[rec.step_name] += rec.input_tokens
        run_totals.append(total)
        for sn, tok in run_step_tokens.items():
            step_tokens[sn].append(tok)

    def _stats(vals: list[int]) -> dict[str, float]:
        if not vals:
            return {"p50": 0, "p95": 0, "max": 0, "min": 0, "cv": 0.0}
        s = sorted(vals)
        n = len(s)
        mean = sum(s) / n
        var = sum((x - mean) ** 2 for x in s) / (n - 1) if n > 1 else 0.0
        std = math.sqrt(var)
        cv = std / mean if mean > 0 else 0.0

        def _pct(p: float) -> float:
            k = (n - 1) * p / 100
            f = int(k)
            c = f + 1
            if c >= n:
                return float(s[f])
            return s[f] + (k - f) * (s[c] - s[f])

        return {
            "p50": int(_pct(50)),
            "p95": int(_pct(95)),
            "max": s[-1],
            "min": s[0],
            "cv": round(cv, 3),
        }

    return {
        "total_input_tokens": _stats(run_totals),
        "per_step": {sn: _stats(vals) for sn, vals in sorted(step_tokens.items())},
    }


def check_input_uniformity(
    records: list[list[StepRecord]],
    patterns: list[Any],
) -> list[str]:
    """Check for suspiciously uniform input distributions."""
    warnings: list[str] = []
    if not records:
        return warnings

    stats = compute_input_stats(records)
    total_cv = stats["total_input_tokens"].get("cv", 0)
    if total_cv < 0.1 and len(records) >= 5:
        warnings.append(
            f"Input length is very uniform (CV={total_cv:.2f}). Production traffic "
            "with variable input lengths may produce different cost distributions."
        )

    step_iters: dict[str, set[int]] = defaultdict(set)
    for run in records:
        run_counts: dict[str, int] = defaultdict(int)
        for rec in run:
            run_counts[rec.step_name] = max(run_counts[rec.step_name], rec.iteration)
        for sn, count in run_counts.items():
            step_iters[sn].add(count)

    for sn, iter_set in step_iters.items():
        if len(iter_set) == 1 and max(iter_set) > 1 and len(records) >= 5:
            count = next(iter(iter_set))
            warnings.append(
                f"All {len(records)} runs executed exactly {count} iterations "
                f"for step '{sn}'. If production inputs trigger variable loop "
                "counts, re-profile with more diverse inputs."
            )

    return warnings


def check_zero_execution_steps(
    records: list[list[StepRecord]],
    workflow_steps: list[str] | None = None,
) -> list[str]:
    """Compare declared vs observed steps to find never-triggered steps."""
    if not workflow_steps or not records:
        return []

    observed: set[str] = set()
    for run in records:
        for rec in run:
            observed.add(rec.step_name)

    warnings: list[str] = []
    for step in workflow_steps:
        if step not in observed:
            warnings.append(
                f"Step '{step}' is defined in the workflow but was never "
                "triggered during profiling. If this step runs in production, "
                "the projection underestimates costs."
            )
    return warnings


def sample_coverage_statement(n: int) -> str:
    """Compute the minimum event frequency detectable at sample size n."""
    min_detectable = 1 - 0.05 ** (1 / n)
    pct = min_detectable * 100
    return (
        f"With {n} profiling runs, events occurring less than ~{pct:.0f}% "
        "of the time may not be represented in the sample."
    )


def format_projection_output(
    p50: float,
    p95: float,
    n: int,
) -> dict[str, Any]:
    """Format projection results with sample-size-appropriate display."""
    if n < 10:
        return {
            "display_mode": "p50_only",
            "p50": p50,
            "range_note": (f"At {n} profiling runs, actual costs could be 0.5×–3× this estimate."),
            "upgrade_note": "Profile with n=20+ for percentile breakdowns.",
        }
    if n < 20:
        return {
            "display_mode": "p50_p95_warning",
            "p50": p50,
            "p95": p95,
            "warning": (
                f"At {n} runs, tail estimates have limited precision. "
                "Profile with n=50+ for reliable p95."
            ),
        }
    return {
        "display_mode": "full",
        "p50": p50,
        "p95": p95,
    }
