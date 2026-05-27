"""Detect non-linear cost patterns: context growth, loop variance, high token variance."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from agentcost.collectors.base import StepRecord
from agentcost.projection.stats import ProfilingStats, compute_stats

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class DetectedPattern:
    """One detected non-linear cost pattern."""

    pattern_type: str
    step_name: str
    severity: str
    evidence: dict[str, Any]
    description: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pattern_type": self.pattern_type,
            "step_name": self.step_name,
            "severity": self.severity,
            "evidence": self.evidence,
            "description": self.description,
        }


def _pearson_r_squared(
    xs: list[float], ys: list[float],
) -> tuple[float, float]:
    """Return (r_squared, slope) for two equal-length lists, or (0, 0) if degenerate."""
    n = len(xs)
    if n < 3:
        return 0.0, 0.0
    sum_x = sum(xs)
    sum_y = sum(ys)
    sum_xy = sum(x * y for x, y in zip(xs, ys, strict=True))
    sum_x2 = sum(x * x for x in xs)
    sum_y2 = sum(y * y for y in ys)

    denom_x = n * sum_x2 - sum_x * sum_x
    denom_y = n * sum_y2 - sum_y * sum_y
    if denom_x == 0 or denom_y == 0:
        return 0.0, 0.0

    numerator = n * sum_xy - sum_x * sum_y
    denom = math.sqrt(denom_x * denom_y)
    r = numerator / denom
    slope = numerator / denom_x
    return r * r, slope


def _detect_context_growth(
    runs: list[list[StepRecord]],
) -> list[DetectedPattern]:
    """Detect steps where context_size grows with iteration number."""
    step_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)

    for run in runs:
        for rec in run:
            if rec.iteration > 1 or any(
                r.step_name == rec.step_name and r.iteration > 1 for r in run
            ):
                step_pairs[rec.step_name].append(
                    (float(rec.iteration), float(rec.context_size)),
                )

    patterns: list[DetectedPattern] = []
    for step_name, pairs in step_pairs.items():
        if len(pairs) < 3:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        r_squared, slope = _pearson_r_squared(xs, ys)
        if r_squared <= 0.7 or slope <= 0:
            continue

        first_iter_contexts = [y for x, y in pairs if x == 1.0]
        max_iter = max(xs)
        last_iter_contexts = [y for x, y in pairs if x == max_iter]
        mean_first = (
            sum(first_iter_contexts) / len(first_iter_contexts)
            if first_iter_contexts else ys[0]
        )
        mean_last = (
            sum(last_iter_contexts) / len(last_iter_contexts)
            if last_iter_contexts else ys[-1]
        )
        ratio = mean_last / mean_first if mean_first > 0 else 0.0

        severity = "danger" if r_squared > 0.85 else "warning"
        patterns.append(DetectedPattern(
            pattern_type="context_growth",
            step_name=step_name,
            severity=severity,
            evidence={
                "r_squared": round(r_squared, 4),
                "slope": round(slope, 2),
                "mean_context_first": round(mean_first, 2),
                "mean_context_last": round(mean_last, 2),
                "n_datapoints": len(pairs),
            },
            description=(
                f"Context grows by ~{slope:.0f} tokens per iteration in step "
                f"'{step_name}' (r²={r_squared:.2f}). At iteration {int(max_iter)}, "
                f"context is {ratio:.1f}x the initial size. Linear projection will "
                f"underestimate costs for high-iteration runs."
            ),
        ))
    return patterns


def _detect_loop_count_variance(
    runs: list[list[StepRecord]],
) -> list[DetectedPattern]:
    """Detect steps where loop iteration count varies significantly across runs."""
    step_max_iter: dict[str, list[int]] = defaultdict(list)
    for run in runs:
        run_maxes: dict[str, int] = {}
        for rec in run:
            cur = run_maxes.get(rec.step_name, 0)
            if rec.iteration > cur:
                run_maxes[rec.step_name] = rec.iteration
        for step_name, max_iter in run_maxes.items():
            step_max_iter[step_name].append(max_iter)

    patterns: list[DetectedPattern] = []
    for step_name, iters in step_max_iter.items():
        if all(i == 1 for i in iters):
            continue
        n = len(iters)
        if n < 2:
            continue
        mean_iter = sum(iters) / n
        if mean_iter == 0:
            continue
        variance = sum((x - mean_iter) ** 2 for x in iters) / (n - 1) if n > 1 else 0.0
        std_iter = math.sqrt(variance)
        cv = std_iter / mean_iter
        if cv <= 0.5:
            continue

        max_i = max(iters)
        min_i = min(iters)
        ratio = max_i / mean_iter if mean_iter > 0 else 0.0
        severity = "danger" if cv > 1.0 or max_i > 3 * mean_iter else "warning"

        patterns.append(DetectedPattern(
            pattern_type="loop_count_variance",
            step_name=step_name,
            severity=severity,
            evidence={
                "cv": round(cv, 4),
                "mean_iterations": round(mean_iter, 2),
                "min_iterations": min_i,
                "max_iterations": max_i,
                "std_iterations": round(std_iter, 4),
            },
            description=(
                f"Loop count for step '{step_name}' varies from {min_i} to {max_i} "
                f"iterations (mean={mean_iter:.1f}, CV={cv:.2f}). Worst-case runs "
                f"cost ~{ratio:.1f}x the average."
            ),
        ))
    return patterns


def _detect_high_token_variance(
    stats: ProfilingStats,
) -> list[DetectedPattern]:
    """Detect steps with heavy-tailed token or cost distributions."""
    patterns: list[DetectedPattern] = []
    for step_name, ss in stats.step_stats.items():
        p50_tok = ss.total_tokens.p50
        p95_tok = ss.total_tokens.p95
        p50_cost = ss.cost.p50
        p95_cost = ss.cost.p95

        ratio_tok = p95_tok / p50_tok if p50_tok > 0 else 0.0
        ratio_cost = p95_cost / p50_cost if p50_cost > 0 else 0.0
        ratio = max(ratio_tok, ratio_cost)

        if ratio <= 3.0:
            continue

        severity = "danger" if ratio > 5.0 else "warning"
        patterns.append(DetectedPattern(
            pattern_type="high_token_variance",
            step_name=step_name,
            severity=severity,
            evidence={
                "p95_p50_ratio_tokens": round(ratio_tok, 4),
                "p95_p50_ratio_cost": round(ratio_cost, 4),
                "p50_tokens": round(p50_tok, 2),
                "p95_tokens": round(p95_tok, 2),
                "p50_cost": round(p50_cost, 6),
                "p95_cost": round(p95_cost, 6),
            },
            description=(
                f"Step '{step_name}' has high token variance: p95 is {ratio_tok:.1f}x "
                f"the median ({p50_tok:.0f} vs {p95_tok:.0f} total tokens). "
                f"Average-based projection will underestimate tail costs."
            ),
        ))
    return patterns


def detect_patterns(
    runs: list[list[StepRecord]],
    stats: ProfilingStats | None = None,
) -> list[DetectedPattern]:
    """Run all pattern detectors and return results sorted by severity (danger first)."""
    if not runs:
        return []
    if stats is None:
        stats = compute_stats(runs)

    patterns: list[DetectedPattern] = []
    patterns.extend(_detect_context_growth(runs))
    patterns.extend(_detect_loop_count_variance(runs))
    patterns.extend(_detect_high_token_variance(stats))

    severity_order = {"danger": 0, "warning": 1}
    patterns.sort(key=lambda p: severity_order.get(p.severity, 2))
    return patterns
