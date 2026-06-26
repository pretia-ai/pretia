"""Compute distributional statistics (p50-p99) from profiling data."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import calculate_cost

logger = logging.getLogger(__name__)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute the p-th percentile (0-100) of sorted data using linear interpolation."""
    n = len(sorted_data)
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def robust_cv(values: list[float]) -> float:
    """Compute MAD-based coefficient of variation. Resistant to outliers.

    The constant 1.4826 makes MAD consistent with standard deviation
    for normal distributions.
    """
    n = len(values)
    if n < 2:
        return 0.0
    s = sorted(values)
    median = s[n // 2]
    if median == 0:
        return 0.0
    deviations = sorted(abs(v - median) for v in values)
    mad = deviations[n // 2]
    return 1.4826 * mad / median


@dataclass(frozen=True, slots=True)
class PercentileStats:
    """Standard set of percentiles for a single metric."""

    min: float
    max: float
    mean: float
    std: float
    p50: float
    p75: float
    p90: float
    p95: float
    p99: float

    def to_dict(self) -> dict[str, float]:
        """Serialize to a JSON-compatible dict."""
        return {
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "std": self.std,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
        }


def compute_percentile_stats(values: list[float]) -> PercentileStats:
    """Compute a full PercentileStats from a list of floats."""
    if not values:
        raise ValueError("Cannot compute stats on empty data")
    n = len(values)
    s = sorted(values)
    mean = sum(s) / n
    if n == 1:
        return PercentileStats(
            min=s[0],
            max=s[0],
            mean=mean,
            std=0.0,
            p50=s[0],
            p75=s[0],
            p90=s[0],
            p95=s[0],
            p99=s[0],
        )
    variance = sum((x - mean) ** 2 for x in s) / (n - 1)
    std = math.sqrt(variance)
    return PercentileStats(
        min=s[0],
        max=s[-1],
        mean=mean,
        std=std,
        p50=_percentile(s, 50),
        p75=_percentile(s, 75),
        p90=_percentile(s, 90),
        p95=_percentile(s, 95),
        p99=_percentile(s, 99),
    )


@dataclass(slots=True)
class StepStats:
    """Per-step statistics across all runs."""

    step_name: str
    step_type: str
    model: str
    call_count: int
    runs_present: int
    input_tokens: PercentileStats
    output_tokens: PercentileStats
    total_tokens: PercentileStats
    cost: PercentileStats
    duration_ms: PercentileStats
    context_size: PercentileStats
    iterations_per_run: PercentileStats
    mean_iterations: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "step_name": self.step_name,
            "step_type": self.step_type,
            "model": self.model,
            "call_count": self.call_count,
            "runs_present": self.runs_present,
            "input_tokens": self.input_tokens.to_dict(),
            "output_tokens": self.output_tokens.to_dict(),
            "total_tokens": self.total_tokens.to_dict(),
            "cost": self.cost.to_dict(),
            "duration_ms": self.duration_ms.to_dict(),
            "context_size": self.context_size.to_dict(),
            "iterations_per_run": self.iterations_per_run.to_dict(),
            "mean_iterations": self.mean_iterations,
        }


@dataclass(slots=True)
class RunStats:
    """Aggregate statistics for a single run."""

    run_index: int
    total_cost: float
    total_tokens: int
    total_input_tokens: int
    total_output_tokens: int
    step_count: int
    duration_ms: int

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "run_index": self.run_index,
            "total_cost": self.total_cost,
            "total_tokens": self.total_tokens,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "step_count": self.step_count,
            "duration_ms": self.duration_ms,
        }


@dataclass(slots=True)
class ProfilingStats:
    """Top-level container for all statistics from a profiling session."""

    step_stats: dict[str, StepStats] = field(default_factory=dict)
    run_stats: list[RunStats] = field(default_factory=list)
    cost_per_run: PercentileStats | None = None
    tokens_per_run: PercentileStats | None = None
    total_runs: int = 0
    total_steps: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the entire structure for JSON persistence."""
        return {
            "step_stats": {k: v.to_dict() for k, v in self.step_stats.items()},
            "run_stats": [r.to_dict() for r in self.run_stats],
            "cost_per_run": self.cost_per_run.to_dict() if self.cost_per_run else None,
            "tokens_per_run": self.tokens_per_run.to_dict() if self.tokens_per_run else None,
            "total_runs": self.total_runs,
            "total_steps": self.total_steps,
        }


def _safe_cost(
    cost_fn: Callable[..., float],
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    try:
        return cost_fn(model, input_tokens, output_tokens)
    except (ValueError, KeyError):
        logger.warning("Unknown model %r in cost calculation — using $0.00", model)
        return 0.0


def compute_stats(
    runs: list[list[StepRecord]],
    cost_fn: Callable[..., float] | None = None,
) -> ProfilingStats:
    """Compute distributional statistics from raw profiling data."""
    if cost_fn is None:
        cost_fn = calculate_cost

    if not runs:
        return ProfilingStats()

    step_records: dict[str, list[StepRecord]] = defaultdict(list)
    step_runs_presence: dict[str, set[int]] = defaultdict(set)
    step_iterations_per_run: dict[str, dict[int, int]] = defaultdict(dict)

    run_stats_list: list[RunStats] = []
    run_costs: list[float] = []
    run_tokens: list[float] = []
    total_step_count = 0

    for run_idx, run in enumerate(runs):
        run_cost = 0.0
        run_total_tokens = 0
        run_input_tokens = 0
        run_output_tokens = 0
        run_duration = 0

        for rec in run:
            step_records[rec.step_name].append(rec)
            step_runs_presence[rec.step_name].add(run_idx)
            cur_max = step_iterations_per_run[rec.step_name].get(run_idx, 0)
            if rec.iteration > cur_max:
                step_iterations_per_run[rec.step_name][run_idx] = rec.iteration

            cost = _safe_cost(cost_fn, rec.model, rec.input_tokens, rec.output_tokens)
            run_cost += cost
            run_total_tokens += rec.input_tokens + rec.output_tokens
            run_input_tokens += rec.input_tokens
            run_output_tokens += rec.output_tokens
            run_duration += rec.duration_ms
            total_step_count += 1

        run_costs.append(run_cost)
        run_tokens.append(float(run_total_tokens))
        run_stats_list.append(
            RunStats(
                run_index=run_idx,
                total_cost=run_cost,
                total_tokens=run_total_tokens,
                total_input_tokens=run_input_tokens,
                total_output_tokens=run_output_tokens,
                step_count=len(run),
                duration_ms=run_duration,
            )
        )

    step_stats_dict: dict[str, StepStats] = {}
    for step_name, records in step_records.items():
        model = records[0].model
        step_type = records[0].step_type

        input_tok_vals = [float(r.input_tokens) for r in records]
        output_tok_vals = [float(r.output_tokens) for r in records]
        total_tok_vals = [float(r.input_tokens + r.output_tokens) for r in records]
        cost_vals = [
            _safe_cost(cost_fn, r.model, r.input_tokens, r.output_tokens) for r in records
        ]
        duration_vals = [float(r.duration_ms) for r in records]
        context_vals = [float(r.context_size) for r in records]

        iter_per_run = [
            float(step_iterations_per_run[step_name].get(ri, 0))
            for ri in range(len(runs))
            if ri in step_runs_presence[step_name]
        ]
        if not iter_per_run:
            iter_per_run = [1.0]

        step_stats_dict[step_name] = StepStats(
            step_name=step_name,
            step_type=step_type,
            model=model,
            call_count=len(records),
            runs_present=len(step_runs_presence[step_name]),
            input_tokens=compute_percentile_stats(input_tok_vals),
            output_tokens=compute_percentile_stats(output_tok_vals),
            total_tokens=compute_percentile_stats(total_tok_vals),
            cost=compute_percentile_stats(cost_vals),
            duration_ms=compute_percentile_stats(duration_vals),
            context_size=compute_percentile_stats(context_vals),
            iterations_per_run=compute_percentile_stats(iter_per_run),
            mean_iterations=sum(iter_per_run) / len(iter_per_run),
        )

    return ProfilingStats(
        step_stats=step_stats_dict,
        run_stats=run_stats_list,
        cost_per_run=compute_percentile_stats(run_costs),
        tokens_per_run=compute_percentile_stats(run_tokens),
        total_runs=len(runs),
        total_steps=total_step_count,
    )
