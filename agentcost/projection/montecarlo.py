"""Monte Carlo simulation for non-linear cost projection."""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from agentcost.collectors.base import StepRecord
from agentcost.pricing.tables import calculate_cost
from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.stats import ProfilingStats

logger = logging.getLogger(__name__)


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


def _safe_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    try:
        return calculate_cost(model, input_tokens, output_tokens)
    except (ValueError, KeyError):
        return 0.0


@dataclass(frozen=True, slots=True)
class PercentileProjection:
    """Projected costs at each percentile."""

    p50: float
    p75: float
    p90: float
    p95: float
    p99: float
    mean: float

    def to_dict(self) -> dict[str, float]:
        """Serialize to a JSON-compatible dict."""
        return {
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "p95": self.p95,
            "p99": self.p99,
            "mean": self.mean,
        }


def _build_percentile_projection(values: list[float]) -> PercentileProjection:
    if not values:
        return PercentileProjection(p50=0, p75=0, p90=0, p95=0, p99=0, mean=0)
    s = sorted(values)
    return PercentileProjection(
        p50=_percentile(s, 50),
        p75=_percentile(s, 75),
        p90=_percentile(s, 90),
        p95=_percentile(s, 95),
        p99=_percentile(s, 99),
        mean=sum(s) / len(s),
    )


@dataclass(frozen=True, slots=True)
class MonteCarloResult:
    """Output of a Monte Carlo cost simulation."""

    n_simulations: int
    monthly_projection: PercentileProjection
    daily_projection: PercentileProjection
    per_run_projection: PercentileProjection
    linear_monthly: PercentileProjection
    log_monthly: PercentileProjection
    convergence_check: bool
    growth_model_delta: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "n_simulations": self.n_simulations,
            "monthly_projection": self.monthly_projection.to_dict(),
            "daily_projection": self.daily_projection.to_dict(),
            "per_run_projection": self.per_run_projection.to_dict(),
            "linear_monthly": self.linear_monthly.to_dict(),
            "log_monthly": self.log_monthly.to_dict(),
            "convergence_check": self.convergence_check,
            "growth_model_delta": self.growth_model_delta,
        }


def _precompute_step_data(
    runs: list[list[StepRecord]],
) -> tuple[
    dict[str, list[float]],
    dict[str, list[int]],
    dict[str, list[float]],
]:
    """Pre-compute per-step cost arrays from raw runs."""
    step_run_costs: dict[str, list[float]] = defaultdict(list)
    step_iterations: dict[str, list[int]] = defaultdict(list)
    step_occurrence_costs: dict[str, list[float]] = defaultdict(list)

    for run in runs:
        run_step_costs: dict[str, float] = defaultdict(float)
        run_step_max_iter: dict[str, int] = {}

        for rec in run:
            cost = _safe_cost(rec.model, rec.input_tokens, rec.output_tokens)
            run_step_costs[rec.step_name] += cost
            step_occurrence_costs[rec.step_name].append(cost)
            cur = run_step_max_iter.get(rec.step_name, 0)
            if rec.iteration > cur:
                run_step_max_iter[rec.step_name] = rec.iteration

        for step_name, total_cost in run_step_costs.items():
            step_run_costs[step_name].append(total_cost)
        for step_name, max_iter in run_step_max_iter.items():
            step_iterations[step_name].append(max_iter)

    return dict(step_run_costs), dict(step_iterations), dict(step_occurrence_costs)


def _precompute_growth_data(
    runs: list[list[StepRecord]],
    growth_steps: set[str],
) -> dict[str, dict[str, Any]]:
    """Extract context growth parameters for steps with context_growth patterns."""
    step_growth: dict[str, dict[str, Any]] = {}

    for step_name in growth_steps:
        iter_context: list[tuple[float, float]] = []
        models: list[str] = []
        output_tokens_list: list[float] = []

        for run in runs:
            for rec in run:
                if rec.step_name == step_name:
                    iter_context.append(
                        (float(rec.iteration), float(rec.context_size)),
                    )
                    models.append(rec.model)
                    output_tokens_list.append(float(rec.output_tokens))

        if not iter_context:
            continue

        first_contexts = [c for i, c in iter_context if i == 1.0]
        base_context = (
            sum(first_contexts) / len(first_contexts)
            if first_contexts
            else iter_context[0][1]
        )

        n = len(iter_context)
        xs = [p[0] for p in iter_context]
        ys = [p[1] for p in iter_context]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den > 0 else 0.0

        step_growth[step_name] = {
            "slope": slope,
            "base_context": base_context,
            "model": models[0] if models else "",
            "mean_output_tokens": (
                sum(output_tokens_list) / len(output_tokens_list)
                if output_tokens_list
                else 0.0
            ),
        }

    return step_growth


def _sample_step_cost(
    step_name: str,
    rng: random.Random,
    step_run_costs: dict[str, list[float]],
    step_iterations: dict[str, list[int]],
    step_occurrence_costs: dict[str, list[float]],
    step_growth: dict[str, dict[str, Any]],
    growth_steps: set[str],
    loop_variance_steps: set[str],
) -> tuple[float, float, float]:
    """Sample cost for one step in one simulated run.

    Returns (average_cost, linear_only_cost, log_only_cost).
    """
    if step_name in growth_steps and step_name in step_growth:
        gd = step_growth[step_name]
        slope = gd["slope"]
        base_context = gd["base_context"]
        model = gd["model"]
        mean_output = gd["mean_output_tokens"]

        iters = step_iterations.get(step_name, [1])
        n_iter = rng.choice(iters)

        median_iter = sorted(iters)[len(iters) // 2]

        linear_cost = 0.0
        log_cost = 0.0
        for k in range(1, n_iter + 1):
            ctx_linear = base_context + slope * k
            ctx_linear = max(ctx_linear, 0)

            if median_iter > 1:
                log_scale = math.log(median_iter + 1)
                ctx_log = base_context + slope * (
                    math.log(k + 1) / log_scale
                ) * median_iter
            else:
                ctx_log = ctx_linear
            ctx_log = max(ctx_log, 0)

            linear_cost += _safe_cost(model, int(ctx_linear), int(mean_output))
            log_cost += _safe_cost(model, int(ctx_log), int(mean_output))

        avg_cost = (linear_cost + log_cost) / 2.0
        return avg_cost, linear_cost, log_cost

    if step_name in loop_variance_steps:
        iters = step_iterations.get(step_name, [1])
        n_iter = rng.choice(iters)
        occ_costs = step_occurrence_costs.get(step_name, [0.0])
        total = sum(rng.choice(occ_costs) for _ in range(n_iter))
        return total, total, total

    costs = step_run_costs.get(step_name, [0.0])
    c = rng.choice(costs)
    return c, c, c


def simulate(
    stats: ProfilingStats,
    patterns: list[DetectedPattern],
    daily_volume: int,
    runs: list[list[StepRecord]],
    n_simulations: int = 10000,
    n_days: int = 30,
    seed: int = 42,
) -> MonteCarloResult:
    """Run Monte Carlo simulation for cost projection."""
    rng = random.Random(seed)  # noqa: S311

    growth_steps: set[str] = set()
    loop_variance_steps: set[str] = set()
    for p in patterns:
        if p.pattern_type == "context_growth":
            growth_steps.add(p.step_name)
        elif p.pattern_type == "loop_count_variance":
            loop_variance_steps.add(p.step_name)

    step_run_costs, step_iterations, step_occurrence_costs = _precompute_step_data(
        runs,
    )
    step_growth = _precompute_growth_data(runs, growth_steps)

    all_step_names = list(step_run_costs.keys())

    sim_run_costs: list[float] = []
    sim_daily_costs: list[float] = []
    sim_monthly_costs: list[float] = []
    sim_linear_monthly: list[float] = []
    sim_log_monthly: list[float] = []

    for _ in range(n_simulations):
        run_avg = 0.0
        run_linear = 0.0
        run_log = 0.0

        for sn in all_step_names:
            avg_c, lin_c, log_c = _sample_step_cost(
                sn,
                rng,
                step_run_costs,
                step_iterations,
                step_occurrence_costs,
                step_growth,
                growth_steps,
                loop_variance_steps,
            )
            run_avg += avg_c
            run_linear += lin_c
            run_log += log_c

        daily_avg = run_avg * daily_volume
        daily_linear = run_linear * daily_volume
        daily_log = run_log * daily_volume

        monthly_avg = daily_avg * n_days
        monthly_linear = daily_linear * n_days
        monthly_log = daily_log * n_days

        sim_run_costs.append(run_avg)
        sim_daily_costs.append(daily_avg)
        sim_monthly_costs.append(monthly_avg)
        sim_linear_monthly.append(monthly_linear)
        sim_log_monthly.append(monthly_log)

    converged = True
    if n_simulations >= 10000:
        sorted_partial = sorted(sim_monthly_costs[:9000])
        sorted_full = sorted(sim_monthly_costs)
        p95_partial = _percentile(sorted_partial, 95)
        p95_full = _percentile(sorted_full, 95)
        if p95_full > 0:
            converged = abs(p95_partial - p95_full) / p95_full < 0.01
        else:
            converged = True

    linear_proj = _build_percentile_projection(sim_linear_monthly)
    log_proj = _build_percentile_projection(sim_log_monthly)

    growth_delta = 0.0
    if log_proj.p95 > 0:
        growth_delta = abs(linear_proj.p95 - log_proj.p95) / log_proj.p95 * 100
    elif linear_proj.p95 > 0:
        growth_delta = 100.0

    return MonteCarloResult(
        n_simulations=n_simulations,
        monthly_projection=_build_percentile_projection(sim_monthly_costs),
        daily_projection=_build_percentile_projection(sim_daily_costs),
        per_run_projection=_build_percentile_projection(sim_run_costs),
        linear_monthly=linear_proj,
        log_monthly=log_proj,
        convergence_check=converged,
        growth_model_delta=growth_delta,
    )
