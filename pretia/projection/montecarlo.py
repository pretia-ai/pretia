"""Monte Carlo simulation for non-linear cost projection."""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import calculate_cost
from pretia.projection.patterns import DetectedPattern
from pretia.projection.stats import ProfilingStats, percentile

logger = logging.getLogger(__name__)


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
        p50=percentile(s, 50),
        p75=percentile(s, 75),
        p90=percentile(s, 90),
        p95=percentile(s, 95),
        p99=percentile(s, 99),
        mean=sum(s) / len(s),
    )


def compute_cvar(simulated_costs: list[float], alpha: float = 0.05) -> float:
    """Conditional Value-at-Risk: mean of the top alpha fraction of costs."""
    if not simulated_costs:
        return 0.0
    sorted_costs = sorted(simulated_costs)
    cutoff_idx = int((1 - alpha) * len(sorted_costs))
    tail = sorted_costs[cutoff_idx:]
    return sum(tail) / len(tail) if tail else sorted_costs[-1]


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
    tail_inflation_factor: float | None = None
    extrapolation_cap_warnings: int = 0
    cvar_95: float = 0.0
    cvar_99: float | None = None

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
            "tail_inflation_factor": self.tail_inflation_factor,
            "extrapolation_cap_warnings": self.extrapolation_cap_warnings,
            "cvar_95": self.cvar_95,
            "cvar_99": self.cvar_99,
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
            sum(first_contexts) / len(first_contexts) if first_contexts else iter_context[0][1]
        )

        n = len(iter_context)
        xs = [p[0] for p in iter_context]
        ys = [p[1] for p in iter_context]
        mean_x = sum(xs) / n
        mean_y = sum(ys) / n
        num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True))
        den = sum((x - mean_x) ** 2 for x in xs)
        slope = num / den if den > 0 else 0.0

        max_observed_context = max(c for _, c in iter_context)

        step_growth[step_name] = {
            "slope": slope,
            "base_context": base_context,
            "model": models[0] if models else "",
            "mean_output_tokens": (
                sum(output_tokens_list) / len(output_tokens_list) if output_tokens_list else 0.0
            ),
            "max_observed_context": max_observed_context,
        }

    return step_growth


def _sample_from_gmm(
    rng: random.Random,
    means: list[float],
    stds: list[float],
    weights: list[float],
) -> float:
    """Sample one cost from a 2-component log-normal mixture."""
    if rng.random() < weights[0]:  # noqa: S311
        log_cost = means[0] + stds[0] * rng.gauss(0, 1)
    else:
        log_cost = means[1] + stds[1] * rng.gauss(0, 1)
    return math.exp(log_cost)


def _build_run_step_cost_map(runs: list[list[StepRecord]]) -> list[dict[str, float]]:
    """Build per-run, per-step cost mapping for whole-run sampling (Fix 3)."""
    result: list[dict[str, float]] = []
    for run in runs:
        step_costs: dict[str, float] = defaultdict(float)
        for rec in run:
            cost = _safe_cost(rec.model, rec.input_tokens, rec.output_tokens)
            step_costs[rec.step_name] += cost
        result.append(dict(step_costs))
    return result


def _clt_aggregate(costs: list[float], n_total: int, z: float) -> float:
    """Aggregate K sampled run costs to a total for n_total runs using CLT."""
    k = len(costs)
    if k == 0 or n_total == 0:
        return 0.0
    mu = sum(costs) / k
    if k > 1:
        var = sum((x - mu) ** 2 for x in costs) / (k - 1)
    else:
        var = 0.0
    result = n_total * mu + z * math.sqrt(n_total * var)
    return max(result, 0.0)


def _sample_step_cost(
    step_name: str,
    rng: random.Random,
    step_run_costs: dict[str, list[float]],
    step_iterations: dict[str, list[int]],
    step_occurrence_costs: dict[str, list[float]],
    step_growth: dict[str, dict[str, Any]],
    growth_steps: set[str],
    loop_variance_steps: set[str],
    max_observed_iters: dict[str, int] | None = None,
) -> tuple[float, float, float, bool]:
    """Sample cost for one step in one simulated run.

    Returns (average_cost, linear_only_cost, log_only_cost, was_capped).
    """
    if step_name in growth_steps and step_name in step_growth:
        gd = step_growth[step_name]
        slope = gd["slope"]
        base_context = gd["base_context"]
        model = gd["model"]
        mean_output = gd["mean_output_tokens"]
        g_type = gd.get("growth_type") or "linear"
        alpha = gd.get("power_law_alpha") or 1.0
        power_base = gd.get("power_law_base") or base_context
        classification = gd.get("growth_classification")
        max_obs_ctx = gd.get("max_observed_context") or float("inf")

        iters = step_iterations.get(step_name, [1])
        n_iter = rng.choice(iters)

        max_obs = (
            max_observed_iters.get(step_name, n_iter) if max_observed_iters is not None else n_iter
        )
        capped = n_iter > max_obs

        median_iter = sorted(iters)[len(iters) // 2]

        linear_cost = 0.0
        log_cost = 0.0
        power_cost = 0.0
        capped_power_cost = 0.0
        for k in range(1, n_iter + 1):
            effective_k = min(k, max_obs)

            ctx_linear = max(base_context + slope * effective_k, 0)

            if median_iter > 1:
                log_scale = math.log(median_iter + 1)
                ctx_log = max(
                    base_context + slope * (math.log(effective_k + 1) / log_scale) * median_iter,
                    0,
                )
            else:
                ctx_log = ctx_linear

            linear_cost += _safe_cost(model, int(ctx_linear), int(mean_output))
            log_cost += _safe_cost(model, int(ctx_log), int(mean_output))

            if g_type == "nonlinear":
                ctx_power = max(power_base * (effective_k**alpha), 0)
                power_cost += _safe_cost(model, int(ctx_power), int(mean_output))
                ctx_capped_p = min(ctx_power, 3 * max_obs_ctx)
                capped_power_cost += _safe_cost(model, int(ctx_capped_p), int(mean_output))

        if g_type == "nonlinear":
            if classification == "sub_linear":
                avg_cost = (log_cost + power_cost) / 2.0
                return avg_cost, power_cost, log_cost, capped
            else:
                avg_cost = (power_cost + capped_power_cost) / 2.0
                return avg_cost, power_cost, capped_power_cost, capped
        else:
            avg_cost = (linear_cost + log_cost) / 2.0
            return avg_cost, linear_cost, log_cost, capped

    if step_name in loop_variance_steps:
        iters = step_iterations.get(step_name, [1])
        n_iter = rng.choice(iters)
        occ_costs = step_occurrence_costs.get(step_name, [0.0])
        total = sum(rng.choice(occ_costs) for _ in range(n_iter))
        return total, total, total, False

    costs = step_run_costs.get(step_name, [0.0])
    c = rng.choice(costs)
    return c, c, c, False


def _inflate_p95(proj: PercentileProjection, factor: float) -> PercentileProjection:
    """Return a new projection with p95 multiplied by the inflation factor."""
    return PercentileProjection(
        p50=proj.p50,
        p75=proj.p75,
        p90=proj.p90,
        p95=proj.p95 * factor,
        p99=proj.p99,
        mean=proj.mean,
    )


def simulate(
    stats: ProfilingStats,
    patterns: list[DetectedPattern],
    daily_volume: int,
    runs: list[list[StepRecord]],
    n_simulations: int = 10000,
    n_days: int = 30,
    seed: int = 42,
    _debug_run_costs: list[float] | None = None,
) -> MonteCarloResult:
    """Run Monte Carlo simulation for cost projection."""
    rng = random.Random(seed)  # noqa: S311

    growth_steps: set[str] = set()
    loop_variance_steps: set[str] = set()
    bimodal_gmm: dict[str, Any] | None = None
    for p in patterns:
        if p.pattern_type == "context_growth":
            growth_steps.add(p.step_name)
        elif p.pattern_type == "loop_count_variance":
            loop_variance_steps.add(p.step_name)
        elif p.pattern_type == "bimodality" and p.gmm_means is not None:
            bimodal_gmm = {
                "means": p.gmm_means,
                "stds": p.gmm_stds,
                "weights": p.gmm_weights,
            }

    step_run_costs, step_iterations, step_occurrence_costs = _precompute_step_data(runs)
    step_growth = _precompute_growth_data(runs, growth_steps)

    for p in patterns:
        if p.pattern_type == "context_growth" and p.step_name in step_growth:
            step_growth[p.step_name]["growth_type"] = getattr(p, "growth_type", None)
            step_growth[p.step_name]["power_law_alpha"] = getattr(p, "power_law_alpha", None)
            step_growth[p.step_name]["power_law_base"] = getattr(p, "power_law_base", None)
            step_growth[p.step_name]["growth_classification"] = getattr(
                p, "growth_classification", None
            )

    all_step_names = list(step_run_costs.keys())
    pattern_steps = (growth_steps | loop_variance_steps) & set(all_step_names)
    pattern_step_list = [sn for sn in all_step_names if sn in pattern_steps]

    # Fix 3: build per-run step cost map for whole-run sampling
    run_step_cost_map = _build_run_step_cost_map(runs)
    n_observed = len(runs)

    # Precompute non-pattern cost per observed run for fast sampling
    run_non_pattern_costs = [
        sum(rcm.get(sn, 0.0) for sn in all_step_names if sn not in pattern_steps)
        for rcm in run_step_cost_map
    ]

    # Fix 4: max observed iterations for extrapolation cap
    max_observed_iters = {sn: max(iters) for sn, iters in step_iterations.items()}

    n_monthly = daily_volume * n_days
    k_samples = max(min(n_monthly, 1000), 1)

    sim_run_costs: list[float] = []
    sim_daily_costs: list[float] = []
    sim_monthly_costs: list[float] = []
    sim_linear_monthly: list[float] = []
    sim_log_monthly: list[float] = []
    extrapolation_cap_count = 0

    for _ in range(n_simulations):
        k_avg: list[float] = []
        k_lin: list[float] = []
        k_log: list[float] = []
        sim_capped = False

        for _ki in range(k_samples):
            if bimodal_gmm is not None:
                # GMM model overrides per-step sampling for the entire run
                run_cost = _sample_from_gmm(
                    rng,
                    bimodal_gmm["means"],
                    bimodal_gmm["stds"],
                    bimodal_gmm["weights"],
                )
                k_avg.append(run_cost)
                k_lin.append(run_cost)
                k_log.append(run_cost)
                continue

            # Fix 3: pick a base observed run to preserve inter-step correlation
            base_run_idx = rng.randrange(n_observed)

            run_avg = run_non_pattern_costs[base_run_idx]
            run_lin = run_avg
            run_log = run_avg

            for sn in pattern_step_list:
                avg_c, lin_c, log_c, capped = _sample_step_cost(
                    sn,
                    rng,
                    step_run_costs,
                    step_iterations,
                    step_occurrence_costs,
                    step_growth,
                    growth_steps,
                    loop_variance_steps,
                    max_observed_iters,
                )
                if capped:
                    sim_capped = True
                run_avg += avg_c
                run_lin += lin_c
                run_log += log_c

            k_avg.append(run_avg)
            k_lin.append(run_lin)
            k_log.append(run_log)

        if sim_capped:
            extrapolation_cap_count += 1

        # Collect one run cost per simulation for per_run projection
        sim_run_costs.append(k_avg[0])
        if _debug_run_costs is not None:
            _debug_run_costs.extend(k_avg)

        # Fix 1: CLT-corrected aggregation from K samples to monthly total
        z = rng.gauss(0, 1)

        monthly_avg = _clt_aggregate(k_avg, n_monthly, z)
        monthly_lin = _clt_aggregate(k_lin, n_monthly, z)
        monthly_log = _clt_aggregate(k_log, n_monthly, z)

        daily_avg = monthly_avg / n_days if n_days > 0 else 0.0

        sim_monthly_costs.append(monthly_avg)
        sim_daily_costs.append(daily_avg)
        sim_linear_monthly.append(monthly_lin)
        sim_log_monthly.append(monthly_log)

    converged = True
    if n_simulations >= 10000:
        sorted_partial = sorted(sim_monthly_costs[:9000])
        sorted_full = sorted(sim_monthly_costs)
        p95_partial = percentile(sorted_partial, 95)
        p95_full = percentile(sorted_full, 95)
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

    monthly_proj = _build_percentile_projection(sim_monthly_costs)
    daily_proj = _build_percentile_projection(sim_daily_costs)
    per_run_proj = _build_percentile_projection(sim_run_costs)

    cvar_95 = compute_cvar(sim_monthly_costs, 0.05)
    cvar_99 = compute_cvar(sim_monthly_costs, 0.01) if n_simulations >= 100 else None

    return MonteCarloResult(
        n_simulations=n_simulations,
        monthly_projection=monthly_proj,
        daily_projection=daily_proj,
        per_run_projection=per_run_proj,
        linear_monthly=linear_proj,
        log_monthly=log_proj,
        convergence_check=converged,
        growth_model_delta=growth_delta,
        tail_inflation_factor=None,
        extrapolation_cap_warnings=extrapolation_cap_count,
        cvar_95=cvar_95,
        cvar_99=cvar_99,
    )
