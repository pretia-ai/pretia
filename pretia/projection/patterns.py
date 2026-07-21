"""Detect non-linear cost patterns: context growth, loop variance, high token variance."""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import MODEL_CACHE_HIT_PRICING, resolve_model
from pretia.projection.stats import ProfilingStats, compute_stats, percentile, robust_cv

logger = logging.getLogger(__name__)

_T_CRITICAL_005 = {
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    20: 2.086,
    25: 2.060,
    30: 2.042,
    40: 2.021,
    50: 2.009,
    60: 2.000,
    80: 1.990,
    100: 1.984,
    120: 1.980,
}


@dataclass(frozen=True, slots=True)
class DetectedPattern:
    """One detected non-linear cost pattern."""

    pattern_type: str
    step_name: str
    severity: str
    evidence: dict[str, Any]
    description: str
    growth_type: str | None = None
    pearson_r_squared: float | None = None
    spearman_rho_squared: float | None = None
    pearson_significant: bool | None = None
    spearman_significant: bool | None = None
    nonlinearity_gap: float | None = None
    power_law_alpha: float | None = None
    power_law_base: float | None = None
    growth_classification: str | None = None
    variance_percentile_used: int | None = None
    step_count_cv: float | None = None
    step_count_min: int | None = None
    step_count_max: int | None = None
    step_count_mean: float | None = None
    bimodal_bic_delta: float | None = None
    bimodal_modes: list[dict[str, float]] | None = None
    gmm_means: list[float] | None = None
    gmm_stds: list[float] | None = None
    gmm_weights: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pattern_type": self.pattern_type,
            "step_name": self.step_name,
            "severity": self.severity,
            "evidence": self.evidence,
            "description": self.description,
            "growth_type": self.growth_type,
            "pearson_r_squared": self.pearson_r_squared,
            "spearman_rho_squared": self.spearman_rho_squared,
            "pearson_significant": self.pearson_significant,
            "spearman_significant": self.spearman_significant,
            "nonlinearity_gap": self.nonlinearity_gap,
            "power_law_alpha": self.power_law_alpha,
            "power_law_base": self.power_law_base,
            "growth_classification": self.growth_classification,
            "variance_percentile_used": self.variance_percentile_used,
            "step_count_cv": self.step_count_cv,
            "step_count_min": self.step_count_min,
            "step_count_max": self.step_count_max,
            "step_count_mean": self.step_count_mean,
            "bimodal_bic_delta": self.bimodal_bic_delta,
            "bimodal_modes": self.bimodal_modes,
            "gmm_means": self.gmm_means,
            "gmm_stds": self.gmm_stds,
            "gmm_weights": self.gmm_weights,
        }


def _pearson_r(
    xs: list[float],
    ys: list[float],
) -> tuple[float, float]:
    """Return (r, slope) for two equal-length lists, or (0, 0) if degenerate."""
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
    return r, slope


def _rank(values: list[float]) -> list[float]:
    """Rank values 1-based, averaging ties."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = sum(range(i + 1, j + 2)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def _is_significant(r_value: float, n: int) -> bool:
    """Check if correlation is significant at p < 0.05 (two-tailed)."""
    df = n - 2
    if df < 3:
        return False
    r_abs = abs(r_value)
    if r_abs >= 1.0:
        return True
    t_stat = r_abs * math.sqrt(df) / math.sqrt(1 - r_abs * r_abs)

    if df > 120:
        t_crit = 1.960
    else:
        t_crit = 1.960
        for k in sorted(_T_CRITICAL_005):
            if k <= df:
                t_crit = _T_CRITICAL_005[k]
    return t_stat > t_crit


def _log_log_regression(
    xs: list[float],
    ys: list[float],
) -> tuple[float, float]:
    """Compute power-law exponent via OLS on log-transformed data.

    Returns (alpha, base) where context ≈ base × iteration^alpha.
    """
    pairs = [(math.log(x), math.log(y)) for x, y in zip(xs, ys, strict=True) if x > 0 and y > 0]
    if len(pairs) < 2:
        return 1.0, 1.0
    lxs = [p[0] for p in pairs]
    lys = [p[1] for p in pairs]
    n = len(pairs)
    mean_lx = sum(lxs) / n
    mean_ly = sum(lys) / n
    num = sum((lx - mean_lx) * (ly - mean_ly) for lx, ly in zip(lxs, lys, strict=True))
    den = sum((lx - mean_lx) ** 2 for lx in lxs)
    if den == 0:
        return 1.0, math.exp(mean_ly)
    alpha = num / den
    intercept = mean_ly - alpha * mean_lx
    base = math.exp(intercept)
    return alpha, base


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
        if len(pairs) < 5:
            continue
        xs = [p[0] for p in pairs]
        ys = [p[1] for p in pairs]
        n = len(pairs)

        pearson_r_val, slope = _pearson_r(xs, ys)
        pearson_r_sq = pearson_r_val * pearson_r_val
        pearson_sig = _is_significant(pearson_r_val, n) if pearson_r_val > 0 else False

        rank_x = _rank(xs)
        rank_y = _rank(ys)
        spearman_r_val, _ = _pearson_r(rank_x, rank_y)
        spearman_r_sq = spearman_r_val * spearman_r_val
        spearman_sig = _is_significant(spearman_r_val, n) if spearman_r_val > 0 else False

        pearson_passes = pearson_r_sq > 0.7 and pearson_sig and pearson_r_val > 0
        spearman_passes = spearman_r_sq > 0.7 and spearman_sig and spearman_r_val > 0

        if not pearson_passes and not spearman_passes:
            continue

        nonlinearity_gap = abs(spearman_r_val - pearson_r_val)

        growth_type: str
        power_law_alpha: float | None = None
        power_law_base: float | None = None
        growth_classification: str | None = None

        if pearson_passes:
            growth_type = "linear"
        else:
            growth_type = "nonlinear"
            alpha, pbase = _log_log_regression(xs, ys)
            power_law_alpha = alpha
            power_law_base = pbase
            growth_classification = "sub_linear" if alpha < 1 else "super_linear"

        sig_r_sqs: list[float] = []
        if pearson_sig and pearson_r_val > 0:
            sig_r_sqs.append(pearson_r_sq)
        if spearman_sig and spearman_r_val > 0:
            sig_r_sqs.append(spearman_r_sq)
        max_sig_r_sq = max(sig_r_sqs) if sig_r_sqs else 0
        severity = "danger" if max_sig_r_sq > 0.85 else "warning"

        first_iter_contexts = [y for x, y in pairs if x == 1.0]
        max_iter = max(xs)
        last_iter_contexts = [y for x, y in pairs if x == max_iter]
        mean_first = (
            sum(first_iter_contexts) / len(first_iter_contexts) if first_iter_contexts else ys[0]
        )
        mean_last = (
            sum(last_iter_contexts) / len(last_iter_contexts) if last_iter_contexts else ys[-1]
        )
        ratio = mean_last / mean_first if mean_first > 0 else 0.0

        description = (
            f"Context grows by ~{slope:.0f} tokens per iteration in step "
            f"'{step_name}' (r²={pearson_r_sq:.2f}, ρ²={spearman_r_sq:.2f}). "
            f"At iteration {int(max_iter)}, context is {ratio:.1f}x the initial size."
        )
        if growth_type == "nonlinear":
            description += (
                f" Non-linear {growth_classification} growth detected (α={power_law_alpha:.2f})."
            )

        patterns.append(
            DetectedPattern(
                pattern_type="context_growth",
                step_name=step_name,
                severity=severity,
                evidence={
                    "r_squared": round(pearson_r_sq, 4),
                    "slope": round(slope, 2),
                    "mean_context_first": round(mean_first, 2),
                    "mean_context_last": round(mean_last, 2),
                    "n_datapoints": n,
                },
                description=description,
                growth_type=growth_type,
                pearson_r_squared=round(pearson_r_sq, 4),
                spearman_rho_squared=round(spearman_r_sq, 4),
                pearson_significant=pearson_sig,
                spearman_significant=spearman_sig,
                nonlinearity_gap=round(nonlinearity_gap, 4),
                power_law_alpha=round(power_law_alpha, 4) if power_law_alpha is not None else None,
                power_law_base=round(power_law_base, 4) if power_law_base is not None else None,
                growth_classification=growth_classification,
            )
        )
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
        cv = robust_cv([float(x) for x in iters])
        if cv <= 0.5:
            continue

        max_i = max(iters)
        min_i = min(iters)
        ratio = max_i / mean_iter if mean_iter > 0 else 0.0
        severity = "danger" if cv > 1.0 or max_i > 3 * mean_iter else "warning"

        patterns.append(
            DetectedPattern(
                pattern_type="loop_count_variance",
                step_name=step_name,
                severity=severity,
                evidence={
                    "cv": round(cv, 4),
                    "mean_iterations": round(mean_iter, 2),
                    "min_iterations": min_i,
                    "max_iterations": max_i,
                },
                description=(
                    f"Loop count for step '{step_name}' varies from {min_i} to {max_i} "
                    f"iterations (mean={mean_iter:.1f}, CV={cv:.2f}). Worst-case runs "
                    f"cost ~{ratio:.1f}x the average."
                ),
            )
        )
    return patterns


def _detect_high_token_variance(
    stats: ProfilingStats,
) -> list[DetectedPattern]:
    """Detect steps with heavy-tailed token or cost distributions."""
    n_runs = stats.total_runs
    use_p90 = n_runs < 30

    patterns: list[DetectedPattern] = []
    for step_name, ss in stats.step_stats.items():
        p50_tok = ss.total_tokens.p50
        p50_cost = ss.cost.p50

        if use_p90:
            tail_tok = ss.total_tokens.p90
            tail_cost = ss.cost.p90
            percentile_used = 90
        else:
            tail_tok = ss.total_tokens.p95
            tail_cost = ss.cost.p95
            percentile_used = 95

        ratio_tok = tail_tok / p50_tok if p50_tok > 0 else 0.0
        ratio_cost = tail_cost / p50_cost if p50_cost > 0 else 0.0
        ratio = max(ratio_tok, ratio_cost)

        if ratio <= 3.0:
            continue

        severity = "danger" if ratio > 5.0 else "warning"
        patterns.append(
            DetectedPattern(
                pattern_type="high_token_variance",
                step_name=step_name,
                severity=severity,
                evidence={
                    "p95_p50_ratio_tokens": round(ratio_tok, 4),
                    "p95_p50_ratio_cost": round(ratio_cost, 4),
                    "p50_tokens": round(p50_tok, 2),
                    "p95_tokens": round(tail_tok, 2),
                    "p50_cost": round(p50_cost, 6),
                    "p95_cost": round(tail_cost, 6),
                },
                description=(
                    f"Step '{step_name}' has high token variance: p{percentile_used} is "
                    f"{ratio_tok:.1f}x the median ({p50_tok:.0f} vs {tail_tok:.0f} total "
                    f"tokens). Average-based projection will underestimate tail costs."
                ),
                variance_percentile_used=percentile_used,
            )
        )
    return patterns


def _detect_step_count_variance(
    runs: list[list[StepRecord]],
) -> list[DetectedPattern]:
    """Detect workflows where the number of active steps varies across runs."""
    if len(runs) < 2:
        return []

    active_counts: list[int] = []
    for run in runs:
        step_tokens: dict[str, int] = defaultdict(int)
        for rec in run:
            step_tokens[rec.step_name] += rec.input_tokens + rec.output_tokens
        active = sum(1 for t in step_tokens.values() if t > 0)
        active_counts.append(active)

    n = len(active_counts)
    mean_ac = sum(active_counts) / n
    if mean_ac == 0:
        return []

    cv = robust_cv([float(x) for x in active_counts])

    max_ac = max(active_counts)
    min_ac = min(active_counts)

    if cv > 0.6 or (min_ac > 0 and max_ac > 2 * min_ac):
        severity = "danger"
    elif cv > 0.3:
        severity = "warning"
    else:
        return []

    return [
        DetectedPattern(
            pattern_type="step_count_variance",
            step_name="_workflow_",
            severity=severity,
            evidence={
                "cv": round(cv, 4),
                "mean_active_steps": round(mean_ac, 2),
                "min_active_steps": min_ac,
                "max_active_steps": max_ac,
            },
            description=(
                f"Active step count varies from {min_ac} to {max_ac} across runs "
                f"(mean={mean_ac:.1f}, CV={cv:.2f}). Routing variance affects cost distribution."
            ),
            step_count_cv=round(cv, 4),
            step_count_min=min_ac,
            step_count_max=max_ac,
            step_count_mean=round(mean_ac, 2),
        )
    ]


def _detect_bimodality(
    runs: list[list[StepRecord]],
    stats: ProfilingStats,
) -> list[DetectedPattern]:
    """Detect bimodal cost distributions via 2-component GMM on log-costs."""
    if len(stats.run_stats) < 15:
        return []

    try:
        from sklearn.mixture import GaussianMixture  # noqa: I001
    except ImportError:
        logger.debug("sklearn not installed — skipping bimodality detection")
        return []

    import numpy as np  # noqa: I001

    costs = [rs.total_cost for rs in stats.run_stats]
    positive_costs = [c for c in costs if c > 0]
    zero_count = len(costs) - len(positive_costs)
    n = len(costs)

    if zero_count > 0 and len(positive_costs) >= 2:
        zero_prop = zero_count / n
        pos_prop = 1 - zero_prop
        pos_sorted = sorted(positive_costs)
        pos_median = pos_sorted[len(pos_sorted) // 2]
        modes = [
            {
                "proportion": round(zero_prop, 3),
                "mean_cost": 0.0,
                "median_cost": 0.0,
                "min_cost": 0.0,
                "max_cost": 0.0,
            },
            {
                "proportion": round(pos_prop, 3),
                "mean_cost": round(sum(positive_costs) / len(positive_costs), 6),
                "median_cost": round(pos_median, 6),
                "min_cost": round(min(positive_costs), 6),
                "max_cost": round(max(positive_costs), 6),
            },
        ]
        return [
            DetectedPattern(
                pattern_type="bimodality",
                step_name="_workflow_",
                severity="warning",
                evidence={"n_runs": n, "zero_cost_runs": zero_count},
                description=(
                    f"Cost distribution is bimodal: {zero_count} runs have zero cost, "
                    f"{len(positive_costs)} runs have positive cost "
                    f"(mean=${sum(positive_costs) / len(positive_costs):.4f}/run)."
                ),
                bimodal_bic_delta=1e10,
                bimodal_modes=modes,
            )
        ]

    if len(positive_costs) < 15:
        return []

    log_costs = [math.log(c) for c in positive_costs]
    log_array = np.array(log_costs).reshape(-1, 1)

    gmm1 = GaussianMixture(n_components=1, random_state=42).fit(log_array)
    gmm2 = GaussianMixture(n_components=2, random_state=42).fit(log_array)
    bic1 = gmm1.bic(log_array)
    bic2 = gmm2.bic(log_array)
    bic_delta = bic1 - bic2

    if bic_delta <= 6:
        return []

    labels = gmm2.predict(log_array)
    means_log = gmm2.means_.flatten()
    covs = gmm2.covariances_.flatten()
    stds_log = [float(math.sqrt(c)) for c in covs]
    weights = gmm2.weights_

    order = np.argsort(means_log)
    modes: list[dict[str, float]] = []
    ordered_means: list[float] = []
    ordered_stds: list[float] = []
    ordered_weights: list[float] = []
    for idx in order:
        ordered_means.append(float(means_log[idx]))
        ordered_stds.append(stds_log[idx])
        ordered_weights.append(float(weights[idx]))
        member_costs = [positive_costs[i] for i in range(len(positive_costs)) if labels[i] == idx]
        if not member_costs:
            continue
        member_sorted = sorted(member_costs)
        modes.append(
            {
                "proportion": round(float(weights[idx]), 3),
                "mean_cost": round(sum(member_costs) / len(member_costs), 6),
                "median_cost": round(member_sorted[len(member_sorted) // 2], 6),
                "min_cost": round(min(member_costs), 6),
                "max_cost": round(max(member_costs), 6),
            }
        )

    return [
        DetectedPattern(
            pattern_type="bimodality",
            step_name="_workflow_",
            severity="warning",
            evidence={
                "n_runs": n,
                "bic_1component": round(bic1, 2),
                "bic_2component": round(bic2, 2),
            },
            description=(
                f"Cost distribution is bimodal (BIC Δ={bic_delta:.1f}). "
                f"Mode A: {modes[0]['proportion']:.0%} of runs at "
                f"${modes[0]['mean_cost']:.4f}/run. "
                f"Mode B: {modes[1]['proportion']:.0%} of runs at "
                f"${modes[1]['mean_cost']:.4f}/run."
                if len(modes) >= 2
                else f"Cost distribution shows bimodal structure (BIC Δ={bic_delta:.1f})."
            ),
            bimodal_bic_delta=round(bic_delta, 2),
            bimodal_modes=modes,
            gmm_means=ordered_means,
            gmm_stds=ordered_stds,
            gmm_weights=ordered_weights,
        )
    ]


def _detect_cache_utilization(
    runs: list[list[StepRecord]],
) -> list[DetectedPattern]:
    """Detect steps where cache utilization is low despite model support."""
    step_cache: dict[str, dict[str, list]] = defaultdict(
        lambda: {"hit": [], "miss": [], "models": []}
    )

    for run in runs:
        for rec in run:
            if rec.cache_hit_tokens is None and rec.cache_miss_tokens is None:
                continue
            hit = rec.cache_hit_tokens or 0
            miss = rec.cache_miss_tokens or 0
            step_cache[rec.step_name]["hit"].append(hit)
            step_cache[rec.step_name]["miss"].append(miss)
            step_cache[rec.step_name]["models"].append(rec.model)

    patterns: list[DetectedPattern] = []
    for step_name, data in step_cache.items():
        hits = data["hit"]
        misses = data["miss"]
        models = data["models"]
        if not hits:
            continue

        total_hit = sum(hits)
        total_miss = sum(misses)
        total = total_hit + total_miss
        if total == 0:
            continue

        cache_hit_ratio = total_hit / total

        model = models[0]
        canonical = model
        try:
            canonical = resolve_model(model)
        except ValueError:
            pass
        supports_caching = canonical in MODEL_CACHE_HIT_PRICING

        if not supports_caching or cache_hit_ratio >= 0.1:
            continue

        patterns.append(
            DetectedPattern(
                pattern_type="cache_utilization_opportunity",
                step_name=step_name,
                severity="warning",
                evidence={
                    "cache_hit_ratio": round(cache_hit_ratio, 4),
                    "total_cache_miss_tokens": total_miss,
                    "model": model,
                },
                description=(
                    f"Step '{step_name}' has low cache utilization "
                    f"(hit ratio={cache_hit_ratio:.1%}) with {model}, "
                    f"which supports prompt caching. Consider restructuring "
                    f"prompts to improve cache hit rate."
                ),
            )
        )
    return patterns


_FRAMEWORK_INTERNAL_NODES = frozenset({"__start__", "__end__", "_route"})


def _detect_zero_execution_steps(
    runs: list[list[StepRecord]],
    graph_steps: list[str] | None = None,
) -> list[DetectedPattern]:
    """Detect graph steps that never executed across any observed run."""
    if graph_steps is None:
        return []

    observed_steps: set[str] = set()
    for run in runs:
        for rec in run:
            observed_steps.add(rec.step_name)

    patterns: list[DetectedPattern] = []
    for step_name in graph_steps:
        if step_name in observed_steps or step_name in _FRAMEWORK_INTERNAL_NODES:
            continue
        patterns.append(
            DetectedPattern(
                pattern_type="zero_execution_step",
                step_name=step_name,
                severity="warning",
                evidence={
                    "step_name": step_name,
                    "total_runs": len(runs),
                    "graph_steps": len(graph_steps),
                    "observed_steps": len(observed_steps),
                },
                description=(
                    f"Graph step '{step_name}' was never executed across "
                    f"{len(runs)} profiling runs. This step may represent a "
                    f"rare code path whose cost is not captured in projections."
                ),
            )
        )
    return patterns


def _detect_output_token_budget(
    runs: list[list[StepRecord]],
) -> list[DetectedPattern]:
    """Detect steps where max_tokens_setting is too loose or risks truncation."""
    step_outputs: dict[str, list[tuple[int, int]]] = defaultdict(list)

    for run in runs:
        for rec in run:
            if rec.max_tokens_setting is None:
                continue
            step_outputs[rec.step_name].append((rec.output_tokens, rec.max_tokens_setting))

    patterns: list[DetectedPattern] = []
    for step_name, pairs in step_outputs.items():
        if not pairs:
            continue

        output_vals = sorted(p[0] for p in pairs)
        max_tokens_setting = pairs[0][1]

        median_output = int(percentile(output_vals, 50))
        p95_output = int(percentile(output_vals, 95))

        if max_tokens_setting > 4 * median_output and median_output > 0:
            suggested = int(round(1.5 * p95_output / 256)) * 256
            if suggested < 256:
                suggested = 256
            patterns.append(
                DetectedPattern(
                    pattern_type="output_token_budget",
                    step_name=step_name,
                    severity="warning",
                    evidence={
                        "max_tokens_setting": max_tokens_setting,
                        "median_output_tokens": median_output,
                        "p95_output_tokens": p95_output,
                        "suggested_max_tokens": suggested,
                        "budget_issue": "too_loose",
                    },
                    description=(
                        f"Step '{step_name}' has max_tokens="
                        f"{max_tokens_setting} but median output is only "
                        f"{median_output} tokens "
                        f"({max_tokens_setting / median_output:.0f}x "
                        f"overhead). Consider reducing to ~{suggested} "
                        f"(1.5x p95)."
                    ),
                )
            )

        if median_output > 0.9 * max_tokens_setting:
            patterns.append(
                DetectedPattern(
                    pattern_type="output_token_budget",
                    step_name=step_name,
                    severity="warning",
                    evidence={
                        "max_tokens_setting": max_tokens_setting,
                        "median_output_tokens": median_output,
                        "p95_output_tokens": p95_output,
                        "budget_issue": "possible_truncation",
                    },
                    description=(
                        f"Step '{step_name}' may be truncating output: "
                        f"median output ({median_output} tokens) is "
                        f"{median_output / max_tokens_setting:.0%} of "
                        f"max_tokens={max_tokens_setting}. Increase the "
                        f"budget to avoid quality loss."
                    ),
                )
            )
    return patterns


# Which detectors require custom cost models in Monte Carlo vs. whole-run resampling.
#
# COST ADJUSTMENT (custom MC model needed):
#   context_growth (linear)    — linear + logarithmic average model
#   context_growth (nonlinear) — power-law model (sub-linear or super-linear)
#   loop_count_variance        — sample iteration count × per-iteration costs
#   bimodality (GMM)           — sample from fitted 2-component log-normal mixture
#
# NO COST ADJUSTMENT (whole-run resampling handles them):
#   high_token_variance            — triggers MC mode; reporting; confidence deduction
#   step_count_variance            — triggers MC mode; whole-run sampling; reporting
#   cache_utilization_opportunity  — reporting only; informational for recommendations
#   zero_execution_step            — reporting only; informational for coverage analysis
#   output_token_budget            — reporting only; informational for recommendations
#
# When bimodality + context_growth/loop_variance co-occur, the GMM model takes
# priority for per-run total cost. Per-step growth models are skipped.
# This is a v1 simplification.
#
# Detectors in the "no cost adjustment" group affect:
#   1. Mode selection: their presence forces Monte Carlo instead of linear projection.
#   2. Confidence scoring: they may trigger deductions (handled in confidence.py).
#   3. Reporting: they provide metadata for richer user-facing output.
#   4. Recommendations: they inform recommendation generation.


def detect_patterns(
    runs: list[list[StepRecord]],
    stats: ProfilingStats | None = None,
    graph_steps: list[str] | None = None,
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
    patterns.extend(_detect_step_count_variance(runs))
    patterns.extend(_detect_bimodality(runs, stats))
    patterns.extend(_detect_cache_utilization(runs))
    patterns.extend(_detect_zero_execution_steps(runs, graph_steps))
    patterns.extend(_detect_output_token_budget(runs))

    severity_order = {"danger": 0, "warning": 1}
    patterns.sort(key=lambda p: severity_order.get(p.severity, 2))
    return patterns
