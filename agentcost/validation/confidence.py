"""Confidence tier computation via jackknife+ conformal prediction intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.stats import StepStats


def compute_effective_sample_size(costs: list[float]) -> float:
    """Compute entropy-based effective sample size from the cost distribution."""
    n = len(costs)
    if n == 0:
        return 0.0

    min_c = min(costs)
    max_c = max(costs)
    if min_c == max_c:
        return 0.0

    n_bins = max(2, int(math.sqrt(n)))
    bin_width = (max_c - min_c) / n_bins

    counts = [0] * n_bins
    for c in costs:
        idx = int((c - min_c) / bin_width)
        idx = min(idx, n_bins - 1)
        counts[idx] += 1

    entropy = 0.0
    for cnt in counts:
        if cnt > 0:
            p = cnt / n
            entropy -= p * math.log(p)

    h_max = math.log(n_bins)
    if h_max == 0:
        return float(n)

    return n * (entropy / h_max)


def compute_conformal_interval(
    costs: list[float],
    alpha: float = 0.10,
) -> tuple[float, float, float]:
    """Jackknife+ conformal prediction interval.

    Returns (point_estimate, interval_lower, interval_upper) for per-run cost.
    Finite-sample coverage >= 1-2*alpha for any distribution.
    """
    n = len(costs)
    if n == 0:
        return 0.0, 0.0, 0.0
    if n == 1:
        return costs[0], costs[0], costs[0]

    mu_hat = sum(costs) / n

    residuals: list[float] = []
    for i in range(n):
        loo_sum = sum(costs) - costs[i]
        mu_minus_i = loo_sum / (n - 1)
        residuals.append(abs(costs[i] - mu_minus_i))

    sorted_r = sorted(residuals)
    q_index = min(math.ceil((1 - alpha) * (n + 1)) - 1, n - 1)
    q_index = max(0, q_index)
    q = sorted_r[q_index]

    return mu_hat, mu_hat - q, mu_hat + q


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Projection confidence derived from conformal prediction interval width."""

    tier: str
    conformal_lower: float
    conformal_upper: float
    monthly_lower: float
    monthly_upper: float
    alpha: float
    relative_width: float
    effective_sample_size: float
    patterns_detected: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "tier": self.tier,
            "conformal_lower": self.conformal_lower,
            "conformal_upper": self.conformal_upper,
            "monthly_lower": self.monthly_lower,
            "monthly_upper": self.monthly_upper,
            "alpha": self.alpha,
            "relative_width": self.relative_width,
            "effective_sample_size": self.effective_sample_size,
            "patterns_detected": list(self.patterns_detected),
        }


def _tier_from_relative_width(relative_width: float) -> str:
    if relative_width < 2.0:
        return "HIGH"
    if relative_width < 5.0:
        return "MODERATE"
    if relative_width < 10.0:
        return "LOW"
    return "VERY_LOW"


def compute_confidence(
    sample_size: int,
    step_stats: dict[str, StepStats],
    patterns: list[DetectedPattern],
    input_source: str = "auto-generate",
    run_costs: list[float] | None = None,
    traffic: int = 1000,
    n_days: int = 30,
    alpha: float = 0.10,
) -> ConfidenceResult:
    """Compute confidence tier from conformal prediction interval width."""
    pattern_names = list({p.pattern_type for p in patterns})

    if run_costs is not None and len(run_costs) >= 2:
        n_eff = compute_effective_sample_size(run_costs)
        mu, per_run_lo, per_run_hi = compute_conformal_interval(run_costs, alpha)
    else:
        n_eff = float(sample_size)
        mu = 0.0
        per_run_lo = 0.0
        per_run_hi = 0.0

    n_monthly = traffic * n_days
    per_run_se = (per_run_hi - mu) if mu > 0 else 0.0

    monthly_point = n_monthly * mu
    z = 1.645 if alpha >= 0.10 else 1.96
    monthly_lo = n_monthly * mu - z * math.sqrt(max(n_monthly, 0)) * per_run_se
    monthly_hi = n_monthly * mu + z * math.sqrt(max(n_monthly, 0)) * per_run_se
    monthly_lo = max(monthly_lo, 0.0)

    if monthly_point > 0:
        relative_width = (monthly_hi - monthly_lo) / monthly_point
    else:
        relative_width = float("inf") if per_run_se > 0 else 0.0

    tier = _tier_from_relative_width(relative_width)

    return ConfidenceResult(
        tier=tier,
        conformal_lower=per_run_lo,
        conformal_upper=per_run_hi,
        monthly_lower=monthly_lo,
        monthly_upper=monthly_hi,
        alpha=alpha,
        relative_width=round(relative_width, 4),
        effective_sample_size=round(n_eff, 2),
        patterns_detected=pattern_names,
    )
