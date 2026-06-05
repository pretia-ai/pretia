"""Generate synthetic per-run cost data from known distributions."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class SyntheticWorkflow:
    """A synthetic workflow with known distribution properties."""

    name: str
    distribution_type: str
    params: dict
    observed_costs: list[float]
    true_p50: float
    true_p95: float
    true_mean: float
    true_std: float
    sample_size: int


def _standard_normal(rng: random.Random) -> float:
    """Box-Muller transform for standard normal draws."""
    u1 = rng.random()
    u2 = rng.random()
    while u1 == 0:
        u1 = rng.random()
    return math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)


def _percentile(sorted_data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) via linear interpolation."""
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


def _reference_percentiles(
    draw_fn,
    n_ref: int = 100_000,
    seed: int = 999_999,
) -> tuple[float, float, float, float]:
    """Compute (p50, p95, mean, std) from a large reference sample."""
    rng = random.Random(seed)
    samples = sorted(draw_fn(rng) for _ in range(n_ref))
    mean = sum(samples) / n_ref
    var = sum((x - mean) ** 2 for x in samples) / (n_ref - 1)
    return (
        _percentile(samples, 50),
        _percentile(samples, 95),
        mean,
        math.sqrt(var),
    )


# ---------------------------------------------------------------------------
# Generator 1: Log-normal
# ---------------------------------------------------------------------------


def generate_lognormal(
    sigma: float,
    n: int,
    seed: int,
    mu: float = 0.0,
) -> SyntheticWorkflow:
    """Generate log-normal cost samples."""
    rng = random.Random(seed)
    costs = [math.exp(mu + sigma * _standard_normal(rng)) for _ in range(n)]

    true_p50 = math.exp(mu)
    true_p95 = math.exp(mu + 1.645 * sigma)
    true_mean = math.exp(mu + sigma**2 / 2)
    true_std = true_mean * math.sqrt(math.exp(sigma**2) - 1)

    return SyntheticWorkflow(
        name=f"lognormal_sigma_{sigma}_n_{n}",
        distribution_type="lognormal",
        params={"mu": mu, "sigma": sigma},
        observed_costs=costs,
        true_p50=true_p50,
        true_p95=true_p95,
        true_mean=true_mean,
        true_std=true_std,
        sample_size=n,
    )


# ---------------------------------------------------------------------------
# Generator 2: Bimodal
# ---------------------------------------------------------------------------


def generate_bimodal(
    mixing: float,
    separation: float,
    n: int,
    seed: int,
    sigma_within: float = 0.3,
) -> SyntheticWorkflow:
    """Generate bimodal cost samples (mixture of two log-normals)."""
    mu_cheap = 0.0
    mu_expensive = math.log(separation)

    rng = random.Random(seed)
    costs = []
    for _ in range(n):
        if rng.random() < mixing:
            costs.append(math.exp(mu_expensive + sigma_within * _standard_normal(rng)))
        else:
            costs.append(math.exp(mu_cheap + sigma_within * _standard_normal(rng)))

    def _draw(r: random.Random) -> float:
        if r.random() < mixing:
            return math.exp(mu_expensive + sigma_within * _standard_normal(r))
        return math.exp(mu_cheap + sigma_within * _standard_normal(r))

    p50, p95, mean, std = _reference_percentiles(_draw)

    return SyntheticWorkflow(
        name=f"bimodal_mix_{mixing}_sep_{separation}_sw_{sigma_within}_n_{n}",
        distribution_type="bimodal",
        params={
            "mixing": mixing,
            "separation": separation,
            "sigma_within": sigma_within,
        },
        observed_costs=costs,
        true_p50=p50,
        true_p95=p95,
        true_mean=mean,
        true_std=std,
        sample_size=n,
    )


# ---------------------------------------------------------------------------
# Generator 3: Pareto-tailed
# ---------------------------------------------------------------------------


def generate_pareto(
    alpha: float,
    n: int,
    seed: int,
    x_min: float = 1.0,
) -> SyntheticWorkflow:
    """Generate Pareto cost samples."""
    rng = random.Random(seed)
    costs = [x_min / (rng.random() ** (1.0 / alpha)) for _ in range(n)]

    true_p50 = x_min * 2 ** (1.0 / alpha)
    true_p95 = x_min * 20 ** (1.0 / alpha)
    if alpha > 1:
        true_mean = x_min * alpha / (alpha - 1)
    else:

        def _draw(r: random.Random) -> float:
            return x_min / (r.random() ** (1.0 / alpha))

        _, _, true_mean, _ = _reference_percentiles(_draw)

    if alpha > 2:
        true_std = x_min * math.sqrt(alpha / ((alpha - 1) ** 2 * (alpha - 2)))
    else:

        def _draw2(r: random.Random) -> float:
            return x_min / (r.random() ** (1.0 / alpha))

        _, _, _, true_std = _reference_percentiles(_draw2)

    return SyntheticWorkflow(
        name=f"pareto_alpha_{alpha}_n_{n}",
        distribution_type="pareto",
        params={"alpha": alpha, "x_min": x_min},
        observed_costs=costs,
        true_p50=true_p50,
        true_p95=true_p95,
        true_mean=true_mean,
        true_std=true_std,
        sample_size=n,
    )


# ---------------------------------------------------------------------------
# Generator 4: Zero-inflated
# ---------------------------------------------------------------------------


def generate_zero_inflated(
    trigger_prob: float,
    n: int,
    seed: int,
    near_zero: float = 0.001,
) -> SyntheticWorkflow:
    """Generate zero-inflated cost samples."""
    rng = random.Random(seed)
    costs = []
    for _ in range(n):
        if rng.random() < trigger_prob:
            costs.append(math.exp(0.0 + 0.5 * _standard_normal(rng)))
        else:
            costs.append(near_zero)

    def _draw(r: random.Random) -> float:
        if r.random() < trigger_prob:
            return math.exp(0.0 + 0.5 * _standard_normal(r))
        return near_zero

    p50, p95, mean, std = _reference_percentiles(_draw)

    return SyntheticWorkflow(
        name=f"zero_inflated_trigger_{trigger_prob}_n_{n}",
        distribution_type="zero_inflated",
        params={"trigger_prob": trigger_prob, "near_zero": near_zero},
        observed_costs=costs,
        true_p50=p50,
        true_p95=p95,
        true_mean=mean,
        true_std=std,
        sample_size=n,
    )


# ---------------------------------------------------------------------------
# Generator 5: Uniform (control)
# ---------------------------------------------------------------------------


def generate_uniform(
    n: int,
    seed: int,
    low: float = 0.5,
    high: float = 1.5,
) -> SyntheticWorkflow:
    """Generate uniform cost samples."""
    rng = random.Random(seed)
    costs = [low + rng.random() * (high - low) for _ in range(n)]

    true_p50 = (low + high) / 2
    true_p95 = low + 0.95 * (high - low)
    true_mean = (low + high) / 2
    true_std = (high - low) / math.sqrt(12)

    return SyntheticWorkflow(
        name=f"uniform_n_{n}",
        distribution_type="uniform",
        params={"low": low, "high": high},
        observed_costs=costs,
        true_p50=true_p50,
        true_p95=true_p95,
        true_mean=true_mean,
        true_std=true_std,
        sample_size=n,
    )


# ---------------------------------------------------------------------------
# Master generator
# ---------------------------------------------------------------------------

_SAMPLE_SIZES = [20, 50, 100, 150, 300]

_LOGNORMAL_SIGMAS = [
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
]

_BIMODAL_MIXINGS = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
_BIMODAL_SEPARATIONS = [2, 3, 5, 10, 15, 20]
_BIMODAL_SIGMA_WITHINS = [0.2, 0.5]

_PARETO_ALPHAS = [1.2, 1.5, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

_ZERO_INFLATED_PROBS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]


def generate_all_synthetic_workflows(
    seed_base: int = 42,
) -> list[SyntheticWorkflow]:
    """Generate 500+ synthetic workflows across all distribution types."""
    workflows: list[SyntheticWorkflow] = []
    counter = 0

    for sigma in _LOGNORMAL_SIGMAS:
        for n in _SAMPLE_SIZES:
            workflows.append(generate_lognormal(sigma, n, seed_base + counter))
            counter += 1

    for mixing in _BIMODAL_MIXINGS:
        for sep in _BIMODAL_SEPARATIONS:
            for sw in _BIMODAL_SIGMA_WITHINS:
                for n in _SAMPLE_SIZES:
                    workflows.append(
                        generate_bimodal(mixing, sep, n, seed_base + counter, sigma_within=sw)
                    )
                    counter += 1

    for alpha in _PARETO_ALPHAS:
        for n in _SAMPLE_SIZES:
            workflows.append(generate_pareto(alpha, n, seed_base + counter))
            counter += 1

    for prob in _ZERO_INFLATED_PROBS:
        for n in _SAMPLE_SIZES:
            workflows.append(generate_zero_inflated(prob, n, seed_base + counter))
            counter += 1

    for n in _SAMPLE_SIZES:
        workflows.append(generate_uniform(n, seed_base + counter))
        counter += 1

    return workflows
