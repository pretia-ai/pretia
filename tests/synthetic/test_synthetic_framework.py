"""Unit tests for the synthetic distribution testing framework."""

from __future__ import annotations

import math

import pytest

from tests.synthetic.generators import (
    SyntheticWorkflow,
    generate_all_synthetic_workflows,
    generate_bimodal,
    generate_lognormal,
    generate_pareto,
    generate_uniform,
    generate_zero_inflated,
)
from tests.synthetic.runner import run_one


class TestLognormalGeneratorMean:
    def test_mean(self):
        wf = generate_lognormal(sigma=0.5, n=10000, seed=42)
        sample_mean = sum(wf.observed_costs) / len(wf.observed_costs)
        analytical_mean = math.exp(0.5**2 / 2)
        assert sample_mean == pytest.approx(analytical_mean, rel=0.05)


class TestLognormalGeneratorReproducible:
    def test_reproducible(self):
        wf1 = generate_lognormal(sigma=0.5, n=100, seed=42)
        wf2 = generate_lognormal(sigma=0.5, n=100, seed=42)
        assert wf1.observed_costs == wf2.observed_costs


class TestBimodalGeneratorMixingRatio:
    def test_mixing_ratio(self):
        wf = generate_bimodal(mixing=0.3, separation=10, n=10000, seed=42)
        midpoint = math.sqrt(10)
        expensive_count = sum(1 for c in wf.observed_costs if c > midpoint)
        ratio = expensive_count / len(wf.observed_costs)
        assert ratio == pytest.approx(0.3, abs=0.05)


class TestParetoGeneratorMinimum:
    def test_minimum(self):
        wf = generate_pareto(alpha=2.0, n=1000, seed=42)
        assert all(c >= 1.0 for c in wf.observed_costs)


class TestZeroInflatedGeneratorZeroFraction:
    def test_zero_fraction(self):
        wf = generate_zero_inflated(trigger_prob=0.1, n=10000, seed=42)
        near_zero = sum(1 for c in wf.observed_costs if c <= 0.002)
        ratio = near_zero / len(wf.observed_costs)
        assert ratio == pytest.approx(0.9, abs=0.05)


class TestUniformGeneratorRange:
    def test_range(self):
        wf = generate_uniform(n=1000, seed=42)
        assert all(0.5 <= c <= 1.5 for c in wf.observed_costs)


class TestSyntheticWorkflowStructure:
    def test_structure(self):
        wf = generate_lognormal(sigma=0.5, n=20, seed=42)
        assert isinstance(wf, SyntheticWorkflow)
        assert len(wf.observed_costs) == 20
        assert wf.true_p50 > 0
        assert wf.true_p95 > wf.true_p50
        assert wf.sample_size == 20
        assert wf.distribution_type == "lognormal"


class TestGenerateAllCount:
    def test_count(self):
        workflows = generate_all_synthetic_workflows(seed_base=42)
        assert len(workflows) >= 500


class TestRunnerReturnsResults:
    def test_returns_results(self):
        workflows = [
            generate_lognormal(0.5, 20, seed=1),
            generate_uniform(20, seed=2),
            generate_bimodal(0.3, 5, 20, seed=3),
        ]
        results = [run_one(wf, daily_volume=10) for wf in workflows]
        assert len(results) == 3
        for r in results:
            assert r.projected_p50 > 0


class TestUniformCalibrationPerfect:
    def test_uniform_near_perfect(self):
        wf = generate_uniform(n=300, seed=42)
        r = run_one(wf, daily_volume=10)
        true_monthly_mean = 10 * 30 * wf.true_mean
        if true_monthly_mean > 0:
            p50_ratio = r.projected_p50 / true_monthly_mean
            assert 0.90 <= p50_ratio <= 1.10
