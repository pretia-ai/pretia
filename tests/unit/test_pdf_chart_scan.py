"""Tests for ChartSpec validation and ScanParams randomization (pure-Python logic)."""

from __future__ import annotations

import random

import pytest

# scan_simulator.py imports PIL.Image at module level.
# Pillow must be installed (it's in the pdf-generation extra).
# Previous code mocked PIL here, but that corrupts PIL for matplotlib
# when pytest collects all tests together.
from pdfs.generators.rendering.chart_renderer import ChartSpec  # noqa: E402
from pdfs.generators.rendering.scan_simulator import (  # noqa: E402
    ScanParams,
    randomize_scan_params,
)


class TestChartSpecValidation:
    def test_valid_bar_chart(self):
        spec = ChartSpec(chart_type="bar", title="Test", data={"A": [1, 2, 3]})
        assert spec.chart_type == "bar"

    def test_valid_line_chart(self):
        spec = ChartSpec(chart_type="line", title="Test", data={"A": [1.0]})
        assert spec.chart_type == "line"

    def test_valid_pie_chart(self):
        spec = ChartSpec(chart_type="pie", title="Test", data={"A": [30, 70]})
        assert spec.chart_type == "pie"

    def test_invalid_chart_type_raises(self):
        with pytest.raises(ValueError, match="chart_type"):
            ChartSpec(chart_type="scatter", title="T", data={"A": [1]})

    def test_empty_data_raises(self):
        with pytest.raises(ValueError, match="data"):
            ChartSpec(chart_type="bar", title="T", data={})

    def test_default_figsize(self):
        spec = ChartSpec(chart_type="bar", title="T", data={"A": [1]})
        assert spec.figsize == (6.5, 4.0)

    def test_frozen(self):
        spec = ChartSpec(chart_type="bar", title="T", data={"A": [1]})
        with pytest.raises(AttributeError):
            spec.title = "New"


class TestScanParamsDefaults:
    def test_defaults_within_spec_ranges(self):
        p = ScanParams()
        assert 150 <= p.dpi <= 200
        assert 0.3 <= p.blur_sigma <= 0.8
        assert 3 <= p.noise_sigma <= 8
        assert 0.8 <= p.contrast_factor <= 1.2

    def test_frozen(self):
        p = ScanParams()
        with pytest.raises(AttributeError):
            p.dpi = 300


class TestRandomizeScanParams:
    def test_deterministic_with_seed(self):
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        p1 = randomize_scan_params(rng1)
        p2 = randomize_scan_params(rng2)
        assert p1 == p2

    def test_different_seeds_differ(self):
        p1 = randomize_scan_params(random.Random(1))
        p2 = randomize_scan_params(random.Random(999))
        assert p1 != p2

    def test_values_within_spec_ranges(self):
        rng = random.Random(42)
        for _ in range(50):
            p = randomize_scan_params(rng)
            assert 150 <= p.dpi <= 200
            assert 0.3 <= p.blur_sigma <= 0.8
            assert -2.0 <= p.rotation_degrees <= 2.0
            assert 3 <= p.noise_sigma <= 8
            assert 0.8 <= p.contrast_factor <= 1.2
