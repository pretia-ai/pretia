"""Tests for visualization utility functions."""

from __future__ import annotations

import pytest

from visualization.utils import add_caption, discover_results, format_workflow_label, save_figure


class TestDiscoverResults:
    def test_finds_comparison_files(self, tmp_path):
        (tmp_path / "W1_comparison_A.json").write_text("{}")
        (tmp_path / "W1_comparison_B.json").write_text("{}")
        (tmp_path / "W13_comparison_C.json").write_text("{}")
        results = discover_results(tmp_path)
        assert "W1" in results
        assert set(results["W1"].keys()) == {"A", "B"}
        assert "W13" in results
        assert "C" in results["W13"]

    def test_finds_legacy_files(self, tmp_path):
        (tmp_path / "W1_synth20.json").write_text("{}")
        (tmp_path / "W1_real500.json").write_text("{}")
        results = discover_results(tmp_path)
        assert "synth20" in results["W1"]
        assert "real500" in results["W1"]

    def test_empty_dir(self, tmp_path):
        results = discover_results(tmp_path)
        assert results == {}

    def test_nonexistent_dir(self, tmp_path):
        results = discover_results(tmp_path / "nope")
        assert results == {}


class TestSaveFigure:
    def test_creates_png_and_pdf(self, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])
        paths = save_figure(fig, tmp_path, "test_fig")
        plt.close(fig)
        assert len(paths) == 2
        assert (tmp_path / "test_fig.png").exists()
        assert (tmp_path / "test_fig.pdf").exists()

    def test_creates_output_dir(self, tmp_path):
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.plot([1, 2])
        nested = tmp_path / "sub" / "dir"
        save_figure(fig, nested, "nested_fig", formats=("png",))
        plt.close(fig)
        assert nested.exists()
        assert (nested / "nested_fig.png").exists()


class TestAddCaption:
    def test_adds_text(self):
        matplotlib = pytest.importorskip("matplotlib")

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        add_caption(fig, "This is a test caption.")
        texts = [t.get_text() for t in fig.texts]
        assert any("test caption" in t for t in texts)
        plt.close(fig)


class TestFormatWorkflowLabel:
    def test_shortens_compound_name(self):
        assert format_workflow_label("W1-support-simple") == "W1"

    def test_keeps_short_name(self):
        assert format_workflow_label("W13") == "W13"

    def test_truncates_long_name(self):
        result = format_workflow_label("some-very-long-name")
        assert len(result) <= 6
