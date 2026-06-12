"""Tests for backtest dataset persistence, indexing, and querying."""

from __future__ import annotations

import json
import time
from pathlib import Path

from tests.backtesting.dataset import (
    load_all_runs,
    load_index,
    load_run,
    query_runs,
    rebuild_index,
    save_backtest_run,
)

_SAMPLE_RUN = {
    "meta": {"backtest_id": "test123", "timestamp": "2026-06-07T12:00:00+00:00"},
    "workflows": {"W1": {"score": {}}},
    "aggregate": {"total_cost_usd": 100.0, "launch_gate": "PASSED"},
}


class TestSaveBacktestRun:
    """save_backtest_run() writes timestamped JSON files."""

    def test_save_backtest_run_creates_file(self, tmp_path: Path) -> None:
        """Saves a run and the file exists on disk."""
        path = save_backtest_run(_SAMPLE_RUN, dataset_dir=tmp_path)

        assert path.exists()
        assert path.parent == tmp_path

    def test_save_backtest_run_filename_format(self, tmp_path: Path) -> None:
        """Filename matches backtest_*.json pattern."""
        path = save_backtest_run(_SAMPLE_RUN, dataset_dir=tmp_path)

        assert path.name.startswith("backtest_")
        assert path.name.endswith(".json")


class TestLoadRun:
    """load_run() and load_all_runs() read persisted data."""

    def test_load_run_roundtrip(self, tmp_path: Path) -> None:
        """Save then load returns the same data."""
        path = save_backtest_run(_SAMPLE_RUN, dataset_dir=tmp_path)
        loaded = load_run(path)

        assert loaded["meta"]["backtest_id"] == "test123"
        assert loaded["workflows"] == _SAMPLE_RUN["workflows"]
        assert loaded["aggregate"] == _SAMPLE_RUN["aggregate"]

    def test_load_all_runs_multiple(self, tmp_path: Path) -> None:
        """Save 2 runs; load_all returns both sorted newest first."""
        run_a = {
            "meta": {
                "backtest_id": "run-a",
                "timestamp": "2026-06-01T00:00:00+00:00",
            },
            "workflows": {"W1": {}},
        }
        save_backtest_run(run_a, dataset_dir=tmp_path)

        # Ensure different filename timestamp by sleeping briefly.
        time.sleep(1.1)

        run_b = {
            "meta": {
                "backtest_id": "run-b",
                "timestamp": "2026-06-02T00:00:00+00:00",
            },
            "workflows": {"W1": {}},
        }
        save_backtest_run(run_b, dataset_dir=tmp_path)

        runs = load_all_runs(dataset_dir=tmp_path)

        assert len(runs) == 2
        # Newest first (by filename, reverse sorted)
        assert runs[0]["meta"]["backtest_id"] == "run-b"
        assert runs[1]["meta"]["backtest_id"] == "run-a"

    def test_load_all_runs_empty(self, tmp_path: Path) -> None:
        """Empty directory returns empty list."""
        result = load_all_runs(dataset_dir=tmp_path)

        assert result == []


class TestRebuildIndex:
    """rebuild_index() creates and populates the index file."""

    def test_rebuild_index_creates_file(self, tmp_path: Path) -> None:
        """Creates dataset_index.json in the target directory."""
        save_backtest_run(_SAMPLE_RUN, dataset_dir=tmp_path)
        index_path = rebuild_index(dataset_dir=tmp_path)

        assert index_path.exists()
        assert index_path.name == "dataset_index.json"

    def test_rebuild_index_content(self, tmp_path: Path) -> None:
        """Index entries have expected fields."""
        save_backtest_run(_SAMPLE_RUN, dataset_dir=tmp_path)
        index_path = rebuild_index(dataset_dir=tmp_path)

        entries = json.loads(index_path.read_text())

        assert len(entries) == 1
        entry = entries[0]
        expected_fields = {
            "backtest_id",
            "timestamp",
            "filename",
            "launch_gate",
            "total_cost_usd",
            "workflow_count",
        }
        assert expected_fields.issubset(set(entry.keys()))
        assert entry["backtest_id"] == "test123"
        assert entry["workflow_count"] == 1


class TestQueryRuns:
    """query_runs() filters by workflow and timestamp."""

    def test_query_runs_by_workflow(self, tmp_path: Path) -> None:
        """Filters by workflow key in workflows dict."""
        run_with_w1 = {
            "meta": {
                "backtest_id": "has-w1",
                "timestamp": "2026-06-07T12:00:00+00:00",
            },
            "workflows": {"W1": {"score": {}}},
        }
        run_without_w1 = {
            "meta": {
                "backtest_id": "no-w1",
                "timestamp": "2026-06-07T13:00:00+00:00",
            },
            "workflows": {"W2": {"score": {}}},
        }
        save_backtest_run(run_with_w1, dataset_dir=tmp_path)
        time.sleep(1.1)
        save_backtest_run(run_without_w1, dataset_dir=tmp_path)

        result = query_runs(dataset_dir=tmp_path, workflow="W1")

        assert len(result) == 1
        assert result[0]["meta"]["backtest_id"] == "has-w1"

    def test_query_runs_by_date(self, tmp_path: Path) -> None:
        """Filters by after/before timestamps."""
        run_early = {
            "meta": {
                "backtest_id": "early",
                "timestamp": "2026-06-01T00:00:00+00:00",
            },
            "workflows": {},
        }
        run_late = {
            "meta": {
                "backtest_id": "late",
                "timestamp": "2026-06-10T00:00:00+00:00",
            },
            "workflows": {},
        }
        save_backtest_run(run_early, dataset_dir=tmp_path)
        time.sleep(1.1)
        save_backtest_run(run_late, dataset_dir=tmp_path)

        result = query_runs(
            dataset_dir=tmp_path,
            after="2026-06-05T00:00:00+00:00",
        )

        assert len(result) == 1
        assert result[0]["meta"]["backtest_id"] == "late"


class TestLoadIndex:
    """load_index() reads the index, auto-rebuilding if needed."""

    def test_load_index_auto_rebuild(self, tmp_path: Path) -> None:
        """If index missing, auto-rebuilds from existing run files."""
        save_backtest_run(_SAMPLE_RUN, dataset_dir=tmp_path)

        # No explicit rebuild_index() call — load_index should handle it.
        index_path = tmp_path / "dataset_index.json"
        assert not index_path.exists()

        entries = load_index(dataset_dir=tmp_path)

        assert index_path.exists()
        assert len(entries) == 1
        assert entries[0]["backtest_id"] == "test123"
