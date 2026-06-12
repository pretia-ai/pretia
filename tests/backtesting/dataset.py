"""Manage the append-only backtest dataset.

Each backtest run is persisted as a timestamped JSON file in
``tests/backtesting/dataset/``. An index file enables fast querying
without scanning every result file.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent / "dataset"

_FILENAME_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


def save_backtest_run(
    run_data: dict[str, Any],
    dataset_dir: Path | None = None,
) -> Path:
    """Write *run_data* as a timestamped JSON file in the dataset directory.

    The *run_data* dict must contain a ``meta.backtest_id`` field.
    Creates the dataset directory if it does not already exist.
    Returns the path to the written file.
    """
    target_dir = dataset_dir or DATASET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    meta = run_data.get("meta", {})
    if "backtest_id" not in meta:
        msg = "run_data must contain a 'meta.backtest_id' field"
        raise ValueError(msg)

    timestamp = datetime.now(UTC).strftime(_FILENAME_TIMESTAMP_FMT)
    filename = f"backtest_{timestamp}.json"
    path = target_dir / filename

    path.write_text(json.dumps(run_data, indent=2, default=str))
    logger.info("Saved backtest run to %s", path)
    return path


def rebuild_index(dataset_dir: Path | None = None) -> Path:
    """Scan all backtest JSON files and rebuild the dataset index.

    Produces ``dataset_index.json`` with one entry per file containing:
    backtest_id, timestamp, filename, launch_gate status, total_cost_usd,
    and workflow_count.  Returns the path to the index file.
    """
    target_dir = dataset_dir or DATASET_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("backtest_*.json")):
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path.name, exc)
            continue

        meta = data.get("meta", {})
        workflows = data.get("workflows", {})

        entries.append(
            {
                "backtest_id": meta.get("backtest_id"),
                "timestamp": meta.get("timestamp"),
                "filename": path.name,
                "launch_gate": meta.get("launch_gate"),
                "total_cost_usd": meta.get("total_cost_usd"),
                "workflow_count": len(workflows),
            }
        )

    index_path = target_dir / "dataset_index.json"
    index_path.write_text(json.dumps(entries, indent=2, default=str))
    logger.info("Rebuilt dataset index with %d entries at %s", len(entries), index_path)
    return index_path


def load_run(path: Path) -> dict[str, Any]:
    """Load and return a single backtest run from *path*."""
    return json.loads(path.read_text())


def load_all_runs(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load all backtest runs, sorted newest-first by filename timestamp."""
    target_dir = dataset_dir or DATASET_DIR
    if not target_dir.exists():
        return []

    runs: list[dict[str, Any]] = []
    for path in sorted(target_dir.glob("backtest_*.json"), reverse=True):
        try:
            runs.append(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping unreadable file %s: %s", path.name, exc)
    return runs


def query_runs(
    dataset_dir: Path | None = None,
    workflow: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict[str, Any]]:
    """Load all runs then filter by workflow presence and/or time range.

    Parameters
    ----------
    workflow:
        If given, only include runs whose ``workflows`` dict contains
        this key.
    after:
        ISO 8601 timestamp lower bound (inclusive) on ``meta.timestamp``.
    before:
        ISO 8601 timestamp upper bound (inclusive) on ``meta.timestamp``.
    """
    runs = load_all_runs(dataset_dir)
    filtered: list[dict[str, Any]] = []

    for run in runs:
        if workflow is not None:
            workflows = run.get("workflows", {})
            if workflow not in workflows:
                continue

        ts = run.get("meta", {}).get("timestamp")

        if after is not None:
            if ts is None or ts < after:
                continue

        if before is not None:
            if ts is None or ts > before:
                continue

        filtered.append(run)

    return filtered


def load_index(dataset_dir: Path | None = None) -> list[dict[str, Any]]:
    """Load the dataset index, rebuilding it first if it does not exist."""
    target_dir = dataset_dir or DATASET_DIR
    index_path = target_dir / "dataset_index.json"

    if not index_path.exists():
        rebuild_index(target_dir)

    if not index_path.exists():
        return []

    return json.loads(index_path.read_text())
