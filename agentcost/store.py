"""Persist and load profiling sessions as JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from agentcost.collectors.base import StepRecord

_DEFAULT_STORAGE_DIR = Path(".agentcost")
_FILENAME_TIMESTAMP_FMT = "%Y%m%d_%H%M%S"


@dataclass
class ProfilingSession:
    """A single profiling session: workflow metadata plus every StepRecord captured."""

    workflow_name: str
    workflow_hash: str
    profiled_at: datetime
    sample_size: int
    input_mode: str
    runs: list[list[StepRecord]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow_name": self.workflow_name,
            "workflow_hash": self.workflow_hash,
            "profiled_at": self.profiled_at.isoformat(),
            "sample_size": self.sample_size,
            "input_mode": self.input_mode,
            "runs": [[record.to_dict() for record in run] for run in self.runs],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfilingSession:
        """Deserialize from a dict produced by `to_dict()`."""
        return cls(
            workflow_name=data["workflow_name"],
            workflow_hash=data["workflow_hash"],
            profiled_at=datetime.fromisoformat(data["profiled_at"]),
            sample_size=data["sample_size"],
            input_mode=data["input_mode"],
            runs=[[StepRecord.from_dict(r) for r in run] for run in data["runs"]],
            metadata=dict(data["metadata"]),
        )


class ProfileStore:
    """Read and write `ProfilingSession`s as JSON files in a storage directory."""

    def __init__(self, storage_dir: Path | None = None) -> None:
        self.storage_dir = storage_dir if storage_dir is not None else _DEFAULT_STORAGE_DIR

    def save(self, session: ProfilingSession) -> Path:
        """Write a session to disk and return its path.

        Filename pattern: ``{workflow}_{YYYYMMDD_HHMMSS}.json``.
        """
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        stamp = session.profiled_at.strftime(_FILENAME_TIMESTAMP_FMT)
        name = self._safe_name(session.workflow_name)
        path = self.storage_dir / f"{name}_{stamp}.json"
        path.write_text(json.dumps(session.to_dict(), indent=2))
        return path

    def load(self, path: Path) -> ProfilingSession:
        """Read a session from a JSON file written by `save()`."""
        return ProfilingSession.from_dict(json.loads(path.read_text()))

    def list_sessions(self, workflow_name: str | None = None) -> list[Path]:
        """List saved session files, newest first; optionally filtered by workflow name."""
        if not self.storage_dir.exists():
            return []
        if workflow_name is None:
            files = list(self.storage_dir.glob("*.json"))
        else:
            files = list(self.storage_dir.glob(f"{self._safe_name(workflow_name)}_*.json"))
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def latest(self, workflow_name: str) -> ProfilingSession | None:
        """Load the most recent session for a workflow, or None if none exists."""
        sessions = self.list_sessions(workflow_name)
        if not sessions:
            return None
        return self.load(sessions[0])

    @staticmethod
    def _safe_name(workflow_name: str) -> str:
        # Workflow names are often paths like "my_agent.py"; collapse to a stable basename
        # so the same workflow always maps to the same filename prefix.
        return Path(workflow_name).stem.replace(" ", "_") or "workflow"
