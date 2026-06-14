"""Tests for ProfilingSession + ProfileStore: round-trip, listing, filtering, readability."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from agentcost.collectors.base import StepRecord
from agentcost.store import ProfileStore, ProfilingSession


def _session(
    *,
    profiled_at: datetime,
    workflow_name: str = "support-agent",
    runs: list[list[StepRecord]] | None = None,
) -> ProfilingSession:
    return ProfilingSession(
        workflow_name=workflow_name,
        workflow_hash="abc123",
        profiled_at=profiled_at,
        sample_size=20,
        input_mode="auto-generate",
        runs=runs if runs is not None else [],
        metadata={"framework": "langgraph", "python": "3.12"},
    )


def _set_mtime(path: Path, mtime: float) -> None:
    os.utime(path, (path.stat().st_atime, mtime))


def test_save_and_load_round_trip(tmp_path, sample_record):
    store = ProfileStore(storage_dir=tmp_path)
    session = _session(
        profiled_at=datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC),
        runs=[
            [sample_record, sample_record, sample_record],
            [sample_record, sample_record, sample_record],
        ],
    )
    path = store.save(session)
    loaded = store.load(path)
    assert loaded == session


def test_filename_pattern(tmp_path):
    store = ProfileStore(storage_dir=tmp_path)
    session = _session(profiled_at=datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC))
    path = store.save(session)
    assert path.name == "support-agent_20260520_143000.json"


def test_save_creates_storage_dir(tmp_path):
    storage = tmp_path / "nested" / "agentcost"
    assert not storage.exists()
    store = ProfileStore(storage_dir=storage)
    store.save(_session(profiled_at=datetime(2026, 1, 1, tzinfo=UTC)))
    assert storage.is_dir()


def test_list_sessions_newest_first(tmp_path):
    store = ProfileStore(storage_dir=tmp_path)
    p1 = store.save(_session(profiled_at=datetime(2026, 1, 1, tzinfo=UTC)))
    p2 = store.save(_session(profiled_at=datetime(2026, 2, 1, tzinfo=UTC)))
    p3 = store.save(_session(profiled_at=datetime(2026, 3, 1, tzinfo=UTC)))

    # Saves happen too fast for mtimes to differ reliably; pin them explicitly.
    _set_mtime(p1, 1_000.0)
    _set_mtime(p2, 2_000.0)
    _set_mtime(p3, 3_000.0)

    assert store.list_sessions() == [p3, p2, p1]


def test_list_sessions_filtered_by_workflow(tmp_path):
    store = ProfileStore(storage_dir=tmp_path)
    support = store.save(
        _session(
            profiled_at=datetime(2026, 1, 1, tzinfo=UTC),
            workflow_name="support-agent",
        )
    )
    store.save(
        _session(
            profiled_at=datetime(2026, 1, 1, tzinfo=UTC),
            workflow_name="other-agent",
        )
    )
    assert store.list_sessions("support-agent") == [support]


def test_list_sessions_returns_empty_when_dir_missing(tmp_path):
    store = ProfileStore(storage_dir=tmp_path / "does-not-exist")
    assert store.list_sessions() == []
    assert store.list_sessions("support-agent") == []


def test_latest_returns_newest(tmp_path):
    store = ProfileStore(storage_dir=tmp_path)
    older = store.save(_session(profiled_at=datetime(2026, 1, 1, tzinfo=UTC)))
    newer = store.save(_session(profiled_at=datetime(2026, 2, 1, tzinfo=UTC)))
    _set_mtime(older, 1_000.0)
    _set_mtime(newer, 2_000.0)

    latest = store.latest("support-agent")
    assert latest is not None
    assert latest.profiled_at == datetime(2026, 2, 1, tzinfo=UTC)


def test_latest_returns_none_when_no_sessions(tmp_path):
    store = ProfileStore(storage_dir=tmp_path)
    assert store.latest("nonexistent") is None


def test_saved_file_is_indented_json(tmp_path):
    store = ProfileStore(storage_dir=tmp_path)
    path = store.save(_session(profiled_at=datetime(2026, 5, 20, tzinfo=UTC)))
    text = path.read_text()
    json.loads(text)
    assert "\n" in text
    assert "  " in text


def test_loaded_session_step_records_are_step_record_instances(tmp_path, sample_record):
    store = ProfileStore(storage_dir=tmp_path)
    session = _session(
        profiled_at=datetime(2026, 5, 20, tzinfo=UTC),
        runs=[[sample_record]],
    )
    path = store.save(session)
    loaded = store.load(path)
    assert isinstance(loaded.runs[0][0], StepRecord)
    assert loaded.runs[0][0] == sample_record


def test_default_storage_dir():
    store = ProfileStore()
    assert store.storage_dir == Path(".agentcost")


# ---------------------------------------------------------------------------
# v3 metadata enrichment fields
# ---------------------------------------------------------------------------


class TestMetadataEnrichment:
    def test_new_fields_default_to_none(self):
        session = _session(profiled_at=datetime(2026, 5, 20, tzinfo=UTC))
        assert session.workflow_id is None
        assert session.run_id is None
        assert session.framework is None
        assert session.agentcost_version is None
        assert session.profiling_cost is None

    def test_new_fields_serialize_roundtrip(self, tmp_path, sample_record):
        store = ProfileStore(storage_dir=tmp_path)
        session = ProfilingSession(
            workflow_name="test-agent",
            workflow_hash="abc123",
            profiled_at=datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC),
            sample_size=10,
            input_mode="single",
            runs=[[sample_record]],
            metadata={},
            workflow_id="test-agent",
            run_id="550e8400-e29b-41d4-a716-446655440000",
            framework="langgraph",
            agentcost_version="0.1.0",
            profiling_cost=1.84,
        )
        path = store.save(session)
        loaded = store.load(path)
        assert loaded.workflow_id == "test-agent"
        assert loaded.run_id == "550e8400-e29b-41d4-a716-446655440000"
        assert loaded.framework == "langgraph"
        assert loaded.agentcost_version == "0.1.0"
        assert loaded.profiling_cost == 1.84

    def test_backward_compat_missing_new_fields(self):
        data = {
            "workflow_name": "old-agent",
            "workflow_hash": "abc",
            "profiled_at": "2026-01-01T00:00:00+00:00",
            "sample_size": 5,
            "input_mode": "single",
            "runs": [],
            "metadata": {},
        }
        session = ProfilingSession.from_dict(data)
        assert session.workflow_id is None
        assert session.run_id is None
        assert session.framework is None
        assert session.agentcost_version is None
        assert session.profiling_cost is None

    def test_new_fields_in_to_dict_output(self):
        session = ProfilingSession(
            workflow_name="test",
            workflow_hash="abc",
            profiled_at=datetime(2026, 1, 1, tzinfo=UTC),
            sample_size=1,
            input_mode="single",
            runs=[],
            metadata={},
            workflow_id="my-workflow",
            run_id="uuid-here",
            framework="generic",
            agentcost_version="0.1.0",
            profiling_cost=2.50,
        )
        d = session.to_dict()
        assert d["workflow_id"] == "my-workflow"
        assert d["run_id"] == "uuid-here"
        assert d["framework"] == "generic"
        assert d["agentcost_version"] == "0.1.0"
        assert d["profiling_cost"] == 2.50

    def test_profiling_cost_accepts_float(self):
        session = ProfilingSession(
            workflow_name="test",
            workflow_hash="abc",
            profiled_at=datetime(2026, 1, 1, tzinfo=UTC),
            sample_size=1,
            input_mode="single",
            runs=[],
            metadata={},
            profiling_cost=1.23,
        )
        d = session.to_dict()
        restored = ProfilingSession.from_dict(d)
        assert restored.profiling_cost == 1.23

    def test_run_id_is_string(self):
        uuid_str = "550e8400-e29b-41d4-a716-446655440000"
        session = ProfilingSession(
            workflow_name="test",
            workflow_hash="abc",
            profiled_at=datetime(2026, 1, 1, tzinfo=UTC),
            sample_size=1,
            input_mode="single",
            runs=[],
            metadata={},
            run_id=uuid_str,
        )
        d = session.to_dict()
        restored = ProfilingSession.from_dict(d)
        assert restored.run_id == uuid_str
