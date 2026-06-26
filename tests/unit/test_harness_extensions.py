"""Tests for harness extensions: save_results metadata, save_as_session, pattern detection."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from typing import Any

import pytest

_has_litellm = True
try:
    import litellm  # noqa: F401
except ModuleNotFoundError:
    _has_litellm = False

from bt_agents.harness.run_workflow import (  # noqa: E402
    _detect_batch_patterns,
    _extract_run_metadata,
    _pricing_table_hash,
    _prompt_hashes,
    load_agent,
    load_prompts,
    save_as_session,
    save_results,
)
from pretia.collectors.base import StepRecord  # noqa: E402
from pretia.store import ProfileStore, ProfilingSession  # noqa: E402


@pytest.fixture
def sample_step_record() -> StepRecord:
    return StepRecord(
        step_name="classify_respond",
        step_type="llm",
        model="claude-haiku-4-5",
        input_tokens=340,
        output_tokens=45,
        context_size=620,
        tool_definitions_tokens=0,
        system_prompt_hash="abc123",
        system_prompt_tokens=280,
        output_format="text",
        is_retry=False,
        iteration=1,
        parent_step=None,
        duration_ms=230,
        timestamp=datetime(2026, 6, 1, 12, 0, 0, tzinfo=UTC),
    )


@pytest.fixture
def multi_step_records() -> list[list[StepRecord]]:
    """Five runs, each with one StepRecord."""
    records = []
    for i in range(5):
        records.append(
            [
                StepRecord(
                    step_name="classify_respond",
                    step_type="llm",
                    model="claude-haiku-4-5",
                    input_tokens=300 + i * 10,
                    output_tokens=40 + i * 5,
                    context_size=600 + i * 10,
                    tool_definitions_tokens=0,
                    system_prompt_hash="abc123",
                    system_prompt_tokens=280,
                    output_format="text",
                    is_retry=False,
                    iteration=1,
                    parent_step=None,
                    duration_ms=200 + i * 20,
                    timestamp=datetime(2026, 6, 1, 12, i, 0, tzinfo=UTC),
                )
            ]
        )
    return records


@pytest.fixture
def sample_prompts() -> dict[str, str]:
    suffix = "{{CACHE_BUST_SUFFIX}}"
    return {
        "classify_respond": f"You are a helpful support agent.\n<!-- session: {suffix} -->",
        "research_draft_loop": f"You are an analyst.\n<!-- session: {suffix} -->",
    }


@pytest.fixture
def sample_inputs() -> list[dict[str, Any]]:
    return [
        {"input": "short question", "tier": "easy", "structural_descriptor": {"tokens": 20}},
        {
            "input": "a longer more complex question",
            "tier": "medium",
            "structural_descriptor": {"tokens": 50},
        },
        {"input": "very detailed question", "tier": "hard"},
        {"input": "simple", "_dry_run": True},
        {"input": "another simple one"},
    ]


class TestExtractRunMetadata:
    def test_captures_input_tier(self, sample_step_record):
        input_data = {"input": "test", "tier": "easy"}
        meta = _extract_run_metadata(input_data, [sample_step_record])
        assert meta["input_tier"] == "easy"

    def test_captures_input_tier_alternate_key(self, sample_step_record):
        input_data = {"input": "test", "input_tier": "hard"}
        meta = _extract_run_metadata(input_data, [sample_step_record])
        assert meta["input_tier"] == "hard"

    def test_tier_none_when_missing(self, sample_step_record):
        input_data = {"input": "test"}
        meta = _extract_run_metadata(input_data, [sample_step_record])
        assert meta["input_tier"] is None

    def test_captures_structural_descriptor(self, sample_step_record):
        input_data = {"input": "test", "structural_descriptor": {"tokens": 100}}
        meta = _extract_run_metadata(input_data, [sample_step_record])
        assert meta["structural_descriptor"] == {"tokens": 100}

    def test_captures_steps_executed(self, sample_step_record):
        meta = _extract_run_metadata({}, [sample_step_record])
        assert meta["steps_executed"] == ["classify_respond"]

    def test_captures_models_used(self, sample_step_record):
        meta = _extract_run_metadata({}, [sample_step_record])
        assert "claude-haiku-4-5" in meta["models_used"]

    def test_captures_step_costs(self, sample_step_record):
        meta = _extract_run_metadata({}, [sample_step_record])
        assert "classify_respond" in meta["step_costs"]
        assert isinstance(meta["step_costs"]["classify_respond"], float)

    def test_captures_max_iteration(self, sample_step_record):
        meta = _extract_run_metadata({}, [sample_step_record])
        assert meta["max_iteration"] == 1

    def test_detects_short_circuit(self):
        intake_record = StepRecord(
            step_name="intake_override",
            step_type="llm",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            context_size=200,
            tool_definitions_tokens=0,
            system_prompt_hash="abc",
            system_prompt_tokens=100,
            output_format="json",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=100,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        meta = _extract_run_metadata({}, [intake_record])
        assert meta["decisions"].get("short_circuited") is True

    def test_detects_routing(self):
        classify = StepRecord(
            step_name="classify",
            step_type="llm",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            context_size=200,
            tool_definitions_tokens=0,
            system_prompt_hash="abc",
            system_prompt_tokens=100,
            output_format="json",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=100,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        route = StepRecord(
            step_name="path_b_moderate",
            step_type="llm",
            model="claude-sonnet-4-6",
            input_tokens=200,
            output_tokens=100,
            context_size=400,
            tool_definitions_tokens=0,
            system_prompt_hash="def",
            system_prompt_tokens=150,
            output_format="text",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=200,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        meta = _extract_run_metadata({}, [classify, route])
        assert meta["decisions"]["classifier_step"] == "classify"
        assert meta["decisions"]["routed_to"] == "path_b_moderate"

    def test_detects_opus_triggered(self):
        review = StepRecord(
            step_name="final_review",
            step_type="llm",
            model="claude-opus-4-7",
            input_tokens=500,
            output_tokens=200,
            context_size=700,
            tool_definitions_tokens=0,
            system_prompt_hash="abc",
            system_prompt_tokens=300,
            output_format="json",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=500,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        meta = _extract_run_metadata({}, [review])
        assert meta["decisions"]["opus_triggered"] is True

    def test_detects_conditional_routing(self):
        routing = StepRecord(
            step_name="conditional_routing",
            step_type="llm",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            context_size=200,
            tool_definitions_tokens=0,
            system_prompt_hash="abc",
            system_prompt_tokens=100,
            output_format="json",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=100,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        other = StepRecord(
            step_name="intake_override",
            step_type="llm",
            model="claude-haiku-4-5",
            input_tokens=100,
            output_tokens=50,
            context_size=200,
            tool_definitions_tokens=0,
            system_prompt_hash="abc",
            system_prompt_tokens=100,
            output_format="json",
            is_retry=False,
            iteration=1,
            parent_step=None,
            duration_ms=100,
            timestamp=datetime(2026, 6, 1, tzinfo=UTC),
        )
        meta = _extract_run_metadata({}, [other, routing])
        assert meta["decisions"]["routing_triggered"] is True


class TestPricingTableHash:
    def test_returns_hex_string(self):
        h = _pricing_table_hash()
        assert isinstance(h, str)
        assert len(h) == 64
        int(h, 16)

    def test_deterministic(self):
        assert _pricing_table_hash() == _pricing_table_hash()


class TestPromptHashes:
    def test_hashes_all_prompts(self, sample_prompts):
        hashes = _prompt_hashes(sample_prompts)
        assert set(hashes.keys()) == set(sample_prompts.keys())
        for v in hashes.values():
            assert isinstance(v, str)
            assert len(v) == 64

    def test_different_content_different_hash(self):
        p1 = {"step": "prompt A"}
        p2 = {"step": "prompt B"}
        assert _prompt_hashes(p1)["step"] != _prompt_hashes(p2)["step"]

    def test_empty_prompts(self):
        assert _prompt_hashes({}) == {}


class TestSaveResults:
    def test_creates_file(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        assert filepath.exists()
        data = json.loads(filepath.read_text())
        assert data["workflow_id"] == "W1"
        assert data["total_runs"] == 5

    def test_includes_per_run_metadata(self, multi_step_records, tmp_path, sample_inputs):
        filepath = save_results(
            "W1",
            multi_step_records,
            str(tmp_path),
            inputs=sample_inputs,
        )
        data = json.loads(filepath.read_text())
        run0 = data["runs"][0]
        assert "metadata" in run0
        assert run0["metadata"]["input_tier"] == "easy"
        assert run0["metadata"]["structural_descriptor"] == {"tokens": 20}

    def test_metadata_none_when_no_inputs(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        data = json.loads(filepath.read_text())
        run0 = data["runs"][0]
        assert run0["metadata"]["input_tier"] is None

    def test_includes_detected_patterns(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        data = json.loads(filepath.read_text())
        assert "detected_patterns" in data
        assert isinstance(data["detected_patterns"], list)

    def test_includes_backtest_id(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        data = json.loads(filepath.read_text())
        assert "backtest_id" in data
        assert isinstance(data["backtest_id"], str)
        assert len(data["backtest_id"]) == 32

    def test_includes_backtest_profile(self, multi_step_records, tmp_path):
        filepath = save_results(
            "W1",
            multi_step_records,
            str(tmp_path),
            backtest_profile="profiling",
        )
        data = json.loads(filepath.read_text())
        assert data["backtest_profile"] == "profiling"

    def test_includes_pricing_table_hash(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        data = json.loads(filepath.read_text())
        assert "pricing_table_hash" in data
        assert len(data["pricing_table_hash"]) == 64

    def test_includes_prompt_hashes(self, multi_step_records, tmp_path, sample_prompts):
        filepath = save_results(
            "W1",
            multi_step_records,
            str(tmp_path),
            prompts=sample_prompts,
        )
        data = json.loads(filepath.read_text())
        assert "prompt_hashes" in data
        assert "classify_respond" in data["prompt_hashes"]

    def test_prompt_hashes_none_without_prompts(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        data = json.loads(filepath.read_text())
        assert data["prompt_hashes"] is None

    def test_unique_backtest_ids(self, multi_step_records, tmp_path):
        fp1 = save_results("W1", multi_step_records, str(tmp_path / "a"))
        fp2 = save_results("W1", multi_step_records, str(tmp_path / "b"))
        d1 = json.loads(fp1.read_text())
        d2 = json.loads(fp2.read_text())
        assert d1["backtest_id"] != d2["backtest_id"]

    def test_steps_serialized_correctly(self, multi_step_records, tmp_path):
        filepath = save_results("W1", multi_step_records, str(tmp_path))
        data = json.loads(filepath.read_text())
        for run in data["runs"]:
            for step in run["steps"]:
                assert "step_name" in step
                assert "model" in step
                assert "input_tokens" in step

    def test_decisions_dict_present(self, multi_step_records, tmp_path, sample_inputs):
        filepath = save_results(
            "W1",
            multi_step_records,
            str(tmp_path),
            inputs=sample_inputs,
        )
        data = json.loads(filepath.read_text())
        for run in data["runs"]:
            assert "decisions" in run["metadata"]
            assert isinstance(run["metadata"]["decisions"], dict)


class TestDetectBatchPatterns:
    def test_returns_list(self, multi_step_records):
        patterns = _detect_batch_patterns(multi_step_records)
        assert isinstance(patterns, list)

    def test_returns_dicts(self, multi_step_records):
        patterns = _detect_batch_patterns(multi_step_records)
        for p in patterns:
            assert isinstance(p, dict)

    def test_empty_records(self):
        patterns = _detect_batch_patterns([])
        assert patterns == []

    def test_dry_run_records_dont_crash(self):
        records = [
            [
                StepRecord(
                    step_name="test",
                    step_type="llm",
                    model="claude-haiku-4-5",
                    input_tokens=0,
                    output_tokens=0,
                    context_size=0,
                    tool_definitions_tokens=0,
                    system_prompt_hash="x",
                    system_prompt_tokens=0,
                    output_format="json",
                    is_retry=False,
                    iteration=1,
                    parent_step=None,
                    duration_ms=0,
                    timestamp=datetime(2026, 6, 1, tzinfo=UTC),
                )
            ]
        ]
        patterns = _detect_batch_patterns(records)
        assert isinstance(patterns, list)


class TestSaveAsSession:
    def test_creates_file(self, multi_step_records, tmp_path):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        assert filepath.exists()

    def test_loadable_by_profile_store(self, multi_step_records, tmp_path):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        assert isinstance(session, ProfilingSession)
        assert session.workflow_name == "W1"
        assert session.sample_size == 5
        assert len(session.runs) == 5

    def test_session_runs_contain_step_records(self, multi_step_records, tmp_path):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        for run in session.runs:
            for rec in run:
                assert isinstance(rec, StepRecord)

    def test_session_metadata_has_backtest_fields(self, multi_step_records, tmp_path):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            backtest_profile="profiling",
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        assert session.metadata["backtest_profile"] == "profiling"
        assert "backtest_id" in session.metadata
        assert "pricing_table_hash" in session.metadata

    def test_session_has_python_version(self, multi_step_records, tmp_path):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        assert session.python_version is not None

    def test_session_has_workflow_hash(self, multi_step_records, tmp_path, sample_prompts):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            prompts=sample_prompts,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        assert isinstance(session.workflow_hash, str)
        assert len(session.workflow_hash) == 64

    def test_session_input_mode_is_backtesting(self, multi_step_records, tmp_path):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        assert session.input_mode == "backtesting"

    def test_session_prompt_hashes_stored(self, multi_step_records, tmp_path, sample_prompts):
        filepath = save_as_session(
            "W1",
            multi_step_records,
            prompts=sample_prompts,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        assert "prompt_hashes" in session.metadata
        assert "classify_respond" in session.metadata["prompt_hashes"]

    def test_session_flows_through_projection(self, multi_step_records, tmp_path):
        """End-to-end: save_as_session → ProfileStore.load → compute_stats."""
        from pretia.projection.stats import compute_stats

        filepath = save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        session = store.load(filepath)
        stats = compute_stats(session.runs)
        assert stats.total_runs == 5
        assert stats.total_steps == 5

    def test_list_sessions_finds_saved(self, multi_step_records, tmp_path):
        save_as_session(
            "W1",
            multi_step_records,
            storage_dir=str(tmp_path),
        )
        store = ProfileStore(storage_dir=tmp_path)
        sessions = store.list_sessions("W1")
        assert len(sessions) >= 1


@pytest.mark.skipif(
    not _has_litellm,
    reason="litellm not installed",
)
class TestEndToEndWithDryRun:
    """Integration: dry-run W1 → save_results → save_as_session → load → compute_stats."""

    def test_full_pipeline(self, tmp_path):
        prompts = load_prompts("W1", "prompts/")
        inputs = [
            {"input": "Help me reset my password", "tier": "easy", "_dry_run": True},
            {"input": "Billing issue with my account", "tier": "medium", "_dry_run": True},
            {
                "input": "Complex API integration question about webhooks",
                "tier": "hard",
                "_dry_run": True,
            },
        ]
        agent = load_agent("W1")
        all_records = []
        for inp in inputs:
            records = asyncio.run(agent.execute(inp, prompts))
            all_records.append(records)

        results_path = save_results(
            "W1",
            all_records,
            str(tmp_path / "results"),
            inputs=inputs,
            prompts=prompts,
            backtest_profile="profiling",
        )
        data = json.loads(results_path.read_text())
        assert data["runs"][0]["metadata"]["input_tier"] == "easy"
        assert data["runs"][1]["metadata"]["input_tier"] == "medium"
        assert data["runs"][2]["metadata"]["input_tier"] == "hard"
        assert data["backtest_profile"] == "profiling"
        assert "detected_patterns" in data

        session_path = save_as_session(
            "W1",
            all_records,
            prompts=prompts,
            backtest_profile="profiling",
            storage_dir=str(tmp_path / "sessions"),
        )
        store = ProfileStore(storage_dir=tmp_path / "sessions")
        session = store.load(session_path)
        assert session.sample_size == 3
        assert session.input_mode == "backtesting"

        from pretia.projection.stats import compute_stats

        stats = compute_stats(session.runs)
        assert stats.total_runs == 3
