"""Tests for ProfileRunner: pipeline orchestration, workflow loading, cost summary."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from pretia.collectors.base import StepRecord
from pretia.runner import ProfileRunner, _build_cost_summary
from pretia.store import ProfileStore


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o-mini",
    input_tokens: int = 100,
    output_tokens: int = 50,
    iteration: int = 1,
) -> StepRecord:
    return StepRecord(
        step_name=step_name,
        step_type="llm",
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        context_size=input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash="abc123",
        system_prompt_tokens=50,
        output_format="text",
        is_retry=False,
        iteration=iteration,
        parent_step=None,
        duration_ms=100,
        timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    )


# ---------------------------------------------------------------------------
# Cost summary calculation
# ---------------------------------------------------------------------------


class TestBuildCostSummary:
    def test_basic_aggregation(self):
        runs = [
            [_make_record("step_a", "gpt-4o-mini", 100, 50)],
            [_make_record("step_a", "gpt-4o-mini", 200, 100)],
        ]
        summary = _build_cost_summary(runs)

        assert "step_a" in summary["per_step"]
        assert len(summary["run_totals"]) == 2
        assert summary["mean_cost_per_run"] > 0
        assert summary["total_session_cost"] > 0

    def test_projection_math(self):
        runs = [[_make_record("s", "gpt-4o-mini", 1000, 500)]]
        summary = _build_cost_summary(runs)
        mean = summary["mean_cost_per_run"]
        assert summary["projection_1000_monthly"] == pytest.approx(
            mean * 1000 * 30,
        )

    def test_unknown_model_zero_cost(self):
        runs = [[_make_record("s", "totally-fake-model", 100, 50)]]
        summary = _build_cost_summary(runs)
        assert summary["mean_cost_per_run"] == 0.0

    def test_uses_monthly_projection_keys(self):
        runs = [[_make_record("s", "gpt-4o-mini", 100, 50)]]
        summary = _build_cost_summary(runs)
        assert "projection_100_monthly" in summary
        assert "projection_1000_monthly" in summary
        assert "projection_10000_monthly" in summary
        assert "projection_100_day" not in summary


# ---------------------------------------------------------------------------
# Workflow loading
# ---------------------------------------------------------------------------


class TestWorkflowLoading:
    def test_found_by_graph_attr(self, tmp_path):
        f = tmp_path / "agent.py"
        f.write_text("graph = 'fake_workflow'\n")
        runner = ProfileRunner(
            workflow_path=str(f),
            single_input="test",
        )
        workflow, _, _mod = runner._load_workflow()
        assert workflow == "fake_workflow"

    def test_found_by_workflow_attr(self, tmp_path):
        f = tmp_path / "agent.py"
        f.write_text("workflow = 'my_wf'\n")
        runner = ProfileRunner(
            workflow_path=str(f),
            single_input="test",
        )
        workflow, _, _mod = runner._load_workflow()
        assert workflow == "my_wf"

    def test_not_found_raises(self, tmp_path):
        f = tmp_path / "empty.py"
        f.write_text("x = 42\n")
        runner = ProfileRunner(
            workflow_path=str(f),
            single_input="test",
        )
        with pytest.raises(Exception, match="Could not find a workflow"):
            runner._load_workflow()


# ---------------------------------------------------------------------------
# Collector auto-detection
# ---------------------------------------------------------------------------


class TestCollectorDetection:
    def test_langgraph_detected(self):
        class _FakeGraph:
            nodes = {"a": None}

            async def ainvoke(self, *a, **kw): ...

        runner = ProfileRunner(
            workflow_path="fake.py",
            single_input="test",
        )
        try:
            coll = runner._select_collector(_FakeGraph())
            assert type(coll).__name__ == "LangGraphCollector"
        except ImportError:
            pytest.skip("langchain-core not installed")

    def test_generic_fallback(self):
        runner = ProfileRunner(
            workflow_path="fake.py",
            single_input="test",
        )
        coll = runner._select_collector(object())
        assert type(coll).__name__ == "GenericCollector"


# ---------------------------------------------------------------------------
# Input mode passthrough
# ---------------------------------------------------------------------------


class TestInputPassthrough:
    @pytest.mark.asyncio
    async def test_single_input(self):
        runner = ProfileRunner(
            workflow_path="fake.py",
            single_input="hello",
        )
        selection, inputs = await runner._resolve_inputs("")
        assert selection.mode == "single"
        assert inputs == ["hello"]

    @pytest.mark.asyncio
    async def test_auto_generate(self):
        runner = ProfileRunner(
            workflow_path="fake.py",
            auto_generate=3,
        )
        with patch(
            "pretia.runner.generate_inputs",
            new_callable=AsyncMock,
            return_value=["a", "b", "c"],
        ):
            selection, inputs = await runner._resolve_inputs(
                "You are a bot.",
            )
        assert selection.mode == "auto-generate"
        assert inputs == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Full pipeline (mocked)
# ---------------------------------------------------------------------------


class TestFullPipeline:
    def test_happy_path(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")
        out_dir = tmp_path / "output"

        records = [
            _make_record("classify", "gpt-4o-mini", 100, 50),
            _make_record("respond", "gpt-4o-mini", 200, 100),
        ]
        mock_runs = [records, records]

        with (
            patch(
                "pretia.runner.ProfileRunner._select_collector",
            ) as mock_coll,
            patch(
                "pretia.runner.generate_inputs",
                new_callable=AsyncMock,
                return_value=["input1", "input2"],
            ),
        ):
            fake_collector = AsyncMock()
            fake_collector.collect = AsyncMock(return_value=mock_runs)
            mock_coll.return_value = fake_collector

            runner = ProfileRunner(
                workflow_path=str(wf),
                auto_generate=2,
                output_dir=str(out_dir),
            )
            session = runner.run_sync()

        assert session.sample_size == 2
        assert len(session.runs) == 2
        assert session.workflow_name == str(wf)
        assert "cost_summary" in session.metadata
        assert "stats" in session.metadata
        assert "patterns" in session.metadata
        assert "projection" in session.metadata
        assert "confidence" in session.metadata

        cost = session.metadata["cost_summary"]
        assert cost["mean_cost_per_run"] > 0

        stats = session.metadata["stats"]
        assert stats["total_runs"] == 2
        assert isinstance(session.metadata["patterns"], list)

        proj = session.metadata["projection"]
        assert proj["method"] in ("linear", "montecarlo")
        assert "projections" in proj
        assert "confidence" in proj

        conf = session.metadata["confidence"]
        assert conf["tier"] in ("HIGH", "MODERATE", "LOW", "VERY_LOW")

    def test_profile_saved_to_disk(self, tmp_path):
        wf = tmp_path / "agent.py"
        wf.write_text("graph = 'fake'\n")
        out_dir = tmp_path / "profiles"

        with (
            patch(
                "pretia.runner.ProfileRunner._select_collector",
            ) as mock_coll,
            patch(
                "pretia.runner.generate_inputs",
                new_callable=AsyncMock,
                return_value=["x"],
            ),
        ):
            fake_collector = AsyncMock()
            fake_collector.collect = AsyncMock(
                return_value=[[_make_record()]],
            )
            mock_coll.return_value = fake_collector

            runner = ProfileRunner(
                workflow_path=str(wf),
                auto_generate=1,
                output_dir=str(out_dir),
            )
            session = runner.run_sync()

        saved = session.metadata["saved_path"]
        assert Path(saved).exists()

        store = ProfileStore(storage_dir=out_dir)
        loaded = store.load(Path(saved))
        assert loaded.workflow_name == str(wf)


# ---------------------------------------------------------------------------
# _post_collect shared pipeline
# ---------------------------------------------------------------------------


class TestPostCollect:
    def test_builds_cost_summary(self, tmp_path):
        runs = [
            [_make_record("classify", "gpt-4o-mini", 100, 50)],
            [_make_record("classify", "gpt-4o-mini", 200, 80)],
        ]
        runner = ProfileRunner(
            workflow_path="dummy.py",
            single_input="test",
            output_dir=str(tmp_path),
        )
        session = runner._post_collect(
            runs,
            workflow_name="test-wf",
            workflow_hash="abc123",
            sample_size=2,
            input_mode="single",
            input_source="single",
            workflow_id="test-wf",
        )
        assert "cost_summary" in session.metadata
        assert session.metadata["cost_summary"]["mean_cost_per_run"] > 0

    def test_detects_patterns(self, tmp_path):
        runs = [
            [_make_record("classify", "gpt-4o-mini", 100, 50)],
            [_make_record("classify", "gpt-4o-mini", 200, 80)],
        ]
        runner = ProfileRunner(
            workflow_path="dummy.py",
            single_input="test",
            output_dir=str(tmp_path),
        )
        session = runner._post_collect(
            runs,
            workflow_name="test-wf",
            workflow_hash="abc123",
            sample_size=2,
            input_mode="single",
            input_source="single",
            workflow_id="test-wf",
        )
        assert "patterns" in session.metadata
        assert isinstance(session.metadata["patterns"], list)

    def test_includes_extra_metadata(self, tmp_path):
        runs = [[_make_record("classify", "gpt-4o-mini", 100, 50)]]
        runner = ProfileRunner(
            workflow_path="dummy.py",
            single_input="test",
            output_dir=str(tmp_path),
        )
        session = runner._post_collect(
            runs,
            workflow_name="test-wf",
            workflow_hash="abc123",
            sample_size=1,
            input_mode="langfuse-analyze",
            input_source="langfuse",
            workflow_id="langfuse-import",
            extra_metadata={"langfuse_trace_count": 5},
        )
        assert session.metadata["langfuse_trace_count"] == 5

    def test_saves_to_store(self, tmp_path):
        runs = [[_make_record("classify", "gpt-4o-mini", 100, 50)]]
        runner = ProfileRunner(
            workflow_path="dummy.py",
            single_input="test",
            output_dir=str(tmp_path),
        )
        session = runner._post_collect(
            runs,
            workflow_name="test-wf",
            workflow_hash="abc123",
            sample_size=1,
            input_mode="single",
            input_source="single",
            workflow_id="test-wf",
        )
        assert "saved_path" in session.metadata
        saved = Path(session.metadata["saved_path"])
        assert saved.exists()
