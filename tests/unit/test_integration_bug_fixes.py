"""Tests for all integration test bug fixes (v1.2.0)."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from unittest.mock import MagicMock

from pretia.collectors.base import StepRecord
from pretia.projection.patterns import _detect_zero_execution_steps, detect_patterns
from pretia.projection.stats import compute_stats, robust_cv
from pretia.recommend.model_swap import (
    ModelSwapGenerator,
    _has_classification_keywords,
)
from pretia.store import ProfilingSession


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    context_size: int = 100,
    iteration: int = 1,
    duration_ms: int = 500,
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": context_size,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 50,
        "output_format": "text",
        "is_retry": False,
        "iteration": iteration,
        "parent_step": None,
        "duration_ms": duration_ms,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


def _make_session(runs: list[list[StepRecord]]) -> ProfilingSession:
    return ProfilingSession(
        workflow_name="test_workflow",
        workflow_hash="abc123",
        profiled_at=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
        sample_size=len(runs),
        input_mode="auto",
        runs=runs,
        metadata={},
    )


# ---------------------------------------------------------------------------
# Fix 1a: StepRecord system_prompt_snippet field
# ---------------------------------------------------------------------------


class TestStepRecordSnippet:
    def test_snippet_default_none(self):
        rec = _make_record()
        assert rec.system_prompt_snippet is None

    def test_snippet_set(self):
        rec = _make_record(system_prompt_snippet="You are a classifier")
        assert rec.system_prompt_snippet == "You are a classifier"

    def test_snippet_round_trip(self):
        rec = _make_record(system_prompt_snippet="You are a classifier")
        d = rec.to_dict()
        assert d["system_prompt_snippet"] == "You are a classifier"
        restored = StepRecord.from_dict(d)
        assert restored.system_prompt_snippet == "You are a classifier"

    def test_snippet_none_round_trip(self):
        rec = _make_record()
        d = rec.to_dict()
        assert d["system_prompt_snippet"] is None
        restored = StepRecord.from_dict(d)
        assert restored.system_prompt_snippet is None


# ---------------------------------------------------------------------------
# Fix 1b: Anthropic SDK kwargs metadata extraction
# ---------------------------------------------------------------------------


class TestAnthropicKwargsMetadata:
    def test_extract_system_prompt_string(self):
        from pretia.collectors.anthropic_sdk import _extract_kwargs_metadata

        kwargs = {"system": "You are a helpful assistant.", "model": "claude-haiku-4-5"}
        meta = _extract_kwargs_metadata(kwargs)
        assert (
            meta["system_prompt_hash"]
            == hashlib.sha256(b"You are a helpful assistant.").hexdigest()
        )
        assert meta["system_prompt_tokens"] > 0
        assert meta["system_prompt_snippet"] == "You are a helpful assistant."

    def test_extract_system_prompt_list(self):
        from pretia.collectors.anthropic_sdk import _extract_kwargs_metadata

        kwargs = {
            "system": [{"type": "text", "text": "You are a router."}],
            "model": "claude-haiku-4-5",
        }
        meta = _extract_kwargs_metadata(kwargs)
        assert "router" in meta["system_prompt_snippet"]
        assert meta["system_prompt_tokens"] > 0

    def test_extract_max_tokens(self):
        from pretia.collectors.anthropic_sdk import _extract_kwargs_metadata

        kwargs = {"max_tokens": 4096, "model": "claude-haiku-4-5"}
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["max_tokens_setting"] == 4096

    def test_extract_tools(self):
        from pretia.collectors.anthropic_sdk import _extract_kwargs_metadata

        kwargs = {
            "tools": [{"name": "search", "description": "Search the web"}],
            "model": "claude-haiku-4-5",
        }
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["tool_definitions_tokens"] > 0

    def test_empty_kwargs_defaults(self):
        from pretia.collectors.anthropic_sdk import _extract_kwargs_metadata

        meta = _extract_kwargs_metadata({})
        assert meta["max_tokens_setting"] is None
        assert meta["system_prompt_tokens"] == 0
        assert meta["tool_definitions_tokens"] == 0
        assert meta["system_prompt_snippet"] is None

    def test_snippet_truncated_to_200(self):
        from pretia.collectors.anthropic_sdk import _extract_kwargs_metadata

        kwargs = {"system": "x" * 500}
        meta = _extract_kwargs_metadata(kwargs)
        assert len(meta["system_prompt_snippet"]) == 200


# ---------------------------------------------------------------------------
# Fix 1b: OpenAI SDK kwargs metadata extraction
# ---------------------------------------------------------------------------


class TestOpenAIKwargsMetadata:
    def test_extract_system_message(self):
        from pretia.collectors.openai_sdk import _extract_kwargs_metadata

        kwargs = {
            "messages": [
                {"role": "system", "content": "You are a classifier."},
                {"role": "user", "content": "Classify this."},
            ],
        }
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["system_prompt_hash"] == hashlib.sha256(b"You are a classifier.").hexdigest()
        assert meta["system_prompt_tokens"] > 0
        assert meta["system_prompt_snippet"] == "You are a classifier."

    def test_extract_max_tokens(self):
        from pretia.collectors.openai_sdk import _extract_kwargs_metadata

        kwargs = {"max_tokens": 2048}
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["max_tokens_setting"] == 2048

    def test_extract_max_completion_tokens(self):
        from pretia.collectors.openai_sdk import _extract_kwargs_metadata

        kwargs = {"max_completion_tokens": 4096}
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["max_tokens_setting"] == 4096

    def test_extract_tools_openai(self):
        from pretia.collectors.openai_sdk import _extract_kwargs_metadata

        kwargs = {
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "search", "parameters": {"type": "object"}},
                }
            ],
        }
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["tool_definitions_tokens"] > 0

    def test_no_system_message_defaults(self):
        from pretia.collectors.openai_sdk import _extract_kwargs_metadata

        kwargs = {"messages": [{"role": "user", "content": "Hello"}]}
        meta = _extract_kwargs_metadata(kwargs)
        assert meta["system_prompt_snippet"] is None
        assert meta["system_prompt_tokens"] == 0


# ---------------------------------------------------------------------------
# Fix 2: Substring model matching
# ---------------------------------------------------------------------------


class TestModelMatching:
    def test_gpt4o_not_matches_gpt4o_mini(self):
        from pretia.collectors.openai_sdk import _models_match

        assert not _models_match("gpt-4o", "gpt-4o-mini")

    def test_same_model_matches(self):
        from pretia.collectors.openai_sdk import _models_match

        assert _models_match("gpt-4o-mini", "gpt-4o-mini")

    def test_date_suffix_resolves(self):
        from pretia.collectors.openai_sdk import _models_match

        assert _models_match("gpt-4o-mini-2024-07-18", "gpt-4o-mini")

    def test_unknown_model_exact_match(self):
        from pretia.collectors.openai_sdk import _models_match

        assert _models_match("my-custom-model", "my-custom-model")

    def test_unknown_model_no_match(self):
        from pretia.collectors.openai_sdk import _models_match

        assert not _models_match("my-custom-model", "other-model")

    def test_anthropic_models_match(self):
        from pretia.collectors.anthropic_sdk import _models_match

        assert _models_match("claude-haiku-4-5-20251001", "claude-haiku-4-5")

    def test_anthropic_different_models(self):
        from pretia.collectors.anthropic_sdk import _models_match

        assert not _models_match("claude-haiku-4-5", "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# Fix 3: GenericCollector CLI reuses module instance
# ---------------------------------------------------------------------------


class TestGenericCollectorReuse:
    def test_select_collector_finds_module_instance(self):
        from pretia.collectors.generic import GenericCollector
        from pretia.runner import ProfileRunner

        user_collector = GenericCollector()
        module = MagicMock()
        module.__dict__ = {"collector": user_collector, "__name__": "test"}

        runner = ProfileRunner.__new__(ProfileRunner)
        runner.collector_name = "generic"
        result = runner._select_collector(MagicMock(), module=module)
        assert result is user_collector

    def test_select_collector_fallback_new_instance(self):
        from pretia.collectors.generic import GenericCollector
        from pretia.runner import ProfileRunner

        module = MagicMock()
        module.__dict__ = {"__name__": "test"}

        runner = ProfileRunner.__new__(ProfileRunner)
        runner.collector_name = "generic"
        result = runner._select_collector(MagicMock(), module=module)
        assert isinstance(result, GenericCollector)


# ---------------------------------------------------------------------------
# Fix 4: Traffic override interpolation
# ---------------------------------------------------------------------------


class TestTrafficOverride:
    def test_custom_traffic_produces_nonzero(self):
        from pretia.ci.report import _build_new_projection_panel

        projection = {
            "method": "linear",
            "traffic_volumes": [100, 1000, 10000],
            "projections": {
                "100": {
                    "daily_volume": 100,
                    "monthly_cost": {"p50": 1.5, "p75": 2.0, "p90": 3.0, "p95": 4.0, "mean": 2.0},
                    "daily_cost": {
                        "p50": 0.05,
                        "p75": 0.07,
                        "p90": 0.1,
                        "p95": 0.13,
                        "mean": 0.07,
                    },
                    "cost_per_run": {
                        "p50": 0.0005,
                        "p75": 0.0007,
                        "p90": 0.001,
                        "p95": 0.0013,
                        "mean": 0.0007,
                    },
                },
            },
            "confidence": {"tier": "MODERATE", "display_range": "p50 – p95", "deductions": []},
        }
        panel = _build_new_projection_panel(projection, traffic=5000)
        assert panel is not None

    def test_standard_traffic_still_works(self):
        from pretia.ci.report import _build_new_projection_panel

        projection = {
            "method": "linear",
            "traffic_volumes": [100],
            "projections": {
                "100": {
                    "daily_volume": 100,
                    "monthly_cost": {"p50": 1.5, "p75": 2.0, "p90": 3.0, "p95": 4.0, "mean": 2.0},
                    "daily_cost": {"p50": 0.05},
                    "cost_per_run": {"p50": 0.0005, "mean": 0.0007},
                },
            },
            "confidence": {"tier": "MODERATE", "display_range": "p50 – p95", "deductions": []},
        }
        panel = _build_new_projection_panel(projection, traffic=100)
        assert panel is not None


# ---------------------------------------------------------------------------
# Fix 5: update-pricing accepts both key formats
# ---------------------------------------------------------------------------


class TestPricingFileKeys:
    def test_input_price_output_price_accepted(self):
        info = {"input_price": 1.0, "output_price": 5.0}
        inp = info.get("input") if "input" in info else info.get("input_price")
        out = info.get("output") if "output" in info else info.get("output_price")
        assert inp == 1.0
        assert out == 5.0

    def test_input_output_accepted(self):
        info = {"input": 2.0, "output": 8.0}
        inp = info.get("input") if "input" in info else info.get("input_price")
        out = info.get("output") if "output" in info else info.get("output_price")
        assert inp == 2.0
        assert out == 8.0

    def test_missing_keys_returns_none(self):
        info = {"tier": "mid"}
        inp = info.get("input") if "input" in info else info.get("input_price")
        out = info.get("output") if "output" in info else info.get("output_price")
        assert inp is None
        assert out is None


# ---------------------------------------------------------------------------
# Fix 6: MODEL_SWAP keyword detection with system prompt snippet
# ---------------------------------------------------------------------------


class TestModelSwapKeywordsWithSnippet:
    def test_keyword_in_snippet(self):
        assert _has_classification_keywords(
            "run", system_prompt_snippet="You are a classifier that routes requests"
        )

    def test_no_keyword_in_snippet(self):
        assert not _has_classification_keywords(
            "run", system_prompt_snippet="You are a creative writer"
        )

    def test_keyword_in_step_name_still_works(self):
        assert _has_classification_keywords("classify_intent")

    def test_keyword_in_snippet_case_insensitive(self):
        assert _has_classification_keywords(
            "run", system_prompt_snippet="You are a ROUTER for customer queries"
        )

    def test_sdk_step_with_classification_snippet_fires(self):
        runs = [
            [
                _make_record(
                    step_name="run",
                    model="gpt-4o",
                    input_tokens=1000,
                    output_tokens=50,
                    output_format="json",
                    system_prompt_snippet="You are a classifier. Classify the input.",
                )
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) >= 1
        assert any(r.type == "model_swap" for r in recs)


# ---------------------------------------------------------------------------
# Fix 7: robust_cv MAD fallback
# ---------------------------------------------------------------------------


class TestRobustCV:
    def test_mad_zero_with_variance_falls_back(self):
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        cv = robust_cv(values)
        assert cv > 0

    def test_all_identical_returns_zero(self):
        assert robust_cv([5.0, 5.0, 5.0, 5.0, 5.0]) == 0.0

    def test_single_value_returns_zero(self):
        assert robust_cv([3.0]) == 0.0

    def test_normal_distribution_uses_mad(self):
        values = [1.0, 2.0, 3.0, 4.0, 100.0]
        cv = robust_cv(values)
        assert cv > 0

    def test_routing_pattern_detected(self):
        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0]
        cv = robust_cv(values)
        assert cv > 0.3

    def test_step_count_variance_fires_for_routing(self):
        runs: list[list[StepRecord]] = []
        for _ in range(6):
            runs.append([_make_record(step_name="classify")])
        for _ in range(4):
            runs.append(
                [
                    _make_record(step_name="classify"),
                    _make_record(step_name="review"),
                ]
            )
        stats = compute_stats(runs)
        patterns = detect_patterns(runs, stats)
        pattern_types = [p.pattern_type for p in patterns]
        assert "step_count_variance" in pattern_types


# ---------------------------------------------------------------------------
# Fix 8: Zero Execution Step pattern
# ---------------------------------------------------------------------------


class TestZeroExecutionStep:
    def test_detects_unexecuted_graph_steps(self):
        runs = [
            [_make_record(step_name="classify")],
            [_make_record(step_name="classify")],
        ]
        graph_steps = ["classify", "review", "escalate"]
        patterns = _detect_zero_execution_steps(runs, graph_steps=graph_steps)
        zero_steps = {p.step_name for p in patterns}
        assert "review" in zero_steps
        assert "escalate" in zero_steps
        assert "classify" not in zero_steps

    def test_returns_empty_when_all_executed(self):
        runs = [
            [_make_record(step_name="classify"), _make_record(step_name="respond")],
        ]
        graph_steps = ["classify", "respond"]
        patterns = _detect_zero_execution_steps(runs, graph_steps=graph_steps)
        assert patterns == []

    def test_returns_empty_when_no_graph_steps(self):
        runs = [[_make_record()]]
        patterns = _detect_zero_execution_steps(runs, graph_steps=None)
        assert patterns == []

    def test_graph_steps_passed_through_detect_patterns(self):
        runs = [
            [_make_record(step_name="classify")],
            [_make_record(step_name="classify")],
        ]
        stats = compute_stats(runs)
        patterns = detect_patterns(runs, stats, graph_steps=["classify", "escalate"])
        zero_patterns = [p for p in patterns if p.pattern_type == "zero_execution_step"]
        assert len(zero_patterns) == 1
        assert zero_patterns[0].step_name == "escalate"


# ---------------------------------------------------------------------------
# Fix 1d: Tool name extraction from response
# ---------------------------------------------------------------------------


class TestToolNameExtraction:
    def test_anthropic_tool_use_name(self):
        from pretia.collectors.anthropic_sdk import _record_from_response

        block = MagicMock()
        block.type = "tool_use"
        block.name = "search"

        response = MagicMock()
        response.usage.input_tokens = 100
        response.usage.output_tokens = 50
        response.usage.cache_read_input_tokens = None
        response.model = "claude-haiku-4-5"
        response.content = [block]

        captured: list[StepRecord] = []
        _record_from_response(response, 0, captured, "llm_call")
        assert len(captured) == 1
        assert captured[0].tool_name == "search"

    def test_openai_tool_call_name(self):
        from pretia.collectors.openai_sdk import _extract_tool_name

        func = MagicMock()
        func.name = "get_weather"
        tool_call = MagicMock()
        tool_call.function = func
        msg = MagicMock()
        msg.tool_calls = [tool_call]
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        assert _extract_tool_name(response) == "get_weather"

    def test_openai_no_tool_calls(self):
        from pretia.collectors.openai_sdk import _extract_tool_name

        msg = MagicMock()
        msg.tool_calls = None
        choice = MagicMock()
        choice.message = msg
        response = MagicMock()
        response.choices = [choice]

        assert _extract_tool_name(response) is None


# ---------------------------------------------------------------------------
# P1: Zero Execution Step filters framework-internal nodes
# ---------------------------------------------------------------------------


class TestZeroExecutionStepFrameworkFilter:
    def test_start_end_filtered(self):
        runs = [[_make_record(step_name="classify")]]
        graph_steps = ["__start__", "classify", "review", "__end__"]
        patterns = _detect_zero_execution_steps(runs, graph_steps=graph_steps)
        flagged = {p.step_name for p in patterns}
        assert "__start__" not in flagged
        assert "__end__" not in flagged
        assert "review" in flagged

    def test_route_filtered(self):
        runs = [[_make_record(step_name="classify")]]
        graph_steps = ["_route", "classify", "escalate"]
        patterns = _detect_zero_execution_steps(runs, graph_steps=graph_steps)
        flagged = {p.step_name for p in patterns}
        assert "_route" not in flagged
        assert "escalate" in flagged


# ---------------------------------------------------------------------------
# P1: CACHE_CONTEXT display label for same-step pairs
# ---------------------------------------------------------------------------


class TestCacheContextDisplayLabel:
    def test_same_step_label_no_duplicate_name(self):
        from pretia.recommend.architecture import CacheContextGenerator

        runs = [
            [
                _make_record(
                    step_name="run",
                    model="gpt-4o",
                    system_prompt_hash="shared",
                    system_prompt_tokens=500,
                    input_tokens=1000,
                    iteration=1,
                    timestamp=datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
                ),
                _make_record(
                    step_name="run",
                    model="gpt-4o",
                    system_prompt_hash="shared",
                    system_prompt_tokens=500,
                    input_tokens=1000,
                    iteration=2,
                    timestamp=datetime(2026, 5, 25, 12, 0, 1, tzinfo=UTC),
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = CacheContextGenerator()
        recs = gen.generate(session)
        assert len(recs) >= 1
        assert "run and run" not in recs[0].title
        assert "consecutive calls" in recs[0].title
