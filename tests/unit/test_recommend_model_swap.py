"""Tests for pretia.recommend.model_swap — ModelSwapGenerator."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import calculate_cost
from pretia.recommend.base import _DEFAULT_DAILY_VOLUME
from pretia.recommend.model_swap import (
    ModelSwapGenerator,
    _detect_provider,
    _has_classification_keywords,
)
from pretia.store import ProfilingSession


def _make_record(
    step_name: str = "classify_intent",
    model: str = "claude-opus-4-7",
    input_tokens: int = 1000,
    output_tokens: int = 100,
    output_format: str = "json",
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": 1200,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 200,
        "output_format": output_format,
        "is_retry": False,
        "iteration": 1,
        "parent_step": None,
        "duration_ms": 500,
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


def _classification_runs(
    step_name: str = "classify_intent",
    model: str = "claude-opus-4-7",
    input_tokens: int = 1000,
    output_tokens: int = 100,
    output_format: str = "json",
    n_runs: int = 5,
) -> list[list[StepRecord]]:
    """Build N identical runs for a simple classification step."""
    return [
        [
            _make_record(
                step_name=step_name,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                output_format=output_format,
            )
        ]
        for _ in range(n_runs)
    ]


class TestDetectProvider:
    def test_anthropic(self) -> None:
        assert _detect_provider("claude-opus-4-7") == "anthropic"

    def test_openai_gpt(self) -> None:
        assert _detect_provider("gpt-4o") == "openai"

    def test_openai_o3(self) -> None:
        assert _detect_provider("o3") == "openai"

    def test_openai_o4(self) -> None:
        assert _detect_provider("o4-mini") == "openai"

    def test_google(self) -> None:
        assert _detect_provider("gemini-2.5-pro") == "google"

    def test_deepseek(self) -> None:
        assert _detect_provider("deepseek-v4-flash") == "deepseek"

    def test_mistral(self) -> None:
        assert _detect_provider("mistral-large-latest") == "mistral"

    def test_qwen(self) -> None:
        assert _detect_provider("qwen3.7-max") == "qwen"

    def test_meta(self) -> None:
        assert _detect_provider("llama-4-maverick") == "meta"

    def test_unknown(self) -> None:
        assert _detect_provider("totally-unknown-model") is None


class TestClassificationKeywords:
    def test_classify(self) -> None:
        assert _has_classification_keywords("classify_intent")

    def test_route(self) -> None:
        assert _has_classification_keywords("route_request")

    def test_label(self) -> None:
        assert _has_classification_keywords("label_ticket")

    def test_filter(self) -> None:
        assert _has_classification_keywords("filter_results")

    def test_no_keywords(self) -> None:
        assert not _has_classification_keywords("generate_response")

    def test_case_insensitive(self) -> None:
        assert _has_classification_keywords("ClassifyIntent")


class TestModelSwapClassification:
    """Test: frontier model + JSON output + low ratio → recommend fast tier."""

    def test_frontier_classification_recommends_fast(self) -> None:
        runs = _classification_runs(model="claude-opus-4-7")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.type == "model_swap"
        assert rec.evidence["recommended_model"] == "claude-haiku-4-5"
        assert rec.evidence["recommended_tier"] == "fast"

    def test_savings_calculation_correct(self) -> None:
        runs = _classification_runs(model="claude-opus-4-7")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]

        current_cost = calculate_cost("claude-opus-4-7", 1000, 100)
        recommended_cost = calculate_cost("claude-haiku-4-5", 1000, 100)
        expected_savings = (current_cost - recommended_cost) * _DEFAULT_DAILY_VOLUME * 30

        assert rec.monthly_savings == pytest.approx(expected_savings, rel=1e-6)
        assert rec.monthly_savings > 0

    def test_openai_frontier_recommends_nano(self) -> None:
        runs = _classification_runs(model="gpt-4o")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        assert recs[0].evidence["recommended_model"] == "gpt-4.1-nano"

    def test_deepseek_frontier_recommends_flash(self) -> None:
        runs = _classification_runs(model="deepseek-v4-pro")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        assert recs[0].evidence["recommended_model"] == "deepseek-v4-flash"

    def test_keyword_step_name_with_text_format(self) -> None:
        """Classification keywords + low ratio → classification even with text."""
        runs = _classification_runs(
            step_name="classify_intent",
            model="claude-opus-4-7",
            output_format="text",
            input_tokens=1000,
            output_tokens=200,
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        assert recs[0].evidence["task_classification"] == "classification"


class TestModelSwapExtraction:
    """Test: frontier model + JSON output + medium ratio → recommend mid tier."""

    def test_frontier_extraction_recommends_mid(self) -> None:
        runs = _classification_runs(
            model="claude-opus-4-7",
            input_tokens=1000,
            output_tokens=500,
            output_format="json",
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        rec = recs[0]
        assert rec.evidence["recommended_model"] == "claude-sonnet-4-6"
        assert rec.evidence["task_classification"] == "structured extraction"

    def test_mid_tier_extraction_no_recommendation(self) -> None:
        """Mid model doing extraction → no recommendation (already mid)."""
        runs = _classification_runs(
            model="claude-sonnet-4-6",
            input_tokens=1000,
            output_tokens=500,
            output_format="json",
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 0


class TestModelSwapNoRecommendation:
    """Test cases where no recommendation should be produced."""

    def test_fast_tier_no_recommendation(self) -> None:
        runs = _classification_runs(model="claude-haiku-4-5")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_high_ratio_no_recommendation(self) -> None:
        """Output/input ratio >= 1.0 → complex generation → no recommendation."""
        runs = _classification_runs(
            model="claude-opus-4-7",
            input_tokens=100,
            output_tokens=200,
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_code_output_no_recommendation(self) -> None:
        runs = _classification_runs(
            model="claude-opus-4-7",
            output_format="code",
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_text_output_no_keywords_no_recommendation(self) -> None:
        """Text output format + no classification keywords → no recommendation."""
        runs = _classification_runs(
            step_name="generate_response",
            model="claude-opus-4-7",
            output_format="text",
            input_tokens=1000,
            output_tokens=200,
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_unknown_model_no_crash(self) -> None:
        runs = _classification_runs(model="totally-unknown-model-xyz")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_empty_runs_no_crash(self) -> None:
        session = _make_session([])
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0

    def test_tool_step_skipped(self) -> None:
        runs = [
            [
                _make_record(
                    step_name="search_web",
                    step_type="tool",
                    model="gpt-4o",
                    output_format="text",
                )
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 0


class TestModelSwapConfidence:
    def test_high_confidence_classification_with_keywords(self) -> None:
        """Low ratio + classification keywords → HIGH confidence."""
        runs = _classification_runs(
            step_name="classify_intent",
            model="claude-opus-4-7",
            input_tokens=1000,
            output_tokens=100,
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].confidence == "HIGH"

    def test_moderate_confidence_extraction(self) -> None:
        """Structured extraction → MODERATE confidence."""
        runs = _classification_runs(
            step_name="extract_data",
            model="claude-opus-4-7",
            input_tokens=1000,
            output_tokens=500,
            output_format="json",
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].confidence == "MODERATE"

    def test_moderate_confidence_json_no_keywords(self) -> None:
        """JSON classification but no keywords and ratio not ultra-low → MODERATE."""
        runs = _classification_runs(
            step_name="parse_document",
            model="claude-opus-4-7",
            input_tokens=1000,
            output_tokens=250,
        )
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert len(recs) == 1
        assert recs[0].confidence == "MODERATE"


class TestModelSwapMultipleSteps:
    def test_independent_recommendations_per_step(self) -> None:
        """Multiple LLM steps each get their own recommendation."""
        runs = [
            [
                _make_record(
                    step_name="classify_intent",
                    model="claude-opus-4-7",
                    input_tokens=1000,
                    output_tokens=100,
                    output_format="json",
                ),
                _make_record(
                    step_name="route_request",
                    model="gpt-4o",
                    input_tokens=800,
                    output_tokens=50,
                    output_format="json",
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 2
        ids = {r.id for r in recs}
        assert "model-swap-classify_intent" in ids
        assert "model-swap-route_request" in ids

    def test_mixed_eligible_and_ineligible(self) -> None:
        """Only eligible steps produce recommendations."""
        runs = [
            [
                _make_record(
                    step_name="classify_intent",
                    model="claude-opus-4-7",
                    input_tokens=1000,
                    output_tokens=100,
                    output_format="json",
                ),
                _make_record(
                    step_name="generate_response",
                    model="claude-opus-4-7",
                    input_tokens=500,
                    output_tokens=2000,
                    output_format="text",
                ),
            ]
            for _ in range(5)
        ]
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        assert recs[0].affected_steps == ["classify_intent"]


class TestModelSwapEvidence:
    def test_evidence_fields_present(self) -> None:
        runs = _classification_runs(model="claude-opus-4-7")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)

        assert len(recs) == 1
        ev = recs[0].evidence
        assert "current_model" in ev
        assert "recommended_model" in ev
        assert "current_tier" in ev
        assert "recommended_tier" in ev
        assert "output_input_ratio" in ev
        assert "output_format" in ev
        assert "task_classification" in ev
        assert "savings_per_call" in ev
        assert "daily_volume" in ev
        assert ev["daily_volume"] == _DEFAULT_DAILY_VOLUME

    def test_recommendation_id_format(self) -> None:
        runs = _classification_runs(step_name="my_step", model="claude-opus-4-7")
        session = _make_session(runs)
        gen = ModelSwapGenerator()
        recs = gen.generate(session)
        assert recs[0].id == "model-swap-my_step"
