"""Tests for StepRecord: construction, validation, frozen behavior, serialization, cost."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord


def test_field_access(sample_record):
    assert sample_record.step_name == "classify_intent"
    assert sample_record.step_type == "llm"
    assert sample_record.model == "claude-haiku-3"
    assert sample_record.input_tokens == 340
    assert sample_record.output_tokens == 45
    assert sample_record.context_size == 620
    assert sample_record.tool_definitions_tokens == 0
    assert sample_record.system_prompt_hash == "a3f8c2d1e5b9"
    assert sample_record.system_prompt_tokens == 280
    assert sample_record.output_format == "json"
    assert sample_record.is_retry is False
    assert sample_record.iteration == 1
    assert sample_record.parent_step is None
    assert sample_record.duration_ms == 230
    assert sample_record.timestamp == datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC)


def test_total_tokens(sample_record):
    assert sample_record.total_tokens == sample_record.input_tokens + sample_record.output_tokens
    assert sample_record.total_tokens == 385


class TestValidation:
    def test_invalid_step_type_raises(self, sample_record):
        with pytest.raises(ValueError, match="step_type"):
            dataclasses.replace(sample_record, step_type="invalid")

    def test_invalid_output_format_raises(self, sample_record):
        with pytest.raises(ValueError, match="output_format"):
            dataclasses.replace(sample_record, output_format="xml")

    def test_negative_input_tokens_raises(self, sample_record):
        with pytest.raises(ValueError, match="input_tokens"):
            dataclasses.replace(sample_record, input_tokens=-1)

    def test_negative_output_tokens_raises(self, sample_record):
        with pytest.raises(ValueError, match="output_tokens"):
            dataclasses.replace(sample_record, output_tokens=-1)

    def test_negative_context_size_raises(self, sample_record):
        with pytest.raises(ValueError, match="context_size"):
            dataclasses.replace(sample_record, context_size=-1)

    def test_negative_duration_raises(self, sample_record):
        with pytest.raises(ValueError, match="duration_ms"):
            dataclasses.replace(sample_record, duration_ms=-1)

    def test_zero_iteration_raises(self, sample_record):
        with pytest.raises(ValueError, match="iteration"):
            dataclasses.replace(sample_record, iteration=0)

    @pytest.mark.parametrize("step_type", ["llm", "tool", "retrieval"])
    def test_valid_step_types_accepted(self, sample_record, step_type):
        record = dataclasses.replace(sample_record, step_type=step_type)
        assert record.step_type == step_type

    @pytest.mark.parametrize("output_format", ["json", "text", "code"])
    def test_valid_output_formats_accepted(self, sample_record, output_format):
        record = dataclasses.replace(sample_record, output_format=output_format)
        assert record.output_format == output_format


def test_record_is_frozen(sample_record):
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample_record.step_name = "different"


class TestSerialization:
    def test_to_dict_from_dict_round_trip(self, sample_record):
        restored = StepRecord.from_dict(sample_record.to_dict())
        assert restored == sample_record

    def test_to_dict_is_json_serializable(self, sample_record):
        # json.dumps would raise TypeError if a datetime or other non-JSON value leaked through.
        json.dumps(sample_record.to_dict())

    def test_timestamp_serialized_as_iso8601(self, sample_record):
        d = sample_record.to_dict()
        assert d["timestamp"] == "2026-05-20T14:30:00+00:00"
        assert datetime.fromisoformat(d["timestamp"]) == sample_record.timestamp

    def test_round_trip_preserves_none_parent(self, sample_record):
        assert sample_record.parent_step is None
        restored = StepRecord.from_dict(sample_record.to_dict())
        assert restored.parent_step is None

    def test_round_trip_preserves_string_parent(self, sample_record):
        record = dataclasses.replace(sample_record, parent_step="planner")
        restored = StepRecord.from_dict(record.to_dict())
        assert restored.parent_step == "planner"


class TestCost:
    def test_cost_calculation(self, sample_record):
        pricing = {"claude-haiku-3": (1e-6, 5e-6)}
        expected = 340 * 1e-6 + 45 * 5e-6
        assert sample_record.cost(pricing) == pytest.approx(expected)

    def test_cost_raises_for_unknown_model(self, sample_record):
        with pytest.raises(ValueError, match="claude-haiku-3"):
            sample_record.cost({"gpt-4o": (1e-6, 5e-6)})

    def test_cost_error_lists_available_models(self, sample_record):
        with pytest.raises(ValueError, match="gpt-4o"):
            sample_record.cost({"gpt-4o": (1e-6, 5e-6)})
