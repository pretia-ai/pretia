"""Tests for profile JSON schema validation and export."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from agentcost.schema import (
    _OPTIONAL_FIELDS,
    _REQUIRED_FIELDS,
    profile_json_schema,
    validate_profile,
)
from agentcost.store import ProfilingSession


def _valid_profile() -> dict:
    """Minimal valid profile dict with all required fields."""
    return {
        "workflow_name": "support-agent",
        "workflow_hash": "abc123",
        "profiled_at": "2026-05-20T14:30:00+00:00",
        "sample_size": 20,
        "input_mode": "auto-generate",
        "runs": [],
        "metadata": {"framework": "langgraph"},
    }


# ---------------------------------------------------------------------------
# validate_profile
# ---------------------------------------------------------------------------


class TestValidateProfile:
    def test_valid_minimal_profile(self):
        assert validate_profile(_valid_profile()) == []

    def test_valid_full_profile(self):
        data = _valid_profile()
        data.update(
            {
                "python_version": "3.12.0",
                "sdk_versions": {"langgraph": "1.0"},
                "api_endpoints": {},
                "git_commit_hash": "abc123",
                "git_branch": "main",
                "git_diff_summary": "none",
                "profiling_start_time": "2026-05-20T14:30:00",
                "profiling_end_time": "2026-05-20T14:35:00",
                "inter_request_delay_ms": 100,
                "workflow_id": "support-agent",
                "run_id": "550e8400-e29b-41d4-a716-446655440000",
                "framework": "langgraph",
                "agentcost_version": "0.1.0",
                "profiling_cost": 1.84,
            }
        )
        assert validate_profile(data) == []

    def test_missing_required_field(self):
        data = _valid_profile()
        del data["workflow_name"]
        errors = validate_profile(data)
        assert len(errors) == 1
        assert "workflow_name" in errors[0]

    def test_missing_multiple_required_fields(self):
        data = _valid_profile()
        del data["workflow_name"]
        del data["sample_size"]
        errors = validate_profile(data)
        assert len(errors) == 2

    def test_wrong_type_required_field(self):
        data = _valid_profile()
        data["sample_size"] = "not_an_int"
        errors = validate_profile(data)
        assert any("sample_size" in e and "int" in e for e in errors)

    def test_wrong_type_optional_field(self):
        data = _valid_profile()
        data["python_version"] = 42
        errors = validate_profile(data)
        assert any("python_version" in e for e in errors)

    def test_optional_field_null_is_valid(self):
        data = _valid_profile()
        data["python_version"] = None
        assert validate_profile(data) == []

    def test_optional_fields_absent_is_valid(self):
        assert validate_profile(_valid_profile()) == []

    def test_negative_sample_size(self):
        data = _valid_profile()
        data["sample_size"] = -1
        errors = validate_profile(data)
        assert any("sample_size" in e and ">= 0" in e for e in errors)

    def test_negative_profiling_cost(self):
        data = _valid_profile()
        data["profiling_cost"] = -0.5
        errors = validate_profile(data)
        assert any("profiling_cost" in e and ">= 0" in e for e in errors)

    def test_runs_must_be_list_of_lists(self):
        data = _valid_profile()
        data["runs"] = ["not", "lists"]
        errors = validate_profile(data)
        assert any("runs[0]" in e for e in errors)

    def test_runs_inner_not_list(self):
        data = _valid_profile()
        data["runs"] = [{"not": "a list"}]
        errors = validate_profile(data)
        assert any("runs[0]" in e for e in errors)

    def test_root_not_dict(self):
        errors = validate_profile("not a dict")  # type: ignore[arg-type]
        assert errors == ["Root must be a JSON object (dict)"]

    def test_roundtrip_with_profiling_session(self):
        session = ProfilingSession(
            workflow_name="test-agent",
            workflow_hash="def456",
            profiled_at=datetime(2026, 5, 20, 14, 30, 0, tzinfo=UTC),
            sample_size=10,
            input_mode="single",
            runs=[],
            metadata={"test": True},
        )
        data = session.to_dict()
        assert validate_profile(data) == []


# ---------------------------------------------------------------------------
# profile_json_schema
# ---------------------------------------------------------------------------


class TestProfileJsonSchema:
    def test_returns_dict(self):
        schema = profile_json_schema()
        assert isinstance(schema, dict)

    def test_has_required_key(self):
        schema = profile_json_schema()
        assert "required" in schema
        for field_name in _REQUIRED_FIELDS:
            assert field_name in schema["required"]

    def test_has_properties_for_all_fields(self):
        schema = profile_json_schema()
        all_fields = set(_REQUIRED_FIELDS) | set(_OPTIONAL_FIELDS)
        for field_name in all_fields:
            assert field_name in schema["properties"], f"Missing property: {field_name}"

    def test_schema_is_json_serializable(self):
        schema = profile_json_schema()
        json.dumps(schema)
