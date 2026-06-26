"""Profile JSON schema definition and validation."""

from __future__ import annotations

from typing import Any

_REQUIRED_FIELDS: dict[str, type | tuple[type, ...]] = {
    "workflow_name": str,
    "workflow_hash": str,
    "profiled_at": str,
    "sample_size": int,
    "input_mode": str,
    "runs": list,
    "metadata": dict,
}

_OPTIONAL_FIELDS: dict[str, type | tuple[type, ...]] = {
    "python_version": str,
    "sdk_versions": dict,
    "api_endpoints": dict,
    "git_commit_hash": str,
    "git_branch": str,
    "git_diff_summary": str,
    "profiling_start_time": str,
    "profiling_end_time": str,
    "inter_request_delay_ms": int,
    "workflow_id": str,
    "run_id": str,
    "framework": str,
    "pretia_version": str,
    "profiling_cost": (int, float),
}


def validate_profile(data: Any) -> list[str]:
    """Validate a profile JSON dict and return a list of errors.

    An empty list means the data is valid.
    """
    if not isinstance(data, dict):
        return ["Root must be a JSON object (dict)"]

    errors: list[str] = []

    for field_name, expected_type in _REQUIRED_FIELDS.items():
        if field_name not in data:
            errors.append(f"Missing required field: {field_name!r}")
        elif not isinstance(data[field_name], expected_type):
            errors.append(
                f"Field {field_name!r} must be {_type_label(expected_type)}, "
                f"got {type(data[field_name]).__name__}"
            )

    for field_name, expected_type in _OPTIONAL_FIELDS.items():
        if field_name in data and data[field_name] is not None:
            if not isinstance(data[field_name], expected_type):
                errors.append(
                    f"Field {field_name!r} must be {_type_label(expected_type)} or null, "
                    f"got {type(data[field_name]).__name__}"
                )

    if "sample_size" in data and isinstance(data["sample_size"], int):
        if data["sample_size"] < 0:
            errors.append("Field 'sample_size' must be >= 0")

    if (
        "profiling_cost" in data
        and data["profiling_cost"] is not None
        and isinstance(data["profiling_cost"], (int, float))
    ):
        if data["profiling_cost"] < 0:
            errors.append("Field 'profiling_cost' must be >= 0")

    if "runs" in data and isinstance(data["runs"], list):
        for i, run in enumerate(data["runs"]):
            if not isinstance(run, list):
                errors.append(f"runs[{i}] must be a list, got {type(run).__name__}")

    return errors


def _type_label(t: type | tuple[type, ...]) -> str:
    if isinstance(t, tuple):
        return " or ".join(cls.__name__ for cls in t)
    return t.__name__


def profile_json_schema() -> dict[str, Any]:
    """Export the profile schema as a JSON Schema dict for documentation."""
    properties: dict[str, Any] = {}

    _json_type_map: dict[type, str] = {
        str: "string",
        int: "integer",
        float: "number",
        list: "array",
        dict: "object",
        bool: "boolean",
    }

    for field_name, expected_type in _REQUIRED_FIELDS.items():
        json_type = _json_type_map.get(expected_type, "string")  # type: ignore[arg-type]
        prop: dict[str, Any] = {"type": json_type}
        if field_name == "profiled_at":
            prop["format"] = "date-time"
        if field_name == "sample_size":
            prop["minimum"] = 0
        if field_name == "runs":
            prop["items"] = {"type": "array"}
        properties[field_name] = prop

    for field_name, expected_type in _OPTIONAL_FIELDS.items():
        if isinstance(expected_type, tuple):
            json_types = [_json_type_map.get(t, "string") for t in expected_type]
            json_types.append("null")
            properties[field_name] = {"type": json_types}
        else:
            json_type = _json_type_map.get(expected_type, "string")
            properties[field_name] = {"type": [json_type, "null"]}
        if field_name == "profiling_cost":
            properties[field_name]["minimum"] = 0

    return {
        "$schema": "https://json-schema.org/draft-07/schema#",
        "title": "Pretia ProfilingSession",
        "type": "object",
        "required": list(_REQUIRED_FIELDS.keys()),
        "properties": properties,
    }
