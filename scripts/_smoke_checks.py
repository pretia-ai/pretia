"""Runtime smoke checks for Stage 4 of validate.py.

Adapted from tests/backtesting/pilot_checks.py for single-run use.
Original functions operated on list[list[StepRecord]] (10 pilot runs);
these accept list[StepRecord] (one run) with tighter thresholds.
"""

from __future__ import annotations

from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import UnrecognizedModelError, calculate_cost
from scripts._validation_types import CheckResult, CheckStatus

_MULTI_PROVIDER_WORKFLOWS = {"W4", "W14", "W15", "W17"}


def _extract_wid(workflow_id: str) -> str:
    parts = workflow_id.split("-")
    if parts:
        candidate = parts[0].upper()
        if candidate.startswith("W") and candidate[1:].isdigit():
            return candidate
    return workflow_id.upper()


def _run_cost(records: list[StepRecord]) -> float:
    total = 0.0
    for r in records:
        total += calculate_cost(
            r.model,
            r.input_tokens,
            r.output_tokens,
            cache_hit_tokens=r.cache_hit_tokens,
            cache_miss_tokens=r.cache_miss_tokens,
        )
    return total


def check_cache_bust(
    workflow_id: str,
    records: list[StepRecord],
) -> CheckResult:
    """Verify no cache hits in DeepSeek/Anthropic records."""
    cache_relevant = [
        r for r in records if "deepseek" in r.model.lower() or "claude" in r.model.lower()
    ]

    if not cache_relevant:
        return CheckResult(
            name="cache_bust",
            status=CheckStatus.PASS,
            details={"note": "not applicable — no DeepSeek/Anthropic models"},
            blocking=True,
        )

    violations = [
        {"step_name": r.step_name, "model": r.model, "cache_hit_tokens": r.cache_hit_tokens}
        for r in cache_relevant
        if r.cache_hit_tokens is not None and r.cache_hit_tokens > 0
    ]

    if violations:
        return CheckResult(
            name="cache_bust",
            status=CheckStatus.FAIL,
            details={"violation_count": len(violations), "violations": violations[:10]},
            blocking=True,
        )

    return CheckResult(
        name="cache_bust",
        status=CheckStatus.PASS,
        details={"checked_records": len(cache_relevant)},
        blocking=True,
    )


def check_template_substitution(
    records: list[StepRecord],
    input_data: dict[str, Any],
) -> CheckResult:
    """Detect unresolved template placeholders in a single input and step names."""
    found: list[dict[str, str]] = []

    for key, value in input_data.items():
        if isinstance(value, str) and ("{{" in value or "}}" in value):
            found.append({"location": f"input.{key}", "value": value[:200]})

    for r in records:
        if "{{" in r.step_name or "}}" in r.step_name:
            found.append({"location": "step_name", "value": r.step_name})

    if found:
        return CheckResult(
            name="template_substitution",
            status=CheckStatus.FAIL,
            details={"placeholder_count": len(found), "examples": found[:10]},
            blocking=True,
        )

    return CheckResult(
        name="template_substitution",
        status=CheckStatus.PASS,
        details={"fields_checked": len(input_data)},
        blocking=True,
    )


def check_cross_provider_accounting(
    workflow_id: str,
    records: list[StepRecord],
) -> CheckResult:
    """Verify every model used in multi-provider workflows has valid pricing."""
    wid = _extract_wid(workflow_id)

    if wid not in _MULTI_PROVIDER_WORKFLOWS:
        return CheckResult(
            name="cross_provider_accounting",
            status=CheckStatus.PASS,
            details={"note": "not applicable — single-provider workflow"},
            blocking=True,
        )

    models_seen: set[str] = set()
    unrecognized: list[str] = []

    for r in records:
        if r.model in models_seen:
            continue
        models_seen.add(r.model)
        try:
            calculate_cost(r.model, r.input_tokens, r.output_tokens)
        except (UnrecognizedModelError, ValueError):
            unrecognized.append(r.model)

    if unrecognized:
        return CheckResult(
            name="cross_provider_accounting",
            status=CheckStatus.FAIL,
            details={"unrecognized_models": unrecognized, "all_models": sorted(models_seen)},
            blocking=True,
        )

    return CheckResult(
        name="cross_provider_accounting",
        status=CheckStatus.PASS,
        details={"models_verified": sorted(models_seen)},
        blocking=True,
    )


def check_finish_reason(
    records: list[StepRecord],
) -> CheckResult:
    """Flag any output truncation in a single run (stricter than pilot 15% threshold)."""
    if not records:
        return CheckResult(
            name="finish_reason",
            status=CheckStatus.PASS,
            details={"total_records": 0, "truncated_count": 0},
            blocking=True,
        )

    truncated_count = sum(1 for r in records if r.output_truncated is True)

    if truncated_count > 0:
        return CheckResult(
            name="finish_reason",
            status=CheckStatus.FAIL,
            details={
                "total_records": len(records),
                "truncated_count": truncated_count,
            },
            blocking=True,
        )

    return CheckResult(
        name="finish_reason",
        status=CheckStatus.PASS,
        details={"total_records": len(records), "truncated_count": 0},
        blocking=True,
    )


def check_output_schema(
    records: list[StepRecord],
) -> CheckResult:
    """Detect JSON output truncation in a single run."""
    json_steps = [r for r in records if r.output_format == "json"]

    if not json_steps:
        return CheckResult(
            name="output_schema",
            status=CheckStatus.PASS,
            details={"json_step_count": 0, "truncated_count": 0},
            blocking=True,
        )

    truncated_count = sum(1 for r in json_steps if r.output_truncated is True)

    if truncated_count > 0:
        return CheckResult(
            name="output_schema",
            status=CheckStatus.FAIL,
            details={"json_step_count": len(json_steps), "truncated_count": truncated_count},
            blocking=True,
        )

    return CheckResult(
        name="output_schema",
        status=CheckStatus.PASS,
        details={"json_step_count": len(json_steps), "truncated_count": 0},
        blocking=True,
    )


def check_cost_plausibility(
    records: list[StepRecord],
    expected_cost_range: tuple[float, float],
) -> CheckResult:
    """Verify single-run cost falls within a plausible range."""
    if not records:
        return CheckResult(
            name="cost_plausibility",
            status=CheckStatus.WARN,
            details={"note": "no records to compute cost"},
            blocking=True,
        )

    try:
        cost = _run_cost(records)
    except (UnrecognizedModelError, ValueError) as exc:
        return CheckResult(
            name="cost_plausibility",
            status=CheckStatus.WARN,
            details={"note": f"cost computation failed: {exc}"},
            blocking=True,
        )

    expected_mid = (expected_cost_range[0] + expected_cost_range[1]) / 2.0
    low = expected_mid * 0.5
    high = expected_mid * 5.0

    in_range = low <= cost <= high
    status = CheckStatus.PASS if in_range else CheckStatus.WARN

    return CheckResult(
        name="cost_plausibility",
        status=status,
        details={
            "run_cost": round(cost, 6),
            "expected_range": [round(expected_cost_range[0], 6), round(expected_cost_range[1], 6)],
            "plausibility_bounds": [round(low, 6), round(high, 6)],
        },
        blocking=True,
    )


def check_nonzero_tokens(
    records: list[StepRecord],
) -> CheckResult:
    """Verify all records have non-zero input and output tokens."""
    if not records:
        return CheckResult(
            name="nonzero_tokens",
            status=CheckStatus.PASS,
            details={"records_checked": 0},
            blocking=True,
        )

    zero_input = [r.step_name for r in records if r.input_tokens == 0]
    zero_output = [r.step_name for r in records if r.output_tokens == 0]

    if zero_input or zero_output:
        return CheckResult(
            name="nonzero_tokens",
            status=CheckStatus.FAIL,
            details={
                "zero_input_steps": zero_input[:10],
                "zero_output_steps": zero_output[:10],
            },
            blocking=True,
        )

    return CheckResult(
        name="nonzero_tokens",
        status=CheckStatus.PASS,
        details={"records_checked": len(records)},
        blocking=True,
    )


def check_step_count(
    workflow_id: str,
    records: list[StepRecord],
    expected_range: tuple[int, int],
) -> CheckResult:
    """Verify step count falls within expected range."""
    count = len(records)
    low, high = expected_range

    if low <= count <= high:
        return CheckResult(
            name="step_count",
            status=CheckStatus.PASS,
            details={"step_count": count, "expected_range": [low, high]},
            blocking=True,
        )

    return CheckResult(
        name="step_count",
        status=CheckStatus.FAIL,
        details={
            "step_count": count,
            "expected_range": [low, high],
            "workflow_id": workflow_id,
        },
        blocking=True,
    )


def run_smoke_checks(
    workflow_id: str,
    records: list[StepRecord],
    input_data: dict[str, Any],
    expected_cost_range: tuple[float, float],
    expected_step_range: tuple[int, int],
) -> list[CheckResult]:
    """Run all smoke checks for one workflow's single-run output."""
    return [
        check_cache_bust(workflow_id, records),
        check_template_substitution(records, input_data),
        check_cross_provider_accounting(workflow_id, records),
        check_finish_reason(records),
        check_output_schema(records),
        check_cost_plausibility(records, expected_cost_range),
        check_nonzero_tokens(records),
        check_step_count(workflow_id, records, expected_step_range),
    ]
