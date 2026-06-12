"""Run pilot calibration checks after 10 pilot runs of a workflow.

Validate that infrastructure works (Layer 1, blocking) and that cost
distributions are plausible (Layer 2, non-blocking).
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass
from typing import Any

from agentcost.collectors.base import StepRecord
from agentcost.pricing.tables import UnrecognizedModelError, calculate_cost

logger = logging.getLogger(__name__)

# Workflow IDs that use multiple providers — cross-provider accounting applies.
_MULTI_PROVIDER_WORKFLOWS = {"W4", "W14", "W15", "W17"}

# Workflow groups for tier-separation thresholds.
_LINEAR_WORKFLOWS = {"W1", "W5", "W9", "W11", "W12", "W18"}
_LOOP_WORKFLOWS = {"W2", "W4", "W15", "W19"}
_ROUTING_WORKFLOWS = {"W13", "W17"}
_RAG_WORKFLOWS = {"W14", "W16"}

# Workflows with routing behavior worth checking path diversity.
_ROUTING_CHECK_WORKFLOWS = {"W1", "W2", "W13", "W17"}

# Workflows that process PDFs.
_PDF_WORKFLOWS = {"W14", "W15", "W16", "W17", "W18"}


@dataclass(frozen=True, slots=True)
class PilotCheckResult:
    """Outcome of a single pilot calibration check."""

    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    details: dict[str, Any]
    layer: int  # 1 or 2
    blocking: bool  # Layer 1 = True, Layer 2 = False


def _extract_workflow_id(workflow_id: str) -> str:
    """Extract the short workflow ID (e.g. 'W13') from a full name."""
    parts = workflow_id.split("-")
    if parts:
        candidate = parts[0].upper()
        if candidate.startswith("W") and candidate[1:].isdigit():
            return candidate
    return workflow_id.upper()


def _flat_records(all_records: list[list[StepRecord]]) -> list[StepRecord]:
    """Flatten nested run records into a single list."""
    return [r for run in all_records for r in run]


def _run_cost(run: list[StepRecord]) -> float:
    """Compute total cost for a single run."""
    total = 0.0
    for r in run:
        total += calculate_cost(
            r.model,
            r.input_tokens,
            r.output_tokens,
            cache_hit_tokens=r.cache_hit_tokens,
            cache_miss_tokens=r.cache_miss_tokens,
        )
    return total


# ---------------------------------------------------------------------------
# Layer 1 checks (infrastructure, blocking)
# ---------------------------------------------------------------------------


def check_cache_bust(
    workflow_id: str,
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Verify cache busting is effective for DeepSeek/Anthropic providers."""
    flat = _flat_records(all_records)
    cache_relevant = [
        r
        for r in flat
        if "deepseek" in r.model.lower() or "claude" in r.model.lower()
    ]

    if not cache_relevant:
        return PilotCheckResult(
            name="cache_bust",
            status="PASS",
            details={"note": "not applicable — no DeepSeek/Anthropic models in workflow"},
            layer=1,
            blocking=True,
        )

    violations = []
    for r in cache_relevant:
        if r.cache_hit_tokens is not None and r.cache_hit_tokens > 0:
            violations.append(
                {
                    "step_name": r.step_name,
                    "model": r.model,
                    "cache_hit_tokens": r.cache_hit_tokens,
                }
            )

    if violations:
        return PilotCheckResult(
            name="cache_bust",
            status="FAIL",
            details={
                "violation_count": len(violations),
                "violations": violations[:10],
            },
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="cache_bust",
        status="PASS",
        details={"checked_records": len(cache_relevant)},
        layer=1,
        blocking=True,
    )


def check_template_substitution(
    all_records: list[list[StepRecord]],
    inputs: list[dict[str, Any]],
) -> PilotCheckResult:
    """Detect unresolved template placeholders in inputs and step names."""
    found: list[dict[str, str]] = []

    for idx, inp in enumerate(inputs):
        for key, value in inp.items():
            if isinstance(value, str) and ("{{" in value or "}}" in value):
                found.append(
                    {"location": f"input[{idx}].{key}", "value": value[:200]}
                )

    flat = _flat_records(all_records)
    for r in flat:
        if "{{" in r.step_name or "}}" in r.step_name:
            found.append({"location": "step_name", "value": r.step_name})

    if found:
        return PilotCheckResult(
            name="template_substitution",
            status="FAIL",
            details={"placeholder_count": len(found), "examples": found[:10]},
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="template_substitution",
        status="PASS",
        details={"inputs_checked": len(inputs)},
        layer=1,
        blocking=True,
    )


def check_cross_provider_accounting(
    workflow_id: str,
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Verify every model used in multi-provider workflows has valid pricing."""
    wid = _extract_workflow_id(workflow_id)

    if wid not in _MULTI_PROVIDER_WORKFLOWS:
        return PilotCheckResult(
            name="cross_provider_accounting",
            status="PASS",
            details={"note": "not applicable — single-provider workflow"},
            layer=1,
            blocking=True,
        )

    flat = _flat_records(all_records)
    models_seen: set[str] = set()
    unrecognized: list[str] = []

    for r in flat:
        if r.model in models_seen:
            continue
        models_seen.add(r.model)
        try:
            calculate_cost(r.model, r.input_tokens, r.output_tokens)
        except (UnrecognizedModelError, ValueError):
            unrecognized.append(r.model)

    if unrecognized:
        return PilotCheckResult(
            name="cross_provider_accounting",
            status="FAIL",
            details={
                "unrecognized_models": unrecognized,
                "all_models": sorted(models_seen),
            },
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="cross_provider_accounting",
        status="PASS",
        details={"models_verified": sorted(models_seen)},
        layer=1,
        blocking=True,
    )


def check_w19_history_accumulation(
    workflow_id: str,
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Verify W19 multi-turn history causes monotonically growing input tokens."""
    wid = _extract_workflow_id(workflow_id)

    if wid != "W19":
        return PilotCheckResult(
            name="w19_history_accumulation",
            status="PASS",
            details={"note": "not applicable — only checked for W19"},
            layer=1,
            blocking=True,
        )

    failures: list[dict[str, Any]] = []

    for run_idx, run in enumerate(all_records):
        if len(run) < 2:
            continue
        first_input = run[0].input_tokens
        last_input = run[-1].input_tokens

        if first_input == 0:
            failures.append(
                {
                    "run": run_idx,
                    "reason": "first record has 0 input_tokens",
                    "first_input": first_input,
                    "last_input": last_input,
                }
            )
            continue

        ratio = last_input / first_input
        if ratio < 1.3:
            failures.append(
                {
                    "run": run_idx,
                    "reason": f"last/first ratio {ratio:.2f} < 1.3",
                    "first_input": first_input,
                    "last_input": last_input,
                    "ratio": ratio,
                }
            )

    if failures:
        return PilotCheckResult(
            name="w19_history_accumulation",
            status="FAIL",
            details={
                "failure_count": len(failures),
                "failures": failures[:10],
                "total_runs": len(all_records),
            },
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="w19_history_accumulation",
        status="PASS",
        details={"runs_checked": len(all_records)},
        layer=1,
        blocking=True,
    )


def check_pdf_validity(
    workflow_id: str,
    inputs: list[dict[str, Any]],
) -> PilotCheckResult:
    """Verify PDF inputs can be opened for document-processing workflows."""
    wid = _extract_workflow_id(workflow_id)

    if wid not in _PDF_WORKFLOWS:
        return PilotCheckResult(
            name="pdf_validity",
            status="PASS",
            details={"note": "not applicable — not a PDF workflow"},
            layer=1,
            blocking=True,
        )

    pdf_paths: list[str] = []
    for inp in inputs:
        for key in ("pdf_path", "document_path"):
            val = inp.get(key)
            if isinstance(val, str) and val:
                pdf_paths.append(val)

    if not pdf_paths:
        return PilotCheckResult(
            name="pdf_validity",
            status="PASS",
            details={"note": "no PDF paths found in inputs"},
            layer=1,
            blocking=True,
        )

    try:
        import pdfplumber  # noqa: F401
    except ImportError:
        return PilotCheckResult(
            name="pdf_validity",
            status="WARN",
            details={
                "note": "pdfplumber not installed — cannot validate PDFs",
                "pdf_paths": pdf_paths,
            },
            layer=1,
            blocking=True,
        )

    invalid: list[dict[str, str]] = []
    for path in pdf_paths:
        try:
            with pdfplumber.open(path) as pdf:
                _ = len(pdf.pages)
        except Exception as exc:
            invalid.append({"path": path, "error": str(exc)})

    if invalid:
        return PilotCheckResult(
            name="pdf_validity",
            status="FAIL",
            details={
                "invalid_count": len(invalid),
                "invalid": invalid,
                "total_pdfs": len(pdf_paths),
            },
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="pdf_validity",
        status="PASS",
        details={"pdfs_validated": len(pdf_paths)},
        layer=1,
        blocking=True,
    )


def check_output_schema(
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Detect JSON output truncation that suggests schema violations."""
    flat = _flat_records(all_records)

    json_steps = [r for r in flat if r.output_format == "json"]
    json_step_count = len(json_steps)

    if json_step_count == 0:
        return PilotCheckResult(
            name="output_schema",
            status="PASS",
            details={"json_step_count": 0, "truncated_count": 0},
            layer=1,
            blocking=True,
        )

    truncated_count = sum(
        1 for r in json_steps if r.output_truncated is True
    )
    truncated_pct = truncated_count / json_step_count

    if truncated_pct > 0.20:
        return PilotCheckResult(
            name="output_schema",
            status="FAIL",
            details={
                "json_step_count": json_step_count,
                "truncated_count": truncated_count,
                "truncated_pct": round(truncated_pct * 100, 1),
            },
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="output_schema",
        status="PASS",
        details={
            "json_step_count": json_step_count,
            "truncated_count": truncated_count,
        },
        layer=1,
        blocking=True,
    )


def check_finish_reason(
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Flag excessive output truncation across all records."""
    flat = _flat_records(all_records)
    total_records = len(flat)

    if total_records == 0:
        return PilotCheckResult(
            name="finish_reason",
            status="PASS",
            details={"total_records": 0, "truncated_count": 0, "truncated_pct": 0.0},
            layer=1,
            blocking=True,
        )

    truncated_count = sum(1 for r in flat if r.output_truncated is True)
    truncated_pct = truncated_count / total_records

    if truncated_pct > 0.15:
        return PilotCheckResult(
            name="finish_reason",
            status="FAIL",
            details={
                "total_records": total_records,
                "truncated_count": truncated_count,
                "truncated_pct": round(truncated_pct * 100, 1),
            },
            layer=1,
            blocking=True,
        )

    return PilotCheckResult(
        name="finish_reason",
        status="PASS",
        details={
            "total_records": total_records,
            "truncated_count": truncated_count,
            "truncated_pct": round(truncated_pct * 100, 1),
        },
        layer=1,
        blocking=True,
    )


# ---------------------------------------------------------------------------
# Layer 2 checks (cost plausibility, non-blocking)
# ---------------------------------------------------------------------------


def check_tier_separation(
    workflow_id: str,
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Verify max/min cost ratio meets the threshold for the workflow group."""
    wid = _extract_workflow_id(workflow_id)

    if wid in _LINEAR_WORKFLOWS:
        threshold = 2.0
    elif wid in _LOOP_WORKFLOWS:
        threshold = 5.0
    elif wid in _ROUTING_WORKFLOWS:
        threshold = 10.0
    elif wid in _RAG_WORKFLOWS:
        threshold = 3.0
    else:
        threshold = 2.0

    run_costs = []
    for run in all_records:
        try:
            run_costs.append(_run_cost(run))
        except (UnrecognizedModelError, ValueError) as exc:
            logger.warning("Skipping run for tier separation: %s", exc)

    if len(run_costs) < 2:
        return PilotCheckResult(
            name="tier_separation",
            status="WARN",
            details={"note": "fewer than 2 valid runs — cannot compute ratio"},
            layer=2,
            blocking=False,
        )

    min_cost = min(run_costs)
    max_cost = max(run_costs)

    if min_cost <= 0:
        ratio = float("inf") if max_cost > 0 else 1.0
    else:
        ratio = max_cost / min_cost

    status = "PASS" if ratio >= threshold else "WARN"

    return PilotCheckResult(
        name="tier_separation",
        status=status,
        details={
            "min_cost": round(min_cost, 6),
            "max_cost": round(max_cost, 6),
            "ratio": round(ratio, 2),
            "threshold": threshold,
        },
        layer=2,
        blocking=False,
    )


def check_routing_ratio(
    workflow_id: str,
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Verify routing workflows exercise multiple paths across pilot runs."""
    wid = _extract_workflow_id(workflow_id)

    if wid not in _ROUTING_CHECK_WORKFLOWS:
        return PilotCheckResult(
            name="routing_ratio",
            status="PASS",
            details={"note": "not applicable — not a routing workflow"},
            layer=2,
            blocking=False,
        )

    # Detect distinct paths by collecting the set of step names per run.
    path_signatures: list[frozenset[str]] = []

    # Path indicator keywords by workflow.
    path_keywords: dict[str, list[str]] = {
        "W13": ["simple", "research", "escalate", "path_a", "path_b", "path_c"],
        "W17": ["simple", "research", "escalate", "path_a", "path_b", "path_c"],
        "W1": ["simple", "complex", "fallback"],
        "W2": ["simple", "complex", "fallback"],
    }

    keywords = path_keywords.get(wid, [])

    for run in all_records:
        step_names = {r.step_name.lower() for r in run}
        if keywords:
            # Build a signature from which keywords appear in step names.
            sig = frozenset(
                kw for kw in keywords if any(kw in sn for sn in step_names)
            )
        else:
            sig = frozenset(step_names)
        path_signatures.append(sig)

    distinct_paths = len(set(path_signatures))

    if distinct_paths >= 2:
        return PilotCheckResult(
            name="routing_ratio",
            status="PASS",
            details={
                "distinct_paths": distinct_paths,
                "total_runs": len(all_records),
            },
            layer=2,
            blocking=False,
        )

    return PilotCheckResult(
        name="routing_ratio",
        status="WARN",
        details={
            "distinct_paths": distinct_paths,
            "total_runs": len(all_records),
            "note": "only 1 path observed — routing diversity may be insufficient",
        },
        layer=2,
        blocking=False,
    )


def check_cost_plausibility(
    all_records: list[list[StepRecord]],
    config: Any,
) -> PilotCheckResult:
    """Verify per-run costs fall within a plausible range of the expected cost."""
    expected_range: tuple[float, float] = config.expected_cost_range
    expected_cost = (expected_range[0] + expected_range[1]) / 2.0

    run_costs: list[float] = []
    for run in all_records:
        try:
            run_costs.append(_run_cost(run))
        except (UnrecognizedModelError, ValueError) as exc:
            logger.warning("Skipping run for cost plausibility: %s", exc)

    if not run_costs:
        return PilotCheckResult(
            name="cost_plausibility",
            status="WARN",
            details={"note": "no valid run costs computed"},
            layer=2,
            blocking=False,
        )

    low_bound = expected_cost * 0.5
    high_bound = expected_cost * 5.0

    out_of_range = [c for c in run_costs if c < low_bound or c > high_bound]
    out_of_range_count = len(out_of_range)

    status = "WARN" if out_of_range_count > 0 else "PASS"

    return PilotCheckResult(
        name="cost_plausibility",
        status=status,
        details={
            "expected_cost": round(expected_cost, 6),
            "expected_range": [round(expected_range[0], 6), round(expected_range[1], 6)],
            "plausibility_bounds": [round(low_bound, 6), round(high_bound, 6)],
            "per_run_costs": [round(c, 6) for c in run_costs],
            "out_of_range_count": out_of_range_count,
        },
        layer=2,
        blocking=False,
    )


def check_detector_preactivation(
    workflow_id: str,
    all_records: list[list[StepRecord]],
) -> PilotCheckResult:
    """Verify that data prerequisites exist for expected detector activations."""
    wid = _extract_workflow_id(workflow_id)

    try:
        from visualization.colors import EXPECTED_DETECTORS
    except ImportError:
        return PilotCheckResult(
            name="detector_preactivation",
            status="WARN",
            details={"note": "visualization.colors not importable — skipping"},
            layer=2,
            blocking=False,
        )

    expected = EXPECTED_DETECTORS.get(wid)
    if expected is None:
        return PilotCheckResult(
            name="detector_preactivation",
            status="PASS",
            details={"note": f"no expected detectors defined for {wid}"},
            layer=2,
            blocking=False,
        )

    active_detectors = {k: v for k, v in expected.items() if v}
    if not active_detectors:
        return PilotCheckResult(
            name="detector_preactivation",
            status="PASS",
            details={"note": "no detectors expected to fire for this workflow"},
            layer=2,
            blocking=False,
        )

    prereqs_met: dict[str, bool] = {}
    prereq_details: dict[str, str] = {}

    for detector in active_detectors:
        if detector == "context_growth":
            prereqs_met[detector], prereq_details[detector] = (
                _check_context_growth_prereq(all_records)
            )
        elif detector == "loop_count_variance":
            prereqs_met[detector], prereq_details[detector] = (
                _check_loop_variance_prereq(all_records)
            )
        elif detector == "bimodality":
            prereqs_met[detector], prereq_details[detector] = (
                _check_bimodality_prereq(all_records)
            )
        elif detector == "step_count_variance":
            prereqs_met[detector], prereq_details[detector] = (
                _check_step_count_variance_prereq(all_records)
            )
        elif detector == "high_token_variance":
            prereqs_met[detector] = True
            prereq_details[detector] = "no specific prerequisite check"
        else:
            prereqs_met[detector] = True
            prereq_details[detector] = "unknown detector — assumed met"

    all_met = all(prereqs_met.values())
    missing = [d for d, met in prereqs_met.items() if not met]

    status = "PASS" if all_met else "WARN"

    return PilotCheckResult(
        name="detector_preactivation",
        status=status,
        details={
            "expected_detectors": list(active_detectors.keys()),
            "prerequisites_met": prereqs_met,
            "prerequisite_details": prereq_details,
            "missing_prerequisites": missing,
        },
        layer=2,
        blocking=False,
    )


def _check_context_growth_prereq(
    all_records: list[list[StepRecord]],
) -> tuple[bool, str]:
    """Check for a positive trend in input_tokens across iterations within runs."""
    positive_trend_runs = 0
    for run in all_records:
        if len(run) < 3:
            continue
        tokens = [r.input_tokens for r in run]
        # Simple check: are later tokens generally larger than earlier ones?
        first_half = tokens[: len(tokens) // 2]
        second_half = tokens[len(tokens) // 2 :]
        if statistics.mean(second_half) > statistics.mean(first_half):
            positive_trend_runs += 1

    if positive_trend_runs >= len(all_records) // 2:
        return True, f"{positive_trend_runs}/{len(all_records)} runs show positive trend"
    return (
        False,
        f"only {positive_trend_runs}/{len(all_records)} runs show positive input_token trend",
    )


def _check_loop_variance_prereq(
    all_records: list[list[StepRecord]],
) -> tuple[bool, str]:
    """Check if max iteration varies across runs with range >= 3."""
    max_iters = []
    for run in all_records:
        if run:
            max_iters.append(max(r.iteration for r in run))

    if len(max_iters) < 2:
        return False, "fewer than 2 runs"

    iter_range = max(max_iters) - min(max_iters)
    if iter_range >= 3:
        return True, f"iteration range = {iter_range} (min={min(max_iters)}, max={max(max_iters)})"
    return (
        False,
        f"iteration range = {iter_range} < 3 (min={min(max_iters)}, max={max(max_iters)})",
    )


def _check_bimodality_prereq(
    all_records: list[list[StepRecord]],
) -> tuple[bool, str]:
    """Check for at least 2 expensive runs and 5 cheap runs using cost median split."""
    run_costs: list[float] = []
    for run in all_records:
        try:
            run_costs.append(_run_cost(run))
        except (UnrecognizedModelError, ValueError):
            continue

    if len(run_costs) < 7:
        return False, f"only {len(run_costs)} valid runs — need at least 7"

    median_cost = statistics.median(run_costs)
    expensive = sum(1 for c in run_costs if c > median_cost)
    cheap = sum(1 for c in run_costs if c <= median_cost)

    if expensive >= 2 and cheap >= 5:
        return True, f"{expensive} expensive, {cheap} cheap (median={median_cost:.6f})"
    return (
        False,
        f"need >=2 expensive and >=5 cheap, got {expensive} expensive, {cheap} cheap",
    )


def _check_step_count_variance_prereq(
    all_records: list[list[StepRecord]],
) -> tuple[bool, str]:
    """Check if at least 2 different step counts appear across runs."""
    step_counts = {len(run) for run in all_records}

    if len(step_counts) >= 2:
        return True, f"{len(step_counts)} distinct step counts: {sorted(step_counts)}"
    return False, f"only 1 step count observed: {sorted(step_counts)}"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_pilot_checks(
    workflow_id: str,
    all_records: list[list[StepRecord]],
    inputs: list[dict[str, Any]],
    config: Any,
) -> list[PilotCheckResult]:
    """Execute all pilot calibration checks for a workflow and return results.

    Runs Layer 1 (infrastructure, blocking) and Layer 2 (cost plausibility,
    non-blocking) checks against the 10 pilot runs.
    """
    # Lazy import to avoid circular dependency.
    from agentcost.validation.suite import BacktestConfig  # noqa: F401

    results: list[PilotCheckResult] = []

    # Layer 1 — infrastructure checks (blocking).
    results.append(check_cache_bust(workflow_id, all_records))
    results.append(check_template_substitution(all_records, inputs))
    results.append(check_cross_provider_accounting(workflow_id, all_records))
    results.append(check_w19_history_accumulation(workflow_id, all_records))
    results.append(check_pdf_validity(workflow_id, inputs))
    results.append(check_output_schema(all_records))
    results.append(check_finish_reason(all_records))

    # Layer 2 — cost plausibility checks (non-blocking).
    results.append(check_tier_separation(workflow_id, all_records))
    results.append(check_routing_ratio(workflow_id, all_records))
    results.append(check_cost_plausibility(all_records, config))
    results.append(check_detector_preactivation(workflow_id, all_records))

    blocking_failures = [r for r in results if r.blocking and r.status == "FAIL"]
    if blocking_failures:
        logger.warning(
            "Pilot checks: %d blocking failure(s) for %s: %s",
            len(blocking_failures),
            workflow_id,
            ", ".join(r.name for r in blocking_failures),
        )

    return results
