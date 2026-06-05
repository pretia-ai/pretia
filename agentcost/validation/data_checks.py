"""Post-profiling data quality checks."""

from __future__ import annotations

from collections import defaultdict

from agentcost.collectors.base import StepRecord


def validate_profiling_data(records: list[list[StepRecord]]) -> list[str]:
    """Check profiling runs for data quality issues.

    Returns a list of warning strings (empty if no issues).
    """
    if not records:
        return []

    n_runs = len(records)
    step_zero_counts: dict[str, int] = defaultdict(int)
    step_present_counts: dict[str, int] = defaultdict(int)

    for run in records:
        step_tokens: dict[str, int] = defaultdict(int)
        steps_seen: set[str] = set()
        for rec in run:
            step_tokens[rec.step_name] += rec.input_tokens + rec.output_tokens
            steps_seen.add(rec.step_name)

        for step_name in steps_seen:
            step_present_counts[step_name] += 1
            if step_tokens[step_name] == 0:
                step_zero_counts[step_name] += 1

    warnings: list[str] = []
    for step_name, present in step_present_counts.items():
        zero_count = step_zero_counts.get(step_name, 0)
        if zero_count == present and present == n_runs:
            warnings.append(
                f"Step '{step_name}' recorded zero tokens across all {n_runs} runs. "
                "This is likely a data collection error — check that the collector "
                "captures usage metadata for this step."
            )
        elif zero_count > present * 0.5:
            pct = zero_count / present * 100
            warnings.append(
                f"Step '{step_name}' recorded zero tokens in {zero_count}/{present} "
                f"runs ({pct:.0f}%). Some runs may have missing usage data."
            )

    return warnings
