"""Feed synthetic cost data through the projection engine."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import UTC, datetime

from agentcost.collectors.base import StepRecord
from agentcost.pricing.tables import MODEL_PRICING, register_model
from agentcost.projection.patterns import detect_patterns
from agentcost.projection.projector import project
from agentcost.projection.stats import compute_stats
from tests.synthetic.generators import SyntheticWorkflow

_SYNTHETIC_MODEL = "_synthetic_unit_cost_"


def _ensure_synthetic_model() -> None:
    if _SYNTHETIC_MODEL not in MODEL_PRICING:
        register_model(_SYNTHETIC_MODEL, input_price=1.0, output_price=0.0)


def _cost_to_record(cost: float, run_idx: int) -> StepRecord:
    """Convert a cost value to a StepRecord with matching token counts."""
    input_tokens = max(1, int(cost * 1_000_000))
    return StepRecord(
        step_name="main",
        step_type="llm",
        model=_SYNTHETIC_MODEL,
        input_tokens=input_tokens,
        output_tokens=0,
        context_size=input_tokens,
        tool_definitions_tokens=0,
        system_prompt_hash="synthetic",
        system_prompt_tokens=0,
        output_format="text",
        is_retry=False,
        iteration=1,
        parent_step=None,
        duration_ms=100,
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
    )


@dataclass
class SyntheticCalibrationResult:
    """Result of projecting one synthetic workflow."""

    workflow: SyntheticWorkflow
    projected_p50: float
    projected_p95: float
    projected_mean: float | None
    patterns_detected: list[str]
    confidence_tier: str
    cvar_95: float | None = None


def run_one(
    wf: SyntheticWorkflow,
    daily_volume: int = 1000,
) -> SyntheticCalibrationResult:
    """Run the projection engine on one synthetic workflow."""
    _ensure_synthetic_model()

    runs = [[_cost_to_record(c, i)] for i, c in enumerate(wf.observed_costs)]
    stats = compute_stats(runs)
    patterns = detect_patterns(runs, stats)
    result = project(stats, patterns, traffic=[daily_volume], runs=runs)

    proj = result.projections[daily_volume]
    mc = result.montecarlo_result

    return SyntheticCalibrationResult(
        workflow=wf,
        projected_p50=proj.monthly_cost.p50,
        projected_p95=proj.monthly_cost.p95,
        projected_mean=proj.monthly_cost.mean,
        patterns_detected=[p.pattern_type for p in patterns],
        confidence_tier=result.confidence.tier,
        cvar_95=mc.cvar_95 if mc else None,
    )


def run_synthetic_calibration(
    workflows: list[SyntheticWorkflow],
    daily_volume: int = 1000,
) -> list[SyntheticCalibrationResult]:
    """Run all synthetic workflows through the projection engine."""
    results: list[SyntheticCalibrationResult] = []
    total = len(workflows)
    for i, wf in enumerate(workflows):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  Processing {i + 1}/{total}...", file=sys.stderr)
        results.append(run_one(wf, daily_volume))
    return results
