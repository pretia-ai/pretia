"""Confidence tier computation for projection reports."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agentcost.projection.patterns import DetectedPattern
from agentcost.projection.stats import StepStats

_MAX_STEP_VARIANCE_DEDUCTION = 30


@dataclass(frozen=True, slots=True)
class ConfidenceResult:
    """Projection confidence assessment."""

    score: int
    tier: str
    display_range: str
    language: str
    deductions: list[str] = field(default_factory=list)
    bonuses: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "score": self.score,
            "tier": self.tier,
            "display_range": self.display_range,
            "language": self.language,
            "deductions": list(self.deductions),
            "bonuses": list(self.bonuses),
        }


def compute_confidence(
    sample_size: int,
    step_stats: dict[str, StepStats],
    patterns: list[DetectedPattern],
    input_source: str = "auto-generate",
) -> ConfidenceResult:
    """Score projection confidence based on sample size, variance, and patterns."""
    score = 100
    deductions: list[str] = []
    bonuses: list[str] = []

    # --- Sample size deductions ---
    if sample_size < 5:
        score -= 40
        deductions.append(
            f"Very small sample size ({sample_size} runs). "
            "Projections have wide uncertainty."
        )
    elif sample_size < 10:
        score -= 40
        deductions.append(
            f"Very small sample size ({sample_size} runs). "
            "Projections have wide uncertainty."
        )
    elif sample_size < 30:
        score -= 20
        deductions.append(
            f"Small sample size ({sample_size} runs). "
            "Consider profiling with 30+ samples."
        )
    elif sample_size < 100:
        score -= 10
        deductions.append(f"Moderate sample size ({sample_size} runs).")

    # --- Per-step variance deductions (capped) ---
    step_deduction_total = 0
    for name, ss in step_stats.items():
        cv = ss.cost.std / ss.cost.mean if ss.cost.mean > 0 else 0.0
        if cv > 1.0:
            penalty = 15
            step_deduction_total += penalty
            deductions.append(
                f"Step '{name}' has very high cost variance (CV={cv:.2f})."
            )
        elif cv > 0.5:
            penalty = 8
            step_deduction_total += penalty
            deductions.append(
                f"Step '{name}' has moderate cost variance (CV={cv:.2f})."
            )

    step_deduction_total = min(step_deduction_total, _MAX_STEP_VARIANCE_DEDUCTION)
    score -= step_deduction_total

    # --- Pattern deductions (deduplicated per step+type) ---
    seen_patterns: set[tuple[str, str]] = set()
    for p in patterns:
        key = (p.pattern_type, p.step_name)
        if key in seen_patterns:
            continue
        seen_patterns.add(key)

        if p.pattern_type == "context_growth":
            score -= 10
            deductions.append(
                f"Context growth detected at step '{p.step_name}'. "
                "Non-linear cost scaling."
            )
        elif p.pattern_type == "loop_count_variance":
            score -= 10
            deductions.append(
                f"High loop count variance at step '{p.step_name}'."
            )
        elif p.pattern_type == "high_token_variance":
            score -= 10
            deductions.append(
                f"High token variance at step '{p.step_name}'."
            )

    # --- Bonuses ---
    if input_source == "langfuse":
        score += 15
        bonuses.append("Based on real production traces (Langfuse import).")
    if sample_size >= 200:
        score += 10
        bonuses.append(f"Large sample size ({sample_size} runs).")

    # --- Clamp and map to tier ---
    score = max(0, min(100, score))

    if score >= 80:
        tier = "HIGH"
        display_range = "p50 – p90"
        language = "projected"
    elif score >= 60:
        tier = "MODERATE"
        display_range = "p50 – p95"
        language = "estimated"
    elif score >= 40:
        tier = "LOW"
        display_range = "p25 – p99"
        language = "estimated"
    else:
        tier = "VERY_LOW"
        display_range = "order of magnitude"
        language = "ballpark"

    return ConfidenceResult(
        score=score,
        tier=tier,
        display_range=display_range,
        language=language,
        deductions=deductions,
        bonuses=bonuses,
    )
