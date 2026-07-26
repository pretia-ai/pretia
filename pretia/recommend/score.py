"""Compute the 0-100 optimization score from recommendations and projected cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pretia.recommend.base import Recommendation

_SCOPE_NOTE = "Score based on detected patterns, model selection, and workflow optimization."

_ZONE_CONFIG: list[tuple[int, str, str, str]] = [
    (30, "red", "needs optimization", "#E53E3E"),
    (70, "amber", "room to improve", "#DD6B20"),
    (100, "green", "well optimized", "#38A169"),
]

_SEVERITY_PENALTY: dict[str, int] = {
    "danger": 25,
    "warning": 12,
}

_MAX_ARCHITECTURE_WASTE = 0.20


def _classify_zone(score: int) -> tuple[str, str, str]:
    """Return (zone, zone_label, zone_color) for a numeric score."""
    for threshold, zone, label, color in _ZONE_CONFIG:
        if score <= threshold:
            return zone, label, color
    return "green", "well optimized", "#38A169"


@dataclass(frozen=True, slots=True)
class OptimizationScore:
    """A 0-100 efficiency score with zone classification."""

    score: int
    zone: str
    zone_label: str
    zone_color: str
    total_savings: float
    waste_pct: float
    recommendation_count: int
    scope_note: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "score": self.score,
            "zone": self.zone,
            "zone_label": self.zone_label,
            "zone_color": self.zone_color,
            "total_savings": self.total_savings,
            "waste_pct": self.waste_pct,
            "recommendation_count": self.recommendation_count,
            "scope_note": self.scope_note,
        }


def compute_score(
    recommendations: list[Recommendation],
    projected_monthly_cost: float,
    daily_volume: int = 10_000,
    patterns: list[dict[str, Any]] | None = None,
) -> OptimizationScore:
    """Compute the optimization score from recommendations, cost, and patterns.

    The score starts at 100 and is reduced by two components:
    - **Savings penalty**: ``waste_pct * 100`` (recoverable spend as % of cost)
    - **Pattern penalty**: each detected pattern subtracts points based on
      severity (danger=-25, warning=-12)

    Architecture recommendations are capped at 20% waste contribution.
    """
    total_savings = 0.0
    design_savings = 0.0
    arch_savings = 0.0
    for r in recommendations:
        orig_vol = r.evidence.get("daily_volume", 10_000) if r.evidence else 10_000
        scale = daily_volume / orig_vol if orig_vol > 0 else 1.0
        scaled = r.monthly_savings * scale
        total_savings += scaled
        if r.type == "architecture":
            arch_savings += scaled
        else:
            design_savings += scaled

    if projected_monthly_cost > 0:
        total_savings = min(total_savings, projected_monthly_cost)
        design_waste = min(design_savings / projected_monthly_cost, 0.8)
        arch_waste = min(arch_savings / projected_monthly_cost, _MAX_ARCHITECTURE_WASTE)
        waste_pct = min(design_waste + arch_waste, 1.0)
    else:
        waste_pct = 0.0

    savings_penalty = waste_pct * 100

    pattern_penalty = 0
    if patterns:
        for p in patterns:
            severity = p.get("severity", "warning")
            pattern_penalty += _SEVERITY_PENALTY.get(severity, 0)

    score = round(100 - savings_penalty - pattern_penalty)
    score = max(0, min(100, score))

    zone, zone_label, zone_color = _classify_zone(score)

    return OptimizationScore(
        score=score,
        zone=zone,
        zone_label=zone_label,
        zone_color=zone_color,
        total_savings=round(total_savings, 2),
        waste_pct=round(waste_pct, 4),
        recommendation_count=len(recommendations),
        scope_note=_SCOPE_NOTE,
    )
