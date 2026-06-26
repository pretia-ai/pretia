"""Compute the 0-100 optimization score from recommendations and projected cost."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pretia.recommend.base import Recommendation

_SCOPE_NOTE = (
    "Score based on model and workflow optimization. "
    "Architecture analysis improving in future versions."
)

_ZONE_CONFIG: list[tuple[int, str, str, str]] = [
    (40, "red", "needs optimization", "#E53E3E"),
    (70, "amber", "room to improve", "#DD6B20"),
    (100, "green", "well optimized", "#38A169"),
]


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
) -> OptimizationScore:
    """Compute the optimization score from recommendations and projected cost."""
    total_savings = sum(r.monthly_savings for r in recommendations)

    if projected_monthly_cost > 0:
        waste_pct = min(total_savings / projected_monthly_cost, 1.0)
    else:
        waste_pct = 0.0

    score = round(100 * (1 - waste_pct))
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
