"""Core data structures and ABC for the recommendation engine."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pretia.store import ProfilingSession

_VALID_TYPES = frozenset({"model_swap", "architecture", "workflow"})
_VALID_CONFIDENCES = frozenset({"HIGH", "MODERATE", "LOW"})

CONFIDENCE_WEIGHTS: dict[str, float] = {
    "HIGH": 1.0,
    "MODERATE": 0.6,
    "LOW": 0.3,
}

_DEFAULT_DAILY_VOLUME = 10_000


def compute_priority(monthly_savings: float, confidence: str) -> int:
    """Return priority score: int(monthly_savings * confidence_weight)."""
    return int(monthly_savings * CONFIDENCE_WEIGHTS.get(confidence, 0.3))


@dataclass(frozen=True, slots=True)
class Recommendation:
    """One actionable cost-optimization recommendation."""

    id: str
    type: str
    title: str
    description: str
    monthly_savings: float
    confidence: str
    affected_steps: list[str]
    evidence: dict[str, Any]
    priority: int

    def __post_init__(self) -> None:
        if self.type not in _VALID_TYPES:
            raise ValueError(f"type must be one of {sorted(_VALID_TYPES)}, got {self.type!r}")
        if self.confidence not in _VALID_CONFIDENCES:
            raise ValueError(
                f"confidence must be one of {sorted(_VALID_CONFIDENCES)}, got {self.confidence!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "monthly_savings": self.monthly_savings,
            "confidence": self.confidence,
            "affected_steps": list(self.affected_steps),
            "evidence": dict(self.evidence),
            "priority": self.priority,
        }


class RecommendationGenerator(ABC):
    """Interface that every recommendation generator implements."""

    @abstractmethod
    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        """Analyze a profiling session and return zero or more recommendations."""
        ...


def _extract_pattern_dicts(profile: ProfilingSession) -> list[dict[str, Any]]:
    """Return pattern dicts from profile metadata, or empty list if absent."""
    patterns = profile.metadata.get("patterns", [])
    if not isinstance(patterns, list):
        return []
    return patterns


def _compute_stats(profile: ProfilingSession) -> Any:
    """Compute ProfilingStats from profile runs."""
    from pretia.projection.stats import compute_stats

    return compute_stats(profile.runs)


def _safe_record_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost for a single record, returning 0.0 for unknown models."""
    try:
        from pretia.pricing.tables import calculate_cost

        return calculate_cost(model, input_tokens, output_tokens)
    except (ValueError, KeyError):
        return 0.0
