"""Generate dollar-denominated optimization recommendations from profiling data."""

from __future__ import annotations

from pretia.recommend.base import Recommendation, RecommendationGenerator
from pretia.recommend.registry import generate_recommendations
from pretia.recommend.score import OptimizationScore, compute_score

__all__ = [
    "OptimizationScore",
    "Recommendation",
    "RecommendationGenerator",
    "compute_score",
    "generate_recommendations",
]

import pretia.recommend.architecture  # noqa: F401
import pretia.recommend.model_swap  # noqa: F401
import pretia.recommend.workflow  # noqa: F401
