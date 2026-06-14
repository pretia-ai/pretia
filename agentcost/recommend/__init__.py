"""Generate dollar-denominated optimization recommendations from profiling data."""

from __future__ import annotations

from agentcost.recommend.base import Recommendation, RecommendationGenerator
from agentcost.recommend.registry import generate_recommendations
from agentcost.recommend.score import OptimizationScore, compute_score

__all__ = [
    "OptimizationScore",
    "Recommendation",
    "RecommendationGenerator",
    "compute_score",
    "generate_recommendations",
]

import agentcost.recommend.architecture  # noqa: F401
import agentcost.recommend.model_swap  # noqa: F401
import agentcost.recommend.workflow  # noqa: F401
