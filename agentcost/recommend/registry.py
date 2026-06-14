"""Generator registry and public API for the recommendation engine."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from agentcost.recommend.base import Recommendation, RecommendationGenerator

if TYPE_CHECKING:
    from agentcost.store import ProfilingSession

logger = logging.getLogger(__name__)

_GENERATORS: list[type[RecommendationGenerator]] = []


def register(cls: type[RecommendationGenerator]) -> type[RecommendationGenerator]:
    """Class decorator that adds a generator to the registry."""
    _GENERATORS.append(cls)
    return cls


def generate_recommendations(profile: ProfilingSession) -> list[Recommendation]:
    """Run all registered generators, deduplicate by id, return sorted by priority."""
    all_recs: list[Recommendation] = []
    for gen_cls in _GENERATORS:
        try:
            recs = gen_cls().generate(profile)
            all_recs.extend(recs)
        except Exception:
            logger.warning("Generator %s failed", gen_cls.__name__, exc_info=True)

    seen: dict[str, Recommendation] = {}
    for rec in all_recs:
        existing = seen.get(rec.id)
        if existing is None or rec.priority > existing.priority:
            seen[rec.id] = rec

    result = list(seen.values())
    result.sort(key=lambda r: r.priority, reverse=True)
    return result
