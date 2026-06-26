"""Cost computation adapter delegating to the canonical Pretia pricing table.

All cost computation in the agent suite goes through this module so there is
a single place to reconcile LiteLLM's pricing (if used) against the codebase.
"""

from __future__ import annotations

import logging

from pretia.pricing.tables import calculate_cost, resolve_model

logger = logging.getLogger(__name__)


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_hit_tokens: int = 0,
    cache_miss_tokens: int = 0,
) -> float:
    """Compute cost in USD using the canonical Pretia pricing table."""
    canonical = resolve_model(model)
    return calculate_cost(
        canonical, input_tokens, output_tokens, cache_hit_tokens, cache_miss_tokens
    )
