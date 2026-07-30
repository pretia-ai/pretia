"""Fetch current model pricing from the LiteLLM community dataset."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

from pretia.pricing.tables import MODEL_ALIASES, MODEL_PRICING

logger = logging.getLogger(__name__)

# Community-maintained, covers all major providers, no auth required.
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

_PER_MILLION = 1_000_000


@dataclass(slots=True)
class RemotePricingResult:
    """Outcome of a remote pricing refresh, scoped to models Pretia knows."""

    changed: dict[str, tuple[float, float]] = field(default_factory=dict)
    unchanged: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_models_dict(self) -> dict[str, dict[str, float]]:
        """Serialize changed prices to the user-override JSON schema."""
        return {
            name: {"input": prices[0], "output": prices[1]}
            for name, prices in self.changed.items()
        }


def is_suspicious_change(old: tuple[float, float], new: tuple[float, float]) -> bool:
    """Flag price changes larger than 2x in either direction.

    Real vendor price adjustments are usually incremental; a >2x jump most
    often means the community dataset entry maps to a different model
    generation and should be verified against the vendor's pricing page.
    """
    for old_price, new_price in zip(old, new, strict=True):
        if old_price <= 0 or new_price <= 0:
            return True
        ratio = new_price / old_price
        if ratio > 2 or ratio < 0.5:
            return True
    return False


def fetch_remote_pricing(
    url: str = LITELLM_PRICING_URL,
    timeout: float = 15.0,
) -> dict[str, Any]:
    """Download the LiteLLM pricing JSON and return it as a dict."""
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
            raw = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise ConnectionError(f"Failed to fetch pricing data from {url}: {exc}") from exc

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Remote pricing data is not valid JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Remote pricing data has an unexpected format (expected an object).")
    return data


def _litellm_lookup(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Index LiteLLM entries by bare model name, stripping 'provider/' prefixes."""
    lookup: dict[str, dict[str, Any]] = {}
    for key, entry in data.items():
        if not isinstance(entry, dict):
            continue
        bare = key.rsplit("/", 1)[-1]
        # Un-prefixed keys win over provider-prefixed duplicates.
        if bare not in lookup or "/" not in key:
            lookup[bare] = entry
    return lookup


def _entry_prices(entry: dict[str, Any]) -> tuple[float, float] | None:
    """Convert a LiteLLM entry's per-token costs to per-million rates."""
    inp = entry.get("input_cost_per_token")
    out = entry.get("output_cost_per_token")
    if not isinstance(inp, (int, float)) or not isinstance(out, (int, float)):
        return None
    if inp < 0 or out < 0:
        return None
    return (round(inp * _PER_MILLION, 6), round(out * _PER_MILLION, 6))


def refresh_known_model_pricing(data: dict[str, Any]) -> RemotePricingResult:
    """Match LiteLLM data against Pretia's known models and report price drift.

    Only models already present in MODEL_PRICING are considered — the remote
    dataset holds thousands of entries with no tier information, and importing
    them wholesale would pollute cost reports.
    """
    lookup = _litellm_lookup(data)

    # Aliases give each canonical model extra names to try against the dataset.
    names_for: dict[str, list[str]] = {name: [name] for name in MODEL_PRICING}
    for alias, canonical in MODEL_ALIASES.items():
        if canonical in names_for:
            names_for[canonical].append(alias)

    result = RemotePricingResult()
    for canonical, candidates in names_for.items():
        prices = None
        for candidate in candidates:
            entry = lookup.get(candidate)
            if entry is not None:
                prices = _entry_prices(entry)
                if prices is not None:
                    break
        if prices is None:
            result.missing.append(canonical)
            continue
        if prices == MODEL_PRICING[canonical]:
            result.unchanged.append(canonical)
        else:
            result.changed[canonical] = prices
    return result
