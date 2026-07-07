"""Look up per-token pricing for all supported LLM models."""

from __future__ import annotations

import pathlib
import re

_DATE_SUFFIX_RE = re.compile(r"-(\d{8}|\d{4}-\d{2}-\d{2})$")

# Per-million-token pricing in USD: (input_price_per_M, output_price_per_M).
# Updated manually when vendors change rates. Numbers reflect publicly
# announced pricing as of May 2026; see vendor docs for the source of truth.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic — https://docs.anthropic.com/en/docs/about-claude/pricing
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
    # Legacy Anthropic — retiring June 15, 2026
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    # OpenAI — https://openai.com/api/pricing/
    "gpt-5.5": (5.00, 30.00),
    "gpt-5.4": (2.50, 10.00),
    "gpt-5.4-nano": (0.12, 0.48),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
    # OpenAI Embeddings — https://openai.com/api/pricing/
    "text-embedding-3-small": (0.02, 0.02),
    "text-embedding-3-large": (0.13, 0.13),
    # Google Gemini — https://ai.google.dev/gemini-api/docs/pricing
    # Prices shown for ≤200k-context tier.
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.5-flash": (0.15, 0.60),
    # Meta Llama via Together AI — https://www.together.ai/pricing
    "llama-4-maverick": (0.27, 0.85),
    "llama-4-scout": (0.10, 0.40),
    # Mistral — https://mistral.ai/pricing
    "mistral-large-latest": (2.00, 6.00),
    "mistral-small-latest": (0.10, 0.30),
    # DeepSeek — https://api-docs.deepseek.com/quick_start/pricing
    # DeepSeek cache-hit pricing is dramatically cheaper ($0.0028/MTok for V4 Flash).
    # v1 uses cache-miss rates for conservative projections. Actual costs will be
    # lower for workflows with stable system prompts that hit cache.
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (0.435, 0.87),
    # Legacy DeepSeek — deprecated 2026-07-24, mapped to V4 Flash pricing.
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.14, 0.28),
    # Qwen (Alibaba Cloud) — https://help.aliyun.com/zh/model-studio/billing
    # DashScope charges more for requests > 200K input tokens; v1 uses the standard tier.
    "qwen3.7-max": (2.50, 7.50),
    "qwen3.7-plus": (0.50, 1.50),
    "qwen3.6-plus": (0.325, 1.95),
    "qwen3.6-max": (1.20, 4.80),
    "qwen3.5-plus": (0.26, 0.78),
    "qwen3.5-omni": (0.26, 0.78),
    "qwen-turbo": (0.033, 0.10),
    "qwen-long": (0.033, 0.10),
}

MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4": "claude-opus-4-7",
    "claude-sonnet-4": "claude-sonnet-4-6",
    "claude-haiku-4": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-7",
    "gpt-5.4-mini": "gpt-5.4-nano",
    "gpt-4.1-micro": "gpt-4.1-nano",
    "o3-mini": "o4-mini",
    "mistral-large": "mistral-large-latest",
    "mistral-small": "mistral-small-latest",
    "deepseek": "deepseek-v4-flash",
    "deepseek-v4": "deepseek-v4-pro",
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "qwen-3.7-max": "qwen3.7-max",
    "qwen-3.7-plus": "qwen3.7-plus",
    "qwen-3.6-plus": "qwen3.6-plus",
    "qwen-3.6-max": "qwen3.6-max",
    "qwen3-max": "qwen3.7-max",
    "qwen3-plus": "qwen3.7-plus",
    "qwen-max": "qwen3.7-max",
    "qwen-plus": "qwen3.7-plus",
    "qwen-max-latest": "qwen3.7-max",
    "qwen-plus-latest": "qwen3.7-plus",
}

# Capability tier, not price. Stored separately because hosted llama/mistral
# pricing varies enough by provider that derived tiers would be misleading.
MODEL_TIERS: dict[str, str] = {
    "claude-opus-4-7": "frontier",
    "claude-opus-4-6": "frontier",
    "claude-opus-4-20250514": "frontier",
    "gpt-5.5": "frontier",
    "gpt-5.4": "mid",
    "o3": "frontier",
    "gemini-2.5-pro": "frontier",
    "mistral-large-latest": "frontier",
    "deepseek-v4-pro": "frontier",
    "claude-sonnet-4-6": "mid",
    "claude-sonnet-4-20250514": "mid",
    "gpt-4.1": "mid",
    "gpt-4o": "mid",
    "o4-mini": "mid",
    "llama-4-maverick": "mid",
    "deepseek-v4-flash": "mid",
    "deepseek-reasoner": "mid",
    "claude-haiku-4-5": "fast",
    "gpt-4.1-mini": "fast",
    "gpt-5.4-nano": "fast",
    "gpt-4.1-nano": "fast",
    "gpt-4o-mini": "fast",
    "text-embedding-3-small": "fast",
    "text-embedding-3-large": "fast",
    "gemini-2.5-flash": "fast",
    "llama-4-scout": "fast",
    "mistral-small-latest": "fast",
    "deepseek-chat": "mid",
    "qwen3.7-max": "frontier",
    "qwen3.6-max": "frontier",
    "qwen3.7-plus": "mid",
    "qwen3.6-plus": "mid",
    "qwen3.5-plus": "mid",
    "qwen3.5-omni": "mid",
    "qwen-turbo": "fast",
    "qwen-long": "fast",
}

PRICING_LAST_UPDATED = "2026-06-05"

MODEL_CACHE_HIT_PRICING: dict[str, float] = {
    # Anthropic — cache read is 10% of standard input price
    "claude-opus-4-7": 0.50,
    "claude-opus-4-6": 0.50,
    "claude-sonnet-4-6": 0.30,
    "claude-haiku-4-5": 0.10,
    "claude-opus-4-20250514": 1.50,
    "claude-sonnet-4-20250514": 0.30,
    # DeepSeek
    "deepseek-v4-flash": 0.0028,
    "deepseek-v4-pro": 0.012,
    "deepseek-chat": 0.0028,
    "deepseek-reasoner": 0.0028,
}

_PER_MILLION = 1_000_000
_VALID_TIERS = frozenset({"frontier", "mid", "fast"})

_USER_PRICING_PATH: pathlib.Path | None = None


def _get_user_pricing_path() -> pathlib.Path:
    global _USER_PRICING_PATH  # noqa: PLW0603
    if _USER_PRICING_PATH is None:
        _USER_PRICING_PATH = pathlib.Path.home() / ".pretia" / "pricing.json"
    return _USER_PRICING_PATH


def _load_user_overrides() -> None:
    """Load user pricing overrides from ~/.pretia/pricing.json if it exists."""
    import json  # noqa: I001

    path = _get_user_pricing_path()
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        models = data.get("models", {})
        for name, info in models.items():
            if "input" in info and "output" in info:
                MODEL_PRICING[name] = (info["input"], info["output"])
            tier = info.get("tier")
            if tier in _VALID_TIERS:
                MODEL_TIERS[name] = tier
        global PRICING_LAST_UPDATED  # noqa: PLW0603
        if "updated" in data:
            PRICING_LAST_UPDATED = data["updated"]
    except (json.JSONDecodeError, OSError, KeyError):
        pass


_load_user_overrides()


class UnrecognizedModelError(ValueError):
    """Raised when a model name is not found in the pricing table."""


def _find_similar_models(model: str, max_results: int = 5) -> list[str]:
    """Find model names that share substrings with the query."""
    query = model.lower().replace("-", "").replace("_", "").replace(".", "")
    candidates: list[tuple[int, str]] = []
    for known in sorted(MODEL_PRICING):
        normalized = known.lower().replace("-", "").replace("_", "").replace(".", "")
        score = 0
        for i in range(min(len(query), len(normalized))):
            if i < len(query) and i < len(normalized) and query[i] == normalized[i]:
                score += 1
        if score >= 3 or query[:4] in normalized or normalized[:4] in query:
            candidates.append((score, known))
    candidates.sort(key=lambda x: -x[0])
    return [c[1] for c in candidates[:max_results]]


def register_model(
    model: str,
    input_price: float,
    output_price: float,
    tier: str = "mid",
) -> None:
    """Register a custom model with per-million-token pricing.

    Prices are per million tokens (e.g., 0.50 = $0.50 per million input tokens).
    """
    MODEL_PRICING[model] = (input_price, output_price)
    if tier in _VALID_TIERS:
        MODEL_TIERS[model] = tier


def resolve_model(model: str) -> str:
    """Return the canonical model name, resolving aliases.

    Raises:
        UnrecognizedModelError: If the model is neither canonical nor a known alias.
    """
    if not model or not isinstance(model, str):
        raise UnrecognizedModelError(
            f"Invalid model name: {model!r}. Expected a non-empty string."
        )
    if model in MODEL_PRICING:
        return model
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # Strip provider date suffixes (e.g. claude-haiku-4-5-20251001 → claude-haiku-4-5)
    base = _DATE_SUFFIX_RE.sub("", model)
    if base != model:
        if base in MODEL_PRICING:
            return base
        if base in MODEL_ALIASES:
            return MODEL_ALIASES[base]
    similar = _find_similar_models(model)
    msg = f"Unknown model {model!r}."
    if similar:
        msg += f" Did you mean: {', '.join(similar)}?"
    msg += (
        f" Add custom pricing with register_model({model!r}, input_price=X, output_price=Y)"
        " where prices are per million tokens."
    )
    raise UnrecognizedModelError(msg)


def get_model_pricing(model: str) -> tuple[float, float]:
    """Return (input_price_per_token, output_price_per_token) for the model."""
    canonical = resolve_model(model)
    per_m_input, per_m_output = MODEL_PRICING[canonical]
    return per_m_input / _PER_MILLION, per_m_output / _PER_MILLION


def calculate_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_hit_tokens: int | None = None,
    cache_miss_tokens: int | None = None,
) -> float:
    """Return the dollar cost of one call, rounded to 6 decimal places."""
    canonical = resolve_model(model)
    input_price, output_price = get_model_pricing(model)

    if cache_hit_tokens is not None and cache_miss_tokens is not None:
        cache_hit_rate = MODEL_CACHE_HIT_PRICING.get(canonical)
        if cache_hit_rate is not None:
            input_cost = cache_miss_tokens * input_price + cache_hit_tokens * (
                cache_hit_rate / _PER_MILLION
            )
        else:
            input_cost = input_tokens * input_price
    else:
        input_cost = input_tokens * input_price

    return round(input_cost + output_tokens * output_price, 6)


def list_models() -> list[str]:
    """Return a sorted list of canonical model names (no aliases)."""
    return sorted(MODEL_PRICING)


def model_tier(model: str) -> str:
    """Return the capability tier of the model: 'frontier', 'mid', or 'fast'."""
    return MODEL_TIERS[resolve_model(model)]


def check_pricing_staleness() -> str | None:
    """Warn if pricing data is more than 30 days old."""
    from datetime import date

    last_updated = date.fromisoformat(PRICING_LAST_UPDATED)
    age_days = (date.today() - last_updated).days
    if age_days > 30:
        return (
            f"Pricing data is {age_days} days old. Provider prices may have changed. "
            "Run `pretia update-pricing` to refresh."
        )
    return None
