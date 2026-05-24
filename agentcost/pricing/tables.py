"""Look up per-token pricing for all supported LLM models."""

from __future__ import annotations

# Per-million-token pricing in USD: (input_price_per_M, output_price_per_M).
# Updated manually when vendors change rates. Numbers reflect publicly
# announced pricing as of mid-2025; see vendor docs for the source of truth.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic — https://www.anthropic.com/pricing
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-haiku-3-5-20241022": (0.80, 4.00),
    "claude-sonnet-3-5-20241022": (3.00, 15.00),
    # OpenAI — https://openai.com/api/pricing/
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.00, 30.00),
    "o1": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Google Gemini — https://ai.google.dev/pricing (≤ 128k-context tier).
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-lite": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-flash": (0.075, 0.30),
    # Meta Llama via Together AI — https://www.together.ai/pricing
    "llama-3.1-70b": (0.88, 0.88),
    "llama-3.1-8b": (0.18, 0.18),
    # Mistral — https://mistral.ai/technology/#pricing
    "mistral-large-latest": (2.00, 6.00),
    "mistral-small-latest": (0.20, 0.60),
}

MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-haiku-3.5": "claude-haiku-3-5-20241022",
    "claude-sonnet-3.5": "claude-sonnet-3-5-20241022",
    "gpt-4o-2024-08-06": "gpt-4o",
    "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
    "gpt-4-turbo-2024-04-09": "gpt-4-turbo",
}

# Capability tier, not price. Stored separately because hosted llama/mistral
# pricing varies enough by provider that derived tiers would be misleading.
MODEL_TIERS: dict[str, str] = {
    "claude-opus-4-20250514": "frontier",
    "gpt-4-turbo": "frontier",
    "o1": "frontier",
    "gemini-1.5-pro": "frontier",
    "claude-sonnet-4-20250514": "mid",
    "claude-sonnet-3-5-20241022": "mid",
    "gpt-4o": "mid",
    "o3-mini": "mid",
    "mistral-large-latest": "mid",
    "gemini-2.0-flash": "mid",
    "llama-3.1-70b": "mid",
    "claude-haiku-3-5-20241022": "fast",
    "gpt-4o-mini": "fast",
    "o1-mini": "fast",
    "gemini-1.5-flash": "fast",
    "gemini-2.0-flash-lite": "fast",
    "mistral-small-latest": "fast",
    "llama-3.1-8b": "fast",
}

_PER_MILLION = 1_000_000
_VALID_TIERS = frozenset({"frontier", "mid", "fast"})


def resolve_model(model: str) -> str:
    """Return the canonical model name, resolving aliases.

    Raises:
        ValueError: If the model is neither canonical nor a known alias.
    """
    if model in MODEL_PRICING:
        return model
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    raise ValueError(f"Unknown model {model!r}. Available models: {sorted(MODEL_PRICING)}")


def get_model_pricing(model: str) -> tuple[float, float]:
    """Return (input_price_per_token, output_price_per_token) for the model."""
    canonical = resolve_model(model)
    per_m_input, per_m_output = MODEL_PRICING[canonical]
    return per_m_input / _PER_MILLION, per_m_output / _PER_MILLION


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the dollar cost of one call, rounded to 6 decimal places."""
    input_price, output_price = get_model_pricing(model)
    return round(input_tokens * input_price + output_tokens * output_price, 6)


def list_models() -> list[str]:
    """Return a sorted list of canonical model names (no aliases)."""
    return sorted(MODEL_PRICING)


def model_tier(model: str) -> str:
    """Return the capability tier of the model: 'frontier', 'mid', or 'fast'."""
    return MODEL_TIERS[resolve_model(model)]
