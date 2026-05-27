"""Look up per-token pricing for all supported LLM models."""

from __future__ import annotations

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
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3": (2.00, 8.00),
    "o4-mini": (1.10, 4.40),
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
    "deepseek-chat": (0.14, 0.28),
    "deepseek-reasoner": (0.55, 2.19),
}

MODEL_ALIASES: dict[str, str] = {
    "claude-opus-4": "claude-opus-4-7",
    "claude-sonnet-4": "claude-sonnet-4-6",
    "claude-haiku-4": "claude-haiku-4-5",
    "claude-haiku": "claude-haiku-4-5",
    "claude-sonnet": "claude-sonnet-4-6",
    "claude-opus": "claude-opus-4-7",
    "gpt-4.1-micro": "gpt-4.1-nano",
    "o3-mini": "o4-mini",
    "mistral-large": "mistral-large-latest",
    "mistral-small": "mistral-small-latest",
    "deepseek": "deepseek-chat",
}

# Capability tier, not price. Stored separately because hosted llama/mistral
# pricing varies enough by provider that derived tiers would be misleading.
MODEL_TIERS: dict[str, str] = {
    "claude-opus-4-7": "frontier",
    "claude-opus-4-6": "frontier",
    "claude-opus-4-20250514": "frontier",
    "gpt-5.5": "frontier",
    "o3": "frontier",
    "gemini-2.5-pro": "frontier",
    "mistral-large-latest": "frontier",
    "claude-sonnet-4-6": "mid",
    "claude-sonnet-4-20250514": "mid",
    "gpt-4.1": "mid",
    "gpt-4o": "mid",
    "o4-mini": "mid",
    "llama-4-maverick": "mid",
    "deepseek-reasoner": "mid",
    "claude-haiku-4-5": "fast",
    "gpt-4.1-mini": "fast",
    "gpt-4.1-nano": "fast",
    "gpt-4o-mini": "fast",
    "gemini-2.5-flash": "fast",
    "llama-4-scout": "fast",
    "mistral-small-latest": "fast",
    "deepseek-chat": "fast",
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
