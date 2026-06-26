"""Recommend cheaper models for steps where task complexity allows it."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pretia.pricing.tables import (
    UnrecognizedModelError,
    calculate_cost,
    model_tier,
    resolve_model,
)
from pretia.recommend.base import (
    _DEFAULT_DAILY_VOLUME,
    Recommendation,
    RecommendationGenerator,
    _compute_stats,
    compute_priority,
)
from pretia.recommend.registry import register

if TYPE_CHECKING:
    from pretia.projection.stats import StepStats
    from pretia.store import ProfilingSession

logger = logging.getLogger(__name__)

_CLASSIFICATION_KEYWORDS = frozenset(
    {"classify", "categorize", "route", "label", "detect", "sort", "filter", "triage", "check"}
)

_MIN_MONTHLY_SAVINGS = 10.0

_PROVIDER_FAST: dict[str, str] = {
    "anthropic": "claude-haiku-4-5",
    "openai": "gpt-4.1-nano",
    "google": "gemini-2.5-flash",
    "deepseek": "deepseek-v4-flash",
    "mistral": "mistral-small-latest",
    "qwen": "qwen-turbo",
    "meta": "llama-4-scout",
}

_PROVIDER_MID: dict[str, str] = {
    "anthropic": "claude-sonnet-4-6",
    "openai": "gpt-4.1",
    "google": "gemini-2.5-pro",
    "deepseek": "deepseek-v4-flash",
    "mistral": "mistral-small-latest",
    "qwen": "qwen3.7-plus",
    "meta": "llama-4-maverick",
}

_MODEL_PREFIX_TO_PROVIDER: list[tuple[str, str]] = [
    ("claude-", "anthropic"),
    ("gpt-", "openai"),
    ("o3", "openai"),
    ("o4-", "openai"),
    ("gemini-", "google"),
    ("deepseek", "deepseek"),
    ("mistral-", "mistral"),
    ("qwen", "qwen"),
    ("llama-", "meta"),
]


def _detect_provider(model: str) -> str | None:
    """Identify the provider from a canonical model name."""
    lower = model.lower()
    for prefix, provider in _MODEL_PREFIX_TO_PROVIDER:
        if lower.startswith(prefix):
            return provider
    return None


def _has_classification_keywords(step_name: str) -> bool:
    """Check if the step name contains classification-related keywords."""
    lower = step_name.lower()
    return any(kw in lower for kw in _CLASSIFICATION_KEYWORDS)


def _dominant_output_format(records_for_step: list) -> str:
    """Return the most common output_format across records for a step."""
    counts: dict[str, int] = {}
    for r in records_for_step:
        counts[r.output_format] = counts.get(r.output_format, 0) + 1
    if not counts:
        return "text"
    return max(counts, key=counts.get)  # type: ignore[arg-type]


@register
class ModelSwapGenerator(RecommendationGenerator):
    """Recommend cheaper models for classification and structured extraction tasks."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        if not profile.runs:
            return []

        stats = _compute_stats(profile)
        recommendations: list[Recommendation] = []

        for step_name, step_stats in stats.step_stats.items():
            rec = self._evaluate_step(step_name, step_stats, profile)
            if rec is not None:
                recommendations.append(rec)

        return recommendations

    def _evaluate_step(
        self,
        step_name: str,
        step_stats: StepStats,
        profile: ProfilingSession,
    ) -> Recommendation | None:
        if step_stats.step_type != "llm":
            return None

        try:
            canonical = resolve_model(step_stats.model)
        except UnrecognizedModelError:
            return None

        try:
            tier = model_tier(canonical)
        except (ValueError, KeyError):
            return None

        if tier == "fast":
            return None

        provider = _detect_provider(canonical)
        if provider is None:
            return None

        median_input = step_stats.input_tokens.p50
        median_output = step_stats.output_tokens.p50
        if median_input <= 0:
            return None

        ratio = median_output / median_input

        step_records = [r for run in profile.runs for r in run if r.step_name == step_name]
        output_format = _dominant_output_format(step_records)

        if output_format == "code":
            return None
        if ratio >= 1.0:
            return None

        has_keywords = _has_classification_keywords(step_name)
        is_classification = False
        is_extraction = False

        if output_format == "json" and ratio < 0.3:
            is_classification = True
        elif has_keywords and ratio < 0.5:
            is_classification = True
        elif output_format == "json" and ratio < 1.0 and tier == "frontier":
            is_extraction = True
        else:
            return None

        if is_classification:
            recommended_model = _PROVIDER_FAST.get(provider)
        elif is_extraction:
            recommended_model = _PROVIDER_MID.get(provider)
        else:
            return None

        if recommended_model is None:
            return None

        try:
            recommended_tier = model_tier(recommended_model)
        except (ValueError, KeyError):
            return None

        if recommended_tier == tier:
            return None

        current_cost = calculate_cost(canonical, int(median_input), int(median_output))
        recommended_cost = calculate_cost(recommended_model, int(median_input), int(median_output))
        savings_per_call = current_cost - recommended_cost
        if savings_per_call <= 0:
            return None

        monthly_savings = savings_per_call * _DEFAULT_DAILY_VOLUME * 30

        if monthly_savings < _MIN_MONTHLY_SAVINGS:
            return None

        if is_classification and ratio < 0.2 and has_keywords:
            confidence = "HIGH"
        else:
            confidence = "MODERATE"

        task_type = "classification" if is_classification else "structured extraction"

        return Recommendation(
            id=f"model-swap-{step_name}",
            type="model_swap",
            title=f"Swap {step_name} to {recommended_model}",
            description=(
                f"{step_name} performs {task_type} "
                f"(output/input ratio: {ratio:.2f}, format: {output_format}). "
                f"Switching from {canonical} ({tier}) to {recommended_model} "
                f"({recommended_tier}) saves ${monthly_savings:,.0f}/month "
                f"at {_DEFAULT_DAILY_VOLUME:,} daily runs."
            ),
            monthly_savings=round(monthly_savings, 2),
            confidence=confidence,
            affected_steps=[step_name],
            evidence={
                "current_model": canonical,
                "recommended_model": recommended_model,
                "current_tier": tier,
                "recommended_tier": recommended_tier,
                "output_input_ratio": round(ratio, 4),
                "output_format": output_format,
                "task_classification": task_type,
                "has_classification_keywords": has_keywords,
                "median_input_tokens": int(median_input),
                "median_output_tokens": int(median_output),
                "savings_per_call": round(savings_per_call, 6),
                "daily_volume": _DEFAULT_DAILY_VOLUME,
            },
            priority=compute_priority(round(monthly_savings, 2), confidence),
        )
