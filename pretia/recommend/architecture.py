"""Architecture recommendations: prompt caching, tool filtering, context dedup."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from pretia.pricing.tables import (
    MODEL_CACHE_HIT_PRICING,
    UnrecognizedModelError,
    get_model_pricing,
    resolve_model,
)
from pretia.recommend.base import (
    _CONSERVATIVE_FACTOR,
    _DEFAULT_DAILY_VOLUME,
    Recommendation,
    RecommendationGenerator,
    _compute_stats,
    _extract_pattern_dicts,
    compute_priority,
)
from pretia.recommend.registry import register

if TYPE_CHECKING:
    from pretia.store import ProfilingSession

logger = logging.getLogger(__name__)

_PER_MILLION = 1_000_000
_MIN_CACHING_SAVINGS = 50.0


@register
class PromptCachingGenerator(RecommendationGenerator):
    """Recommend enabling prompt caching for steps with low cache hit rates."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        patterns = _extract_pattern_dicts(profile)
        cache_patterns = [
            p for p in patterns if p.get("pattern_type") == "cache_utilization_opportunity"
        ]
        if not cache_patterns:
            return []

        recommendations: list[Recommendation] = []
        for pattern in cache_patterns:
            rec = self._evaluate_pattern(pattern)
            if rec is not None:
                recommendations.append(rec)
        return recommendations

    def _evaluate_pattern(self, pattern: dict[str, Any]) -> Recommendation | None:
        step_name = pattern.get("step_name", "")
        evidence = pattern.get("evidence", {})
        cache_hit_ratio = evidence.get("cache_hit_ratio", 0.0)
        cache_miss_tokens = evidence.get(
            "mean_cache_miss_tokens_per_run",
            evidence.get("total_cache_miss_tokens", 0),
        )
        model = evidence.get("model", "")

        if not model or cache_miss_tokens <= 0:
            return None

        try:
            canonical = resolve_model(model)
        except UnrecognizedModelError:
            return None

        cache_hit_rate_per_m = MODEL_CACHE_HIT_PRICING.get(canonical)
        if cache_hit_rate_per_m is None:
            return None

        standard_input_rate = get_model_pricing(canonical)[0]
        cache_hit_rate = cache_hit_rate_per_m / _PER_MILLION

        savings_per_token = standard_input_rate - cache_hit_rate
        if savings_per_token <= 0:
            return None

        mean_sys = evidence.get("mean_system_prompt_tokens", 0)
        mean_tool = evidence.get("mean_tool_def_tokens", 0)
        mean_inp = evidence.get("mean_input_tokens", 0)
        cacheable_frac = (
            min((mean_sys + mean_tool) / mean_inp, 1.0) if mean_inp > 0 else 0.5
        )
        effective_miss = cache_miss_tokens * cacheable_frac

        monthly_savings = round(
            effective_miss * savings_per_token * _CONSERVATIVE_FACTOR
            * _DEFAULT_DAILY_VOLUME * 30,
            2,
        )

        if monthly_savings < _MIN_CACHING_SAVINGS:
            return None

        if standard_input_rate > 0:
            pct_reduction = savings_per_token / standard_input_rate * 100
        else:
            pct_reduction = 0.0

        return Recommendation(
            id=f"prompt-caching-{step_name}",
            type="architecture",
            title=f"Enable prompt caching for {step_name}",
            description=(
                f"{step_name} sends {cache_miss_tokens:,.0f} cache-miss tokens per run "
                f"on {canonical} (cache hit ratio: {cache_hit_ratio:.1%}). "
                f"Enabling prompt caching saves ${monthly_savings:,.0f}/month "
                f"at {_DEFAULT_DAILY_VOLUME:,} daily runs. "
                f"Estimate is conservative. Actual savings may be higher."
            ),
            monthly_savings=monthly_savings,
            confidence="HIGH",
            affected_steps=[step_name],
            evidence={
                "model": canonical,
                "cache_hit_ratio": cache_hit_ratio,
                "cache_miss_tokens": cache_miss_tokens,
                "cacheable_fraction": round(cacheable_frac, 4),
                "standard_input_rate_per_token": standard_input_rate,
                "cache_hit_rate_per_token": cache_hit_rate,
                "pct_reduction": round(pct_reduction, 1),
                "daily_volume": _DEFAULT_DAILY_VOLUME,
            },
            priority=compute_priority(monthly_savings, "HIGH"),
        )


@register
class ToolFilterGenerator(RecommendationGenerator):
    """Recommend filtering tool definitions when most are unused."""

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
        step_stats: Any,
        profile: ProfilingSession,
    ) -> Recommendation | None:
        if step_stats.step_type != "llm":
            return None

        median_input = step_stats.input_tokens.p50
        median_tool_def = 0.0
        per_run_tool_def: list[int] = []
        for run in profile.runs:
            run_total = sum(
                r.tool_definitions_tokens for r in run if r.step_name == step_name
            )
            if any(r.step_name == step_name for r in run):
                per_run_tool_def.append(run_total)

        if not per_run_tool_def:
            return None

        per_run_tool_def.sort()
        n = len(per_run_tool_def)
        median_tool_def = (
            per_run_tool_def[n // 2]
            if n % 2 == 1
            else (per_run_tool_def[n // 2 - 1] + per_run_tool_def[n // 2]) / 2
        )

        if median_input <= 0 or median_tool_def <= 0:
            return None

        tool_def_share = median_tool_def / median_input
        if tool_def_share <= 0.3:
            return None

        used_tools: set[str] = set()
        has_tool_data = False
        for run in profile.runs:
            for r in run:
                if r.step_name == step_name and r.tool_name is not None:
                    used_tools.add(r.tool_name)
                    has_tool_data = True

        if not has_tool_data:
            if median_tool_def > 0:
                n_used = 0
                fraction_used = 0.0
                savings_tokens = median_tool_def
            else:
                return None
        else:
            n_used = len(used_tools)
            if n_used == 0:
                fraction_used = 0.0
                savings_tokens = median_tool_def
            else:
                if median_tool_def > 0:
                    total_estimated = max(n_used, int(median_tool_def / 200))
                else:
                    total_estimated = n_used
                if total_estimated <= n_used:
                    return None
                fraction_used = n_used / total_estimated
                savings_tokens = median_tool_def * (1 - fraction_used)

        if savings_tokens <= 0:
            return None

        try:
            input_price = get_model_pricing(step_stats.model)[0]
        except (ValueError, KeyError):
            return None

        monthly_savings = round(
            savings_tokens * input_price * _CONSERVATIVE_FACTOR
            * _DEFAULT_DAILY_VOLUME * 30,
            2,
        )

        if monthly_savings < 10.0:
            return None

        return Recommendation(
            id=f"tool-filter-{step_name}",
            type="architecture",
            title=f"Filter tool definitions for {step_name}",
            description=(
                f"{step_name} passes {int(median_tool_def):,} tool definition tokens "
                f"({tool_def_share:.0%} of input) but "
                + (
                    "none are ever called. "
                    if n_used == 0
                    else (
                        f"only uses {n_used} tool{'s' if n_used != 1 else ''} "
                        f"({', '.join(sorted(used_tools))}). "
                    )
                )
                + f"Filtering unused tools saves ${monthly_savings:,.0f}/month "
                f"at {_DEFAULT_DAILY_VOLUME:,} daily runs. "
                f"Estimate is conservative. Actual savings may be higher."
            ),
            monthly_savings=monthly_savings,
            confidence="MODERATE",
            affected_steps=[step_name],
            evidence={
                "tool_definition_share": round(tool_def_share, 4),
                "median_tool_def_tokens": int(median_tool_def),
                "median_input_tokens": int(median_input),
                "used_tools": sorted(used_tools),
                "n_used_tools": n_used,
                "fraction_used": round(fraction_used, 4),
                "savings_tokens": int(savings_tokens),
                "daily_volume": _DEFAULT_DAILY_VOLUME,
            },
            priority=compute_priority(monthly_savings, "MODERATE"),
        )


@register
class CacheContextGenerator(RecommendationGenerator):
    """Recommend eliminating redundant system prompts in consecutive steps."""

    def generate(self, profile: ProfilingSession) -> list[Recommendation]:
        if not profile.runs:
            return []

        pair_run_costs: dict[tuple[str, str], list[float]] = {}

        for run in profile.runs:
            sorted_steps = sorted(run, key=lambda r: r.timestamp)
            run_pair_totals: dict[tuple[str, str], float] = defaultdict(float)

            for i in range(1, len(sorted_steps)):
                prev = sorted_steps[i - 1]
                curr = sorted_steps[i]

                if not (
                    prev.system_prompt_hash == curr.system_prompt_hash and prev.system_prompt_hash
                ):
                    continue

                is_different_step = prev.step_name != curr.step_name
                is_same_step_no_growth = (
                    prev.step_name == curr.step_name
                    and curr.iteration > prev.iteration
                    and curr.input_tokens <= prev.input_tokens * 1.1
                )
                if not (is_different_step or is_same_step_no_growth):
                    continue

                pair_key = (
                    min(prev.step_name, curr.step_name),
                    max(prev.step_name, curr.step_name),
                )

                try:
                    canonical = resolve_model(curr.model)
                    input_price = get_model_pricing(canonical)[0]
                except (ValueError, KeyError, UnrecognizedModelError):
                    continue

                cache_rate = MODEL_CACHE_HIT_PRICING.get(canonical)
                if cache_rate is not None:
                    net_rate = input_price - cache_rate / _PER_MILLION
                else:
                    net_rate = input_price * 0.5
                redundant_cost = curr.system_prompt_tokens * net_rate
                run_pair_totals[pair_key] += redundant_cost

            for pair_key, total in run_pair_totals.items():
                pair_run_costs.setdefault(pair_key, []).append(total)

        recommendations: list[Recommendation] = []
        for (step_a, step_b), costs in pair_run_costs.items():
            avg_redundant = sum(costs) / len(costs) if costs else 0.0
            monthly_savings = round(
                avg_redundant * _CONSERVATIVE_FACTOR * _DEFAULT_DAILY_VOLUME * 30, 2
            )

            if monthly_savings < 10.0:
                continue

            avg_tokens = 0
            for run in profile.runs:
                for r in run:
                    if r.step_name in (step_a, step_b):
                        avg_tokens = r.system_prompt_tokens
                        break
                if avg_tokens > 0:
                    break

            conservative_note = " Estimate is conservative. Actual savings may be higher."
            if step_a == step_b:
                title = f"Eliminate redundant system prompt across consecutive calls in {step_a}"
                desc = (
                    f"{step_a} sends identical system prompts "
                    f"({avg_tokens:,} tokens) on consecutive iterations. "
                    f"Restructuring to share context or enabling prompt caching "
                    f"saves ${monthly_savings:,.0f}/month "
                    f"at {_DEFAULT_DAILY_VOLUME:,} daily runs."
                    + conservative_note
                )
            else:
                title = f"Eliminate redundant system prompt in {step_a} and {step_b}"
                desc = (
                    f"{step_a} and {step_b} send identical system prompts "
                    f"({avg_tokens:,} tokens) in consecutive calls. "
                    f"Restructuring to share context or enabling prompt caching "
                    f"saves ${monthly_savings:,.0f}/month "
                    f"at {_DEFAULT_DAILY_VOLUME:,} daily runs."
                    + conservative_note
                )

            recommendations.append(
                Recommendation(
                    id=f"cache-context-{step_a}-{step_b}",
                    type="architecture",
                    title=title,
                    description=desc,
                    monthly_savings=monthly_savings,
                    confidence="HIGH",
                    affected_steps=[step_a, step_b],
                    evidence={
                        "system_prompt_tokens": avg_tokens,
                        "occurrences": len(costs),
                        "avg_redundant_cost_per_run": round(avg_redundant, 6),
                        "daily_volume": _DEFAULT_DAILY_VOLUME,
                    },
                    priority=compute_priority(monthly_savings, "HIGH"),
                )
            )

        return recommendations
