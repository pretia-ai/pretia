"""Tests for the pricing tables: lookup, aliasing, cost math, tiering, integration."""

from __future__ import annotations

import dataclasses

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.pricing import (
    calculate_cost,
    get_model_pricing,
    list_models,
    model_tier,
    resolve_model,
)
from agentcost.pricing.tables import MODEL_ALIASES, MODEL_PRICING, MODEL_TIERS


class TestGetModelPricing:
    def test_returns_positive_floats(self):
        in_price, out_price = get_model_pricing("gpt-4o")
        assert isinstance(in_price, float)
        assert isinstance(out_price, float)
        assert in_price > 0
        assert out_price > 0

    def test_returns_per_token_scale(self):
        in_price, out_price = get_model_pricing("gpt-4o")
        assert in_price < 1e-3
        assert out_price < 1e-3

    def test_alias_returns_same_pricing(self):
        assert get_model_pricing("claude-sonnet-4") == get_model_pricing("claude-sonnet-4-6")

    @pytest.mark.parametrize("model", list_models())
    def test_every_canonical_model_has_pricing(self, model):
        in_price, out_price = get_model_pricing(model)
        assert in_price > 0
        assert out_price > 0


class TestResolveModel:
    def test_alias_resolves_to_canonical(self):
        assert resolve_model("claude-sonnet-4") == "claude-sonnet-4-6"

    def test_canonical_returns_itself(self):
        assert resolve_model("claude-sonnet-4-6") == "claude-sonnet-4-6"

    def test_unknown_model_raises_value_error(self):
        with pytest.raises(ValueError, match="nonexistent-model"):
            resolve_model("nonexistent-model")

    @pytest.mark.parametrize("alias", sorted(MODEL_ALIASES))
    def test_every_alias_maps_to_existing_canonical(self, alias):
        assert resolve_model(alias) in MODEL_PRICING


class TestCalculateCost:
    def test_returns_positive_float(self):
        cost = calculate_cost("gpt-4o-mini", input_tokens=1000, output_tokens=500)
        assert isinstance(cost, float)
        assert cost > 0

    def test_zero_tokens_is_zero_cost(self):
        assert calculate_cost("gpt-4o-mini", 0, 0) == 0.0

    def test_alias_and_canonical_agree(self):
        a = calculate_cost("claude-sonnet-4", 1000, 500)
        b = calculate_cost("claude-sonnet-4-6", 1000, 500)
        assert a == b

    def test_math_matches_pricing_dict(self):
        in_per_m, out_per_m = MODEL_PRICING["gpt-4o"]
        input_tokens, output_tokens = 10_000, 2_000
        expected = round(
            (in_per_m / 1_000_000) * input_tokens + (out_per_m / 1_000_000) * output_tokens,
            6,
        )
        assert calculate_cost("gpt-4o", input_tokens, output_tokens) == expected

    def test_result_is_rounded_to_six_decimals(self):
        in_price, out_price = get_model_pricing("gpt-4o-mini")
        raw = 1 * in_price + 1 * out_price
        assert calculate_cost("gpt-4o-mini", 1, 1) == round(raw, 6)

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="totally-fake-model"):
            calculate_cost("totally-fake-model", 100, 100)


class TestModelTier:
    def test_opus_is_frontier(self):
        assert model_tier("claude-opus-4-7") == "frontier"

    def test_mini_is_fast(self):
        assert model_tier("gpt-4o-mini") == "fast"

    def test_alias_returns_tier(self):
        assert model_tier("claude-sonnet-4") == "mid"

    @pytest.mark.parametrize("model", list_models())
    def test_every_model_has_a_tier(self, model):
        assert model_tier(model) in {"frontier", "mid", "fast"}


class TestListModels:
    def test_returns_sorted_strings(self):
        models = list_models()
        assert all(isinstance(m, str) for m in models)
        assert models == sorted(models)

    def test_length_matches_pricing_dict(self):
        assert len(list_models()) == len(MODEL_PRICING)

    def test_contains_only_canonical_names(self):
        assert set(list_models()) == set(MODEL_PRICING)

    def test_aliases_excluded(self):
        canonical = set(list_models())
        pure_aliases = set(MODEL_ALIASES) - set(MODEL_PRICING)
        assert canonical.isdisjoint(pure_aliases)


class TestStructuralInvariants:
    """Catch drift between MODEL_PRICING, MODEL_ALIASES, and MODEL_TIERS."""

    def test_every_model_has_a_tier_entry(self):
        assert set(MODEL_PRICING) == set(MODEL_TIERS)

    def test_every_alias_targets_an_existing_model(self):
        assert set(MODEL_ALIASES.values()).issubset(MODEL_PRICING)

    def test_pricing_covers_all_required_providers(self):
        models = " ".join(list_models())
        assert "claude" in models
        assert "gpt" in models
        assert "gemini" in models
        assert "llama" in models
        assert "mistral" in models
        assert "deepseek" in models


class TestDeepSeekPricing:
    def test_deepseek_v4_flash_in_pricing_table(self):
        in_per_m, out_per_m = MODEL_PRICING["deepseek-v4-flash"]
        assert in_per_m == 0.14
        assert out_per_m == 0.28

    def test_deepseek_v4_pro_in_pricing_table(self):
        in_per_m, out_per_m = MODEL_PRICING["deepseek-v4-pro"]
        assert in_per_m == 0.435
        assert out_per_m == 0.87

    def test_deepseek_alias_resolution(self):
        assert resolve_model("deepseek") == "deepseek-v4-flash"
        assert resolve_model("deepseek-v4") == "deepseek-v4-pro"
        assert resolve_model("deepseek-flash") == "deepseek-v4-flash"
        assert resolve_model("deepseek-pro") == "deepseek-v4-pro"

    def test_deepseek_legacy_aliases(self):
        assert resolve_model("deepseek-chat") in MODEL_PRICING
        assert resolve_model("deepseek-reasoner") in MODEL_PRICING

    def test_deepseek_model_tier(self):
        assert model_tier("deepseek-v4-pro") == "frontier"
        assert model_tier("deepseek-v4-flash") == "mid"
        assert model_tier("deepseek-chat") == "mid"
        assert model_tier("deepseek-reasoner") == "mid"

    def test_calculate_cost_deepseek_v4_flash(self):
        cost = calculate_cost("deepseek-v4-flash", 1_000_000, 500_000)
        expected = 1_000_000 * 0.14 / 1e6 + 500_000 * 0.28 / 1e6
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_calculate_cost_deepseek_v4_pro(self):
        cost = calculate_cost("deepseek-v4-pro", 1_000_000, 500_000)
        expected = 1_000_000 * 0.435 / 1e6 + 500_000 * 0.87 / 1e6
        assert cost == pytest.approx(expected, abs=1e-6)

    def test_deepseek_extreme_budget_comparison(self):
        ds_cost = calculate_cost("deepseek-v4-flash", 100_000, 50_000)
        opus_cost = calculate_cost("claude-opus-4-7", 100_000, 50_000)
        assert ds_cost < opus_cost * 0.05


class TestStepRecordIntegration:
    def test_calculate_cost_from_step_record(self, sample_record):
        record = dataclasses.replace(sample_record, model="gpt-4o-mini")
        cost = calculate_cost(record.model, record.input_tokens, record.output_tokens)
        in_per_m, out_per_m = MODEL_PRICING["gpt-4o-mini"]
        expected = round(
            (in_per_m / 1_000_000) * record.input_tokens
            + (out_per_m / 1_000_000) * record.output_tokens,
            6,
        )
        assert cost == expected

    def test_step_record_built_with_pricing_model(self, sample_record):
        record = dataclasses.replace(sample_record, model="gpt-4o-mini")
        pricing = {record.model: get_model_pricing(record.model)}
        assert isinstance(record, StepRecord)
        assert record.cost(pricing) == pytest.approx(
            calculate_cost(record.model, record.input_tokens, record.output_tokens),
            abs=1e-6,
        )


# ---------------------------------------------------------------------------
# Pricing table structural validation
# ---------------------------------------------------------------------------

# Models referenced by backtesting workflow configs (from tests/backtesting/workflows/_shared.py)
_BACKTESTING_MODELS = [
    "claude-haiku-4-5",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
    "gpt-4.1-nano",
    "gpt-4.1",
    "gemini-2.5-flash",
    "qwen-turbo",
    "qwen3.6-plus",
    "deepseek-v4-flash",
]


class TestAllBacktestingModelsInPricingTable:
    @pytest.mark.parametrize("model", _BACKTESTING_MODELS)
    def test_model_in_table(self, model):
        cost = calculate_cost(model, input_tokens=1000, output_tokens=1000)
        assert cost > 0


class TestPricingTableNoZeroPrices:
    @pytest.mark.parametrize("model", sorted(MODEL_PRICING.keys()))
    def test_no_zero_price(self, model):
        input_price, output_price = MODEL_PRICING[model]
        assert input_price > 0, f"{model} has zero input price"
        assert output_price > 0, f"{model} has zero output price"


class TestPricingTableInputCheaperThanOutput:
    @pytest.mark.parametrize("model", sorted(MODEL_PRICING.keys()))
    def test_input_leq_output(self, model):
        input_price, output_price = MODEL_PRICING[model]
        assert input_price <= output_price, (
            f"{model}: input ${input_price}/MTok > output ${output_price}/MTok — "
            "unusual pricing; verify against provider docs"
        )


# ---------------------------------------------------------------------------
# Anthropic cache-hit pricing
# ---------------------------------------------------------------------------


class TestAnthropicCacheHitPricing:
    @pytest.mark.parametrize(
        "model,expected_rate",
        [
            ("claude-opus-4-7", 0.50),
            ("claude-opus-4-6", 0.50),
            ("claude-sonnet-4-6", 0.30),
            ("claude-haiku-4-5", 0.10),
            ("claude-opus-4-20250514", 1.50),
            ("claude-sonnet-4-20250514", 0.30),
        ],
    )
    def test_anthropic_cache_hit_rate_exists(self, model, expected_rate):
        from agentcost.pricing.tables import MODEL_CACHE_HIT_PRICING

        assert model in MODEL_CACHE_HIT_PRICING
        assert MODEL_CACHE_HIT_PRICING[model] == expected_rate

    def test_anthropic_cache_rate_is_ten_percent_of_input(self):
        from agentcost.pricing.tables import MODEL_CACHE_HIT_PRICING

        for model in MODEL_PRICING:
            if not model.startswith("claude-"):
                continue
            input_price = MODEL_PRICING[model][0]
            expected_cache = input_price * 0.10
            assert model in MODEL_CACHE_HIT_PRICING, f"{model} missing cache pricing"
            assert MODEL_CACHE_HIT_PRICING[model] == pytest.approx(expected_cache, rel=1e-6)

    def test_calculate_cost_anthropic_with_cache_cheaper(self):
        cost_no_cache = calculate_cost("claude-haiku-4-5", input_tokens=2000, output_tokens=500)
        cost_with_cache = calculate_cost(
            "claude-haiku-4-5",
            input_tokens=2000,
            output_tokens=500,
            cache_hit_tokens=1500,
            cache_miss_tokens=500,
        )
        assert cost_with_cache < cost_no_cache


class TestCacheHitPricingInvariants:
    def test_all_cache_models_in_pricing_table(self):
        from agentcost.pricing.tables import MODEL_CACHE_HIT_PRICING

        assert set(MODEL_CACHE_HIT_PRICING).issubset(MODEL_PRICING)

    def test_cache_hit_rate_less_than_input_price(self):
        from agentcost.pricing.tables import MODEL_CACHE_HIT_PRICING

        for model, cache_rate in MODEL_CACHE_HIT_PRICING.items():
            input_price = MODEL_PRICING[model][0]
            assert cache_rate < input_price, (
                f"{model}: cache rate {cache_rate} >= input price {input_price}"
            )
