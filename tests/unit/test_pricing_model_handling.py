"""Tests for unrecognized model handling and custom model registration."""

from __future__ import annotations

import pytest

from agentcost.pricing import calculate_cost, register_model
from agentcost.pricing.tables import MODEL_PRICING, UnrecognizedModelError


class TestUnrecognizedModelRaisesError:
    def test_raises(self):
        with pytest.raises(UnrecognizedModelError, match="nonexistent-model-v9"):
            calculate_cost("nonexistent-model-v9", 1000, 500)

    def test_message_includes_suggestion(self):
        with pytest.raises(UnrecognizedModelError, match="register_model"):
            calculate_cost("nonexistent-model-v9", 1000, 500)


class TestUnrecognizedModelSimilarNames:
    def test_similar_names_suggested(self):
        with pytest.raises(UnrecognizedModelError, match="Did you mean") as exc_info:
            calculate_cost("claude-sonnet-4.6", 1000, 500)
        assert "claude-sonnet-4-6" in str(exc_info.value)


class TestRegisterCustomModel:
    def test_register_and_calculate(self):
        register_model("custom-test-model-xyz", input_price=0.50, output_price=1.50)
        try:
            cost = calculate_cost("custom-test-model-xyz", 1000, 500)
            expected = (1000 / 1_000_000 * 0.50) + (500 / 1_000_000 * 1.50)
            assert cost == pytest.approx(expected, abs=1e-8)
        finally:
            MODEL_PRICING.pop("custom-test-model-xyz", None)


class TestRegisterModelOverridesError:
    def test_no_error_after_register(self):
        register_model("custom-override-test-abc", input_price=1.0, output_price=2.0)
        try:
            cost = calculate_cost("custom-override-test-abc", 100, 100)
            assert cost > 0
        finally:
            MODEL_PRICING.pop("custom-override-test-abc", None)


class TestKnownModelsStillWork:
    @pytest.mark.parametrize(
        "model",
        [
            "claude-sonnet-4-6",
            "gpt-4o",
            "qwen3.7-max",
            "deepseek-chat",
            "gemini-2.5-flash",
        ],
    )
    def test_known_model(self, model):
        cost = calculate_cost(model, 1000, 500)
        assert cost > 0
