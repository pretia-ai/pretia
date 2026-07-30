"""Tests for remote pricing refresh from the LiteLLM community dataset."""

from __future__ import annotations

import pytest

from pretia.pricing.remote import (
    RemotePricingResult,
    _entry_prices,
    _litellm_lookup,
    is_suspicious_change,
    refresh_known_model_pricing,
)
from pretia.pricing.tables import MODEL_PRICING


def _litellm_entry(input_per_token: float, output_per_token: float) -> dict:
    return {
        "input_cost_per_token": input_per_token,
        "output_cost_per_token": output_per_token,
        "max_tokens": 8192,
    }


class TestEntryPrices:
    def test_converts_per_token_to_per_million(self):
        entry = _litellm_entry(2.5e-6, 1e-5)
        assert _entry_prices(entry) == (2.5, 10.0)

    def test_missing_costs_returns_none(self):
        assert _entry_prices({"max_tokens": 8192}) is None

    def test_non_numeric_costs_returns_none(self):
        assert _entry_prices({"input_cost_per_token": "a", "output_cost_per_token": 1e-6}) is None

    def test_negative_costs_returns_none(self):
        entry = {"input_cost_per_token": -1e-6, "output_cost_per_token": 1e-6}
        assert _entry_prices(entry) is None


class TestLitellmLookup:
    def test_strips_provider_prefix(self):
        data = {"anthropic/claude-test": _litellm_entry(1e-6, 2e-6)}
        lookup = _litellm_lookup(data)
        assert "claude-test" in lookup

    def test_unprefixed_key_wins_over_prefixed(self):
        data = {
            "openai/gpt-test": _litellm_entry(9e-6, 9e-6),
            "gpt-test": _litellm_entry(1e-6, 2e-6),
        }
        lookup = _litellm_lookup(data)
        assert lookup["gpt-test"]["input_cost_per_token"] == 1e-6

    def test_skips_non_dict_entries(self):
        lookup = _litellm_lookup({"sample_spec": "string value"})
        assert lookup == {}


class TestRefreshKnownModelPricing:
    def test_changed_price_is_reported(self):
        current = MODEL_PRICING["gpt-4o"]
        data = {"gpt-4o": _litellm_entry((current[0] + 1) / 1e6, current[1] / 1e6)}
        result = refresh_known_model_pricing(data)
        assert "gpt-4o" in result.changed
        assert result.changed["gpt-4o"] == (current[0] + 1, current[1])

    def test_matching_price_is_unchanged(self):
        current = MODEL_PRICING["gpt-4o"]
        data = {"gpt-4o": _litellm_entry(current[0] / 1e6, current[1] / 1e6)}
        result = refresh_known_model_pricing(data)
        assert "gpt-4o" in result.unchanged
        assert "gpt-4o" not in result.changed

    def test_unknown_remote_models_are_ignored(self):
        data = {"some-brand-new-model": _litellm_entry(1e-6, 2e-6)}
        result = refresh_known_model_pricing(data)
        assert "some-brand-new-model" not in result.changed
        assert "some-brand-new-model" not in result.unchanged

    def test_model_absent_from_remote_is_missing(self):
        result = refresh_known_model_pricing({})
        assert "gpt-4o" in result.missing

    def test_alias_resolves_to_canonical(self):
        # "mistral-large" aliases "mistral-large-latest"; remote data only has the alias.
        current = MODEL_PRICING["mistral-large-latest"]
        data = {"mistral-large": _litellm_entry((current[0] + 0.5) / 1e6, current[1] / 1e6)}
        result = refresh_known_model_pricing(data)
        assert "mistral-large-latest" in result.changed

    def test_canonical_name_preferred_over_alias(self):
        current = MODEL_PRICING["mistral-large-latest"]
        data = {
            "mistral-large-latest": _litellm_entry((current[0] + 1) / 1e6, current[1] / 1e6),
            "mistral-large": _litellm_entry(9e-6, 9e-6),
        }
        result = refresh_known_model_pricing(data)
        assert result.changed["mistral-large-latest"] == (current[0] + 1, current[1])


class TestIsSuspiciousChange:
    def test_small_increase_not_suspicious(self):
        assert is_suspicious_change((2.0, 6.0), (2.5, 7.0)) is False

    def test_more_than_double_is_suspicious(self):
        assert is_suspicious_change((2.0, 6.0), (8.0, 24.0)) is True

    def test_less_than_half_is_suspicious(self):
        assert is_suspicious_change((2.0, 6.0), (0.5, 1.5)) is True

    def test_exactly_double_not_suspicious(self):
        assert is_suspicious_change((1.0, 2.0), (2.0, 4.0)) is False

    def test_zero_new_price_is_suspicious(self):
        assert is_suspicious_change((0.13, 0.13), (0.13, 0.0)) is True


class TestToModelsDict:
    def test_serializes_to_override_schema(self):
        result = RemotePricingResult(changed={"gpt-4o": (3.0, 12.0)})
        assert result.to_models_dict() == {"gpt-4o": {"input": 3.0, "output": 12.0}}


class TestFetchRemotePricing:
    def test_network_error_raises_connection_error(self, monkeypatch):
        import urllib.error
        import urllib.request

        def _fail(*args, **kwargs):
            raise urllib.error.URLError("unreachable")

        monkeypatch.setattr(urllib.request, "urlopen", _fail)
        from pretia.pricing.remote import fetch_remote_pricing

        with pytest.raises(ConnectionError):
            fetch_remote_pricing()

    def test_invalid_json_raises_value_error(self, monkeypatch):
        import io
        import urllib.request

        class _FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"not json"))
        from pretia.pricing.remote import fetch_remote_pricing

        with pytest.raises(ValueError):
            fetch_remote_pricing()

    def test_non_object_json_raises_value_error(self, monkeypatch):
        import io
        import urllib.request

        class _FakeResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: _FakeResponse(b"[1, 2, 3]"))
        from pretia.pricing.remote import fetch_remote_pricing

        with pytest.raises(ValueError):
            fetch_remote_pricing()
