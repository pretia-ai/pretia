from __future__ import annotations

import json

import pytest

from agents.harness.tool_sim import (
    TOOL_SCHEMAS,
    _calculator,
    _unit_converter,
    _web_search,
    simulate_tool_call,
)

# ---------------------------------------------------------------------------
# TOOL_SCHEMAS
# ---------------------------------------------------------------------------


class TestToolSchemas:
    """Validate the structure of TOOL_SCHEMAS."""

    def test_schema_count(self):
        assert len(TOOL_SCHEMAS) == 3

    def test_each_schema_has_type_and_function_name(self):
        for schema in TOOL_SCHEMAS:
            assert schema["type"] == "function"
            assert "name" in schema["function"]

    def test_expected_tool_names_present(self):
        names = {s["function"]["name"] for s in TOOL_SCHEMAS}
        assert names == {"web_search", "calculator", "unit_converter"}


# ---------------------------------------------------------------------------
# Calculator
# ---------------------------------------------------------------------------


class TestCalculator:
    """Test the calculator tool via its internal function."""

    def test_addition(self):
        assert "5" in _calculator("2 + 3")

    def test_multiplication_float(self):
        assert "10" in _calculator("2.5 * 4")

    def test_exponentiation(self):
        assert "1024" in _calculator("2 ** 10")

    def test_sqrt(self):
        assert "4" in _calculator("sqrt(16)")

    def test_abs(self):
        assert "5" in _calculator("abs(-5)")

    def test_empty_expression_raises(self):
        with pytest.raises(ValueError, match="Empty expression"):
            _calculator("")

    def test_unsafe_expression_raises(self):
        with pytest.raises(ValueError):
            _calculator("__import__('os')")


# ---------------------------------------------------------------------------
# Unit converter
# ---------------------------------------------------------------------------


class TestUnitConverter:
    """Test the unit_converter tool."""

    @staticmethod
    def _extract_converted(result_str: str) -> float:
        """Parse the numeric converted value from '1 km = 0.6214 mi'."""
        rhs = result_str.split("=")[1].strip()
        return float(rhs.split()[0])

    def test_km_to_mi(self):
        result = _unit_converter(1, "km", "mi")
        assert self._extract_converted(result) == pytest.approx(0.6214, abs=0.01)

    def test_f_to_c(self):
        result = _unit_converter(32, "F", "C")
        assert self._extract_converted(result) == pytest.approx(0, abs=0.1)

    def test_kg_to_lb(self):
        result = _unit_converter(100, "kg", "lb")
        assert self._extract_converted(result) == pytest.approx(220.46, abs=0.1)

    def test_mi_to_km(self):
        result = _unit_converter(1, "mi", "km")
        assert self._extract_converted(result) == pytest.approx(1.6093, abs=0.01)

    def test_unsupported_pair_raises(self):
        with pytest.raises(ValueError, match="Unsupported conversion"):
            _unit_converter(1, "parsec", "lightyear")


# ---------------------------------------------------------------------------
# Web search
# ---------------------------------------------------------------------------


class TestWebSearch:
    """Test the web_search tool."""

    def test_exchange_rate_query(self):
        result = _web_search("USD to EUR exchange rate")
        assert len(result) > 0

    def test_unknown_query_returns_generic(self):
        result = _web_search("completely obscure nonsensical query xyz123")
        assert isinstance(result, str)
        assert len(result) > 0


# ---------------------------------------------------------------------------
# simulate_tool_call dispatch
# ---------------------------------------------------------------------------


class TestSimulateToolCall:
    """Test the top-level dispatch function."""

    def test_calculator_dispatch(self):
        raw = simulate_tool_call("calculator", {"expression": "2+3"})
        payload = json.loads(raw)
        assert "result" in payload

    def test_web_search_dispatch(self):
        raw = simulate_tool_call("web_search", {"query": "test"})
        payload = json.loads(raw)
        assert "result" in payload

    def test_unknown_tool_returns_error(self):
        raw = simulate_tool_call("unknown_tool", {})
        payload = json.loads(raw)
        assert "error" in payload
        assert "Unknown tool" in payload["error"]
