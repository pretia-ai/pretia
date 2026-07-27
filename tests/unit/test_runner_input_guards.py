"""Tests for ProfileRunner._validate_inputs: coercion, blank detection, error hints."""

from __future__ import annotations

import pytest

from pretia.inputs.selector import InputSelection
from pretia.runner import ProfileRunner


def _selection(mode: str = "manual") -> InputSelection:
    """Build a minimal InputSelection for testing."""
    return InputSelection(mode=mode, inputs=[], message="test")


class TestValidateInputs:
    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="0 inputs"):
            ProfileRunner._validate_inputs([], _selection("manual"))

    def test_file_mode_empty_hint(self):
        with pytest.raises(ValueError, match="inputs file"):
            ProfileRunner._validate_inputs([], _selection("file"))

    def test_langfuse_mode_hint(self):
        with pytest.raises(ValueError, match="LANGFUSE"):
            ProfileRunner._validate_inputs([], _selection("langfuse"))

    def test_auto_generate_hint(self):
        with pytest.raises(ValueError, match="generator"):
            ProfileRunner._validate_inputs([], _selection("auto-generate"))

    def test_all_blank_raises(self):
        with pytest.raises(ValueError):
            ProfileRunner._validate_inputs(["", "  ", "\n"], _selection("manual"))

    def test_non_string_coerced(self):
        result = ProfileRunner._validate_inputs(
            [42, "ok"],  # type: ignore[list-item]
            _selection("manual"),
        )
        assert result == ["42", "ok"]

    def test_valid_strings_pass(self):
        result = ProfileRunner._validate_inputs(
            ["hello", "world"],
            _selection("manual"),
        )
        assert result == ["hello", "world"]

    def test_dict_inputs_pass(self):
        result = ProfileRunner._validate_inputs(
            [{"key": "val"}],  # type: ignore[list-item]
            _selection("manual"),
        )
        assert result == [{"key": "val"}]
