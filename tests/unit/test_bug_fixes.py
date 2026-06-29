"""Tests for specific bug fixes (BUG-1, BUG-7, BUG-16)."""

from __future__ import annotations

import types
from unittest.mock import patch

import pytest

from pretia.runner import _find_workflow

# ---------------------------------------------------------------------------
# BUG-1: _find_workflow prefers ainvoke-capable canonical names
# ---------------------------------------------------------------------------


class TestFindWorkflowPreference:
    def test_prefers_app_with_ainvoke_over_graph_without(self):
        mod = types.ModuleType("test_mod")

        class FakeStateGraph:
            pass

        class FakeCompiledGraph:
            async def ainvoke(self, payload, config=None):
                pass

            def invoke(self, payload, config=None):
                pass

            nodes = {}

        mod.graph = FakeStateGraph()
        mod.app = FakeCompiledGraph()

        result = _find_workflow(mod, None)
        assert result is mod.app
        assert hasattr(result, "ainvoke")

    def test_returns_graph_if_it_has_ainvoke(self):
        mod = types.ModuleType("test_mod")

        class FakeCompiledGraph:
            async def ainvoke(self, payload, config=None):
                pass

        mod.graph = FakeCompiledGraph()

        result = _find_workflow(mod, None)
        assert result is mod.graph

    def test_falls_back_to_non_ainvoke_canonical_name(self):
        mod = types.ModuleType("test_mod")
        mod.workflow = lambda x: x

        result = _find_workflow(mod, None)
        assert result is mod.workflow


# ---------------------------------------------------------------------------
# BUG-7: Langfuse import produces friendly error
# ---------------------------------------------------------------------------


class TestLangfuseImportGuard:
    def test_create_langfuse_client_friendly_error(self, monkeypatch):
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        monkeypatch.setenv("LANGFUSE_HOST", "https://test.langfuse.com")

        blocked = {"langfuse": None, "langfuse.api": None, "langfuse.api.client": None}
        with patch.dict("sys.modules", blocked):
            from pretia.inputs.importer import create_langfuse_client

            with pytest.raises(ImportError, match="pip install pretia\\[langfuse\\]"):
                create_langfuse_client()


# ---------------------------------------------------------------------------
# BUG-16: _safe_cost returns 0.0 silently for empty model
# ---------------------------------------------------------------------------


class TestSafeCostEmptyModel:
    def test_empty_model_returns_zero_no_warning(self, caplog):
        from pretia.pricing.tables import calculate_cost
        from pretia.projection.stats import _safe_cost

        result = _safe_cost(calculate_cost, "", 100, 50)
        assert result == 0.0
        assert "Unknown model" not in caplog.text

    def test_none_model_returns_zero_no_warning(self, caplog):
        from pretia.pricing.tables import calculate_cost
        from pretia.projection.stats import _safe_cost

        result = _safe_cost(calculate_cost, None, 100, 50)
        assert result == 0.0
        assert "Unknown model" not in caplog.text

    def test_valid_model_still_works(self):
        from pretia.pricing.tables import calculate_cost
        from pretia.projection.stats import _safe_cost

        result = _safe_cost(calculate_cost, "gpt-4o", 1000, 500)
        assert result > 0
