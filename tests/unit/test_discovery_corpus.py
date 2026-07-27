"""Discovery corpus: realistic fixture scripts with pinned expectations.

Every file in tests/unit/discovery_corpus/ represents one workflow shape found
in the wild. This test loads each via the real module loader, runs _find_workflow,
and asserts the expected outcome. New bug reports become new fixtures.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import click
import pytest

from pretia.runner import (
    AmbiguousEntrypointError,
    EntrypointError,
    _callable_accepts_single_input,
    _find_workflow,
    _load_workflow_module,
)

_CORPUS_DIR = Path(__file__).parent / "discovery_corpus"

EXPECTATIONS: dict[str, dict] = {
    "compiled_graph": {
        "outcome": "found",
        "has_invoke": True,
    },
    "canonical_workflow_fn": {
        "outcome": "found",
    },
    "sole_async_fn": {
        "outcome": "found",
    },
    "sole_sync_fn": {
        "outcome": "found",
    },
    "class_agent": {
        "outcome": "found",
    },
    "class_agent_async": {
        "outcome": "found",
        "is_coroutine": True,
    },
    "instance_canonical": {
        "outcome": "found",
    },
    "instance_noncanonical": {
        "outcome": "found",
    },
    "agent_factory": {
        "outcome": "found",
    },
    "graph_builder": {
        "outcome": "found",
        "has_invoke": True,
    },
    "cli_script": {
        "outcome": "entrypoint_error",
        "message_contains": ["run_agent"],
    },
    "two_plain_fns": {
        "outcome": "ambiguous",
        "candidates_contain": ["foo", "bar"],
    },
    "two_agent_classes": {
        "outcome": "ambiguous",
    },
    "fastapi_shaped": {
        "outcome": "found",
    },
    "imported_util_trap": {
        "outcome": "entrypoint_error",
    },
    "empty_module": {
        "outcome": "none",
    },
    "main_guard_only": {
        "outcome": "none",
    },
    "llm_object_only": {
        "outcome": "none",
    },
    "callable_instance": {
        "outcome": "found",
    },
    "class_ctor_args_with_instance": {
        "outcome": "found",
    },
    "sync_generator": {
        "outcome": "found",
    },
    "async_generator": {
        "outcome": "found",
    },
    "click_command": {
        "outcome": "found",
    },
    "crewai_shaped": {
        "outcome": "none",
    },
    "wrapped_function": {
        "outcome": "found",
    },
    "all_export": {
        "outcome": "found",
    },
}


@pytest.fixture(params=sorted(EXPECTATIONS.keys()))
def corpus_fixture(request):
    """Yield (fixture_stem, fixture_path, expected_spec)."""
    stem = request.param
    path = _CORPUS_DIR / f"{stem}.py"
    assert path.exists(), f"Missing corpus fixture: {path}"
    return stem, path, EXPECTATIONS[stem]


class TestDiscoveryCorpus:
    def test_fixture_outcome(self, corpus_fixture):
        stem, path, spec = corpus_fixture
        module = _load_workflow_module(str(path))
        mod_name = module.__name__
        try:
            prov: dict[str, str] = {}
            result = _find_workflow(module, provenance=prov)

            expected = spec["outcome"]
            if expected == "found":
                assert result is not None, f"{stem}: expected found, got None"
                if spec.get("has_invoke"):
                    assert hasattr(result, "ainvoke") or hasattr(result, "invoke")
                elif spec.get("is_coroutine"):
                    assert asyncio.iscoroutinefunction(result)
                else:
                    assert callable(result)
                    assert _callable_accepts_single_input(result) or hasattr(result, "ainvoke")
                assert prov.get("rule"), f"{stem}: provenance rule missing"
            elif expected == "none":
                assert result is None, f"{stem}: expected None, got {result}"
            else:
                pytest.fail(f"{stem}: expected {expected} but _find_workflow returned {result}")

        except EntrypointError as exc:
            assert spec["outcome"] == "entrypoint_error", (
                f"{stem}: unexpected EntrypointError: {exc}"
            )
            assert exc.wrapper_snippet, f"{stem}: wrapper_snippet is empty"
            for substr in spec.get("message_contains", []):
                assert substr in str(exc), f"{stem}: expected '{substr}' in error message"

        except AmbiguousEntrypointError as exc:
            assert spec["outcome"] == "ambiguous", (
                f"{stem}: unexpected AmbiguousEntrypointError: {exc}"
            )
            assert len(exc.candidates) >= 2
            for name in spec.get("candidates_contain", []):
                all_text = " ".join(str(c) for c in exc.candidates)
                assert name in all_text, f"{stem}: expected '{name}' in candidates"

        finally:
            sys.modules.pop(mod_name, None)


class TestCorpusCompleteness:
    def test_every_fixture_has_expectation(self):
        """Every .py file in corpus dir (except __init__) has an EXPECTATIONS entry."""
        fixture_stems = {p.stem for p in _CORPUS_DIR.glob("*.py") if p.name != "__init__.py"}
        expected_stems = set(EXPECTATIONS.keys())
        missing = fixture_stems - expected_stems
        assert not missing, f"Fixtures without expectations: {missing}"

    def test_every_expectation_has_fixture(self):
        """Every EXPECTATIONS key has a corresponding .py file."""
        fixture_stems = {p.stem for p in _CORPUS_DIR.glob("*.py") if p.name != "__init__.py"}
        expected_stems = set(EXPECTATIONS.keys())
        orphans = expected_stems - fixture_stems
        assert not orphans, f"Expectations without fixtures: {orphans}"


class TestProductionShapes:
    """Package and sibling-import fixtures loaded through the real loader."""

    def test_relative_import_package(self):
        """A package with relative imports loads and discovers workflow."""
        path = _CORPUS_DIR / "pkg_fixture" / "agent.py"
        module = _load_workflow_module(str(path))
        mod_name = module.__name__
        try:
            result = _find_workflow(module)
            assert result is not None
            assert result("hello") == "Hello"
        finally:
            sys.modules.pop(mod_name, None)

    def test_sibling_import(self):
        """A flat directory with sibling import loads and discovers workflow."""
        path = _CORPUS_DIR / "sibling_fixture" / "agent.py"
        module = _load_workflow_module(str(path))
        mod_name = module.__name__
        try:
            result = _find_workflow(module)
            assert result is not None
            assert "Response:" in result("test")
        finally:
            sys.modules.pop(mod_name, None)

    def test_ipynb_rejected_with_hint(self, tmp_path):
        """A .ipynb file gives a clear error with nbconvert suggestion."""
        nb = tmp_path / "agent.ipynb"
        nb.write_text("{}")
        with pytest.raises(click.UsageError, match="nbconvert"):
            _load_workflow_module(str(nb))

    def test_sys_exit_at_module_level(self, tmp_path):
        """A script calling sys.exit() during import gives a clear hint."""
        f = tmp_path / "exitscript.py"
        f.write_text("import sys\nsys.exit(1)\n")
        with pytest.raises(click.UsageError, match="sys.exit"):
            _load_workflow_module(str(f))

    def test_streamlit_import_rejected(self, tmp_path):
        """A Streamlit app is rejected before execution."""
        f = tmp_path / "stapp.py"
        f.write_text("import streamlit as st\nst.title('hello')\n")
        with pytest.raises(click.UsageError, match="Streamlit"):
            _load_workflow_module(str(f))

    def test_gradio_with_launch_rejected(self, tmp_path):
        """A Gradio script with module-level .launch() is rejected."""
        f = tmp_path / "grapp.py"
        f.write_text(
            "import gradio as gr\n"
            "demo = gr.Interface(fn=lambda x: x, inputs='text', outputs='text')\n"
            "demo.launch()\n"
        )
        with pytest.raises(click.UsageError, match="Gradio"):
            _load_workflow_module(str(f))

    def test_gradio_import_only_not_blocked(self, tmp_path):
        """Gradio import without .launch() passes the source scan."""
        f = tmp_path / "grutil.py"
        f.write_text(
            "from __future__ import annotations\nimport gradio\ndef workflow(inp): return inp\n"
        )
        try:
            module = _load_workflow_module(str(f))
            sys.modules.pop(module.__name__, None)
        except ImportError:
            pass  # gradio not installed is fine; no UsageError about "Gradio server"

    def test_api_key_error_diagnosed(self, tmp_path):
        """An API key error at import gives config diagnosis, not UsageError."""
        f = tmp_path / "needskey.py"
        f.write_text(
            "class OpenAIError(Exception): pass\n"
            "raise OpenAIError('The api_key client option must be set')\n"
        )
        with pytest.raises(click.UsageError, match="API/auth"):
            _load_workflow_module(str(f))

    def test_crewai_wrapper_hint_contains_kickoff(self):
        """CrewAI-shaped module's wrapper snippet mentions kickoff."""
        from pretia.wrapper_hint import build_wrapper_snippet

        path = _CORPUS_DIR / "crewai_shaped.py"
        module = _load_workflow_module(str(path))
        mod_name = module.__name__
        try:
            snippet = build_wrapper_snippet(module, [])
            assert "kickoff" in snippet
        finally:
            sys.modules.pop(mod_name, None)
