"""Tests for pdfs/generators/_llm.py: pure-Python helpers (no API calls)."""

from __future__ import annotations

from pdfs.generators._llm import (
    _build_system_prompt,
    _build_user_prompt,
    compute_generator_hash,
    corpus_spec,
    count_tokens,
)


class TestBuildSystemPrompt:
    def test_contains_document_type(self):
        prompt = _build_system_prompt("policy_document", "health_insurance", None)
        assert "policy_document" in prompt

    def test_contains_domain(self):
        prompt = _build_system_prompt("report", "corporate_finance", None)
        assert "corporate_finance" in prompt

    def test_includes_provider_when_given(self):
        prompt = _build_system_prompt("policy", "insurance", "United Healthcare")
        assert "United Healthcare" in prompt

    def test_omits_provider_when_none(self):
        prompt = _build_system_prompt("report", "finance", None)
        assert "None" not in prompt


class TestBuildUserPrompt:
    def test_contains_target_pages(self):
        sections = [{"title": "Intro", "target_pages": 3}]
        prompt = _build_user_prompt(sections, 10, 8000, None, "well_structured")
        assert "10-page" in prompt

    def test_contains_section_titles(self):
        sections = [{"title": "Coverage Overview", "target_pages": 2}]
        prompt = _build_user_prompt(sections, 5, 4000, None, "well_structured")
        assert "Coverage Overview" in prompt

    def test_includes_key_values(self):
        prompt = _build_user_prompt([], 5, 4000, {"deductible": "$1,500"}, "well_structured")
        assert "$1,500" in prompt

    def test_poorly_structured_instructions(self):
        prompt = _build_user_prompt([], 5, 4000, None, "poorly_structured")
        assert "messy" in prompt.lower() or "inconsistent" in prompt.lower()


class TestCountTokens:
    def test_returns_positive_int(self):
        result = count_tokens("hello world this is a test")
        assert isinstance(result, int)
        assert result > 0

    def test_empty_string_returns_zero(self):
        assert count_tokens("") == 0

    def test_longer_text_returns_more(self):
        short = count_tokens("hello")
        long = count_tokens("hello world this is a much longer piece of text")
        assert long > short


class TestComputeGeneratorHash:
    def test_deterministic(self, tmp_path):
        f = tmp_path / "gen.py"
        f.write_text("print('hello')")
        h1 = compute_generator_hash(str(f))
        h2 = compute_generator_hash(str(f))
        assert h1 == h2
        assert len(h1) == 64

    def test_changes_with_content(self, tmp_path):
        f = tmp_path / "gen.py"
        f.write_text("v1")
        h1 = compute_generator_hash(str(f))
        f.write_text("v2")
        h2 = compute_generator_hash(str(f))
        assert h1 != h2


class TestCorpusSpec:
    def test_has_required_keys(self, tmp_path):
        f = tmp_path / "gen.py"
        f.write_text("x")
        spec = corpus_spec(str(f), seed=42, n_inputs=50)
        assert "generator_version" in spec
        assert "seed" in spec
        assert spec["seed"] == 42
        assert spec["n_inputs"] == 50
        assert "generated_at" in spec
        assert spec["generator_version"].startswith("sha256:")
