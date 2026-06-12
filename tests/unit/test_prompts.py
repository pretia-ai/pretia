"""Structural verification tests for backtesting system prompts.

Catches drift between prompts/manifest.json and the actual .txt prompt files.
Modeled on tests/unit/test_pricing.py::TestStructuralInvariants.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
MANIFEST_PATH = PROMPTS_DIR / "manifest.json"

ANTHROPIC_MODELS = {"claude-haiku-4-5", "claude-sonnet-4-6", "claude-opus-4-7"}

REQUIRED_MANIFEST_KEYS = {
    "workflow_id",
    "workflow_name",
    "step_name",
    "file_path",
    "target_model",
    "token_budget_min",
    "token_budget_max",
    "output_format",
    "has_cache_bust",
    "cost_critical_elements",
    "measured_token_count",
}

CACHE_BUST_PLACEHOLDER = "CACHE_BUST_SUFFIX"


@pytest.fixture(scope="module")
def manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def prompt_files(manifest):
    """Dict of file_path -> file content for all manifest entries."""
    result = {}
    for entry in manifest["prompts"]:
        path = PROMPTS_DIR / entry["file_path"]
        result[entry["file_path"]] = path.read_text(encoding="utf-8")
    return result


def _find_prompt(manifest, workflow_id, step_name):
    for entry in manifest["prompts"]:
        if entry["workflow_id"] == workflow_id and entry["step_name"] == step_name:
            return entry
    raise ValueError(f"No manifest entry for {workflow_id}/{step_name}")


def _read_prompt(file_path):
    return (PROMPTS_DIR / file_path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Class 1: Manifest Integrity
# ---------------------------------------------------------------------------


class TestManifestIntegrity:
    """Catch drift between manifest.json and actual prompt files on disk."""

    def test_manifest_loads_valid_json(self):
        data = json.loads(MANIFEST_PATH.read_text())
        assert "prompts" in data
        assert isinstance(data["prompts"], list)

    def test_manifest_has_27_entries(self, manifest):
        assert len(manifest["prompts"]) == 27

    def test_every_manifest_entry_has_required_fields(self, manifest):
        for entry in manifest["prompts"]:
            missing = REQUIRED_MANIFEST_KEYS - set(entry.keys())
            assert not missing, f"{entry.get('file_path', '?')}: missing keys {missing}"

    def test_every_manifest_file_exists_on_disk(self, manifest):
        for entry in manifest["prompts"]:
            path = PROMPTS_DIR / entry["file_path"]
            assert path.exists(), f"Missing file: {entry['file_path']}"

    def test_no_orphan_prompt_files(self, manifest):
        manifest_paths = {entry["file_path"] for entry in manifest["prompts"]}
        actual_files = set()
        for txt in PROMPTS_DIR.rglob("*.txt"):
            actual_files.add(str(txt.relative_to(PROMPTS_DIR)))
        orphans = actual_files - manifest_paths
        assert not orphans, f"Orphan files not in manifest: {orphans}"


# ---------------------------------------------------------------------------
# Class 2: Token Budgets
# ---------------------------------------------------------------------------


class TestTokenBudgets:
    """Verify token counts stay within specified budget ranges."""

    def test_budget_ranges_are_valid(self, manifest):
        for entry in manifest["prompts"]:
            assert entry["token_budget_min"] <= entry["token_budget_max"], (
                f"{entry['file_path']}: min ({entry['token_budget_min']}) > "
                f"max ({entry['token_budget_max']})"
            )

    def test_measured_count_within_budget(self, manifest):
        for entry in manifest["prompts"]:
            lo, hi = entry["token_budget_min"], entry["token_budget_max"]
            measured = entry["measured_token_count"]
            assert lo <= measured <= hi, (
                f"{entry['file_path']}: measured {measured} outside budget {lo}-{hi}"
            )

    def test_actual_file_tokens_match_manifest(self, manifest):
        tiktoken = pytest.importorskip("tiktoken")
        enc = tiktoken.get_encoding("cl100k_base")
        for entry in manifest["prompts"]:
            content = _read_prompt(entry["file_path"])
            actual = len(enc.encode(content))
            lo, hi = entry["token_budget_min"], entry["token_budget_max"]
            assert lo <= actual <= hi, (
                f"{entry['file_path']}: actual {actual} tokens outside budget {lo}-{hi}"
            )


# ---------------------------------------------------------------------------
# Class 3: Cache-Bust Placement
# ---------------------------------------------------------------------------


class TestCacheBustPlacement:
    """Verify cache-bust placeholder presence and position."""

    def test_cache_bust_present_when_expected(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if entry["has_cache_bust"]:
                content = prompt_files[entry["file_path"]]
                assert CACHE_BUST_PLACEHOLDER in content, (
                    f"{entry['file_path']}: missing cache-bust placeholder"
                )

    def test_cache_bust_absent_when_not_expected(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if not entry["has_cache_bust"]:
                content = prompt_files[entry["file_path"]]
                assert CACHE_BUST_PLACEHOLDER not in content, (
                    f"{entry['file_path']}: has cache-bust but should not"
                )

    def test_cache_bust_at_end_of_file(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if entry["has_cache_bust"]:
                content = prompt_files[entry["file_path"]]
                tail = content[-150:]
                assert CACHE_BUST_PLACEHOLDER in tail, (
                    f"{entry['file_path']}: cache-bust not at end of file"
                )


# ---------------------------------------------------------------------------
# Class 4: Cost-Critical Annotations
# ---------------------------------------------------------------------------


class TestCostCriticalAnnotations:
    """Verify COST-CRITICAL inline annotations match manifest declarations."""

    def test_cost_critical_annotation_present(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if entry["cost_critical_elements"]:
                content = prompt_files[entry["file_path"]]
                assert "COST-CRITICAL" in content, (
                    f"{entry['file_path']}: has cost-critical elements but no annotation in file"
                )

    def test_annotation_count_gte_elements(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if not entry["cost_critical_elements"]:
                continue
            content = prompt_files[entry["file_path"]]
            count = content.count("COST-CRITICAL")
            expected = len(entry["cost_critical_elements"])
            assert count >= expected, (
                f"{entry['file_path']}: {count} annotations < {expected} declared elements"
            )


# ---------------------------------------------------------------------------
# Class 5: Output Format Consistency
# ---------------------------------------------------------------------------


class TestOutputFormatConsistency:
    """Verify JSON schema presence and enforcement text alignment."""

    def test_json_output_files_contain_schema(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if entry["output_format"] != "json":
                continue
            content = prompt_files[entry["file_path"]]
            quoted_fields = re.findall(r'"[a-z_]+"', content)
            assert len(quoted_fields) >= 3, (
                f"{entry['file_path']}: JSON-output file has fewer than 3 "
                f"quoted field names — schema may be missing"
            )

    def test_json_output_files_contain_enforcement(self, manifest, prompt_files):
        enforcement_patterns = ["valid JSON", "JSON only", "single JSON object"]
        for entry in manifest["prompts"]:
            if entry["output_format"] != "json":
                continue
            content = prompt_files[entry["file_path"]]
            has_enforcement = any(p in content for p in enforcement_patterns)
            assert has_enforcement, (
                f"{entry['file_path']}: JSON-output file missing enforcement text"
            )

    def test_non_anthropic_json_has_extra_enforcement(self, manifest, prompt_files):
        extra = "Your entire response must be a single JSON object"
        for entry in manifest["prompts"]:
            if entry["output_format"] != "json":
                continue
            if entry["target_model"] in ANTHROPIC_MODELS:
                continue
            content = prompt_files[entry["file_path"]]
            assert extra in content, (
                f"{entry['file_path']}: non-Anthropic JSON-output file missing extra enforcement"
            )


# ---------------------------------------------------------------------------
# Class 6: Cross-Workflow Consistency
# ---------------------------------------------------------------------------


class TestCrossWorkflowConsistency:
    """Catch consistency drift across paired workflows."""

    def test_w1_w11_word_count_constraints_identical(self, prompt_files):
        w1 = prompt_files["w01_support_simple/step1_classify_respond.txt"]
        w11 = prompt_files["w11_support_qwen/step1_classify_respond.txt"]
        assert "100 to 200 words" in w1
        assert "100 to 200 words" in w11
        assert "200 to 400 words" in w1
        assert "200 to 400 words" in w11

    def test_w1_w11_techflow_pricing_identical(self, prompt_files):
        w1 = prompt_files["w01_support_simple/step1_classify_respond.txt"]
        w11 = prompt_files["w11_support_qwen/step1_classify_respond.txt"]
        for price in ["$49/month", "$199/month", "$499/month"]:
            assert price in w1, f"W1 missing {price}"
            assert price in w11, f"W11 missing {price}"

    def test_w11_has_qwen_reinforcement(self, prompt_files):
        w1 = prompt_files["w01_support_simple/step1_classify_respond.txt"]
        w11 = prompt_files["w11_support_qwen/step1_classify_respond.txt"]
        assert "Respond in English only" in w11
        assert "Respond in English only" not in w1

    def test_w14_w15_citation_format_identical(self, prompt_files):
        w14 = prompt_files["w14_simple_rag/step4_generate_answer.txt"]
        w15 = prompt_files["w15_multihop_rag/step5_generate_answer.txt"]
        citation = "[Source: {document_name}, page {X}]"
        assert citation in w14, "W14 missing citation format"
        assert citation in w15, "W15 missing citation format"

    def test_w17_has_all_override_rules(self, prompt_files):
        content = prompt_files["w17_claims_agent/step1_intake_override.txt"]
        assert "inactive" in content
        assert "missing" in content.lower() or "Missing" in content
        assert "5000" in content
        assert "code" in content.lower() and "inconsistency" in content.lower()

    def test_w17_has_all_function_schemas(self, prompt_files):
        content = prompt_files["w17_claims_agent/step3_evaluate_decide.txt"]
        functions = [
            "approve_pre_authorization",
            "approve_claim_payment",
            "deny_claim",
            "request_missing_documentation",
            "route_to_senior_reviewer",
            "route_to_coding_review",
        ]
        for fn in functions:
            assert fn in content, f"W17 step3 missing function schema: {fn}"

    def test_w17_has_all_claim_types(self, prompt_files):
        content = prompt_files["w17_claims_agent/step3_evaluate_decide.txt"]
        for claim_type in ["pre_approval", "standard", "appeal"]:
            assert claim_type in content or claim_type.replace("_", "-") in content, (
                f"W17 step3 missing claim type: {claim_type}"
            )


# ---------------------------------------------------------------------------
# Domain Content Checks (parametrized)
# ---------------------------------------------------------------------------


class TestDomainContent:
    """Verify domain-specific content is present in the right prompts."""

    @pytest.mark.parametrize(
        "file_path",
        [
            "w01_support_simple/step1_classify_respond.txt",
            "w02_support_complex/step2_research_draft_loop.txt",
            "w11_support_qwen/step1_classify_respond.txt",
        ],
    )
    @pytest.mark.parametrize("term", ["TechFlow", "$49", "$199", "$499"])
    def test_techflow_domain_content(self, file_path, term):
        content = _read_prompt(file_path)
        assert term in content, f"{file_path} missing TechFlow term: {term}"

    @pytest.mark.parametrize(
        "file_path",
        [
            "w09_sales_outreach/step1_qualify.txt",
            "w09_sales_outreach/step2_draft_email.txt",
        ],
    )
    def test_novacrm_domain_content(self, file_path):
        content = _read_prompt(file_path)
        assert "NovaCRM" in content

    @pytest.mark.parametrize(
        "file_path",
        [
            "w14_simple_rag/step4_generate_answer.txt",
            "w15_multihop_rag/step5_generate_answer.txt",
        ],
    )
    @pytest.mark.parametrize("provider", ["United Healthcare", "Aetna"])
    def test_insurance_domain_content(self, file_path, provider):
        content = _read_prompt(file_path)
        assert provider in content, f"{file_path} missing provider: {provider}"

    def test_cloudops_domain_content(self):
        content = _read_prompt("w19_multi_turn/step1_respond.txt")
        for term in ["CloudOps", "$99", "$399"]:
            assert term in content, f"W19 missing CloudOps term: {term}"

    @pytest.mark.parametrize(
        "file_path",
        [
            "w17_claims_agent/step1_intake_override.txt",
            "w17_claims_agent/step3_evaluate_decide.txt",
        ],
    )
    def test_w17_insurance_providers(self, file_path):
        content = _read_prompt(file_path)
        for provider in ["United Healthcare", "Aetna", "Cigna"]:
            assert provider in content, f"{file_path} missing provider: {provider}"


# ---------------------------------------------------------------------------
# Class 7: Prompt Parseability
# ---------------------------------------------------------------------------


class TestPromptParseability:
    """Verify prompt files are well-formed and free of format errors."""

    def test_files_are_valid_utf8(self, manifest):
        for entry in manifest["prompts"]:
            path = PROMPTS_DIR / entry["file_path"]
            path.read_text(encoding="utf-8")

    def test_files_are_not_empty(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            content = prompt_files[entry["file_path"]]
            assert len(content) >= 100, (
                f"{entry['file_path']}: only {len(content)} chars — too short"
            )

    def test_no_markdown_headers(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            content = prompt_files[entry["file_path"]]
            for i, line in enumerate(content.splitlines(), 1):
                assert not re.match(r"^#{1,3} ", line), (
                    f"{entry['file_path']}:{i}: markdown header found: {line[:60]}"
                )

    def test_no_markdown_code_fences(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            content = prompt_files[entry["file_path"]]
            for i, line in enumerate(content.splitlines(), 1):
                assert "```" not in line, f"{entry['file_path']}:{i}: markdown code fence found"

    def test_embedded_json_schemas_are_valid(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            if entry["output_format"] != "json":
                continue
            content = prompt_files[entry["file_path"]]
            blocks = re.findall(r"\{[^{}]*\}", content)
            parsed_any = False
            for block in blocks:
                try:
                    json.loads(block)
                    parsed_any = True
                    break
                except json.JSONDecodeError:
                    continue
            assert parsed_any, (
                f"{entry['file_path']}: no parseable JSON block found in JSON-output prompt"
            )

    def test_no_unresolved_template_variables(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            content = prompt_files[entry["file_path"]]
            templates = re.findall(r"\{\{[^}]+\}\}", content)
            unexpected = [t for t in templates if "CACHE_BUST_SUFFIX" not in t]
            assert not unexpected, (
                f"{entry['file_path']}: unresolved template variables: {unexpected}"
            )

    def test_no_meta_instructions_leaked(self, manifest, prompt_files):
        forbidden = ["AgentCost", "profiling run", "backtesting", "projection engine"]
        for entry in manifest["prompts"]:
            content = prompt_files[entry["file_path"]]
            for term in forbidden:
                assert term not in content, (
                    f"{entry['file_path']}: meta-instruction leaked: '{term}'"
                )

    def test_no_trailing_whitespace_lines(self, manifest, prompt_files):
        for entry in manifest["prompts"]:
            content = prompt_files[entry["file_path"]]
            for i, line in enumerate(content.splitlines(), 1):
                if line and line != line.rstrip():
                    raise AssertionError(f"{entry['file_path']}:{i}: trailing whitespace")
