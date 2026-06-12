#!/usr/bin/env python
"""Unified validation: 4-stage gate for the AgentCost backtesting suite.

Usage::

    python scripts/validate.py                  # All 4 stages
    python scripts/validate.py --stage 1        # Data readiness only
    python scripts/validate.py --skip-live      # Stages 1-3 (no API calls)
    python scripts/validate.py --output r.json  # JSON report
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from scripts._validation_types import CheckResult, CheckStatus, StageResult, ValidationReport

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
INPUTS_PROF = ROOT / "inputs" / "generated" / "profiling"
INPUTS_GT = ROOT / "inputs" / "generated" / "ground_truth"
PDFS_PROF = ROOT / "pdfs" / "generated" / "profiling"
PDFS_GT = ROOT / "pdfs" / "generated" / "ground_truth"

ACTIVE_WORKFLOWS = [
    "W01", "W02", "W05", "W09", "W11", "W12", "W13",
    "W14", "W15", "W16", "W17", "W18", "W19",
]
ACTIVE_WF_SHORT = [
    "W1", "W2", "W5", "W9", "W11", "W12", "W13",
    "W14", "W15", "W16", "W17", "W18", "W19",
]
DROPPED_WORKFLOWS = ["W04"]

CANONICAL_DIR_MAP: dict[str, str] = {
    "W01": "w01", "W02": "w02", "W05": "w5", "W09": "w09",
    "W11": "w11", "W12": "w12", "W13": "w13", "W14": "w14",
    "W15": "w15", "W16": "w16", "W17": "w17", "W18": "w18", "W19": "w19",
}

EXPECTED_GT_MIN: dict[str, int] = {
    "W01": 200, "W02": 200, "W05": 200, "W09": 200, "W11": 200,
    "W12": 200, "W13": 200, "W14": 200, "W15": 200, "W16": 200,
    "W17": 200, "W18": 200, "W19": 200,
}

PDF_WORKFLOWS = {"W14", "W15", "W16", "W17", "W18"}

EMBEDDING_PATHS = {
    "W14_prof": ROOT / "pdfs" / "w14_corpus" / "embeddings.json",
    "W14_gt": ROOT / "pdfs" / "w14_corpus" / "embeddings_gt.json",
    "W15_prof": ROOT / "pdfs" / "w15_corpus" / "embeddings.json",
    "W15_gt": ROOT / "pdfs" / "w15_corpus" / "embeddings_gt.json",
    "W17_prof": ROOT / "pdfs" / "w17_corpus" / "embeddings.json",
}

REQUIRED_API_KEYS: dict[str, list[str]] = {
    "ANTHROPIC_API_KEY": ["W1", "W5", "W13", "W14", "W16", "W17"],
    "OPENAI_API_KEY": ["W9", "W14", "W15", "W17"],
    "DEEPSEEK_API_KEY": ["W2", "W12", "W15", "W18", "W19"],
    "DASHSCOPE_API_KEY": ["W11"],
    "GEMINI_API_KEY": ["W15"],
}

EXPECTED_STEPS: dict[str, tuple[int, int]] = {
    "W1": (1, 1), "W5": (1, 1), "W11": (1, 1), "W12": (1, 1), "W18": (1, 1),
    "W9": (2, 2),
    "W2": (2, 20), "W13": (2, 4), "W14": (2, 3), "W15": (3, 12),
    "W16": (3, 30), "W17": (1, 4), "W19": (8, 8),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_input_dir(base: Path, wf: str) -> Path | None:
    canonical = CANONICAL_DIR_MAP.get(wf)
    if canonical:
        d = base / canonical
        if d.is_dir():
            return d
    wf_num = wf.replace("W", "").replace("w", "")
    for pattern in [f"w{wf_num.zfill(2)}", f"w{wf_num}"]:
        d = base / pattern
        if d.is_dir():
            return d
    return None


def _count_json(d: Path) -> int:
    if not d.is_dir():
        return 0
    return len(list(d.glob("*.json")))


def _wf_short(wf: str) -> str:
    num = int(wf.replace("W", "").replace("w", ""))
    return f"W{num}"


def _load_manifest() -> list[dict[str, Any]]:
    path = PROMPTS_DIR / "manifest.json"
    with open(path) as f:
        return json.load(f).get("prompts", [])


def _load_dotenv() -> None:
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _cr(name: str, status: str, details: dict[str, Any], blocking: bool = True,
        check_id: str | None = None) -> CheckResult:
    return CheckResult(
        name=name,
        status=CheckStatus(status),
        details=details,
        blocking=blocking,
        check_id=check_id,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 1: DATA READINESS (no network, <5s)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage_1(
    prompts_dir: Path = PROMPTS_DIR,
    inputs_prof: Path = INPUTS_PROF,
    inputs_gt: Path = INPUTS_GT,
    pdfs_prof: Path = PDFS_PROF,
    pdfs_gt: Path = PDFS_GT,
) -> StageResult:
    t0 = time.monotonic()
    checks: list[CheckResult] = []

    # --- Prompts ---
    checks.extend(_check_prompts(prompts_dir))

    # --- Inputs ---
    checks.extend(_check_inputs(inputs_prof, inputs_gt))

    # --- PDFs ---
    checks.extend(_check_pdfs(pdfs_prof, pdfs_gt))

    return StageResult(
        stage=1,
        name="Data Readiness",
        checks=tuple(checks),
        duration_s=time.monotonic() - t0,
    )


def _check_prompts(prompts_dir: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    if not (prompts_dir / "manifest.json").exists():
        results.append(_cr("prompt_manifest", "FAIL", {"error": "manifest.json not found"}))
        return results

    manifest = json.load(open(prompts_dir / "manifest.json")).get("prompts", [])
    active_entries = [e for e in manifest if e["workflow_id"] not in DROPPED_WORKFLOWS]

    # File existence
    missing = [
        e["file_path"] for e in active_entries
        if not (prompts_dir / e["file_path"]).exists()
    ]
    if missing:
        results.append(_cr("prompt_file_existence", "FAIL", {"missing": missing}, check_id="1.1"))
    else:
        results.append(_cr(
            "prompt_file_existence", "PASS",
            {"active_prompts": len(active_entries)}, check_id="1.1",
        ))

    # JSON schema completeness
    json_steps = [e for e in active_entries if e.get("output_format") == "json"]
    missing_schema = []
    for entry in json_steps:
        fp = prompts_dir / entry["file_path"]
        if fp.exists():
            text = fp.read_text(encoding="utf-8")
            if "{" not in text or '"' not in text:
                missing_schema.append(f"{entry['workflow_id']}/{entry['step_name']}")
    if missing_schema:
        results.append(_cr(
            "json_schema_completeness", "WARN",
            {"missing": missing_schema}, blocking=False, check_id="1.5",
        ))
    else:
        results.append(_cr(
            "json_schema_completeness", "PASS",
            {"json_step_count": len(json_steps)}, check_id="1.5",
        ))

    # Domain content verification
    domain_checks: list[tuple[str, str, list[str]]] = [
        ("W01", "w01_support_simple/step1_classify_respond.txt",
         ["TechFlow", "$49", "$199", "$499"]),
        ("W11", "w11_support_qwen/step1_classify_respond.txt",
         ["TechFlow", "$49", "$199", "$499"]),
        ("W09", "w09_sales_outreach/step1_qualify.txt", ["NovaCRM"]),
        ("W19", "w19_multi_turn/step1_respond.txt", ["CloudOps"]),
        ("W17", "w17_claims_agent/step1_intake_override.txt",
         ["inactive", "missing", "5000"]),
        ("W17", "w17_claims_agent/step3_evaluate_decide.txt",
         ["approve_pre_authorization", "approve_claim_payment", "deny_claim",
          "request_missing_documentation", "route_to_senior_reviewer", "route_to_coding_review"]),
    ]
    domain_failures: list[str] = []
    for _wf, relpath, keywords in domain_checks:
        fp = prompts_dir / relpath
        if not fp.exists():
            domain_failures.append(f"{_wf}: file missing {relpath}")
            continue
        text = fp.read_text(encoding="utf-8")
        for kw in keywords:
            if kw.lower() not in text.lower():
                domain_failures.append(f"{_wf}/{relpath}: missing '{kw}'")

    for _wf, relpath in [("W14", "w14_simple_rag/step4_generate_answer.txt"),
                         ("W15", "w15_multihop_rag/step5_generate_answer.txt")]:
        fp = prompts_dir / relpath
        if fp.exists():
            text = fp.read_text(encoding="utf-8").lower()
            if "insurance" not in text and "health" not in text and "policy" not in text:
                domain_failures.append(f"{_wf}: no insurance/health/policy reference")

    if domain_failures:
        status = "FAIL" if len(domain_failures) > 3 else "WARN"
        results.append(_cr(
            "domain_content", status,
            {"failures": domain_failures},
            blocking=len(domain_failures) > 3, check_id="1.6",
        ))
    else:
        results.append(_cr("domain_content", "PASS", {}, check_id="1.6"))

    # W1/W11 parity
    w1_path = prompts_dir / "w01_support_simple/step1_classify_respond.txt"
    w11_path = prompts_dir / "w11_support_qwen/step1_classify_respond.txt"
    if w1_path.exists() and w11_path.exists():
        w1_lines = w1_path.read_text(encoding="utf-8").splitlines()
        w11_lines = w11_path.read_text(encoding="utf-8").splitlines()
        total_lines = max(len(w1_lines), len(w11_lines))
        diffs = sum(
            1 for i in range(min(len(w1_lines), len(w11_lines)))
            if w1_lines[i] != w11_lines[i]
        ) + abs(len(w1_lines) - len(w11_lines))
        diff_pct = (diffs / total_lines * 100) if total_lines > 0 else 0
        if diff_pct <= 25:
            results.append(_cr("w1_w11_parity", "PASS", {"diff_pct": diff_pct}, check_id="1.7"))
        else:
            results.append(_cr(
                "w1_w11_parity", "WARN",
                {"diff_pct": diff_pct}, blocking=False, check_id="1.7",
            ))
    else:
        results.append(_cr(
            "w1_w11_parity", "FAIL",
            {"error": "missing prompt files"}, check_id="1.7",
        ))

    # W17 cross-check
    w17_issues: list[str] = []
    intake_path = prompts_dir / "w17_claims_agent/step1_intake_override.txt"
    eval_path = prompts_dir / "w17_claims_agent/step3_evaluate_decide.txt"
    for path, rules in [
        (intake_path, ["inactive", "missing", "5000", "inconsistency"]),
        (eval_path, ["pre-authorization", "standard", "appeal",
                     "approve_pre_authorization", "approve_claim_payment", "deny_claim",
                     "request_missing_documentation", "route_to_senior_reviewer",
                     "route_to_coding_review"]),
    ]:
        if path.exists():
            text = path.read_text(encoding="utf-8").lower()
            for rule in rules:
                if rule not in text:
                    w17_issues.append(f"{path.name}: missing '{rule}'")
        else:
            w17_issues.append(f"{path.name} missing")

    if w17_issues:
        status = "FAIL" if len(w17_issues) > 2 else "WARN"
        results.append(_cr(
            "w17_crosscheck", status,
            {"issues": w17_issues}, blocking=len(w17_issues) > 2, check_id="1.8",
        ))
    else:
        results.append(_cr("w17_crosscheck", "PASS", {}, check_id="1.8"))

    return results


def _check_inputs(inputs_prof: Path, inputs_gt: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    # File counts
    count_issues: list[str] = []
    for wf in ACTIVE_WORKFLOWS:
        wf_short = _wf_short(wf)
        prof_dir = _resolve_input_dir(inputs_prof, wf)
        gt_dir = _resolve_input_dir(inputs_gt, wf)

        if prof_dir is None:
            count_issues.append(f"{wf_short}: profiling dir not found")
        else:
            n = _count_json(prof_dir)
            if wf == "W11":
                if n < 50:
                    count_issues.append(f"{wf_short}: profiling has {n} (need >=50)")
            elif n != 50:
                count_issues.append(f"{wf_short}: profiling has {n} (expected 50)")

        if gt_dir is None:
            count_issues.append(f"{wf_short}: ground_truth dir not found")
        else:
            n = _count_json(gt_dir)
            min_expected = EXPECTED_GT_MIN.get(wf, 200)
            if n < min_expected:
                count_issues.append(f"{wf_short}: ground_truth has {n} (need >={min_expected})")

    if count_issues:
        results.append(_cr("input_file_counts", "FAIL", {"issues": count_issues}, check_id="2.1"))
    else:
        results.append(_cr("input_file_counts", "PASS", {}, check_id="2.1"))

    # Per-set + cross-set via verify_inputs
    try:
        from inputs.validation.verify_inputs import load_inputs as vi_load
        from inputs.validation.verify_inputs import run_all_checks, run_cross_checks
        per_set_failures: list[str] = []
        cross_failures: list[str] = []

        for wf in ACTIVE_WORKFLOWS:
            wf_short = _wf_short(wf)
            prof_dir = _resolve_input_dir(inputs_prof, wf)
            gt_dir = _resolve_input_dir(inputs_gt, wf)
            if prof_dir is None or gt_dir is None:
                continue
            try:
                prof_inputs = vi_load(str(prof_dir))
                gt_inputs = vi_load(str(gt_dir))
            except Exception as exc:
                per_set_failures.append(f"{wf_short}: load error: {exc}")
                continue

            for name, passed, msg in run_all_checks(prof_inputs, "profiling", wf_short):
                if not passed:
                    per_set_failures.append(f"{wf_short}/prof/{name}: {msg.split(chr(10))[0]}")
            for name, passed, msg in run_all_checks(gt_inputs, "ground_truth", wf_short):
                if not passed:
                    per_set_failures.append(f"{wf_short}/gt/{name}: {msg.split(chr(10))[0]}")
            for name, passed, msg in run_cross_checks(prof_inputs, gt_inputs, wf_short):
                if not passed:
                    cross_failures.append(f"{wf_short}/{name}: {msg.split(chr(10))[0]}")

        if per_set_failures:
            results.append(_cr(
                "input_per_set_checks", "WARN",
                {"failures": per_set_failures[:30]}, blocking=False, check_id="2.2",
            ))
        else:
            results.append(_cr("input_per_set_checks", "PASS", {}, check_id="2.2"))

        if cross_failures:
            results.append(_cr(
                "input_cross_set_checks", "WARN",
                {"failures": cross_failures[:20]}, blocking=False, check_id="2.14",
            ))
        else:
            results.append(_cr("input_cross_set_checks", "PASS", {}, check_id="2.14"))

    except ImportError as exc:
        results.append(_cr(
            "input_per_set_checks", "WARN",
            {"error": f"verify_inputs import: {exc}"}, blocking=False, check_id="2.2",
        ))

    return results


def _check_pdfs(pdfs_prof: Path, pdfs_gt: Path) -> list[CheckResult]:
    results: list[CheckResult] = []

    pdf_corpora: list[tuple[str, Path, str, str]] = [
        ("W14/W15 profiling", pdfs_prof / "w14_w15_corpus" / "pdfs", "W14", "profiling"),
        ("W14/W15 GT", pdfs_gt / "w14_w15_corpus" / "pdfs", "W14", "ground_truth"),
        ("W16 profiling", pdfs_prof / "w16", "W16", "profiling"),
        ("W16 GT", pdfs_gt / "w16", "W16", "ground_truth"),
        ("W17 profiling", pdfs_prof / "w17", "W17", "profiling"),
        ("W18 profiling", pdfs_prof / "w18", "W18", "profiling"),
        ("W18 GT", pdfs_gt / "w18", "W18", "ground_truth"),
    ]

    try:
        from pdfs.validation.verify_pdfs import verify_corpus

        for label, corpus_path, wf, profile in pdf_corpora:
            if not corpus_path.exists():
                if "GT" in label and "W17" in label:
                    continue
                results.append(_cr(
                    f"pdf_corpus_{label}", "FAIL",
                    {"error": f"dir missing: {corpus_path}"}, check_id="3.1",
                ))
                continue
            pdfs = list(corpus_path.rglob("*.pdf"))
            if not pdfs:
                if "W17 GT" in label:
                    continue
                results.append(_cr(
                    f"pdf_corpus_{label}", "WARN",
                    {"error": f"no PDFs in {corpus_path}"}, blocking=False, check_id="3.1",
                ))
                continue
            try:
                report = verify_corpus(corpus_path, wf, profile)
                fails = sum(
                    1 for r in report.pdf_reports for c in r.checks if c.status == "FAIL"
                )
                if fails > 0:
                    results.append(_cr(
                        f"pdf_corpus_{label}", "WARN",
                        {"fail_count": fails, "pdf_count": len(pdfs)},
                        blocking=False, check_id="3.1",
                    ))
                else:
                    results.append(_cr(
                        f"pdf_corpus_{label}", "PASS",
                        {"pdf_count": len(pdfs)}, check_id="3.1",
                    ))
            except Exception as exc:
                results.append(_cr(
                    f"pdf_corpus_{label}", "WARN",
                    {"error": str(exc)}, blocking=False, check_id="3.1",
                ))
    except ImportError:
        results.append(_cr(
            "pdf_validity", "WARN",
            {"error": "pdfplumber/verify_pdfs not available"},
            blocking=False, check_id="3.1",
        ))

    # Embeddings existence
    for label, emb_path in EMBEDDING_PATHS.items():
        if emb_path.exists():
            try:
                data = json.loads(emb_path.read_text(encoding="utf-8"))
                chunks = data.get("chunks", [])
                if chunks:
                    dim = len(chunks[0].get("embedding", []))
                    results.append(_cr(
                        f"embeddings_{label}", "PASS",
                        {"chunks": len(chunks), "dim": dim}, check_id="3.8",
                    ))
                else:
                    results.append(_cr(
                        f"embeddings_{label}", "FAIL",
                        {"error": "0 chunks"}, check_id="3.8",
                    ))
            except Exception as exc:
                results.append(_cr(
                    f"embeddings_{label}", "FAIL",
                    {"error": str(exc)}, check_id="3.8",
                ))
        else:
            results.append(_cr(
                f"embeddings_{label}", "FAIL",
                {"error": f"missing: {emb_path}"}, check_id="3.8",
            ))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 2: INFRASTRUCTURE READINESS (no API calls, <5s)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage_2() -> StageResult:
    t0 = time.monotonic()
    checks: list[CheckResult] = []

    # API keys
    _load_dotenv()
    missing_keys: list[str] = []
    for key, workflows in REQUIRED_API_KEYS.items():
        if not os.environ.get(key, ""):
            missing_keys.append(f"{key} ({', '.join(workflows)})")
    if missing_keys:
        checks.append(_cr("api_keys", "FAIL", {"missing": missing_keys}, check_id="6.1"))
    else:
        checks.append(_cr(
            "api_keys", "PASS",
            {"count": len(REQUIRED_API_KEYS)}, check_id="6.1",
        ))

    # Agent loads
    load_failures: list[str] = []
    loaded_agents: dict[str, Any] = {}
    for wf in ACTIVE_WF_SHORT:
        try:
            from bt_agents.harness.run_workflow import load_agent
            agent = load_agent(wf)
            if not hasattr(agent, "execute"):
                load_failures.append(f"{wf}: no execute() method")
            else:
                loaded_agents[wf] = agent
        except Exception as exc:
            load_failures.append(f"{wf}: {exc}")
    if load_failures:
        checks.append(_cr("agent_loads", "FAIL", {"failures": load_failures}, check_id="4.1"))
    else:
        checks.append(_cr(
            "agent_loads", "PASS",
            {"count": len(loaded_agents)}, check_id="4.1",
        ))

    # Prompt wiring
    prompt_issues: list[str] = []
    for wf in ACTIVE_WF_SHORT:
        try:
            from bt_agents.harness.run_workflow import load_prompts
            load_prompts(wf)
        except Exception as exc:
            prompt_issues.append(f"{wf}: {exc}")
    if prompt_issues:
        checks.append(_cr("prompt_wiring", "FAIL", {"issues": prompt_issues}, check_id="4.2"))
    else:
        checks.append(_cr("prompt_wiring", "PASS", {}, check_id="4.2"))

    # Model pricing + LiteLLM consistency
    pricing_issues: list[str] = []
    try:
        from agentcost.pricing.tables import MODEL_PRICING, resolve_model
        from bt_agents.providers.llm import LITELLM_MODEL_MAP

        manifest = _load_manifest()
        active_models: set[str] = set()
        for entry in manifest:
            if entry["workflow_id"] not in DROPPED_WORKFLOWS:
                active_models.add(entry["target_model"])
                if entry.get("alternate_model"):
                    active_models.add(entry["alternate_model"])

        for model in sorted(active_models):
            try:
                canonical = resolve_model(model)
                if canonical not in MODEL_PRICING:
                    pricing_issues.append(f"{model}: not in MODEL_PRICING")
            except Exception as exc:
                pricing_issues.append(f"{model}: {exc}")
            if model not in LITELLM_MODEL_MAP:
                pricing_issues.append(f"{model}: not in LITELLM_MODEL_MAP")

        agent_models = set(LITELLM_MODEL_MAP.keys())
        for m in sorted(agent_models):
            try:
                resolve_model(m)
            except Exception:
                pricing_issues.append(f"{m}: LITELLM model without pricing")
    except ImportError as exc:
        pricing_issues.append(f"import error: {exc}")

    if pricing_issues:
        checks.append(_cr(
            "pricing_consistency", "WARN",
            {"issues": pricing_issues}, blocking=False, check_id="5.3",
        ))
    else:
        checks.append(_cr("pricing_consistency", "PASS", {}, check_id="5.3"))

    # StepRecord schema + projection smoke test
    try:
        from agentcost.collectors.base import StepRecord
        from agentcost.projection.stats import compute_stats

        test_records: list[list[StepRecord]] = []
        for i in range(20):
            sr = StepRecord(
                step_name="test_step",
                step_type="llm",
                model="claude-haiku-4-5",
                input_tokens=200 + i * 10,
                output_tokens=100 + i * 5,
                context_size=200 + i * 10,
                tool_definitions_tokens=0,
                system_prompt_hash="abc123",
                system_prompt_tokens=500,
                output_format="text",
                is_retry=False,
                iteration=1,
                parent_step=None,
                duration_ms=1000,
                timestamp=datetime.now(UTC),
            )
            test_records.append([sr])

        stats = compute_stats(test_records)
        if stats and stats.step_stats:
            checks.append(_cr(
                "schema_and_projection", "PASS",
                {"step_stats_count": len(stats.step_stats)}, check_id="5.1",
            ))
        else:
            checks.append(_cr(
                "schema_and_projection", "FAIL",
                {"error": "empty results"}, check_id="5.1",
            ))
    except Exception as exc:
        checks.append(_cr(
            "schema_and_projection", "FAIL",
            {"error": str(exc)}, check_id="5.1",
        ))

    # Backtesting configs
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from tests.backtesting.configs import BACKTESTING_CONFIGS; "
             "names = [c.name.split('-')[0].upper() for c in BACKTESTING_CONFIGS]; "
             "print(','.join(names))"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
        if result.returncode == 0:
            config_wfs = set(result.stdout.strip().split(","))
            missing = set(ACTIVE_WF_SHORT) - config_wfs
            if missing:
                checks.append(_cr(
                    "backtesting_configs", "WARN",
                    {"missing": sorted(missing)}, blocking=False, check_id="5.4",
                ))
            else:
                checks.append(_cr(
                    "backtesting_configs", "PASS",
                    {"count": len(config_wfs)}, check_id="5.4",
                ))
        else:
            checks.append(_cr(
                "backtesting_configs", "WARN",
                {"error": result.stderr[:200]}, blocking=False, check_id="5.4",
            ))
    except Exception as exc:
        checks.append(_cr(
            "backtesting_configs", "WARN",
            {"error": str(exc)}, blocking=False, check_id="5.4",
        ))

    # Concurrency settings
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, json; sys.path.insert(0, '.'); "
             "from tests.backtesting.concurrency import PROVIDER_PARALLEL; "
             "print(json.dumps(PROVIDER_PARALLEL))"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
        if result.returncode == 0:
            providers = json.loads(result.stdout)
            high = {k: v for k, v in providers.items() if v > 50}
            if high:
                checks.append(_cr(
                    "concurrency_settings", "WARN",
                    {"high_parallel": high}, blocking=False, check_id="6.2",
                ))
            else:
                checks.append(_cr(
                    "concurrency_settings", "PASS",
                    {"providers": providers}, check_id="6.2",
                ))
    except Exception as exc:
        checks.append(_cr(
            "concurrency_settings", "WARN",
            {"error": str(exc)}, blocking=False, check_id="6.2",
        ))

    # Output dirs
    for d in [ROOT / "results", ROOT / "reports"]:
        d.mkdir(parents=True, exist_ok=True)
    checks.append(_cr("output_dirs", "PASS", {}, check_id="6.4"))

    return StageResult(
        stage=2,
        name="Infrastructure Readiness",
        checks=tuple(checks),
        duration_s=time.monotonic() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 3: ENGINE CALIBRATION (no API calls, ~10s)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage_3() -> StageResult:
    t0 = time.monotonic()
    checks: list[CheckResult] = []

    try:
        from tests.synthetic.calibration import compute_calibration_report
        from tests.synthetic.generators import generate_all_synthetic_workflows
        from tests.synthetic.runner import run_synthetic_calibration

        workflows = generate_all_synthetic_workflows()
        results = run_synthetic_calibration(workflows)
        report = compute_calibration_report(workflows, results)

        p50_ok = report.p50_calibration_pct >= 85.0
        p95_ok = report.p95_coverage_pct >= 70.0

        if p50_ok and p95_ok:
            checks.append(_cr("synthetic_calibration", "PASS", {
                "p50_calibration": report.p50_calibration_pct,
                "p95_coverage": report.p95_coverage_pct,
                "workflow_count": len(workflows),
            }))
        else:
            checks.append(_cr("synthetic_calibration", "FAIL", {
                "p50_calibration": report.p50_calibration_pct,
                "p95_coverage": report.p95_coverage_pct,
                "p50_target": 85.0,
                "p95_target": 70.0,
            }))

    except ImportError as exc:
        checks.append(_cr(
            "synthetic_calibration", "WARN",
            {"error": f"import: {exc}"}, blocking=False,
        ))
    except Exception as exc:
        checks.append(_cr(
            "synthetic_calibration", "FAIL",
            {"error": str(exc)},
        ))

    return StageResult(
        stage=3,
        name="Engine Calibration",
        checks=tuple(checks),
        duration_s=time.monotonic() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# STAGE 4: LIVE SMOKE TEST (API calls, ~$0.15, ~2 min)
# ═══════════════════════════════════════════════════════════════════════════

def run_stage_4() -> StageResult:
    t0 = time.monotonic()
    checks: list[CheckResult] = []
    total_cost = 0.0

    _load_dotenv()

    try:
        from bt_agents.harness.run_workflow import load_agent, load_prompts
    except ImportError as exc:
        checks.append(_cr("live_smoke", "FAIL", {"error": f"import: {exc}"}))
        return StageResult(
            stage=4, name="Live Smoke Test",
            checks=tuple(checks), duration_s=time.monotonic() - t0,
        )

    from scripts._smoke_checks import run_smoke_checks

    for wf in ACTIVE_WF_SHORT:
        try:
            agent = load_agent(wf)
            prompts = load_prompts(wf)
        except Exception as exc:
            checks.append(_cr(f"smoke_{wf}", "FAIL", {"error": f"load: {exc}"}))
            continue

        # Get cheapest input
        padded = f"W{wf.replace('W', '').zfill(2)}"
        prof_dir = _resolve_input_dir(INPUTS_PROF, padded)
        input_data: dict[str, Any] = {"input": f"Smoke test {wf}", "tier": "easy"}
        if prof_dir:
            easy_files = sorted(prof_dir.glob("*easy*.json"))
            if easy_files:
                try:
                    inp = json.loads(easy_files[0].read_text())
                    input_data = inp.get("input_data", {})
                    input_data["tier"] = inp.get("tier", "easy")
                except Exception:  # noqa: S110
                    pass

        try:
            records = asyncio.run(agent.execute(input_data, prompts))
            if not records:
                checks.append(_cr(f"smoke_{wf}", "FAIL", {"error": "0 StepRecords"}))
                continue

            # Run smoke checks
            expected_cost = EXPECTED_STEPS.get(wf, (1, 20))
            wf_checks = run_smoke_checks(
                workflow_id=wf,
                records=records,
                input_data=input_data,
                expected_cost_range=(0.0001, 1.0),
                expected_step_range=expected_cost,
            )
            checks.extend(wf_checks)

            # Track cost
            try:
                from agentcost.pricing.tables import calculate_cost
                run_cost = sum(
                    calculate_cost(r.model, r.input_tokens, r.output_tokens)
                    for r in records
                )
                total_cost += run_cost
            except Exception:  # noqa: S110
                pass

            if total_cost > 10.0:
                checks.append(_cr(
                    "budget_exceeded", "FAIL",
                    {"total_cost": total_cost, "stopped_at": wf},
                ))
                break

        except Exception as exc:
            checks.append(_cr(f"smoke_{wf}", "FAIL", {"error": str(exc)[:200]}))

    checks.append(_cr(
        "live_smoke_cost", "PASS" if total_cost <= 10.0 else "WARN",
        {"total_cost_usd": round(total_cost, 4)}, blocking=False,
    ))

    return StageResult(
        stage=4,
        name="Live Smoke Test",
        checks=tuple(checks),
        duration_s=time.monotonic() - t0,
    )


# ═══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════

def run_validation(
    *,
    stages: list[int] | None = None,
    skip_live: bool = False,
    output: Path | None = None,
) -> ValidationReport:
    requested = stages or ([1, 2, 3] if skip_live else [1, 2, 3, 4])
    completed: list[StageResult] = []
    t0 = time.monotonic()
    api_cost = 0.0

    stage_fns: dict[int, Any] = {
        1: run_stage_1,
        2: run_stage_2,
        3: run_stage_3,
        4: run_stage_4,
    }

    for stage_num in sorted(requested):
        fn = stage_fns.get(stage_num)
        if fn is None:
            continue

        result = fn()
        completed.append(result)

        if stage_num == 4:
            for c in result.checks:
                cost = c.details.get("total_cost_usd")
                if cost is not None:
                    api_cost = cost

        if not result.passed and stage_num < max(requested):
            break

    report = ValidationReport(
        timestamp=datetime.now(UTC).isoformat(),
        stages=tuple(completed),
        total_duration_s=time.monotonic() - t0,
        api_cost_usd=api_cost,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report.to_json())

    return report


# ═══════════════════════════════════════════════════════════════════════════
# CLI REPORT
# ═══════════════════════════════════════════════════════════════════════════

def format_report(report: ValidationReport) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("UNIFIED VALIDATION REPORT")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("=" * 72)

    total_pass = total_fail = total_warn = 0

    for stage in report.stages:
        lines.append("")
        icon = "PASS" if stage.passed else "FAIL"
        lines.append(
            f"--- Stage {stage.stage}: {stage.name} [{icon}] ({stage.duration_s:.1f}s) ---"
        )
        for check in stage.checks:
            sym = {"PASS": "+", "FAIL": "X", "WARN": "!"}[check.status.value]
            lines.append(f"  [{sym}] {check.name}")
            if check.status != CheckStatus.PASS and check.details:
                for k, v in check.details.items():
                    lines.append(f"       {k}: {v}")

            if check.status == CheckStatus.PASS:
                total_pass += 1
            elif check.status == CheckStatus.FAIL:
                total_fail += 1
            else:
                total_warn += 1

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL, {total_warn} WARN")
    verdict = "PASS" if report.passed else "FAIL"
    lines.append(f"VERDICT: {verdict} (max stage passed: {report.max_passed_stage})")
    if report.api_cost_usd > 0:
        lines.append(f"API COST: ${report.api_cost_usd:.4f}")
    lines.append(f"DURATION: {report.total_duration_s:.1f}s")
    lines.append("=" * 72)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

@click.command()
@click.option("--stage", type=int, default=None, help="Run only this stage (1-4).")
@click.option("--skip-live", is_flag=True, default=False, help="Skip stage 4 (no API calls).")
@click.option("--output", type=click.Path(), default=None, help="JSON report output path.")
def main(stage: int | None, skip_live: bool, output: str | None) -> None:
    """Unified validation: 4-stage gate for AgentCost backtesting."""
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    stage_list = [stage] if stage else None
    report = run_validation(
        stages=stage_list,
        skip_live=skip_live,
        output=Path(output) if output else None,
    )

    click.echo(format_report(report))

    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
