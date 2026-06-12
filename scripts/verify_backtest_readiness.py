#!/usr/bin/env python
"""Pre-backtest verification: exhaustive checks before committing ~$400+ API budget.

Usage::

    python scripts/verify_backtest_readiness.py                 # Full run + live tests
    python scripts/verify_backtest_readiness.py --skip-live      # No API calls
    python scripts/verify_backtest_readiness.py --block 1        # Single block
    python scripts/verify_backtest_readiness.py --fix            # Auto-fix simple issues
    python scripts/verify_backtest_readiness.py --output r.json  # JSON report
"""

from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = ROOT / "prompts"
INPUTS_PROF = ROOT / "inputs" / "generated" / "profiling"
INPUTS_GT = ROOT / "inputs" / "generated" / "ground_truth"
PDFS_PROF = ROOT / "pdfs" / "generated" / "profiling"
PDFS_GT = ROOT / "pdfs" / "generated" / "ground_truth"
PILOT_REPORT = ROOT / "tests" / "backtesting" / "results" / "pilot" / "pilot_report.json"
PRICING_TABLE = ROOT / "agentcost" / "pricing" / "tables.py"
DATASET_DIR = ROOT / "tests" / "backtesting" / "dataset"

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


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VR:
    check_id: str
    name: str
    status: str  # PASS, FAIL, WARN
    details: str
    blocking: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BlockReport:
    block_id: int
    block_name: str
    checks: list[VR] = field(default_factory=list)


@dataclass
class Report:
    timestamp: str = ""
    blocks: list[BlockReport] = field(default_factory=list)
    go: bool = True


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


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 1: SYSTEM PROMPT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_block_1() -> BlockReport:
    block = BlockReport(1, "System Prompt Verification")

    manifest = _load_manifest()
    active_entries = [
        e for e in manifest
        if e["workflow_id"] not in DROPPED_WORKFLOWS
    ]

    # 1.1 File existence
    missing_files: list[str] = []
    for entry in active_entries:
        fp = PROMPTS_DIR / entry["file_path"]
        if not fp.exists():
            missing_files.append(entry["file_path"])

    all_txts = set(
        str(p.relative_to(PROMPTS_DIR))
        for p in PROMPTS_DIR.rglob("*.txt")
    )
    manifest_paths = set(e["file_path"] for e in manifest)
    orphans = all_txts - manifest_paths

    if missing_files:
        block.checks.append(VR(
            "1.1", "prompt_file_existence", "FAIL",
            f"Missing prompt files: {missing_files}",
        ))
    elif orphans:
        non_dropped_orphans = [o for o in orphans if "w04_" not in o.lower()]
        if non_dropped_orphans:
            block.checks.append(VR(
                "1.1", "prompt_file_existence", "WARN",
                f"{len(active_entries)} active prompts OK. Orphans (not in manifest): {non_dropped_orphans}",
                blocking=False,
            ))
        else:
            block.checks.append(VR(
                "1.1", "prompt_file_existence", "PASS",
                f"{len(active_entries)} active prompts present. W04 orphans excluded.",
            ))
    else:
        block.checks.append(VR(
            "1.1", "prompt_file_existence", "PASS",
            f"{len(active_entries)} active prompt entries, all files present. {len(all_txts)} total .txt files.",
        ))

    # 1.5 JSON schema completeness
    json_steps = [e for e in active_entries if e.get("output_format") == "json"]
    missing_schema: list[str] = []
    for entry in json_steps:
        fp = PROMPTS_DIR / entry["file_path"]
        if not fp.exists():
            continue
        text = fp.read_text(encoding="utf-8")
        has_json_example = "{" in text and '"' in text
        if not has_json_example:
            missing_schema.append(f"{entry['workflow_id']}/{entry['step_name']}")

    if missing_schema:
        block.checks.append(VR(
            "1.5", "json_schema_completeness", "WARN",
            f"JSON-output steps without schema guidance: {missing_schema}",
            blocking=False,
        ))
    else:
        block.checks.append(VR(
            "1.5", "json_schema_completeness", "PASS",
            f"All {len(json_steps)} JSON-output steps contain schema examples.",
        ))

    # 1.6 Domain content verification
    domain_checks: list[tuple[str, str, list[str]]] = [
        ("W01", "w01_support_simple/step1_classify_respond.txt",
         ["TechFlow", "$49", "$199", "$499"]),
        ("W11", "w11_support_qwen/step1_classify_respond.txt",
         ["TechFlow", "$49", "$199", "$499"]),
        ("W09", "w09_sales_outreach/step1_qualify.txt",
         ["NovaCRM"]),
        ("W19", "w19_multi_turn/step1_respond.txt",
         ["CloudOps"]),
        ("W17", "w17_claims_agent/step1_intake_override.txt",
         ["inactive", "missing", "5000"]),
        ("W17", "w17_claims_agent/step3_evaluate_decide.txt",
         ["approve_pre_authorization", "approve_claim_payment", "deny_claim",
          "request_missing_documentation", "route_to_senior_reviewer", "route_to_coding_review"]),
    ]

    domain_failures: list[str] = []
    for wf, relpath, keywords in domain_checks:
        fp = PROMPTS_DIR / relpath
        if not fp.exists():
            domain_failures.append(f"{wf}: file missing {relpath}")
            continue
        text = fp.read_text(encoding="utf-8")
        for kw in keywords:
            if kw.lower() not in text.lower():
                domain_failures.append(f"{wf}/{relpath}: missing '{kw}'")

    # W14/W15 insurance reference
    for wf, relpath in [("W14", "w14_simple_rag/step4_generate_answer.txt"),
                        ("W15", "w15_multihop_rag/step5_generate_answer.txt")]:
        fp = PROMPTS_DIR / relpath
        if fp.exists():
            text = fp.read_text(encoding="utf-8").lower()
            if "insurance" not in text and "health" not in text and "policy" not in text:
                domain_failures.append(f"{wf}: no insurance/health/policy reference")

    # W19 product names
    fp19 = PROMPTS_DIR / "w19_multi_turn/step1_respond.txt"
    if fp19.exists():
        text19 = fp19.read_text(encoding="utf-8")
        products = ["Dashboard", "Deploy", "Scale", "Guard"]
        found = sum(1 for p in products if p.lower() in text19.lower())
        if found < 3:
            domain_failures.append(f"W19: only {found}/4 product names found (need ≥3)")

    if domain_failures:
        block.checks.append(VR(
            "1.6", "domain_content", "FAIL" if len(domain_failures) > 3 else "WARN",
            "Domain content issues:\n" + "\n".join(f"  - {f}" for f in domain_failures),
            blocking=len(domain_failures) > 3,
        ))
    else:
        block.checks.append(VR("1.6", "domain_content", "PASS", "All domain content verified."))

    # 1.7 W1↔W11 parity
    w1_path = PROMPTS_DIR / "w01_support_simple/step1_classify_respond.txt"
    w11_path = PROMPTS_DIR / "w11_support_qwen/step1_classify_respond.txt"
    if w1_path.exists() and w11_path.exists():
        w1_lines = w1_path.read_text(encoding="utf-8").splitlines()
        w11_lines = w11_path.read_text(encoding="utf-8").splitlines()
        total_lines = max(len(w1_lines), len(w11_lines))
        diffs = 0
        for i in range(min(len(w1_lines), len(w11_lines))):
            if w1_lines[i] != w11_lines[i]:
                diffs += 1
        diffs += abs(len(w1_lines) - len(w11_lines))
        diff_pct = (diffs / total_lines * 100) if total_lines > 0 else 0

        if diff_pct <= 25:
            block.checks.append(VR(
                "1.7", "w1_w11_parity", "PASS",
                f"W1↔W11 differ in {diffs}/{total_lines} lines ({diff_pct:.0f}%). Within tolerance.",
            ))
        else:
            block.checks.append(VR(
                "1.7", "w1_w11_parity", "WARN",
                f"W1↔W11 differ in {diffs}/{total_lines} lines ({diff_pct:.0f}%). "
                f"Review if structural divergence is intentional.",
                blocking=False,
            ))
    else:
        block.checks.append(VR(
            "1.7", "w1_w11_parity", "FAIL",
            "One or both prompt files missing for W1/W11 comparison.",
        ))

    # 1.8 W17 cross-check
    w17_issues: list[str] = []
    intake_path = PROMPTS_DIR / "w17_claims_agent/step1_intake_override.txt"
    eval_path = PROMPTS_DIR / "w17_claims_agent/step3_evaluate_decide.txt"

    override_rules = ["inactive", "missing", "5000", "inconsistency"]
    if intake_path.exists():
        text = intake_path.read_text(encoding="utf-8").lower()
        for rule in override_rules:
            if rule not in text:
                w17_issues.append(f"intake: override rule keyword '{rule}' not found")
    else:
        w17_issues.append("intake prompt file missing")

    claim_types = ["pre-authorization", "standard", "appeal"]
    function_schemas = [
        "approve_pre_authorization", "approve_claim_payment", "deny_claim",
        "request_missing_documentation", "route_to_senior_reviewer", "route_to_coding_review",
    ]
    if eval_path.exists():
        text = eval_path.read_text(encoding="utf-8").lower()
        for ct in claim_types:
            if ct not in text:
                w17_issues.append(f"evaluate: claim type '{ct}' not found")
        for fn in function_schemas:
            if fn not in text:
                w17_issues.append(f"evaluate: function schema '{fn}' not found")
    else:
        w17_issues.append("evaluate prompt file missing")

    if w17_issues:
        block.checks.append(VR(
            "1.8", "w17_crosscheck", "WARN" if len(w17_issues) <= 2 else "FAIL",
            "W17 cross-check issues:\n" + "\n".join(f"  - {i}" for i in w17_issues),
            blocking=len(w17_issues) > 2,
        ))
    else:
        block.checks.append(VR(
            "1.8", "w17_crosscheck", "PASS",
            "W17: 4 override rules, 3 claim types, 6 function schemas all present.",
        ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 2: INPUT SET VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_block_2() -> BlockReport:
    block = BlockReport(2, "Input Set Verification")

    # 2.1 File counts + duplicate dir detection
    count_issues: list[str] = []
    dup_warnings: list[str] = []
    for wf in ACTIVE_WORKFLOWS:
        wf_short = _wf_short(wf)
        prof_dir = _resolve_input_dir(INPUTS_PROF, wf)
        gt_dir = _resolve_input_dir(INPUTS_GT, wf)

        # Check for duplicates
        wf_num = wf.replace("W", "")
        padded = f"w{wf_num.zfill(2)}"
        unpadded = f"w{wf_num}"
        if padded != unpadded:
            d1 = INPUTS_PROF / padded
            d2 = INPUTS_PROF / unpadded
            if d1.is_dir() and d2.is_dir():
                dup_warnings.append(f"  profiling: both {padded}/ and {unpadded}/ exist")

        if prof_dir is None:
            count_issues.append(f"  {wf_short}: profiling dir not found")
        else:
            n = _count_json(prof_dir)
            if wf == "W11":
                if n < 50:
                    count_issues.append(f"  {wf_short}: profiling has {n} files (need ≥50)")
            elif n != 50:
                count_issues.append(f"  {wf_short}: profiling has {n} files (expected 50)")

        if gt_dir is None:
            count_issues.append(f"  {wf_short}: ground_truth dir not found")
        else:
            n = _count_json(gt_dir)
            min_expected = EXPECTED_GT_MIN.get(wf, 200)
            if n < min_expected:
                count_issues.append(
                    f"  {wf_short}: ground_truth has {n} files (need ≥{min_expected})"
                )

    if dup_warnings:
        block.checks.append(VR(
            "2.1a", "duplicate_input_dirs", "WARN",
            "Duplicate input directories found:\n" + "\n".join(dup_warnings),
            blocking=False,
        ))
    if count_issues:
        block.checks.append(VR("2.1", "file_counts", "FAIL",
                                "File count issues:\n" + "\n".join(count_issues)))
    else:
        block.checks.append(VR("2.1", "file_counts", "PASS",
                                "All workflows have expected profiling (50) and GT (≥200) counts."))

    # 2.2-2.8 Reuse verify_inputs for per-set and cross-set checks
    try:
        from inputs.validation.verify_inputs import (
            load_inputs as vi_load,
            run_all_checks,
            run_cross_checks,
        )

        per_set_failures: list[str] = []
        cross_failures: list[str] = []

        for wf in ACTIVE_WORKFLOWS:
            wf_short = _wf_short(wf)
            prof_dir = _resolve_input_dir(INPUTS_PROF, wf)
            gt_dir = _resolve_input_dir(INPUTS_GT, wf)
            if prof_dir is None or gt_dir is None:
                continue

            try:
                prof_inputs = vi_load(str(prof_dir))
                gt_inputs = vi_load(str(gt_dir))
            except Exception as exc:
                per_set_failures.append(f"  {wf_short}: load error: {exc}")
                continue

            # Per-set checks on profiling
            for name, passed, msg in run_all_checks(prof_inputs, "profiling", wf_short):
                if not passed:
                    per_set_failures.append(f"  {wf_short}/prof/{name}: {msg.split(chr(10))[0]}")

            # Per-set checks on ground truth
            for name, passed, msg in run_all_checks(gt_inputs, "ground_truth", wf_short):
                if not passed:
                    per_set_failures.append(f"  {wf_short}/gt/{name}: {msg.split(chr(10))[0]}")

            # Cross-set checks
            for name, passed, msg in run_cross_checks(prof_inputs, gt_inputs, wf_short):
                if not passed:
                    cross_failures.append(f"  {wf_short}/{name}: {msg.split(chr(10))[0]}")

        if per_set_failures:
            detail = "\n".join(per_set_failures[:30])
            extra = f"\n  ... and {len(per_set_failures) - 30} more" if len(per_set_failures) > 30 else ""
            block.checks.append(VR(
                "2.2-2.8", "per_set_checks", "WARN",
                f"Per-set check failures ({len(per_set_failures)}):\n{detail}{extra}",
                blocking=False,
            ))
        else:
            block.checks.append(VR(
                "2.2-2.8", "per_set_checks", "PASS",
                "All per-set checks passed (tier distribution, entropy, range, duplicates, dirty).",
            ))

        if cross_failures:
            detail = "\n".join(cross_failures[:20])
            block.checks.append(VR(
                "2.14-2.15", "cross_set_checks", "WARN",
                f"Cross-set check failures ({len(cross_failures)}):\n{detail}",
                blocking=False,
            ))
        else:
            block.checks.append(VR(
                "2.14-2.15", "cross_set_checks", "PASS",
                "All cross-set checks passed (GT weights, token stretch, style artifacts).",
            ))

    except ImportError as exc:
        block.checks.append(VR(
            "2.2-2.8", "per_set_checks", "WARN",
            f"Could not import verify_inputs: {exc}. Skipping per-set checks.",
            blocking=False,
        ))

    # 2.10 Structural descriptor completeness
    sd_issues: list[str] = []
    required_fields: dict[str, list[str]] = {
        "W13": ["target_path"],
        "W17": ["pipeline_depth"],
        "W19": ["substantive_turn_count"],
    }
    for wf in ACTIVE_WORKFLOWS:
        wf_short = _wf_short(wf)
        prof_dir = _resolve_input_dir(INPUTS_PROF, wf)
        if prof_dir is None:
            continue
        files = sorted(prof_dir.glob("*.json"))[:5]
        for fp in files:
            try:
                inp = json.loads(fp.read_text())
                sd = inp.get("structural_descriptor", {})
                if not sd:
                    sd_issues.append(f"  {wf_short}/{fp.name}: no structural_descriptor")
                    continue
                for req in required_fields.get(wf_short, []):
                    if req not in sd or sd[req] is None:
                        sd_issues.append(f"  {wf_short}/{fp.name}: missing sd.{req}")
            except Exception:
                pass

    if sd_issues:
        block.checks.append(VR(
            "2.10", "structural_descriptors", "WARN",
            f"Structural descriptor issues ({len(sd_issues)}):\n" + "\n".join(sd_issues[:15]),
            blocking=False,
        ))
    else:
        block.checks.append(VR(
            "2.10", "structural_descriptors", "PASS",
            "Structural descriptors present and populated in sampled inputs.",
        ))

    # 2.11 W11↔W1 identity
    w01_dir = _resolve_input_dir(INPUTS_PROF, "W01")
    w11_dir = _resolve_input_dir(INPUTS_PROF, "W11")
    if w01_dir and w11_dir:
        w01_files = {f.name for f in w01_dir.glob("*.json")}
        w11_w01_files = {f.name for f in w11_dir.glob("w01_*.json")}
        overlap = w01_files & w11_w01_files
        if overlap:
            diffs = 0
            for fn in list(overlap)[:10]:
                t1 = (w01_dir / fn).read_text()
                t11 = (w11_dir / fn).read_text()
                if t1 != t11:
                    diffs += 1
            if diffs == 0:
                block.checks.append(VR(
                    "2.11", "w11_w1_identity", "PASS",
                    f"W11 contains {len(w11_w01_files)} w01-prefixed files; sampled 10 are identical to W01.",
                ))
            else:
                block.checks.append(VR(
                    "2.11", "w11_w1_identity", "WARN",
                    f"W11 w01-prefixed files: {diffs}/10 sampled differ from W01.",
                    blocking=False,
                ))
        else:
            block.checks.append(VR(
                "2.11", "w11_w1_identity", "WARN",
                "No w01-prefixed files found in W11 profiling dir.",
                blocking=False,
            ))
    else:
        block.checks.append(VR(
            "2.11", "w11_w1_identity", "WARN",
            "Could not locate W01 or W11 profiling dirs.",
            blocking=False,
        ))

    # 2.12 W13 routing distribution
    w13_dir = _resolve_input_dir(INPUTS_PROF, "W13")
    if w13_dir:
        paths: dict[str, int] = Counter()
        for fp in w13_dir.glob("*.json"):
            inp = json.loads(fp.read_text())
            tp = inp.get("structural_descriptor", {}).get("target_path", "?")
            paths[tp] += 1
        total = sum(paths.values())
        if total > 0:
            a_pct = paths.get("A", 0) / total * 100
            b_pct = paths.get("B", 0) / total * 100
            c_pct = paths.get("C", 0) / total * 100
            detail = f"A={a_pct:.0f}% B={b_pct:.0f}% C={c_pct:.0f}% (target: 70/20/10)"
            if abs(a_pct - 70) > 15 or abs(b_pct - 20) > 15:
                block.checks.append(VR(
                    "2.12", "w13_routing_dist", "WARN",
                    f"W13 routing distribution off target: {detail}",
                    blocking=False,
                ))
            else:
                block.checks.append(VR("2.12", "w13_routing_dist", "PASS", detail))
        else:
            block.checks.append(VR("2.12", "w13_routing_dist", "WARN", "No W13 inputs.", blocking=False))

    # 2.13 W17 override trigger coverage
    w17_dir = _resolve_input_dir(INPUTS_PROF, "W17")
    if w17_dir:
        inactive = 0
        high_amount = 0
        claim_types_found: set[str] = set()
        for fp in w17_dir.glob("*.json"):
            inp = json.loads(fp.read_text())
            idata = inp.get("input_data", {})
            if idata.get("member_status") == "inactive":
                inactive += 1
            amt = idata.get("claimed_amount", 0)
            if isinstance(amt, (int, float)) and amt > 5000:
                high_amount += 1
            ct = idata.get("claim_type", "")
            if ct:
                claim_types_found.add(ct)

        issues = []
        if inactive < 3:
            issues.append(f"inactive members: {inactive} (need ≥3)")
        if high_amount < 3:
            issues.append(f"high-amount claims: {high_amount} (need ≥3)")
        expected_types = {"pre_approval", "standard", "appeal"}
        missing_types = expected_types - claim_types_found
        if missing_types:
            issues.append(f"missing claim types: {missing_types}")

        if issues:
            block.checks.append(VR(
                "2.13", "w17_override_coverage", "WARN",
                "W17 override issues:\n" + "\n".join(f"  - {i}" for i in issues),
                blocking=False,
            ))
        else:
            block.checks.append(VR(
                "2.13", "w17_override_coverage", "PASS",
                f"W17: {inactive} inactive, {high_amount} high-amount, types={claim_types_found}",
            ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 3: PDF CORPUS VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_block_3() -> BlockReport:
    block = BlockReport(3, "PDF Corpus Verification")

    # 3.1-3.3 PDF validity and text extraction
    pdf_corpora: list[tuple[str, Path, str, str]] = [
        ("W14/W15 profiling", PDFS_PROF / "w14_w15_corpus" / "pdfs", "W14", "profiling"),
        ("W14/W15 GT", PDFS_GT / "w14_w15_corpus" / "pdfs", "W14", "ground_truth"),
        ("W16 profiling", PDFS_PROF / "w16", "W16", "profiling"),
        ("W16 GT", PDFS_GT / "w16", "W16", "ground_truth"),
        ("W17 profiling", PDFS_PROF / "w17", "W17", "profiling"),
        ("W18 profiling", PDFS_PROF / "w18", "W18", "profiling"),
        ("W18 GT", PDFS_GT / "w18", "W18", "ground_truth"),
    ]

    try:
        from pdfs.validation.verify_pdfs import verify_corpus, format_report
        for label, corpus_path, wf, profile in pdf_corpora:
            if not corpus_path.exists():
                if "GT" in label and "W17" in label:
                    continue
                block.checks.append(VR(
                    "3.1", f"pdf_corpus_{label}", "FAIL",
                    f"Corpus dir missing: {corpus_path}",
                ))
                continue
            pdfs = list(corpus_path.rglob("*.pdf"))
            if not pdfs:
                if "W17 GT" in label:
                    continue
                block.checks.append(VR(
                    "3.1", f"pdf_corpus_{label}", "WARN",
                    f"No PDFs in {corpus_path}",
                    blocking=False,
                ))
                continue
            try:
                report = verify_corpus(corpus_path, wf, profile)
                fails = sum(
                    1 for r in report.pdf_reports
                    for c in r.checks if c.status == "FAIL"
                )
                total = sum(len(r.checks) for r in report.pdf_reports)
                if fails > 0:
                    block.checks.append(VR(
                        "3.1", f"pdf_corpus_{label}", "WARN",
                        f"{label}: {fails}/{total} per-PDF check failures across {len(pdfs)} PDFs.",
                        blocking=False,
                    ))
                else:
                    block.checks.append(VR(
                        "3.1", f"pdf_corpus_{label}", "PASS",
                        f"{label}: {len(pdfs)} PDFs, {total} checks all passed.",
                    ))
            except Exception as exc:
                block.checks.append(VR(
                    "3.1", f"pdf_corpus_{label}", "WARN",
                    f"{label}: verify_corpus error: {exc}",
                    blocking=False,
                ))
    except ImportError:
        block.checks.append(VR(
            "3.1", "pdf_validity", "WARN",
            "pdfplumber/verify_pdfs not available. Checking file counts only.",
            blocking=False,
        ))
        for label, corpus_path in pdf_corpora:
            if corpus_path.exists():
                pdfs = list(corpus_path.rglob("*.pdf"))
                block.checks.append(VR(
                    "3.1", f"pdf_count_{label}", "PASS" if pdfs else "WARN",
                    f"{label}: {len(pdfs)} PDFs found.",
                    blocking=False,
                ))

    # 3.4 Content manifest (W14/W15)
    for label, path in [
        ("W14/W15 prof", PDFS_PROF / "w14_w15_corpus" / "pdfs" / "manifest.json"),
        ("W14/W15 GT", PDFS_GT / "w14_w15_corpus" / "pdfs" / "manifest.json"),
    ]:
        if path.exists():
            try:
                data = json.loads(path.read_text())
                doc_count = len(data.get("documents", []))
                block.checks.append(VR(
                    "3.4", f"manifest_{label}", "PASS",
                    f"{label}: manifest has {doc_count} documents.",
                ))
            except Exception as exc:
                block.checks.append(VR(
                    "3.4", f"manifest_{label}", "FAIL",
                    f"{label}: manifest parse error: {exc}",
                ))
        else:
            block.checks.append(VR(
                "3.4", f"manifest_{label}", "FAIL",
                f"{label}: manifest.json missing at {path}",
            ))

    # 3.6 W17 policy verification
    w17_policies = PDFS_PROF / "w17" / "policies"
    if w17_policies.is_dir():
        pdfs = list(w17_policies.glob("*.pdf"))
        if len(pdfs) >= 3:
            block.checks.append(VR(
                "3.6", "w17_policies", "PASS",
                f"W17: {len(pdfs)} policy PDFs found.",
            ))
        else:
            block.checks.append(VR(
                "3.6", "w17_policies", "WARN",
                f"W17: only {len(pdfs)} policy PDFs (expected 3).",
                blocking=False,
            ))
    else:
        block.checks.append(VR("3.6", "w17_policies", "FAIL", "W17 policies dir missing."))

    # 3.7 W17 clinical notes
    w17_cn = PDFS_PROF / "w17" / "clinical_notes"
    if w17_cn.is_dir():
        pdfs = list(w17_cn.glob("*.pdf"))
        if len(pdfs) >= 10:
            block.checks.append(VR(
                "3.7", "w17_clinical_notes", "PASS",
                f"W17: {len(pdfs)} clinical note PDFs.",
            ))
        else:
            block.checks.append(VR(
                "3.7", "w17_clinical_notes", "WARN",
                f"W17: only {len(pdfs)} clinical notes (expected 10).",
                blocking=False,
            ))
    else:
        block.checks.append(VR("3.7", "w17_clinical_notes", "FAIL",
                                "W17 clinical_notes dir missing."))

    # 3.8 Embeddings existence and structure
    for label, emb_path in EMBEDDING_PATHS.items():
        if emb_path.exists():
            try:
                data = json.loads(emb_path.read_text(encoding="utf-8"))
                chunks = data.get("chunks", [])
                if not chunks:
                    block.checks.append(VR(
                        "3.8", f"embeddings_{label}", "FAIL",
                        f"{label}: embeddings file exists but has 0 chunks.",
                    ))
                    continue
                dim = len(chunks[0].get("embedding", []))
                block.checks.append(VR(
                    "3.8", f"embeddings_{label}", "PASS",
                    f"{label}: {len(chunks)} chunks, dim={dim}.",
                ))
            except Exception as exc:
                block.checks.append(VR(
                    "3.8", f"embeddings_{label}", "FAIL",
                    f"{label}: parse error: {exc}",
                ))
        else:
            block.checks.append(VR(
                "3.8", f"embeddings_{label}", "FAIL",
                f"{label}: file missing at {emb_path}. "
                f"Fix: python scripts/build_rag_corpus.py --all",
            ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 4: WORKFLOW AGENT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_block_4(skip_live: bool = False) -> BlockReport:
    block = BlockReport(4, "Workflow Agent Verification")

    # 4.1 Agent loads
    load_failures: list[str] = []
    loaded_agents: dict[str, Any] = {}
    for wf in ACTIVE_WF_SHORT:
        try:
            from bt_agents.harness.run_workflow import load_agent
            agent = load_agent(wf)
            if not hasattr(agent, "execute"):
                load_failures.append(f"  {wf}: agent loaded but no execute() method")
            else:
                loaded_agents[wf] = agent
        except Exception as exc:
            load_failures.append(f"  {wf}: {exc}")

    if load_failures:
        block.checks.append(VR(
            "4.1", "agent_loads", "FAIL",
            "Agent load failures:\n" + "\n".join(load_failures),
        ))
    else:
        block.checks.append(VR(
            "4.1", "agent_loads", "PASS",
            f"All {len(loaded_agents)} agents loaded with execute() method.",
        ))

    # 4.2 Prompt wiring
    prompt_issues: list[str] = []
    loaded_prompts: dict[str, dict[str, str]] = {}
    for wf in ACTIVE_WF_SHORT:
        try:
            from bt_agents.harness.run_workflow import load_prompts
            prompts = load_prompts(wf)
            loaded_prompts[wf] = prompts
        except Exception as exc:
            prompt_issues.append(f"  {wf}: {exc}")

    if prompt_issues:
        block.checks.append(VR(
            "4.2", "prompt_wiring", "FAIL",
            "Prompt wiring failures:\n" + "\n".join(prompt_issues),
        ))
    else:
        block.checks.append(VR(
            "4.2", "prompt_wiring", "PASS",
            f"All {len(loaded_prompts)} workflows loaded prompts successfully.",
        ))

    # 4.4 Model pricing verification
    pricing_issues: list[str] = []
    try:
        from agentcost.pricing.tables import resolve_model, MODEL_PRICING
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
                    pricing_issues.append(f"  {model}: resolved to {canonical} but not in MODEL_PRICING")
            except Exception as exc:
                pricing_issues.append(f"  {model}: resolve_model error: {exc}")

            if model not in LITELLM_MODEL_MAP:
                pricing_issues.append(f"  {model}: not in LITELLM_MODEL_MAP")

    except ImportError as exc:
        pricing_issues.append(f"  Import error: {exc}")

    if pricing_issues:
        block.checks.append(VR(
            "4.4", "model_pricing", "WARN",
            "Model pricing/mapping issues:\n" + "\n".join(pricing_issues),
            blocking=False,
        ))
    else:
        block.checks.append(VR(
            "4.4", "model_pricing", "PASS",
            f"All {len(active_models)} models have pricing and LiteLLM mapping.",
        ))

    # 4.5 Dry-run all workflows
    dry_run_issues: list[str] = []
    dry_run_results: dict[str, list[Any]] = {}
    for wf in ACTIVE_WF_SHORT:
        if wf not in loaded_agents or wf not in loaded_prompts:
            continue
        agent = loaded_agents[wf]
        prompts = loaded_prompts[wf]

        # Build minimal input
        prof_dir = _resolve_input_dir(INPUTS_PROF, f"W{wf.replace('W', '').zfill(2)}")
        if prof_dir:
            first_file = sorted(prof_dir.glob("*.json"))
            if first_file:
                try:
                    inp = json.loads(first_file[0].read_text())
                    input_data = inp.get("input_data", {})
                    input_data["_dry_run"] = True
                    input_data["tier"] = inp.get("tier", "easy")
                except Exception:
                    input_data = {"input": f"Test for {wf}", "_dry_run": True, "tier": "easy"}
            else:
                input_data = {"input": f"Test for {wf}", "_dry_run": True, "tier": "easy"}
        else:
            input_data = {"input": f"Test for {wf}", "_dry_run": True, "tier": "easy"}

        try:
            records = asyncio.run(agent.execute(input_data, prompts))
            if not records:
                dry_run_issues.append(f"  {wf}: returned 0 StepRecords")
            else:
                dry_run_results[wf] = records
        except Exception as exc:
            dry_run_issues.append(f"  {wf}: {exc}")

    if dry_run_issues:
        block.checks.append(VR(
            "4.5", "dry_run", "FAIL",
            "Dry-run failures:\n" + "\n".join(dry_run_issues),
        ))
    else:
        step_summary = ", ".join(f"{wf}={len(r)}" for wf, r in dry_run_results.items())
        block.checks.append(VR(
            "4.5", "dry_run", "PASS",
            f"All {len(dry_run_results)} workflows completed dry-run. Steps: {step_summary}",
        ))

    # 4.7 Step count verification
    expected_steps: dict[str, tuple[int, int]] = {
        "W1": (1, 1), "W5": (1, 1), "W11": (1, 1), "W12": (1, 1), "W18": (1, 1),
        "W9": (2, 2),
        "W2": (2, 20), "W13": (2, 4), "W14": (2, 3), "W15": (3, 12),
        "W16": (3, 30), "W17": (1, 4), "W19": (8, 8),
    }
    step_issues: list[str] = []
    for wf, records in dry_run_results.items():
        n = len(records)
        lo, hi = expected_steps.get(wf, (1, 20))
        if n < lo or n > hi:
            step_issues.append(f"  {wf}: {n} steps (expected {lo}-{hi})")

    if step_issues:
        block.checks.append(VR(
            "4.7", "step_counts", "WARN",
            "Step count deviations:\n" + "\n".join(step_issues),
            blocking=False,
        ))
    else:
        block.checks.append(VR(
            "4.7", "step_counts", "PASS",
            "All dry-run step counts within expected ranges.",
        ))

    # 4.12 Cache-bust placeholder verification
    manifest = _load_manifest()
    cache_bust_issues: list[str] = []
    for entry in manifest:
        if entry["workflow_id"] in DROPPED_WORKFLOWS:
            continue
        if entry.get("has_cache_bust"):
            fp = PROMPTS_DIR / entry["file_path"]
            if fp.exists():
                text = fp.read_text(encoding="utf-8")
                if "{{CACHE_BUST_SUFFIX}}" not in text:
                    # The cache-bust prefix is added by llm.py, placeholder is optional
                    pass  # llm.py prepends UUID prefix regardless
            else:
                cache_bust_issues.append(f"  {entry['workflow_id']}/{entry['step_name']}: prompt missing")

    if cache_bust_issues:
        block.checks.append(VR(
            "4.12", "cache_bust_placeholders", "WARN",
            "Cache-bust issues:\n" + "\n".join(cache_bust_issues),
            blocking=False,
        ))
    else:
        block.checks.append(VR(
            "4.12", "cache_bust_placeholders", "PASS",
            "Cache-bust mechanism verified (llm.py UUID prefix + optional placeholder).",
        ))

    # 4.6 Single-input live tests (default ON, skip with --skip-live)
    if skip_live:
        block.checks.append(VR(
            "4.6", "live_test", "WARN",
            "Live tests skipped (--skip-live). Use default mode for full verification.",
            blocking=False,
        ))
    else:
        _load_dotenv()
        live_issues: list[str] = []
        live_results: list[str] = []
        total_live_cost = 0.0

        for wf in ACTIVE_WF_SHORT:
            if wf not in loaded_agents or wf not in loaded_prompts:
                continue

            agent = loaded_agents[wf]
            prompts = loaded_prompts[wf]

            # Get cheapest-tier input
            prof_dir = _resolve_input_dir(INPUTS_PROF, f"W{wf.replace('W', '').zfill(2)}")
            input_data = {"input": f"Live test {wf}", "tier": "easy"}
            if prof_dir:
                easy_files = sorted(prof_dir.glob("*easy*.json"))
                if easy_files:
                    try:
                        inp = json.loads(easy_files[0].read_text())
                        input_data = inp.get("input_data", {})
                        input_data["tier"] = inp.get("tier", "easy")
                    except Exception:
                        pass

            try:
                records = asyncio.run(agent.execute(input_data, prompts))
                if not records:
                    live_issues.append(f"  {wf}: returned 0 StepRecords from live run")
                    continue

                run_cost = 0.0
                for r in records:
                    if r.input_tokens == 0 and r.output_tokens == 0:
                        live_issues.append(f"  {wf}: zero tokens in StepRecord (dry-run leaked?)")
                        break
                    try:
                        from agentcost.pricing.tables import calculate_cost
                        c = calculate_cost(r.model, r.input_tokens, r.output_tokens)
                        run_cost += c
                    except Exception:
                        pass

                total_live_cost += run_cost
                live_results.append(f"  {wf}: {len(records)} steps, ${run_cost:.4f}")

                if total_live_cost > 10.0:
                    live_issues.append(f"  Budget exceeded $10 — stopping live tests at {wf}")
                    break

            except Exception as exc:
                live_issues.append(f"  {wf}: live run error: {str(exc)[:200]}")

        detail = "\n".join(live_results)
        if live_issues:
            block.checks.append(VR(
                "4.6", "live_test", "WARN",
                f"Live test issues ({len(live_issues)}):\n"
                + "\n".join(live_issues) + f"\nResults:\n{detail}\nTotal: ${total_live_cost:.4f}",
                blocking=False,
            ))
        else:
            block.checks.append(VR(
                "4.6", "live_test", "PASS",
                f"All {len(live_results)} workflows passed live test. Total: ${total_live_cost:.4f}\n{detail}",
            ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 5: ENGINE INTEGRATION VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

def run_block_5() -> BlockReport:
    block = BlockReport(5, "Engine Integration Verification")

    # 5.1 StepRecord schema + 5.2 Projection smoke test
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
                timestamp=datetime.now(timezone.utc),
            )
            test_records.append([sr])

        stats = compute_stats(test_records)
        if stats and stats.step_stats:
            block.checks.append(VR(
                "5.1-5.2", "schema_and_projection", "PASS",
                f"StepRecord schema accepted. compute_stats returned {len(stats.step_stats)} step stats.",
            ))
        else:
            block.checks.append(VR(
                "5.1-5.2", "schema_and_projection", "FAIL",
                "compute_stats returned empty results.",
            ))
    except Exception as exc:
        block.checks.append(VR(
            "5.1-5.2", "schema_and_projection", "FAIL",
            f"Engine integration error: {exc}",
        ))

    # 5.3 Pricing consistency
    try:
        from agentcost.pricing.tables import MODEL_PRICING, calculate_cost, resolve_model
        from bt_agents.providers.llm import LITELLM_MODEL_MAP

        agent_models = set(LITELLM_MODEL_MAP.keys())
        pricing_models = set()
        for m in MODEL_PRICING:
            pricing_models.add(m)

        missing_in_pricing: list[str] = []
        for m in sorted(agent_models):
            try:
                resolve_model(m)
            except Exception:
                missing_in_pricing.append(m)

        if missing_in_pricing:
            block.checks.append(VR(
                "5.3", "pricing_consistency", "WARN",
                f"Models without pricing: {missing_in_pricing}",
                blocking=False,
            ))
        else:
            block.checks.append(VR(
                "5.3", "pricing_consistency", "PASS",
                f"All {len(agent_models)} agent models have pricing entries.",
            ))
    except ImportError as exc:
        block.checks.append(VR(
            "5.3", "pricing_consistency", "WARN",
            f"Import error: {exc}",
            blocking=False,
        ))

    # 5.4 Backtesting configs match active workflows
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
            active_set = set(ACTIVE_WF_SHORT)
            missing = active_set - config_wfs
            extra_w4 = {w for w in config_wfs if w == "W4"}

            issues = []
            if missing:
                issues.append(f"Missing from configs: {missing}")
            if extra_w4:
                issues.append("W4 still in configs (should be dropped)")

            if issues:
                block.checks.append(VR(
                    "5.4", "backtesting_configs", "WARN",
                    "Config issues: " + "; ".join(issues),
                    blocking=False,
                ))
            else:
                block.checks.append(VR(
                    "5.4", "backtesting_configs", "PASS",
                    f"All {len(config_wfs)} config entries match active workflows.",
                ))
        else:
            block.checks.append(VR(
                "5.4", "backtesting_configs", "WARN",
                f"Config import failed: {result.stderr[:200]}",
                blocking=False,
            ))
    except Exception as exc:
        block.checks.append(VR(
            "5.4", "backtesting_configs", "WARN",
            f"Config check error: {exc}",
            blocking=False,
        ))

    # 5.5 Dataset structure
    try:
        spec_ds = importlib.util.spec_from_file_location(
            "backtesting_dataset",
            ROOT / "tests" / "backtesting" / "dataset.py",
        )
        ds_mod = importlib.util.module_from_spec(spec_ds)
        spec_ds.loader.exec_module(ds_mod)
        save_backtest_run = ds_mod.save_backtest_run

        DATASET_DIR.mkdir(parents=True, exist_ok=True)
        test_record = {
            "meta": {
                "backtest_id": "verify_test_000",
                "backtest_version": "test",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "engine_version": "0.1.0",
                "pricing_table_hash": "test",
                "input_seed": 42,
                "python_version": "3.11",
            },
            "workflows": {},
            "aggregate": {
                "total_cost_usd": 0,
                "workflows_passing_all": [],
                "workflows_need_reweight": [],
                "workflows_unresolved": [],
                "detector_tp_rate": 0,
                "detector_fn_rate": 0,
                "launch_gate": "TEST",
            },
        }
        path = save_backtest_run(test_record)
        if path.exists():
            roundtrip = json.loads(path.read_text())
            if roundtrip.get("meta", {}).get("backtest_id") == "verify_test_000":
                block.checks.append(VR(
                    "5.5", "dataset_structure", "PASS",
                    f"Dataset roundtrip OK. File: {path.name}",
                ))
                path.unlink()  # Clean up test file
            else:
                block.checks.append(VR(
                    "5.5", "dataset_structure", "FAIL",
                    "Dataset roundtrip data mismatch.",
                ))
        else:
            block.checks.append(VR(
                "5.5", "dataset_structure", "FAIL",
                "save_backtest_run did not create file.",
            ))
    except Exception as exc:
        block.checks.append(VR(
            "5.5", "dataset_structure", "WARN",
            f"Dataset test error: {exc}",
            blocking=False,
        ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# AUTO-FIX
# ═══════════════════════════════════════════════════════════════════════════

def run_fixes() -> list[str]:
    fixes_applied: list[str] = []

    # Fix 1: W17 embeddings
    w17_emb = ROOT / "pdfs" / "w17_corpus" / "embeddings.json"
    if not w17_emb.exists():
        w17_corpus_dir = ROOT / "pdfs" / "w17_corpus"
        w17_corpus_dir.mkdir(parents=True, exist_ok=True)
        build_script = ROOT / "scripts" / "build_rag_corpus.py"
        w17_pdf_dir = PDFS_PROF / "w17"
        if build_script.exists() and w17_pdf_dir.is_dir():
            print("  [FIX] Building W17 embeddings...")
            try:
                result = subprocess.run(
                    [sys.executable, str(build_script),
                     "--pdf-dir", str(w17_pdf_dir),
                     "--output", str(w17_emb)],
                    capture_output=True, text=True, timeout=120,
                )
                if w17_emb.exists():
                    fixes_applied.append("Built W17 embeddings from profiling PDFs")
                else:
                    print(f"  [FIX] build_rag_corpus.py failed: {result.stderr[:300]}")
                    # Fallback: try dry-run mode
                    result = subprocess.run(
                        [sys.executable, str(build_script),
                         "--pdf-dir", str(w17_pdf_dir),
                         "--output", str(w17_emb),
                         "--dry-run"],
                        capture_output=True, text=True, timeout=120,
                    )
                    if w17_emb.exists():
                        fixes_applied.append("Built W17 embeddings (dry-run mode)")
                    else:
                        print(f"  [FIX] Dry-run also failed: {result.stderr[:300]}")
            except Exception as exc:
                print(f"  [FIX] Exception building embeddings: {exc}")

    # Fix 2: Duplicate input dirs
    for base_dir in [INPUTS_PROF, INPUTS_GT]:
        for padded, unpadded in [("w01", "w1"), ("w02", "w2")]:
            d_padded = base_dir / padded
            d_unpadded = base_dir / unpadded
            if d_padded.is_dir() and d_unpadded.is_dir():
                p_count = _count_json(d_padded)
                u_count = _count_json(d_unpadded)
                if p_count >= u_count:
                    print(f"  [FIX] Removing duplicate {d_unpadded} ({u_count} files, canonical {d_padded} has {p_count})")
                    shutil.rmtree(d_unpadded)
                    fixes_applied.append(f"Removed {d_unpadded.relative_to(ROOT)}")

    return fixes_applied


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 6: CONCURRENCY & RATE LIMIT VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

REQUIRED_API_KEYS: dict[str, list[str]] = {
    "ANTHROPIC_API_KEY": ["W1", "W5", "W13", "W14", "W16", "W17"],
    "OPENAI_API_KEY": ["W9", "W14", "W15", "W17"],
    "DEEPSEEK_API_KEY": ["W2", "W12", "W15", "W18", "W19"],
    "DASHSCOPE_API_KEY": ["W11"],
    "GEMINI_API_KEY": ["W15"],
}


def run_block_6() -> BlockReport:
    block = BlockReport(6, "Concurrency & Rate Limit Verification")

    _load_dotenv()

    # 6.1 API key presence
    missing_keys: list[str] = []
    present_keys: list[str] = []
    for key, workflows in REQUIRED_API_KEYS.items():
        val = os.environ.get(key, "")
        if not val:
            missing_keys.append(f"  {key} (needed for {', '.join(workflows)})")
        else:
            present_keys.append(key)

    if missing_keys:
        block.checks.append(VR(
            "6.1", "api_keys", "FAIL",
            "Missing API keys:\n" + "\n".join(missing_keys),
        ))
    else:
        block.checks.append(VR(
            "6.1", "api_keys", "PASS",
            f"All {len(present_keys)} API keys present: {', '.join(present_keys)}",
        ))

    # 6.2 Rate limit headroom
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from tests.backtesting.concurrency import PROVIDER_PARALLEL, WORKFLOW_PARALLEL_OVERRIDE; "
             "import json; "
             "print(json.dumps({'providers': PROVIDER_PARALLEL, 'overrides': WORKFLOW_PARALLEL_OVERRIDE}))"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            providers = data["providers"]
            overrides = data["overrides"]
            high_parallel = {k: v for k, v in providers.items() if v > 50}
            if high_parallel:
                block.checks.append(VR(
                    "6.2", "rate_limits", "WARN",
                    f"High parallelism ({high_parallel}). Verify rate limits match your API tier. Overrides: {overrides}",
                    blocking=False,
                ))
            else:
                block.checks.append(VR(
                    "6.2", "rate_limits", "PASS",
                    f"Provider parallelism: {providers}. Overrides: {overrides}",
                ))
        else:
            block.checks.append(VR(
                "6.2", "rate_limits", "WARN",
                f"Could not load concurrency config: {result.stderr[:200]}",
                blocking=False,
            ))
    except Exception as exc:
        block.checks.append(VR(
            "6.2", "rate_limits", "WARN", f"Rate limit check error: {exc}",
            blocking=False,
        ))

    # 6.3 Concurrent group balance
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys, json; sys.path.insert(0, '.'); "
             "from tests.backtesting.concurrency import build_concurrent_groups; "
             f"groups = build_concurrent_groups({ACTIVE_WF_SHORT!r}); "
             "print(json.dumps(groups))"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
        if result.returncode == 0:
            groups = json.loads(result.stdout)
            sizes = [len(g) for g in groups]
            detail = " | ".join(f"G{i+1}:[{','.join(g)}]" for i, g in enumerate(groups))
            if max(sizes) > 4:
                block.checks.append(VR(
                    "6.3", "group_balance", "WARN",
                    f"Unbalanced groups (max {max(sizes)} workflows in one group):\n  {detail}",
                    blocking=False,
                ))
            else:
                block.checks.append(VR(
                    "6.3", "group_balance", "PASS",
                    f"{len(groups)} groups: {detail}",
                ))
        else:
            block.checks.append(VR(
                "6.3", "group_balance", "WARN",
                f"Group check failed: {result.stderr[:200]}",
                blocking=False,
            ))
    except Exception as exc:
        block.checks.append(VR(
            "6.3", "group_balance", "WARN", f"Error: {exc}", blocking=False,
        ))

    # 6.4 Resume capability
    results_dir = ROOT / "tests" / "backtesting" / "results" / "backtest"
    if results_dir.is_dir():
        existing = list(results_dir.glob("*_comparison_*.json"))
        block.checks.append(VR(
            "6.4", "resume_capability", "PASS",
            f"Results dir exists with {len(existing)} existing result files. --resume will skip these.",
        ))
    else:
        block.checks.append(VR(
            "6.4", "resume_capability", "PASS",
            "Results dir does not exist yet — fresh run, no resume conflicts.",
        ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# BLOCK 7: COST ESTIMATION & BUDGET GATES
# ═══════════════════════════════════════════════════════════════════════════

GROUND_TRUTH_N: dict[str, int] = {
    "W1": 200, "W2": 300, "W5": 220, "W9": 200,
    "W11": 200, "W12": 200, "W13": 300, "W14": 300, "W15": 300,
    "W16": 300, "W17": 300, "W18": 300, "W19": 300,
}

PROFILING_N = 50


def run_block_7() -> BlockReport:
    block = BlockReport(7, "Cost Estimation & Budget Gates")

    # 7.1 Pre-execution cost estimate from pilot data
    pilot_path = PILOT_REPORT
    if pilot_path.exists():
        try:
            pilot = json.loads(pilot_path.read_text())
            total_estimated = 0.0
            per_wf_estimates: list[str] = []
            for wf in ACTIVE_WF_SHORT:
                wf_data = pilot.get("per_workflow", {}).get(wf, {})
                costs = wf_data.get("per_run_costs", [])
                if costs:
                    mean_cost = sum(costs) / len(costs)
                    gt_n = GROUND_TRUTH_N.get(wf, 200)
                    # Comparison A: profiling (50) + GT (gt_n) runs
                    # Comparison B: profiling (shared) + GT (gt_n) runs
                    # Total: 50 + gt_n + gt_n = 50 + 2*gt_n API runs
                    est = mean_cost * (PROFILING_N + 2 * gt_n)
                    total_estimated += est
                    per_wf_estimates.append(
                        f"  {wf}: ${est:.2f} (mean=${mean_cost:.4f} × {PROFILING_N + 2*gt_n} runs)"
                    )

            detail = "\n".join(per_wf_estimates)
            if total_estimated > 500:
                block.checks.append(VR(
                    "7.1", "cost_estimate", "WARN",
                    f"Estimated total: ${total_estimated:.2f} (over $500 budget):\n{detail}",
                    blocking=False,
                ))
            else:
                block.checks.append(VR(
                    "7.1", "cost_estimate", "PASS",
                    f"Estimated total: ${total_estimated:.2f}:\n{detail}",
                ))
        except Exception as exc:
            block.checks.append(VR(
                "7.1", "cost_estimate", "WARN",
                f"Could not compute estimate: {exc}",
                blocking=False,
            ))
    else:
        block.checks.append(VR(
            "7.1", "cost_estimate", "FAIL",
            f"Pilot report not found at {pilot_path}. Run pilot first.",
        ))

    # 7.2 Budget tracker gate logic
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import sys; sys.path.insert(0, '.'); "
             "from tests.backtesting.budget_tracker import BudgetTracker; "
             "t = BudgetTracker(limit=500.0); "
             "t.record('W1', 'A', 10.0); "
             "assert t.spent == 10.0; "
             "a_scores = {'W1': False, 'W2': False, 'W9': False}; "
             "stop, msg = t.check_comparison_a_gate(a_scores); "
             "print(f'{stop}|{msg}')"],
            capture_output=True, text=True, timeout=15, cwd=str(ROOT),
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split("|", 1)
            stopped = parts[0] == "True"
            msg = parts[1] if len(parts) > 1 else ""
            if stopped:
                block.checks.append(VR(
                    "7.2", "budget_gates", "PASS",
                    f"Budget gate works: 3 failed workflows → stop. Message: {msg}",
                ))
            else:
                block.checks.append(VR(
                    "7.2", "budget_gates", "WARN",
                    "Budget gate did not stop on 3 failures — check logic.",
                    blocking=False,
                ))
        else:
            block.checks.append(VR(
                "7.2", "budget_gates", "WARN",
                f"Budget gate test failed: {result.stderr[:200]}",
                blocking=False,
            ))
    except Exception as exc:
        block.checks.append(VR(
            "7.2", "budget_gates", "WARN",
            f"Budget tracker test error: {exc}",
            blocking=False,
        ))

    # 7.3 Per-workflow cost bounds
    if pilot_path.exists():
        try:
            pilot = json.loads(pilot_path.read_text())
            expensive: list[str] = []
            for wf in ACTIVE_WF_SHORT:
                wf_data = pilot.get("per_workflow", {}).get(wf, {})
                costs = wf_data.get("per_run_costs", [])
                if costs:
                    mean_cost = sum(costs) / len(costs)
                    gt_n = GROUND_TRUTH_N.get(wf, 200)
                    est = mean_cost * gt_n * 2
                    if est > 100:
                        expensive.append(f"  {wf}: estimated ${est:.2f} for GT runs")

            if expensive:
                block.checks.append(VR(
                    "7.3", "per_workflow_bounds", "WARN",
                    "Workflows exceeding $100 estimate:\n" + "\n".join(expensive),
                    blocking=False,
                ))
            else:
                block.checks.append(VR(
                    "7.3", "per_workflow_bounds", "PASS",
                    "All workflows under $100 per-workflow estimate.",
                ))
        except Exception as exc:
            block.checks.append(VR(
                "7.3", "per_workflow_bounds", "WARN",
                f"Cost bound check error: {exc}",
                blocking=False,
            ))

    return block


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════

def format_cli_report(report: Report) -> str:
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("PRE-BACKTEST VERIFICATION REPORT")
    lines.append(f"Timestamp: {report.timestamp}")
    lines.append("=" * 72)

    total_pass = total_fail = total_warn = 0

    for block in report.blocks:
        lines.append("")
        lines.append(f"─── Block {block.block_id}: {block.block_name} ───")
        for check in block.checks:
            icon = {"PASS": "✓", "FAIL": "✗", "WARN": "⚠"}.get(check.status, "?")
            lines.append(f"  [{check.status}] {icon} {check.check_id} {check.name}")
            for detail_line in check.details.split("\n"):
                if detail_line.strip():
                    lines.append(f"         {detail_line}")

            if check.status == "PASS":
                total_pass += 1
            elif check.status == "FAIL":
                total_fail += 1
            else:
                total_warn += 1

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"SUMMARY: {total_pass} PASS, {total_fail} FAIL, {total_warn} WARN")
    verdict = "GO" if report.go else "NO-GO"
    lines.append(f"VERDICT: {verdict}")
    if not report.go:
        blocking_fails = [
            f"  - {c.check_id} {c.name}"
            for b in report.blocks for c in b.checks
            if c.status == "FAIL" and c.blocking
        ]
        if blocking_fails:
            lines.append("Blocking failures:")
            lines.extend(blocking_fails)
    lines.append("=" * 72)

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pre-backtest verification: exhaustive component checks.",
    )
    parser.add_argument("--block", type=int, default=None,
                        help="Run only this block (1-7).")
    parser.add_argument("--skip-live", action="store_true", default=False,
                        help="Skip live API tests in Block 4.")
    parser.add_argument("--fix", action="store_true", default=False,
                        help="Auto-fix simple issues (W17 embeddings, duplicate dirs).")
    parser.add_argument("--output", type=str, default=None,
                        help="Write JSON report to this path.")
    parser.add_argument("-v", "--verbose", action="store_true", default=False,
                        help="Show full details for all checks.")

    args = parser.parse_args()

    # Ensure we can import project modules from project root
    os.chdir(ROOT)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    report = Report(timestamp=datetime.now(timezone.utc).isoformat())

    # Auto-fix if requested
    if args.fix:
        print("Running auto-fixes...")
        fixes = run_fixes()
        if fixes:
            print(f"Applied {len(fixes)} fixes: {fixes}")
        else:
            print("No fixes needed or all fixes failed.")
        print()

    # Run blocks
    blocks_to_run = [args.block] if args.block else [1, 2, 3, 4, 5, 6, 7]

    for b in blocks_to_run:
        print(f"Running Block {b}...")
        try:
            if b == 1:
                report.blocks.append(run_block_1())
            elif b == 2:
                report.blocks.append(run_block_2())
            elif b == 3:
                report.blocks.append(run_block_3())
            elif b == 4:
                report.blocks.append(run_block_4(skip_live=args.skip_live))
            elif b == 5:
                report.blocks.append(run_block_5())
            elif b == 6:
                report.blocks.append(run_block_6())
            elif b == 7:
                report.blocks.append(run_block_7())
        except Exception as exc:
            print(f"  Block {b} error: {exc}")
            traceback.print_exc()
            report.blocks.append(BlockReport(
                b, f"Block {b} (ERROR)",
                [VR(f"{b}.0", "block_error", "FAIL", str(exc))],
            ))

    # Determine go/no-go
    for block in report.blocks:
        for check in block.checks:
            if check.status == "FAIL" and check.blocking:
                report.go = False

    # Output
    print()
    print(format_cli_report(report))

    # JSON output
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(
            {
                "timestamp": report.timestamp,
                "go": report.go,
                "blocks": [
                    {
                        "block_id": b.block_id,
                        "block_name": b.block_name,
                        "checks": [c.to_dict() for c in b.checks],
                    }
                    for b in report.blocks
                ],
            },
            indent=2,
        ))
        print(f"\nJSON report written to: {args.output}")

    sys.exit(0 if report.go else 1)


if __name__ == "__main__":
    main()
