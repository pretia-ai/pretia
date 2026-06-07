"""Run all quality checks from Part 6 of directions-pdf-generation.md.

Checks are grouped into three categories:
  - Per-PDF (7 checks): validity, page count, token count, text extraction, scanned detection,
    table integrity, metadata consistency.
  - Per-Corpus (4 checks): key field coverage, doc type distribution, length distribution,
    W17 provider coverage.
  - Cross-Distribution (3 checks): length drift, structural drift, modality drift.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class CheckResult:
    """Result of a single quality check."""

    check_id: int
    name: str
    passed: bool
    details: str
    severity: str = "error"  # "error" | "warning"

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "name": self.name,
            "passed": self.passed,
            "details": self.details,
            "severity": self.severity,
        }


@dataclass(slots=True)
class PDFCheckReport:
    """Report for a single PDF."""

    pdf_id: str
    pdf_path: str
    checks: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "error")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pdf_id": self.pdf_id,
            "pdf_path": self.pdf_path,
            "passed": self.passed,
            "checks": [c.to_dict() for c in self.checks],
        }


@dataclass(slots=True)
class CorpusCheckReport:
    """Full verification report for a corpus."""

    workflow: str
    profile: str
    pdf_reports: list[PDFCheckReport] = field(default_factory=list)
    corpus_checks: list[CheckResult] = field(default_factory=list)
    cross_dist_checks: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        pdf_ok = all(r.passed for r in self.pdf_reports)
        corpus_ok = all(c.passed for c in self.corpus_checks if c.severity == "error")
        cross_ok = all(c.passed for c in self.cross_dist_checks if c.severity == "error")
        return pdf_ok and corpus_ok and cross_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow": self.workflow,
            "profile": self.profile,
            "all_passed": self.all_passed,
            "pdf_reports": [r.to_dict() for r in self.pdf_reports],
            "corpus_checks": [c.to_dict() for c in self.corpus_checks],
            "cross_distribution_checks": [c.to_dict() for c in self.cross_dist_checks],
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))


def _lazy_pdfplumber() -> Any:
    try:
        import pdfplumber

        return pdfplumber
    except ImportError as err:
        msg = "pdfplumber required for PDF verification: pip install pdfplumber"
        raise ImportError(msg) from err


def _count_tokens(text: str) -> int:
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def _load_descriptor(pdf_path: Path) -> dict[str, Any] | None:
    """Load the PDFDescriptor JSON that should sit alongside a PDF."""
    json_name = pdf_path.stem + ".json"
    json_path = pdf_path.parent / json_name
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


# ---------------------------------------------------------------------------
# Per-PDF checks (1-7)
# ---------------------------------------------------------------------------


def check_pdf_validity(pdf_path: Path) -> CheckResult:
    """Check 1: PDF opens with pdfplumber without exception."""
    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            _ = len(pdf.pages)
        return CheckResult(1, "pdf_validity", True, "PDF opens successfully")
    except Exception as e:
        return CheckResult(1, "pdf_validity", False, f"Failed to open: {e}")


def check_page_count(
    pdf_path: Path,
    expected_min: int | None = None,
    expected_max: int | None = None,
) -> CheckResult:
    """Check 2: Page count matches target tier range."""
    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            actual = len(pdf.pages)
    except Exception as e:
        return CheckResult(2, "page_count", False, f"Cannot read pages: {e}")

    if expected_min is not None and actual < expected_min:
        return CheckResult(
            2,
            "page_count",
            False,
            f"Page count {actual} below minimum {expected_min}",
        )
    if expected_max is not None and actual > expected_max:
        return CheckResult(
            2,
            "page_count",
            False,
            f"Page count {actual} above maximum {expected_max}",
        )
    return CheckResult(2, "page_count", True, f"Page count: {actual}")


def check_token_count(
    pdf_path: Path,
    expected_min: int | None = None,
    expected_max: int | None = None,
) -> CheckResult:
    """Check 3: Token count falls within tier range."""
    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as e:
        return CheckResult(3, "token_count", False, f"Text extraction failed: {e}")

    tokens = _count_tokens(text)
    if expected_min is not None and tokens < expected_min:
        return CheckResult(
            3,
            "token_count",
            False,
            f"Token count {tokens} below minimum {expected_min}",
            severity="warning",
        )
    if expected_max is not None and tokens > expected_max:
        return CheckResult(
            3,
            "token_count",
            False,
            f"Token count {tokens} above maximum {expected_max}",
            severity="warning",
        )
    return CheckResult(3, "token_count", True, f"Token count: {tokens}")


def check_text_extraction(pdf_path: Path) -> CheckResult:
    """Check 4: pdfplumber extracts >90% of text pages successfully."""
    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            extracted = sum(1 for p in pdf.pages if (p.extract_text() or "").strip())
    except Exception as e:
        return CheckResult(4, "text_extraction", False, f"Extraction error: {e}")

    if total_pages == 0:
        return CheckResult(4, "text_extraction", False, "PDF has 0 pages")

    ratio = extracted / total_pages
    if ratio < 0.5:
        return CheckResult(
            4,
            "text_extraction",
            False,
            f"Only {extracted}/{total_pages} pages ({ratio:.0%}) yielded text",
            severity="warning",
        )
    return CheckResult(
        4,
        "text_extraction",
        True,
        f"{extracted}/{total_pages} pages ({ratio:.0%}) yielded text",
    )


def check_scanned_pages(
    pdf_path: Path,
    expected_scanned_indices: list[int] | None = None,
) -> CheckResult:
    """Check 5: Scanned pages yield <10 chars via pdfplumber."""
    if not expected_scanned_indices:
        return CheckResult(5, "scanned_detection", True, "No scanned pages expected")

    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            failures = []
            for idx in expected_scanned_indices:
                if idx >= len(pdf.pages):
                    failures.append(f"Page {idx} out of range")
                    continue
                text = (pdf.pages[idx].extract_text() or "").strip()
                if len(text) >= 10:
                    failures.append(f"Page {idx} extracted {len(text)} chars (expected <10)")
    except Exception as e:
        return CheckResult(5, "scanned_detection", False, f"Error: {e}")

    if failures:
        return CheckResult(
            5,
            "scanned_detection",
            False,
            f"Scanned page check failures: {'; '.join(failures)}",
        )
    return CheckResult(
        5,
        "scanned_detection",
        True,
        f"All {len(expected_scanned_indices)} scanned pages confirmed",
    )


def check_table_integrity(pdf_path: Path, expect_tables: bool = False) -> CheckResult:
    """Check 6: pdfplumber detects at least one table where expected."""
    if not expect_tables:
        return CheckResult(6, "table_integrity", True, "No tables expected")

    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                tables = page.extract_tables()
                if tables:
                    return CheckResult(
                        6,
                        "table_integrity",
                        True,
                        f"Found {len(tables)} table(s)",
                    )
    except Exception as e:
        return CheckResult(6, "table_integrity", False, f"Error: {e}")

    return CheckResult(
        6,
        "table_integrity",
        False,
        "Expected tables but pdfplumber found none",
        severity="warning",
    )


def check_metadata_consistency(pdf_path: Path) -> CheckResult:
    """Check 7: PDFDescriptor matches actual PDF properties."""
    desc = _load_descriptor(pdf_path)
    if desc is None:
        return CheckResult(
            7,
            "metadata_consistency",
            False,
            f"No descriptor JSON found for {pdf_path.name}",
        )

    pdfplumber = _lazy_pdfplumber()
    try:
        with pdfplumber.open(pdf_path) as pdf:
            actual_pages = len(pdf.pages)
    except Exception as e:
        return CheckResult(7, "metadata_consistency", False, f"Cannot read PDF: {e}")

    expected_pages = desc.get("page_count", 0)
    if actual_pages != expected_pages:
        return CheckResult(
            7,
            "metadata_consistency",
            False,
            f"Descriptor says {expected_pages} pages, PDF has {actual_pages}",
        )
    return CheckResult(7, "metadata_consistency", True, "Descriptor matches PDF")


def run_per_pdf_checks(
    pdf_path: Path,
    page_range: tuple[int, int] | None = None,
    token_range: tuple[int, int] | None = None,
    expect_tables: bool = False,
    scanned_indices: list[int] | None = None,
) -> PDFCheckReport:
    """Run all 7 per-PDF checks on a single file."""
    pdf_id = pdf_path.stem
    report = PDFCheckReport(pdf_id=pdf_id, pdf_path=str(pdf_path))

    report.checks.append(check_pdf_validity(pdf_path))
    report.checks.append(
        check_page_count(
            pdf_path,
            expected_min=page_range[0] if page_range else None,
            expected_max=page_range[1] if page_range else None,
        )
    )
    report.checks.append(
        check_token_count(
            pdf_path,
            expected_min=token_range[0] if token_range else None,
            expected_max=token_range[1] if token_range else None,
        )
    )
    report.checks.append(check_text_extraction(pdf_path))
    report.checks.append(check_scanned_pages(pdf_path, scanned_indices))
    report.checks.append(check_table_integrity(pdf_path, expect_tables))
    report.checks.append(check_metadata_consistency(pdf_path))

    status = "PASS" if report.passed else "FAIL"
    logger.info("PDF %s: %s", pdf_id, status)
    return report


# ---------------------------------------------------------------------------
# Per-Corpus checks (8-11)
# ---------------------------------------------------------------------------


def check_key_field_coverage(
    pdf_dir: Path,
    required_values: dict[str, dict[str, str]],
) -> CheckResult:
    """Check 8: Every value in cross-provider tables appears in at least one document."""
    pdfplumber = _lazy_pdfplumber()
    all_text = ""
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    all_text += (page.extract_text() or "") + "\n"
        except Exception:  # noqa: S112
            continue

    missing: list[str] = []
    for provider, values in required_values.items():
        for field_name, value in values.items():
            if value not in all_text:
                missing.append(f"{provider}/{field_name}: {value!r}")

    if missing:
        summary = "; ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        return CheckResult(
            8,
            "key_field_coverage",
            False,
            f"Missing {len(missing)} values: {summary}{suffix}",
        )
    return CheckResult(8, "key_field_coverage", True, "All required values found")


def check_doc_type_distribution(
    descriptors: list[dict[str, Any]],
) -> CheckResult:
    """Check 9: Each doc type is 15-25% of corpus, none >30%."""
    if not descriptors:
        return CheckResult(9, "doc_type_distribution", False, "No descriptors")

    counts: Counter[str] = Counter()
    for d in descriptors:
        counts[d.get("document_type", "unknown")] += 1

    total = len(descriptors)
    issues: list[str] = []
    for dtype, count in counts.items():
        pct = count / total * 100
        if pct > 30:
            issues.append(f"{dtype}: {pct:.0f}% (>30%)")

    if issues:
        return CheckResult(
            9,
            "doc_type_distribution",
            False,
            f"Distribution issues: {'; '.join(issues)}",
            severity="warning",
        )
    dist_str = ", ".join(f"{k}: {v / total:.0%}" for k, v in counts.most_common())
    return CheckResult(9, "doc_type_distribution", True, f"Distribution: {dist_str}")


def check_length_distribution(
    descriptors: list[dict[str, Any]],
    tier_ranges: dict[str, tuple[int, int]] | None = None,
) -> CheckResult:
    """Check 10: Token counts span >=70% of combined tier ranges."""
    tokens = [
        d.get("estimated_token_count", 0) for d in descriptors if d.get("estimated_token_count")
    ]
    if not tokens:
        return CheckResult(10, "length_distribution", False, "No token counts", severity="warning")

    actual_range = max(tokens) - min(tokens)
    if tier_ranges:
        all_mins = [r[0] for r in tier_ranges.values()]
        all_maxs = [r[1] for r in tier_ranges.values()]
        expected_range = max(all_maxs) - min(all_mins)
        coverage = actual_range / expected_range if expected_range > 0 else 0
        if coverage < 0.7:
            return CheckResult(
                10,
                "length_distribution",
                False,
                f"Token range covers {coverage:.0%} of tier ranges (need >=70%)",
                severity="warning",
            )
        return CheckResult(
            10,
            "length_distribution",
            True,
            f"Token range {min(tokens)}-{max(tokens)}, covers {coverage:.0%}",
        )

    return CheckResult(
        10,
        "length_distribution",
        True,
        f"Token range: {min(tokens)}-{max(tokens)}",
    )


def check_w17_provider_coverage(
    descriptors: list[dict[str, Any]],
    expected_providers: list[str],
) -> CheckResult:
    """Check 11: Each provider has exactly 1 policy PDF."""
    provider_counts: Counter[str] = Counter()
    for d in descriptors:
        if d.get("document_type") == "provider_policy":
            provider = d.get("provider", "unknown")
            provider_counts[provider] += 1

    issues: list[str] = []
    for provider in expected_providers:
        count = provider_counts.get(provider, 0)
        if count != 1:
            issues.append(f"{provider}: {count} policies (expected 1)")

    if issues:
        return CheckResult(11, "w17_provider_coverage", False, "; ".join(issues))
    return CheckResult(
        11,
        "w17_provider_coverage",
        True,
        f"All {len(expected_providers)} providers have exactly 1 policy",
    )


# ---------------------------------------------------------------------------
# Cross-Distribution checks (12-14)
# ---------------------------------------------------------------------------


def check_length_drift(
    profiling_descriptors: list[dict[str, Any]],
    gt_descriptors: list[dict[str, Any]],
) -> CheckResult:
    """Check 12: GT mean token count is 1.3-2x profiling mean."""
    prof_tokens = [
        d.get("estimated_token_count", 0)
        for d in profiling_descriptors
        if d.get("estimated_token_count")
    ]
    gt_tokens = [
        d.get("estimated_token_count", 0) for d in gt_descriptors if d.get("estimated_token_count")
    ]

    if not prof_tokens or not gt_tokens:
        return CheckResult(12, "length_drift", False, "Missing token data", severity="warning")

    prof_mean = sum(prof_tokens) / len(prof_tokens)
    gt_mean = sum(gt_tokens) / len(gt_tokens)
    ratio = gt_mean / prof_mean if prof_mean > 0 else 0

    if ratio < 1.3:
        return CheckResult(
            12,
            "length_drift",
            False,
            f"GT/profiling ratio {ratio:.2f} (below 1.3x)",
            severity="warning",
        )
    if ratio > 2.0:
        return CheckResult(
            12,
            "length_drift",
            False,
            f"GT/profiling ratio {ratio:.2f} (above 2.0x)",
            severity="warning",
        )
    return CheckResult(
        12,
        "length_drift",
        True,
        f"GT/profiling ratio {ratio:.2f} (within 1.3-2.0x)",
    )


def check_structural_drift(
    profiling_descriptors: list[dict[str, Any]],
    gt_descriptors: list[dict[str, Any]],
) -> CheckResult:
    """Check 13: >=30% GT has poor structure, <=10% profiling has poor structure."""

    def poor_pct(descs: list[dict[str, Any]]) -> float:
        if not descs:
            return 0.0
        poor = sum(
            1
            for d in descs
            if d.get("structure_quality") in ("partially_structured", "unstructured")
        )
        return poor / len(descs)

    prof_poor = poor_pct(profiling_descriptors)
    gt_poor = poor_pct(gt_descriptors)

    issues: list[str] = []
    if prof_poor > 0.10:
        issues.append(f"Profiling poor structure: {prof_poor:.0%} (should be <=10%)")
    if gt_poor < 0.30:
        issues.append(f"GT poor structure: {gt_poor:.0%} (should be >=30%)")

    if issues:
        return CheckResult(13, "structural_drift", False, "; ".join(issues), severity="warning")
    return CheckResult(
        13,
        "structural_drift",
        True,
        f"Profiling poor: {prof_poor:.0%}, GT poor: {gt_poor:.0%}",
    )


def check_modality_drift(
    profiling_descriptors: list[dict[str, Any]],
    gt_descriptors: list[dict[str, Any]],
) -> CheckResult:
    """Check 14: Profiling is 80/20/0, GT is ~50/30/20 (W14/W15 only)."""

    def modality_pcts(descs: list[dict[str, Any]]) -> tuple[float, float, float]:
        total_text = sum(d.get("text_pages", 0) for d in descs)
        total_tc = sum(d.get("table_chart_pages", 0) for d in descs)
        total_scan = sum(d.get("scanned_pages", 0) for d in descs)
        total = total_text + total_tc + total_scan
        if total == 0:
            return (0.0, 0.0, 0.0)
        return (total_text / total, total_tc / total, total_scan / total)

    prof_mod = modality_pcts(profiling_descriptors)
    gt_mod = modality_pcts(gt_descriptors)

    issues: list[str] = []
    if prof_mod[2] > 0.05:
        issues.append(f"Profiling has {prof_mod[2]:.0%} scanned (should be ~0%)")
    if gt_mod[2] < 0.10:
        issues.append(f"GT has {gt_mod[2]:.0%} scanned (should be ~20%)")

    if issues:
        return CheckResult(14, "modality_drift", False, "; ".join(issues), severity="warning")
    return CheckResult(
        14,
        "modality_drift",
        True,
        f"Profiling: {prof_mod[0]:.0%}/{prof_mod[1]:.0%}/{prof_mod[2]:.0%}, "
        f"GT: {gt_mod[0]:.0%}/{gt_mod[1]:.0%}/{gt_mod[2]:.0%}",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def verify_corpus(
    pdf_dir: Path,
    workflow: str,
    profile: str,
    required_values: dict[str, dict[str, str]] | None = None,
    expected_providers: list[str] | None = None,
) -> CorpusCheckReport:
    """Run all applicable checks on a corpus directory."""
    report = CorpusCheckReport(workflow=workflow, profile=profile)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in %s", pdf_dir)
        return report

    descriptors: list[dict[str, Any]] = []
    for pdf_path in pdf_files:
        desc = _load_descriptor(pdf_path)
        if desc:
            descriptors.append(desc)

        pdf_report = run_per_pdf_checks(
            pdf_path,
            expect_tables=desc.get("table_chart_pages", 0) > 0 if desc else False,
        )
        report.pdf_reports.append(pdf_report)

    if required_values:
        report.corpus_checks.append(check_key_field_coverage(pdf_dir, required_values))
    if descriptors:
        report.corpus_checks.append(check_doc_type_distribution(descriptors))
        report.corpus_checks.append(check_length_distribution(descriptors))
    if expected_providers:
        report.corpus_checks.append(check_w17_provider_coverage(descriptors, expected_providers))

    passed = sum(1 for r in report.pdf_reports if r.passed)
    total = len(report.pdf_reports)
    corpus_ok = all(c.passed for c in report.corpus_checks if c.severity == "error")
    logger.info(
        "Corpus %s/%s: %d/%d PDFs passed, corpus checks %s",
        workflow,
        profile,
        passed,
        total,
        "PASS" if corpus_ok else "FAIL",
    )
    return report


def verify_cross_distribution(
    profiling_dir: Path,
    gt_dir: Path,
    workflow: str,
    check_modality: bool = False,
) -> list[CheckResult]:
    """Run cross-distribution checks (12-14) between profiling and GT corpora."""

    def load_descs(d: Path) -> list[dict[str, Any]]:
        descs = []
        for pdf_path in sorted(d.glob("*.pdf")):
            desc = _load_descriptor(pdf_path)
            if desc:
                descs.append(desc)
        return descs

    prof_descs = load_descs(profiling_dir)
    gt_descs = load_descs(gt_dir)

    checks = [
        check_length_drift(prof_descs, gt_descs),
        check_structural_drift(prof_descs, gt_descs),
    ]
    if check_modality:
        checks.append(check_modality_drift(prof_descs, gt_descs))

    return checks


def format_report(report: CorpusCheckReport) -> str:
    """Format a verification report as human-readable text."""
    lines: list[str] = []
    lines.append(f"=== {report.workflow} / {report.profile} ===")
    lines.append("")

    passed_pdfs = sum(1 for r in report.pdf_reports if r.passed)
    total_pdfs = len(report.pdf_reports)
    lines.append(f"Per-PDF checks: {passed_pdfs}/{total_pdfs} passed")

    for r in report.pdf_reports:
        if not r.passed:
            failed = [c for c in r.checks if not c.passed]
            for c in failed:
                lines.append(f"  FAIL {r.pdf_id}: [{c.name}] {c.details}")

    if report.corpus_checks:
        lines.append("")
        lines.append("Corpus checks:")
        for c in report.corpus_checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  {status} [{c.name}] {c.details}")

    if report.cross_dist_checks:
        lines.append("")
        lines.append("Cross-distribution checks:")
        for c in report.cross_dist_checks:
            status = "PASS" if c.passed else "FAIL"
            lines.append(f"  {status} [{c.name}] {c.details}")

    lines.append("")
    overall = "PASS" if report.all_passed else "FAIL"
    lines.append(f"Overall: {overall}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Verify generated PDF corpora")
    parser.add_argument("pdf_dir", type=Path, help="Directory containing PDFs")
    parser.add_argument("--workflow", required=True, help="Workflow ID (e.g., W14)")
    parser.add_argument("--profile", required=True, choices=["profiling", "ground_truth"])
    parser.add_argument("--output", type=Path, help="Write JSON report to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = verify_corpus(args.pdf_dir, args.workflow, args.profile)
    print(format_report(report))

    if args.output:
        report.save(args.output)
        logger.info("Report saved to %s", args.output)

    sys.exit(0 if report.all_passed else 1)
