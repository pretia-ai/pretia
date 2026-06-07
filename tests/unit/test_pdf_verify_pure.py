"""Tests for pure-Python parts of verify_pdfs and verify_content_coverage (no pdfplumber)."""

from __future__ import annotations

import json

from pdfs.validation.verify_content_coverage import format_coverage_report
from pdfs.validation.verify_pdfs import (
    CheckResult,
    CorpusCheckReport,
    PDFCheckReport,
    format_report,
)


class TestCheckResultToDict:
    def test_round_trip(self):
        cr = CheckResult(1, "pdf_validity", True, "OK")
        d = cr.to_dict()
        assert d["check_id"] == 1
        assert d["name"] == "pdf_validity"
        assert d["passed"] is True
        assert d["severity"] == "error"

    def test_warning_severity(self):
        cr = CheckResult(3, "token_count", False, "Low", severity="warning")
        assert cr.to_dict()["severity"] == "warning"


class TestPDFCheckReportPassed:
    def test_all_pass(self):
        report = PDFCheckReport(
            pdf_id="test",
            pdf_path="/tmp/test.pdf",
            checks=[
                CheckResult(1, "a", True, "ok"),
                CheckResult(2, "b", True, "ok"),
            ],
        )
        assert report.passed is True

    def test_error_failure(self):
        report = PDFCheckReport(
            pdf_id="test",
            pdf_path="/tmp/test.pdf",
            checks=[
                CheckResult(1, "a", True, "ok"),
                CheckResult(2, "b", False, "bad", severity="error"),
            ],
        )
        assert report.passed is False

    def test_warning_does_not_fail(self):
        report = PDFCheckReport(
            pdf_id="test",
            pdf_path="/tmp/test.pdf",
            checks=[
                CheckResult(1, "a", True, "ok"),
                CheckResult(2, "b", False, "warn", severity="warning"),
            ],
        )
        assert report.passed is True


class TestCorpusCheckReportAllPassed:
    def test_all_pass(self):
        report = CorpusCheckReport(
            workflow="W17",
            profile="profiling",
            pdf_reports=[
                PDFCheckReport("a", "/a.pdf", [CheckResult(1, "x", True, "ok")]),
            ],
            corpus_checks=[CheckResult(8, "coverage", True, "ok")],
        )
        assert report.all_passed is True

    def test_pdf_failure_fails_overall(self):
        report = CorpusCheckReport(
            workflow="W17",
            profile="profiling",
            pdf_reports=[
                PDFCheckReport("a", "/a.pdf", [CheckResult(1, "x", False, "bad")]),
            ],
        )
        assert report.all_passed is False

    def test_corpus_check_failure(self):
        report = CorpusCheckReport(
            workflow="W14",
            profile="profiling",
            pdf_reports=[
                PDFCheckReport("a", "/a.pdf", [CheckResult(1, "x", True, "ok")]),
            ],
            corpus_checks=[CheckResult(8, "coverage", False, "missing")],
        )
        assert report.all_passed is False


class TestCorpusCheckReportSave:
    def test_save_and_reload(self, tmp_path):
        report = CorpusCheckReport(
            workflow="W16",
            profile="ground_truth",
            pdf_reports=[
                PDFCheckReport("a", "/a.pdf", [CheckResult(1, "x", True, "ok")]),
            ],
            corpus_checks=[],
        )
        path = tmp_path / "report.json"
        report.save(path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["workflow"] == "W16"
        assert data["all_passed"] is True
        assert len(data["pdf_reports"]) == 1


class TestFormatReport:
    def test_contains_workflow(self):
        report = CorpusCheckReport(
            workflow="W17",
            profile="profiling",
            pdf_reports=[],
            corpus_checks=[],
        )
        text = format_report(report)
        assert "W17" in text
        assert "profiling" in text

    def test_shows_failed_checks(self):
        report = CorpusCheckReport(
            workflow="W14",
            profile="profiling",
            pdf_reports=[
                PDFCheckReport(
                    "bad",
                    "/bad.pdf",
                    [CheckResult(1, "validity", False, "corrupt file")],
                ),
            ],
        )
        text = format_report(report)
        assert "FAIL" in text
        assert "corrupt file" in text


class TestFormatCoverageReport:
    def test_all_found(self):
        result = {
            "workflow": "W17",
            "total_values": 21,
            "found": 21,
            "missing": 0,
            "coverage": 1.0,
            "all_found": True,
            "details": {},
        }
        text = format_coverage_report(result)
        assert "PASS" in text
        assert "100%" in text

    def test_missing_values(self):
        result = {
            "workflow": "W14/W15",
            "total_values": 24,
            "found": 20,
            "missing": 4,
            "coverage": 0.833,
            "all_found": False,
            "details": {
                "Aetna": {"deductible": False, "copay": True},
            },
        }
        text = format_coverage_report(result)
        assert "FAIL" in text
        assert "4" in text or "Missing" in text
