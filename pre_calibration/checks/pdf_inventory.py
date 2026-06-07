"""Verify PDF corpus exists for document-processing workflows."""

from __future__ import annotations

from pathlib import Path

from pre_calibration.pre_calibration import CheckResult

PDF_WORKFLOWS = ["w14", "w15", "w16", "w17", "w18"]


async def check(*, pdfs_dir: Path = Path("pdfs/generated")) -> CheckResult:
    """Verify PDF files exist for document-processing workflows."""
    details: dict = {}
    missing = []

    if not pdfs_dir.is_dir():
        return CheckResult(
            name="pdf_inventory",
            status="FAIL",
            details={"error": f"PDFs directory not found: {pdfs_dir}"},
            blocking=True,
        )

    total_pdfs = 0
    for wf in PDF_WORKFLOWS:
        wf_dir = pdfs_dir / wf
        if not wf_dir.is_dir():
            # Try finding PDFs at the root level
            pdfs = list(pdfs_dir.glob(f"{wf}*.pdf"))
            if not pdfs:
                missing.append(wf)
            else:
                total_pdfs += len(pdfs)
        else:
            pdfs = list(wf_dir.glob("*.pdf"))
            if not pdfs:
                missing.append(wf)
            total_pdfs += len(pdfs)

    details["pdf_count"] = total_pdfs
    if missing:
        details["missing_workflows"] = missing

    status = "FAIL" if missing else "PASS"
    return CheckResult(
        name="pdf_inventory",
        status=status,
        details=details,
        blocking=True,
    )
