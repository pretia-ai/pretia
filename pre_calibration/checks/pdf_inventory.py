"""Verify PDF corpus exists for document-processing workflows."""

from __future__ import annotations

from pathlib import Path

from pre_calibration.pre_calibration import CheckResult

PDF_WORKFLOWS = ["w14", "w15", "w16", "w17", "w18"]

# Some workflows share a PDF corpus under a combined directory name.
_SHARED_CORPUS_DIRS: dict[str, list[str]] = {
    "w14": ["w14_w15_corpus"],
    "w15": ["w14_w15_corpus", "w15_supplement"],
}


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
        found = False
        # Check direct subdirectory (pdfs/generated/w14/)
        wf_dir = pdfs_dir / wf
        if wf_dir.is_dir():
            pdfs = list(wf_dir.rglob("*.pdf"))
            if pdfs:
                total_pdfs += len(pdfs)
                found = True
        # Check profiling/ground_truth subdirectories (pdfs/generated/profiling/w14/)
        for profile in ("profiling", "ground_truth"):
            profile_dir = pdfs_dir / profile / wf
            if profile_dir.is_dir():
                pdfs = list(profile_dir.rglob("*.pdf"))
                if pdfs:
                    total_pdfs += len(pdfs)
                    found = True
        # Check root-level files (pdfs/generated/w14-*.pdf)
        if not found:
            pdfs = list(pdfs_dir.glob(f"{wf}*.pdf"))
            if pdfs:
                total_pdfs += len(pdfs)
                found = True
            # Also check inside profiling/ground_truth with glob
            for profile in ("profiling", "ground_truth"):
                pdfs = list((pdfs_dir / profile).glob(f"{wf}*.pdf"))
                if pdfs:
                    total_pdfs += len(pdfs)
                    found = True
        # Check shared corpus directories (e.g., w14_w15_corpus for w14/w15)
        if not found:
            for shared_name in _SHARED_CORPUS_DIRS.get(wf, []):
                for base in [pdfs_dir, pdfs_dir / "profiling", pdfs_dir / "ground_truth"]:
                    shared_dir = base / shared_name
                    if shared_dir.is_dir():
                        pdfs = list(shared_dir.rglob("*.pdf"))
                        if pdfs:
                            total_pdfs += len(pdfs)
                            found = True
                            break
                if found:
                    break
        if not found:
            missing.append(wf)

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
