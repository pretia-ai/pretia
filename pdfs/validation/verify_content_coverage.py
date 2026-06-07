"""Verify key field coverage for W14/W15/W17 corpora.

Searches generated PDF text for the exact cross-provider values specified in
the directions-pdf-generation.md spec. These values are load-bearing — the
input generators create queries and claims targeting them.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from pdfs.generators._types import W14_W15_PROVIDER_VALUES, W17_PROVIDER_VALUES

logger = logging.getLogger(__name__)


def _lazy_pdfplumber() -> Any:
    try:
        import pdfplumber

        return pdfplumber
    except ImportError as err:
        raise ImportError("pdfplumber required: pip install pdfplumber") from err


def extract_corpus_text(pdf_dir: Path) -> str:
    """Extract all text from all PDFs in a directory."""
    pdfplumber = _lazy_pdfplumber()
    texts: list[str] = []
    for pdf_file in sorted(pdf_dir.glob("*.pdf")):
        try:
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    texts.append(page.extract_text() or "")
        except Exception as e:
            logger.warning("Failed to read %s: %s", pdf_file.name, e)
    return "\n".join(texts)


def verify_w14_w15_coverage(pdf_dir: Path) -> dict[str, Any]:
    """Verify all W14/W15 cross-provider values appear in the corpus."""
    corpus_text = extract_corpus_text(pdf_dir)

    results: dict[str, dict[str, bool]] = {}
    missing_count = 0
    found_count = 0

    for provider, values in W14_W15_PROVIDER_VALUES.items():
        results[provider] = {}
        for field_name, value in values.items():
            found = value in corpus_text
            results[provider][field_name] = found
            if found:
                found_count += 1
            else:
                missing_count += 1
                logger.warning(
                    "MISSING: %s / %s = %r",
                    provider,
                    field_name,
                    value,
                )

    total = found_count + missing_count
    return {
        "workflow": "W14/W15",
        "total_values": total,
        "found": found_count,
        "missing": missing_count,
        "coverage": found_count / total if total > 0 else 0,
        "all_found": missing_count == 0,
        "details": results,
    }


def verify_w17_coverage(pdf_dir: Path) -> dict[str, Any]:
    """Verify all W17 cross-provider policy values appear in the corpus."""
    corpus_text = extract_corpus_text(pdf_dir)

    results: dict[str, dict[str, bool]] = {}
    missing_count = 0
    found_count = 0

    for provider, values in W17_PROVIDER_VALUES.items():
        results[provider] = {}
        for field_name, value in values.items():
            found = value in corpus_text
            results[provider][field_name] = found
            if found:
                found_count += 1
            else:
                missing_count += 1
                logger.warning(
                    "MISSING: %s / %s = %r",
                    provider,
                    field_name,
                    value,
                )

    total = found_count + missing_count
    return {
        "workflow": "W17",
        "total_values": total,
        "found": found_count,
        "missing": missing_count,
        "coverage": found_count / total if total > 0 else 0,
        "all_found": missing_count == 0,
        "details": results,
    }


def format_coverage_report(result: dict[str, Any]) -> str:
    """Format a coverage check result as human-readable text."""
    lines: list[str] = []
    lines.append(f"=== {result['workflow']} Key Field Coverage ===")
    lines.append(
        f"Coverage: {result['found']}/{result['total_values']} ({result['coverage']:.0%})"
    )

    if result["missing"] > 0:
        lines.append(f"\nMissing {result['missing']} values:")
        for provider, fields in result["details"].items():
            for field_name, found in fields.items():
                if not found:
                    lines.append(f"  {provider} / {field_name}")

    status = "PASS" if result["all_found"] else "FAIL"
    lines.append(f"\nResult: {status}")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    import json
    import sys

    parser = argparse.ArgumentParser(description="Verify key field coverage in PDF corpora")
    parser.add_argument("pdf_dir", type=Path, help="Directory containing PDFs")
    parser.add_argument(
        "--workflow",
        required=True,
        choices=["W14", "W15", "W17"],
        help="Which workflow's values to check",
    )
    parser.add_argument("--output", type=Path, help="Write JSON result to this path")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.workflow in ("W14", "W15"):
        result = verify_w14_w15_coverage(args.pdf_dir)
    else:
        result = verify_w17_coverage(args.pdf_dir)

    print(format_coverage_report(result))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2))
        logger.info("Result saved to %s", args.output)

    sys.exit(0 if result["all_found"] else 1)
