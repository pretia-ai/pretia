"""Generate long-document corpus for W18 single-pass processing backtesting.

Produce 50 (profiling) or 500 (ground truth) PDFs across 4 document types
and 4-5 difficulty tiers.  Tiers are TOKEN-based (30K-100K), not page-based.
Each PDF is one backtesting run input -- the entire document is processed in
a single LLM pass, so total token count is the primary cost driver.

Model: deepseek-v4-pro for all documents.  All docs use sectional generation
(30-100 pages each).

Content density:
- Profiling: consistent ~800 tokens/page.
- Ground truth: variable 500-1200 tokens/page.

Modality: text-only (no scanned pages, no charts).  Tables rendered as text.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from pathlib import Path
from typing import Any

from pdfs.generators._llm import count_tokens, generate_document_content, run_concurrent
from pdfs.generators._types import W18_TIERS, DocumentSpec, SectionSpec
from pdfs.generators.rendering.pdf_assembler import PDFDescriptor, write_descriptor
from pdfs.generators.rendering.text_renderer import PageLayout, render_markdown_to_pdf

logger = logging.getLogger(__name__)

_MODEL = "deepseek-v4-pro"

# Consistent density for profiling, variable for ground truth.
_PROFILING_DENSITY = 800  # tokens per page
_GT_DENSITY_RANGE = (500, 1200)  # tokens per page (min, max)

# ---------------------------------------------------------------------------
# Document-type templates
# ---------------------------------------------------------------------------

_DOC_TYPE_DEFS: dict[str, dict[str, Any]] = {
    "annual_report": {
        "domain": "corporate_finance",
        "page_range": (30, 60),
        "section_templates": [
            ("Letter to Shareholders", "text"),
            ("Company Overview", "text"),
            ("Financial Highlights", "mixed"),
            ("Business Segments", "text"),
            ("Revenue Analysis", "mixed"),
            ("Operating Expenses", "mixed"),
            ("Capital Expenditures", "text"),
            ("Risk Factors", "text"),
            ("Corporate Governance", "text"),
            ("Board of Directors", "text"),
            ("Sustainability and ESG", "text"),
            ("Financial Statements", "mixed"),
            ("Notes to Financial Statements", "text"),
            ("Auditor's Report", "text"),
            ("Shareholder Information", "text"),
        ],
    },
    "legal_deposition": {
        "domain": "legal",
        "page_range": (40, 100),
        "section_templates": [
            ("Appearances", "text"),
            ("Preliminary Statements", "text"),
            ("Direct Examination", "text"),
            ("Cross Examination", "text"),
            ("Redirect Examination", "text"),
            ("Exhibit Discussion", "text"),
            ("Stipulations", "text"),
            ("Objections and Rulings", "text"),
            ("Continued Testimony", "text"),
            ("Closing Statements", "text"),
        ],
    },
    "technical_spec": {
        "domain": "engineering",
        "page_range": (30, 80),
        "section_templates": [
            ("Scope and Purpose", "text"),
            ("Referenced Documents", "text"),
            ("Definitions and Acronyms", "text"),
            ("System Overview", "text"),
            ("Functional Requirements", "mixed"),
            ("Performance Requirements", "mixed"),
            ("Interface Requirements", "mixed"),
            ("Design Constraints", "text"),
            ("Security Requirements", "text"),
            ("Test and Verification", "mixed"),
            ("Configuration Management", "text"),
            ("Quality Assurance", "text"),
            ("Appendix A: Data Dictionary", "mixed"),
            ("Appendix B: Compliance Matrix", "mixed"),
        ],
    },
    "regulatory_filing": {
        "domain": "regulatory_compliance",
        "page_range": (50, 100),
        "section_templates": [
            ("Cover Sheet and Filing Information", "text"),
            ("Executive Summary", "text"),
            ("Entity Description", "text"),
            ("Regulatory Framework", "text"),
            ("Compliance Methodology", "text"),
            ("Risk Assessment", "text"),
            ("Capital Adequacy", "mixed"),
            ("Liquidity Analysis", "text"),
            ("Operational Risk", "text"),
            ("Market Risk Disclosures", "text"),
            ("Internal Controls", "text"),
            ("Remediation Plans", "text"),
            ("Legal Proceedings", "text"),
            ("Certifications", "text"),
            ("Exhibits and Schedules", "mixed"),
        ],
    },
}

_ALL_DOC_TYPES = list(_DOC_TYPE_DEFS.keys())

# ---------------------------------------------------------------------------
# Tier distribution weights
# ---------------------------------------------------------------------------

_PROFILING_TIER_WEIGHTS: dict[str, float] = {
    "easy": 0.30,
    "medium": 0.30,
    "hard": 0.25,
    "edge": 0.15,
}

_GT_TIER_WEIGHTS: dict[str, float] = {
    "easy": 0.25,
    "medium": 0.25,
    "hard": 0.25,
    "edge": 0.15,
    "extreme": 0.10,
}

_PROFILING_COUNT = 50
_GT_COUNT = 500


# ---------------------------------------------------------------------------
# Corpus planning helpers
# ---------------------------------------------------------------------------


def _distribute_items(
    n: int,
    weights: dict[str, float],
    rng: random.Random,
) -> list[str]:
    """Assign n items to categories according to weights, shuffle deterministically."""
    assignments: list[str] = []
    categories = list(weights.keys())
    counts = {cat: int(round(n * w)) for cat, w in weights.items()}

    # Fix rounding so total == n.
    diff = n - sum(counts.values())
    if diff > 0:
        for _ in range(diff):
            cat = rng.choice(categories)
            counts[cat] += 1
    elif diff < 0:
        for _ in range(-diff):
            candidates = [c for c in categories if counts[c] > 0]
            cat = rng.choice(candidates)
            counts[cat] -= 1

    for cat in categories:
        assignments.extend([cat] * counts[cat])

    rng.shuffle(assignments)
    return assignments


def _estimate_pages(target_tokens: int, density: int) -> int:
    """Estimate page count from target token count and density."""
    return max(1, round(target_tokens / density))


def _choose_density(profile: str, rng: random.Random) -> int:
    """Select content density (tokens/page) based on profile type."""
    if profile == "profiling":
        return _PROFILING_DENSITY
    lo, hi = _GT_DENSITY_RANGE
    return rng.randint(lo, hi)


def _build_sections(
    doc_type: str,
    target_pages: int,
    rng: random.Random,
) -> list[SectionSpec]:
    """Build section specs by distributing pages across type-specific templates."""
    templates = _DOC_TYPE_DEFS[doc_type]["section_templates"]

    # Pick a reasonable section count: use all templates, but scale for long docs.
    n_sections = len(templates)
    if target_pages > 80:
        # Add extra sections for very long documents.
        extra = min(5, (target_pages - 80) // 10)
        n_sections += extra

    # Cycle templates if needed.
    selected: list[tuple[str, str]] = []
    for i in range(n_sections):
        title, content_type = templates[i % len(templates)]
        if i >= len(templates):
            title = f"{title} (continued {i // len(templates) + 1})"
        selected.append((title, content_type))

    # Distribute pages across sections.
    base_pages = max(1, target_pages // n_sections)
    remainder = target_pages - base_pages * n_sections
    page_alloc = [base_pages] * n_sections
    for i in range(max(0, remainder)):
        page_alloc[i % n_sections] += 1

    # For legal depositions, make section lengths more uneven (Q&A format).
    if doc_type == "legal_deposition":
        _skew_pages(page_alloc, rng)

    return [
        SectionSpec(
            title=title,
            target_pages=max(1, pages),
            content_type=content_type,
        )
        for (title, content_type), pages in zip(selected, page_alloc, strict=True)
    ]


def _skew_pages(page_alloc: list[int], rng: random.Random) -> None:
    """Redistribute pages to create uneven section lengths (in-place).

    Move pages from shorter sections to longer ones to simulate the
    naturally uneven flow of deposition testimony.
    """
    total = sum(page_alloc)
    n = len(page_alloc)
    for _ in range(n // 2):
        src = rng.randrange(n)
        dst = rng.randrange(n)
        if src != dst and page_alloc[src] > 1:
            transfer = rng.randint(1, max(1, page_alloc[src] // 2))
            page_alloc[src] -= transfer
            page_alloc[dst] += transfer
    # Ensure total is preserved.
    current = sum(page_alloc)
    if current != total:
        page_alloc[0] += total - current


def _plan_document(
    idx: int,
    tier: str,
    profile: str,
    rng: random.Random,
) -> DocumentSpec:
    """Plan a single document: choose type, token target, pages, sections."""
    doc_type = rng.choice(_ALL_DOC_TYPES)
    tier_def = W18_TIERS[tier]

    # Token range
    token_key = "profiling_tokens" if profile == "profiling" else "gt_tokens"
    token_range = tier_def[token_key]
    if token_range is None:
        # Extreme tier not available for profiling -- fallback to GT range.
        token_range = tier_def["gt_tokens"]
    min_tokens, max_tokens = token_range
    target_tokens = rng.randint(min_tokens, max_tokens)

    # Derive page count from token target and density.
    density = _choose_density(profile, rng)
    target_pages = _estimate_pages(target_tokens, density)

    # Clamp to doc-type page range.
    type_min, type_max = _DOC_TYPE_DEFS[doc_type]["page_range"]
    target_pages = max(type_min, min(type_max, target_pages))

    sections = _build_sections(doc_type, target_pages, rng)

    # Structure quality: profiling always well-structured; legal depositions
    # are inherently less structured.
    if doc_type == "legal_deposition":
        structure_quality = "poorly_structured"
    else:
        structure_quality = "well_structured"

    return DocumentSpec(
        doc_id=f"w18-{doc_type}-{tier}-{idx:04d}",
        workflow="w18",
        profile=profile,
        document_type=doc_type,
        domain=_DOC_TYPE_DEFS[doc_type]["domain"],
        sections=sections,
        target_page_count=target_pages,
        target_token_count=target_tokens,
        generation_model=_MODEL,
        modality_mix=(1.0, 0.0, 0.0),  # text-only
        structure_quality=structure_quality,
    )


# ---------------------------------------------------------------------------
# Token verification
# ---------------------------------------------------------------------------


def _verify_token_count(
    pdf_path: Path,
    target_tokens: int,
    tier: str,
    profile: str,
    doc_id: str,
) -> int:
    """Extract text from rendered PDF and verify token count against tier range.

    Return the actual token count. Log a warning if outside the tier range.
    """
    try:
        import pdfplumber
    except ImportError:
        logger.warning(
            "pdfplumber not installed -- skipping token verification for %s",
            doc_id,
        )
        return target_tokens

    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    actual_tokens = count_tokens(text)

    tier_def = W18_TIERS[tier]
    token_key = "profiling_tokens" if profile == "profiling" else "gt_tokens"
    token_range = tier_def[token_key]
    if token_range is None:
        token_range = tier_def["gt_tokens"]
    lo, hi = token_range

    if actual_tokens < lo or actual_tokens > hi:
        logger.warning(
            "Token count mismatch for %s: target=%d, actual=%d, tier %s range=[%d, %d]",
            doc_id,
            target_tokens,
            actual_tokens,
            tier,
            lo,
            hi,
        )

    return actual_tokens


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def _generate_single_document(
    spec: DocumentSpec,
    tier: str,
    output_dir: Path,
    api_key: str | None,
) -> PDFDescriptor:
    """Generate one long document PDF from a DocumentSpec."""
    sections_dicts = [
        {
            "title": s.title,
            "target_pages": s.target_pages,
            "content_type": s.content_type,
            "key_values": dict(s.key_values),
        }
        for s in spec.sections
    ]

    content = await generate_document_content(
        document_type=spec.document_type.replace("_", " "),
        domain=spec.domain.replace("_", " "),
        sections=sections_dicts,
        target_pages=spec.target_page_count,
        model=_MODEL,
        structure_quality=spec.structure_quality,
        api_key=api_key,
    )

    # Render to PDF.
    pdf_path = output_dir / f"{spec.doc_id}.pdf"
    layout = PageLayout(body_font_size_pt=10, line_spacing=1.15)
    render_markdown_to_pdf(content, pdf_path, layout=layout)

    # Count rendered pages.
    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    page_count = count_pdf_pages(pdf_path)

    # Verify token count against tier range.
    target_tokens = spec.target_token_count or (spec.target_page_count * _PROFILING_DENSITY)
    actual_tokens = _verify_token_count(pdf_path, target_tokens, tier, spec.profile, spec.doc_id)

    # Determine content density label.
    if page_count > 0:
        actual_density = actual_tokens / page_count
        if actual_density > 900:
            density_label = "dense"
        elif actual_density < 600:
            density_label = "sparse"
        else:
            density_label = "mixed"
    else:
        density_label = "mixed"

    descriptor = PDFDescriptor(
        pdf_id=spec.doc_id,
        workflow="w18",
        profile=spec.profile,
        document_type=spec.document_type,
        page_count=page_count,
        estimated_token_count=actual_tokens,
        text_pages=page_count,
        table_chart_pages=0,
        scanned_pages=0,
        section_count=len(spec.sections),
        structure_quality=spec.structure_quality,
        content_density=density_label,
        generation_model=_MODEL,
    )
    write_descriptor(descriptor, output_dir)

    logger.info(
        "Generated W18 %s [%s]: %d pages, %d tokens (target %d), tier=%s",
        spec.document_type,
        spec.doc_id,
        page_count,
        actual_tokens,
        target_tokens,
        tier,
    )
    return descriptor


async def generate_w18_corpus(
    output_dir: Path,
    profile: str,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Generate the full W18 long-document corpus.

    Args:
        output_dir: Directory to write PDFs and descriptor JSON files.
        profile: Either "profiling" (50 docs) or "ground_truth" (500 docs).
        seed: Random seed for deterministic tier/type assignment.
        api_key: Optional API key for the generation model.

    Returns:
        List of PDFDescriptor objects, one per generated document.
    """
    if profile not in ("profiling", "ground_truth"):
        raise ValueError(f"profile must be 'profiling' or 'ground_truth', got {profile!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)  # noqa: S311

    n_docs = _PROFILING_COUNT if profile == "profiling" else _GT_COUNT
    tier_weights = _PROFILING_TIER_WEIGHTS if profile == "profiling" else _GT_TIER_WEIGHTS
    tier_assignments = _distribute_items(n_docs, tier_weights, rng)

    # Plan all documents up front for deterministic assignment.
    specs_with_tiers = [
        (_plan_document(i, tier, profile, rng), tier) for i, tier in enumerate(tier_assignments)
    ]

    logger.info(
        "Planned %d W18 documents (%s): %s",
        n_docs,
        profile,
        {t: tier_assignments.count(t) for t in sorted(set(tier_assignments))},
    )

    # Generate concurrently (10 parallel LLM calls by default).
    tasks = [
        _generate_single_document(spec, tier, output_dir, api_key)
        for spec, tier in specs_with_tiers
    ]
    for i, (spec, _) in enumerate(specs_with_tiers):
        logger.info("Queued document %d/%d: %s", i + 1, n_docs, spec.doc_id)

    descriptors = await run_concurrent(tasks, concurrency=50)

    logger.info(
        "Completed W18 corpus: %d documents in %s",
        len(descriptors),
        output_dir,
    )
    return descriptors


def generate_w18_corpus_sync(
    output_dir: Path,
    profile: str,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Synchronous wrapper around generate_w18_corpus."""
    return asyncio.run(
        generate_w18_corpus(output_dir, profile=profile, seed=seed, api_key=api_key)
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate W18 long-document PDFs for single-pass backtesting.",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["profiling", "ground_truth"],
        default="profiling",
        help="Corpus profile: 'profiling' (50 docs) or 'ground_truth' (500 docs).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=None,
        help="Override document count (default: 50 profiling / 500 ground_truth).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: pdfs/generated/{profile}/w18/).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else Path(f"pdfs/generated/{args.profile}/w18/")

    # Patch corpus size if --n is provided.
    if args.n is not None:
        import pdfs.generators.w18_long_documents as _self

        if args.profile == "profiling":
            _self._PROFILING_COUNT = args.n  # type: ignore[attr-defined]
        else:
            _self._GT_COUNT = args.n  # type: ignore[attr-defined]

    results = generate_w18_corpus_sync(out, profile=args.profile, seed=args.seed)
    for desc in results:
        print(  # noqa: T201
            f"  {desc.document_type} [{desc.content_density}]: {desc.pdf_id} "
            f"({desc.page_count} pages, {desc.estimated_token_count} tokens)"
        )
