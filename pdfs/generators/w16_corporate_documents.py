"""Generate corporate document corpus for W16 map-reduce analysis backtesting.

Produce 50 (profiling) or 500 (ground truth) PDFs spanning 5 document types
and 4-5 difficulty tiers.  Each PDF is one backtesting run input -- the
map-reduce workflow splits it into N sections for parallel summarisation,
so section count is the primary cost driver.

Model: deepseek-v4-pro for all documents.  For docs >60 pages the
generate_document_content helper automatically switches to sectional generation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from pathlib import Path
from typing import Any

from pdfs.generators._llm import count_tokens, generate_document_content, run_concurrent
from pdfs.generators._types import W16_TIERS, DocumentSpec, SectionSpec
from pdfs.generators.rendering.pdf_assembler import PDFDescriptor, write_descriptor
from pdfs.generators.rendering.text_renderer import PageLayout, render_markdown_to_pdf

logger = logging.getLogger(__name__)

_MODEL = "deepseek-v4-pro"

# ---------------------------------------------------------------------------
# Document-type templates
# ---------------------------------------------------------------------------

# (doc_type_key, human_label, domain, section_templates, content_mix)
# section_templates: list of (title, content_type) that get scaled to the
# target section count -- templates are cycled or truncated as needed.

_DOC_TYPE_DEFS: dict[str, dict[str, Any]] = {
    "annual_report": {
        "domain": "corporate_finance",
        "section_templates": [
            ("Letter to Shareholders", "text"),
            ("Financial Highlights", "table_heavy"),
            ("Business Overview", "mixed"),
            ("Revenue Breakdown", "table_heavy"),
            ("Risk Factors", "text"),
            ("Corporate Governance", "text"),
            ("Sustainability Report", "mixed"),
            ("Financial Statements", "table_heavy"),
            ("Notes to Financial Statements", "text"),
            ("Auditor's Report", "text"),
        ],
        "modality_mix": (0.7, 0.3, 0.0),  # text, tables, scanned
    },
    "regulatory_filing": {
        "domain": "regulatory_compliance",
        "section_templates": [
            ("Filing Summary", "text"),
            ("Entity Information", "mixed"),
            ("Regulatory Background", "text"),
            ("Compliance Assessment", "mixed"),
            ("Data Tables", "table_heavy"),
            ("Risk Disclosures", "text"),
            ("Legal Opinions", "text"),
            ("Exhibits and Schedules", "table_heavy"),
            ("Certification", "text"),
        ],
        "modality_mix": (0.8, 0.2, 0.0),
    },
    "research_paper": {
        "domain": "academic_research",
        "section_templates": [
            ("Abstract", "text"),
            ("Introduction", "text"),
            ("Literature Review", "text"),
            ("Methodology", "mixed"),
            ("Results", "table_heavy"),
            ("Discussion", "text"),
            ("Conclusion", "text"),
            ("References", "text"),
        ],
        "modality_mix": (0.85, 0.15, 0.0),
    },
    "meeting_transcript": {
        "domain": "corporate_governance",
        "section_templates": [
            ("Call to Order", "text"),
            ("Roll Call and Attendance", "text"),
            ("Previous Minutes Approval", "text"),
            ("Financial Report Discussion", "text"),
            ("Strategic Initiatives", "text"),
            ("Open Discussion", "text"),
            ("Motions and Votes", "text"),
            ("Adjournment", "text"),
        ],
        "modality_mix": (1.0, 0.0, 0.0),
    },
    "technical_spec": {
        "domain": "engineering",
        "section_templates": [
            ("Scope and Purpose", "text"),
            ("Definitions and Abbreviations", "text"),
            ("System Requirements", "mixed"),
            ("Architecture Overview", "text"),
            ("Interface Specifications", "table_heavy"),
            ("Performance Requirements", "table_heavy"),
            ("Test Procedures", "mixed"),
            ("Compliance Matrix", "table_heavy"),
            ("Appendices", "mixed"),
        ],
        "modality_mix": (0.7, 0.3, 0.0),
    },
}

# All doc type keys
_ALL_DOC_TYPES = list(_DOC_TYPE_DEFS.keys())

# ---------------------------------------------------------------------------
# Tier distribution weights
# ---------------------------------------------------------------------------

_PROFILING_TIER_WEIGHTS: dict[str, float] = {
    "easy": 0.35,
    "medium": 0.30,
    "hard": 0.25,
    "edge": 0.10,
}

_GT_TIER_WEIGHTS: dict[str, float] = {
    "easy": 0.30,
    "medium": 0.25,
    "hard": 0.25,
    "edge": 0.15,
    "extreme": 0.05,
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


def _pick_doc_type(tier: str, rng: random.Random) -> str:
    """Select a document type valid for the given tier."""
    allowed = W16_TIERS[tier]["doc_types"]
    return rng.choice(allowed)


def _build_sections(
    doc_type: str,
    n_sections: int,
    target_pages: int,
    structure_quality: str,
    rng: random.Random,
) -> list[SectionSpec]:
    """Build section specs for a document by cycling through type-specific templates."""
    templates = _DOC_TYPE_DEFS[doc_type]["section_templates"]

    # Cycle templates to fill n_sections.
    selected: list[tuple[str, str]] = []
    for i in range(n_sections):
        title, content_type = templates[i % len(templates)]
        # Avoid duplicate titles when cycling by appending a suffix.
        if i >= len(templates):
            title = f"{title} (continued {i // len(templates) + 1})"
        selected.append((title, content_type))

    # Distribute pages across sections.
    base_pages = max(1, target_pages // n_sections)
    remainder = target_pages - base_pages * n_sections
    page_alloc = [base_pages] * n_sections
    for i in range(max(0, remainder)):
        page_alloc[i % n_sections] += 1

    specs: list[SectionSpec] = []
    for (title, content_type), pages in zip(selected, page_alloc, strict=True):
        # For poorly structured docs, occasionally switch content type.
        if structure_quality == "poorly_structured" and rng.random() < 0.3:
            content_type = rng.choice(["text", "mixed", "table_heavy"])
        specs.append(
            SectionSpec(
                title=title,
                target_pages=max(1, pages),
                content_type=content_type,
            )
        )
    return specs


def _plan_document(
    idx: int,
    tier: str,
    profile: str,
    rng: random.Random,
) -> DocumentSpec:
    """Plan a single document: choose type, pages, sections, structure quality."""
    doc_type = _pick_doc_type(tier, rng)
    tier_def = W16_TIERS[tier]

    # Page range
    page_key = "profiling_pages" if profile == "profiling" else "gt_pages"
    page_range = tier_def[page_key]
    if page_range is None:
        # Extreme tier not available for profiling -- shouldn't happen due to
        # weight distribution, but guard anyway.
        page_range = tier_def["gt_pages"]
    min_pages, max_pages = page_range
    target_pages = rng.randint(min_pages, max_pages)

    # Section count
    sec_min, sec_max = tier_def["sections"]
    n_sections = rng.randint(sec_min, sec_max)

    # Structure quality: profiling is always well-structured; GT is ~30% poorly structured.
    if profile == "profiling":
        structure_quality = "well_structured"
    else:
        structure_quality = "poorly_structured" if rng.random() < 0.30 else "well_structured"

    sections = _build_sections(doc_type, n_sections, target_pages, structure_quality, rng)
    doc_def = _DOC_TYPE_DEFS[doc_type]

    return DocumentSpec(
        doc_id=f"w16-{doc_type}-{tier}-{idx:04d}",
        workflow="w16",
        profile=profile,
        document_type=doc_type,
        domain=doc_def["domain"],
        sections=sections,
        target_page_count=target_pages,
        generation_model=_MODEL,
        modality_mix=doc_def["modality_mix"],
        structure_quality=structure_quality,
    )


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def _generate_single_document(
    spec: DocumentSpec,
    output_dir: Path,
    api_key: str | None,
) -> PDFDescriptor:
    """Generate one corporate document PDF from a DocumentSpec."""
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
    token_count = count_tokens(content)

    # Estimate table/chart pages from modality mix.
    text_frac, table_frac, _scan_frac = spec.modality_mix
    table_chart_pages = int(round(page_count * table_frac))
    text_pages = page_count - table_chart_pages

    descriptor = PDFDescriptor(
        pdf_id=spec.doc_id,
        workflow="w16",
        profile=spec.profile,
        document_type=spec.document_type,
        page_count=page_count,
        estimated_token_count=token_count,
        text_pages=text_pages,
        table_chart_pages=table_chart_pages,
        scanned_pages=0,
        section_count=len(spec.sections),
        structure_quality=spec.structure_quality,
        content_density="dense" if spec.document_type == "regulatory_filing" else "mixed",
        generation_model=_MODEL,
    )
    write_descriptor(descriptor, output_dir)

    logger.info(
        "Generated W16 %s [%s]: %d pages (%d target), %d sections, %d tokens",
        spec.document_type,
        spec.doc_id,
        page_count,
        spec.target_page_count,
        len(spec.sections),
        token_count,
    )
    return descriptor


async def generate_w16_corpus(
    output_dir: Path,
    profile: str,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Generate the full W16 corporate document corpus.

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
    specs = [_plan_document(i, tier, profile, rng) for i, tier in enumerate(tier_assignments)]

    logger.info(
        "Planned %d W16 documents (%s): %s",
        n_docs,
        profile,
        {t: tier_assignments.count(t) for t in sorted(set(tier_assignments))},
    )

    # Generate concurrently (10 parallel LLM calls by default).
    tasks = [_generate_single_document(spec, output_dir, api_key) for spec in specs]
    for i, spec in enumerate(specs):
        logger.info("Queued document %d/%d: %s", i + 1, n_docs, spec.doc_id)

    descriptors = await run_concurrent(tasks)

    logger.info(
        "Completed W16 corpus: %d documents in %s",
        len(descriptors),
        output_dir,
    )
    return descriptors


def generate_w16_corpus_sync(
    output_dir: Path,
    profile: str,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Synchronous wrapper around generate_w16_corpus."""
    return asyncio.run(
        generate_w16_corpus(output_dir, profile=profile, seed=seed, api_key=api_key)
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate W16 corporate document PDFs for map-reduce backtesting.",
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
        help="Output directory (default: pdfs/generated/{profile}/w16/).",
    )
    args = parser.parse_args()

    out = Path(args.output_dir) if args.output_dir else Path(f"pdfs/generated/{args.profile}/w16/")

    # Patch corpus size if --n is provided.
    if args.n is not None:
        import pdfs.generators.w16_corporate_documents as _self

        if args.profile == "profiling":
            _self._PROFILING_COUNT = args.n  # type: ignore[attr-defined]
        else:
            _self._GT_COUNT = args.n  # type: ignore[attr-defined]

    results = generate_w16_corpus_sync(out, profile=args.profile, seed=args.seed)
    for desc in results:
        print(  # noqa: T201
            f"  {desc.document_type} [{desc.structure_quality}]: {desc.pdf_id} "
            f"({desc.page_count} pages, {desc.section_count} sections, "
            f"{desc.estimated_token_count} tokens)"
        )
