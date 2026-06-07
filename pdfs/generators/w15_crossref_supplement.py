"""Generate W15 cross-reference supplement documents for multi-hop RAG.

Extend the base W14/W15 corpus with three additional document types designed to
force multi-hop retrieval: coverage comparisons, appeals handbooks, and
policy amendments. Information for multi-hop queries is deliberately spread
across non-adjacent documents so no single document contains a complete answer.

Consume the base corpus manifest (from w14_w15_insurance_corpus.py) to create
valid cross-references into existing documents.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pdfs.generators._llm import count_tokens, generate_document_content
from pdfs.generators._types import (
    W14_W15_PROVIDER_VALUES,
    W15_SUPPLEMENT_DOC_TYPES,
    ContentManifest,
    ContentManifestEntry,
    ContentManifestSection,
    SectionSpec,
)
from pdfs.generators.rendering.pdf_assembler import (
    PageSource,
    PDFDescriptor,
    assemble_pdf,
    write_descriptor,
)
from pdfs.generators.rendering.scan_simulator import rasterize_pdf_pages
from pdfs.generators.rendering.text_renderer import PageLayout, render_markdown_to_pdf

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROVIDERS = list(W14_W15_PROVIDER_VALUES.keys())

# Model assignment: coverage_comparison -> Sonnet (cross-ref accuracy), rest -> DeepSeek
_MODEL_FOR_DOCTYPE: dict[str, str] = {
    "coverage_comparison": "claude-sonnet-4-6",
    "appeals_handbook": "deepseek-v4-pro",
    "amendment_rider": "deepseek-v4-pro",
}

# 8 supplement docs per distribution
_SUPPLEMENT_COUNT = 8

# Modality distribution mirrors W14 settings
_MODALITY_MIX: dict[str, tuple[float, float, float]] = {
    "profiling": (0.80, 0.20, 0.0),
    "ground_truth": (0.50, 0.30, 0.20),
}

_POOR_STRUCTURE_RATE: dict[str, float] = {
    "profiling": 0.10,
    "ground_truth": 0.30,
}

_DOCTYPE_LABELS: dict[str, str] = {
    "coverage_comparison": "Coverage Comparison Guide",
    "appeals_handbook": "Appeals and Grievances Handbook",
    "amendment_rider": "Provider-Specific Amendment/Rider",
}


# ---------------------------------------------------------------------------
# Section templates per supplement doc type
# ---------------------------------------------------------------------------


def _coverage_comparison_sections(
    pages: int,
    provider: str,
    base_manifest: ContentManifest,
    rng: random.Random,
) -> list[SectionSpec]:
    """Build sections for a coverage comparison guide.

    Cross-reference specific plan types within a provider by pulling section
    titles and key facts from the base manifest.
    """
    base = max(1, pages // 4)
    remainder = max(1, pages - 3 * base)

    # Extract cross-reference data from base manifest
    xref_facts = _extract_provider_xrefs(provider, base_manifest)

    return [
        SectionSpec(
            title="Plan Tier Overview",
            target_pages=base,
            content_type="table_heavy",
            key_values={"cross_ref_source": "base_corpus_sbc"},
        ),
        SectionSpec(
            title="Side-by-Side Coverage Comparison",
            target_pages=remainder,
            content_type="table_heavy",
            key_values=xref_facts,
        ),
        SectionSpec(
            title="Cost Scenario Analysis",
            target_pages=base,
            content_type="mixed",
        ),
        SectionSpec(
            title="Choosing the Right Plan",
            target_pages=base,
            content_type="text",
        ),
    ]


def _appeals_handbook_sections(
    pages: int,
    provider: str,
    base_manifest: ContentManifest,
    rng: random.Random,
) -> list[SectionSpec]:
    """Build sections for an appeals and grievances handbook."""
    base = max(1, pages // 4)
    remainder = max(1, pages - 3 * base)

    return [
        SectionSpec(
            title="Understanding Denials and Adverse Determinations",
            target_pages=base,
            content_type="text",
            key_values={"cross_ref_source": "base_corpus_detailed_policy"},
        ),
        SectionSpec(
            title="Internal Appeal Process",
            target_pages=remainder,
            content_type="mixed",
        ),
        SectionSpec(
            title="External Review Rights",
            target_pages=base,
            content_type="text",
        ),
        SectionSpec(
            title="Grievance Filing Procedures and Timelines",
            target_pages=base,
            content_type="text",
        ),
    ]


def _amendment_rider_sections(
    pages: int,
    provider: str,
    base_manifest: ContentManifest,
    rng: random.Random,
) -> list[SectionSpec]:
    """Build sections for an amendment/rider document.

    Deliberately short -- modifies specific clauses from the base policy.
    """
    base = max(1, pages // 3)
    remainder = max(1, pages - 2 * base)

    # Find a specific base document to reference
    ref_doc = _find_base_doc(provider, "detailed_policy", base_manifest)
    ref_info: dict[str, str] = {}
    if ref_doc:
        ref_info["amends_document"] = ref_doc.pdf_id
        if ref_doc.sections:
            ref_info["amends_section"] = ref_doc.sections[0].title

    return [
        SectionSpec(
            title="Amendment Scope and Effective Date",
            target_pages=base,
            content_type="text",
            key_values=ref_info,
        ),
        SectionSpec(
            title="Modified Coverage Terms",
            target_pages=remainder,
            content_type="mixed",
        ),
        SectionSpec(
            title="Superseded Provisions",
            target_pages=base,
            content_type="text",
        ),
    ]


_SECTION_BUILDERS: dict[str, Any] = {
    "coverage_comparison": _coverage_comparison_sections,
    "appeals_handbook": _appeals_handbook_sections,
    "amendment_rider": _amendment_rider_sections,
}


# ---------------------------------------------------------------------------
# Cross-reference helpers
# ---------------------------------------------------------------------------


def _extract_provider_xrefs(
    provider: str,
    manifest: ContentManifest,
) -> dict[str, str]:
    """Pull key facts from the base manifest for cross-referencing.

    Gather deductible, copay, and coverage facts from the provider's base
    documents so the comparison guide can reference them accurately.
    """
    xrefs: dict[str, str] = {}
    for doc in manifest.documents:
        if doc.provider != provider:
            continue
        for section in doc.sections:
            for fact in section.key_facts:
                # Preserve the fact with its source document for traceability
                key = f"{doc.document_type}::{section.title}::{fact.split(':')[0].strip()}"
                xrefs[key] = fact
                # Cap at 10 cross-refs to keep prompts manageable
                if len(xrefs) >= 10:
                    return xrefs
    return xrefs


def _find_base_doc(
    provider: str,
    doc_type: str,
    manifest: ContentManifest,
) -> ContentManifestEntry | None:
    """Find the first base corpus document matching provider and type."""
    for doc in manifest.documents:
        if doc.provider == provider and doc.document_type == doc_type:
            return doc
    return None


# ---------------------------------------------------------------------------
# Page modality assignment (shared with w14_w15)
# ---------------------------------------------------------------------------


def _assign_page_modalities(
    page_count: int,
    modality_mix: tuple[float, float, float],
    rng: random.Random,
) -> tuple[list[int], list[int], list[int]]:
    """Assign each page index to text, table/chart, or scanned.

    Return (text_indices, table_chart_indices, scanned_indices).
    """
    text_frac, table_frac, scan_frac = modality_mix
    n_table = max(0, round(page_count * table_frac))
    n_scan = max(0, round(page_count * scan_frac))
    n_text = max(0, page_count - n_table - n_scan)

    indices = list(range(page_count))
    rng.shuffle(indices)

    text_idx = sorted(indices[:n_text])
    table_idx = sorted(indices[n_text : n_text + n_table])
    scan_idx = sorted(indices[n_text + n_table :])

    return text_idx, table_idx, scan_idx


# ---------------------------------------------------------------------------
# Scan simulation
# ---------------------------------------------------------------------------


def _should_apply_scanning(profile: str, rng: random.Random) -> bool:
    """Decide whether a GT supplement PDF gets scanned pages (~40% of GT PDFs)."""
    if profile != "ground_truth":
        return False
    return rng.random() < 0.40


def _pick_scanned_pages(page_count: int, rng: random.Random) -> list[int]:
    """Select 30-50% of pages to rasterize as scanned."""
    scan_frac = rng.uniform(0.30, 0.50)
    n_scan = max(1, round(page_count * scan_frac))
    indices = list(range(page_count))
    rng.shuffle(indices)
    return sorted(indices[:n_scan])


# ---------------------------------------------------------------------------
# Document plan
# ---------------------------------------------------------------------------


_supplement_plan_counter = 0


@dataclass(slots=True)
class _SupplementPlan:
    """Internal plan for one supplement document."""

    provider: str
    doc_type: str
    page_count: int
    model: str
    structure_quality: str
    sections: list[SectionSpec]
    doc_id: str = ""
    modality_mix: tuple[float, float, float] = (1.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if not self.doc_id:
            global _supplement_plan_counter  # noqa: PLW0603
            _supplement_plan_counter += 1
            short = self.provider.lower().replace(" ", "_")
            self.doc_id = f"w15_{short}_{self.doc_type}_{_supplement_plan_counter:03d}"


def _pick_page_count(
    doc_type: str,
    profile: str,
    rng: random.Random,
) -> int:
    """Select page count from the supplement doc type range."""
    lo, hi = W15_SUPPLEMENT_DOC_TYPES[doc_type]
    if profile == "profiling":
        upper = lo + max(1, int((hi - lo) * 0.4))
        return rng.randint(lo, upper)
    return rng.randint(lo, hi)


def _build_supplement_plan(
    profile: str,
    base_manifest: ContentManifest,
    rng: random.Random,
) -> list[_SupplementPlan]:
    """Build list of supplement documents to generate.

    Distribute 8 documents across providers and supplement types, ensuring
    at least one coverage comparison (the most important for multi-hop).
    """
    modality = _MODALITY_MIX[profile]
    poor_rate = _POOR_STRUCTURE_RATE[profile]
    doc_types = list(W15_SUPPLEMENT_DOC_TYPES.keys())

    plans: list[_SupplementPlan] = []

    # Guarantee one coverage_comparison per provider (3 docs)
    for provider in PROVIDERS:
        dt = "coverage_comparison"
        pages = _pick_page_count(dt, profile, rng)
        is_poor = rng.random() < poor_rate
        sections = _SECTION_BUILDERS[dt](pages, provider, base_manifest, rng)
        plans.append(
            _SupplementPlan(
                provider=provider,
                doc_type=dt,
                page_count=pages,
                model=_MODEL_FOR_DOCTYPE[dt],
                structure_quality="poorly_structured" if is_poor else "well_structured",
                sections=sections,
                modality_mix=modality,
            )
        )

    # Fill remaining slots (8 - 3 = 5) with random types and providers
    remaining = _SUPPLEMENT_COUNT - len(plans)
    for _ in range(remaining):
        provider = rng.choice(PROVIDERS)
        dt = rng.choice(doc_types)
        pages = _pick_page_count(dt, profile, rng)
        is_poor = rng.random() < poor_rate
        sections = _SECTION_BUILDERS[dt](pages, provider, base_manifest, rng)
        plans.append(
            _SupplementPlan(
                provider=provider,
                doc_type=dt,
                page_count=pages,
                model=_MODEL_FOR_DOCTYPE[dt],
                structure_quality="poorly_structured" if is_poor else "well_structured",
                sections=sections,
                modality_mix=modality,
            )
        )

    return plans


# ---------------------------------------------------------------------------
# Single-document generation
# ---------------------------------------------------------------------------


async def _generate_one_supplement(
    plan: _SupplementPlan,
    output_dir: Path,
    profile: str,
    base_manifest: ContentManifest,
    rng: random.Random,
    api_key: str | None,
) -> tuple[PDFDescriptor, ContentManifestEntry]:
    """Generate a single supplement PDF and return its descriptor and manifest entry."""
    provider_kvs = dict(W14_W15_PROVIDER_VALUES[plan.provider])

    # Build cross-reference context for the LLM prompt
    xref_context = _build_xref_prompt_context(plan.provider, plan.doc_type, base_manifest)

    # Build section dicts
    section_dicts = [
        {
            "title": s.title,
            "target_pages": s.target_pages,
            "content_type": s.content_type,
            "key_values": dict(s.key_values),
        }
        for s in plan.sections
    ]

    # Merge cross-reference context into key_values for the LLM
    generation_kvs = dict(provider_kvs)
    generation_kvs.update(xref_context)

    logger.info(
        "Generating supplement %s for %s (%d pages, model=%s)",
        plan.doc_type,
        plan.provider,
        plan.page_count,
        plan.model,
    )

    content = await generate_document_content(
        document_type=_DOCTYPE_LABELS.get(plan.doc_type, plan.doc_type),
        domain="health insurance",
        sections=section_dicts,
        target_pages=plan.page_count,
        model=plan.model,
        provider_name=plan.provider,
        key_values=generation_kvs,
        structure_quality=plan.structure_quality,
        api_key=api_key,
    )

    token_count = count_tokens(content)

    # Render base PDF
    pdf_filename = f"{plan.doc_id}.pdf"
    pdf_path = output_dir / pdf_filename
    render_markdown_to_pdf(content, pdf_path, layout=PageLayout())

    # Count actual pages
    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    actual_pages = count_pdf_pages(pdf_path)

    # Modality assignment
    text_indices, table_indices, scan_indices = _assign_page_modalities(
        actual_pages,
        plan.modality_mix,
        rng,
    )

    # Scan simulation for ground truth
    final_scanned_count = 0
    if _should_apply_scanning(profile, rng):
        pages_to_scan = _pick_scanned_pages(actual_pages, rng)
        if pages_to_scan:
            scanned_images = rasterize_pdf_pages(pdf_path, pages_to_scan, rng=rng)

            page_sources: list[PageSource] = []
            scanned_set = set(pages_to_scan)
            scan_img_idx = 0

            for page_idx in range(actual_pages):
                if page_idx in scanned_set:
                    page_sources.append(
                        PageSource(
                            source_type="scanned_image",
                            pil_image=scanned_images[scan_img_idx],
                        )
                    )
                    scan_img_idx += 1
                    final_scanned_count += 1
                else:
                    page_sources.append(
                        PageSource(
                            source_type="text_pdf",
                            source_path=pdf_path,
                            source_page_index=page_idx,
                        )
                    )

            assembled_path = output_dir / f"{plan.doc_id}_assembled.pdf"
            assemble_pdf(page_sources, assembled_path)
            assembled_path.replace(pdf_path)

    # Build descriptor
    descriptor = PDFDescriptor(
        pdf_id=plan.doc_id,
        workflow="w15",
        profile=profile,
        document_type=plan.doc_type,
        page_count=actual_pages,
        estimated_token_count=token_count,
        text_pages=len(text_indices),
        table_chart_pages=len(table_indices),
        scanned_pages=final_scanned_count or len(scan_indices),
        section_count=len(plan.sections),
        key_fields_present=list(provider_kvs.keys()),
        provider=plan.provider,
        structure_quality=plan.structure_quality,
        content_density="mixed",
        generation_model=plan.model,
    )
    write_descriptor(descriptor, output_dir / "descriptors")

    # Build manifest entry
    manifest_sections = _build_manifest_sections(plan.sections, actual_pages)
    manifest_entry = ContentManifestEntry(
        pdf_id=plan.doc_id,
        pdf_filename=pdf_filename,
        provider=plan.provider,
        document_type=plan.doc_type,
        page_count=actual_pages,
        estimated_token_count=token_count,
        sections=manifest_sections,
    )

    return descriptor, manifest_entry


def _build_xref_prompt_context(
    provider: str,
    doc_type: str,
    base_manifest: ContentManifest,
) -> dict[str, str]:
    """Build cross-reference context to inject into the generation prompt.

    Inform the LLM about specific base documents and sections so it can
    create valid, traceable cross-references.
    """
    context: dict[str, str] = {}

    if doc_type == "coverage_comparison":
        # Reference SBC and member handbook for comparison data
        for target_type in ("sbc", "member_handbook", "formulary"):
            doc = _find_base_doc(provider, target_type, base_manifest)
            if doc:
                section_titles = ", ".join(s.title for s in doc.sections[:3])
                context[f"xref_{target_type}_doc"] = doc.pdf_id
                context[f"xref_{target_type}_sections"] = section_titles

    elif doc_type == "appeals_handbook":
        # Reference detailed policy for appeal procedures
        doc = _find_base_doc(provider, "detailed_policy", base_manifest)
        if doc:
            context["xref_policy_doc"] = doc.pdf_id
            appeal_sections = [
                s
                for s in doc.sections
                if any(kw in s.title.lower() for kw in ("appeal", "claim", "grievance", "review"))
            ]
            if appeal_sections:
                context["xref_policy_appeal_section"] = appeal_sections[0].title

    elif doc_type == "amendment_rider":
        # Reference detailed policy being amended
        doc = _find_base_doc(provider, "detailed_policy", base_manifest)
        if doc:
            context["amends_document_id"] = doc.pdf_id
            if doc.sections:
                # Pick a section to modify
                context["amended_section"] = doc.sections[0].title

    return context


def _build_manifest_sections(
    sections: list[SectionSpec],
    total_pages: int,
) -> list[ContentManifestSection]:
    """Build manifest sections with estimated page ranges and key facts."""
    manifest_sections: list[ContentManifestSection] = []
    page_cursor = 1

    for sec in sections:
        end_page = min(page_cursor + sec.target_pages - 1, total_pages)

        facts: list[str] = []
        for k, v in sec.key_values.items():
            facts.append(f"{k}: {v}")

        manifest_sections.append(
            ContentManifestSection(
                title=sec.title,
                page_range=(page_cursor, end_page),
                key_facts=facts,
            )
        )
        page_cursor = end_page + 1

    return manifest_sections


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def generate_w15_supplement(
    output_dir: Path,
    profile: str,
    base_manifest: ContentManifest,
    seed: int = 42,
    api_key: str | None = None,
) -> tuple[list[PDFDescriptor], ContentManifest]:
    """Generate W15 cross-reference supplement documents.

    Return (descriptors, supplement_manifest). The base_manifest parameter
    provides cross-reference targets from the W14 base corpus, ensuring
    supplement documents reference valid sections and facts.
    """
    if profile not in ("profiling", "ground_truth"):
        raise ValueError(f"profile must be 'profiling' or 'ground_truth', got {profile!r}")

    rng = random.Random(seed)  # noqa: S311
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plans = _build_supplement_plan(profile, base_manifest, rng)
    logger.info(
        "Supplement plan: %d documents (profile=%s)",
        len(plans),
        profile,
    )

    descriptors: list[PDFDescriptor] = []
    manifest_entries: list[ContentManifestEntry] = []

    for i, plan in enumerate(plans):
        logger.info(
            "Supplement %d/%d: %s / %s",
            i + 1,
            len(plans),
            plan.provider,
            plan.doc_type,
        )
        descriptor, entry = await _generate_one_supplement(
            plan=plan,
            output_dir=output_dir,
            profile=profile,
            base_manifest=base_manifest,
            rng=rng,
            api_key=api_key,
        )
        descriptors.append(descriptor)
        manifest_entries.append(entry)

    # Build supplement manifest
    corpus_id = f"w15_supplement_{profile}_{seed}"
    manifest = ContentManifest(
        corpus_id=corpus_id,
        workflow="w15",
        profile=profile,
        generated_at=datetime.now(UTC).isoformat(),
        documents=manifest_entries,
    )
    manifest.save(output_dir / "supplement_manifest.json")

    logger.info(
        "Supplement generation complete: %d PDFs, manifest saved to %s",
        len(descriptors),
        output_dir / "supplement_manifest.json",
    )

    return descriptors, manifest


def generate_w15_supplement_sync(
    output_dir: Path,
    profile: str,
    base_manifest: ContentManifest,
    seed: int = 42,
    api_key: str | None = None,
) -> tuple[list[PDFDescriptor], ContentManifest]:
    """Synchronous wrapper around generate_w15_supplement."""
    return asyncio.run(
        generate_w15_supplement(
            output_dir=output_dir,
            profile=profile,
            base_manifest=base_manifest,
            seed=seed,
            api_key=api_key,
        )
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate W15 cross-reference supplement documents for multi-hop RAG.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=["profiling", "ground_truth"],
        help="Distribution profile: profiling (small, clean) or ground_truth (large, messy).",
    )
    parser.add_argument(
        "--base-manifest",
        type=Path,
        required=True,
        help="Path to the W14/W15 base corpus manifest.json.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic structural decisions (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/w15_supplement"),
        help="Directory for generated PDFs and manifest (default: output/w15_supplement).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key override (falls back to ANTHROPIC_API_KEY / DEEPSEEK_API_KEY env vars).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = _parse_args()
    base = ContentManifest.load(args.base_manifest)
    descriptors, manifest = generate_w15_supplement_sync(
        output_dir=args.output_dir,
        profile=args.profile,
        base_manifest=base,
        seed=args.seed,
        api_key=args.api_key,
    )
    print(f"Generated {len(descriptors)} supplement PDFs -> {args.output_dir}")
    print(f"Supplement manifest: {args.output_dir / 'supplement_manifest.json'}")
