"""Generate the shared W14/W15 health insurance PDF corpus.

Produce a realistic document collection across three providers (United Healthcare,
Aetna, BlueCross BlueShield) with five document types. The corpus serves two
purposes: RAG retrieval profiling and ground truth distribution evaluation.

PDF generation runs BEFORE input generation -- the ContentManifest produced here
is consumed by downstream query generators.
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

from pdfs.generators._llm import count_tokens, generate_document_content, run_concurrent
from pdfs.generators._types import (
    W14_DOC_TYPES,
    W14_W15_PROVIDER_VALUES,
    ContentManifest,
    ContentManifestEntry,
    ContentManifestSection,
    SectionSpec,
)
from pdfs.generators.rendering.chart_renderer import ChartSpec, render_chart_to_pdf_page
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

# Model assignment: detailed_policy -> Sonnet, everything else -> DeepSeek
_MODEL_FOR_DOCTYPE: dict[str, str] = {
    "sbc": "deepseek-v4-pro",
    "formulary": "deepseek-v4-pro",
    "network_directory": "deepseek-v4-pro",
    "detailed_policy": "claude-sonnet-4-6",
    "member_handbook": "deepseek-v4-pro",
}

# Corpus size targets per provider
_CORPUS_SIZES: dict[str, tuple[int, int]] = {
    "profiling": (5, 7),  # 5-7 docs per provider -> 15-21 total
    "ground_truth": (13, 20),  # 13-20 docs per provider -> 39-60 total
}

# Modality distribution — proportion of PAGES across the corpus:
# (text_frac, table_chart_frac, scanned_frac)
_MODALITY_MIX: dict[str, tuple[float, float, float]] = {
    "profiling": (0.65, 0.20, 0.15),
    "ground_truth": (0.50, 0.30, 0.20),
}

# Structure quality thresholds (fraction of docs with poor structure)
_POOR_STRUCTURE_RATE: dict[str, float] = {
    "profiling": 0.10,
    "ground_truth": 0.30,
}

# Content type mapping for section spec generation
_DOCTYPE_CONTENT_TYPES: dict[str, str] = {
    "sbc": "table_heavy",
    "formulary": "table_heavy",
    "network_directory": "table_heavy",
    "detailed_policy": "text",
    "member_handbook": "mixed",
}

# Human-readable document type names for prompts
_DOCTYPE_LABELS: dict[str, str] = {
    "sbc": "Summary of Benefits and Coverage (SBC)",
    "formulary": "Prescription Drug Formulary",
    "network_directory": "Provider Network Directory",
    "detailed_policy": "Detailed Policy Document",
    "member_handbook": "Member Handbook",
}


# ---------------------------------------------------------------------------
# Section templates per document type
# ---------------------------------------------------------------------------


def _sbc_sections(pages: int, rng: random.Random) -> list[SectionSpec]:
    """Build section specs for an SBC document."""
    base = max(1, pages // 5)
    return [
        SectionSpec("Coverage Overview", base, "table_heavy"),
        SectionSpec("Deductibles and Out-of-Pocket Limits", base, "table_heavy"),
        SectionSpec("Common Medical Events", max(1, pages - 4 * base), "table_heavy"),
        SectionSpec("Excluded Services", base, "text"),
        SectionSpec("Grievance and Appeals Rights", base, "text"),
    ]


def _formulary_sections(pages: int, rng: random.Random) -> list[SectionSpec]:
    """Build section specs for a formulary document."""
    base = max(1, pages // 4)
    return [
        SectionSpec("Tier Structure and Cost Sharing", base, "table_heavy"),
        SectionSpec("Generic Medications (Tier 1-2)", max(1, pages - 3 * base), "table_heavy"),
        SectionSpec("Specialty Medications (Tier 3-4)", base, "table_heavy"),
        SectionSpec("Prior Authorization Requirements", base, "mixed"),
    ]


def _network_directory_sections(pages: int, rng: random.Random) -> list[SectionSpec]:
    """Build section specs for a network directory."""
    base = max(1, pages // 3)
    return [
        SectionSpec("Primary Care Physicians", base, "table_heavy"),
        SectionSpec("Specialists", max(1, pages - 2 * base), "table_heavy"),
        SectionSpec("Hospitals and Facilities", base, "table_heavy"),
    ]


def _detailed_policy_sections(pages: int, rng: random.Random) -> list[SectionSpec]:
    """Build section specs for a detailed policy document."""
    # Allocate roughly evenly across many sections
    n_sections = rng.randint(8, 12)
    base = max(1, pages // n_sections)
    remainder = pages - base * n_sections

    titles = [
        "Definitions and General Provisions",
        "Eligibility and Enrollment",
        "Covered Medical Services",
        "Preventive Care Benefits",
        "Mental Health and Substance Abuse",
        "Prescription Drug Coverage",
        "Emergency and Urgent Care",
        "Prior Authorization and Utilization Review",
        "Claims Processing and Payment",
        "Appeals and External Review",
        "Coordination of Benefits",
        "Termination and Continuation of Coverage",
    ]
    sections = []
    for i, title in enumerate(titles[:n_sections]):
        extra = 1 if i < remainder else 0
        sections.append(SectionSpec(title, base + extra, "text"))
    return sections


def _member_handbook_sections(pages: int, rng: random.Random) -> list[SectionSpec]:
    """Build section specs for a member handbook."""
    base = max(1, pages // 5)
    return [
        SectionSpec("Welcome and Plan Overview", base, "text"),
        SectionSpec("How Your Plan Works", max(1, pages - 4 * base), "mixed"),
        SectionSpec("Understanding Your Costs", base, "table_heavy"),
        SectionSpec("Getting Care", base, "mixed"),
        SectionSpec("Your Rights and Responsibilities", base, "text"),
    ]


_SECTION_BUILDERS: dict[str, Any] = {
    "sbc": _sbc_sections,
    "formulary": _formulary_sections,
    "network_directory": _network_directory_sections,
    "detailed_policy": _detailed_policy_sections,
    "member_handbook": _member_handbook_sections,
}


# ---------------------------------------------------------------------------
# Page-count resolution
# ---------------------------------------------------------------------------


def _pick_page_count(
    doc_type: str,
    profile: str,
    rng: random.Random,
) -> int:
    """Select page count from the doc type range, biased by profile.

    Profiling uses the lower end; ground truth uses the full range.
    """
    lo, hi = W14_DOC_TYPES[doc_type]
    if profile == "profiling":
        # Lower 40% of range
        upper = lo + max(1, int((hi - lo) * 0.4))
        return rng.randint(lo, upper)
    return rng.randint(lo, hi)


# ---------------------------------------------------------------------------
# Document plan
# ---------------------------------------------------------------------------


_doc_plan_counter = 0


@dataclass(slots=True)
class _DocPlan:
    """Internal plan for one document to generate."""

    provider: str
    doc_type: str
    page_count: int
    model: str
    structure_quality: str
    sections: list[SectionSpec]
    doc_id: str = ""

    def __post_init__(self) -> None:
        if not self.doc_id:
            global _doc_plan_counter  # noqa: PLW0603
            _doc_plan_counter += 1
            provider_slug = self.provider.lower().replace(" ", "_")
            self.doc_id = f"w14_{provider_slug}_{self.doc_type}_{_doc_plan_counter:03d}"


def _build_corpus_plan(
    profile: str,
    rng: random.Random,
) -> list[_DocPlan]:
    """Build the full list of documents to generate for all providers."""
    lo_per_provider, hi_per_provider = _CORPUS_SIZES[profile]
    poor_rate = _POOR_STRUCTURE_RATE[profile]
    doc_types = list(W14_DOC_TYPES.keys())

    plans: list[_DocPlan] = []

    for provider in PROVIDERS:
        target_count = rng.randint(lo_per_provider, hi_per_provider)

        provider_plans: list[_DocPlan] = []
        for dt in doc_types:
            pages = _pick_page_count(dt, profile, rng)
            is_poor = rng.random() < poor_rate
            sections = _SECTION_BUILDERS[dt](pages, rng)
            provider_plans.append(
                _DocPlan(
                    provider=provider,
                    doc_type=dt,
                    page_count=pages,
                    model=_MODEL_FOR_DOCTYPE[dt],
                    structure_quality="poorly_structured" if is_poor else "well_structured",
                    sections=sections,
                )
            )

        extras_needed = max(0, target_count - len(provider_plans))
        for _ in range(extras_needed):
            dt = rng.choice(doc_types)
            pages = _pick_page_count(dt, profile, rng)
            is_poor = rng.random() < poor_rate
            sections = _SECTION_BUILDERS[dt](pages, rng)
            provider_plans.append(
                _DocPlan(
                    provider=provider,
                    doc_type=dt,
                    page_count=pages,
                    model=_MODEL_FOR_DOCTYPE[dt],
                    structure_quality="poorly_structured" if is_poor else "well_structured",
                    sections=sections,
                )
            )

        plans.extend(provider_plans)

    return plans


# ---------------------------------------------------------------------------
# Per-page modality assignment
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
# Key-value injection
# ---------------------------------------------------------------------------


def _provider_key_values(provider: str) -> dict[str, str]:
    """Return the load-bearing key-value pairs for a provider."""
    return dict(W14_W15_PROVIDER_VALUES[provider])


def _distribute_key_values(
    key_values: dict[str, str],
    sections: list[SectionSpec],
) -> list[SectionSpec]:
    """Distribute provider key values across sections so each appears at least once.

    Assign values to whichever section is thematically closest, falling back to
    round-robin for unmatched keys.
    """
    # Thematic mapping: key substring -> section title substring
    affinity: dict[str, list[str]] = {
        "deductible": ["deductible", "cost", "overview", "how your plan"],
        "out_of_pocket": ["deductible", "cost", "overview", "out-of-pocket"],
        "copay": ["cost", "emergency", "common medical", "how your plan"],
        "mri": ["prior authorization", "utilization", "covered medical"],
        "mental_health": ["mental health", "covered medical", "how your plan"],
        "pre_existing": ["eligibility", "general provision", "overview"],
        "out_of_network": ["coverage", "how your plan", "cost"],
        "prescription": ["formulary", "prescription", "tier"],
    }

    updated: list[SectionSpec] = list(sections)
    assigned: set[str] = set()

    for kv_key, kv_val in key_values.items():
        placed = False
        kv_lower = kv_key.lower()

        # Try affinity match
        for pattern, section_hints in affinity.items():
            if pattern in kv_lower:
                for i, sec in enumerate(updated):
                    sec_lower = sec.title.lower()
                    if any(hint in sec_lower for hint in section_hints):
                        new_kvs = dict(sec.key_values)
                        new_kvs[kv_key] = kv_val
                        updated[i] = SectionSpec(
                            title=sec.title,
                            target_pages=sec.target_pages,
                            content_type=sec.content_type,
                            key_values=new_kvs,
                        )
                        placed = True
                        assigned.add(kv_key)
                        break
                if placed:
                    break

        # Fallback: round-robin across sections
        if not placed:
            idx = len(assigned) % len(updated)
            sec = updated[idx]
            new_kvs = dict(sec.key_values)
            new_kvs[kv_key] = kv_val
            updated[idx] = SectionSpec(
                title=sec.title,
                target_pages=sec.target_pages,
                content_type=sec.content_type,
                key_values=new_kvs,
            )
            assigned.add(kv_key)

    return updated


# ---------------------------------------------------------------------------
# Scan simulation for ground truth
# ---------------------------------------------------------------------------


def _should_apply_scanning(
    profile: str,
    rng: random.Random,
) -> bool:
    """Decide whether a PDF gets scanned pages (~35% profiling, ~40% GT)."""
    _, _, scan_frac = _MODALITY_MIX[profile]
    if scan_frac == 0:
        return False
    return rng.random() < 0.40


def _pick_scanned_pages(
    page_count: int,
    rng: random.Random,
) -> list[int]:
    """Select 30-50% of pages to rasterize as scanned."""
    scan_frac = rng.uniform(0.30, 0.50)
    n_scan = max(1, round(page_count * scan_frac))
    indices = list(range(page_count))
    rng.shuffle(indices)
    return sorted(indices[:n_scan])


# ---------------------------------------------------------------------------
# Chart generation for mixed-modality pages
# ---------------------------------------------------------------------------

_CHART_TEMPLATES: list[dict[str, Any]] = [
    {
        "chart_type": "bar",
        "title": "Annual Cost Comparison by Category",
        "data": {
            "In-Network": [250, 1500, 40, 200, 300],
            "Out-of-Network": [500, 3000, 80, 400, 600],
        },
        "x_labels": ["ER Copay", "Deductible", "PCP Visit", "Specialist", "Rx Tier 3"],
        "y_label": "Cost ($)",
    },
    {
        "chart_type": "pie",
        "title": "Premium Allocation Breakdown",
        "data": {"Allocation": [45, 25, 15, 10, 5]},
        "x_labels": ["Medical Services", "Pharmacy", "Administration", "Reserves", "Preventive"],
    },
    {
        "chart_type": "bar",
        "title": "Member Cost Sharing Summary",
        "data": {
            "Bronze": [6500, 8500, 50, 300],
            "Silver": [4000, 6500, 35, 200],
            "Gold": [1500, 3500, 20, 150],
        },
        "x_labels": ["Deductible", "OOP Max", "PCP Copay", "ER Copay"],
        "y_label": "Cost ($)",
    },
    {
        "chart_type": "line",
        "title": "Historical Premium Trends (2020-2025)",
        "data": {
            "Individual": [450, 475, 510, 540, 570, 600],
            "Family": [1200, 1260, 1340, 1420, 1500, 1580],
        },
        "x_labels": ["2020", "2021", "2022", "2023", "2024", "2025"],
        "y_label": "Monthly Premium ($)",
    },
]


def _embed_charts(
    pdf_path: Path,
    plan: _DocPlan,
    n_charts: int,
    rng: random.Random,
    output_dir: Path,
) -> int:
    """Generate and insert chart pages into an existing PDF.

    Returns the number of chart pages inserted.
    """
    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    templates = rng.sample(_CHART_TEMPLATES, min(n_charts, len(_CHART_TEMPLATES)))
    chart_pages: list[Path] = []

    for i, tmpl in enumerate(templates):
        spec = ChartSpec(**tmpl)
        chart_pdf = output_dir / f"_chart_{plan.doc_id}_{i}.pdf"
        render_chart_to_pdf_page(spec, chart_pdf)
        chart_pages.append(chart_pdf)

    if not chart_pages:
        return 0

    # Assemble: original pages + chart pages interleaved
    original_pages = count_pdf_pages(pdf_path)
    page_sources: list[PageSource] = []

    # Insert charts at evenly spaced positions
    insert_positions = set()
    if original_pages > 0:
        step = max(1, original_pages // (len(chart_pages) + 1))
        for ci in range(len(chart_pages)):
            insert_positions.add(min((ci + 1) * step, original_pages - 1))

    chart_idx = 0
    for page_idx in range(original_pages):
        page_sources.append(
            PageSource(
                source_type="text_pdf",
                source_path=pdf_path,
                source_page_index=page_idx,
            )
        )
        if page_idx in insert_positions and chart_idx < len(chart_pages):
            page_sources.append(
                PageSource(
                    source_type="table_pdf",
                    source_path=chart_pages[chart_idx],
                    source_page_index=0,
                )
            )
            chart_idx += 1

    assembled = output_dir / f"_assembled_{plan.doc_id}.pdf"
    assemble_pdf(page_sources, assembled)
    assembled.replace(pdf_path)

    # Clean up temp chart PDFs
    for cp in chart_pages:
        cp.unlink(missing_ok=True)

    return len(chart_pages)


# ---------------------------------------------------------------------------
# Single-document generation
# ---------------------------------------------------------------------------


async def _generate_one_document(
    plan: _DocPlan,
    output_dir: Path,
    profile: str,
    rng: random.Random,
    api_key: str | None,
) -> tuple[PDFDescriptor, ContentManifestEntry]:
    """Generate a single PDF and return its descriptor and manifest entry."""
    provider_kvs = _provider_key_values(plan.provider)
    enriched_sections = _distribute_key_values(provider_kvs, plan.sections)

    # Build section dicts for the LLM call
    section_dicts = [
        {
            "title": s.title,
            "target_pages": s.target_pages,
            "content_type": s.content_type,
            "key_values": dict(s.key_values),
        }
        for s in enriched_sections
    ]

    logger.info(
        "Generating %s for %s (%d pages, model=%s)",
        plan.doc_type,
        plan.provider,
        plan.page_count,
        plan.model,
    )

    # Generate content via LLM
    content = await generate_document_content(
        document_type=_DOCTYPE_LABELS.get(plan.doc_type, plan.doc_type),
        domain="health insurance",
        sections=section_dicts,
        target_pages=plan.page_count,
        model=plan.model,
        provider_name=plan.provider,
        key_values=provider_kvs,
        structure_quality=plan.structure_quality,
        api_key=api_key,
    )

    token_count = count_tokens(content)

    # Render the base PDF (text + embedded tables)
    pdf_filename = f"{plan.doc_id}.pdf"
    pdf_path = output_dir / pdf_filename
    render_markdown_to_pdf(content, pdf_path, layout=PageLayout())

    # Count actual pages
    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    actual_pages = count_pdf_pages(pdf_path)

    # Embed charts — ~20% of pages for profiling, ~30% for GT
    _, table_chart_frac, _ = _MODALITY_MIX[profile]
    chart_count = 0
    if table_chart_frac > 0 and actual_pages >= 4:
        n_charts = max(1, round(actual_pages * table_chart_frac * 0.5))
        _embed_charts(pdf_path, plan, n_charts, rng, output_dir)
        chart_count = n_charts
        actual_pages = count_pdf_pages(pdf_path)

    # Scanned pages — GT only, ~40% of GT docs, 30-50% of pages within those
    final_scanned_count = 0
    if _should_apply_scanning(profile, rng):
        pages_to_scan = _pick_scanned_pages(actual_pages, rng)
        if pages_to_scan:
            scanned_images = rasterize_pdf_pages(pdf_path, pages_to_scan, rng=rng)

            # Rebuild PDF with scanned pages replacing originals
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

            # Write to temp then replace
            assembled_path = output_dir / f"{plan.doc_id}_assembled.pdf"
            assemble_pdf(page_sources, assembled_path)
            assembled_path.replace(pdf_path)

    # Build descriptor
    descriptor = PDFDescriptor(
        pdf_id=plan.doc_id,
        workflow="w14_w15",
        profile=profile,
        document_type=plan.doc_type,
        page_count=actual_pages,
        estimated_token_count=token_count,
        text_pages=actual_pages - chart_count - final_scanned_count,
        table_chart_pages=chart_count,
        scanned_pages=final_scanned_count,
        section_count=len(enriched_sections),
        key_fields_present=list(provider_kvs.keys()),
        provider=plan.provider,
        structure_quality=plan.structure_quality,
        content_density="dense" if plan.doc_type == "detailed_policy" else "mixed",
        generation_model=plan.model,
    )

    # Write descriptor JSON
    write_descriptor(descriptor, output_dir / "descriptors")

    # Build manifest entry
    manifest_sections = _build_manifest_sections(enriched_sections, actual_pages, provider_kvs)
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


def _build_manifest_sections(
    sections: list[SectionSpec],
    total_pages: int,
    provider_kvs: dict[str, str],
) -> list[ContentManifestSection]:
    """Build manifest sections with estimated page ranges and key facts."""
    manifest_sections: list[ContentManifestSection] = []
    page_cursor = 1

    for sec in sections:
        end_page = min(page_cursor + sec.target_pages - 1, total_pages)

        # Key facts: combine section key_values with any provider values placed here
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


async def generate_w14_w15_corpus(
    output_dir: Path,
    profile: str,
    seed: int = 42,
    api_key: str | None = None,
) -> tuple[list[PDFDescriptor], ContentManifest]:
    """Generate W14/W15 base insurance corpus.

    Return (descriptors, manifest). The manifest indexes every key fact with
    its document ID, section title, and page range for downstream consumption
    by input generators.
    """
    if profile not in ("profiling", "ground_truth"):
        raise ValueError(f"profile must be 'profiling' or 'ground_truth', got {profile!r}")

    rng = random.Random(seed)  # noqa: S311
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build generation plan
    plans = _build_corpus_plan(profile, rng)
    logger.info(
        "Corpus plan: %d documents across %d providers (profile=%s)",
        len(plans),
        len(PROVIDERS),
        profile,
    )

    # Generate documents concurrently (10 parallel LLM calls).
    # Each task gets its own RNG derived from the master seed for determinism.
    tasks = []
    for i, plan in enumerate(plans):
        logger.info(
            "Queued document %d/%d: %s / %s",
            i + 1,
            len(plans),
            plan.provider,
            plan.doc_type,
        )
        doc_rng = random.Random(seed + i + 1)  # noqa: S311
        tasks.append(
            _generate_one_document(
                plan=plan,
                output_dir=output_dir,
                profile=profile,
                rng=doc_rng,
                api_key=api_key,
            )
        )

    results = await run_concurrent(tasks)
    descriptors = [r[0] for r in results]
    manifest_entries = [r[1] for r in results]

    # Build and save manifest
    corpus_id = f"w14_w15_{profile}_{seed}"
    manifest = ContentManifest(
        corpus_id=corpus_id,
        workflow="w14_w15",
        profile=profile,
        generated_at=datetime.now(UTC).isoformat(),
        documents=manifest_entries,
    )
    manifest.save(output_dir / "manifest.json")

    logger.info(
        "Corpus generation complete: %d PDFs, manifest saved to %s",
        len(descriptors),
        output_dir / "manifest.json",
    )

    return descriptors, manifest


def generate_w14_w15_corpus_sync(
    output_dir: Path,
    profile: str,
    seed: int = 42,
    api_key: str | None = None,
) -> tuple[list[PDFDescriptor], ContentManifest]:
    """Synchronous wrapper around generate_w14_w15_corpus."""
    return asyncio.run(
        generate_w14_w15_corpus(
            output_dir=output_dir,
            profile=profile,
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
        description="Generate W14/W15 base health insurance PDF corpus.",
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=["profiling", "ground_truth"],
        help="Distribution profile: profiling (small, clean) or ground_truth (large, messy).",
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
        default=None,
        help="Directory for generated PDFs and manifest.",
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
    output_dir = args.output_dir or Path(f"pdfs/generated/{args.profile}/w14_w15_corpus/pdfs")
    descriptors, manifest = generate_w14_w15_corpus_sync(
        output_dir=output_dir,
        profile=args.profile,
        seed=args.seed,
        api_key=args.api_key,
    )
    print(f"Generated {len(descriptors)} PDFs -> {output_dir}")
    print(f"Manifest: {output_dir / 'manifest.json'}")
