"""Generate W17 provider policy PDFs for insurance claims backtesting.

Produce 3 provider-specific policy documents (United Healthcare, Aetna, Cigna),
each 25-40 pages.  These are the same corpus used for both profiling and ground
truth -- cost drift comes from claims (inputs), not policies.

Model: claude-sonnet-4-6 (precision-critical).  Cross-provider values from
W17_PROVIDER_VALUES are embedded verbatim and verified post-generation.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import random
from pathlib import Path

from pdfs.generators._llm import count_tokens, generate_document_content
from pdfs.generators._types import W17_PROVIDER_VALUES, DocumentSpec, SectionSpec
from pdfs.generators.rendering.pdf_assembler import PDFDescriptor, write_descriptor
from pdfs.generators.rendering.text_renderer import PageLayout, render_markdown_to_pdf

logger = logging.getLogger(__name__)

_MODEL = "claude-sonnet-4-6"

# Eight required sections per policy document with target page ranges.
_POLICY_SECTIONS: list[tuple[str, int, int, str]] = [
    # (title, min_pages, max_pages, content_type)
    ("Coverage Overview", 2, 3, "text"),
    ("Covered Services", 5, 8, "mixed"),
    ("Exclusions", 3, 5, "text"),
    ("Prior Authorization Requirements", 3, 5, "mixed"),
    ("Cost Sharing", 2, 3, "table_heavy"),
    ("Claims Processing", 3, 5, "text"),
    ("Appeals Process", 2, 4, "text"),
    ("Definitions", 2, 3, "text"),
]

# Map W17_PROVIDER_VALUES keys to the sections where each value belongs.
_VALUE_SECTION_MAP: dict[str, list[str]] = {
    "mri_prior_auth": ["Prior Authorization Requirements", "Covered Services"],
    "pre_existing_exclusion": ["Exclusions", "Coverage Overview"],
    "appeal_deadline": ["Appeals Process"],
    "medical_necessity_standard": [
        "Prior Authorization Requirements",
        "Claims Processing",
    ],
    "er_definition": ["Coverage Overview", "Covered Services"],
    "experimental_exclusion": ["Exclusions"],
    "max_out_of_pocket": ["Cost Sharing", "Coverage Overview"],
}


def _build_section_specs(
    provider: str,
    seed: int,
) -> list[SectionSpec]:
    """Build the 8 SectionSpec objects for a provider policy document.

    Distribute provider-specific key values to the sections where they
    naturally belong so the LLM can embed them in the right context.
    """
    rng = random.Random(seed)  # noqa: S311 — deterministic seed for reproducibility, not crypto
    provider_values = W17_PROVIDER_VALUES[provider]

    specs: list[SectionSpec] = []
    for title, min_p, max_p, content_type in _POLICY_SECTIONS:
        target_pages = rng.randint(min_p, max_p)

        # Collect the key-value pairs that belong in this section.
        section_kv: dict[str, str] = {}
        for key, target_sections in _VALUE_SECTION_MAP.items():
            if title in target_sections and key in provider_values:
                section_kv[key] = provider_values[key]

        specs.append(
            SectionSpec(
                title=title,
                target_pages=target_pages,
                content_type=content_type,
                key_values=section_kv,
            )
        )
    return specs


def _build_document_spec(provider: str, seed: int) -> DocumentSpec:
    """Build a full DocumentSpec for one provider's policy document."""
    sections = _build_section_specs(provider, seed)
    total_pages = sum(s.target_pages for s in sections)

    return DocumentSpec(
        doc_id=f"w17-policy-{provider.lower().replace(' ', '-')}",
        workflow="w17",
        profile="profiling",
        document_type="insurance_policy",
        domain="health_insurance",
        sections=sections,
        target_page_count=total_pages,
        generation_model=_MODEL,
        provider=provider,
        structure_quality="well_structured",
    )


def _verify_provider_values(content: str, provider: str) -> list[str]:
    """Verify that all W17_PROVIDER_VALUES for a provider appear in the content.

    Return list of missing value keys (empty means all present).
    """
    provider_values = W17_PROVIDER_VALUES[provider]
    missing: list[str] = []
    for key, value in provider_values.items():
        if value not in content:
            missing.append(key)
            logger.warning(
                "Provider %s: required value %r (%s) not found in generated content",
                provider,
                value,
                key,
            )
    return missing


_KEY_SECTION_MAP: dict[str, str] = {
    "mri_prior_auth": "Prior Authorization Requirements",
    "pre_existing_exclusion": "Exclusions",
    "appeal_deadline": "Appeals Process",
    "medical_necessity_standard": "Covered Services",
    "er_definition": "Coverage Overview",
    "experimental_exclusion": "Exclusions",
    "max_out_of_pocket": "Cost Sharing",
}


def _patch_missing_values(content: str, provider: str, missing: list[str]) -> str:
    """Inject missing required values into the generated content.

    Append each missing value near the relevant section heading. If the heading
    isn't found, append to the end of the document.
    """
    provider_values = W17_PROVIDER_VALUES[provider]
    for key in missing:
        value = provider_values[key]
        target_section = _KEY_SECTION_MAP.get(key, "")
        label = key.replace("_", " ").title()
        snippet = f"\n\n**{label}:** {value}\n"

        if target_section and target_section in content:
            idx = content.index(target_section) + len(target_section)
            nl = content.find("\n", idx)
            if nl != -1:
                content = content[:nl] + snippet + content[nl:]
            else:
                content += snippet
        else:
            content += snippet

        logger.info("Patched missing value %s = %r for %s", key, value, provider)
    return content


async def _generate_single_policy(
    provider: str,
    output_dir: Path,
    seed: int,
    api_key: str | None,
) -> PDFDescriptor:
    """Generate one provider's policy document as a PDF with descriptor."""
    spec = _build_document_spec(provider, seed)

    # Build flat key_values dict with all provider values for the top-level prompt.
    provider_values = W17_PROVIDER_VALUES[provider]

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
        document_type="insurance_policy",
        domain="health_insurance",
        sections=sections_dicts,
        target_pages=spec.target_page_count,
        model=_MODEL,
        provider_name=provider,
        key_values=provider_values,
        structure_quality="well_structured",
        api_key=api_key,
    )

    # Verify all provider-specific values appear verbatim.
    missing = _verify_provider_values(content, provider)
    if missing:
        logger.warning(
            "Provider %s: %d required values missing, patching content: %s",
            provider,
            len(missing),
            missing,
        )
        content = _patch_missing_values(content, provider, missing)
        # Re-verify after patching.
        still_missing = _verify_provider_values(content, provider)
        if still_missing:
            logger.error(
                "Provider %s: %d values STILL missing after patching: %s",
                provider,
                len(still_missing),
                still_missing,
            )

    # Render to PDF.
    pdf_path = output_dir / f"{spec.doc_id}.pdf"
    layout = PageLayout(body_font_size_pt=10, line_spacing=1.2)
    render_markdown_to_pdf(content, pdf_path, layout=layout)

    # Count pages in the rendered PDF.
    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    page_count = count_pdf_pages(pdf_path)
    token_count = count_tokens(content)

    final_missing = _verify_provider_values(content, provider)
    present_keys = [k for k in provider_values if k not in final_missing]

    descriptor = PDFDescriptor(
        pdf_id=spec.doc_id,
        workflow="w17",
        profile="profiling",
        document_type="insurance_policy",
        page_count=page_count,
        estimated_token_count=token_count,
        text_pages=page_count,
        table_chart_pages=0,
        scanned_pages=0,
        section_count=len(spec.sections),
        key_fields_present=present_keys,
        provider=provider,
        structure_quality="well_structured",
        content_density="dense",
        generation_model=_MODEL,
    )
    write_descriptor(descriptor, output_dir)

    logger.info(
        "Generated %s policy: %d pages, %d tokens, %d/%d values present",
        provider,
        page_count,
        token_count,
        len(present_keys),
        len(provider_values),
    )
    return descriptor


async def generate_w17_policies(
    output_dir: Path,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Generate all 3 W17 provider policy PDFs."""
    output_dir.mkdir(parents=True, exist_ok=True)
    providers = list(W17_PROVIDER_VALUES.keys())

    descriptors: list[PDFDescriptor] = []
    for i, provider in enumerate(providers):
        # Offset seed per provider for deterministic but distinct content.
        provider_seed = seed + i
        descriptor = await _generate_single_policy(
            provider=provider,
            output_dir=output_dir,
            seed=provider_seed,
            api_key=api_key,
        )
        descriptors.append(descriptor)

    logger.info(
        "Generated %d W17 policy documents in %s",
        len(descriptors),
        output_dir,
    )
    return descriptors


def generate_w17_policies_sync(
    output_dir: Path,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Synchronous wrapper."""
    return asyncio.run(generate_w17_policies(output_dir, seed=seed, api_key=api_key))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate W17 insurance policy PDFs for backtesting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation (default: 42).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pdfs/generated/profiling/w17/policies/",
        help="Output directory for generated PDFs.",
    )
    args = parser.parse_args()

    results = generate_w17_policies_sync(Path(args.output_dir), seed=args.seed)
    for desc in results:
        print(  # noqa: T201
            f"  {desc.provider}: {desc.pdf_id} "
            f"({desc.page_count} pages, {desc.estimated_token_count} tokens)"
        )
