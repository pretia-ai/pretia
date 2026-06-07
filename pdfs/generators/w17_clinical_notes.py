"""Generate W17 clinical notes and supporting documentation for claims backtesting.

Produce 10 clinical note PDFs (1-3 pages each) with a controlled distribution:
  - 3 straightforward (clear diagnosis, clear procedure, complete docs)
  - 3 ambiguous (vague diagnosis, missing prior treatment history)
  - 2 conflicting (diagnosis suggests X but procedure is for Y)
  - 2 incomplete (missing sections, partial notes)

Also produce 5 supporting documents:
  - 3 itemized bills
  - 2 prior authorization forms

Model: deepseek-v4-pro for all documents.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from pdfs.generators._llm import count_tokens, generate_content
from pdfs.generators.rendering.pdf_assembler import PDFDescriptor, write_descriptor
from pdfs.generators.rendering.text_renderer import PageLayout, render_markdown_to_pdf

logger = logging.getLogger(__name__)

_MODEL = "deepseek-v4-pro"


@dataclass(frozen=True, slots=True)
class ClinicalNoteProfile:
    """Define the characteristics of a clinical note for generation."""

    note_id: str
    category: str  # "straightforward" | "ambiguous" | "conflicting" | "incomplete"
    chief_complaint: str
    diagnosis_code: str  # ICD-10
    procedure_code: str  # CPT
    target_pages: int
    special_instructions: str


# 10 clinical note profiles covering the required distribution.
_CLINICAL_NOTE_PROFILES: list[ClinicalNoteProfile] = [
    # --- 3 straightforward ---
    ClinicalNoteProfile(
        note_id="cn-01",
        category="straightforward",
        chief_complaint="Chronic lower back pain with radiculopathy",
        diagnosis_code="M54.5",
        procedure_code="72148",
        target_pages=2,
        special_instructions=(
            "Clear diagnosis and procedure match. Complete documentation with full "
            "history, physical exam findings, and treatment plan. Prior conservative "
            "treatment (PT, NSAIDs) documented over 6 weeks."
        ),
    ),
    ClinicalNoteProfile(
        note_id="cn-02",
        category="straightforward",
        chief_complaint="Right knee osteoarthritis, bone-on-bone",
        diagnosis_code="M17.11",
        procedure_code="27447",
        target_pages=3,
        special_instructions=(
            "Textbook knee replacement candidacy. Complete imaging results, failed "
            "conservative treatment documented (injections, PT, bracing). All sections "
            "filled, clear medical necessity."
        ),
    ),
    ClinicalNoteProfile(
        note_id="cn-03",
        category="straightforward",
        chief_complaint="Type 2 diabetes with peripheral neuropathy",
        diagnosis_code="E11.42",
        procedure_code="95907",
        target_pages=2,
        special_instructions=(
            "Routine nerve conduction study referral. Well-documented progression of "
            "neuropathy symptoms, medication history, HbA1c trends. Complete and clear."
        ),
    ),
    # --- 3 ambiguous ---
    ClinicalNoteProfile(
        note_id="cn-04",
        category="ambiguous",
        chief_complaint="Persistent headaches and dizziness",
        diagnosis_code="R51.9",
        procedure_code="70553",
        target_pages=2,
        special_instructions=(
            "Vague chief complaint using non-specific symptom codes. Diagnosis is "
            "'unspecified headache' -- could be migraine, tension, or secondary. No "
            "prior treatment history documented. MRI brain ordered but clinical "
            "justification is thin."
        ),
    ),
    ClinicalNoteProfile(
        note_id="cn-05",
        category="ambiguous",
        chief_complaint="Generalized abdominal discomfort",
        diagnosis_code="R10.9",
        procedure_code="74178",
        target_pages=1,
        special_instructions=(
            "Unspecified abdominal pain. CT abdomen/pelvis ordered. Missing prior "
            "treatment attempts. Physical exam findings are vague ('mild diffuse "
            "tenderness'). No red flag symptoms documented to justify imaging."
        ),
    ),
    ClinicalNoteProfile(
        note_id="cn-06",
        category="ambiguous",
        chief_complaint="Fatigue and general malaise, 3 months",
        diagnosis_code="R53.83",
        procedure_code="80053",
        target_pages=2,
        special_instructions=(
            "Non-specific fatigue workup. Comprehensive metabolic panel ordered. "
            "History of present illness is vague and rambling. Missing family history "
            "and social history sections. No prior labs documented."
        ),
    ),
    # --- 2 conflicting ---
    ClinicalNoteProfile(
        note_id="cn-07",
        category="conflicting",
        chief_complaint="Left shoulder pain after fall",
        diagnosis_code="M75.12",
        procedure_code="29827",
        target_pages=2,
        special_instructions=(
            "Diagnosis says rotator cuff tear (left) but the procedure code is for "
            "arthroscopic rotator cuff repair of the RIGHT shoulder. Physical exam "
            "describes left shoulder findings but surgical consent references right. "
            "Create a realistic documentation error."
        ),
    ),
    ClinicalNoteProfile(
        note_id="cn-08",
        category="conflicting",
        chief_complaint="Chest pain with exertion",
        diagnosis_code="I25.10",
        procedure_code="43239",
        target_pages=3,
        special_instructions=(
            "Diagnosis is atherosclerotic heart disease but the procedure code is "
            "for upper GI endoscopy with biopsy. Assessment mentions cardiac workup "
            "but the plan section orders gastroenterology procedure. Include both "
            "cardiac and GI complaints in the HPI to make the conflict realistic."
        ),
    ),
    # --- 2 incomplete ---
    ClinicalNoteProfile(
        note_id="cn-09",
        category="incomplete",
        chief_complaint="Wrist fracture follow-up",
        diagnosis_code="S52.501A",
        procedure_code="25600",
        target_pages=1,
        special_instructions=(
            "Partial clinical note. Missing physical examination section entirely. "
            "Assessment is one line. No imaging results documented despite referencing "
            "'X-ray shows...' in the plan. Missing provider signature line."
        ),
    ),
    ClinicalNoteProfile(
        note_id="cn-10",
        category="incomplete",
        chief_complaint="Post-surgical follow-up, hip replacement",
        diagnosis_code="Z96.641",
        procedure_code="99214",
        target_pages=1,
        special_instructions=(
            "Sparse follow-up note. Only chief complaint and a brief assessment "
            "present. Missing HPI, physical exam, and plan sections. Date of surgery "
            "referenced but not documented. Reads like a half-finished note."
        ),
    ),
]


@dataclass(frozen=True, slots=True)
class SupportingDocProfile:
    """Define the characteristics of a supporting document for generation."""

    doc_id: str
    doc_type: str  # "itemized_bill" | "prior_auth_form"
    target_pages: int
    special_instructions: str


_SUPPORTING_DOC_PROFILES: list[SupportingDocProfile] = [
    # --- 3 itemized bills ---
    SupportingDocProfile(
        doc_id="sd-bill-01",
        doc_type="itemized_bill",
        target_pages=2,
        special_instructions=(
            "Itemized hospital bill for knee replacement surgery. Include facility "
            "charges, surgeon fees, anesthesia, implant costs, post-op room charges, "
            "pharmacy charges. Use realistic CPT codes and dollar amounts. Total "
            "should be $45,000-$65,000 range."
        ),
    ),
    SupportingDocProfile(
        doc_id="sd-bill-02",
        doc_type="itemized_bill",
        target_pages=1,
        special_instructions=(
            "Outpatient imaging bill for MRI of lumbar spine. Include facility fee, "
            "professional fee (radiologist interpretation), and contrast if applicable. "
            "Total in $1,500-$3,500 range. Use CPT 72148."
        ),
    ),
    SupportingDocProfile(
        doc_id="sd-bill-03",
        doc_type="itemized_bill",
        target_pages=1,
        special_instructions=(
            "Emergency department visit bill. Include ED facility fee (99285), "
            "physician charge, labs (CBC, BMP), CT scan, IV fluids. Total in "
            "$5,000-$12,000 range."
        ),
    ),
    # --- 2 prior authorization forms ---
    SupportingDocProfile(
        doc_id="sd-auth-01",
        doc_type="prior_auth_form",
        target_pages=2,
        special_instructions=(
            "Prior authorization request form for MRI lumbar spine. Include patient "
            "demographics, referring physician info, diagnosis (M54.5), procedure "
            "(72148), clinical justification section, prior conservative treatment "
            "documented. Include checkbox-style fields rendered as text."
        ),
    ),
    SupportingDocProfile(
        doc_id="sd-auth-02",
        doc_type="prior_auth_form",
        target_pages=2,
        special_instructions=(
            "Prior authorization request for elective knee replacement. Include "
            "patient demographics, diagnosis (M17.11), procedure (27447), failed "
            "conservative treatment summary, imaging results summary. Insurance "
            "plan information fields."
        ),
    ),
]


def _build_clinical_note_prompt(profile: ClinicalNoteProfile) -> tuple[str, str]:
    """Build system and user prompts for a clinical note.

    Return (system_prompt, user_prompt).
    """
    system_prompt = (
        "You are a physician writing clinical documentation. Generate realistic "
        "clinical notes that follow standard medical documentation practices. "
        "Use proper medical terminology, real ICD-10 and CPT code formats, and "
        "standard SOAP note structure where appropriate.\n\n"
        "Requirements:\n"
        "- Use markdown formatting with clear section headers.\n"
        "- Include realistic but fictional patient demographics.\n"
        "- All medical content should be clinically coherent within the note's "
        "intended characteristics.\n"
        "- Do NOT include meta-commentary about the document."
    )

    target_tokens = profile.target_pages * 800

    user_prompt = (
        f"Write a {profile.target_pages}-page clinical note "
        f"(approximately {target_tokens} tokens).\n\n"
        f"Category: {profile.category}\n"
        f"Chief complaint: {profile.chief_complaint}\n"
        f"Diagnosis code (ICD-10): {profile.diagnosis_code}\n"
        f"Procedure code (CPT): {profile.procedure_code}\n\n"
        f"Special characteristics:\n{profile.special_instructions}\n\n"
        "Required sections (include all unless the special instructions say to omit):\n"
        "1. Patient Demographics (name, DOB, MRN, date of visit)\n"
        "2. Chief Complaint\n"
        "3. History of Present Illness (HPI)\n"
        "4. Past Medical History\n"
        "5. Physical Examination\n"
        "6. Assessment and Plan\n"
        "7. Prior Treatment History\n"
        "8. Diagnosis Codes and Procedure Codes\n\n"
        f"The ICD-10 code {profile.diagnosis_code} and CPT code "
        f"{profile.procedure_code} MUST appear explicitly in the note.\n\n"
        "Output the clinical note in markdown."
    )
    return system_prompt, user_prompt


def _build_supporting_doc_prompt(profile: SupportingDocProfile) -> tuple[str, str]:
    """Build system and user prompts for a supporting document.

    Return (system_prompt, user_prompt).
    """
    if profile.doc_type == "itemized_bill":
        system_prompt = (
            "You are generating a realistic itemized medical bill. Use standard "
            "healthcare billing formats with CPT codes, HCPCS codes, revenue codes, "
            "and realistic dollar amounts. Format as a structured document with "
            "clear line items.\n\n"
            "Requirements:\n"
            "- Line items must use real CPT/HCPCS code formats.\n"
            "- Dollar amounts must be realistic for US healthcare.\n"
            "- Totals must equal the sum of line items.\n"
            "- Use markdown with tables for line items.\n"
            "- Do NOT include meta-commentary."
        )
    else:
        system_prompt = (
            "You are generating a realistic prior authorization request form for "
            "a health insurance company. Include standard fields: patient info, "
            "provider info, diagnosis, requested procedure, clinical justification, "
            "and supporting documentation references.\n\n"
            "Requirements:\n"
            "- Use real ICD-10 and CPT code formats.\n"
            "- Include checkbox-style fields rendered as text (e.g., '[X] Urgent').\n"
            "- Clinical justification must reference specific findings.\n"
            "- Use markdown formatting.\n"
            "- Do NOT include meta-commentary."
        )

    target_tokens = profile.target_pages * 800
    user_prompt = (
        f"Generate a {profile.target_pages}-page {profile.doc_type.replace('_', ' ')} "
        f"(approximately {target_tokens} tokens).\n\n"
        f"{profile.special_instructions}\n\n"
        "Output in markdown."
    )
    return system_prompt, user_prompt


async def _generate_single_clinical_note(
    profile: ClinicalNoteProfile,
    output_dir: Path,
    api_key: str | None,
) -> PDFDescriptor:
    """Generate one clinical note PDF with descriptor."""
    doc_id = f"w17-{profile.note_id}"
    system_prompt, user_prompt = _build_clinical_note_prompt(profile)

    target_tokens = profile.target_pages * 800
    content = await generate_content(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=_MODEL,
        max_tokens=min(target_tokens * 2, 8192),
        api_key=api_key,
    )

    pdf_path = output_dir / f"{doc_id}.pdf"
    layout = PageLayout(body_font_size_pt=10, line_spacing=1.2)
    render_markdown_to_pdf(content, pdf_path, layout=layout)

    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    page_count = count_pdf_pages(pdf_path)
    token_count = count_tokens(content)

    descriptor = PDFDescriptor(
        pdf_id=doc_id,
        workflow="w17",
        profile="profiling",
        document_type="clinical_note",
        page_count=page_count,
        estimated_token_count=token_count,
        text_pages=page_count,
        table_chart_pages=0,
        scanned_pages=0,
        section_count=8,
        key_fields_present=[profile.diagnosis_code, profile.procedure_code],
        provider=None,
        structure_quality=(
            "poorly_structured" if profile.category == "incomplete" else "well_structured"
        ),
        content_density="sparse" if profile.category == "incomplete" else "dense",
        generation_model=_MODEL,
    )
    write_descriptor(descriptor, output_dir)

    logger.info(
        "Generated clinical note %s (%s): %d pages, %d tokens",
        doc_id,
        profile.category,
        page_count,
        token_count,
    )
    return descriptor


async def _generate_single_supporting_doc(
    profile: SupportingDocProfile,
    output_dir: Path,
    api_key: str | None,
) -> PDFDescriptor:
    """Generate one supporting document PDF with descriptor."""
    doc_id = f"w17-{profile.doc_id}"
    system_prompt, user_prompt = _build_supporting_doc_prompt(profile)

    target_tokens = profile.target_pages * 800
    content = await generate_content(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=_MODEL,
        max_tokens=min(target_tokens * 2, 8192),
        api_key=api_key,
    )

    pdf_path = output_dir / f"{doc_id}.pdf"
    layout = PageLayout(body_font_size_pt=10, line_spacing=1.15)
    render_markdown_to_pdf(content, pdf_path, layout=layout)

    from pdfs.generators.rendering.text_renderer import count_pdf_pages

    page_count = count_pdf_pages(pdf_path)
    token_count = count_tokens(content)

    descriptor = PDFDescriptor(
        pdf_id=doc_id,
        workflow="w17",
        profile="profiling",
        document_type=profile.doc_type,
        page_count=page_count,
        estimated_token_count=token_count,
        text_pages=page_count,
        table_chart_pages=0,
        scanned_pages=0,
        section_count=1,
        key_fields_present=[],
        provider=None,
        structure_quality="well_structured",
        content_density="dense" if profile.doc_type == "itemized_bill" else "mixed",
        generation_model=_MODEL,
    )
    write_descriptor(descriptor, output_dir)

    logger.info(
        "Generated supporting doc %s (%s): %d pages, %d tokens",
        doc_id,
        profile.doc_type,
        page_count,
        token_count,
    )
    return descriptor


async def generate_w17_clinical_notes(
    output_dir: Path,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Generate 10 clinical note PDFs with controlled distribution.

    Distribution: 3 straightforward, 3 ambiguous, 2 conflicting, 2 incomplete.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptors: list[PDFDescriptor] = []
    for profile in _CLINICAL_NOTE_PROFILES:
        descriptor = await _generate_single_clinical_note(
            profile=profile,
            output_dir=output_dir,
            api_key=api_key,
        )
        descriptors.append(descriptor)

    logger.info(
        "Generated %d W17 clinical notes in %s",
        len(descriptors),
        output_dir,
    )
    return descriptors


async def generate_w17_supporting_docs(
    output_dir: Path,
    seed: int = 42,
    api_key: str | None = None,
) -> list[PDFDescriptor]:
    """Generate 5 supporting document PDFs (3 bills, 2 prior auth forms)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    descriptors: list[PDFDescriptor] = []
    for profile in _SUPPORTING_DOC_PROFILES:
        descriptor = await _generate_single_supporting_doc(
            profile=profile,
            output_dir=output_dir,
            api_key=api_key,
        )
        descriptors.append(descriptor)

    logger.info(
        "Generated %d W17 supporting docs in %s",
        len(descriptors),
        output_dir,
    )
    return descriptors


async def _generate_all(
    output_dir: Path,
    seed: int,
    api_key: str | None,
) -> tuple[list[PDFDescriptor], list[PDFDescriptor]]:
    """Generate both clinical notes and supporting docs."""
    notes_dir = output_dir / "clinical_notes"
    support_dir = output_dir / "supporting_docs"

    notes = await generate_w17_clinical_notes(notes_dir, seed=seed, api_key=api_key)
    supporting = await generate_w17_supporting_docs(support_dir, seed=seed, api_key=api_key)
    return notes, supporting


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate W17 clinical notes and supporting docs for backtesting.",
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
        default="pdfs/generated/profiling/w17/",
        help="Base output directory for generated PDFs.",
    )
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Generate clinical notes only (skip supporting docs).",
    )
    parser.add_argument(
        "--supporting-only",
        action="store_true",
        help="Generate supporting docs only (skip clinical notes).",
    )
    args = parser.parse_args()

    base_dir = Path(args.output_dir)

    if args.notes_only:
        results = asyncio.run(
            generate_w17_clinical_notes(base_dir / "clinical_notes", seed=args.seed)
        )
        for desc in results:
            print(  # noqa: T201
                f"  {desc.pdf_id} ({desc.page_count} pages, {desc.estimated_token_count} tokens)"
            )
    elif args.supporting_only:
        results = asyncio.run(
            generate_w17_supporting_docs(base_dir / "supporting_docs", seed=args.seed)
        )
        for desc in results:
            print(  # noqa: T201
                f"  {desc.pdf_id} ({desc.page_count} pages, {desc.estimated_token_count} tokens)"
            )
    else:
        notes, supporting = asyncio.run(_generate_all(base_dir, seed=args.seed, api_key=None))
        print(f"\nClinical notes ({len(notes)}):")  # noqa: T201
        for desc in notes:
            print(  # noqa: T201
                f"  {desc.pdf_id} ({desc.page_count} pages, {desc.estimated_token_count} tokens)"
            )
        print(f"\nSupporting docs ({len(supporting)}):")  # noqa: T201
        for desc in supporting:
            print(  # noqa: T201
                f"  {desc.pdf_id} ({desc.page_count} pages, {desc.estimated_token_count} tokens)"
            )
