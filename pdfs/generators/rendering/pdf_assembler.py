"""Combine pages from different sources into a single PDF and manage descriptors."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_VALID_SOURCE_TYPES = frozenset({"text_pdf", "table_pdf", "chart_image", "scanned_image"})
_PDF_SOURCE_TYPES = frozenset({"text_pdf", "table_pdf"})
_IMAGE_SOURCE_TYPES = frozenset({"chart_image", "scanned_image"})

# US Letter dimensions in points (72 points per inch)
_LETTER_WIDTH = 612
_LETTER_HEIGHT = 792


@dataclass(slots=True)
class PageSource:
    """Describe one page to include in an assembled PDF.

    Use source_path for file-based sources or pil_image for in-memory PIL Images.
    """

    source_type: str
    source_path: Path | None = None
    source_page_index: int = 0
    pil_image: Any | None = None  # PIL.Image.Image, typed as Any to avoid hard dep

    def __post_init__(self) -> None:
        """Validate source_type and ensure at least one source is provided."""
        if self.source_type not in _VALID_SOURCE_TYPES:
            msg = (
                f"Invalid source_type {self.source_type!r}, "
                f"expected one of {sorted(_VALID_SOURCE_TYPES)}"
            )
            raise ValueError(msg)
        if self.source_type in _PDF_SOURCE_TYPES and self.source_path is None:
            msg = f"source_path is required for source_type {self.source_type!r}"
            raise ValueError(msg)
        if (
            self.source_type in _IMAGE_SOURCE_TYPES
            and self.source_path is None
            and self.pil_image is None
        ):
            msg = (
                f"Either source_path or pil_image is required for source_type {self.source_type!r}"
            )
            raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class PDFDescriptor:
    """Metadata describing a generated PDF for traceability and validation."""

    pdf_id: str
    workflow: str
    profile: str  # "profiling" | "ground_truth"
    document_type: str
    page_count: int
    estimated_token_count: int
    text_pages: int
    table_chart_pages: int
    scanned_pages: int
    section_count: int
    key_fields_present: list[str] = field(default_factory=list)
    provider: str | None = None  # W17-specific
    structure_quality: str | None = None  # "well_structured" | "partially_structured" | ...
    content_density: str | None = None  # "dense" | "sparse" | "mixed"
    generation_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pdf_id": self.pdf_id,
            "workflow": self.workflow,
            "profile": self.profile,
            "document_type": self.document_type,
            "page_count": self.page_count,
            "estimated_token_count": self.estimated_token_count,
            "text_pages": self.text_pages,
            "table_chart_pages": self.table_chart_pages,
            "scanned_pages": self.scanned_pages,
            "section_count": self.section_count,
            "key_fields_present": list(self.key_fields_present),
            "provider": self.provider,
            "structure_quality": self.structure_quality,
            "content_density": self.content_density,
            "generation_model": self.generation_model,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PDFDescriptor:
        """Deserialize from a dict produced by `to_dict()`."""
        return cls(
            pdf_id=data["pdf_id"],
            workflow=data["workflow"],
            profile=data["profile"],
            document_type=data["document_type"],
            page_count=data["page_count"],
            estimated_token_count=data["estimated_token_count"],
            text_pages=data["text_pages"],
            table_chart_pages=data["table_chart_pages"],
            scanned_pages=data["scanned_pages"],
            section_count=data["section_count"],
            key_fields_present=list(data.get("key_fields_present", [])),
            provider=data.get("provider"),
            structure_quality=data.get("structure_quality"),
            content_density=data.get("content_density"),
            generation_model=data.get("generation_model"),
        )


def assemble_pdf(pages: list[PageSource], output_path: Path) -> Path:
    """Assemble multiple page sources into a single PDF at output_path.

    Handles four source types:
    - text_pdf / table_pdf: extract a specific page from an existing PDF via pypdf.
    - chart_image / scanned_image: convert a PIL Image or image file to a full-page
      PDF (US Letter) via reportlab, then merge.

    Return the output path on success.
    """
    from pypdf import PdfWriter

    writer = PdfWriter()

    for i, page in enumerate(pages):
        if page.source_type in _PDF_SOURCE_TYPES:
            _append_pdf_page(writer, page, page_index=i)
        else:
            _append_image_page(writer, page, page_index=i)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("wb") as f:
        writer.write(f)

    logger.info("Assembled %d pages into %s", len(pages), output_path)
    return output_path


def _append_pdf_page(writer: Any, page: PageSource, *, page_index: int) -> None:
    """Extract a single page from a source PDF and append to the writer."""
    from pypdf import PdfReader

    reader = PdfReader(str(page.source_path))
    total = len(reader.pages)
    idx = page.source_page_index

    if idx >= total:
        logger.warning(
            "Page %d: source_page_index %d exceeds page count %d in %s, clamping to last page",
            page_index,
            idx,
            total,
            page.source_path,
        )
        idx = total - 1

    writer.add_page(reader.pages[idx])
    logger.debug(
        "Page %d: extracted page %d from %s (%s)",
        page_index,
        idx,
        page.source_path,
        page.source_type,
    )


def _append_image_page(writer: Any, page: PageSource, *, page_index: int) -> None:
    """Convert an image to a letter-sized PDF page and append to the writer."""
    import io

    from PIL import Image
    from pypdf import PdfReader
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    # Resolve the PIL image
    if page.pil_image is not None:
        img = page.pil_image
        source_desc = "in-memory image"
    else:
        img = Image.open(page.source_path)
        source_desc = str(page.source_path)

    # Convert to RGB if needed (e.g. RGBA or palette mode)
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")

    # Scale image to fit letter page with margins
    margin = 0.5 * inch
    max_w = _LETTER_WIDTH - 2 * margin
    max_h = _LETTER_HEIGHT - 2 * margin
    img_w, img_h = img.size
    scale = min(max_w / img_w, max_h / img_h)
    draw_w = img_w * scale
    draw_h = img_h * scale

    # Center on the page
    x = (_LETTER_WIDTH - draw_w) / 2
    y = (_LETTER_HEIGHT - draw_h) / 2

    # Render via reportlab into an in-memory buffer
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=(_LETTER_WIDTH, _LETTER_HEIGHT))

    # Save image to a temporary bytes buffer for reportlab
    img_buf = io.BytesIO()
    img.save(img_buf, format="PNG")
    img_buf.seek(0)

    from reportlab.lib.utils import ImageReader

    c.drawImage(ImageReader(img_buf), x, y, width=draw_w, height=draw_h)
    c.showPage()
    c.save()

    # Merge the single-page PDF into the writer
    buf.seek(0)
    reader = PdfReader(buf)
    writer.add_page(reader.pages[0])

    logger.debug(
        "Page %d: rendered %s as image page (%s)",
        page_index,
        source_desc,
        page.source_type,
    )


def write_descriptor(descriptor: PDFDescriptor, output_dir: Path) -> Path:
    """Write a PDFDescriptor as {pdf_id}.json in output_dir.

    Return the path to the written JSON file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{descriptor.pdf_id}.json"
    json_path.write_text(json.dumps(descriptor.to_dict(), indent=2))
    logger.info("Wrote descriptor %s", json_path)
    return json_path


def load_descriptor(json_path: Path) -> PDFDescriptor:
    """Load a PDFDescriptor from a JSON file."""
    data = json.loads(json_path.read_text())
    logger.debug("Loaded descriptor from %s", json_path)
    return PDFDescriptor.from_dict(data)
