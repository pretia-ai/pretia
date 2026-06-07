"""Shared rendering utilities for PDF generation."""

from __future__ import annotations

from pdfs.generators.rendering.chart_renderer import (
    ChartSpec,
    render_chart_to_pdf_page,
    render_chart_to_png,
)
from pdfs.generators.rendering.pdf_assembler import (
    PageSource,
    PDFDescriptor,
    assemble_pdf,
    write_descriptor,
)
from pdfs.generators.rendering.scan_simulator import (
    ScanParams,
    randomize_scan_params,
    rasterize_pdf_pages,
)
from pdfs.generators.rendering.table_renderer import (
    TableData,
    TableStyle,
    render_table_to_pdf,
    table_to_html,
)
from pdfs.generators.rendering.text_renderer import (
    PageLayout,
    render_html_to_pdf,
    render_markdown_to_pdf,
)

__all__ = [
    "ChartSpec",
    "PDFDescriptor",
    "PageLayout",
    "PageSource",
    "ScanParams",
    "TableData",
    "TableStyle",
    "assemble_pdf",
    "randomize_scan_params",
    "rasterize_pdf_pages",
    "render_chart_to_pdf_page",
    "render_chart_to_png",
    "render_html_to_pdf",
    "render_markdown_to_pdf",
    "render_table_to_pdf",
    "table_to_html",
    "write_descriptor",
]
