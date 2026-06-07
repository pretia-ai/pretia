"""Render table data to PDF pages (via reportlab) and HTML fragments (for WeasyPrint)."""

from __future__ import annotations

import html
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TableStyle:
    """Control visual appearance of rendered tables."""

    border_style: Literal["full", "header_only", "minimal"] = "full"
    header_bg_color: str = "#E8E8E8"
    font_size_pt: int = 9
    col_widths: tuple[float, ...] | None = None


@dataclass(frozen=True, slots=True)
class TableData:
    """Hold header + row content for a single table."""

    headers: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)
    caption: str | None = None


_DEFAULT_STYLE = TableStyle()


# ---------------------------------------------------------------------------
# PDF rendering (reportlab)
# ---------------------------------------------------------------------------

_LETTER_WIDTH = 612  # points (8.5 in)
_LETTER_HEIGHT = 792  # points (11 in)
_MARGIN = 72  # 1 inch


def _build_reportlab_table(
    table: TableData,
    style: TableStyle,
    available_width: float,
) -> list[Any]:
    """Construct reportlab flowables for a table.

    Return a list that optionally starts with a caption paragraph,
    followed by the Table object.
    """
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, Table
    from reportlab.platypus.tables import TableStyle as RLTableStyle

    styles = getSampleStyleSheet()

    # -- resolve column widths ------------------------------------------------
    n_cols = len(table.headers) if table.headers else (len(table.rows[0]) if table.rows else 0)
    if style.col_widths is not None:
        col_widths: list[float] | None = list(style.col_widths)
    elif n_cols > 0:
        col_widths = [available_width / n_cols] * n_cols
    else:
        col_widths = None

    # -- assemble data grid ---------------------------------------------------
    data: list[list[str]] = []
    if table.headers:
        data.append(table.headers)
    data.extend(table.rows)

    rl_table = Table(data, colWidths=col_widths)

    # -- style commands -------------------------------------------------------
    cmds: list[tuple[Any, ...]] = [
        ("FONTSIZE", (0, 0), (-1, -1), style.font_size_pt),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]

    header_row_idx = 0 if table.headers else -1

    # header background
    if table.headers:
        cmds.append(
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(style.header_bg_color)),
        )
        cmds.append(("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"))

    # borders
    if style.border_style == "full":
        cmds.append(("GRID", (0, 0), (-1, -1), 0.5, colors.black))
    elif style.border_style == "header_only":
        cmds.append(("BOX", (0, 0), (-1, -1), 0.5, colors.black))
        if header_row_idx >= 0:
            cmds.append(
                ("LINEBELOW", (0, header_row_idx), (-1, header_row_idx), 0.5, colors.black),
            )
    elif style.border_style == "minimal":
        if header_row_idx >= 0:
            cmds.append(
                ("LINEBELOW", (0, header_row_idx), (-1, header_row_idx), 0.5, colors.black),
            )

    rl_table.setStyle(RLTableStyle(cmds))

    # -- optional caption -----------------------------------------------------
    flowables: list[Any] = []
    if table.caption:
        caption_style = styles["Normal"].clone("caption")
        caption_style.fontSize = style.font_size_pt + 1
        caption_style.fontName = "Helvetica-Bold"
        caption_style.spaceAfter = 6
        flowables.append(Paragraph(table.caption, caption_style))

    flowables.append(rl_table)
    return flowables


def render_table_to_pdf(
    table: TableData,
    output_path: Path,
    style: TableStyle = _DEFAULT_STYLE,
) -> Path:
    """Render a single table to a standalone Letter-size PDF page.

    Return *output_path* after writing.
    """
    from reportlab.platypus import SimpleDocTemplate

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_width = _LETTER_WIDTH - 2 * _MARGIN

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(_LETTER_WIDTH, _LETTER_HEIGHT),
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    flowables = _build_reportlab_table(table, style, available_width)
    doc.build(flowables)

    log.info("Wrote table PDF to %s", output_path)
    return output_path


def render_tables_to_pages(
    tables: list[TableData],
    output_path: Path,
    style: TableStyle = _DEFAULT_STYLE,
) -> Path:
    """Render multiple tables as sequential pages in a single PDF.

    Each table starts on a new page. Return *output_path* after writing.
    """
    from reportlab.platypus import PageBreak, SimpleDocTemplate

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    available_width = _LETTER_WIDTH - 2 * _MARGIN

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(_LETTER_WIDTH, _LETTER_HEIGHT),
        leftMargin=_MARGIN,
        rightMargin=_MARGIN,
        topMargin=_MARGIN,
        bottomMargin=_MARGIN,
    )

    all_flowables: list[Any] = []
    for i, tbl in enumerate(tables):
        if i > 0:
            all_flowables.append(PageBreak())
        all_flowables.extend(_build_reportlab_table(tbl, style, available_width))

    doc.build(all_flowables)

    log.info("Wrote %d tables to %s", len(tables), output_path)
    return output_path


# ---------------------------------------------------------------------------
# HTML rendering (for WeasyPrint embedding)
# ---------------------------------------------------------------------------


def _border_css(style: TableStyle) -> tuple[str, str, str]:
    """Return ``(table_css, th_css, td_css)`` border declarations for the given style."""
    if style.border_style == "full":
        table_css = "border-collapse: collapse;"
        cell_css = "border: 1px solid #000;"
        th_css = cell_css
    elif style.border_style == "header_only":
        table_css = "border-collapse: collapse; border: 1px solid #000;"
        th_css = "border-bottom: 1px solid #000;"
        cell_css = "border: none;"
    else:  # minimal
        table_css = "border-collapse: collapse;"
        th_css = "border-bottom: 1px solid #000;"
        cell_css = "border: none;"
    return table_css, th_css, cell_css


def table_to_html(table: TableData, style: TableStyle = _DEFAULT_STYLE) -> str:
    """Convert *table* to an HTML ``<table>`` fragment with inline CSS.

    Suitable for embedding inside a WeasyPrint HTML document.
    """
    table_css, th_css, td_css = _border_css(style)

    parts: list[str] = []

    # optional caption
    if table.caption:
        parts.append(
            f'<p style="font-weight: bold; font-size: {style.font_size_pt + 1}pt; '
            f'margin-bottom: 4px;">{html.escape(table.caption)}</p>'
        )

    parts.append(f'<table style="{table_css} font-size: {style.font_size_pt}pt; width: 100%;">')

    # column widths
    if style.col_widths is not None and table.headers:
        total = sum(style.col_widths)
        parts.append("<colgroup>")
        for w in style.col_widths:
            pct = (w / total) * 100 if total else 0
            parts.append(f'  <col style="width: {pct:.1f}%;">')
        parts.append("</colgroup>")

    # header row
    if table.headers:
        parts.append("<thead>")
        parts.append("<tr>")
        for hdr in table.headers:
            parts.append(
                f'  <th style="{th_css} background-color: {style.header_bg_color}; '
                f'padding: 3px; text-align: left;">{html.escape(hdr)}</th>'
            )
        parts.append("</tr>")
        parts.append("</thead>")

    # body rows
    parts.append("<tbody>")
    for row in table.rows:
        parts.append("<tr>")
        for cell in row:
            parts.append(
                f'  <td style="{td_css} padding: 3px; '
                f'vertical-align: top;">{html.escape(cell)}</td>'
            )
        parts.append("</tr>")
    parts.append("</tbody>")

    parts.append("</table>")
    return "\n".join(parts)
