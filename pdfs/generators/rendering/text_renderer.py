"""Convert structured text content (markdown) to multi-page PDFs via reportlab."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PageLayout:
    """Define page geometry and typography for PDF rendering."""

    page_size: tuple[float, float] = (8.5, 11.0)
    margin_inches: float = 1.0
    body_font: str = "Times-Roman"
    body_font_size_pt: int = 11
    heading_font: str = "Helvetica-Bold"
    heading_font_size_pt: int = 13
    line_spacing: float = 1.15
    paragraph_spacing_pt: int = 6
    page_numbers: bool = True


DEFAULT_LAYOUT = PageLayout()

# ---------------------------------------------------------------------------
# Markdown -> HTML (kept for table_to_html embedding and tests)
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_HR_RE = re.compile(r"^-{3,}$", re.MULTILINE)
_UL_ITEM_RE = re.compile(r"^[ \t]*[-*+]\s+(.+)$")
_OL_ITEM_RE = re.compile(r"^[ \t]*\d+\.\s+(.+)$")


def _inline_formatting(text: str) -> str:
    """Apply bold and italic transformations to inline text."""
    text = _BOLD_RE.sub(r"<strong>\1</strong>", text)
    text = _ITALIC_RE.sub(r"<em>\1</em>", text)
    return text


def _convert_markdown_body(markdown_text: str) -> str:
    """Transform markdown source into HTML body content."""
    lines = markdown_text.split("\n")
    html_parts: list[str] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        if _HR_RE.match(line.strip()):
            html_parts.append("<hr>")
            i += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            content = _inline_formatting(heading_match.group(2).strip())
            html_parts.append(f"<h{level}>{content}</h{level}>")
            i += 1
            continue

        if _UL_ITEM_RE.match(line):
            items: list[str] = []
            while i < len(lines) and _UL_ITEM_RE.match(lines[i]):
                match = _UL_ITEM_RE.match(lines[i])
                assert match is not None
                items.append(f"<li>{_inline_formatting(match.group(1))}</li>")
                i += 1
            html_parts.append("<ul>" + "".join(items) + "</ul>")
            continue

        if _OL_ITEM_RE.match(line):
            items = []
            while i < len(lines) and _OL_ITEM_RE.match(lines[i]):
                match = _OL_ITEM_RE.match(lines[i])
                assert match is not None
                items.append(f"<li>{_inline_formatting(match.group(1))}</li>")
                i += 1
            html_parts.append("<ol>" + "".join(items) + "</ol>")
            continue

        para_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            next_line = lines[i]
            if (
                _HEADING_RE.match(next_line)
                or _HR_RE.match(next_line.strip())
                or _UL_ITEM_RE.match(next_line)
                or _OL_ITEM_RE.match(next_line)
            ):
                break
            para_lines.append(next_line)
            i += 1
        if para_lines:
            content = _inline_formatting(" ".join(para_lines))
            html_parts.append(f"<p>{content}</p>")

    return "\n".join(html_parts)


def _build_css(layout: PageLayout) -> str:
    """Generate a CSS stylesheet from layout parameters."""
    w_in, h_in = layout.page_size
    margin = layout.margin_inches

    page_counter = ""
    if layout.page_numbers:
        page_counter = (
            '@bottom-center { content: "Page " counter(page) " of " counter(pages); '
            f"font-family: {layout.body_font}; font-size: 9pt; color: #666; }}"
        )

    page_rule = f"@page {{ size: {w_in}in {h_in}in; margin: {margin}in; "
    if layout.page_numbers:
        page_rule += page_counter
    else:
        page_rule += "}"

    body_css = (
        f"body {{ font-family: {layout.body_font}; "
        f"font-size: {layout.body_font_size_pt}pt; "
        f"line-height: {layout.line_spacing}; "
        "color: #222; }}"
    )

    paragraph_css = f"p {{ margin: 0 0 {layout.paragraph_spacing_pt}pt 0; text-align: justify; }}"

    heading_shared = (
        f"font-family: {layout.heading_font}; color: #111; "
        f"margin-top: {layout.paragraph_spacing_pt * 2}pt; "
        f"margin-bottom: {layout.paragraph_spacing_pt}pt;"
    )
    h1_css = f"h1 {{ {heading_shared} font-size: {layout.heading_font_size_pt + 5}pt; }}"
    h2_css = f"h2 {{ {heading_shared} font-size: {layout.heading_font_size_pt + 2}pt; }}"
    h3_css = f"h3 {{ {heading_shared} font-size: {layout.heading_font_size_pt}pt; }}"

    list_css = (
        f"ul, ol {{ margin: 0 0 {layout.paragraph_spacing_pt}pt 20pt; padding: 0; }} "
        f"li {{ margin-bottom: {layout.paragraph_spacing_pt // 2}pt; }}"
    )

    hr_css = "hr { border: none; border-top: 1px solid #999; margin: 12pt 0; }"

    return "\n".join(
        [page_rule, body_css, paragraph_css, h1_css, h2_css, h3_css, list_css, hr_css]
    )


def markdown_to_html(markdown_text: str, layout: PageLayout = DEFAULT_LAYOUT) -> str:
    """Convert markdown text to a fully styled HTML document."""
    body_html = _convert_markdown_body(markdown_text)
    css = _build_css(layout)

    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<style>\n{css}\n</style>\n"
        "</head>\n"
        f"<body>\n{body_html}\n</body>\n"
        "</html>"
    )


# ---------------------------------------------------------------------------
# Markdown -> parsed blocks for reportlab
# ---------------------------------------------------------------------------

_STRIP_BOLD = re.compile(r"\*\*(.+?)\*\*")
_STRIP_ITALIC = re.compile(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)")
_TABLE_SEP_RE = re.compile(r"^\|[\s:-]+\|", re.MULTILINE)
_TABLE_ROW_RE = re.compile(r"^\|(.+)\|$")

_CODE_FENCE_RE = re.compile(r"^```\w*\s*$", re.MULTILINE)

_UNICODE_REPLACEMENTS: dict[str, str] = {
    "–": "-",  # en-dash
    "—": "--",  # em-dash
    "‘": "'",  # left single quote
    "’": "'",  # right single quote
    "“": '"',  # left double quote
    "”": '"',  # right double quote
    "•": "-",  # bullet
    "…": "...",  # ellipsis
    " ": " ",  # non-breaking space
    "′": "'",  # prime
    "″": '"',  # double prime
    "·": "-",  # middle dot
    "●": "-",  # black circle
    "■": "-",  # black square
    "−": "-",  # minus sign
    "×": "x",  # multiplication sign
    "≤": "<=",  # less-than-or-equal
    "≥": ">=",  # greater-than-or-equal
    "®": "(R)",  # registered trademark
    "™": "(TM)",  # trademark
    "©": "(C)",  # copyright
    "§": "Section ",  # section sign
    "¹": "1",  # superscript 1
    "²": "2",  # superscript 2
    "³": "3",  # superscript 3
    "†": "*",  # dagger
    "‡": "**",  # double dagger
    "°": " degrees",  # degree sign
    "½": "1/2",  # fraction one half
    "¼": "1/4",  # fraction one quarter
    "¾": "3/4",  # fraction three quarters
    "‐": "-",  # hyphen
    "‑": "-",  # non-breaking hyphen
    "―": "--",  # horizontal bar
    "⁃": "-",  # hyphen bullet
    "◦": "-",  # white bullet
    "‣": "-",  # triangular bullet
    "«": "<<",  # left guillemet
    "»": ">>",  # right guillemet
    "☐": "[ ]",  # ballot box (unchecked)
    "☑": "[X]",  # ballot box (checked)
    "☒": "[X]",  # ballot box with X
    "✓": "[X]",  # check mark
    "✗": "[ ]",  # ballot X
    "✔": "[X]",  # heavy check mark
    "→": "->",  # right arrow
    "←": "<-",  # left arrow
    "↓": "v",  # down arrow
    "↑": "^",  # up arrow
    "€": "EUR",  # euro sign
    "£": "GBP",  # pound sign
    "¥": "JPY",  # yen sign
    "¢": "c",  # cent sign
    "ℓ": "l",  # script small l
    "№": "No.",  # numero sign
    "⁴": "4",  # superscript 4
    "⁵": "5",  # superscript 5
    "⁶": "6",  # superscript 6
    "⁷": "7",  # superscript 7
    "⁸": "8",  # superscript 8
    "⁹": "9",  # superscript 9
    "⁰": "0",  # superscript 0
}


def _sanitize_text(text: str) -> str:
    """Replace Unicode characters that reportlab's default fonts can't render."""
    # Strip markdown code fences that LLMs sometimes wrap output in
    text = _CODE_FENCE_RE.sub("", text)
    # Replace known problematic characters
    for old, new in _UNICODE_REPLACEMENTS.items():
        text = text.replace(old, new)
    # Strip any remaining non-ASCII that slipped through
    return text.encode("ascii", errors="replace").decode("ascii")


_BOLD_ITALIC_RE = re.compile(r"\*{3}(.+?)\*{3}")


def _md_to_rl_markup(text: str) -> str:
    """Convert markdown inline formatting to reportlab XML markup."""
    text = _sanitize_text(text)
    # Strip all HTML tags the LLM may have injected
    text = re.sub(r"<[^>]+>", "", text)
    # Convert markdown links [text](url) to just text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Bold+italic combo first (***text***) to avoid nesting issues
    text = _BOLD_ITALIC_RE.sub(r"<b><i>\1</i></b>", text)
    text = _STRIP_BOLD.sub(r"<b>\1</b>", text)
    text = _STRIP_ITALIC.sub(r"<i>\1</i>", text)
    # Fix any mis-nested tags that reportlab would reject
    text = re.sub(r"<b><i>(.+?)</b></i>", r"<b><i>\1</i></b>", text)
    text = re.sub(r"<i><b>(.+?)</i></b>", r"<i><b>\1</b></i>", text)
    return text


def _parse_table_rows(lines: list[str], start: int) -> tuple[list[dict], int]:
    """Parse a markdown table starting at *start*, return (blocks, next_index).

    Produces one "table" block with headers and rows.
    """
    header_line = lines[start].strip()
    if not _TABLE_ROW_RE.match(header_line):
        return [], start

    headers = [c.strip() for c in header_line.strip("|").split("|")]
    i = start + 1

    if i < len(lines) and _TABLE_SEP_RE.match(lines[i].strip()):
        i += 1

    rows: list[list[str]] = []
    while i < len(lines):
        row_line = lines[i].strip()
        if not _TABLE_ROW_RE.match(row_line):
            break
        cells = [c.strip() for c in row_line.strip("|").split("|")]
        rows.append(cells)
        i += 1

    return [{"type": "table", "headers": headers, "rows": rows}], i


def _parse_markdown_blocks(markdown_text: str) -> list[dict]:
    """Parse markdown into a list of block dicts for reportlab rendering.

    Each block is {"type": "heading"|"paragraph"|"list"|"hr", ...}.
    """
    lines = markdown_text.split("\n")
    blocks: list[dict] = []
    i = 0

    while i < len(lines):
        line = lines[i]

        if not line.strip():
            i += 1
            continue

        if _HR_RE.match(line.strip()):
            blocks.append({"type": "hr"})
            i += 1
            continue

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            level = len(heading_match.group(1))
            blocks.append(
                {
                    "type": "heading",
                    "level": level,
                    "text": _md_to_rl_markup(heading_match.group(2).strip()),
                }
            )
            i += 1
            continue

        if _UL_ITEM_RE.match(line):
            items: list[str] = []
            while i < len(lines) and _UL_ITEM_RE.match(lines[i]):
                match = _UL_ITEM_RE.match(lines[i])
                assert match is not None
                items.append(_md_to_rl_markup(match.group(1)))
                i += 1
            blocks.append({"type": "list", "ordered": False, "items": items})
            continue

        if _OL_ITEM_RE.match(line):
            items = []
            while i < len(lines) and _OL_ITEM_RE.match(lines[i]):
                match = _OL_ITEM_RE.match(lines[i])
                assert match is not None
                items.append(_md_to_rl_markup(match.group(1)))
                i += 1
            blocks.append({"type": "list", "ordered": True, "items": items})
            continue

        if _TABLE_ROW_RE.match(line.strip()):
            table_blocks, i = _parse_table_rows(lines, i)
            blocks.extend(table_blocks)
            continue

        para_lines: list[str] = []
        while i < len(lines) and lines[i].strip():
            next_line = lines[i]
            if (
                _HEADING_RE.match(next_line)
                or _HR_RE.match(next_line.strip())
                or _UL_ITEM_RE.match(next_line)
                or _OL_ITEM_RE.match(next_line)
                or _TABLE_ROW_RE.match(next_line.strip())
            ):
                break
            para_lines.append(next_line)
            i += 1
        if para_lines:
            blocks.append(
                {
                    "type": "paragraph",
                    "text": _md_to_rl_markup(" ".join(para_lines)),
                }
            )

    return blocks


# ---------------------------------------------------------------------------
# Markdown -> PDF (reportlab)
# ---------------------------------------------------------------------------


def render_markdown_to_pdf(
    markdown_text: str,
    output_path: Path,
    layout: PageLayout = DEFAULT_LAYOUT,
) -> Path:
    """Render markdown text to a PDF file using reportlab.

    Create parent directories when they do not exist.
    """
    from reportlab.lib.colors import HexColor
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        HRFlowable,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    margin = layout.margin_inches * inch
    page_w = layout.page_size[0] * inch
    page_h = layout.page_size[1] * inch

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=(page_w, page_h),
        leftMargin=margin,
        rightMargin=margin,
        topMargin=margin,
        bottomMargin=margin,
    )

    base_styles = getSampleStyleSheet()
    leading = layout.body_font_size_pt * layout.line_spacing

    body_style = ParagraphStyle(
        "body",
        parent=base_styles["Normal"],
        fontName=layout.body_font,
        fontSize=layout.body_font_size_pt,
        leading=leading,
        spaceAfter=layout.paragraph_spacing_pt,
        alignment=TA_JUSTIFY,
    )
    h1_style = ParagraphStyle(
        "md_h1",
        parent=base_styles["Heading1"],
        fontName=layout.heading_font,
        fontSize=layout.heading_font_size_pt + 5,
        leading=(layout.heading_font_size_pt + 5) * 1.2,
        spaceBefore=layout.paragraph_spacing_pt * 2,
        spaceAfter=layout.paragraph_spacing_pt,
    )
    h2_style = ParagraphStyle(
        "md_h2",
        parent=base_styles["Heading2"],
        fontName=layout.heading_font,
        fontSize=layout.heading_font_size_pt + 2,
        leading=(layout.heading_font_size_pt + 2) * 1.2,
        spaceBefore=layout.paragraph_spacing_pt * 2,
        spaceAfter=layout.paragraph_spacing_pt,
    )
    h3_style = ParagraphStyle(
        "md_h3",
        parent=base_styles["Heading3"],
        fontName=layout.heading_font,
        fontSize=layout.heading_font_size_pt,
        leading=layout.heading_font_size_pt * 1.2,
        spaceBefore=layout.paragraph_spacing_pt,
        spaceAfter=layout.paragraph_spacing_pt,
    )
    h4_style = ParagraphStyle(
        "md_h4",
        parent=body_style,
        fontName=layout.heading_font,
        fontSize=layout.body_font_size_pt + 1,
        leading=(layout.body_font_size_pt + 1) * 1.2,
        spaceBefore=layout.paragraph_spacing_pt,
        spaceAfter=layout.paragraph_spacing_pt // 2,
    )
    list_style = ParagraphStyle(
        "md_list",
        parent=body_style,
        alignment=TA_LEFT,
    )

    heading_styles = {1: h1_style, 2: h2_style, 3: h3_style, 4: h4_style, 5: h4_style, 6: h4_style}

    def _safe_para(text: str, style: Any) -> Any:
        """Create a Paragraph, stripping all markup on parse failure."""
        try:
            return Paragraph(text, style)
        except Exception:
            plain = re.sub(r"<[^>]+>", "", text)
            plain = plain.replace("<", "&lt;").replace(">", "&gt;")
            plain = plain.replace("&", "&amp;") if "&amp;" not in plain else plain
            try:
                return Paragraph(plain, style)
            except Exception:
                return Paragraph("(rendering error)", style)

    blocks = _parse_markdown_blocks(markdown_text)
    story: list = []

    for block in blocks:
        btype = block["type"]

        if btype == "heading":
            style = heading_styles.get(block["level"], h3_style)
            story.append(_safe_para(block["text"], style))

        elif btype == "paragraph":
            story.append(_safe_para(block["text"], body_style))

        elif btype == "list":
            for idx, item_text in enumerate(block["items"], 1):
                prefix = f"{idx}." if block["ordered"] else "-"
                story.append(_safe_para(f"{prefix}  {item_text}", list_style))
            story.append(Spacer(1, layout.paragraph_spacing_pt))

        elif btype == "table":
            cell_style = ParagraphStyle(
                "table_cell",
                parent=body_style,
                fontSize=layout.body_font_size_pt - 1,
                leading=(layout.body_font_size_pt - 1) * 1.2,
                spaceAfter=0,
                alignment=TA_LEFT,
            )
            header_cell_style = ParagraphStyle(
                "table_header",
                parent=cell_style,
                fontName=layout.heading_font,
            )
            header_row = [
                _safe_para(_md_to_rl_markup(h), header_cell_style) for h in block["headers"]
            ]
            data_rows = [header_row]
            for row in block["rows"]:
                data_rows.append([_safe_para(_md_to_rl_markup(c), cell_style) for c in row])
            avail_w = (layout.page_size[0] - 2 * layout.margin_inches) * inch
            n_cols = len(block["headers"])
            col_w = avail_w / max(n_cols, 1)
            tbl = Table(data_rows, colWidths=[col_w] * n_cols)
            tbl.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#E8E8E8")),
                        ("FONTNAME", (0, 0), (-1, 0), layout.heading_font),
                        ("FONTSIZE", (0, 0), (-1, -1), layout.body_font_size_pt - 1),
                        ("GRID", (0, 0), (-1, -1), 0.5, HexColor("#CCCCCC")),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("TOPPADDING", (0, 0), (-1, -1), 4),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                        ("LEFTPADDING", (0, 0), (-1, -1), 6),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ]
                )
            )
            story.append(tbl)
            story.append(Spacer(1, layout.paragraph_spacing_pt))

        elif btype == "hr":
            story.append(Spacer(1, 6))
            story.append(HRFlowable(width="100%", thickness=0.5, color="#999999"))
            story.append(Spacer(1, 6))

    if not story:
        story.append(Paragraph(" ", body_style))

    page_num_fn = None
    if layout.page_numbers:

        def page_num_fn(canvas: Any, doc: Any) -> None:
            canvas.saveState()
            canvas.setFont(layout.body_font, 9)
            text = f"Page {doc.page}"
            canvas.drawCentredString(page_w / 2.0, margin * 0.5, text)
            canvas.restoreState()

    log.info("Rendering PDF to %s", output_path)

    build_args = {"onFirstPage": page_num_fn, "onLaterPages": page_num_fn} if page_num_fn else {}
    try:
        doc.build(story, **build_args)
    except Exception as exc:
        log.warning("Build failed (%s), retrying without tables", exc)
        # Replace Table flowables with plain-text paragraphs
        safe_story = []
        for flowable in story:
            if type(flowable).__name__ == "Table":
                safe_story.append(_safe_para("(table omitted due to size)", body_style))
            else:
                safe_story.append(flowable)
        doc2 = SimpleDocTemplate(
            str(output_path),
            pagesize=(page_w, page_h),
            leftMargin=margin,
            rightMargin=margin,
            topMargin=margin,
            bottomMargin=margin,
        )
        doc2.build(safe_story, **build_args)

    log.debug("PDF written: %s (%d bytes)", output_path, output_path.stat().st_size)
    return output_path


def render_html_to_pdf(
    html_content: str,
    output_path: Path,
    layout: PageLayout = DEFAULT_LAYOUT,
) -> Path:
    """Render HTML to PDF. Extracts text content and renders via reportlab."""
    text = re.sub(r"<[^>]+>", "", html_content)
    return render_markdown_to_pdf(text, output_path, layout=layout)


# ---------------------------------------------------------------------------
# PDF page counting (pypdf)
# ---------------------------------------------------------------------------


def count_pdf_pages(pdf_path: Path) -> int:
    """Return the number of pages in a PDF file."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    log.debug("Page count for %s: %d", pdf_path, page_count)
    return page_count
