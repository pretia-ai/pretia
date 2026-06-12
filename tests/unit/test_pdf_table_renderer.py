"""Tests for TableData/TableStyle models and table_to_html rendering."""

from __future__ import annotations

import sys
import types

if "pdfs.generators.rendering" not in sys.modules:
    import importlib

    for _pkg in ("pdfs", "pdfs.generators"):
        if _pkg not in sys.modules:
            importlib.import_module(_pkg)
    _placeholder = types.ModuleType("pdfs.generators.rendering")
    _placeholder.__path__ = ["pdfs/generators/rendering"]
    sys.modules["pdfs.generators.rendering"] = _placeholder

from pdfs.generators.rendering.table_renderer import (
    TableData,
    TableStyle,
    table_to_html,
)


class TestTableToHtml:
    def test_basic_table(self):
        table = TableData(headers=["Name", "Value"], rows=[["A", "1"], ["B", "2"]])
        html = table_to_html(table)
        assert "<table" in html
        assert "<thead>" in html
        assert "<tbody>" in html
        assert "Name" in html
        assert "A" in html

    def test_full_border_style(self):
        table = TableData(headers=["X"], rows=[["Y"]])
        html = table_to_html(table, TableStyle(border_style="full"))
        assert "border" in html

    def test_header_only_border_style(self):
        table = TableData(headers=["X"], rows=[["Y"]])
        html = table_to_html(table, TableStyle(border_style="header_only"))
        assert "<table" in html

    def test_minimal_border_style(self):
        table = TableData(headers=["X"], rows=[["Y"]])
        html = table_to_html(table, TableStyle(border_style="minimal"))
        assert "<table" in html

    def test_html_escapes_content(self):
        table = TableData(headers=["Col"], rows=[["<script>alert('xss')</script>"]])
        html = table_to_html(table)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_caption_rendered(self):
        table = TableData(headers=["A"], rows=[["B"]], caption="My Caption")
        html = table_to_html(table)
        assert "My Caption" in html

    def test_no_caption_when_none(self):
        table = TableData(headers=["A"], rows=[["B"]], caption=None)
        html = table_to_html(table)
        assert "<caption>" not in html

    def test_header_bg_color(self):
        style = TableStyle(header_bg_color="#FF0000")
        table = TableData(headers=["A"], rows=[["B"]])
        html = table_to_html(table, style)
        assert "#FF0000" in html
