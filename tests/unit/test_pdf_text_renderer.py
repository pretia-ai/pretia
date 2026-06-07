"""Tests for PageLayout defaults, frozen behavior, and markdown-to-HTML conversion."""

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

from pdfs.generators.rendering.text_renderer import (
    DEFAULT_LAYOUT,
    PageLayout,
    markdown_to_html,
)

import pytest


class TestPageLayout:
    def test_defaults_match_spec(self):
        layout = PageLayout()
        assert layout.page_size == (8.5, 11.0)
        assert layout.margin_inches == 1.0
        assert layout.body_font_size_pt == 11
        assert layout.heading_font_size_pt == 13
        assert layout.line_spacing == 1.15
        assert layout.paragraph_spacing_pt == 6
        assert layout.page_numbers is True

    def test_frozen(self):
        with pytest.raises(AttributeError):
            DEFAULT_LAYOUT.margin_inches = 2.0


class TestMarkdownToHtml:
    def test_heading_h1(self):
        html = markdown_to_html("# Title")
        assert "<h1>" in html and "Title" in html

    def test_heading_h2(self):
        html = markdown_to_html("## Subtitle")
        assert "<h2>" in html

    def test_heading_h3(self):
        html = markdown_to_html("### Section")
        assert "<h3>" in html

    def test_bold(self):
        html = markdown_to_html("This is **bold** text")
        assert "<strong>bold</strong>" in html

    def test_italic(self):
        html = markdown_to_html("This is *italic* text")
        assert "<em>italic</em>" in html

    def test_paragraph(self):
        html = markdown_to_html("First paragraph.\n\nSecond paragraph.")
        assert "<p>" in html

    def test_unordered_list(self):
        html = markdown_to_html("- item one\n- item two")
        assert "<li>" in html

    def test_horizontal_rule(self):
        html = markdown_to_html("---")
        assert "<hr" in html

    def test_contains_page_css(self):
        html = markdown_to_html("Hello")
        assert "@page" in html

    def test_page_numbers_in_css(self):
        html = markdown_to_html("Hello", PageLayout(page_numbers=True))
        assert "counter" in html.lower() or "page" in html.lower()

    def test_custom_layout_font_size(self):
        layout = PageLayout(body_font_size_pt=14)
        html = markdown_to_html("Text", layout)
        assert "14pt" in html

    def test_returns_full_html_document(self):
        html = markdown_to_html("Hello")
        assert html.strip().startswith("<!DOCTYPE html>") or html.strip().startswith("<html")
