"""Tests for PDFDescriptor round-trip serialization, write/load, and PageSource validation."""

from __future__ import annotations

import sys
import types

# pdf_assembler lives under pdfs.generators.rendering, whose __init__.py imports
# scan_simulator (PIL dependency).  Register a lightweight placeholder so we can
# reach the submodule without triggering that heavy import chain.
if "pdfs.generators.rendering" not in sys.modules:
    import importlib

    for _pkg in ("pdfs", "pdfs.generators"):
        if _pkg not in sys.modules:
            importlib.import_module(_pkg)
    _placeholder = types.ModuleType("pdfs.generators.rendering")
    _placeholder.__path__ = ["pdfs/generators/rendering"]
    sys.modules["pdfs.generators.rendering"] = _placeholder

import pytest

from pdfs.generators.rendering.pdf_assembler import (
    PageSource,
    PDFDescriptor,
    load_descriptor,
    write_descriptor,
)


class TestPDFDescriptorRoundTrip:
    def test_minimal_round_trip(self):
        desc = PDFDescriptor(
            pdf_id="test_001",
            workflow="W17",
            profile="profiling",
            document_type="provider_policy",
            page_count=30,
            estimated_token_count=24000,
            text_pages=30,
            table_chart_pages=0,
            scanned_pages=0,
            section_count=8,
        )
        assert PDFDescriptor.from_dict(desc.to_dict()) == desc

    def test_with_optional_fields(self):
        desc = PDFDescriptor(
            pdf_id="test_002",
            workflow="W16",
            profile="ground_truth",
            document_type="annual_report",
            page_count=45,
            estimated_token_count=36000,
            text_pages=35,
            table_chart_pages=10,
            scanned_pages=0,
            section_count=12,
            provider=None,
            structure_quality="poorly_structured",
            content_density="dense",
            generation_model="deepseek-v4-pro",
        )
        assert PDFDescriptor.from_dict(desc.to_dict()) == desc

    def test_key_fields_list(self):
        desc = PDFDescriptor(
            pdf_id="test_003",
            workflow="W14",
            profile="profiling",
            document_type="sbc",
            page_count=12,
            estimated_token_count=9600,
            text_pages=10,
            table_chart_pages=2,
            scanned_pages=0,
            section_count=5,
            key_fields_present=["deductible", "oop_max", "er_copay"],
            provider="United Healthcare",
        )
        reloaded = PDFDescriptor.from_dict(desc.to_dict())
        assert reloaded.key_fields_present == ["deductible", "oop_max", "er_copay"]
        assert reloaded.provider == "United Healthcare"


class TestWriteLoadDescriptor:
    def test_write_and_load(self, tmp_path):
        desc = PDFDescriptor(
            pdf_id="roundtrip",
            workflow="W18",
            profile="profiling",
            document_type="regulatory_filing",
            page_count=80,
            estimated_token_count=64000,
            text_pages=80,
            table_chart_pages=0,
            scanned_pages=0,
            section_count=15,
        )
        json_path = write_descriptor(desc, tmp_path)
        assert json_path.exists()
        assert json_path.name == "roundtrip.json"
        loaded = load_descriptor(json_path)
        assert loaded == desc


class TestPageSourceValidation:
    def test_valid_text_pdf(self, tmp_path):
        f = tmp_path / "test.pdf"
        f.write_text("dummy")
        ps = PageSource(source_type="text_pdf", source_path=f)
        assert ps.source_type == "text_pdf"

    def test_invalid_source_type_raises(self):
        with pytest.raises(ValueError, match="source_type"):
            PageSource(source_type="invalid_type")

    def test_pdf_source_without_path_raises(self):
        with pytest.raises(ValueError, match="source_path"):
            PageSource(source_type="text_pdf")
