"""Tests for pdfs/generators/_types.py: dataclass round-trips and reference data constants."""

from __future__ import annotations

from pdfs.generators._types import (
    W14_DOC_TYPES,
    W14_W15_PROVIDER_VALUES,
    W16_TIERS,
    W17_PROVIDER_VALUES,
    W18_TIERS,
    ContentManifest,
    ContentManifestEntry,
    ContentManifestSection,
    DocumentSpec,
    SectionSpec,
)


class TestContentManifestSectionRoundTrip:
    def test_round_trip(self):
        section = ContentManifestSection(
            title="Deductibles",
            page_range=(3, 5),
            key_facts=["Deductible: $1,500", "ER copay: $250"],
        )
        assert ContentManifestSection.from_dict(section.to_dict()) == section

    def test_page_range_tuple_preserved(self):
        data = {"title": "X", "page_range": [1, 3], "key_facts": []}
        assert isinstance(ContentManifestSection.from_dict(data).page_range, tuple)


class TestContentManifestEntryRoundTrip:
    def test_round_trip(self):
        entry = ContentManifestEntry(
            pdf_id="doc-001",
            pdf_filename="sbc_united.pdf",
            provider="United Healthcare",
            document_type="sbc",
            page_count=12,
            estimated_token_count=9600,
            sections=[
                ContentManifestSection(
                    title="Deductibles",
                    page_range=(1, 4),
                    key_facts=["Deductible: $1,500"],
                ),
                ContentManifestSection(
                    title="Copays",
                    page_range=(5, 8),
                    key_facts=["ER copay: $250", "PCP copay: $30"],
                ),
            ],
        )
        assert ContentManifestEntry.from_dict(entry.to_dict()) == entry


class TestContentManifestRoundTrip:
    def _make_manifest(self) -> ContentManifest:
        return ContentManifest(
            corpus_id="test-corpus-001",
            workflow="w14",
            profile="profiling",
            generated_at="2026-01-15T12:00:00+00:00",
            documents=[
                ContentManifestEntry(
                    pdf_id="doc-001",
                    pdf_filename="sbc_united.pdf",
                    provider="United Healthcare",
                    document_type="sbc",
                    page_count=10,
                    estimated_token_count=8000,
                    sections=[
                        ContentManifestSection(
                            title="Overview",
                            page_range=(1, 3),
                            key_facts=["Plan type: PPO"],
                        ),
                    ],
                ),
                ContentManifestEntry(
                    pdf_id="doc-002",
                    pdf_filename="formulary_aetna.pdf",
                    provider="Aetna",
                    document_type="formulary",
                    page_count=7,
                    estimated_token_count=5600,
                    sections=[
                        ContentManifestSection(
                            title="Drug List",
                            page_range=(1, 7),
                            key_facts=["Tier 1: generics"],
                        ),
                    ],
                ),
            ],
        )

    def test_round_trip(self):
        manifest = self._make_manifest()
        assert ContentManifest.from_dict(manifest.to_dict()) == manifest

    def test_to_dict_computes_counts(self):
        manifest = self._make_manifest()
        data = manifest.to_dict()
        assert data["provider_count"] == 2
        assert data["document_count"] == 2

    def test_create_factory(self):
        manifest = ContentManifest.create(
            corpus_id="factory-test",
            workflow="w15",
            profile="ground_truth",
        )
        assert manifest.corpus_id == "factory-test"
        assert manifest.workflow == "w15"
        assert manifest.profile == "ground_truth"
        assert manifest.documents == []
        assert len(manifest.generated_at) > 0

    def test_save_load(self, tmp_path):
        manifest = self._make_manifest()
        path = tmp_path / "manifests" / "test_manifest.json"
        manifest.save(path)
        loaded = ContentManifest.load(path)
        assert loaded == manifest


class TestDocumentSpecToDict:
    def test_serializes_sections(self):
        spec = DocumentSpec(
            doc_id="spec-001",
            workflow="w14",
            profile="profiling",
            document_type="sbc",
            domain="health_insurance",
            sections=[
                SectionSpec(
                    title="Deductibles",
                    target_pages=3,
                    content_type="text",
                    key_values={"deductible": "$1,500"},
                ),
                SectionSpec(
                    title="Copays",
                    target_pages=2,
                    content_type="table_heavy",
                ),
            ],
            target_page_count=10,
        )
        data = spec.to_dict()
        assert len(data["sections"]) == 2
        assert data["sections"][0]["title"] == "Deductibles"
        assert data["sections"][1]["content_type"] == "table_heavy"

    def test_default_model(self):
        spec = DocumentSpec(
            doc_id="spec-002",
            workflow="w14",
            profile="profiling",
            document_type="sbc",
            domain="health_insurance",
            sections=[],
            target_page_count=5,
        )
        assert spec.generation_model == "deepseek-v4-pro"


class TestReferenceDataConstants:
    def test_w14_w15_providers(self):
        assert len(W14_W15_PROVIDER_VALUES) == 3
        for _provider, values in W14_W15_PROVIDER_VALUES.items():
            assert len(values) == 8

    def test_w17_providers(self):
        assert len(W17_PROVIDER_VALUES) == 3
        for _provider, values in W17_PROVIDER_VALUES.items():
            assert len(values) == 7

    def test_w14_doc_types(self):
        assert set(W14_DOC_TYPES.keys()) == {
            "sbc",
            "formulary",
            "network_directory",
            "detailed_policy",
            "member_handbook",
        }

    def test_w16_tiers(self):
        assert "easy" in W16_TIERS
        assert "extreme" in W16_TIERS
        assert W16_TIERS["extreme"]["profiling_pages"] is None

    def test_w18_tiers(self):
        assert "easy" in W18_TIERS
        assert W18_TIERS["easy"]["profiling_tokens"] == (30_000, 40_000)
