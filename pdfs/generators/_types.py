"""Shared dataclasses for PDF generation pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class SectionSpec:
    """One section of a document to be generated."""

    title: str
    target_pages: int
    content_type: str  # "text" | "table_heavy" | "mixed"
    key_values: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DocumentSpec:
    """Full specification for one document to generate."""

    doc_id: str
    workflow: str
    profile: str  # "profiling" | "ground_truth"
    document_type: str
    domain: str
    sections: list[SectionSpec]
    target_page_count: int
    target_token_count: int | None = None
    generation_model: str = "deepseek-v4-pro"
    provider: str | None = None
    modality_mix: tuple[float, float, float] = (1.0, 0.0, 0.0)
    structure_quality: str = "well_structured"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "doc_id": self.doc_id,
            "workflow": self.workflow,
            "profile": self.profile,
            "document_type": self.document_type,
            "domain": self.domain,
            "sections": [
                {
                    "title": s.title,
                    "target_pages": s.target_pages,
                    "content_type": s.content_type,
                    "key_values": dict(s.key_values),
                }
                for s in self.sections
            ],
            "target_page_count": self.target_page_count,
            "target_token_count": self.target_token_count,
            "generation_model": self.generation_model,
            "provider": self.provider,
            "modality_mix": list(self.modality_mix),
            "structure_quality": self.structure_quality,
        }


@dataclass(frozen=True, slots=True)
class ContentManifestSection:
    """One section within a document in the content manifest."""

    title: str
    page_range: tuple[int, int]
    key_facts: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "title": self.title,
            "page_range": list(self.page_range),
            "key_facts": list(self.key_facts),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentManifestSection:
        """Deserialize from dict."""
        return cls(
            title=data["title"],
            page_range=tuple(data["page_range"]),
            key_facts=data["key_facts"],
        )


@dataclass(frozen=True, slots=True)
class ContentManifestEntry:
    """One document's entry in the W14/W15 content manifest."""

    pdf_id: str
    pdf_filename: str
    provider: str
    document_type: str
    page_count: int
    estimated_token_count: int
    sections: list[ContentManifestSection]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "pdf_id": self.pdf_id,
            "pdf_filename": self.pdf_filename,
            "provider": self.provider,
            "document_type": self.document_type,
            "page_count": self.page_count,
            "estimated_token_count": self.estimated_token_count,
            "sections": [s.to_dict() for s in self.sections],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentManifestEntry:
        """Deserialize from dict."""
        return cls(
            pdf_id=data["pdf_id"],
            pdf_filename=data["pdf_filename"],
            provider=data["provider"],
            document_type=data["document_type"],
            page_count=data["page_count"],
            estimated_token_count=data["estimated_token_count"],
            sections=[ContentManifestSection.from_dict(s) for s in data["sections"]],
        )


@dataclass(frozen=True, slots=True)
class ContentManifest:
    """Full content manifest for a W14/W15 corpus."""

    corpus_id: str
    workflow: str
    profile: str
    generated_at: str
    documents: list[ContentManifestEntry]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "corpus_id": self.corpus_id,
            "workflow": self.workflow,
            "profile": self.profile,
            "generated_at": self.generated_at,
            "provider_count": len({d.provider for d in self.documents}),
            "document_count": len(self.documents),
            "documents": [d.to_dict() for d in self.documents],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContentManifest:
        """Deserialize from dict."""
        return cls(
            corpus_id=data["corpus_id"],
            workflow=data["workflow"],
            profile=data["profile"],
            generated_at=data["generated_at"],
            documents=[ContentManifestEntry.from_dict(d) for d in data["documents"]],
        )

    def save(self, path: Path) -> None:
        """Write manifest as JSON to the given path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Path) -> ContentManifest:
        """Load manifest from a JSON file."""
        data = json.loads(path.read_text())
        return cls.from_dict(data)

    @classmethod
    def create(
        cls,
        corpus_id: str,
        workflow: str,
        profile: str,
    ) -> ContentManifest:
        """Create an empty manifest with timestamp."""
        return cls(
            corpus_id=corpus_id,
            workflow=workflow,
            profile=profile,
            generated_at=datetime.now(UTC).isoformat(),
            documents=[],
        )


# --- Cross-provider reference data ---
# These tables are load-bearing: the generated documents must contain these exact values,
# and the input generators create queries/claims targeting them.

W14_W15_PROVIDER_VALUES: dict[str, dict[str, str]] = {
    "United Healthcare": {
        "in_network_deductible": "$1,500",
        "out_of_pocket_max": "$6,500",
        "er_copay": "$250",
        "mri_prior_auth": "Required, 5-day turnaround",
        "mental_health_visits": "20/year, $40 copay",
        "pre_existing_waiting": "None (ACA compliant)",
        "out_of_network_coverage": "60% after $3,000 deductible",
        "prescription_tiers": "4 tiers",
    },
    "Aetna": {
        "in_network_deductible": "$2,000",
        "out_of_pocket_max": "$7,500",
        "er_copay": "$300",
        "mri_prior_auth": "Required, 3-day turnaround",
        "mental_health_visits": "Unlimited, $50 copay",
        "pre_existing_waiting": "None",
        "out_of_network_coverage": "50% after $4,000 deductible",
        "prescription_tiers": "5 tiers",
    },
    "BlueCross BlueShield": {
        "in_network_deductible": "$1,200",
        "out_of_pocket_max": "$5,800",
        "er_copay": "$200",
        "mri_prior_auth": "Not required for in-network",
        "mental_health_visits": "30/year, $35 copay",
        "pre_existing_waiting": "None",
        "out_of_network_coverage": "70% after $2,500 deductible",
        "prescription_tiers": "4 tiers",
    },
}

W17_PROVIDER_VALUES: dict[str, dict[str, str]] = {
    "United Healthcare": {
        "mri_prior_auth": "Yes, all MRI",
        "pre_existing_exclusion": "None (ACA)",
        "appeal_deadline": "180 days from denial",
        "medical_necessity_standard": "Generally accepted medical practice",
        "er_definition": '"Prudent layperson" standard',
        "experimental_exclusion": "Excludes Phase I/II trials",
        "max_out_of_pocket": "$7,500 individual",
    },
    "Aetna": {
        "mri_prior_auth": "Yes, non-emergency only",
        "pre_existing_exclusion": "None (ACA)",
        "appeal_deadline": "90 days from denial",
        "medical_necessity_standard": "Evidence-based, peer-reviewed",
        "er_definition": '"Prudent layperson" standard',
        "experimental_exclusion": "Excludes all clinical trials",
        "max_out_of_pocket": "$8,200 individual",
    },
    "Cigna": {
        "mri_prior_auth": "No (in-network only)",
        "pre_existing_exclusion": "6-month lookback for specific conditions",
        "appeal_deadline": "120 days from denial",
        "medical_necessity_standard": "Clinically appropriate and effective",
        "er_definition": '"Reasonable person" standard',
        "experimental_exclusion": "Excludes Phase I only",
        "max_out_of_pocket": "$6,800 individual",
    },
}

# W14/W15 document types with page ranges
W14_DOC_TYPES: dict[str, tuple[int, int]] = {
    "sbc": (8, 15),
    "formulary": (5, 10),
    "network_directory": (3, 8),
    "detailed_policy": (20, 40),
    "member_handbook": (10, 20),
}

# W15 additional document types
W15_SUPPLEMENT_DOC_TYPES: dict[str, tuple[int, int]] = {
    "coverage_comparison": (10, 15),
    "appeals_handbook": (8, 12),
    "amendment_rider": (3, 5),
}

# W16 tier definitions
W16_TIERS: dict[str, dict[str, Any]] = {
    "easy": {
        "profiling_pages": (3, 10),
        "gt_pages": (5, 15),
        "sections": (3, 5),
        "doc_types": ["research_paper", "technical_spec"],
    },
    "medium": {
        "profiling_pages": (12, 30),
        "gt_pages": (18, 45),
        "sections": (6, 12),
        "doc_types": ["annual_report", "regulatory_filing"],
    },
    "hard": {
        "profiling_pages": (35, 65),
        "gt_pages": (50, 85),
        "sections": (13, 18),
        "doc_types": ["annual_report", "regulatory_filing"],
    },
    "edge": {
        "profiling_pages": (1, 100),
        "gt_pages": (1, 100),
        "sections": (1, 20),
        "doc_types": [
            "annual_report",
            "regulatory_filing",
            "research_paper",
            "meeting_transcript",
            "technical_spec",
        ],
    },
    "extreme": {
        "profiling_pages": None,
        "gt_pages": (80, 100),
        "sections": (15, 20),
        "doc_types": ["regulatory_filing"],
    },
}

# W18 tier definitions (token-based)
W18_TIERS: dict[str, dict[str, Any]] = {
    "easy": {
        "profiling_tokens": (30_000, 40_000),
        "gt_tokens": (30_000, 50_000),
        "pages": (30, 45),
    },
    "medium": {
        "profiling_tokens": (40_000, 60_000),
        "gt_tokens": (50_000, 75_000),
        "pages": (45, 70),
    },
    "hard": {
        "profiling_tokens": (60_000, 80_000),
        "gt_tokens": (70_000, 95_000),
        "pages": (70, 90),
    },
    "edge": {
        "profiling_tokens": (80_000, 100_000),
        "gt_tokens": (85_000, 100_000),
        "pages": (90, 100),
    },
    "extreme": {
        "profiling_tokens": None,
        "gt_tokens": (90_000, 100_000),
        "pages": (95, 100),
    },
}
