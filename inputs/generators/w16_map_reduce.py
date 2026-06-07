"""Thin wrapper around existing PDF descriptors for W16 (Map-Reduce Summarization).

Read JSON descriptors from ``pdfs/generated/{profile}/w16/`` and wrap them in
the tagged input format. Falls back to stub generation when descriptor
directories are missing (dry-run / CI).
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

logger = logging.getLogger(__name__)

_PDF_BASE = Path(__file__).resolve().parents[2] / "pdfs" / "generated"

_PROFILE_DIR_MAP = {
    "profiling": "profiling",
    "ground_truth": "ground_truth",
}


def _classify_tier_w16(page_count: int, section_count: int, structure_quality: str) -> str:
    """Assign a tier based on page count, section count, and structure quality."""
    if page_count >= 80 and structure_quality == "poorly_structured":
        return "extreme"
    if page_count == 1 or 80 <= page_count <= 100:
        return "edge"
    if 30 <= page_count <= 80 and 12 <= section_count <= 18:
        return "hard"
    if 10 <= page_count <= 40 and 6 <= section_count <= 10:
        return "medium"
    if 3 <= page_count <= 12 and 3 <= section_count <= 4:
        return "easy"

    # Fallback: use page_count ranges when section_count doesn't match neatly
    if page_count <= 12:
        return "easy"
    if page_count <= 40:
        return "medium"
    if page_count <= 80:
        return "hard"
    return "edge"


def _structural_descriptor(desc: dict[str, Any]) -> dict[str, Any]:
    """Build a structural descriptor from a PDF descriptor."""
    return {
        "page_count": desc["page_count"],
        "expected_section_count": desc["section_count"],
        "has_clear_structure": desc.get("structure_quality") == "well_structured",
        "content_type": desc.get("document_type", "annual_report"),
    }


def _stub_descriptor(rng: random.Random, tier: str, idx: int) -> dict[str, Any]:
    """Generate a stub PDF descriptor when real files are unavailable."""
    tier_ranges = {
        "easy": {"pages": (3, 12), "sections": (3, 4)},
        "medium": {"pages": (10, 40), "sections": (6, 10)},
        "hard": {"pages": (30, 80), "sections": (12, 18)},
        "edge": {"pages": (80, 100), "sections": (5, 15)},
        "extreme": {"pages": (80, 100), "sections": (5, 10)},
    }
    ranges = tier_ranges.get(tier, tier_ranges["medium"])
    page_count = rng.randint(*ranges["pages"])
    section_count = rng.randint(*ranges["sections"])

    return {
        "pdf_id": f"w16-stub-{tier}-{idx:04d}",
        "workflow": "w16",
        "document_type": "annual_report",
        "page_count": page_count,
        "estimated_token_count": page_count * rng.randint(400, 800),
        "section_count": section_count,
        "structure_quality": (
            "poorly_structured" if tier == "extreme" else "well_structured"
        ),
        "content_density": rng.choice(["dense", "mixed", "sparse"]),
    }


class W16MapReduceGenerator(BaseInputGenerator):
    """Wrap existing W16 PDF descriptors as tagged inputs."""

    workflow_id = "W16"
    dirty_types: list[str] = []  # W16 not listed in dirty input table

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._descriptors: dict[str, list[dict[str, Any]]] = {}

    def _load_descriptors(self, profile: str) -> list[dict[str, Any]]:
        """Load and cache PDF descriptors for a profile."""
        if profile in self._descriptors:
            return self._descriptors[profile]

        dir_name = _PROFILE_DIR_MAP.get(profile, profile)
        desc_dir = _PDF_BASE / dir_name / "w16"

        if not desc_dir.exists():
            logger.warning(
                "PDF descriptor directory %s not found; falling back to stub generation",
                desc_dir,
            )
            self._descriptors[profile] = []
            return []

        descriptors = []
        for json_path in sorted(desc_dir.glob("*.json")):
            try:
                data = json.loads(json_path.read_text())
                descriptors.append(data)
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Skipping %s: %s", json_path, exc)

        self._descriptors[profile] = descriptors
        return descriptors

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Wrap one PDF descriptor (or stub) as a GeneratedInput."""
        descriptors = self._load_descriptors(profile)

        if descriptors:
            # Pick a descriptor matching the tier if possible, otherwise any
            tier_matches = [
                d for d in descriptors
                if _classify_tier_w16(
                    d["page_count"], d["section_count"],
                    d.get("structure_quality", "well_structured"),
                ) == tier
            ]
            if tier_matches:
                desc = rng.choice(tier_matches)
            else:
                desc = rng.choice(descriptors)
        else:
            desc = _stub_descriptor(rng, tier, idx)

        struct_desc = _structural_descriptor(desc)

        input_data = {
            "input": "<document text placeholder>",
            "pdf_id": desc["pdf_id"],
        }
        input_text = json.dumps(input_data)

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow=self.workflow_id,
            profile=profile,
            tier=tier,
            token_count=desc.get("estimated_token_count", self.estimate_tokens(input_text)),
            is_dirty=False,  # W16 has no dirty types
            dirty_type=None,
            structural_descriptor=struct_desc,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(W16MapReduceGenerator)
