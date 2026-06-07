"""Thin wrapper around existing PDF descriptors for W18 (Long Document QA).

Read JSON descriptors from ``pdfs/generated/{profile}/w18/`` and wrap them in
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


def _classify_tier_w18(
    estimated_token_count: int,
    page_count: int,
    structure_quality: str,
) -> str:
    """Assign a tier based on estimated token count."""
    t = estimated_token_count
    if t >= 90_000:
        return "extreme"
    if t >= 80_000:
        return "edge"
    if t >= 60_000:
        return "hard"
    if t >= 40_000:
        return "medium"
    if t >= 30_000:
        return "easy"

    # Below 30K: still classify by relative position
    if t >= 20_000:
        return "easy"
    return "easy"


def _structural_descriptor(desc: dict[str, Any]) -> dict[str, Any]:
    """Build a structural descriptor from a PDF descriptor."""
    return {
        "estimated_token_count": desc["estimated_token_count"],
        "page_count": desc["page_count"],
        "content_type": desc.get("document_type", "annual_report"),
        "has_tables": desc.get("table_chart_pages", 0) > 0,
        "has_numerical_data": desc.get("table_chart_pages", 0) > 0,
    }


def _stub_descriptor(rng: random.Random, tier: str, idx: int) -> dict[str, Any]:
    """Generate a stub PDF descriptor when real files are unavailable."""
    tier_token_ranges = {
        "easy": (30_000, 50_000),
        "medium": (40_000, 75_000),
        "hard": (60_000, 95_000),
        "edge": (80_000, 100_000),
        "extreme": (90_000, 100_000),
    }
    lo, hi = tier_token_ranges.get(tier, (40_000, 75_000))
    token_count = rng.randint(lo, hi)
    # Approximate pages from token count (~700 tokens per page)
    page_count = max(1, token_count // 700)
    table_pages = rng.randint(0, max(1, page_count // 5))

    return {
        "pdf_id": f"w18-stub-{tier}-{idx:04d}",
        "workflow": "w18",
        "document_type": "annual_report",
        "page_count": page_count,
        "estimated_token_count": token_count,
        "table_chart_pages": table_pages,
        "section_count": rng.randint(8, 20),
        "structure_quality": rng.choice(["well_structured", "poorly_structured"]),
        "content_density": rng.choice(["dense", "mixed", "sparse"]),
    }


def _apply_near_limit_dirty(
    desc: dict[str, Any], rng: random.Random
) -> dict[str, Any]:
    """Push token count toward the 100K limit for near-limit dirty inputs."""
    desc = dict(desc)
    target = rng.randint(95_000, 100_000)
    desc["estimated_token_count"] = target
    # Adjust page count proportionally
    desc["page_count"] = max(1, target // 700)
    return desc


class W18LongDocumentGenerator(BaseInputGenerator):
    """Wrap existing W18 PDF descriptors as tagged inputs."""

    workflow_id = "W18"
    dirty_types = ["near_limit"]

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed)
        self._descriptors: dict[str, list[dict[str, Any]]] = {}

    def _load_descriptors(self, profile: str) -> list[dict[str, Any]]:
        """Load and cache PDF descriptors for a profile."""
        if profile in self._descriptors:
            return self._descriptors[profile]

        dir_name = _PROFILE_DIR_MAP.get(profile, profile)
        desc_dir = _PDF_BASE / dir_name / "w18"

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
            # Pick a descriptor matching the tier if possible
            tier_matches = [
                d for d in descriptors
                if _classify_tier_w18(
                    d["estimated_token_count"],
                    d["page_count"],
                    d.get("structure_quality", "well_structured"),
                ) == tier
            ]
            if tier_matches:
                desc = rng.choice(tier_matches)
            else:
                desc = rng.choice(descriptors)
        else:
            desc = _stub_descriptor(rng, tier, idx)

        # Apply near-limit mutation for dirty inputs
        if is_dirty and dirty_type == "near_limit":
            desc = _apply_near_limit_dirty(desc, rng)

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
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=struct_desc,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(W18LongDocumentGenerator)
