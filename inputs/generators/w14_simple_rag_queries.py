"""Input generator for W14: Simple RAG queries against PDF insurance corpus.

Generate factoid, comparison, synthesis, and edge queries that target
specific facts from the corpus manifest.  Falls back to hardcoded
insurance-domain query pools when the manifest is unavailable.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

# ---------------------------------------------------------------------------
# Token-length targets  (chars ≈ tokens * 4)
# ---------------------------------------------------------------------------
_TOKEN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "profiling": {
        "easy":   (15, 40),
        "medium": (30, 80),
        "hard":   (40, 120),
        "edge":   (5, 500),
    },
    "ground_truth": {
        "easy":     (20, 60),
        "medium":   (50, 130),
        "hard":     (60, 200),
        "edge":     (5, 600),
        "extreme":  (80, 300),
    },
}

# ---------------------------------------------------------------------------
# Copy-paste dirty artifacts
# ---------------------------------------------------------------------------
_COPY_PASTE_ARTIFACTS = [
    "\n\nSent from my iPhone",
    "\n\n> On Mon, Jan 5, 2026, John wrote:\n> What is the deductible?",
    "\n\n---\nThis email and any attachments are confidential.",
    "\n\nFYI — forwarding from the customer portal.",
    "\n\n[image: company_logo.png]",
]

# ---------------------------------------------------------------------------
# Fallback query pools (used when manifest is unavailable)
# ---------------------------------------------------------------------------
_FALLBACK_FACTOID = [
    "What is the in-network deductible for United Healthcare?",
    "What is the emergency room copay for Aetna?",
    "Does Blue Cross Blue Shield cover mental health visits?",
    "What is the out-of-pocket maximum for United Healthcare?",
    "How many prescription drug tiers does Aetna have?",
    "What is the copay for a primary care visit under Blue Cross?",
    "Is prior authorization required for MRI scans with United Healthcare?",
    "What is the coinsurance rate for specialist visits under Aetna?",
    "Does United Healthcare cover telehealth visits?",
    "What is the annual limit on physical therapy visits for Blue Cross?",
    "What is the generic drug copay for United Healthcare formulary?",
    "How long is the waiting period for pre-existing conditions under Aetna?",
    "What preventive care services are covered at no cost?",
    "What is the out-of-network deductible for Blue Cross?",
    "Does United Healthcare offer dental coverage as part of the base plan?",
]

_FALLBACK_COMPARISON = [
    "Compare emergency room copays between United Healthcare and Aetna.",
    "How do in-network deductibles differ between all three providers?",
    "Which provider has the lowest out-of-pocket maximum?",
    "Compare mental health coverage limits across United Healthcare and Blue Cross.",
    "How do prescription drug tier structures compare between Aetna and United Healthcare?",
    "Which plan offers better out-of-network coverage?",
    "Compare the prior authorization requirements between all providers.",
    "How do specialist visit copays differ between Blue Cross and Aetna?",
    "Compare telehealth coverage and copays across all three plans.",
    "Which provider has the most comprehensive preventive care coverage?",
]

_FALLBACK_SYNTHESIS = [
    ("For a family of four with one member needing regular specialist visits and "
     "another on multiple prescriptions, which provider offers the best overall value?"),
    ("Analyze the total annual cost exposure for a patient requiring monthly MRI scans "
     "across all three providers, including prior authorization delays."),
    ("Compare the appeal processes, required documentation, and timelines across all "
     "three providers for a denied specialist referral."),
    ("If an employee switches from United Healthcare to Aetna mid-year, what coverage "
     "gaps should they expect during the transition?"),
    ("Evaluate which provider offers the most comprehensive coverage for a chronic "
     "condition requiring monthly specialist visits, quarterly lab work, and daily medication."),
    ("Assess the financial impact of an unexpected ER visit followed by a 3-day hospital "
     "stay under each provider's plan."),
    ("Compare how each provider handles out-of-state emergency care and what additional "
     "costs a member might face."),
    ("Synthesize the grievance and appeals rights across all documents and identify which "
     "provider offers the most member-friendly process."),
]

_FALLBACK_EDGE = [
    "",
    "???",
    "Cual es el deducible de United Healthcare?",
    "a" * 5000,
    "What is the coverage for experimental gene therapy using CRISPR-Cas9?",
]

_FALLBACK_EXTREME = [
    ("Provide a comprehensive comparison of every benefit category, copay, coinsurance "
     "rate, deductible, out-of-pocket maximum, prior authorization requirement, appeal "
     "process, and exclusion across all providers and all document types in the corpus."),
    ("For each provider, list every key fact mentioned in every section of every document, "
     "organized by provider and document type."),
    ("Create a decision matrix for choosing between all available plans considering: "
     "chronic illness management, emergency care frequency, prescription drug needs, "
     "mental health utilization, preventive care preferences, and out-of-network travel needs."),
]


class SimpleRagQueryGenerator(BaseInputGenerator):
    """Generate simple RAG queries targeting the PDF insurance corpus (W14)."""

    workflow_id = "W14"
    dirty_types = ["copy_pasted_artifacts", "near_empty"]

    _token_ranges_lookup = _TOKEN_RANGES

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True
        self._manifest: dict[str, Any] | None = None
        self._manifest_loaded = False

    def get_rewritable_text(self, inp: GeneratedInput) -> str | None:
        return "query"

    def get_llm_instruction(self, inp: GeneratedInput) -> str:
        qt = inp.structural_descriptor.get("query_type", "factoid")
        return (
            f"Generate a {inp.tier}-difficulty health insurance query ({qt}). "
            f"Topics: deductibles, copays, coverage, exclusions, appeals. "
            f"Providers: United Healthcare, Aetna, Cigna."
        )

    def _load_manifest(self, profile: str) -> dict[str, Any] | None:
        """Load corpus manifest if available."""
        if self._manifest_loaded:
            return self._manifest

        self._manifest_loaded = True
        manifest_path = Path(f"pdfs/generated/{profile}/w14_w15_corpus/pdfs/manifest.json")
        if not manifest_path.exists():
            # Try common alternate locations
            alt = Path(f"pdfs/generated/profiling/w14_w15_corpus/pdfs/manifest.json")
            if alt.exists():
                manifest_path = alt
            else:
                return None

        try:
            self._manifest = json.loads(manifest_path.read_text())
            return self._manifest
        except (json.JSONDecodeError, OSError):
            return None

    def _extract_providers(self, manifest: dict[str, Any]) -> list[str]:
        """Extract unique provider names from manifest."""
        providers: set[str] = set()
        for doc in manifest.get("documents", []):
            if "provider" in doc:
                providers.add(doc["provider"])
        return sorted(providers)

    def _extract_facts(
        self, manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Extract all key facts with their document and section context."""
        facts = []
        for doc in manifest.get("documents", []):
            provider = doc.get("provider", "Unknown")
            doc_type = doc.get("document_type", "unknown")
            for section in doc.get("sections", []):
                section_title = section.get("title", "")
                for fact in section.get("key_facts", []):
                    facts.append({
                        "fact": fact,
                        "provider": provider,
                        "document_type": doc_type,
                        "section": section_title,
                        "pdf_id": doc.get("pdf_id", ""),
                    })
        return facts

    def _generate_factoid_query(
        self, rng: random.Random, manifest: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a single-fact lookup query."""
        if manifest is None:
            query = rng.choice(_FALLBACK_FACTOID)
            return query, {
                "query_type": "factoid",
                "expected_chunk_count": 1,
                "target_providers": [],
                "answerable": True,
            }

        facts = self._extract_facts(manifest)
        if not facts:
            query = rng.choice(_FALLBACK_FACTOID)
            return query, {
                "query_type": "factoid",
                "expected_chunk_count": 1,
                "target_providers": [],
                "answerable": True,
            }

        fact = rng.choice(facts)
        # Parse fact string like "in_network_deductible: $1,500"
        key = fact["fact"].split(":")[0].strip().replace("_", " ")

        templates = [
            f"What is the {key} for {fact['provider']}?",
            f"Can you tell me the {key} under the {fact['provider']} plan?",
            f"Look up the {key} in the {fact['provider']} {fact['document_type']}.",
            f"What does {fact['provider']} list as the {key}?",
            f"Find the {key} information from {fact['provider']}.",
        ]
        query = rng.choice(templates)
        descriptor = {
            "query_type": "factoid",
            "expected_chunk_count": 1,
            "target_providers": [fact["provider"]],
            "answerable": True,
        }
        return query, descriptor

    def _generate_comparison_query(
        self, rng: random.Random, manifest: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a query comparing facts across 2-3 providers/documents."""
        if manifest is None:
            query = rng.choice(_FALLBACK_COMPARISON)
            return query, {
                "query_type": "comparison",
                "expected_chunk_count": rng.randint(2, 4),
                "target_providers": [],
                "answerable": True,
            }

        providers = self._extract_providers(manifest)
        facts = self._extract_facts(manifest)
        if len(providers) < 2 or not facts:
            query = rng.choice(_FALLBACK_COMPARISON)
            return query, {
                "query_type": "comparison",
                "expected_chunk_count": rng.randint(2, 4),
                "target_providers": [],
                "answerable": True,
            }

        # Find a fact key that appears for multiple providers
        fact_keys: dict[str, list[str]] = {}
        for f in facts:
            key = f["fact"].split(":")[0].strip().replace("_", " ")
            fact_keys.setdefault(key, [])
            if f["provider"] not in fact_keys[key]:
                fact_keys[key].append(f["provider"])

        shared_keys = [(k, v) for k, v in fact_keys.items() if len(v) >= 2]
        if not shared_keys:
            query = rng.choice(_FALLBACK_COMPARISON)
            return query, {
                "query_type": "comparison",
                "expected_chunk_count": rng.randint(2, 4),
                "target_providers": providers[:2],
                "answerable": True,
            }

        key, key_providers = rng.choice(shared_keys)
        target = rng.sample(key_providers, min(rng.randint(2, 3), len(key_providers)))

        templates = [
            f"Compare the {key} between {' and '.join(target)}.",
            f"How do {' and '.join(target)} differ on {key}?",
            f"What are the differences in {key} across {', '.join(target)}?",
        ]
        query = rng.choice(templates)
        return query, {
            "query_type": "comparison",
            "expected_chunk_count": len(target),
            "target_providers": target,
            "answerable": True,
        }

    def _generate_synthesis_query(
        self, rng: random.Random, manifest: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a query requiring synthesis across 3+ documents."""
        if manifest is None:
            query = rng.choice(_FALLBACK_SYNTHESIS)
            return query, {
                "query_type": "synthesis",
                "expected_chunk_count": rng.randint(4, 8),
                "target_providers": [],
                "answerable": True,
            }

        providers = self._extract_providers(manifest)
        facts = self._extract_facts(manifest)

        if len(providers) < 2:
            query = rng.choice(_FALLBACK_SYNTHESIS)
            return query, {
                "query_type": "synthesis",
                "expected_chunk_count": rng.randint(4, 8),
                "target_providers": providers,
                "answerable": True,
            }

        # Pick multiple fact keys across providers
        sampled_facts = rng.sample(facts, min(4, len(facts)))
        keys_mentioned = list({f["fact"].split(":")[0].strip().replace("_", " ")
                               for f in sampled_facts})
        providers_mentioned = list({f["provider"] for f in sampled_facts})

        templates = [
            (f"Analyze the {', '.join(keys_mentioned[:3])} across "
             f"{' and '.join(providers_mentioned)} and recommend the best option "
             f"for a patient with chronic conditions."),
            (f"Synthesize information about {' and '.join(keys_mentioned[:2])} from "
             f"all providers to determine which plan minimizes total out-of-pocket costs."),
            (f"Compare {', '.join(keys_mentioned[:3])} across {', '.join(providers_mentioned)} "
             f"and explain the trade-offs for different patient profiles."),
        ]
        query = rng.choice(templates)
        return query, {
            "query_type": "synthesis",
            "expected_chunk_count": rng.randint(4, 8),
            "target_providers": providers_mentioned,
            "answerable": True,
        }

    def _generate_edge_query(
        self, rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        """Generate an edge-case query (unanswerable, non-English, etc.)."""
        query = rng.choice(_FALLBACK_EDGE)
        answerable = bool(query.strip()) and len(query) < 1000 and query.isascii()
        return query, {
            "query_type": "edge",
            "expected_chunk_count": 0,
            "target_providers": [],
            "answerable": answerable,
        }

    def _generate_extreme_query(
        self, rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a broad query spanning all documents."""
        query = rng.choice(_FALLBACK_EXTREME)
        return query, {
            "query_type": "synthesis",
            "expected_chunk_count": rng.randint(8, 15),
            "target_providers": [],
            "answerable": True,
        }

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one RAG query input."""
        manifest = self._load_manifest(profile)
        corpus_path = f"pdfs/generated/{profile}/w14_w15_corpus/"

        if tier == "easy":
            query, descriptor = self._generate_factoid_query(rng, manifest)
        elif tier == "medium":
            query, descriptor = self._generate_comparison_query(rng, manifest)
        elif tier == "hard":
            query, descriptor = self._generate_synthesis_query(rng, manifest)
        elif tier == "extreme":
            query, descriptor = self._generate_extreme_query(rng)
        else:
            query, descriptor = self._generate_edge_query(rng)

        # Pad/truncate query to fit the target token range for this tier+profile
        ranges = _TOKEN_RANGES.get(profile, _TOKEN_RANGES["profiling"])
        if tier in ranges:
            tmin, tmax = ranges[tier]

            query = self.pad_to_token_range(query, tmin, tmax, rng)

        # Apply dirty input
        if is_dirty:
            if dirty_type == "copy_pasted_artifacts":
                artifact = rng.choice(_COPY_PASTE_ARTIFACTS)
                query = query + artifact
            elif dirty_type == "near_empty":
                query = rng.choice(["", " ", "?", "...", "hi"])

        # Apply style shift for GT
        query = self.apply_style_shift(query, profile)

        token_count = self.estimate_tokens(query)

        input_data: dict[str, Any] = {
            "query": query,
            "corpus_path": corpus_path,
            "input": query,
            "expected_chunk_count": descriptor["expected_chunk_count"],
        }

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W14",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=descriptor,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(SimpleRagQueryGenerator)
