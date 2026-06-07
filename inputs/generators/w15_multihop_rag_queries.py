"""Input generator for W15: Multi-hop RAG queries against PDF insurance corpus.

Generate queries requiring iterative retrieval (1-4 hops) with increasing
complexity.  Shares the same corpus dependency as W14 but produces compound
queries that cannot be answered from a single retrieval step.
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
        "edge":   (20, 150),
    },
    "ground_truth": {
        "easy":     (20, 60),
        "medium":   (50, 130),
        "hard":     (60, 200),
        "edge":     (20, 200),
        "extreme":  (80, 300),
    },
}

# ---------------------------------------------------------------------------
# Hop counts per tier
# ---------------------------------------------------------------------------
_HOP_COUNTS: dict[str, int] = {
    "easy":    1,
    "medium":  2,
    "hard":    3,
    "edge":    4,
    "extreme": 4,
}

# ---------------------------------------------------------------------------
# Copy-paste dirty artifacts
# ---------------------------------------------------------------------------
_COPY_PASTE_ARTIFACTS = [
    "\n\n-- Forwarded message --\nFrom: benefits@company.com",
    "\n\nSent from my iPad",
    "\n\n> Original question below:\n> (see above)",
    "\n\n[Attachment: screenshot_2026.png]",
]

# ---------------------------------------------------------------------------
# Information spread mapping
# ---------------------------------------------------------------------------
_SPREAD_BY_TIER: dict[str, str] = {
    "easy":    "single_section",
    "medium":  "multi_section",
    "hard":    "multi_document",
    "edge":    "multi_document",
    "extreme": "multi_document",
}

# ---------------------------------------------------------------------------
# Fallback query templates by hop count
# ---------------------------------------------------------------------------
_FALLBACK_1HOP = [
    "What is Aetna's copay for ER visits?",
    "What is the in-network deductible for United Healthcare?",
    "Does Blue Cross Blue Shield require prior authorization for MRI?",
    "What is the out-of-pocket maximum listed in the Aetna SBC?",
    "How many prescription drug tiers does United Healthcare have?",
    "What mental health visit limits does Blue Cross impose?",
    "What is the coinsurance rate for outpatient surgery under Aetna?",
    "What preventive care services are covered at no cost by United Healthcare?",
    "What is the generic drug copay in the Aetna formulary?",
    "Does Blue Cross cover telehealth visits, and what is the copay?",
    "What is the specialist visit copay for United Healthcare?",
    "What urgent care copay does Aetna charge for in-network visits?",
    "What is the out-of-network deductible for Blue Cross?",
    "How many days supply does United Healthcare allow for mail-order prescriptions?",
    "What is the hospital admission copay for Aetna members?",
]

_FALLBACK_2HOP = [
    ("What is United Healthcare's ER copay, and does it get waived if the "
     "patient is admitted to the hospital?"),
    ("Look up Aetna's prescription formulary tiers and then check whether "
     "the SBC mentions any additional drug cost-sharing provisions."),
    ("Find the mental health visit limit for Blue Cross, then determine "
     "what happens after that limit is reached according to the appeals section."),
    ("What is the prior authorization turnaround time for MRI at United Healthcare, "
     "and what section of the SBC describes the appeals process if authorization is denied?"),
    ("Compare the in-network deductible in the SBC with any deductible "
     "information mentioned in the formulary for Aetna."),
    ("Find the out-of-pocket maximum for United Healthcare and then check "
     "whether prescription drug costs count toward that maximum."),
    ("What specialist visit copay does Blue Cross charge, and do referrals "
     "from a PCP reduce that copay?"),
    ("Look up the preventive care coverage for Aetna and check if there are "
     "any age-specific exclusions mentioned in the excluded services section."),
    ("What is United Healthcare's coinsurance for outpatient surgery, and does "
     "the plan require a second opinion before approving the procedure?"),
    ("Find Blue Cross's generic drug copay and then check the formulary for "
     "which specific medications fall under the generic tier."),
]

_FALLBACK_3HOP = [
    ("If a United Healthcare member is denied MRI coverage, what are the "
     "appeal steps, required documentation, and timeline?"),
    ("For an Aetna member needing monthly specialist visits plus daily medication, "
     "calculate the annual copay burden by cross-referencing the specialist copay, "
     "prescription tier structure, and out-of-pocket maximum."),
    ("Trace the full patient journey for a Blue Cross member seeking mental health "
     "treatment: find the visit limit, check what happens at the limit, and identify "
     "the appeals process for extending coverage."),
    ("A United Healthcare member needs an out-of-network specialist. Find the "
     "out-of-network deductible, the coinsurance rate after deductible, and whether "
     "balance billing protections exist in the plan."),
    ("For a patient switching from Aetna to Blue Cross mid-year, compare the "
     "deductible reset policies, check if any accumulated costs transfer, and "
     "identify any waiting period for pre-existing conditions."),
    ("Determine the total cost of a 3-day hospital stay under United Healthcare "
     "by combining the admission copay, daily coinsurance, and checking whether "
     "the out-of-pocket maximum caps the total exposure."),
    ("Find Aetna's prior authorization requirements for specialty drugs, check "
     "the formulary for step therapy requirements, and identify the appeals "
     "process if the preferred drug causes adverse effects."),
    ("Analyze Blue Cross coverage for a pregnancy from prenatal through delivery: "
     "find preventive care coverage for checkups, hospital admission copay for "
     "delivery, and any postnatal care limits."),
]

_FALLBACK_4HOP = [
    ("For a family of four enrolled in United Healthcare where one child has asthma, "
     "the other needs orthodontics, the mother requires quarterly specialist visits, "
     "and the father is on three daily medications: find each person's primary cost "
     "driver across the SBC, formulary, and provider directory, then calculate the "
     "combined annual out-of-pocket exposure."),
    ("A Blue Cross member has a complex chronic condition requiring: monthly specialist "
     "visits (check copay), quarterly MRI scans (check prior auth), daily Tier 3 "
     "medication (check formulary cost), and an annual hospital procedure (check "
     "inpatient coinsurance). Cross-reference all four cost components and determine "
     "if the out-of-pocket maximum provides relief."),
    ("Compare the end-to-end appeals process across all three providers for a denied "
     "claim: find the initial denial notification timeline, internal appeal procedures, "
     "external review rights, and state regulatory complaint options for each."),
    ("Evaluate which provider is best for someone who: travels frequently out of state "
     "(check out-of-network coverage), uses telehealth regularly (check telehealth copay), "
     "needs specialty medications (check formulary tiers), and wants comprehensive "
     "preventive care (check preventive benefits). Score each provider across all four "
     "dimensions."),
    ("Trace the cost implications of a medical emergency while traveling: ER visit "
     "copay/coinsurance (possibly out-of-network), ambulance coverage, hospital admission, "
     "and follow-up specialist care back home. Calculate for each provider and determine "
     "which plan minimizes the total financial exposure."),
]

_FALLBACK_4HOP_UNANSWERABLE = [
    ("What is the coverage for CRISPR gene therapy clinical trials, including the "
     "experimental drug formulary classification, the prior authorization pathway "
     "for investigational procedures, the appeals process for denied experimental "
     "coverage, and the relevant state mandates?"),
    ("Find the pediatric dental orthodontic coverage limits, cross-reference with "
     "the orthodontic provider network directory, check if orthodontic appliances "
     "are covered under durable medical equipment, and determine the coordination "
     "of benefits with a secondary dental plan."),
]


class MultihopRagQueryGenerator(BaseInputGenerator):
    """Generate multi-hop RAG queries requiring iterative retrieval (W15)."""

    workflow_id = "W15"
    dirty_types = ["copy_pasted_artifacts"]

    _token_ranges_lookup = _TOKEN_RANGES

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True
        self._manifest: dict[str, Any] | None = None
        self._manifest_loaded = False

    def get_rewritable_text(self, inp: GeneratedInput) -> str | None:
        return "query"

    def get_llm_instruction(self, inp: GeneratedInput) -> str:
        hops = inp.structural_descriptor.get("expected_hop_count", 1)
        return (
            f"Generate a {inp.tier}-difficulty multi-hop insurance query "
            f"requiring {hops} retrieval hops across multiple document sections."
        )

    def _load_manifest(self, profile: str) -> dict[str, Any] | None:
        """Load corpus manifest if available."""
        if self._manifest_loaded:
            return self._manifest

        self._manifest_loaded = True
        manifest_path = Path(f"pdfs/generated/{profile}/w14_w15_corpus/pdfs/manifest.json")
        if not manifest_path.exists():
            alt = Path("pdfs/generated/profiling/w14_w15_corpus/pdfs/manifest.json")
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
        providers: set[str] = set()
        for doc in manifest.get("documents", []):
            if "provider" in doc:
                providers.add(doc["provider"])
        return sorted(providers)

    def _extract_facts(
        self, manifest: dict[str, Any],
    ) -> list[dict[str, Any]]:
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

    def _generate_1hop(
        self, rng: random.Random, manifest: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a 1-hop query answerable from single retrieval."""
        if manifest:
            facts = self._extract_facts(manifest)
            if facts:
                fact = rng.choice(facts)
                key = fact["fact"].split(":")[0].strip().replace("_", " ")
                templates = [
                    f"What is the {key} for {fact['provider']}?",
                    f"Find the {key} in the {fact['provider']} {fact['document_type']}.",
                    f"Look up {fact['provider']}'s {key}.",
                ]
                query = rng.choice(templates)
                return query, {
                    "expected_hop_count": 1,
                    "query_type": "factoid",
                    "information_spread": "single_section",
                    "target_providers": [fact["provider"]],
                    "answerable": True,
                }

        query = rng.choice(_FALLBACK_1HOP)
        return query, {
            "expected_hop_count": 1,
            "query_type": "factoid",
            "information_spread": "single_section",
            "target_providers": [],
            "answerable": True,
        }

    def _generate_2hop(
        self, rng: random.Random, manifest: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a 2-hop query requiring cross-referencing 2 sources."""
        if manifest:
            facts = self._extract_facts(manifest)
            providers = self._extract_providers(manifest)
            if len(facts) >= 2:
                pair = rng.sample(facts, 2)
                keys = [f["fact"].split(":")[0].strip().replace("_", " ") for f in pair]
                target_providers = list({f["provider"] for f in pair})

                templates = [
                    (f"Find the {keys[0]} for {pair[0]['provider']} and then check "
                     f"the {keys[1]} in the {pair[1]['document_type']}."),
                    (f"What is {pair[0]['provider']}'s {keys[0]}, and how does it "
                     f"relate to the {keys[1]} listed in the {pair[1]['section']} section?"),
                    (f"Look up the {keys[0]} from {pair[0]['provider']}'s "
                     f"{pair[0]['document_type']} and cross-reference it with "
                     f"the {keys[1]} from {pair[1]['provider']}."),
                ]
                query = rng.choice(templates)
                return query, {
                    "expected_hop_count": 2,
                    "query_type": "cross_reference",
                    "information_spread": "multi_section",
                    "target_providers": target_providers,
                    "answerable": True,
                }

        query = rng.choice(_FALLBACK_2HOP)
        return query, {
            "expected_hop_count": 2,
            "query_type": "cross_reference",
            "information_spread": "multi_section",
            "target_providers": [],
            "answerable": True,
        }

    def _generate_3hop(
        self, rng: random.Random, manifest: dict[str, Any] | None,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a 3-hop query requiring synthesis across many sections."""
        if manifest:
            facts = self._extract_facts(manifest)
            providers = self._extract_providers(manifest)
            if len(facts) >= 3 and len(providers) >= 2:
                triple = rng.sample(facts, 3)
                keys = [f["fact"].split(":")[0].strip().replace("_", " ") for f in triple]
                target_providers = list({f["provider"] for f in triple})

                query = (
                    f"For a patient using {target_providers[0]}, find the {keys[0]}, "
                    f"then check the {keys[1]} in the {triple[1]['document_type']}"
                )
                if len(target_providers) > 1:
                    query += (
                        f", and compare with {target_providers[-1]}'s {keys[2]} "
                        f"from the {triple[2]['section']} section"
                    )
                query += "."
                return query, {
                    "expected_hop_count": 3,
                    "query_type": "synthesis",
                    "information_spread": "multi_document",
                    "target_providers": target_providers,
                    "answerable": True,
                }

        query = rng.choice(_FALLBACK_3HOP)
        return query, {
            "expected_hop_count": 3,
            "query_type": "synthesis",
            "information_spread": "multi_document",
            "target_providers": [],
            "answerable": True,
        }

    def _generate_4hop_unanswerable(
        self, rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a 4-hop unanswerable query (edge tier)."""
        query = rng.choice(_FALLBACK_4HOP_UNANSWERABLE)
        return query, {
            "expected_hop_count": 4,
            "query_type": "unanswerable",
            "information_spread": "multi_document",
            "target_providers": [],
            "answerable": False,
        }

    def _generate_4hop_extreme(
        self, rng: random.Random,
    ) -> tuple[str, dict[str, Any]]:
        """Generate a 4-hop query with substantial context per hop (extreme tier)."""
        query = rng.choice(_FALLBACK_4HOP)
        return query, {
            "expected_hop_count": 4,
            "query_type": "synthesis",
            "information_spread": "multi_document",
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
        """Generate one multi-hop RAG query."""
        manifest = self._load_manifest(profile)
        corpus_path = f"pdfs/generated/{profile}/w14_w15_corpus/"

        if tier == "easy":
            query, descriptor = self._generate_1hop(rng, manifest)
        elif tier == "medium":
            query, descriptor = self._generate_2hop(rng, manifest)
        elif tier == "hard":
            query, descriptor = self._generate_3hop(rng, manifest)
        elif tier == "edge":
            query, descriptor = self._generate_4hop_unanswerable(rng)
        elif tier == "extreme":
            query, descriptor = self._generate_4hop_extreme(rng)
        else:
            query, descriptor = self._generate_1hop(rng, manifest)

        # Pad/truncate query to fit the target token range for this tier+profile
        ranges = _TOKEN_RANGES.get(profile, _TOKEN_RANGES["profiling"])
        if tier in ranges:
            tmin, tmax = ranges[tier]

            query = self.pad_to_token_range(query, tmin, tmax, rng)

        # Apply dirty input
        if is_dirty and dirty_type == "copy_pasted_artifacts":
            artifact = rng.choice(_COPY_PASTE_ARTIFACTS)
            query = query + artifact

        # Apply style shift for GT
        query = self.apply_style_shift(query, profile)

        token_count = self.estimate_tokens(query)

        input_data: dict[str, Any] = {
            "query": query,
            "corpus_path": corpus_path,
            "input": query,
            "expected_chunk_count": descriptor.get("expected_hop_count", 1),
        }

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W15",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=descriptor,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(MultihopRagQueryGenerator)
