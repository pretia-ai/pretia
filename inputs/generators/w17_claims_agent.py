"""Template-based input generator for W17 (Claims Processing Agent).

Generate structured JSON claims that trigger known pipeline behaviors:
intake short-circuits, standard processing, clinical review, appeals,
high-amount routing, and code mismatch routing. No LLM calls needed.
"""

from __future__ import annotations

import json
import random
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

# --- Claim field pools ---

_PROVIDERS = ["United Healthcare", "Aetna", "Cigna"]

_ACTIONS = [
    "intake_shortcircuit",
    "standard_adjudication",
    "pre_approval_clinical",
    "appeal_review",
    "high_amount_routing",
    "code_mismatch_routing",
]

# ICD-10 diagnosis codes paired with matching CPT procedure codes
_MATCHED_CODE_PAIRS: list[tuple[str, str]] = [
    ("M54.5", "72148"),   # low back pain → lumbar MRI
    ("I10", "93000"),     # hypertension → ECG
    ("E11.9", "83036"),   # type 2 diabetes → HbA1c
    ("J06.9", "99213"),   # upper respiratory infection → office visit
    ("K21.0", "43239"),   # GERD → upper GI endoscopy
    ("M17.11", "27447"),  # primary osteoarthritis knee → total knee replacement
    ("G43.909", "95819"), # migraine → EEG
    ("N18.3", "90960"),   # CKD stage 3 → dialysis
    ("C50.911", "19301"), # breast cancer → mastectomy
    ("F32.1", "90834"),   # major depressive disorder → psychotherapy
]

# Mismatched code pairs for edge cases (diagnosis and procedure are unrelated)
_MISMATCHED_CODE_PAIRS: list[tuple[str, str]] = [
    ("S52.501A", "72148"),  # forearm fracture → lumbar MRI
    ("J06.9", "27447"),     # upper respiratory infection → knee replacement
    ("E11.9", "19301"),     # diabetes → mastectomy
    ("I10", "43239"),       # hypertension → upper GI endoscopy
    ("M54.5", "90960"),     # low back pain → dialysis
]

_CLINICAL_NOTES = [
    "Patient presents with persistent symptoms despite conservative treatment over 6 weeks.",
    "Imaging reveals significant structural abnormality requiring surgical intervention.",
    "Lab results confirm diagnosis. Recommend continued pharmaceutical management.",
    "Follow-up visit shows improvement with current treatment plan. No changes indicated.",
    "Patient reports worsening symptoms. Specialist referral recommended for further evaluation.",
    "Post-operative recovery proceeding normally. Physical therapy initiated.",
]

_DENIAL_REASONS = [
    "Service not medically necessary per clinical guidelines.",
    "Pre-authorization not obtained prior to service.",
    "Provider out of network for member's plan.",
    "Duplicate claim submission.",
    "Benefit exhausted for this service category.",
]

_APPEAL_DOCS = [
    "Peer-reviewed literature supporting medical necessity attached.",
    "Updated clinical notes from treating physician with additional justification.",
    "Independent medical opinion letter from board-certified specialist.",
    "New diagnostic imaging results demonstrating clinical need.",
]

_SERVICE_DATE_TEMPLATES = [
    "2024-01-{day:02d}",
    "2024-03-{day:02d}",
    "2024-05-{day:02d}",
    "2024-07-{day:02d}",
    "2024-09-{day:02d}",
    "2024-11-{day:02d}",
]


def _make_claim_id(rng: random.Random) -> str:
    return f"CLM-2024-{rng.randint(100000, 999999)}"


def _make_member_id(rng: random.Random) -> str:
    return f"MEM-{rng.randint(100000, 999999)}"


def _make_service_date(rng: random.Random) -> str:
    template = rng.choice(_SERVICE_DATE_TEMPLATES)
    return template.format(day=rng.randint(1, 28))


class W17ClaimsAgentGenerator(BaseInputGenerator):
    """Generate structured JSON claims for the W17 claims-processing pipeline."""

    workflow_id = "W17"
    dirty_types = ["adversarial_routing"]

    @property
    def tier_weights(self) -> dict[str, dict[str, float]]:
        return {
            "profiling": {
                "easy": 0.15,
                "medium": 0.40,
                "hard": 0.30,
                "edge": 0.15,
            },
            "ground_truth": {
                "easy": 0.20,
                "medium": 0.35,
                "hard": 0.25,
                "edge": 0.15,
                "extreme": 0.05,
            },
        }

    def _build_easy_claim(self, rng: random.Random, idx: int) -> dict[str, Any]:
        """Inactive member or missing docs → intake short-circuit."""
        dx, cpt = rng.choice(_MATCHED_CODE_PAIRS)
        use_inactive = rng.random() < 0.5
        return {
            "claim_id": _make_claim_id(rng),
            "member_id": _make_member_id(rng),
            "member_status": "inactive" if use_inactive else "active",
            "claim_type": "standard",
            "provider": _PROVIDERS[idx % len(_PROVIDERS)],
            "diagnosis_code": dx,
            "procedure_code": cpt,
            "claimed_amount": round(rng.uniform(100, 1500), 2),
            "service_date": _make_service_date(rng),
            "clinical_note": None if not use_inactive else rng.choice(_CLINICAL_NOTES),
            "itemized_bill": None,  # missing required doc triggers short-circuit
            "prior_denial_reason": None,
            "appeal_documentation": None,
        }

    def _build_medium_claim(self, rng: random.Random, idx: int) -> dict[str, Any]:
        """Standard claim, all docs present, clear coverage."""
        dx, cpt = rng.choice(_MATCHED_CODE_PAIRS)
        return {
            "claim_id": _make_claim_id(rng),
            "member_id": _make_member_id(rng),
            "member_status": "active",
            "claim_type": "standard",
            "provider": _PROVIDERS[idx % len(_PROVIDERS)],
            "diagnosis_code": dx,
            "procedure_code": cpt,
            "claimed_amount": round(rng.uniform(200, 4500), 2),
            "service_date": _make_service_date(rng),
            "clinical_note": rng.choice(_CLINICAL_NOTES),
            "itemized_bill": f"Itemized bill for procedure {cpt}: service fee, facility fee, supplies.",
            "prior_denial_reason": None,
            "appeal_documentation": None,
        }

    def _build_hard_claim(self, rng: random.Random, idx: int) -> dict[str, Any]:
        """Pre-approval with clinical review OR appeal with new evidence."""
        is_appeal = rng.random() < 0.5
        dx, cpt = rng.choice(_MATCHED_CODE_PAIRS)

        if is_appeal:
            return {
                "claim_id": _make_claim_id(rng),
                "member_id": _make_member_id(rng),
                "member_status": "active",
                "claim_type": "appeal",
                "provider": _PROVIDERS[idx % len(_PROVIDERS)],
                "diagnosis_code": dx,
                "procedure_code": cpt,
                "claimed_amount": round(rng.uniform(1000, 4500), 2),
                "service_date": _make_service_date(rng),
                "clinical_note": rng.choice(_CLINICAL_NOTES),
                "itemized_bill": f"Itemized bill for procedure {cpt}: detailed line items.",
                "prior_denial_reason": rng.choice(_DENIAL_REASONS),
                "appeal_documentation": rng.choice(_APPEAL_DOCS),
            }

        return {
            "claim_id": _make_claim_id(rng),
            "member_id": _make_member_id(rng),
            "member_status": "active",
            "claim_type": "pre_approval",
            "provider": _PROVIDERS[idx % len(_PROVIDERS)],
            "diagnosis_code": dx,
            "procedure_code": cpt,
            "claimed_amount": round(rng.uniform(2000, 4500), 2),
            "service_date": _make_service_date(rng),
            "clinical_note": rng.choice(_CLINICAL_NOTES),
            "itemized_bill": f"Itemized bill for procedure {cpt}: pre-authorization request.",
            "prior_denial_reason": None,
            "appeal_documentation": None,
        }

    def _build_edge_claim(self, rng: random.Random, idx: int) -> dict[str, Any]:
        """High amount (>5000) or code mismatch → routing flag."""
        use_mismatch = rng.random() < 0.5

        if use_mismatch:
            dx, cpt = rng.choice(_MISMATCHED_CODE_PAIRS)
            amount = round(rng.uniform(500, 4500), 2)
        else:
            dx, cpt = rng.choice(_MATCHED_CODE_PAIRS)
            amount = round(rng.uniform(5001, 25000), 2)

        return {
            "claim_id": _make_claim_id(rng),
            "member_id": _make_member_id(rng),
            "member_status": "active",
            "claim_type": rng.choice(["standard", "pre_approval"]),
            "provider": _PROVIDERS[idx % len(_PROVIDERS)],
            "diagnosis_code": dx,
            "procedure_code": cpt,
            "claimed_amount": amount,
            "service_date": _make_service_date(rng),
            "clinical_note": rng.choice(_CLINICAL_NOTES),
            "itemized_bill": f"Itemized bill for procedure {cpt}: detailed breakdown.",
            "prior_denial_reason": None,
            "appeal_documentation": None,
        }

    def _build_extreme_claim(self, rng: random.Random, idx: int) -> dict[str, Any]:
        """Complex appeal + high amount + edge documentation (GT only)."""
        dx, cpt = rng.choice(_MISMATCHED_CODE_PAIRS)
        return {
            "claim_id": _make_claim_id(rng),
            "member_id": _make_member_id(rng),
            "member_status": "active",
            "claim_type": "appeal",
            "provider": _PROVIDERS[idx % len(_PROVIDERS)],
            "diagnosis_code": dx,
            "procedure_code": cpt,
            "claimed_amount": round(rng.uniform(8000, 50000), 2),
            "service_date": _make_service_date(rng),
            "clinical_note": rng.choice(_CLINICAL_NOTES),
            "itemized_bill": (
                f"Itemized bill for procedure {cpt}: multiple line items, "
                "facility fee, anesthesia, specialist consultation, imaging."
            ),
            "prior_denial_reason": rng.choice(_DENIAL_REASONS),
            "appeal_documentation": " ".join(
                rng.sample(_APPEAL_DOCS, min(3, len(_APPEAL_DOCS)))
            ),
        }

    def _structural_descriptor(self, claim: dict[str, Any], tier: str) -> dict[str, Any]:
        """Derive the expected structural descriptor from a claim."""
        overrides: list[str] = []
        short_circuit = False
        routing = False
        depth = "full_pipeline"

        if claim["member_status"] == "inactive":
            short_circuit = True
            depth = "intake_only"
        elif claim["itemized_bill"] is None:
            short_circuit = True
            depth = "intake_only"

        if claim["claimed_amount"] > 5000:
            overrides.append("high_amount")
            routing = True

        # Check for code mismatch
        matched_codes = {pair for pair in _MATCHED_CODE_PAIRS}
        pair = (claim["diagnosis_code"], claim["procedure_code"])
        if pair not in matched_codes:
            overrides.append("code_inconsistency")
            routing = True

        if routing and not short_circuit:
            depth = "full_pipeline_routed"

        if claim["claim_type"] == "pre_approval":
            depth = depth if routing else "full_pipeline_clinical"
        elif claim["claim_type"] == "appeal":
            depth = depth if routing else "full_pipeline_appeal"

        return {
            "claim_type": claim["claim_type"],
            "overrides_expected": overrides,
            "short_circuit_expected": short_circuit,
            "routing_expected": routing,
            "pipeline_depth": depth,
        }

    def _apply_adversarial_routing(
        self, claim: dict[str, Any], rng: random.Random
    ) -> dict[str, Any]:
        """Mutate a claim to test adversarial routing detection."""
        claim = dict(claim)
        mutation = rng.choice(["conflicting_status", "amount_boundary", "code_swap"])

        if mutation == "conflicting_status":
            # Active member with fields that suggest inactive processing
            claim["member_status"] = "active"
            claim["claimed_amount"] = 0.01
        elif mutation == "amount_boundary":
            # Amount exactly at the routing threshold
            claim["claimed_amount"] = 5000.00
        else:
            # Swap dx/cpt codes (put procedure code where diagnosis should be)
            claim["diagnosis_code"] = claim["procedure_code"]
            claim["procedure_code"] = claim["diagnosis_code"]

        return claim

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one claims input for the given tier."""
        builders = {
            "easy": self._build_easy_claim,
            "medium": self._build_medium_claim,
            "hard": self._build_hard_claim,
            "edge": self._build_edge_claim,
            "extreme": self._build_extreme_claim,
        }
        builder = builders.get(tier, self._build_medium_claim)
        claim = builder(rng, idx)

        if is_dirty and dirty_type == "adversarial_routing":
            claim = self._apply_adversarial_routing(claim, rng)

        descriptor = self._structural_descriptor(claim, tier)
        claim_text = json.dumps(claim, indent=2)

        if profile == "ground_truth":
            claim_text = self.apply_style_shift(claim_text, profile)

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow=self.workflow_id,
            profile=profile,
            tier=tier,
            token_count=self.estimate_tokens(claim_text),
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=descriptor,
            input_data=claim,
        )

    def generate_batch(
        self,
        profile: str,
        n: int,
    ) -> list[GeneratedInput]:
        """Override to guarantee all 6 actions appear >= 3 times in profiling."""
        inputs = super().generate_batch(profile, n)

        if profile != "profiling":
            return inputs

        # Count action coverage from structural descriptors
        action_counts: dict[str, int] = {a: 0 for a in _ACTIONS}
        for inp in inputs:
            desc = inp.structural_descriptor
            if desc["short_circuit_expected"]:
                action_counts["intake_shortcircuit"] += 1
            elif desc["claim_type"] == "standard" and not desc["routing_expected"]:
                action_counts["standard_adjudication"] += 1
            elif desc["claim_type"] == "pre_approval" and not desc["routing_expected"]:
                action_counts["pre_approval_clinical"] += 1
            elif desc["claim_type"] == "appeal" and not desc["routing_expected"]:
                action_counts["appeal_review"] += 1

            for override in desc["overrides_expected"]:
                if override == "high_amount":
                    action_counts["high_amount_routing"] += 1
                elif override == "code_inconsistency":
                    action_counts["code_mismatch_routing"] += 1

        # Patch any under-represented actions by replacing medium-tier inputs
        medium_indices = [
            i for i, inp in enumerate(inputs)
            if inp.tier == "medium" and not inp.is_dirty
        ]

        for action, count in action_counts.items():
            while count < 3 and medium_indices:
                patch_idx = medium_indices.pop(0)
                patched = self._build_action_targeted_claim(action, self.rng, patch_idx)
                desc = self._structural_descriptor(patched, "medium")
                claim_text = json.dumps(patched, indent=2)

                inputs[patch_idx] = GeneratedInput(
                    id=inputs[patch_idx].id,
                    workflow=self.workflow_id,
                    profile=profile,
                    tier=inputs[patch_idx].tier,
                    token_count=self.estimate_tokens(claim_text),
                    is_dirty=False,
                    dirty_type=None,
                    structural_descriptor=desc,
                    input_data=patched,
                )
                count += 1

        return inputs

    def _build_action_targeted_claim(
        self, action: str, rng: random.Random, idx: int
    ) -> dict[str, Any]:
        """Build a claim that specifically triggers the named action."""
        if action == "intake_shortcircuit":
            return self._build_easy_claim(rng, idx)
        if action == "standard_adjudication":
            return self._build_medium_claim(rng, idx)
        if action == "pre_approval_clinical":
            dx, cpt = rng.choice(_MATCHED_CODE_PAIRS)
            claim = self._build_medium_claim(rng, idx)
            claim["claim_type"] = "pre_approval"
            claim["diagnosis_code"] = dx
            claim["procedure_code"] = cpt
            claim["claimed_amount"] = round(rng.uniform(2000, 4500), 2)
            return claim
        if action == "appeal_review":
            claim = self._build_hard_claim(rng, idx)
            claim["claim_type"] = "appeal"
            # Ensure matched codes so routing is not triggered
            dx, cpt = rng.choice(_MATCHED_CODE_PAIRS)
            claim["diagnosis_code"] = dx
            claim["procedure_code"] = cpt
            claim["claimed_amount"] = round(rng.uniform(1000, 4500), 2)
            claim["prior_denial_reason"] = rng.choice(_DENIAL_REASONS)
            claim["appeal_documentation"] = rng.choice(_APPEAL_DOCS)
            return claim
        if action == "high_amount_routing":
            claim = self._build_medium_claim(rng, idx)
            claim["claimed_amount"] = round(rng.uniform(5001, 25000), 2)
            return claim
        if action == "code_mismatch_routing":
            dx, cpt = rng.choice(_MISMATCHED_CODE_PAIRS)
            claim = self._build_medium_claim(rng, idx)
            claim["diagnosis_code"] = dx
            claim["procedure_code"] = cpt
            return claim

        return self._build_medium_claim(rng, idx)


if __name__ == "__main__":
    add_cli(W17ClaimsAgentGenerator)
