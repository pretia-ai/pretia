"""Input generator for W4: Compliance document review.

Produce structured compliance documents (contracts, HR policies, regulatory
filings) with planted issues at varying severity levels.  Dry-run mode uses
template pools — no LLM calls required.
"""

from __future__ import annotations

import random
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

# ---------------------------------------------------------------------------
# Token-length targets  (chars ≈ tokens * 4)
# ---------------------------------------------------------------------------
_TOKEN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "profiling": {
        "easy":   (500, 1_500),
        "medium": (1_500, 4_000),
        "hard":   (4_000, 10_000),
        "edge":   (100, 12_000),
    },
    "ground_truth": {
        "easy":     (800, 2_500),
        "medium":   (2_500, 7_000),
        "hard":     (6_000, 15_000),
        "edge":     (100, 15_000),
        "extreme":  (12_000, 25_000),
    },
}

# ---------------------------------------------------------------------------
# Issue templates per severity
# ---------------------------------------------------------------------------
_CRITICAL_ISSUES = [
    {"type": "missing_clause", "desc": "Indemnification clause absent"},
    {"type": "missing_clause", "desc": "Limitation of liability clause missing"},
    {"type": "missing_gdpr", "desc": "No GDPR data-processing addendum"},
    {"type": "missing_gdpr", "desc": "Right-to-erasure provision not referenced"},
    {"type": "incorrect_reference", "desc": "References repealed regulation 45 CFR 164.502"},
    {"type": "missing_clause", "desc": "Force majeure clause absent"},
    {"type": "missing_clause", "desc": "Governing law clause missing"},
    {"type": "incorrect_reference", "desc": "Cites superseded ISO 27001:2013 instead of :2022"},
]

_MAJOR_ISSUES = [
    {"type": "ambiguous_language", "desc": "\"Reasonable efforts\" undefined in performance SLA"},
    {"type": "ambiguous_language", "desc": "Termination-for-convenience notice period unspecified"},
    {"type": "incorrect_reference", "desc": "Internal cross-reference to Section 4.3 but Section 4 has only 2 subsections"},
    {"type": "ambiguous_language", "desc": "\"Material breach\" not defined"},
    {"type": "incorrect_reference", "desc": "Exhibit A referenced but not attached"},
    {"type": "ambiguous_language", "desc": "\"Commercially reasonable\" standard not benchmarked"},
    {"type": "missing_clause", "desc": "Audit rights clause lacks frequency specification"},
    {"type": "incorrect_reference", "desc": "Schedule B pricing references expired rate card"},
]

_MINOR_ISSUES = [
    {"type": "ambiguous_language", "desc": "Inconsistent capitalisation of defined terms"},
    {"type": "ambiguous_language", "desc": "Paragraph numbering skip from 3.2 to 3.4"},
    {"type": "incorrect_reference", "desc": "Footer references wrong agreement date"},
    {"type": "ambiguous_language", "desc": "Duplicate definition of 'Affiliate' in Sections 1 and 7"},
    {"type": "incorrect_reference", "desc": "Table of contents lists Section 12 but document ends at Section 11"},
    {"type": "ambiguous_language", "desc": "Mixed use of 'shall' and 'will' without distinction"},
]

# ---------------------------------------------------------------------------
# Issue count per tier
# ---------------------------------------------------------------------------
_ISSUE_COUNTS: dict[str, tuple[int, int, int]] = {
    # (min_minor, min_major, min_critical)
    "easy":    (0, 0, 0),
    "medium":  (1, 2, 0),
    "hard":    (2, 3, 3),
    "edge":    (0, 0, 0),
    "extreme": (3, 4, 3),
}

# Document type weights: ~35% contract, ~35% hr_policy, ~30% regulatory_filing
_DOC_TYPES = ["contract", "contract", "contract", "contract", "contract", "contract", "contract",
              "hr_policy", "hr_policy", "hr_policy", "hr_policy", "hr_policy", "hr_policy", "hr_policy",
              "regulatory_filing", "regulatory_filing", "regulatory_filing", "regulatory_filing",
              "regulatory_filing", "regulatory_filing"]

# ---------------------------------------------------------------------------
# Template pools — 5-8 per tier per doc type
# ---------------------------------------------------------------------------

def _contract_sections(rng: random.Random, n_sections: int) -> list[tuple[str, str]]:
    """Return (title, body_template) pairs for a contract."""
    pool = [
        ("DEFINITIONS", "For purposes of this Agreement, the following terms have the meanings set forth below.\n"
         "\"Affiliate\" means any entity that directly or indirectly controls, is controlled by, or is "
         "under common control with a party.\n\"Confidential Information\" means all non-public information "
         "disclosed by either party to the other.\n\"Effective Date\" means the date first written above.\n"
         "\"Services\" means the consulting, development, and support services described in Exhibit A."),
        ("SCOPE OF SERVICES", "Provider shall perform the Services described in Exhibit A attached hereto. "
         "Services shall be performed in a professional and workmanlike manner consistent with generally "
         "accepted industry standards. Provider shall assign qualified personnel with appropriate experience "
         "and expertise to perform the Services."),
        ("TERM AND TERMINATION", "This Agreement shall commence on the Effective Date and continue for an "
         "initial term of twelve (12) months (the \"Initial Term\"), unless earlier terminated in accordance "
         "with this Section. Either party may terminate this Agreement for convenience upon sixty (60) days "
         "prior written notice to the other party."),
        ("COMPENSATION AND PAYMENT", "Client shall pay Provider the fees set forth in Schedule B attached "
         "hereto. All invoices are due and payable within thirty (30) days of receipt. Late payments shall "
         "bear interest at the rate of 1.5% per month or the maximum rate permitted by applicable law, "
         "whichever is less."),
        ("INTELLECTUAL PROPERTY", "All intellectual property rights in work product created by Provider in "
         "the course of performing the Services (\"Work Product\") shall be owned exclusively by Client. "
         "Provider hereby assigns to Client all right, title, and interest in and to the Work Product. "
         "Provider retains ownership of its pre-existing intellectual property."),
        ("CONFIDENTIALITY", "Each party agrees to hold the other party's Confidential Information in strict "
         "confidence and not to disclose such information to any third party without the prior written "
         "consent of the disclosing party. This obligation shall survive termination of this Agreement for "
         "a period of three (3) years."),
        ("REPRESENTATIONS AND WARRANTIES", "Each party represents and warrants that: (a) it has the legal "
         "power and authority to enter into this Agreement; (b) this Agreement constitutes a valid and "
         "binding obligation; and (c) the execution of this Agreement does not conflict with any other "
         "agreement to which it is a party."),
        ("LIMITATION OF LIABILITY", "IN NO EVENT SHALL EITHER PARTY BE LIABLE FOR ANY INDIRECT, INCIDENTAL, "
         "SPECIAL, CONSEQUENTIAL, OR PUNITIVE DAMAGES, REGARDLESS OF THE CAUSE OF ACTION OR THE THEORY OF "
         "LIABILITY. EACH PARTY'S TOTAL AGGREGATE LIABILITY UNDER THIS AGREEMENT SHALL NOT EXCEED THE FEES "
         "PAID OR PAYABLE IN THE TWELVE (12) MONTHS PRECEDING THE CLAIM."),
        ("INDEMNIFICATION", "Each party (the \"Indemnifying Party\") shall indemnify, defend, and hold "
         "harmless the other party from and against any and all claims, damages, losses, and expenses "
         "(including reasonable attorneys' fees) arising out of the Indemnifying Party's breach of this "
         "Agreement or negligent or willful acts or omissions."),
        ("GOVERNING LAW AND DISPUTE RESOLUTION", "This Agreement shall be governed by and construed in "
         "accordance with the laws of the State of Delaware, without regard to its conflict of laws "
         "principles. Any dispute arising under this Agreement shall be resolved through binding arbitration "
         "administered by the American Arbitration Association."),
        ("DATA PROTECTION", "Provider shall comply with all applicable data protection laws, including the "
         "General Data Protection Regulation (GDPR) and the California Consumer Privacy Act (CCPA). "
         "Provider shall implement appropriate technical and organizational measures to protect personal "
         "data processed on behalf of Client."),
        ("FORCE MAJEURE", "Neither party shall be liable for any failure or delay in performing its "
         "obligations under this Agreement to the extent that such failure or delay results from a Force "
         "Majeure Event. \"Force Majeure Event\" means any event beyond the reasonable control of the "
         "affected party, including natural disasters, war, terrorism, pandemics, and government actions."),
    ]
    rng.shuffle(pool)
    return pool[:n_sections]


def _hr_sections(rng: random.Random, n_sections: int) -> list[tuple[str, str]]:
    pool = [
        ("PURPOSE AND SCOPE", "This policy establishes guidelines for employee conduct, compensation, "
         "and benefits applicable to all full-time and part-time employees of the Company. This policy "
         "supersedes all prior policies and handbooks on the same subject matter."),
        ("EQUAL EMPLOYMENT OPPORTUNITY", "The Company is committed to providing equal employment "
         "opportunities to all employees and applicants without regard to race, color, religion, sex, "
         "national origin, age, disability, genetic information, veteran status, or any other protected "
         "characteristic under federal, state, or local law."),
        ("COMPENSATION AND BENEFITS", "Employee compensation is reviewed annually and adjusted based on "
         "performance, market conditions, and internal equity. Benefits include health insurance, dental "
         "and vision coverage, 401(k) retirement plan with employer match up to 4%, paid time off, and "
         "employee assistance programs."),
        ("LEAVE POLICIES", "Employees are entitled to the following leave benefits: (a) Annual Leave: 15 "
         "business days per calendar year for employees with less than 5 years of service, 20 days for "
         "employees with 5 or more years; (b) Sick Leave: 10 days per year; (c) FMLA Leave: up to 12 "
         "weeks of unpaid leave as required by the Family and Medical Leave Act."),
        ("CODE OF CONDUCT", "All employees are expected to conduct themselves in a professional manner "
         "that reflects positively on the Company. Employees shall avoid conflicts of interest, maintain "
         "confidentiality of proprietary information, and comply with all applicable laws and regulations. "
         "Violations of this Code may result in disciplinary action up to and including termination."),
        ("ANTI-HARASSMENT POLICY", "The Company prohibits harassment of any kind, including but not "
         "limited to sexual harassment, bullying, and discrimination. Employees who experience or witness "
         "harassment should report it immediately to their supervisor or Human Resources. All complaints "
         "will be investigated promptly and confidentially."),
        ("REMOTE WORK POLICY", "Eligible employees may work remotely up to three (3) days per week with "
         "manager approval. Remote workers must maintain a dedicated workspace, ensure reliable internet "
         "connectivity, and remain available during core business hours (9:00 AM to 3:00 PM local time)."),
        ("PERFORMANCE MANAGEMENT", "Performance reviews are conducted semi-annually in June and December. "
         "Reviews evaluate job performance, goal achievement, and alignment with company values. Employees "
         "receive ratings on a five-point scale from 'Needs Improvement' to 'Exceptional'."),
        ("DISCIPLINARY PROCEDURES", "The Company follows a progressive discipline approach: (1) Verbal "
         "Warning, (2) Written Warning, (3) Final Written Warning, (4) Termination. Severe misconduct "
         "may result in immediate termination without prior warnings."),
        ("DATA PRIVACY AND SECURITY", "Employees must comply with the Company's data privacy and security "
         "policies. Personal data of employees, customers, and partners must be handled in accordance with "
         "applicable data protection regulations including GDPR and CCPA. Employees must report data "
         "breaches within 24 hours of discovery."),
    ]
    rng.shuffle(pool)
    return pool[:n_sections]


def _regulatory_sections(rng: random.Random, n_sections: int) -> list[tuple[str, str]]:
    pool = [
        ("EXECUTIVE SUMMARY", "This filing presents the annual compliance report for fiscal year 2024 "
         "as required under SEC Rule 10b-5 and Regulation S-K. The report covers material developments, "
         "risk factors, and internal control assessments for the reporting period ending December 31, 2024."),
        ("RISK FACTORS", "The following risk factors may materially affect the Company's business, "
         "financial condition, and results of operations: (a) regulatory changes in key markets; "
         "(b) cybersecurity threats and data breach risks; (c) competitive pressures from emerging "
         "fintech companies; (d) interest rate volatility; (e) geopolitical instability affecting "
         "supply chains."),
        ("INTERNAL CONTROLS", "Management is responsible for establishing and maintaining adequate "
         "internal controls over financial reporting as defined in Rules 13a-15(f) and 15d-15(f) under "
         "the Securities Exchange Act of 1934. Our internal control framework is based on the COSO 2013 "
         "Internal Control — Integrated Framework."),
        ("FINANCIAL STATEMENTS", "The consolidated financial statements have been prepared in accordance "
         "with U.S. Generally Accepted Accounting Principles (GAAP). Revenue recognition follows ASC 606. "
         "All material inter-company transactions have been eliminated in consolidation."),
        ("COMPLIANCE CERTIFICATIONS", "The undersigned officers certify that: (i) this report does not "
         "contain any untrue statement of material fact; (ii) the financial statements fairly present the "
         "financial condition; (iii) internal controls are effective as of the assessment date; and "
         "(iv) all significant deficiencies have been disclosed to the audit committee."),
        ("REGULATORY FRAMEWORK", "The Company operates under the regulatory oversight of the SEC, FINRA, "
         "OCC, and applicable state regulators. Material regulatory developments during the reporting "
         "period include the adoption of final rules under the Dodd-Frank Act Section 619 (Volcker Rule) "
         "and updates to Basel III capital requirements."),
        ("AUDIT COMMITTEE REPORT", "The Audit Committee has reviewed and discussed the audited financial "
         "statements with management and the independent auditor. The Committee has received written "
         "disclosures and the letter from the independent auditor required by PCAOB Ethics and "
         "Independence Rule 3526."),
        ("ENVIRONMENTAL AND SOCIAL GOVERNANCE", "The Company is committed to sustainable business "
         "practices. During the reporting period, we reduced Scope 1 and 2 emissions by 15% compared "
         "to the prior year baseline. Our diversity metrics show 42% female representation at the "
         "management level, exceeding the industry average of 34%."),
        ("LEGAL PROCEEDINGS", "The Company is party to various legal proceedings arising in the ordinary "
         "course of business. Management believes that the resolution of these matters will not have a "
         "material adverse effect on the Company's financial position, results of operations, or cash "
         "flows. Refer to Note 18 of the financial statements for additional detail."),
        ("SUBSEQUENT EVENTS", "Management has evaluated subsequent events through the date the financial "
         "statements were issued. On January 15, 2025, the Company completed the acquisition of XYZ Corp "
         "for approximately $450 million in cash and stock. No other material subsequent events were "
         "identified."),
    ]
    rng.shuffle(pool)
    return pool[:n_sections]


_SECTION_BUILDERS = {
    "contract": _contract_sections,
    "hr_policy": _hr_sections,
    "regulatory_filing": _regulatory_sections,
}

# Copy-paste dirty artifacts
_COPY_PASTE_ARTIFACTS = [
    "\n\n--- Page Break ---\n[Header: CONFIDENTIAL DRAFT — DO NOT DISTRIBUTE]\n\n",
    "\n\n[Note to self: need to verify this clause with legal]\n",
    "\n\nTrack Changes: DELETED — \"Provider shall not be liable\" INSERTED — \"Provider shall be liable\"\n",
    "\n\n>>> Copied from template v2.3 — UPDATE BEFORE SENDING <<<\n",
    "\n\n[TODO: Insert client name here] agrees to the following terms...\n",
    "\n\nComment [JD1]: Is this still accurate after the 2024 amendment?\n",
    "\n\n[REDLINE: Previous version had 90-day notice period, changed to 60]\n",
    "\n\n/* INTERNAL NOTE: This section was drafted by outside counsel and has not been reviewed */\n",
]


class ComplianceReviewGenerator(BaseInputGenerator):
    """Generate compliance documents with planted issues for W4."""

    workflow_id = "W4"
    dirty_types = ["copy_pasted_artifacts"]

    _token_ranges_lookup = _TOKEN_RANGES

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True

    def get_rewritable_text(self, inp: GeneratedInput) -> str | None:
        return "document_text"

    def get_llm_instruction(self, inp: GeneratedInput) -> str:
        dt = inp.structural_descriptor.get("document_type", "contract")
        issues = len(inp.structural_descriptor.get("planted_issues", []))
        return f"Generate a {inp.tier}-difficulty {dt} for compliance review with {issues} issues."

    def _pick_doc_type(self, rng: random.Random) -> str:
        return rng.choice(_DOC_TYPES)

    def _pick_issues(
        self, tier: str, rng: random.Random, n_sections: int,
    ) -> list[dict[str, str]]:
        """Select planted issues appropriate for the tier."""
        min_minor, min_major, min_critical = _ISSUE_COUNTS.get(tier, (0, 0, 0))

        if tier == "easy":
            # 0-1 minor issues
            count = rng.randint(0, 1)
            pool = rng.sample(_MINOR_ISSUES, min(count, len(_MINOR_ISSUES)))
            issues = [{"severity": "minor", **i} for i in pool]
        elif tier == "medium":
            minor = rng.sample(_MINOR_ISSUES, min(min_minor, len(_MINOR_ISSUES)))
            major = rng.sample(_MAJOR_ISSUES, min(min_major, len(_MAJOR_ISSUES)))
            issues = ([{"severity": "minor", **i} for i in minor]
                      + [{"severity": "major", **i} for i in major])
        elif tier in ("hard", "extreme"):
            n_minor = rng.randint(min_minor, min_minor + 2)
            n_major = rng.randint(min_major, min_major + 2)
            n_critical = rng.randint(min_critical, min_critical + 2)
            minor = rng.sample(_MINOR_ISSUES, min(n_minor, len(_MINOR_ISSUES)))
            major = rng.sample(_MAJOR_ISSUES, min(n_major, len(_MAJOR_ISSUES)))
            critical = rng.sample(_CRITICAL_ISSUES, min(n_critical, len(_CRITICAL_ISSUES)))
            issues = ([{"severity": "minor", **i} for i in minor]
                      + [{"severity": "major", **i} for i in major]
                      + [{"severity": "critical", **i} for i in critical])
        else:
            # edge — random mix
            total = rng.randint(0, 6)
            all_pool = (
                [{"severity": "minor", **i} for i in _MINOR_ISSUES]
                + [{"severity": "major", **i} for i in _MAJOR_ISSUES]
                + [{"severity": "critical", **i} for i in _CRITICAL_ISSUES]
            )
            rng.shuffle(all_pool)
            issues = all_pool[:total]

        # Assign sections
        for issue in issues:
            sec = rng.randint(1, n_sections)
            subsec = rng.randint(1, 4)
            issue["section"] = f"Section {sec}.{subsec}"

        return issues

    def _build_document(
        self,
        doc_type: str,
        tier: str,
        profile: str,
        rng: random.Random,
        target_tokens: int,
        issues: list[dict[str, str]],
    ) -> tuple[str, int]:
        """Build a template document and return (text, section_count)."""
        # Decide section count based on tier
        if tier == "easy":
            n_sections = rng.randint(4, 6)
        elif tier == "medium":
            n_sections = rng.randint(6, 8)
        elif tier in ("hard", "extreme"):
            n_sections = rng.randint(8, 12)
        else:
            n_sections = rng.randint(3, 12)

        builder = _SECTION_BUILDERS[doc_type]
        sections = builder(rng, n_sections)

        # Assemble document
        doc_type_label = doc_type.upper().replace("_", " ")
        lines = [
            f"{doc_type_label}",
            "=" * len(doc_type_label),
            f"Effective Date: January 1, 202{rng.randint(3, 6)}",
            "",
        ]

        for i, (title, body) in enumerate(sections, 1):
            lines.append(f"SECTION {i}. {title}")
            lines.append("-" * (len(title) + len(str(i)) + 10))
            lines.append("")
            # Repeat body to reach target length
            lines.append(body)
            lines.append("")

        text = "\n".join(lines)

        # Stretch or trim to approximate target token count
        target_chars = target_tokens * 4
        while len(text) < target_chars:
            # Add supplementary clauses
            extra_section = rng.choice(sections)
            text += f"\n\nSUPPLEMENTARY — {extra_section[0]}\n{extra_section[1]}\n"
        if len(text) > target_chars * 1.3:
            text = text[:target_chars]

        return text, len(sections)

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one compliance document with planted issues."""
        doc_type = self._pick_doc_type(rng)

        # Pick target token count within tier range
        ranges = _TOKEN_RANGES.get(profile, _TOKEN_RANGES["profiling"])
        lo, hi = ranges.get(tier, (500, 1_500))
        target_tokens = rng.randint(lo, hi)

        # Determine section count first for issue assignment
        if tier == "easy":
            n_sections = rng.randint(4, 6)
        elif tier == "medium":
            n_sections = rng.randint(6, 8)
        elif tier in ("hard", "extreme"):
            n_sections = rng.randint(8, 12)
        else:
            n_sections = rng.randint(3, 12)

        issues = self._pick_issues(tier, rng, n_sections)

        text, section_count = self._build_document(
            doc_type, tier, profile, rng, target_tokens, issues,
        )

        # Pad/truncate to fit the exact token range for this tier+profile
        text = self.pad_to_token_range(text, lo, hi, rng)

        # Apply dirty artifacts
        if is_dirty and dirty_type == "copy_pasted_artifacts":
            n_artifacts = rng.randint(1, 3)
            for _ in range(n_artifacts):
                artifact = rng.choice(_COPY_PASTE_ARTIFACTS)
                insert_pos = rng.randint(len(text) // 4, 3 * len(text) // 4)
                text = text[:insert_pos] + artifact + text[insert_pos:]

        # Apply style shift for GT
        text = self.apply_style_shift(text, profile)

        token_count = self.estimate_tokens(text)

        # Build structural descriptor
        issue_descriptors = [
            {
                "severity": iss["severity"],
                "type": iss["type"],
                "section": iss["section"],
            }
            for iss in issues
        ]

        # Expected iteration range based on issue count
        n_issues = len(issues)
        if n_issues <= 1:
            expected_iter = [1, 2]
        elif n_issues <= 4:
            expected_iter = [3, 4]
        elif n_issues <= 8:
            expected_iter = [4, 6]
        else:
            expected_iter = [5, 8]

        structural_descriptor: dict[str, Any] = {
            "document_type": doc_type,
            "planted_issues": issue_descriptors,
            "expected_iteration_range": expected_iter,
            "total_sections": section_count,
        }

        input_data: dict[str, Any] = {
            "document_text": text,
            "document_type": doc_type,
            "input": text,
        }

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W4",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=structural_descriptor,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(ComplianceReviewGenerator)
