"""W12 business document entity extraction generator.

Generate business documents (reports, memos, financial summaries,
correspondence) for entity extraction by DeepSeek models.
Dirty types: mixed Unicode.
"""

from __future__ import annotations

import random
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

# ── Document templates per type and tier ─────────────────────────────────

_COMPANY_NAMES = [
    "Meridian Holdings", "Atlas Corp", "Pinnacle Industries", "Vanguard Systems",
    "Sterling Partners", "Crestview Capital", "Summit Dynamics", "Ironbridge LLC",
    "Northstar Group", "Horizon Ventures", "Cascade Technologies", "Sentinel Corp",
    "Apex Global", "Beacon Enterprises", "Conduit Financial", "Drummond & Associates",
]

_PERSON_NAMES = [
    "Sarah Mitchell", "James Nakamura", "Elena Rodriguez", "Marcus Thompson",
    "Priya Sharma", "David Kowalski", "Amara Okafor", "Chen Wei",
    "Lisa Fernandez", "Robert Andersen", "Yuki Tanaka", "Michael O'Brien",
    "Fatima Al-Rashid", "Thomas Mueller", "Aisha Patel", "Carlos Dubois",
]

_TITLES = [
    "CEO", "CFO", "COO", "VP of Finance", "Director of Operations",
    "Chief Revenue Officer", "Head of Strategy", "Managing Director",
    "Senior VP of Sales", "Controller", "Treasurer", "Board Member",
]

# ── Easy templates (short, clear structure) ──────────────────────────────

_EASY_REPORT_TEMPLATES = [
    "QUARTERLY PERFORMANCE SUMMARY\nCompany: {company}\nPeriod: Q{q} {year}\nPrepared by: {person}, {title}\n\nRevenue: ${revenue:,.0f}\nOperating Expenses: ${opex:,.0f}\nNet Income: ${net:,.0f}\nHeadcount: {hc}\n\nKey Highlights:\n- Revenue grew {growth}% year-over-year\n- New client acquisitions: {clients}\n- Customer retention rate: {retention}%\n\nOutlook: Management expects continued growth in Q{nq} driven by expansion in the {segment} segment.",
    "MONTHLY STATUS REPORT\nFrom: {person}, {title}\nTo: Board of Directors\nDate: {month} {year}\nRe: {company} Operations Update\n\nOperational Metrics:\n- Units processed: {units:,}\n- Average processing time: {time} hours\n- Error rate: {error}%\n- Customer satisfaction: {csat}/10\n\nBudget Status:\n- Allocated: ${budget:,.0f}\n- Spent: ${spent:,.0f}\n- Variance: {variance}%\n\nAction Items:\n1. Review vendor contracts by end of month\n2. Hire {hires} additional staff for {dept} department\n3. Complete {project} migration by {deadline}",
    "EXECUTIVE BRIEFING\n{company} | {month} {year}\n\nPrepared for: {person}, {title}\n\nMarket Position:\n- Market share: {share}%\n- Rank: #{rank} in {industry}\n- YoY growth: {growth}%\n\nFinancial Snapshot:\n- ARR: ${arr:,.0f}\n- MRR: ${mrr:,.0f}\n- Burn rate: ${burn:,.0f}/month\n- Runway: {runway} months\n\nStrategic Priorities:\n1. {priority1}\n2. {priority2}\n3. {priority3}",
    "INVESTMENT MEMORANDUM\nSubject: {company} Series {round} Funding\nDate: {month} {year}\nLead: {person}, {title}\n\nCompany Overview:\n- Founded: {founded}\n- Industry: {industry}\n- Employees: {hc}\n- HQ: {city}\n\nFinancials:\n- Revenue (TTM): ${revenue:,.0f}\n- Growth Rate: {growth}% YoY\n- Gross Margin: {margin}%\n\nProposed Terms:\n- Round size: ${round_size:,.0f}\n- Pre-money valuation: ${valuation:,.0f}\n- Lead investor: {investor}",
]

_EASY_MEMO_TEMPLATES = [
    "INTERNAL MEMO\nTo: All Staff\nFrom: {person}, {title}\nDate: {month} {year}\nSubject: {subject}\n\n{body}\n\nPlease direct questions to {person2} at ext. {ext}.",
    "MEMO\nTo: {person}, {title}\nFrom: {person2}, {title2}\nDate: {month} {year}\nRe: {subject}\n\nAfter reviewing the Q{q} results for {company}, I recommend the following actions:\n\n1. {action1}\n2. {action2}\n3. {action3}\n\nThe estimated cost impact is ${cost:,.0f} over {months} months.\n\nPlease respond by {deadline}.",
]

_EASY_FINANCIAL_TEMPLATES = [
    "FINANCIAL SUMMARY\n{company}\nFor the period ending {month} {year}\n\nIncome Statement:\n  Revenue: ${revenue:,.0f}\n  COGS: ${cogs:,.0f}\n  Gross Profit: ${gp:,.0f}\n  SG&A: ${sga:,.0f}\n  EBITDA: ${ebitda:,.0f}\n  Net Income: ${net:,.0f}\n\nBalance Sheet Highlights:\n  Total Assets: ${assets:,.0f}\n  Total Liabilities: ${liabilities:,.0f}\n  Shareholders Equity: ${equity:,.0f}\n\nCash Flow:\n  Operating: ${cfo:,.0f}\n  Investing: ${cfi:,.0f}\n  Financing: ${cff:,.0f}",
    "BUDGET vs ACTUAL REPORT\n{company} | {month} {year}\nDepartment: {dept}\nManager: {person}\n\nCategory          Budget       Actual     Variance\nPersonnel     ${p_budget:>10,.0f}  ${p_actual:>10,.0f}  {p_var:>8}%\nTechnology    ${t_budget:>10,.0f}  ${t_actual:>10,.0f}  {t_var:>8}%\nMarketing     ${m_budget:>10,.0f}  ${m_actual:>10,.0f}  {m_var:>8}%\nOperations    ${o_budget:>10,.0f}  ${o_actual:>10,.0f}  {o_var:>8}%\n\nTotal         ${total_b:>10,.0f}  ${total_a:>10,.0f}  {total_v:>8}%\n\nNotes: {notes}",
]

_EASY_CORRESPONDENCE_TEMPLATES = [
    "From: {person} <{email1}>\nTo: {person2} <{email2}>\nDate: {month} {year}\nSubject: {subject}\n\nDear {person2},\n\nThank you for your inquiry regarding {topic}. {body}\n\nPlease find the relevant details below:\n- Contract value: ${value:,.0f}\n- Term: {term} months\n- Effective date: {date}\n\nBest regards,\n{person}\n{title}, {company}",
    "Dear {person},\n\nI am writing to confirm the terms discussed during our meeting on {date}.\n\n{company} agrees to provide {service} for a total fee of ${fee:,.0f}, payable in {payments} installments.\n\nKey milestones:\n1. {milestone1} - {date1}\n2. {milestone2} - {date2}\n3. {milestone3} - {date3}\n\nPlease sign and return the attached agreement by {deadline}.\n\nSincerely,\n{person2}\n{title2}, {company2}",
]

# ── Document type pools ──────────────────────────────────────────────────

_DOC_TYPES = ["report", "memo", "financial_summary", "correspondence"]

# ── Token ranges ─────────────────────────────────────────────────────────

_TOKEN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "profiling": {
        "easy": (200, 600),
        "medium": (600, 2000),
        "hard": (2000, 5000),
        "edge": (100, 5000),
    },
    "ground_truth": {
        "easy": (300, 1000),
        "medium": (1000, 3500),
        "hard": (3000, 8000),
        "edge": (100, 8000),
        "extreme": (6000, 15000),
    },
}

# ── Filler paragraphs for padding ────────────────────────────────────────

_FILLER_PARAGRAPHS = [
    "\nThe management team has identified several strategic initiatives for the upcoming quarter. These include expanding into adjacent markets, optimizing the supply chain to reduce costs by an estimated 12%, and investing in technology infrastructure to support anticipated growth. The board has approved a capital expenditure budget of $2.5 million for these initiatives.\n",
    "\nRegulatory developments continue to shape the competitive landscape. Recent changes to data privacy regulations will require additional compliance investments estimated at $350,000 annually. The legal team is working with external counsel to ensure full compliance by the regulatory deadline. Meanwhile, industry consolidation presents both opportunities for strategic acquisitions and competitive threats from larger combined entities.\n",
    "\nCustomer acquisition metrics remain strong with a 23% increase in new enterprise clients compared to the prior quarter. The average contract value has increased from $85,000 to $112,000, reflecting successful upselling efforts by the sales team. Customer churn has decreased to 4.2% from 5.8%, attributable to the implementation of the new customer success program launched in January.\n",
    "\nThe technology division completed the migration of legacy systems to the new cloud infrastructure ahead of schedule and $180,000 under budget. System uptime has improved to 99.97% from 99.82%, and average response times have decreased by 34%. The team is now focused on implementing automated monitoring and predictive maintenance capabilities.\n",
    "\nHuman resources reports that the organization has grown to 847 employees, a net increase of 62 over the prior quarter. Key hires include a new VP of Engineering, three senior data scientists, and a Chief Information Security Officer. Employee satisfaction scores from the latest survey averaged 4.1 out of 5.0, up from 3.8 in the previous cycle.\n",
    "\nThe marketing department executed a multi-channel campaign that generated 12,450 qualified leads at a cost per lead of $42, well below the industry benchmark of $65. Social media engagement increased 156% following the launch of the thought leadership content series. The brand awareness survey conducted in March showed a 15-point improvement in unaided recall among the target demographic.\n",
    "\nOperational efficiency metrics show continued improvement across all business units. The order fulfillment cycle time has been reduced from 4.2 days to 2.8 days through process automation and workflow optimization. Quality defect rates have decreased to 0.3% from 0.7%, and the on-time delivery rate has improved to 98.5% from 96.1%.\n",
    "\nResearch and development expenditures totaled $4.8 million for the quarter, representing 14% of revenue. The R&D team filed 7 patent applications and released 3 major product updates. The product roadmap for the next fiscal year includes investments in artificial intelligence capabilities, enhanced analytics features, and a new mobile platform.\n",
    "\nThe risk management committee has identified and assessed 23 risk factors across operational, financial, regulatory, and strategic categories. The top three risks by potential impact are: cybersecurity threats (estimated impact $5-15M), supply chain disruption (estimated impact $3-8M), and key personnel retention (estimated impact $2-6M). Mitigation plans have been developed for each.\n",
    "\nInternational operations contributed 28% of total revenue, up from 22% in the same period last year. The European division grew 45% year-over-year, driven primarily by the German and UK markets. The Asia-Pacific region showed mixed results, with strong performance in Japan and Australia offset by challenges in the Chinese market due to regulatory changes.\n",
    "\nPartnerships and alliances activity was robust during the quarter. The company signed strategic partnership agreements with three Fortune 500 companies, entered into a technology licensing arrangement with a leading university research lab, and expanded its reseller network by adding 18 certified partners across 7 countries. Total partner-sourced revenue increased 67% year-over-year.\n",
    "\nThe sustainability initiative made significant progress toward ESG goals. Carbon emissions were reduced by 15% through energy efficiency improvements and the transition to renewable energy sources for two major facilities. The company achieved ISO 14001 certification for its environmental management system and published its first comprehensive ESG report, receiving a B+ rating from the leading ESG analytics firm.\n",
]

_SECTION_HEADERS = [
    "\n--- DETAILED ANALYSIS ---\n",
    "\n--- ADDITIONAL CONTEXT ---\n",
    "\n--- SUPPLEMENTARY DATA ---\n",
    "\n--- APPENDIX ---\n",
    "\n--- SUPPORTING INFORMATION ---\n",
    "\n--- RISK ASSESSMENT ---\n",
    "\n--- MARKET ANALYSIS ---\n",
    "\n--- COMPETITIVE LANDSCAPE ---\n",
]


def _fill_template_vars(template: str, rng: random.Random) -> str:
    """Replace template placeholders with realistic random values."""
    companies = list(_COMPANY_NAMES)
    rng.shuffle(companies)
    persons = list(_PERSON_NAMES)
    rng.shuffle(persons)
    titles = list(_TITLES)
    rng.shuffle(titles)

    months = ["January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"]

    replacements: dict[str, Any] = {
        "company": companies[0], "company2": companies[1] if len(companies) > 1 else companies[0],
        "person": persons[0], "person2": persons[1] if len(persons) > 1 else persons[0],
        "title": titles[0], "title2": titles[1] if len(titles) > 1 else titles[0],
        "month": rng.choice(months), "year": rng.choice([2024, 2025, 2026]),
        "q": rng.randint(1, 4), "nq": rng.randint(1, 4),
        "revenue": rng.uniform(500_000, 50_000_000),
        "opex": rng.uniform(300_000, 30_000_000),
        "net": rng.uniform(50_000, 10_000_000),
        "hc": rng.randint(20, 2000),
        "growth": rng.randint(5, 85),
        "clients": rng.randint(5, 200),
        "retention": rng.randint(85, 99),
        "segment": rng.choice(["enterprise", "mid-market", "SMB", "government", "healthcare"]),
        "units": rng.randint(1000, 500000),
        "time": round(rng.uniform(0.5, 48.0), 1),
        "error": round(rng.uniform(0.1, 5.0), 2),
        "csat": round(rng.uniform(7.0, 9.8), 1),
        "budget": rng.uniform(100_000, 5_000_000),
        "spent": rng.uniform(80_000, 4_500_000),
        "variance": rng.randint(-15, 15),
        "hires": rng.randint(2, 20),
        "dept": rng.choice(["Engineering", "Sales", "Marketing", "Operations", "Finance"]),
        "project": rng.choice(["cloud", "CRM", "ERP", "data warehouse", "platform"]),
        "deadline": f"{rng.choice(months)} {rng.randint(1, 28)}, {rng.choice([2025, 2026])}",
        "share": rng.randint(2, 35),
        "rank": rng.randint(1, 20),
        "industry": rng.choice(["SaaS", "Fintech", "Healthcare IT", "E-commerce", "Cybersecurity"]),
        "arr": rng.uniform(1_000_000, 100_000_000),
        "mrr": rng.uniform(80_000, 8_000_000),
        "burn": rng.uniform(100_000, 3_000_000),
        "runway": rng.randint(6, 36),
        "round": rng.choice(["A", "B", "C", "D"]),
        "founded": rng.randint(2015, 2023),
        "city": rng.choice(["San Francisco", "New York", "London", "Berlin", "Singapore", "Austin"]),
        "round_size": rng.uniform(5_000_000, 100_000_000),
        "valuation": rng.uniform(20_000_000, 500_000_000),
        "investor": rng.choice(["Sequoia Capital", "a16z", "Accel Partners", "Lightspeed", "Index Ventures"]),
        "margin": rng.randint(55, 85),
        "priority1": rng.choice(["Expand enterprise sales team", "Launch APAC operations", "Build AI features"]),
        "priority2": rng.choice(["Reduce churn to <3%", "Achieve SOC 2 Type II", "Improve NPS to 70+"]),
        "priority3": rng.choice(["Complete Series B", "Open London office", "Hire VP Engineering"]),
        "subject": rng.choice(["Q2 Planning", "Budget Revision", "Vendor Review", "Org Restructuring",
                                "Product Launch Timeline", "Partnership Proposal"]),
        "body": rng.choice([
            "After careful review of the current quarter performance, we recommend adjusting our strategy.",
            "The team has completed the analysis and the findings are summarized below.",
            "Based on the latest market data, we propose the following changes to our approach.",
        ]),
        "ext": rng.randint(1000, 9999),
        "action1": "Reallocate budget from underperforming segments",
        "action2": "Accelerate hiring for critical engineering roles",
        "action3": "Initiate vendor renegotiation for cost optimization",
        "cost": rng.uniform(50_000, 2_000_000),
        "months": rng.randint(3, 18),
        "email1": f"{persons[0].split()[0].lower()}@{companies[0].lower().replace(' ', '')}.com",
        "email2": f"{persons[1].split()[0].lower()}@{companies[1].lower().replace(' ', '')}.com" if len(persons) > 1 else "contact@example.com",
        "topic": rng.choice(["the proposed partnership", "consulting services", "license renewal", "merger terms"]),
        "value": rng.uniform(100_000, 10_000_000),
        "term": rng.choice([12, 24, 36]),
        "date": f"{rng.choice(months)} {rng.randint(1, 28)}, {rng.choice([2025, 2026])}",
        "date1": f"{rng.choice(months)} 2025", "date2": f"{rng.choice(months)} 2025",
        "date3": f"{rng.choice(months)} 2026",
        "service": rng.choice(["consulting services", "technology integration", "managed analytics", "advisory"]),
        "fee": rng.uniform(50_000, 5_000_000),
        "payments": rng.choice([2, 3, 4, 6]),
        "milestone1": "Phase 1 - Assessment", "milestone2": "Phase 2 - Implementation",
        "milestone3": "Phase 3 - Validation",
        "cogs": rng.uniform(200_000, 20_000_000),
        "gp": rng.uniform(200_000, 25_000_000),
        "sga": rng.uniform(100_000, 10_000_000),
        "ebitda": rng.uniform(50_000, 15_000_000),
        "assets": rng.uniform(1_000_000, 100_000_000),
        "liabilities": rng.uniform(500_000, 50_000_000),
        "equity": rng.uniform(500_000, 50_000_000),
        "cfo": rng.uniform(100_000, 20_000_000),
        "cfi": rng.uniform(-10_000_000, -100_000),
        "cff": rng.uniform(-5_000_000, 5_000_000),
        "p_budget": rng.uniform(500_000, 3_000_000),
        "p_actual": rng.uniform(450_000, 3_200_000),
        "p_var": rng.randint(-10, 10),
        "t_budget": rng.uniform(200_000, 1_000_000),
        "t_actual": rng.uniform(180_000, 1_100_000),
        "t_var": rng.randint(-15, 15),
        "m_budget": rng.uniform(100_000, 800_000),
        "m_actual": rng.uniform(90_000, 850_000),
        "m_var": rng.randint(-12, 12),
        "o_budget": rng.uniform(150_000, 600_000),
        "o_actual": rng.uniform(140_000, 650_000),
        "o_var": rng.randint(-8, 8),
        "total_b": rng.uniform(1_000_000, 5_000_000),
        "total_a": rng.uniform(900_000, 5_500_000),
        "total_v": rng.randint(-10, 10),
        "notes": rng.choice([
            "Personnel overspend due to accelerated hiring plan.",
            "Technology savings from cloud migration offset marketing overspend.",
            "All categories within acceptable variance thresholds.",
            "Operations under budget due to delayed facility expansion.",
        ]),
    }

    result = template
    for key, val in replacements.items():
        placeholder = "{" + key + "}"
        if placeholder in result:
            if isinstance(val, float):
                # Let the format spec in the template handle it
                pass
            else:
                result = result.replace(placeholder, str(val))

    # Handle remaining format specs for floats
    try:
        result = result.format(**replacements)
    except (KeyError, ValueError, IndexError):
        # If complex format specs fail, do simple replacement
        for key, val in replacements.items():
            placeholder = "{" + key
            if placeholder in result:
                # Try to replace {key:format} patterns
                import re as _re
                pattern = r"\{" + key + r"(?::[^}]*)?\}"
                result = _re.sub(pattern, str(val), result)

    return result


def _select_templates(doc_type: str, rng: random.Random) -> list[str]:
    """Select template pool for a document type."""
    if doc_type == "report":
        return _EASY_REPORT_TEMPLATES
    elif doc_type == "memo":
        return _EASY_MEMO_TEMPLATES
    elif doc_type == "financial_summary":
        return _EASY_FINANCIAL_TEMPLATES
    elif doc_type == "correspondence":
        return _EASY_CORRESPONDENCE_TEMPLATES
    return _EASY_REPORT_TEMPLATES


def _pad_document(text: str, target_min: int, target_max: int, rng: random.Random) -> str:
    """Pad document to target token range by adding sections and paragraphs."""
    tokens = max(1, len(text) // 4)
    target = rng.randint(target_min, target_max)

    while tokens < target:
        header = rng.choice(_SECTION_HEADERS)
        paragraph = rng.choice(_FILLER_PARAGRAPHS)
        text += header + paragraph
        tokens = max(1, len(text) // 4)

    if tokens > target_max:
        text = text[: target_max * 4]

    return text


def _apply_dirty(text: str, dirty_type: str, rng: random.Random) -> str:
    """Apply mixed Unicode substitution."""
    if dirty_type == "mixed_unicode":
        substitutions = {
            "a": "а", "e": "е", "o": "о", "c": "с",
            "A": "А", "E": "Е", "O": "О", "C": "С",
            "p": "р", "P": "Р", "s": "ѕ", "i": "і",
        }
        chars = list(text)
        count = 0
        indices = list(range(len(chars)))
        rng.shuffle(indices)
        for idx in indices:
            if chars[idx] in substitutions and rng.random() < 0.2:
                chars[idx] = substitutions[chars[idx]]
                count += 1
                if count >= 15:
                    break
        return "".join(chars)

    return text


class W12ExtractionDeepSeekGenerator(BaseInputGenerator):
    """Generate business documents for entity extraction."""

    workflow_id = "W12"
    dirty_types = ["mixed_unicode"]

    _token_ranges_lookup = _TOKEN_RANGES

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True

    def get_rewritable_text(self, inp: GeneratedInput) -> str | None:
        return "document_text"

    def get_llm_instruction(self, inp: GeneratedInput) -> str:
        dt = inp.structural_descriptor.get("document_type", "report")
        ec = inp.structural_descriptor.get("entity_count", 5)
        return f"Generate a {inp.tier}-difficulty {dt} document with ~{ec} extractable entities."

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one business document for entity extraction."""
        # Rotate document types evenly
        doc_type = _DOC_TYPES[idx % len(_DOC_TYPES)]
        templates = _select_templates(doc_type, rng)
        template = templates[idx % len(templates)]

        text = _fill_template_vars(template, rng)

        # Determine token range
        range_key = profile if profile in _TOKEN_RANGES else "profiling"
        if tier in _TOKEN_RANGES[range_key]:
            tmin, tmax = _TOKEN_RANGES[range_key][tier]
        elif tier == "extreme" and profile == "ground_truth":
            tmin, tmax = 6000, 15000
        else:
            tmin, tmax = 200, 600

        text = _pad_document(text, tmin, tmax, rng)
        text = self.apply_style_shift(text, profile)

        if is_dirty and dirty_type:
            text = _apply_dirty(text, dirty_type, rng)

        token_count = self.estimate_tokens(text)

        # Count entities in the generated text (approximate)
        entity_count = 0
        for name in _COMPANY_NAMES + _PERSON_NAMES:
            if name in text:
                entity_count += 1
        entity_count = max(entity_count, rng.randint(2, 8))

        has_tables = any(kw in text for kw in ["Budget", "Actual", "Variance", "Category"])
        has_numerical = any(c == "$" for c in text) or any(c == "%" for c in text)

        structural_descriptor: dict[str, Any] = {
            "document_type": doc_type,
            "entity_count": entity_count,
            "has_tables": has_tables,
            "has_numerical_data": has_numerical,
        }

        input_data: dict[str, Any] = {
            "document_text": text,
            "input": text,
        }

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W12",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=structural_descriptor,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(W12ExtractionDeepSeekGenerator)
