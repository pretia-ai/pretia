"""W09 sales outreach lead profile generator.

Generate lead profiles for sales qualification scoring.
Dirty types: typos, near-empty.
"""

from __future__ import annotations

import random
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

# ── Pools ────────────────────────────────────────────────────────────────

_COMPANY_NAMES = [
    "Nextera Analytics", "PulsePoint Labs", "VelocityStack", "CloudBridge Solutions",
    "DataForge Inc", "QuantumLeap AI", "SynapseFlow", "TerraCode Systems",
    "NovaEdge Technologies", "ArcLight Digital", "Prism Software Group",
    "ZenithOps", "Clearwave Data", "IronPeak Solutions", "BlueStar Innovations",
    "VertexAI Corp", "StreamLine SaaS", "OmniTech Partners", "FusionGrid Labs",
    "Catalyst Cloud", "BrightPath Analytics", "CodeSphere Inc", "ApexWare Solutions",
    "NimbleOps", "TrueNorth Data", "PixelForge Studios", "RapidScale Tech",
    "EchoLogic Systems", "SummitView Software", "CoreStack Global",
]

_INDUSTRIES_HIGH_FIT = [
    "SaaS", "Fintech", "Developer Tools", "Cloud Infrastructure",
    "AI/ML Platform", "Data Analytics", "Cybersecurity",
]

_INDUSTRIES_MEDIUM_FIT = [
    "Healthcare IT", "EdTech", "Digital Marketing", "E-commerce",
    "Logistics Tech", "PropTech", "InsurTech",
]

_INDUSTRIES_LOW_FIT = [
    "Manufacturing", "Retail", "Agriculture", "Construction",
    "Mining", "Food & Beverage", "Hospitality", "Textiles",
]

_TECH_STACKS = [
    "Python", "TypeScript", "React", "AWS", "GCP", "Azure",
    "Kubernetes", "Docker", "PostgreSQL", "MongoDB", "Redis",
    "Terraform", "GitHub Actions", "Jenkins", "Datadog",
    "Snowflake", "dbt", "Airflow", "Kafka", "Elasticsearch",
    "FastAPI", "Django", "Node.js", "Next.js", "GraphQL",
]

_SIGNALS_STRONG = [
    "Series B funding raised ($25M)",
    "Hiring 10+ engineers this quarter",
    "Published blog post about scaling AI infrastructure",
    "CTO spoke at KubeCon about ML deployment challenges",
    "Moved to microservices architecture (LinkedIn post by VP Eng)",
    "Competitor contract expiring Q3 2026",
    "Posted RFP for cost monitoring tools",
    "Engineering team doubled in last 6 months",
    "Launched new AI product line needing cost controls",
    "Active GitHub org with 50+ public repos",
]

_SIGNALS_MODERATE = [
    "Attended webinar on AI cost management",
    "Downloaded whitepaper on LLM deployment costs",
    "Visited pricing page twice in the last month",
    "Followed company on LinkedIn",
    "Mentioned cost optimization in quarterly earnings call",
    "Small engineering team but growing headcount",
    "Using basic monitoring but no cost-specific tools",
]

_SIGNALS_WEAK = [
    "Generic website visit (homepage only)",
    "Unsubscribed from newsletter then resubscribed",
    "No recent hiring activity",
    "Minimal online engineering presence",
    "Last funding round was 3+ years ago",
]

_ENGAGEMENT_EVENTS = [
    "Signed up for free trial",
    "Completed product demo",
    "Attended live webinar",
    "Downloaded case study PDF",
    "Opened 3 emails in last 2 weeks",
    "Clicked pricing page CTA",
    "Submitted contact form",
    "Booked discovery call",
    "Watched product video (full)",
    "Visited documentation site",
]

_FIRST_NAMES = [
    "Alex", "Jordan", "Morgan", "Casey", "Taylor",
    "Riley", "Avery", "Quinn", "Cameron", "Dakota",
    "Jamie", "Reese", "Skyler", "Hayden", "Kendall",
    "Parker", "Drew", "Blake", "Emerson", "Sage",
    "Rowan", "Finley", "Addison", "Peyton", "Kai",
]

_LAST_NAMES = [
    "Chen", "Rodriguez", "Patel", "Thompson", "Nakamura",
    "Williams", "Garcia", "Mueller", "Kim", "Andersen",
    "O'Brien", "Singh", "Johansson", "Al-Rashid", "Kowalski",
    "Fernandez", "Tanaka", "Dubois", "Okafor", "Svensson",
    "Bennett", "Liu", "Rossi", "Yamamoto", "Petrov",
]

_TITLES_SENIOR = [
    "VP of Engineering", "CTO", "Head of AI/ML", "Director of Platform",
    "Chief Architect", "SVP of Technology", "Head of Infrastructure",
]

_TITLES_MID = [
    "Engineering Manager", "Senior Staff Engineer", "Principal Engineer",
    "Tech Lead", "DevOps Lead", "ML Engineering Manager",
]

_TITLES_JUNIOR = [
    "Software Engineer", "DevOps Engineer", "Data Engineer",
    "Backend Developer", "Cloud Engineer",
]

# ── Token ranges ─────────────────────────────────────────────────────────

_TOKEN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "profiling": {
        "easy": (150, 250),
        "medium": (100, 200),
        "hard": (60, 150),
        "edge": (30, 200),
    },
    "ground_truth": {
        "easy": (200, 400),
        "medium": (150, 350),
        "hard": (80, 250),
        "edge": (30, 300),
        "extreme": (300, 600),
    },
}

# ── Extended signal descriptions for GT token stretch ───────────────────

_SIGNAL_DESCRIPTIONS_LONG = [
    "Recently presented at re:Invent 2025 on migrating legacy ML pipelines to serverless architecture, highlighting cost overruns and optimization strategies for production inference workloads across multiple cloud regions.",
    "Published a detailed technical blog series on their engineering blog about the challenges of scaling LLM-based features from prototype to production, with specific mentions of token cost management and prompt optimization techniques.",
    "Announced a major partnership with a leading cloud provider to co-develop cost-monitoring tools for AI workloads, signaling strategic commitment to infrastructure cost transparency across their product suite.",
    "Their VP of Engineering gave a keynote at QCon on the hidden costs of agentic AI systems, specifically calling out the lack of pre-deployment cost estimation tools in the current ecosystem.",
    "Filed for a patent on adaptive prompt compression techniques that reduce LLM API costs by up to 40%, indicating deep investment in the cost optimization space for AI-powered features.",
    "Conducted a public RFP for AI cost management and observability platforms, with requirements including per-agent cost attribution, anomaly detection, and integration with existing FinOps workflows.",
    "Their engineering team published an open-source tool for benchmarking LLM inference costs across providers, gathering 2,000+ GitHub stars in the first week and significant community engagement.",
    "CEO mentioned in a Bloomberg interview that AI infrastructure costs are their fastest-growing expense line item, representing 23% of total cloud spend and projected to reach 35% by Q4.",
    "Completed a SOC 2 Type II audit that specifically evaluated AI governance and cost controls, suggesting mature procurement processes that would value detailed cost projections.",
    "Launched an internal AI Center of Excellence with a dedicated budget for tooling and optimization, actively evaluating vendors for cost monitoring, prompt management, and model selection.",
    "Their data science team presented a case study at NeurIPS showing how they reduced inference costs by 60% through careful model routing and caching strategies across their product suite.",
    "Recently migrated from a single-model architecture to a multi-model routing system, creating new challenges in cost prediction and optimization that align directly with our value proposition.",
    "The CTO published a LinkedIn article titled 'Why We're Spending $2M/Month on AI and How We Plan to Cut It in Half' which received widespread attention in the developer community.",
    "Participated as a design partner for three competing AI cost management tools before dropping out, suggesting they have specific requirements that existing solutions don't fully address.",
    "Their quarterly earnings call transcript mentions AI cost management as a top-3 engineering priority for the next fiscal year, with dedicated headcount being allocated to the initiative.",
    "Hosted an internal hackathon focused on reducing LLM costs, with the winning project implementing a context window optimizer that reduced average prompt length by 30% without quality degradation.",
    "Recently acquired a small startup specializing in AI model performance benchmarking, signaling strategic intent to build or buy cost optimization capabilities for their growing AI portfolio.",
    "Their head of platform engineering spoke at a private CTO roundtable about the challenges of predicting AI infrastructure costs during sprint planning, citing a 3x variance between estimates and actuals.",
    "Published their annual technology radar which listed 'AI cost observability' as a technology to adopt, placing it in the same priority tier as security scanning and performance monitoring.",
    "Their recruiting page shows 5 open positions specifically mentioning 'AI cost optimization' or 'LLM infrastructure efficiency' in the job description, indicating organizational commitment.",
    "Completed a comprehensive vendor evaluation of FinOps tools and found none that adequately handle AI-specific cost patterns like token-based pricing, context window growth, and agent loop costs.",
]


def _apply_dirty(text: str, dirty_type: str, rng: random.Random) -> str:
    """Apply dirty transformation to a serialized lead profile."""
    if dirty_type == "typos":
        chars = list(text)
        n_typos = rng.randint(2, 5)
        for _ in range(n_typos):
            if len(chars) < 4:
                break
            idx = rng.randint(1, len(chars) - 2)
            op = rng.choice(["swap", "drop", "double"])
            if op == "swap" and idx < len(chars) - 1:
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            elif op == "drop":
                chars.pop(idx)
            else:
                chars.insert(idx, chars[idx])
        return "".join(chars)

    if dirty_type == "near_empty":
        options = [
            '{"lead_profile": {}}',
            '{"lead_profile": {"company_name": ""}}',
            '{"lead_profile": null}',
            "{}",
        ]
        return rng.choice(options)

    return text


class W09SalesOutreachGenerator(BaseInputGenerator):
    """Generate lead profiles for sales qualification scoring."""

    workflow_id = "W09"
    dirty_types = ["typos", "near_empty"]

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True

    def _build_lead_profile(
        self, tier: str, profile: str, rng: random.Random,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Build a lead profile dict and structural descriptor for the tier."""
        company = rng.choice(_COMPANY_NAMES)
        first = rng.choice(_FIRST_NAMES)
        last = rng.choice(_LAST_NAMES)
        email = f"{first.lower()}.{last.lower()}@{company.lower().replace(' ', '').replace(',', '')}.com"

        # Use longer signal descriptions for ground_truth to stretch tokens
        use_long_signals = profile == "ground_truth"

        if tier == "easy":
            # Hot lead: high-fit industry, large company, strong signals
            industry = rng.choice(_INDUSTRIES_HIGH_FIT)
            employee_count = rng.randint(200, 5000)
            tech_stack = rng.sample(_TECH_STACKS, rng.randint(5, 10))
            if use_long_signals:
                n_signals = rng.randint(4, 7)
                signals = rng.sample(_SIGNAL_DESCRIPTIONS_LONG, min(n_signals, len(_SIGNAL_DESCRIPTIONS_LONG)))
            else:
                signals = rng.sample(_SIGNALS_STRONG, rng.randint(3, 5))
            engagement = rng.sample(_ENGAGEMENT_EVENTS, rng.randint(3, 6))
            title = rng.choice(_TITLES_SENIOR)
            expected_rating = "hot"
            expected_score = rng.randint(7, 9)
            profile_completeness = round(rng.uniform(0.85, 1.0), 2)
            industry_fit = "high"

        elif tier == "medium":
            # Warm lead: moderate fit, some signals
            industry = rng.choice(_INDUSTRIES_MEDIUM_FIT)
            employee_count = rng.randint(50, 500)
            tech_stack = rng.sample(_TECH_STACKS, rng.randint(3, 6))
            if use_long_signals:
                n_signals = rng.randint(2, 5)
                signals = rng.sample(_SIGNAL_DESCRIPTIONS_LONG, min(n_signals, len(_SIGNAL_DESCRIPTIONS_LONG)))
            else:
                signals = rng.sample(_SIGNALS_MODERATE, rng.randint(1, 3))
            engagement = rng.sample(_ENGAGEMENT_EVENTS, rng.randint(1, 3))
            title = rng.choice(_TITLES_MID)
            expected_rating = "warm"
            expected_score = rng.randint(4, 6)
            profile_completeness = round(rng.uniform(0.55, 0.85), 2)
            industry_fit = "medium"

        elif tier in ("hard", "extreme"):
            # Cold lead: low-fit industry, small company, sparse
            industry = rng.choice(_INDUSTRIES_LOW_FIT)
            employee_count = rng.randint(5, 50)
            tech_stack = rng.sample(_TECH_STACKS, rng.randint(0, 2))
            if use_long_signals:
                n_signals = rng.randint(1, 3)
                signals = rng.sample(_SIGNAL_DESCRIPTIONS_LONG, min(n_signals, len(_SIGNAL_DESCRIPTIONS_LONG)))
            else:
                signals = rng.sample(_SIGNALS_WEAK, rng.randint(0, 2))
            engagement = rng.sample(_ENGAGEMENT_EVENTS, rng.randint(0, 1))
            title = rng.choice(_TITLES_JUNIOR)
            expected_rating = "cold"
            expected_score = rng.randint(1, 3)
            profile_completeness = round(rng.uniform(0.15, 0.55), 2)
            industry_fit = "low"

        else:  # edge
            # Edge: could be anything, possibly incomplete
            if rng.random() < 0.5:
                industry = rng.choice(
                    _INDUSTRIES_HIGH_FIT + _INDUSTRIES_MEDIUM_FIT + _INDUSTRIES_LOW_FIT
                )
            else:
                industry = rng.choice(["Unknown", "", "Other", "N/A"])
            employee_count = rng.choice([0, 1, rng.randint(1, 10000)])
            tech_stack = rng.sample(_TECH_STACKS, rng.randint(0, 3))
            signals = []
            engagement = rng.sample(_ENGAGEMENT_EVENTS, rng.randint(0, 1))
            title = rng.choice(_TITLES_JUNIOR + _TITLES_MID + _TITLES_SENIOR + ["", "Intern"])
            expected_rating = rng.choice(["hot", "warm", "cold"])
            expected_score = rng.randint(1, 9)
            profile_completeness = round(rng.uniform(0.0, 0.4), 2)
            industry_fit = "low"

        lead_profile = {
            "company_name": company,
            "industry": industry,
            "employee_count": employee_count,
            "tech_stack": tech_stack,
            "recent_signals": signals,
            "engagement": engagement,
            "contact": {
                "first_name": first,
                "last_name": last,
                "title": title,
                "email": email,
            },
        }

        structural_descriptor = {
            "expected_rating": expected_rating,
            "expected_score": expected_score,
            "profile_completeness": profile_completeness,
            "industry_fit": industry_fit,
        }

        return lead_profile, structural_descriptor

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one lead profile for sales qualification."""
        lead_profile, structural_descriptor = self._build_lead_profile(tier, profile, rng)

        input_data: dict[str, Any] = {
            "lead_profile": lead_profile,
        }

        # For dirty near_empty, replace the entire input
        if is_dirty and dirty_type == "near_empty":
            near_empty_profiles: list[dict[str, Any]] = [
                {"lead_profile": {}},
                {"lead_profile": {"company_name": ""}},
                {"lead_profile": {"company_name": "Unknown", "industry": "", "employee_count": 0,
                                   "tech_stack": [], "recent_signals": [], "engagement": [],
                                   "contact": {"first_name": "", "last_name": "", "title": "", "email": ""}}},
                {},
            ]
            input_data = rng.choice(near_empty_profiles)
        elif is_dirty and dirty_type == "typos":
            # Introduce typos in company name and title
            company = lead_profile["company_name"]
            if len(company) > 3:
                ci = rng.randint(1, len(company) - 2)
                company = company[:ci] + company[ci + 1] + company[ci] + company[ci + 2:]
            lead_profile["company_name"] = company

            title = lead_profile["contact"]["title"]
            if len(title) > 3:
                ti = rng.randint(1, len(title) - 2)
                title = title[:ti] + title[ti + 1] + title[ti] + title[ti + 2:]
            lead_profile["contact"]["title"] = title
            input_data = {"lead_profile": lead_profile}

        import json
        serialized = json.dumps(input_data, indent=2)

        # Pad/truncate serialized JSON to fit the target token range
        ranges = _TOKEN_RANGES.get(profile, _TOKEN_RANGES["profiling"])
        if tier in ranges:
            tmin, tmax = ranges[tier]
            serialized = self.pad_to_token_range(serialized, tmin, tmax, rng)

        token_count = self.estimate_tokens(serialized)

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W09",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=structural_descriptor,
            input_data=input_data,
        )


if __name__ == "__main__":
    add_cli(W09SalesOutreachGenerator)
