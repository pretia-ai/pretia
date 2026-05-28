"""Shared utilities for backtesting workflows."""

from __future__ import annotations

from typing import Any

_ANTHROPIC_TIERS = {
    "haiku": "claude-haiku-4-5",
    "sonnet": "claude-sonnet-4-6",
    "opus": "claude-opus-4-7",
}

_OPENAI_TIERS = {
    "nano": "gpt-4.1-nano",
    "mini": "gpt-4.1-mini",
    "standard": "gpt-4.1",
    "flagship": "gpt-5.5",
}

_GEMINI_TIERS = {
    "flash": "gemini-2.5-flash",
    "pro": "gemini-2.5-pro",
}

_DEEPSEEK_TIERS = {
    "flash": "deepseek-v4-flash",
    "pro": "deepseek-v4-pro",
}

_QWEN_TIERS = {
    "turbo": "qwen-turbo",
    "plus": "qwen3.6-plus",
    "max": "qwen3.7-max",
}


def get_anthropic_model(tier: str) -> str:
    """Return a canonical Anthropic model string validated against the pricing table."""
    try:
        from agentcost.pricing.tables import resolve_model
        return resolve_model(_ANTHROPIC_TIERS[tier])
    except Exception:
        return _ANTHROPIC_TIERS[tier]


def get_openai_model(tier: str) -> str:
    """Return a canonical OpenAI model string validated against the pricing table."""
    try:
        from agentcost.pricing.tables import resolve_model
        return resolve_model(_OPENAI_TIERS[tier])
    except Exception:
        return _OPENAI_TIERS[tier]


def get_gemini_model(tier: str) -> str:
    """Return a canonical Gemini model string validated against the pricing table."""
    try:
        from agentcost.pricing.tables import resolve_model
        return resolve_model(_GEMINI_TIERS[tier])
    except Exception:
        return _GEMINI_TIERS[tier]


def get_deepseek_model(tier: str) -> str:
    """Return a canonical DeepSeek model string validated against the pricing table."""
    try:
        from agentcost.pricing.tables import resolve_model
        return resolve_model(_DEEPSEEK_TIERS[tier])
    except Exception:
        return _DEEPSEEK_TIERS[tier]


def get_qwen_model(tier: str) -> str:
    """Return a canonical Qwen model string validated against the pricing table."""
    try:
        from agentcost.pricing.tables import resolve_model
        return resolve_model(_QWEN_TIERS[tier])
    except Exception:
        return _QWEN_TIERS[tier]


def stub_tool(name: str, data: dict[str, Any]) -> dict[str, Any]:
    """Return canned data for a tool stub."""
    return {"tool": name, **data}


CANNED_FAQ = {
    "billing": (
        "ProjectFlow Billing FAQ\n\n"
        "Subscription Plans: We offer Starter ($29/mo), Professional ($79/mo), and "
        "Enterprise (custom pricing). All plans include unlimited projects. Professional "
        "adds advanced reporting, priority support, and API access. Enterprise adds SSO, "
        "audit logs, and a dedicated account manager.\n\n"
        "Payment Methods: We accept all major credit cards (Visa, Mastercard, Amex) and "
        "ACH bank transfers for annual plans. Enterprise customers can pay via invoice "
        "with NET-30 terms.\n\n"
        "Refund Policy: We offer a full refund within 14 days of purchase. After 14 days, "
        "we prorate remaining time if you downgrade. No refunds for partial months."
    ),
    "technical": (
        "ProjectFlow Technical FAQ\n\n"
        "API Rate Limits: Free tier gets 100 requests/minute, Professional gets 1,000/min, "
        "Enterprise gets 10,000/min. Rate limit headers are included in every response: "
        "X-RateLimit-Remaining and X-RateLimit-Reset.\n\n"
        "Integrations: We support Slack, GitHub, Jira, Asana, and Zapier out of the box. "
        "Custom integrations are available via our REST API and webhooks. Webhook events "
        "include task.created, task.updated, task.completed, and sprint.closed.\n\n"
        "Data Export: All plans support CSV export. Professional and Enterprise support "
        "JSON and XML export via the API. Enterprise adds scheduled exports to S3."
    ),
    "account": (
        "ProjectFlow Account FAQ\n\n"
        "Password Reset: Click 'Forgot Password' on the login page. Enter your email and "
        "we'll send a reset link valid for 24 hours. If the link expires, request a new "
        "one. If you don't receive the email, check your spam folder or contact support.\n\n"
        "Team Management: Admins can invite team members from Settings > Team. Each member "
        "can be assigned a role: Admin (full access), Member (create/edit), or Viewer "
        "(read-only). Members are billed per seat on Professional and Enterprise plans.\n\n"
        "Account Deletion: Contact support@projectflow.io to request account deletion. "
        "We'll export your data within 48 hours and delete the account within 30 days."
    ),
    "general": (
        "ProjectFlow General FAQ\n\n"
        "Getting Started: Create your first project from the dashboard. Add tasks, set "
        "due dates, and assign team members. Use boards for visual workflow management "
        "or lists for traditional task tracking. Our onboarding guide walks you through "
        "the first 5 minutes: docs.projectflow.io/getting-started.\n\n"
        "Mobile Apps: ProjectFlow is available on iOS and Android. Download from the "
        "App Store or Google Play. The mobile app supports push notifications, task "
        "creation, and commenting. Full editing requires the web app.\n\n"
        "Uptime: We maintain 99.9% uptime SLA for Professional and Enterprise plans. "
        "Status page: status.projectflow.io."
    ),
    "escalation": (
        "ProjectFlow Escalation Procedures\n\n"
        "If you're experiencing a critical issue affecting your team's productivity, "
        "please contact our priority support team at urgent@projectflow.io or call "
        "+1-800-555-FLOW. Enterprise customers have a dedicated Slack channel.\n\n"
        "For billing disputes over $500, our finance team will review within 2 business "
        "days. For data loss incidents, we have a 4-hour response SLA for Enterprise "
        "customers.\n\n"
        "If you're unsatisfied with a support interaction, you can request supervisor "
        "review by replying to any support email with 'ESCALATE' in the subject line."
    ),
}

CANNED_SEARCH_RESULTS = [
    (
        "Market Analysis Report (2026)\n\n"
        "The global SaaS market reached $420 billion in 2025, growing at 18% CAGR. "
        "AI-powered tools represent the fastest-growing segment at 35% year-over-year. "
        "Key drivers include enterprise automation, remote work infrastructure, and "
        "generative AI integration. North America accounts for 45% of revenue, followed "
        "by Europe (28%) and Asia-Pacific (20%). The mid-market segment ($10M-$500M ARR) "
        "is seeing the most competitive dynamics."
    ),
    (
        "Industry Trends Deep Dive\n\n"
        "Three trends are reshaping the B2B software landscape: (1) AI-native products "
        "that embed intelligence in core workflows, not as add-ons. (2) Usage-based "
        "pricing replacing per-seat models — 62% of new SaaS companies now offer some "
        "form of consumption pricing. (3) Vertical specialization over horizontal "
        "platforms — the winners are going deep in specific industries rather than broad."
    ),
    (
        "Competitive Landscape Summary\n\n"
        "The market is consolidating around three tiers: (1) Platform companies (Salesforce, "
        "Microsoft, Google) that bundle AI across their suite. (2) Category leaders "
        "(Datadog, Snowflake, Figma) that dominate specific workflows. (3) AI-native "
        "startups (Cursor, Vercel, Anthropic) that are redefining how software is built "
        "and used. The key battleground is developer tools, where AI code assistants "
        "have reached 70% adoption among professional developers."
    ),
]

CANNED_COMPANY_PROFILES = {
    "techcorp": (
        "TechCorp Inc. — Series C startup, $85M raised, 200 employees. "
        "Builds developer productivity tools. Key products: CodeAssist (AI pair programmer), "
        "DeployFast (CI/CD platform). Revenue: $12M ARR, growing 150% YoY. "
        "ICP: engineering teams at mid-market SaaS companies. Pain points: slow release "
        "cycles, high infrastructure costs, developer burnout."
    ),
    "megabank": (
        "MegaBank Financial — Fortune 500, 45,000 employees. Global retail and investment "
        "banking. Technology budget: $2.3B/year. Currently modernizing legacy systems. "
        "Key initiatives: cloud migration (AWS), AI risk modeling, digital customer "
        "experience. Pain points: regulatory compliance overhead, technical debt, talent "
        "retention. Decision cycle: 6-12 months. Requires SOC 2 and FedRAMP."
    ),
    "greenstart": (
        "GreenStart Energy — Seed-stage climate tech startup, $3M raised, 15 employees. "
        "Building a carbon credit marketplace using blockchain verification. Pre-revenue, "
        "targeting $1M ARR by end of year. ICP: corporate sustainability officers. "
        "Pain points: carbon credit fraud, lack of standardization, high verification costs."
    ),
}

CANNED_DOCUMENTS = {
    "invoice": (
        "INVOICE #INV-2026-0847\n"
        "Date: March 15, 2026\n"
        "Due Date: April 14, 2026\n\n"
        "From: Acme Consulting LLC\n"
        "123 Business Ave, Suite 400\n"
        "San Francisco, CA 94105\n\n"
        "To: Widget Corp International\n"
        "456 Enterprise Blvd\n"
        "New York, NY 10001\n\n"
        "Description                    Qty    Rate      Amount\n"
        "Strategy Consulting (hourly)    40    $250.00   $10,000.00\n"
        "Market Research Report           1    $5,000    $5,000.00\n"
        "Travel Expenses (NYC trip)       1    $1,247.50 $1,247.50\n\n"
        "Subtotal: $16,247.50\n"
        "Tax (8.875%): $1,441.97\n"
        "Total Due: $17,689.47\n\n"
        "Payment Terms: NET-30. Late payments subject to 1.5% monthly interest."
    ),
    "contract": (
        "MASTER SERVICES AGREEMENT\n\n"
        "This Agreement is entered into as of January 1, 2026 between DataFlow Inc. "
        "('Provider') and RetailMax Corp ('Client').\n\n"
        "1. SERVICES: Provider shall deliver data analytics and machine learning model "
        "development services as described in each Statement of Work.\n\n"
        "2. TERM: Initial term of 24 months, auto-renewing for 12-month periods unless "
        "either party provides 90 days written notice.\n\n"
        "3. FEES: Client shall pay $45,000 per month for the base platform. Additional "
        "usage billed at $0.002 per API call above 10M monthly. Annual commitment: "
        "$540,000.\n\n"
        "4. CONFIDENTIALITY: Both parties agree to protect Confidential Information for "
        "5 years following disclosure.\n\n"
        "5. LIABILITY: Provider's total liability shall not exceed fees paid in the "
        "preceding 12 months."
    ),
}

CANNED_CODE_DIFFS = {
    "small_refactor": (
        "diff --git a/src/auth/login.py b/src/auth/login.py\n"
        "--- a/src/auth/login.py\n"
        "+++ b/src/auth/login.py\n"
        "@@ -15,8 +15,12 @@ def authenticate(username: str, password: str) -> User | None:\n"
        "-    user = db.query(User).filter(User.username == username).first()\n"
        "-    if user and check_password(password, user.password_hash):\n"
        "-        return user\n"
        "-    return None\n"
        "+    user = db.query(User).filter(User.username == username).first()\n"
        "+    if user is None:\n"
        "+        log.info('Login failed: unknown user %s', username)\n"
        "+        return None\n"
        "+    if not check_password(password, user.password_hash):\n"
        "+        log.warning('Login failed: bad password for %s', username)\n"
        "+        return None\n"
        "+    log.info('Login success: %s', username)\n"
        "+    return user\n"
    ),
    "large_feature": (
        "diff --git a/src/billing/subscriptions.py b/src/billing/subscriptions.py\n"
        "--- a/src/billing/subscriptions.py\n"
        "+++ b/src/billing/subscriptions.py\n"
        "@@ -1,5 +1,45 @@\n"
        "+from datetime import datetime, timedelta\n"
        "+from decimal import Decimal\n"
        "+from typing import Optional\n"
        "+\n"
        "+from src.models import Plan, Subscription, User\n"
        "+from src.payments import PaymentGateway\n"
        "+from src.notifications import send_email\n"
        "+\n"
        "+class SubscriptionManager:\n"
        "+    def __init__(self, gateway: PaymentGateway):\n"
        "+        self.gateway = gateway\n"
        "+\n"
        "+    def upgrade(self, user: User, new_plan: Plan) -> Subscription:\n"
        "+        current = user.subscription\n"
        "+        if current.plan.tier >= new_plan.tier:\n"
        "+            raise ValueError('Cannot downgrade via upgrade endpoint')\n"
        "+        prorated = self._calculate_proration(current, new_plan)\n"
        "+        charge = self.gateway.charge(user, prorated)\n"
        "+        if not charge.success:\n"
        "+            raise PaymentError(charge.error)\n"
        "+        current.plan = new_plan\n"
        "+        current.upgraded_at = datetime.utcnow()\n"
        "+        send_email(user.email, 'plan_upgraded', plan=new_plan.name)\n"
        "+        return current\n"
        "+\n"
        "+    def _calculate_proration(self, sub: Subscription, new: Plan) -> Decimal:\n"
        "+        remaining_days = (sub.period_end - datetime.utcnow()).days\n"
        "+        daily_old = sub.plan.price / Decimal('30')\n"
        "+        daily_new = new.price / Decimal('30')\n"
        "+        return (daily_new - daily_old) * remaining_days\n"
    ),
}
