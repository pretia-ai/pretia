"""Input generator for W5: Multimodal document extraction.

Produce document inputs with modality drift between profiling and ground
truth.  Profiling uses 70% text / 30% image; GT uses 40% text / 60% image.
Dry-run mode uses template stubs — image content is a placeholder string.
When dry_run is False, image-modality inputs render the document text as a
scanned PNG via the existing PDF rendering + scan-simulation pipeline.
"""

from __future__ import annotations

import logging
import random
import tempfile
from pathlib import Path
from typing import Any

from inputs.generators._base import BaseInputGenerator, GeneratedInput, add_cli

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Token-length targets  (chars ≈ tokens * 4)
# ---------------------------------------------------------------------------
_TOKEN_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "profiling": {
        "easy":   (300, 800),
        "medium": (800, 2_000),
        "hard":   (1_500, 6_400),
        "edge":   (100, 6_400),
    },
    "ground_truth": {
        "easy":     (500, 1_200),
        "medium":   (1_200, 3_500),
        "hard":     (2_000, 8_000),
        "edge":     (100, 8_000),
        "extreme":  (5_000, 12_000),
    },
}

# ---------------------------------------------------------------------------
# Modality ratios
# ---------------------------------------------------------------------------
_MODALITY_RATIOS: dict[str, dict[str, float]] = {
    "profiling":     {"text": 0.70, "image": 0.20, "mixed": 0.10},
    "ground_truth":  {"text": 0.40, "image": 0.40, "mixed": 0.20},
}

# ---------------------------------------------------------------------------
# Document subtypes and field-count ranges per tier
# ---------------------------------------------------------------------------
_SUBTYPES = ["invoice", "receipt", "business_card", "form", "table"]
_SUBTYPE_WEIGHTS = [0.30, 0.25, 0.15, 0.15, 0.15]

_FIELD_RANGES: dict[str, tuple[int, int]] = {
    "easy":    (4, 6),
    "medium":  (8, 12),
    "hard":    (13, 20),
    "edge":    (1, 8),
    "extreme": (20, 35),
}

# ---------------------------------------------------------------------------
# Invoice templates (12)
# ---------------------------------------------------------------------------
_INVOICE_FIELDS = [
    "Invoice Number", "Invoice Date", "Due Date", "Vendor Name", "Vendor Address",
    "Vendor Tax ID", "Customer Name", "Customer Address", "Customer PO Number",
    "Line Item 1 Description", "Line Item 1 Quantity", "Line Item 1 Unit Price",
    "Line Item 1 Amount", "Line Item 2 Description", "Line Item 2 Quantity",
    "Line Item 2 Unit Price", "Line Item 2 Amount", "Line Item 3 Description",
    "Line Item 3 Quantity", "Line Item 3 Unit Price", "Line Item 3 Amount",
    "Subtotal", "Tax Rate", "Tax Amount", "Shipping", "Discount",
    "Total Amount Due", "Payment Terms", "Bank Name", "Bank Account Number",
    "Routing Number", "SWIFT Code", "Currency", "Notes", "Authorized Signature",
]

_INVOICE_TEMPLATES = [
    {"vendor": "Acme Industrial Supply Co.", "style": "standard", "currency": "USD"},
    {"vendor": "Global Tech Solutions Ltd.", "style": "modern", "currency": "USD"},
    {"vendor": "Berlin Manufacturing GmbH", "style": "european", "currency": "EUR"},
    {"vendor": "Sakura Electronics K.K.", "style": "japanese_bilingual", "currency": "JPY"},
    {"vendor": "Metro Office Supplies", "style": "simple", "currency": "USD"},
    {"vendor": "CloudServe Infrastructure Inc.", "style": "saas_recurring", "currency": "USD"},
    {"vendor": "Precision Medical Instruments", "style": "medical", "currency": "USD"},
    {"vendor": "Sunshine Catering & Events", "style": "service", "currency": "USD"},
    {"vendor": "Nordic Timber Exports AS", "style": "international", "currency": "NOK"},
    {"vendor": "QuickPrint Digital Services", "style": "minimal", "currency": "USD"},
    {"vendor": "Pacific Rim Logistics Pte Ltd", "style": "freight", "currency": "SGD"},
    {"vendor": "Heritage Construction LLC", "style": "construction", "currency": "USD"},
]

# ---------------------------------------------------------------------------
# Receipt templates (12)
# ---------------------------------------------------------------------------
_RECEIPT_FIELDS = [
    "Store Name", "Store Address", "Store Phone", "Transaction Date",
    "Transaction Time", "Cashier", "Register Number", "Item 1", "Item 1 Price",
    "Item 2", "Item 2 Price", "Item 3", "Item 3 Price", "Item 4", "Item 4 Price",
    "Subtotal", "Tax", "Total", "Payment Method", "Card Last Four",
    "Authorization Code", "Return Policy",
]

_RECEIPT_TEMPLATES = [
    {"store": "Walmart Supercenter #4521", "type": "retail"},
    {"store": "Whole Foods Market", "type": "grocery"},
    {"store": "CVS Pharmacy #8832", "type": "pharmacy"},
    {"store": "Home Depot #1247", "type": "hardware"},
    {"store": "Starbucks Reserve Roastery", "type": "food_beverage"},
    {"store": "Best Buy #945", "type": "electronics"},
    {"store": "Target Store T-2210", "type": "retail"},
    {"store": "Costco Wholesale #412", "type": "wholesale"},
    {"store": "Shell Gas Station", "type": "fuel"},
    {"store": "Marriott Hotel Downtown", "type": "hotel"},
    {"store": "Delta Air Lines", "type": "airline"},
    {"store": "Uber Rides", "type": "rideshare"},
]

# ---------------------------------------------------------------------------
# Business card templates (11)
# ---------------------------------------------------------------------------
_CARD_FIELDS = [
    "Full Name", "Title", "Company", "Department", "Email", "Phone",
    "Mobile", "Fax", "Address Line 1", "Address Line 2", "City",
    "State/Province", "Zip/Postal Code", "Country", "Website",
    "LinkedIn URL", "Twitter Handle",
]

_CARD_TEMPLATES = [
    {"name": "Sarah J. Mitchell", "title": "VP of Engineering", "company": "TechVenture Inc."},
    {"name": "Dr. Robert Chen", "title": "Chief Medical Officer", "company": "HealthFirst Labs"},
    {"name": "Maria Gonzalez", "title": "Partner", "company": "Gonzalez & Associates LLP"},
    {"name": "James Okonkwo", "title": "Director of Operations", "company": "Nexus Logistics"},
    {"name": "Yuki Tanaka", "title": "Senior Architect", "company": "Tanaka Design Studio"},
    {"name": "Hans Mueller", "title": "Managing Director", "company": "Deutsche Finanz AG"},
    {"name": "Priya Sharma", "title": "Lead Data Scientist", "company": "AI Dynamics"},
    {"name": "Michael O'Brien", "title": "CEO & Founder", "company": "GreenEnergy Solutions"},
    {"name": "Lisa Park", "title": "Marketing Director", "company": "BrandWave Media"},
    {"name": "Ahmed Al-Rashid", "title": "Regional Manager", "company": "Gulf Trading Co."},
    {"name": "Catherine Dubois", "title": "Head of Research", "company": "Institut Pasteur"},
]

# ---------------------------------------------------------------------------
# Form templates (11)
# ---------------------------------------------------------------------------
_FORM_FIELDS = [
    "Form Title", "Form Number", "Date", "Applicant Name", "Date of Birth",
    "SSN/Tax ID", "Address", "City", "State", "Zip Code", "Phone",
    "Email", "Employer Name", "Employer Address", "Annual Income",
    "Filing Status", "Dependents Count", "Signature", "Checkbox: I Agree",
    "Dropdown: State", "Text Area: Additional Notes",
]

_FORM_TEMPLATES = [
    {"title": "W-4 Employee Withholding Certificate", "type": "tax"},
    {"title": "I-9 Employment Eligibility Verification", "type": "employment"},
    {"title": "Insurance Claim Form", "type": "insurance"},
    {"title": "Loan Application", "type": "financial"},
    {"title": "Patient Intake Form", "type": "medical"},
    {"title": "Rental Application", "type": "housing"},
    {"title": "Passport Renewal Application", "type": "government"},
    {"title": "Vehicle Registration Form", "type": "dmv"},
    {"title": "Permit Application - Building", "type": "construction"},
    {"title": "Scholarship Application", "type": "education"},
    {"title": "Customs Declaration Form", "type": "border"},
]

# ---------------------------------------------------------------------------
# Table templates (11)
# ---------------------------------------------------------------------------
_TABLE_FIELDS = [
    "Column 1 Header", "Column 2 Header", "Column 3 Header", "Column 4 Header",
    "Row Count", "Cell Values", "Total Row", "Notes Column",
    "Merged Cells", "Nested Headers",
]

_TABLE_TEMPLATES = [
    {"title": "Quarterly Revenue by Region", "rows": 12, "cols": 5},
    {"title": "Employee Attendance Log", "rows": 20, "cols": 4},
    {"title": "Product Inventory Status", "rows": 15, "cols": 6},
    {"title": "Budget Allocation FY2025", "rows": 8, "cols": 7},
    {"title": "Lab Test Results Panel", "rows": 10, "cols": 5},
    {"title": "Shipping Manifest", "rows": 25, "cols": 8},
    {"title": "Vendor Price Comparison", "rows": 6, "cols": 5},
    {"title": "Student Grade Report", "rows": 30, "cols": 6},
    {"title": "Project Timeline Milestones", "rows": 12, "cols": 4},
    {"title": "Server Uptime Metrics", "rows": 7, "cols": 6},
    {"title": "Nutritional Facts Comparison", "rows": 10, "cols": 5},
]

# ---------------------------------------------------------------------------
# Edge-case qualities
# ---------------------------------------------------------------------------
_EDGE_QUALITIES = [
    "blurry", "rotated_90", "rotated_180", "upside_down", "low_contrast",
    "partial_occlusion", "blank_page", "watermarked", "stamped_over_text",
    "coffee_stained", "faded_ink", "crumpled",
]


def _render_document_image(
    text_content: str,
    output_path: Path,
    rng: random.Random,
    edge_quality: str = "",
) -> Path:
    """Render document text as a scanned PNG image.

    Uses the existing PDF rendering + scan-simulation pipeline:
    1. Render text as a 1-page PDF via reportlab (text_renderer)
    2. Rasterize the PDF page to a PIL Image (scan_simulator)
    3. Save as PNG

    Returns the path to the saved PNG.
    """
    from pdfs.generators.rendering.scan_simulator import (
        ScanParams,
        randomize_scan_params,
        rasterize_pdf_page,
    )
    from pdfs.generators.rendering.text_renderer import render_markdown_to_pdf

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Map edge qualities to degraded scan parameters
    quality_overrides: dict[str, ScanParams] = {
        "blurry": ScanParams(dpi=120, blur_sigma=2.5, noise_sigma=6.0),
        "rotated_90": ScanParams(rotation_degrees=90.0),
        "rotated_180": ScanParams(rotation_degrees=180.0),
        "upside_down": ScanParams(rotation_degrees=180.0),
        "low_contrast": ScanParams(contrast_factor=0.4),
        "faded_ink": ScanParams(contrast_factor=0.35, noise_sigma=12.0),
        "crumpled": ScanParams(blur_sigma=1.5, noise_sigma=10.0, contrast_factor=0.7),
        "coffee_stained": ScanParams(contrast_factor=0.6, noise_sigma=8.0),
    }

    with tempfile.TemporaryDirectory() as tmp:
        pdf_path = Path(tmp) / "doc.pdf"
        render_markdown_to_pdf(text_content, pdf_path)

        if edge_quality and edge_quality in quality_overrides:
            params = quality_overrides[edge_quality]
        else:
            params = randomize_scan_params(rng)

        pil_image = rasterize_pdf_page(pdf_path, 0, params)

    pil_image.save(str(output_path), format="PNG")
    logger.debug("Rendered document image: %s", output_path)
    return output_path


def _select_fields(
    subtype: str, n_fields: int, rng: random.Random,
) -> list[str]:
    """Select n_fields from the field pool for the given subtype."""
    if subtype == "invoice":
        pool = _INVOICE_FIELDS
    elif subtype == "receipt":
        pool = _RECEIPT_FIELDS
    elif subtype == "business_card":
        pool = _CARD_FIELDS
    elif subtype == "form":
        pool = _FORM_FIELDS
    else:
        pool = _TABLE_FIELDS

    n = min(n_fields, len(pool))
    return rng.sample(pool, n)


_STREET_NAMES = [
    "Main St", "Oak Ave", "Elm Blvd", "Park Dr", "Cedar Ln",
    "Market St", "Broadway", "5th Ave", "Industrial Pkwy", "Commerce Way",
]
_CITIES = [
    "San Francisco, CA 94105", "New York, NY 10001", "Chicago, IL 60601",
    "Austin, TX 78701", "Seattle, WA 98101", "Boston, MA 02101",
    "Denver, CO 80202", "Portland, OR 97201", "Miami, FL 33101",
]
_FIRST_NAMES = ["James", "Maria", "David", "Sarah", "Michael", "Lisa", "Robert", "Anna"]
_LAST_NAMES = ["Smith", "Johnson", "Chen", "Garcia", "Kim", "Patel", "Miller", "Brown"]
_BANK_NAMES = ["Chase", "Bank of America", "Wells Fargo", "Citibank", "US Bank"]
_PAYMENT_METHODS = ["Visa", "Mastercard", "Amex", "Cash", "Debit", "Apple Pay"]
_DEPARTMENTS = ["Engineering", "Sales", "Marketing", "Operations", "Finance", "HR"]
_COUNTRIES = ["United States", "Canada", "United Kingdom", "Germany", "Japan"]
_TABLE_TEXT_VALS = [
    "Northeast", "Southeast", "Midwest", "West", "Central",
    "Q1", "Q2", "Q3", "Q4", "YTD", "Active", "Pending", "Shipped",
    "In Stock", "Low Stock", "Approved", "Under Review", "Completed",
]


def _gen_address(rng: random.Random) -> str:
    return f"{rng.randint(100, 9999)} {rng.choice(_STREET_NAMES)}"


def _gen_full_name(rng: random.Random) -> str:
    return f"{rng.choice(_FIRST_NAMES)} {rng.choice(_LAST_NAMES)}"


def _gen_phone(rng: random.Random) -> str:
    return (
        f"+1 ({rng.randint(200, 999)}) "
        f"{rng.randint(200, 999)}-{rng.randint(1000, 9999)}"
    )


def _invoice_field_val(
    field: str, rng: random.Random, template: dict[str, Any],
) -> str:
    currency = template.get("currency", "USD")
    if "Amount" in field or "Price" in field or "Total" in field:
        return f"{currency} {rng.uniform(10, 10000):.2f}"
    if "Date" in field:
        return f"202{rng.randint(3, 6)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if "Quantity" in field:
        return str(rng.randint(1, 100))
    if "Rate" in field:
        return f"{rng.uniform(5, 15):.1f}%"
    if "Description" in field:
        items = ["Consulting Services", "Software License", "Hardware Kit",
                 "Training Session", "Support Plan", "Data Migration"]
        return rng.choice(items)
    if field == "Vendor Address" or field == "Customer Address":
        return f"{_gen_address(rng)}, {rng.choice(_CITIES)}"
    if field == "Customer Name":
        return _gen_full_name(rng)
    if "PO Number" in field or "Number" in field or "ID" in field:
        return f"INV-{rng.randint(100000, 999999)}"
    if field == "Bank Name":
        return rng.choice(_BANK_NAMES)
    if "Account" in field:
        return f"****{rng.randint(1000, 9999)}"
    if "SWIFT" in field:
        return f"{''.join(rng.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=8))}"
    if field == "Payment Terms":
        return rng.choice(["NET-30", "NET-60", "Due on Receipt", "NET-15"])
    if field == "Notes":
        return "Thank you for your business."
    if field == "Authorized Signature":
        return _gen_full_name(rng)
    if field == "Shipping" or field == "Discount":
        return f"{currency} {rng.uniform(0, 500):.2f}"
    return f"{currency} {rng.uniform(1, 1000):.2f}"


def _receipt_field_val(
    field: str, rng: random.Random, template: dict[str, Any],
) -> str:
    if "Price" in field or "Total" in field or "Tax" in field or "Subtotal" in field:
        return f"${rng.uniform(0.99, 299.99):.2f}"
    if "Date" in field:
        return f"{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/202{rng.randint(3, 6)}"
    if "Time" in field:
        return f"{rng.randint(6, 23):02d}:{rng.randint(0, 59):02d}"
    if "Four" in field:
        return f"****{rng.randint(1000, 9999)}"
    if "Item" in field and "Price" not in field:
        items = ["Organic Milk 1gal", "Whole Wheat Bread", "Chicken Breast 2lb",
                 "AA Batteries 8pk", "Paper Towels 6-roll", "Coffee Beans 12oz",
                 "Shampoo 16oz", "USB-C Cable 6ft", "LED Bulb 4pk"]
        return rng.choice(items)
    if field == "Store Name":
        return template.get("store", "Store")
    if field == "Store Address":
        return f"{_gen_address(rng)}, {rng.choice(_CITIES)}"
    if field == "Store Phone":
        return _gen_phone(rng)
    if field == "Cashier":
        return _gen_full_name(rng)
    if field == "Register Number":
        return f"REG-{rng.randint(1, 20):02d}"
    if field == "Authorization Code":
        return f"{rng.randint(100000, 999999)}"
    if field == "Payment Method":
        return rng.choice(_PAYMENT_METHODS)
    if field == "Return Policy":
        return f"Returns accepted within {rng.choice([14, 30, 60, 90])} days with receipt."
    return f"${rng.uniform(1, 100):.2f}"


def _card_field_val(
    field: str, rng: random.Random, template: dict[str, Any],
) -> str:
    if "Email" in field:
        name = template.get("name", "John Doe").lower().replace(" ", ".")
        name = name.replace("dr. ", "")
        company = template.get("company", "co").lower().replace(" ", "")
        return f"{name}@{company}.com"
    if "Phone" in field or "Mobile" in field or "Fax" in field:
        return _gen_phone(rng)
    if "LinkedIn" in field:
        slug = template.get("name", "johndoe").lower().replace(" ", "-")
        return f"linkedin.com/in/{slug}"
    if "Twitter" in field:
        slug = template.get("name", "johndoe").lower().replace(" ", "")
        return f"@{slug}"
    if field == "Website":
        company = template.get("company", "company").lower().replace(" ", "")
        return f"www.{company}.com"
    if field == "Department":
        return rng.choice(_DEPARTMENTS)
    if "Address" in field:
        return _gen_address(rng)
    if field == "City":
        return rng.choice(_CITIES).split(",")[0]
    if "State" in field:
        return rng.choice(["CA", "NY", "TX", "WA", "MA", "IL", "CO", "FL"])
    if "Zip" in field or "Postal" in field:
        return str(rng.randint(10000, 99999))
    if field == "Country":
        return rng.choice(_COUNTRIES)
    return _gen_full_name(rng)


def _form_field_val(
    field: str, rng: random.Random, template: dict[str, Any],
) -> str:
    if "Date" in field:
        return f"202{rng.randint(3, 6)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    if "Income" in field:
        return f"${rng.randint(25000, 250000):,}"
    if "SSN" in field or "Tax ID" in field:
        return f"XXX-XX-{rng.randint(1000, 9999)}"
    if "Checkbox" in field:
        return "[X]" if rng.random() > 0.3 else "[ ]"
    if "Count" in field:
        return str(rng.randint(0, 5))
    if "Name" in field and "Employer" not in field:
        return _gen_full_name(rng)
    if "Employer Name" in field:
        employers = ["Acme Corp", "TechVenture Inc", "Metro Services LLC",
                     "Pacific Holdings", "Summit Group"]
        return rng.choice(employers)
    if "Address" in field or "Employer Address" in field:
        return f"{_gen_address(rng)}, {rng.choice(_CITIES)}"
    if field == "Phone":
        return _gen_phone(rng)
    if field == "Email":
        return f"{_gen_full_name(rng).lower().replace(' ', '.')}@email.com"
    if field == "City":
        return rng.choice(_CITIES).split(",")[0]
    if field == "State":
        return rng.choice(["CA", "NY", "TX", "WA", "MA"])
    if "Zip" in field:
        return str(rng.randint(10000, 99999))
    if "Filing Status" in field:
        return rng.choice(["Single", "Married Filing Jointly", "Head of Household"])
    if "Signature" in field:
        return _gen_full_name(rng)
    if "Dropdown" in field:
        return rng.choice(["CA", "NY", "TX", "WA", "FL"])
    if "Text Area" in field:
        return "No additional notes at this time."
    if "Form Number" in field or "Number" in field:
        return f"F-{rng.randint(1000, 9999)}"
    if "Form Title" in field:
        return template.get("title", "Form")
    return _gen_full_name(rng)


def _generate_text_content(
    subtype: str, fields: list[str], rng: random.Random, template: dict[str, Any],
) -> str:
    """Build a markdown representation of the document.

    Uses markdown headings + bullet lists so the PDF renderer produces
    properly structured pages with line breaks.
    """
    lines: list[str] = []

    if subtype == "invoice":
        vendor = template.get("vendor", "Vendor Co.")
        lines.append(f"# INVOICE")
        lines.append("")
        lines.append(f"**From:** {vendor}")
        lines.append("")
        for field in fields:
            val = _invoice_field_val(field, rng, template)
            lines.append(f"- **{field}:** {val}")

    elif subtype == "receipt":
        store = template.get("store", "Store")
        lines.append(f"# {store}")
        lines.append("")
        for field in fields:
            val = _receipt_field_val(field, rng, template)
            lines.append(f"- **{field}:** {val}")

    elif subtype == "business_card":
        name = template.get("name", "John Doe")
        title = template.get("title", "Manager")
        company = template.get("company", "Company")
        lines.append(f"# {name}")
        lines.append(f"**{title}** | {company}")
        lines.append("")
        for field in fields:
            if field in ("Full Name", "Title", "Company"):
                continue
            val = _card_field_val(field, rng, template)
            lines.append(f"- **{field}:** {val}")

    elif subtype == "form":
        form_title = template.get("title", "Form")
        lines.append(f"# {form_title}")
        lines.append(f"*Type: {template.get('type', 'general')}*")
        lines.append("")
        for field in fields:
            val = _form_field_val(field, rng, template)
            lines.append(f"- **{field}:** {val}")

    else:  # table
        title = template.get("title", "Data Table")
        rows = template.get("rows", 10)
        cols = min(template.get("cols", 4), len(fields))
        lines.append(f"# {title}")
        lines.append("")
        headers = fields[:cols]
        lines.append("| " + " | ".join(headers) + " |")
        lines.append("| " + " | ".join("---" for _ in headers) + " |")
        for _ in range(min(rows, 20)):
            row_vals = []
            for h in headers:
                low = h.lower()
                if any(k in low for k in ("total", "amount", "price", "value")):
                    row_vals.append(f"${rng.uniform(10, 9999):.2f}")
                elif any(k in low for k in ("count", "row", "qty")):
                    row_vals.append(str(rng.randint(1, 500)))
                else:
                    row_vals.append(rng.choice(_TABLE_TEXT_VALS))
            lines.append("| " + " | ".join(row_vals) + " |")

    return "\n".join(lines)


def _generate_image_placeholder(
    subtype: str, n_fields: int, rng: random.Random, template: dict[str, Any],
    quality: str = "high",
) -> str:
    """Build a placeholder for an image-based document."""
    desc_parts = [subtype]
    if subtype == "invoice":
        desc_parts.append(f"from {template.get('vendor', 'vendor')}")
    elif subtype == "receipt":
        desc_parts.append(f"from {template.get('store', 'store')}")
    elif subtype == "business_card":
        desc_parts.append(f"for {template.get('name', 'person')}")
    elif subtype == "form":
        desc_parts.append(f"({template.get('title', 'form')})")
    else:
        desc_parts.append(f"({template.get('title', 'table')})")

    quality_note = ""
    if quality != "high":
        quality_note = f", quality={quality}"

    return f"[IMAGE: {' '.join(desc_parts)} with {n_fields} fields{quality_note}]"


class MultimodalExtractionGenerator(BaseInputGenerator):
    """Generate multimodal document extraction inputs for W5."""

    workflow_id = "W5"
    dirty_types: list[str] = []  # W5 not in dirty input table

    def __init__(self, seed: int = 42) -> None:
        super().__init__(seed=seed)
        self.dry_run = True
        self._forced_modalities: dict[int, str] = {}
        self._batch_idx: int | None = None
        self._image_output_dir: str | None = None

    def _pick_subtype(self, rng: random.Random) -> str:
        """Select document subtype with weighted distribution."""
        r = rng.random()
        cumulative = 0.0
        for st, w in zip(_SUBTYPES, _SUBTYPE_WEIGHTS):
            cumulative += w
            if r <= cumulative:
                return st
        return _SUBTYPES[-1]

    def _pick_template(
        self, subtype: str, rng: random.Random,
    ) -> dict[str, Any]:
        """Select a random template for the given subtype."""
        pools: dict[str, list[dict[str, Any]]] = {
            "invoice": _INVOICE_TEMPLATES,
            "receipt": _RECEIPT_TEMPLATES,
            "business_card": _CARD_TEMPLATES,
            "form": _FORM_TEMPLATES,
            "table": _TABLE_TEMPLATES,
        }
        return rng.choice(pools[subtype])

    def _assign_modality(
        self, profile: str, rng: random.Random, tier: str,
    ) -> str:
        """Assign modality — uses forced assignment when set by generate_batch."""
        if self._forced_modalities and self._batch_idx is not None:
            forced = self._forced_modalities.get(self._batch_idx)
            if forced is not None:
                return forced

        if tier == "edge":
            return rng.choice(["image", "mixed"])

        ratios = _MODALITY_RATIOS.get(profile, _MODALITY_RATIOS["profiling"])
        r = rng.random()
        cumulative = 0.0
        for modality, weight in ratios.items():
            cumulative += weight
            if r <= cumulative:
                return modality
        return "text"

    def _pick_resolution(self, tier: str, rng: random.Random) -> str:
        if tier == "edge":
            return rng.choice(["low", "low", "medium"])
        if tier in ("hard", "extreme"):
            return rng.choice(["medium", "high", "low"])
        return rng.choice(["high", "high", "medium"])

    def generate_single(
        self,
        tier: str,
        profile: str,
        rng: random.Random,
        idx: int,
        is_dirty: bool = False,
        dirty_type: str | None = None,
    ) -> GeneratedInput:
        """Generate one multimodal extraction input."""
        subtype = self._pick_subtype(rng)
        template = self._pick_template(subtype, rng)
        modality = self._assign_modality(profile, rng, tier)
        resolution = self._pick_resolution(tier, rng)

        # Field count
        lo, hi = _FIELD_RANGES.get(tier, (4, 6))
        field_count = rng.randint(lo, hi)

        fields = _select_fields(subtype, field_count, rng)

        # Edge-case quality
        edge_quality = ""
        if tier == "edge":
            edge_quality = rng.choice(_EDGE_QUALITIES)

        # Build content based on modality
        handwriting_present = False
        image_path: str | None = None
        input_id = self.make_id(profile, tier, idx)

        if modality == "text":
            content = _generate_text_content(subtype, fields, rng, template)
        elif modality == "image":
            # Generate the document text (used for both placeholder and real image)
            doc_text = _generate_text_content(subtype, fields, rng, template)
            handwriting_present = rng.random() > 0.7

            if not self.dry_run:
                img_dir = Path(self._image_output_dir or "inputs/generated/images/w5")
                img_path = img_dir / f"{input_id}.png"
                _render_document_image(doc_text, img_path, rng, edge_quality)
                image_path = str(img_path)
                field_sample = rng.sample(fields, min(3, len(fields)))
                content = (
                    f"Extract all {field_count} fields from this {subtype}. "
                    f"Include {', '.join(field_sample)}, and any other visible data."
                )
            else:
                quality = edge_quality if edge_quality else resolution
                content = _generate_image_placeholder(
                    subtype, field_count, rng, template, quality,
                )
        else:
            # mixed: text part + image part
            text_fields = fields[: len(fields) // 2]
            image_fields = fields[len(fields) // 2 :]
            text_part = _generate_text_content(subtype, text_fields, rng, template)
            image_doc_text = _generate_text_content(subtype, image_fields, rng, template)
            handwriting_present = rng.random() > 0.8

            if not self.dry_run:
                img_dir = Path(self._image_output_dir or "inputs/generated/images/w5")
                img_path = img_dir / f"{input_id}.png"
                _render_document_image(image_doc_text, img_path, rng, edge_quality)
                image_path = str(img_path)
                img_field_sample = rng.sample(image_fields, min(2, len(image_fields)))
                content = (
                    f"{text_part}\n\n"
                    f"The attached image contains additional fields including "
                    f"{', '.join(img_field_sample)}. Extract those as well."
                )
            else:
                quality = edge_quality if edge_quality else resolution
                image_part = _generate_image_placeholder(
                    subtype, field_count - len(text_fields), rng, template, quality,
                )
                content = f"{text_part}\n\n--- IMAGE SECTION ---\n{image_part}"

        # Pad/truncate text-only content to hit target token range.
        # Image and mixed inputs skip padding — their cost driver is image
        # resolution (vision tokens), not text length.
        if modality == "text":
            ranges = _TOKEN_RANGES.get(profile, _TOKEN_RANGES["profiling"])
            if tier in ranges:
                tmin, tmax = ranges[tier]
                content = self.pad_to_token_range(content, tmin, tmax, rng)

        # Apply style shift for GT text-based inputs
        if modality == "text":
            content = self.apply_style_shift(content, profile)

        token_count = self.estimate_tokens(content)

        structural_descriptor: dict[str, Any] = {
            "modality": modality,
            "document_subtype": subtype,
            "field_count": field_count,
            "image_resolution": resolution,
            "handwriting_present": handwriting_present,
        }
        if edge_quality:
            structural_descriptor["edge_quality"] = edge_quality

        input_data: dict[str, Any] = {
            "modality": modality,
            "content": content,
            "document_subtype": subtype,
            "input": content,
        }
        if image_path:
            input_data["image_path"] = image_path

        return GeneratedInput(
            id=self.make_id(profile, tier, idx),
            workflow="W5",
            profile=profile,
            tier=tier,
            token_count=token_count,
            is_dirty=is_dirty,
            dirty_type=dirty_type,
            structural_descriptor=structural_descriptor,
            input_data=input_data,
        )

    def generate_batch(
        self,
        profile: str,
        n: int,
    ) -> list[GeneratedInput]:
        """Generate batch enforcing modality ratios at the batch level.

        Override the base to pre-assign modalities so batch-level distribution
        matches the target ratios (70/20/10 profiling, 40/40/20 GT) instead
        of relying on per-input random draws which drift for small n.
        """
        self.rng = random.Random(self.seed)
        tiers = self.allocate_tiers(profile, n)
        dirty_map = self.select_dirty_indices(n, tiers)

        ratios = _MODALITY_RATIOS.get(profile, _MODALITY_RATIOS["profiling"])
        non_edge_indices = [i for i, t in enumerate(tiers) if t != "edge"]
        edge_indices = [i for i, t in enumerate(tiers) if t == "edge"]

        modality_assignments: dict[int, str] = {}

        for i in edge_indices:
            modality_assignments[i] = self.rng.choice(["image", "mixed"])

        n_non_edge = len(non_edge_indices)
        if n_non_edge > 0:
            text_count = round(n_non_edge * ratios.get("text", 0.5))
            image_count = round(n_non_edge * ratios.get("image", 0.3))
            mixed_count = n_non_edge - text_count - image_count

            assignments = (
                ["text"] * text_count
                + ["image"] * image_count
                + ["mixed"] * max(0, mixed_count)
            )
            while len(assignments) < n_non_edge:
                assignments.append("text")
            assignments = assignments[:n_non_edge]
            self.rng.shuffle(assignments)

            for i, modality in zip(non_edge_indices, assignments):
                modality_assignments[i] = modality

        self._forced_modalities = modality_assignments

        inputs: list[GeneratedInput] = []
        for idx in range(n):
            tier = tiers[idx]
            is_dirty = idx in dirty_map
            dirty_type_val = dirty_map.get(idx)

            self._batch_idx = idx
            inp = self.generate_single(
                tier=tier,
                profile=profile,
                rng=self.rng,
                idx=idx,
                is_dirty=is_dirty,
                dirty_type=dirty_type_val,
            )
            inputs.append(inp)

        self._forced_modalities = {}
        self._batch_idx = None
        return inputs


if __name__ == "__main__":
    add_cli(MultimodalExtractionGenerator)
