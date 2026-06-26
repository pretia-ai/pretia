# Directions: PDF Generation for W14, W15, W16, W17, W18

**Purpose:** This file specifies how Claude Code must build the PDF generation system — the code that produces synthetic PDF corpora for the five workflows that process documents. Each workflow gets its own corpus, generated specifically to exercise that workflow's cost-driving dimensions. There is no shared pre-curated corpus.

**Context for Claude Code:** You have the Pretia codebase (including the PDF processing pipeline with page-level classification, chunking, and embedding), `projection-engine-recommendation-addition-2.md` (PDF pipeline design, W17 architecture), `cross-cutting-robustness.md`, and the technical spec. This file provides the per-workflow document specifications and the profiling-vs-ground-truth differentiation.

**What you are building:** A generator module per workflow that produces tagged PDF files with controlled characteristics (page count, section structure, modality, content density). Each generated PDF carries metadata describing its structural properties.

---

## Part 1: Generation Pipeline

### Why Generate, Not Curate

Pre-curated corpora are uncontrolled — you can't design specific token counts, section structures, or content that answers specific queries. Generated corpora give full control over the cost-driving dimensions: document length, modality mix, section count, and content complexity. This enables designed drift between profiling and ground truth sets.

### The Two-Stage Pipeline

**Stage 1 — Content generation (LLM):** DeepSeek V4 Flash generates the text content: section titles, body text, table data, policy language. The LLM produces structured markdown or JSON describing the document's content and layout.

**Stage 2 — PDF rendering (Python):** Code converts the generated content into actual PDF files. This stage controls the visual presentation: fonts, page layout, table formatting, chart insertion, and — for W14/W15 — page rasterization to simulate scanned documents.

Never ask the LLM to produce a PDF directly. The LLM produces text; the code produces PDFs.

### Rendering Tools

Use **WeasyPrint** as the primary renderer (markdown/HTML → PDF). It handles:
- Multi-page documents with headers/footers
- Tables with borders and cell styling
- Page numbers
- Consistent typography

Use **reportlab** for:
- Programmatic table generation with fine layout control
- Embedding matplotlib-generated charts
- Page-level image insertion

Use **matplotlib** for:
- Bar charts, line charts, pie charts embedded in documents
- Simple diagrams and figures
- Data visualizations referenced in the text

Use **PIL/Pillow** for:
- Page rasterization (text page → image → embedded in PDF) to simulate scanned documents
- Adding noise, blur, skew, and rotation to simulate scan quality degradation

### Three Rendering Modes

**Mode 1 — Text-only:** Clean, extractable text. pdfplumber extracts it at $0 cost. No vision processing triggered. This is the baseline mode for all workflows.

**Mode 2 — Tables and charts:** Text with embedded tables (reportlab) and charts (matplotlib). The text remains extractable, but the charts and complex tables may trigger the "mixed" page classification (embedded_image_area > 40%) which routes to vision processing. Cost: moderate — vision only on chart-heavy pages.

**Mode 3 — Scanned:** Text pages are rasterized to images via PIL, then embedded in the PDF as full-page images. pdfplumber extracts no text. The page-level classifier routes these to full-page vision processing. Cost: high — every scanned page incurs vision token cost.

**v1 scope for Mode 3:** Scanned pages are generated **only for W14 and W15**, where vision token cost is a meaningful cost driver in the RAG pipeline. W16, W17, and W18 do not use scanned pages because:
- W16 (map-reduce): Haiku processes extracted text per section. Vision routing doesn't affect the map-reduce cost model.
- W17 (claims): Policy documents are text-based reference material. Scanned policies would add vision cost without testing a different cost pattern.
- W18 (long document): Uses DeepSeek V4, which doesn't have vision. Scanned pages would just fail extraction.

---

## Part 2: Profiling vs. Ground Truth PDF Differences

The same three drift dimensions from `directions-input-generation.md` apply to PDFs:

### Dimension 1: Length

Profiling PDFs are shorter. Ground truth PDFs are longer within each tier. The per-workflow tier tables below specify both ranges.

### Dimension 2: Structural Cleanliness

**Profiling PDFs:** Clean structure. Clear section headings, consistent formatting, logical flow, numbered sections. The splitting step (W16) and chunking pipeline find natural boundaries easily.

**Ground truth PDFs:** Messier structure. Some documents have:
- Inconsistent heading levels (H2 followed by H4, skipping H3)
- Missing section numbers
- Run-on sections without clear boundaries
- Tables that interrupt text flow mid-paragraph
- Footnotes and appendices that contain substantive content (not just references)
- Headers that don't match the content below them

This affects W16 directly (the splitter produces more variable section counts) and W14/W15 indirectly (chunking produces less clean chunks, affecting retrieval quality).

### Dimension 3: Modality Mix (W14/W15 Only)

**Profiling PDFs (W14/W15):** 80% text-only pages, 20% pages with tables/charts. 0% scanned pages.

**Ground truth PDFs (W14/W15):** 50% text-only pages, 30% pages with tables/charts, 20% scanned pages. The scanned pages trigger vision processing, adding a cost dimension absent from profiling.

**All other workflows (W16, W17, W18):** 100% text-only and tables/charts in both profiling and ground truth. No scanned pages.

---

## Part 3: Per-Workflow PDF Specifications

---

### W14 — Simple RAG Corpus

**Domain:** Health insurance policies and coverage documents.

**Corpus size:**
- Profiling: 15–20 PDFs
- Ground truth: 40–60 PDFs

The corpus is the document collection that RAG retrieves from. More documents = more potential retrieval matches = wider retrieval size variance.

**Insurance providers:** United Healthcare, Aetna, BlueCross BlueShield. Each provider gets 5–7 PDFs in profiling, 13–20 in ground truth.

**Document types (5 per provider):**

1. **Summary of Benefits and Coverage (SBC):** 8–15 pages. Structured: coverage categories, copays, deductibles, out-of-pocket maximums. Tabular format. One per provider per plan type (Gold, Silver, Bronze).
2. **Formulary / Drug Coverage List:** 5–10 pages. Table-heavy: drug names, tiers, prior authorization requirements, quantity limits.
3. **Provider Network Directory (excerpt):** 3–8 pages. Names, specialties, locations, accepting-new-patients flags. Tabular.
4. **Detailed Policy Document:** 20–40 pages. Dense text. Coverage criteria, exclusions, appeals process, definitions, regulatory compliance. This is the primary document for complex queries.
5. **Member Handbook / FAQ:** 10–20 pages. Mixed: text explanations, tables summarizing key numbers, occasional flowcharts (rendered as simple charts).

**Content requirements:**

Each provider's documents must contain meaningfully different values for key fields so that cross-provider queries produce different answers:

| Field | United Healthcare | Aetna | BlueCross BlueShield |
|---|---|---|---|
| In-network deductible (individual) | $1,500 | $2,000 | $1,200 |
| Out-of-pocket maximum | $6,500 | $7,500 | $5,800 |
| ER copay | $250 | $300 | $200 |
| MRI prior authorization | Required, 5-day turnaround | Required, 3-day turnaround | Not required for in-network |
| Mental health visits | 20/year, $40 copay | Unlimited, $50 copay | 30/year, $35 copay |
| Pre-existing condition waiting period | None (ACA compliant) | None | None |
| Out-of-network coverage | 60% after $3,000 deductible | 50% after $4,000 deductible | 70% after $2,500 deductible |
| Prescription tiers | 4 tiers | 5 tiers | 4 tiers |

These values must appear explicitly in the generated documents. The RAG queries from `directions-input-generation.md` target these fields — if the documents don't contain them, retrieval returns nothing useful.

**Page-level modality distribution:**

- Profiling PDFs: ~80% text pages, ~20% table/chart pages. No scanned pages.
- Ground truth PDFs: ~50% text pages, ~30% table/chart pages, ~20% scanned pages (rasterized from text, simulating older/faxed documents). The scanned pages trigger vision processing in the PDF pipeline.

**Structural descriptor per PDF:**
```json
{
  "pdf_id": "w14_prof_uhc_sbc_gold_001",
  "workflow": "W14",
  "profile": "profiling",
  "provider": "United Healthcare",
  "document_type": "sbc | formulary | network_directory | detailed_policy | member_handbook",
  "page_count": 12,
  "estimated_token_count": 8500,
  "text_pages": 10,
  "table_chart_pages": 2,
  "scanned_pages": 0,
  "section_count": 6,
  "key_fields_present": ["deductible", "oop_max", "er_copay", "mri_prior_auth"]
}
```

---

### W15 — Multi-Hop RAG Corpus

**Domain:** Same as W14 (health insurance). W15 uses the same corpus as W14 with additional documents designed for multi-hop retrieval.

**Corpus composition:**
- Base: W14's full corpus (reuse, don't regenerate)
- Additional cross-reference documents: 5–10 PDFs per distribution

**Additional document types for multi-hop:**

6. **Coverage Comparison Guide:** 10–15 pages. Compares coverage across plan types (Gold vs Silver vs Bronze) within a single provider. Forces multi-hop: the query asks "What's better for someone who needs frequent MRI scans?" → retrieval needs the comparison guide + the formulary + the detailed policy's prior auth section.
7. **Appeals and Grievances Handbook:** 8–12 pages. Detailed appeal procedures, timelines, documentation requirements. Separated from the main policy document so that "How do I appeal a denied MRI?" requires hopping from the denial conditions (in the detailed policy) to the appeal process (in this handbook).
8. **Provider-Specific Amendment / Rider:** 3–5 pages. Modifies base policy for specific conditions or circumstances. Creates intentional conflicts with the main policy document that the agent must reconcile.

**Cross-reference design principle:** Information that answers multi-hop queries must be deliberately spread across non-adjacent documents. If a query asks "What are all the scenarios where a claim would be denied, and what appeal options exist for each?", the answer requires:
- Denial scenarios (in the detailed policy document, Section 4)
- Appeal timelines (in the appeals handbook, Section 2)
- Exceptions for specific procedures (in the amendment/rider)

No single document contains the complete answer. This forces the sufficiency assessor to say "sufficient: false" and trigger additional retrieval hops.

**Modality distribution:** Same as W14. The additional documents follow the same profiling (80/20/0) and ground truth (50/30/20) modality splits.

---

### W16 — Map-Reduce Corpus

**Domain:** Corporate analysis — annual reports, regulatory filings, research papers, meeting transcripts.

**Corpus size:**
- Profiling: 50 PDFs (one per profiling run — each run processes one document)
- Ground truth: 500 PDFs (one per ground truth run)

Unlike W14/W15 where the corpus is a shared document collection queried by different inputs, W16's input IS the PDF. Each profiling/ground truth run processes a different document.

**Document types (5 categories):**

1. **Annual Reports:** 15–60 pages. Sections: letter to shareholders, financial highlights, business segments, risk factors, financial statements. Well-structured with clear headings.
2. **Regulatory Filings:** 20–80 pages. Dense, formal language. Sections: purpose, scope, methodology, findings, enforcement actions, appendices. Often has numbered sections and subsections.
3. **Research Papers:** 8–25 pages. Sections: abstract, introduction, methodology, results, discussion, conclusion, references. Most structured of the document types.
4. **Meeting Transcripts:** 10–40 pages. Least structured — speaker labels with dialogue, occasional agenda items, minimal headings. Tests the splitter's ability to find section boundaries in unstructured text.
5. **Technical Specifications:** 15–50 pages. Sections: overview, requirements, architecture, implementation, testing, appendices. Mix of text and tables.

**Tier-to-page-count mapping:**

| Tier | Profiling pages | GT pages | Expected sections (N) | Document types |
|---|---|---|---|---|
| Easy | 3–10 | 5–15 | 3–5 | Research papers, short specs |
| Medium | 12–30 | 18–45 | 6–12 | Annual reports, regulatory filings |
| Hard | 35–65 | 50–85 | 13–18 | Long annual reports, dense filings |
| Edge | 1–100 | 1–100 | 1–20 | 1-page memo (degenerate), 100-page transcript |
| Extreme (GT only) | — | 80–100 | 15–20 | Dense filing with poor structure |

**Section structure control:**

The section count N is the primary cost driver (N parallel Haiku calls). The generator must control N by:
- **Well-structured documents:** Clear headings (H1/H2) that the splitter will find. Predictable N.
- **Poorly-structured documents (ground truth):** Headings missing or inconsistent, topic shifts without markers, content that could be one section or three depending on interpretation. Variable N.

Profiling documents should be well-structured (predictable N). Ground truth should include ~30% poorly-structured documents (variable N) to test the splitter's robustness.

**Modality:** Text-only and tables. No scanned pages. Charts are optional (matplotlib-generated, embedded in annual reports and research papers).

---

### W17 — Claims Agent Policy Corpus

**Domain:** Health insurance claims processing. Three providers: United Healthcare, Aetna, Cigna.

**W17's corpus is special:** Unlike other workflows, W17's PDFs are reference material consulted during claims processing. The agent retrieves from the correct provider's policy to evaluate a claim. The corpus is small but each document is load-bearing — using the wrong provider's policy must produce a wrong decision.

**Corpus size:**
- 3 provider policy PDFs (one per provider), each 25–40 pages
- 10 clinical note PDFs (attached to specific claims, 1–3 pages each)
- 5 supporting documentation PDFs (itemized bills, prior authorization forms, 1–2 pages each)
- Same corpus for profiling and ground truth (the drift is in the claims, not the policies)

**Provider policy document structure:**

Each provider's policy PDF must include all of the following sections, with provider-specific values:

1. **Coverage Overview** (2–3 pages): Plan types, effective dates, member eligibility criteria.
2. **Covered Services** (5–8 pages): Detailed list of covered procedures with conditions, organized by service category (preventive, diagnostic, surgical, emergency, mental health, rehabilitation).
3. **Exclusions** (3–5 pages): What is NOT covered. Each provider has different exclusions.
4. **Prior Authorization Requirements** (3–5 pages): Which procedures require prior auth, documentation requirements, turnaround times, emergency exceptions.
5. **Cost Sharing** (2–3 pages): Deductibles, copays, coinsurance, out-of-pocket maximums. Tables.
6. **Claims Processing** (3–5 pages): How to file, timelines, what documentation is required for each claim type (standard, pre-approval, appeal).
7. **Appeals Process** (2–4 pages): How to appeal a denial, required documentation, timelines, levels of appeal.
8. **Definitions** (2–3 pages): Medical terminology, "medical necessity" definition, "experimental treatment" definition.

**Critical cross-provider differences (must be in the generated content):**

| Policy element | United Healthcare | Aetna | Cigna |
|---|---|---|---|
| MRI prior auth required? | Yes, all MRI | Yes, non-emergency only | No (in-network only) |
| Pre-existing condition exclusion | None (ACA) | None (ACA) | 6-month lookback for specific conditions |
| Appeal deadline | 180 days from denial | 90 days from denial | 120 days from denial |
| Medical necessity standard | "Generally accepted medical practice" | "Evidence-based, peer-reviewed" | "Clinically appropriate and effective" |
| Emergency room definition | "Prudent layperson" standard | "Prudent layperson" standard | "Reasonable person" standard |
| Experimental treatment exclusion | Excludes Phase I/II trials | Excludes all clinical trials | Excludes Phase I only |
| Maximum out-of-pocket | $7,500 individual | $8,200 individual | $6,800 individual |

These differences are what make W17's evaluation step cost-variable: a claim that's straightforward under United Healthcare (MRI covered, no prior auth issues) may require additional evaluation under Aetna (prior auth check needed) or be denied under Cigna (pre-existing condition lookback). The claims in `directions-input-generation.md` are designed to exercise these differences.

**Clinical note PDFs:**

Short (1–3 pages), free-form medical text. Include:
- Patient complaint, history of present illness
- Physical examination findings
- Assessment and plan
- Prior treatment history (relevant for appeals)
- Diagnosis codes (ICD-10) and procedure codes (CPT)

Generate 10 clinical notes covering:
- 3 straightforward cases (clear diagnosis, clear procedure, complete documentation)
- 3 cases with ambiguous documentation (vague diagnosis, missing prior treatment history)
- 2 cases with conflicting information (diagnosis suggests X, but procedure is for Y)
- 2 cases with incomplete documentation (missing sections, partial notes)

The clinical notes pair with specific claims in the input set. The structural descriptor in `directions-input-generation.md` for W17 references clinical note IDs.

**Modality:** All text. No scanned pages, no charts. These are reference documents that the pipeline processes via pdfplumber text extraction.

---

### W18 — Long-Document Corpus

**Domain:** Long-form analysis — annual reports, legal depositions, technical specifications, regulatory filings.

**Corpus size:**
- Profiling: 50 PDFs (one per run)
- Ground truth: 500 PDFs (one per run)

**Document types (4 categories):**

1. **Annual Reports:** 30–60 pages. Mix of narrative and financial data. Sections: executive summary, business overview, market analysis, financial statements, risk factors, outlook.
2. **Legal Depositions:** 40–100 pages. Mostly dialogue (Q&A format). Minimal structure — page after page of testimony. Tests the model's ability to extract findings from unstructured text.
3. **Technical Specifications:** 30–80 pages. Detailed requirements, architecture descriptions, API documentation, test plans. Structured with numbered sections.
4. **Regulatory Filings:** 50–100 pages. Dense, formal. Sections: background, findings of fact, legal analysis, orders, appendices. Heavy on cross-references.

**Tier-to-token-count mapping:**

| Tier | Profiling tokens | GT tokens | Page range |
|---|---|---|---|
| Easy | 30,000–40,000 | 30,000–50,000 | 30–45 pages |
| Medium | 40,000–60,000 | 50,000–75,000 | 45–70 pages |
| Hard | 60,000–80,000 | 70,000–95,000 | 70–90 pages |
| Edge | 80,000–100,000 | 85,000–100,000 | 90–100+ pages |
| Extreme (GT only) | — | 90,000–100,000 | 95–100+ pages |

**Content density control:**

The token-to-page ratio matters. A "dense" page (~1,200 tokens) contributes more to input cost than a "sparse" page (~500 tokens). Control this by:
- Dense pages: Long paragraphs, no white space, small margins. Regulatory filings and depositions.
- Sparse pages: Tables with white space, short paragraphs, headers taking up half the page. Annual reports with charts.
- Average: ~800 tokens/page.

Profiling documents should have consistent density. Ground truth documents should have variable density (some dense, some sparse, some mixed) to widen the token count variance within a tier.

**Modality:** Text-only. No scanned pages (DeepSeek V4 doesn't have vision). Tables are acceptable (rendered as text tables, not images). No charts (DeepSeek processes text only).

**Token count verification:** After rendering each PDF, extract text and count tokens using the DeepSeek tokenizer (or tiktoken cl100k_base as proxy). The actual token count must fall within the tier's specified range. Regenerate any PDF that falls outside range.

---

## Part 4: Content Generation Details

### Generation Model

**DeepSeek V4 Pro** for any document over ~15 pages. **DeepSeek V4 Flash** for short documents under ~15 pages. **Sonnet 4.6** only for the small number of precision-critical reference documents where internal consistency is load-bearing.

**Tier A — Sonnet 4.6 (precision-critical, small count):**
- W17 provider policy documents (3 documents, 25–40 pages each). These are load-bearing — specific values at specific locations, internally consistent cross-references, provider-specific differences that must drive different claim outcomes. Getting these wrong invalidates the entire W17 backtest. Cost: ~$1–2 total for 3 documents.
- W14/W15 detailed policy documents (5–10 documents, 20–40 pages each). RAG retrieves from these. If a document contradicts itself ("MRI requires prior auth" in Section 3, "MRI covered without preconditions" in Section 7), retrieval returns garbage. Small count, high precision requirement.

**Tier B — DeepSeek V4 Pro (long documents):**
- W16 corporate documents over ~15 pages (medium/hard/extreme tiers)
- W18 all documents (30–100 pages, always long)
- W4 compliance documents over ~10 pages (from `directions-input-generation.md`)
- Any document exceeding ~15 pages that isn't in Tier A

For documents under ~60 pages, Pro can generate the full content in a single call or two large calls (split at natural section boundaries). For very long documents (60–100+ pages), use sectional generation: Pro generates the full outline (section titles, key data points per section, target page count per section), then Pro generates each section independently (5–10 pages per call). The rendering code assembles sections into the final PDF. Pro handles both the outline and the sections — Flash's coherence degrades on substantive content at this scale, producing hallucinated cross-references, inconsistent financial data, and drifting topic focus.

**Tier C — DeepSeek V4 Flash (short, structured):**
- W14/W15 SBCs, formularies, network directories (5–15 pages, mostly tabular)
- W17 clinical notes (1–3 pages)
- W17 supporting documentation (1–2 pages)
- W16 short documents (easy tier, 3–10 pages)
- Any document under ~15 pages

Flash handles these directly — they're short, structured, and don't require long-range coherence.

### Content Generation Prompt Structure

Each content generation call should include:

1. **Document type and domain** — "Generate a 25-page health insurance policy document for United Healthcare."
2. **Required sections** — list of section titles with target page counts.
3. **Required values** — specific numbers (deductibles, copays, etc.) that must appear. Provide these as a table of key-value pairs.
4. **Structural constraints** — heading format, table locations, whether to include an appendix.
5. **Token target** — "Target approximately 18,000 tokens of content."
6. **Style instructions** — formal/technical language level, whether to use legal jargon, whether tables should have full borders or be minimal.

For **Tier B sectional generation** (documents over ~60 pages), the outline call includes items 1–6 above and produces a structured plan. Each section call receives: the outline (for context on where this section fits), the section's specific requirements (from the outline), and the key values that must appear in that section. The section call does NOT receive the full text of other sections — only the outline. Both the outline and section calls use DeepSeek V4 Pro.

The prompt should NOT include:
- Instructions about PDF rendering (that's the code's job)
- References to Pretia or backtesting
- Instructions to make the content "diverse" (diversity comes from varying the input parameters across calls, not from a single call)

### Substantive Content Requirement

Generated content must be factually coherent and internally consistent. Specific requirements:

- **Financial data:** Numbers must add up. If revenue is $10M and costs are $7M, profit must be $3M. If a table shows quarterly revenue, the total row must equal the sum.
- **Policy language:** Rules must be consistent within a document. If Section 3 says "MRI requires prior authorization," Section 7 must not say "MRI is covered without preconditions" (unless the document is deliberately designed to contain contradictions for W15 multi-hop testing).
- **Legal text:** Cross-references must be valid. "As defined in Section 2.3" must reference an actual Section 2.3 that contains a definition.
- **Medical content:** ICD-10 and CPT codes must use real code formats (M54.5, 72148) even though the specific mappings can be fictional. Don't use obviously fake codes like "XXX-000."

This matters because RAG retrieval matches on content semantics. Lorem ipsum or incoherent content produces retrieval results that don't test the generation step meaningfully.

---

## Part 5: Rendering Specifications

### Text-Only Pages

```
Page layout:
  - Letter size (8.5" × 11")
  - Margins: 1" all sides
  - Font: 11pt serif (Times or similar) for body, 13pt sans-serif (Helvetica) for headings
  - Line spacing: 1.15
  - Paragraph spacing: 6pt after
  - Page numbers: bottom center

Target density: ~800 tokens/page at standard settings.
```

### Table Pages

Tables should be generated programmatically (reportlab) from the structured data in the LLM output. Vary table styles:
- Some with full borders (formal documents)
- Some with header-only borders (modern reports)
- Some spanning full page width, others inline with text
- Column counts: 3–8

Table pages trigger the "mixed" page classification when the table occupies >40% of the page area. This is intentional for W14/W15 ground truth (adds moderate vision cost).

### Chart Pages

Generate simple charts using matplotlib:
- Bar charts for financial data (quarterly revenue, expense breakdowns)
- Line charts for trends (year-over-year metrics)
- Pie charts for distributions (market share, budget allocation)

Save as PNG, embed in PDF. Chart pages always trigger image detection in the page classifier.

Charts are used in W14/W15 (ground truth) and optionally in W16 (annual reports). Not used in W17 or W18.

### Scanned Pages (W14/W15 Ground Truth Only)

To simulate scanned documents:

1. Render the text page normally (WeasyPrint → PDF → extract page as image via pdf2image).
2. Apply degradation via PIL:
   - Resolution: 150–200 DPI (not 300 — scanned docs are often lower quality)
   - Gaussian blur: sigma 0.3–0.8
   - Rotation: ±0.5–2 degrees
   - Noise: Gaussian noise, sigma 3–8
   - Contrast variation: 0.8–1.2×
3. Embed the degraded image back into a new PDF page.

Scanned pages cost ~3–5× more than text pages in vision token processing. Control the scan degradation level to produce realistic (not extreme) quality variation.

**Distribution within ground truth PDFs:**
- Not every ground truth PDF has scanned pages. ~40% of ground truth PDFs contain scanned pages.
- Within those PDFs, ~30–50% of pages are scanned (not all pages — simulate a partially-scanned document where some pages are digital and some are faxed/scanned).
- The specific pages that get scanned should be randomly selected but tagged in the metadata.

---

## Part 6: Quality Checks

### Per-PDF Checks (Automated)

1. **PDF validity:** Open with pdfplumber without exception. All pages accessible.
2. **Page count:** Matches the target tier range.
3. **Token count:** Extract text (pdfplumber for text pages, skip scanned pages), count tokens. Must fall within the tier's specified range.
4. **Text extraction success:** For text-only and table pages, pdfplumber must extract >90% of the expected text. If extraction fails on >10% of text pages, the rendering is broken.
5. **Scanned page detection:** For PDFs with scanned pages, verify pdfplumber extracts <10 characters from those pages (confirming they're properly rasterized).
6. **Table integrity:** For documents with tables, verify at least one table is detected by pdfplumber's `extract_tables()`.
7. **Metadata consistency:** The structural descriptor matches the actual PDF (page count, section count, modality distribution).

### Per-Corpus Checks

8. **Key field coverage (W14/W15/W17):** Every value in the cross-provider tables above must appear in at least one document in the corpus. Run a text search across all PDFs to verify.
9. **Document type distribution:** Each document type should comprise 15–25% of the corpus. No single type >30%.
10. **Length distribution:** Token counts across the corpus span at least 70% of the combined tier ranges (same non-uniformity requirement as input generation).
11. **W17 provider coverage:** Each provider (UHC, Aetna, Cigna) has exactly one policy PDF. The three policies contain all the specified cross-provider differences.

### Cross-Distribution Checks (Profiling vs. Ground Truth)

12. **Length drift:** Mean token count of ground truth PDFs is 1.3–2× profiling mean (by tier).
13. **Structural drift:** ≥30% of ground truth PDFs have "poor" structure quality (missing/inconsistent headings). ≤10% of profiling PDFs have poor structure.
14. **Modality drift (W14/W15 only):** Profiling modality is 80/20/0 (text/table+chart/scanned). Ground truth modality is approximately 50/30/20. Verify by counting page types across all PDFs.

---

## Part 7: Coordination with Input Generation

PDF generation runs **before** input generation for W14/W15 (because query generation needs to know what's in the corpus).

### W14/W15 Query-Corpus Alignment

After generating the W14/W15 corpus, extract and index the content. Then pass the content index to the W14/W15 query generators in `directions-input-generation.md`. The query generator uses this index to:
- Create easy queries that match 1–2 specific chunks
- Create medium queries that match 3–5 chunks across sections
- Create hard queries that require synthesizing across documents
- Create edge queries that match nothing (testing empty retrieval)

This means the PDF generator must output a **content manifest** alongside the PDFs:

```json
{
  "corpus_id": "w14_profiling_v1",
  "documents": [
    {
      "pdf_id": "w14_prof_uhc_sbc_gold_001",
      "provider": "United Healthcare",
      "document_type": "sbc",
      "sections": [
        {
          "title": "Deductibles and Copays",
          "page_range": [3, 5],
          "key_facts": [
            "Individual in-network deductible: $1,500",
            "ER copay: $250",
            "Out-of-pocket maximum: $6,500"
          ]
        }
      ]
    }
  ]
}
```

The query generator reads this manifest to create queries targeting specific facts at specific locations, ensuring the queries are answerable from the corpus.

### W17 Claim-Policy Alignment

The W17 claims in `directions-input-generation.md` reference specific providers. The claims evaluation step retrieves from the matching provider's policy PDF. The claim generator must know what policy values each provider has (from the cross-provider table above) to create claims that exercise the differences:
- A claim for an MRI under Cigna should be straightforward (no prior auth for in-network).
- The same MRI claim under United Healthcare requires prior authorization evaluation.
- An appeal under Aetna has a 90-day deadline; the same appeal under Cigna has 120 days.

### W16/W18 — No Coordination Needed

For W16 and W18, the input IS the PDF. No separate query generation step. The PDF generator produces the complete input.

---

## File Manifest

```
pdfs/
  generators/
    w14_w15_insurance_corpus.py    # Shared generator for W14/W15 corpus
    w15_crossref_supplement.py     # Additional multi-hop documents for W15
    w16_corporate_documents.py     # Per-document generator for W16
    w17_policy_documents.py        # Provider policy generator
    w17_clinical_notes.py          # Clinical note generator
    w18_long_documents.py          # Long document generator
    
    rendering/
      text_renderer.py             # WeasyPrint-based text-to-PDF
      table_renderer.py            # reportlab table generation
      chart_renderer.py            # matplotlib chart generation
      scan_simulator.py            # PIL-based page rasterization
      pdf_assembler.py             # Combines rendered elements into final PDF
    
  generated/
    profiling/
      w14_w15_corpus/              # Shared corpus for W14/W15 profiling
        pdfs/                      # The PDF files
        manifest.json              # Content manifest for query generation
      w16/                         # One PDF per profiling run
      w17/
        policies/                  # 3 provider policy PDFs
        clinical_notes/            # 10 clinical note PDFs
        supporting_docs/           # 5 supporting documentation PDFs
      w18/                         # One PDF per profiling run
    ground_truth/
      w14_w15_corpus/              # Larger corpus for W14/W15 ground truth
        pdfs/
        manifest.json
      w15_supplement/              # Additional multi-hop documents
        pdfs/
        manifest.json
      w16/                         # One PDF per ground truth run
      w17/                         # Same policies, different clinical notes optional
      w18/                         # One PDF per ground truth run
      
  validation/
    verify_pdfs.py                 # Runs all per-PDF and per-corpus checks
    verify_content_coverage.py     # Checks key field coverage for W14/W15/W17
```

Each generator must be runnable standalone:
```
python w14_w15_insurance_corpus.py --profile profiling --seed 42 --output-dir generated/profiling/w14_w15_corpus/
python w16_corporate_documents.py --profile ground_truth --n 500 --seed 42 --output-dir generated/ground_truth/w16/
python w17_policy_documents.py --seed 42 --output-dir generated/profiling/w17/policies/
```

The rendering modules are shared utilities imported by all generators. They should not contain workflow-specific logic.
