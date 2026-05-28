"""Data Extraction (Complex) -- 5-step workflow with validation loop (1-5 iterations).

Steps:
    parse_document    (Haiku 4.5)  -> identify doc type
    extract_fields    (Sonnet 4.6) -> extract into JSON
    cross_reference   (Sonnet 4.6) -> cross-reference against canned records
    resolve_conflicts (Opus 4.7)   -> analyze conflicts; loop back if corrections needed
    format_output     (Haiku 4.5)  -> format final JSON with confidence scores
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        CANNED_DOCUMENTS,
        get_anthropic_model,
    )

    # ── State ────────────────────────────────────────────────────────────

    class ExtractionComplexState(TypedDict):
        input: str
        doc_type: str
        structure: str
        extracted_json: str
        conflicts: str
        resolution: str
        final_json: str
        iteration_count: int

    # ── Node functions ───────────────────────────────────────────────────

    def parse_document(state: ExtractionComplexState) -> dict:
        """Identify the document type from the input text."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=128,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a document classifier. Analyze the provided document and "
                        "identify its type (invoice, contract, report, letter, etc.) and "
                        "describe its overall structure (sections, tables, key fields). "
                        "Respond with a short JSON object: "
                        '{"doc_type": "...", "structure": "..."}'
                    ),
                },
                {"role": "user", "content": state["input"]},
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        # Extract doc_type heuristically from response
        doc_type = "unknown"
        for dt in ("invoice", "contract", "report", "letter"):
            if dt in raw.lower():
                doc_type = dt
                break
        return {"doc_type": doc_type, "structure": raw, "iteration_count": 0}

    def extract_fields(state: ExtractionComplexState) -> dict:
        """Extract structured fields from the document into JSON."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        iteration = state.get("iteration_count", 0) + 1
        prior_resolution = state.get("resolution", "")
        extra_context = ""
        if prior_resolution:
            extra_context = (
                f"\n\nPrevious extraction had conflicts. Apply these corrections:\n"
                f"{prior_resolution}"
            )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a data extraction specialist. Extract all structured fields "
                        "from the document into a JSON object. Include dates, amounts, names, "
                        "addresses, line items, and any other relevant fields. Be precise with "
                        "numbers and formatting."
                        f"{extra_context}"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document type: {state.get('doc_type', 'unknown')}\n"
                        f"Structure analysis: {state.get('structure', '')}\n\n"
                        f"Document:\n{state['input']}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"extracted_json": raw, "iteration_count": iteration}

    def cross_reference(state: ExtractionComplexState) -> dict:
        """Cross-reference extracted data against canned database records."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=512,
        )
        # Build a canned reference database from CANNED_DOCUMENTS
        reference_db = "\n---\n".join(
            f"[{key}]: {value}" for key, value in CANNED_DOCUMENTS.items()
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a data validation specialist. Compare the extracted JSON "
                        "against the reference database records below. Identify any "
                        "discrepancies, missing fields, or conflicting values. List each "
                        "conflict with the field name, extracted value, and reference value."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extracted JSON:\n{state.get('extracted_json', '')}\n\n"
                        f"Reference Database:\n{reference_db}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"conflicts": raw}

    def resolve_conflicts(state: ExtractionComplexState) -> dict:
        """Analyze conflicts and decide corrections. May trigger re-extraction."""
        llm = ChatAnthropic(
            model=get_anthropic_model("opus"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an expert data reconciliation analyst. Review the conflicts "
                        "found during cross-referencing. For each conflict, decide the correct "
                        "value based on the source document and reference data. If corrections "
                        "are needed, clearly state them. End your response with either "
                        "'NEEDS_CORRECTION: true' if the extracted data should be re-extracted "
                        "with corrections, or 'NEEDS_CORRECTION: false' if the data is "
                        "acceptable."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Extracted JSON:\n{state.get('extracted_json', '')}\n\n"
                        f"Conflicts found:\n{state.get('conflicts', '')}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"resolution": raw}

    def format_output(state: ExtractionComplexState) -> dict:
        """Format the final validated JSON with confidence scores."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a data formatting specialist. Take the extracted and "
                        "validated JSON data and produce a final clean JSON output. Add a "
                        "'confidence' field (0.0-1.0) for each extracted value based on "
                        "how certain the extraction is. Include a top-level "
                        "'extraction_metadata' object with doc_type, total_fields, "
                        "avg_confidence, and iteration_count."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document type: {state.get('doc_type', 'unknown')}\n"
                        f"Extracted data:\n{state.get('extracted_json', '')}\n"
                        f"Resolution notes:\n{state.get('resolution', '')}\n"
                        f"Iterations: {state.get('iteration_count', 1)}"
                    ),
                },
            ]
        )
        raw = result.content if isinstance(result.content, str) else str(result.content)
        return {"final_json": raw}

    # ── Routing ──────────────────────────────────────────────────────────

    def should_re_extract(state: ExtractionComplexState) -> str:
        """Decide whether to loop back to extract_fields or proceed to format_output."""
        iteration = state.get("iteration_count", 0)
        if iteration >= 5:
            return "format_output"
        resolution = state.get("resolution", "")
        normalized = resolution.lower().replace(" ", "")
        if "needs_correction:true" in normalized:
            return "extract_fields"
        return "format_output"

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(ExtractionComplexState)
    builder.add_node("parse_document", parse_document)
    builder.add_node("extract_fields", extract_fields)
    builder.add_node("cross_reference", cross_reference)
    builder.add_node("resolve_conflicts", resolve_conflicts)
    builder.add_node("format_output", format_output)

    builder.set_entry_point("parse_document")
    builder.add_edge("parse_document", "extract_fields")
    builder.add_edge("extract_fields", "cross_reference")
    builder.add_edge("cross_reference", "resolve_conflicts")
    builder.add_conditional_edges(
        "resolve_conflicts",
        should_re_extract,
        {"extract_fields": "extract_fields", "format_output": "format_output"},
    )
    builder.add_edge("format_output", END)

    graph = builder.compile()

except ImportError:
    graph = None
