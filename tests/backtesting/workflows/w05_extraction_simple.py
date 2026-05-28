"""Data Extraction (Simple) -- 3-step linear workflow, no loops. Haiku + Sonnet.

Steps:
    parse_document  (Haiku 4.5)  -> identify doc type, extract structure
    extract_fields  (Sonnet 4.6) -> extract structured JSON
    validate        (Haiku 4.5)  -> validate extracted JSON
"""

from __future__ import annotations

try:
    from typing import TypedDict

    from langchain_anthropic import ChatAnthropic
    from langgraph.graph import END, StateGraph

    from tests.backtesting.workflows._shared import (
        get_anthropic_model,
    )

    _EXTRACTION_SYSTEM = (
        "You are a document data extraction specialist. Extract structured data "
        "from the provided text into the specified JSON format. Be precise. When a "
        "field is ambiguous, include your best guess and set confidence to 'low'. "
        "Always return valid JSON."
    )

    # ── State ────────────────────────────────────────────────────────────

    class ExtractionSimpleState(TypedDict):
        input: str
        doc_type: str
        structure: str
        extracted_json: str
        validated_json: str

    # ── Node functions ───────────────────────────────────────────────────

    def parse_document(state: ExtractionSimpleState) -> dict:
        """Identify document type and extract its high-level structure."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=512,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a document analysis specialist. Examine the provided "
                        "document and determine:\n"
                        "1. Document type (invoice, contract, receipt, letter, report, etc.)\n"
                        "2. Key sections and their locations\n"
                        "3. The fields that should be extracted\n\n"
                        "Respond in this format:\n"
                        "TYPE: <document_type>\n"
                        "SECTIONS: <comma-separated list>\n"
                        "FIELDS: <comma-separated list of extractable fields>"
                    ),
                },
                {"role": "user", "content": f"Document:\n\n{state['input']}"},
            ]
        )
        content = result.content if isinstance(result.content, str) else str(result.content)
        # Extract doc type from the response (first line after TYPE:)
        doc_type = "unknown"
        for line in content.splitlines():
            if line.strip().upper().startswith("TYPE:"):
                doc_type = line.split(":", 1)[1].strip().lower()
                break
        return {"doc_type": doc_type, "structure": content}

    def extract_fields(state: ExtractionSimpleState) -> dict:
        """Extract structured fields into JSON."""
        llm = ChatAnthropic(
            model=get_anthropic_model("sonnet"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        f"{_EXTRACTION_SYSTEM}\n\n"
                        "Based on the document structure analysis, extract all identified "
                        "fields into a JSON object. Include a 'confidence' field for each "
                        "extracted value ('high', 'medium', or 'low'). Wrap the entire "
                        "response in a JSON code block."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document type: {state['doc_type']}\n\n"
                        f"Structure analysis:\n{state['structure']}\n\n"
                        f"Original document:\n{state['input']}"
                    ),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"extracted_json": text}

    def validate(state: ExtractionSimpleState) -> dict:
        """Validate the extracted JSON for completeness and correctness."""
        llm = ChatAnthropic(
            model=get_anthropic_model("haiku"),
            max_tokens=1024,
        )
        result = llm.invoke(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a data validation specialist. Review the extracted JSON "
                        "against the original document. Check for:\n"
                        "1. Missing fields that should have been extracted\n"
                        "2. Incorrect values (compare against source)\n"
                        "3. Formatting issues (dates, currencies, numbers)\n"
                        "4. Valid JSON structure\n\n"
                        "Return the corrected JSON with a 'validation_status' field set to "
                        "'valid' or 'corrected'. If corrected, include a "
                        "'corrections_made' array describing what was fixed. "
                        "Always return valid JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Document type: {state['doc_type']}\n\n"
                        f"Original document:\n{state['input']}\n\n"
                        f"Extracted JSON:\n{state['extracted_json']}"
                    ),
                },
            ]
        )
        text = result.content if isinstance(result.content, str) else str(result.content)
        return {"validated_json": text}

    # ── Graph ────────────────────────────────────────────────────────────

    builder = StateGraph(ExtractionSimpleState)
    builder.add_node("parse_document", parse_document)
    builder.add_node("extract_fields", extract_fields)
    builder.add_node("validate", validate)

    builder.set_entry_point("parse_document")
    builder.add_edge("parse_document", "extract_fields")
    builder.add_edge("extract_fields", "validate")
    builder.add_edge("validate", END)

    graph = builder.compile()

except ImportError:
    graph = None
