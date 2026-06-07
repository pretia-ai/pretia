"""Async LLM content generation helpers for PDF generation pipeline.

Reuses provider resolution patterns from agentcost/inputs/generator.py.
Two-stage pipeline: LLM generates structured text content, Python renders to PDF.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
from datetime import UTC, datetime
from typing import Any

logger = logging.getLogger(__name__)

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"


def _try_import(name: str) -> Any | None:
    try:
        return __import__(name)
    except ImportError:
        return None


def _resolve_api_key(model: str, api_key: str | None = None) -> tuple[str, str, str]:
    """Resolve provider, API key, and base URL for a model.

    Returns (provider, resolved_key, base_url).
    """
    if model.startswith("claude-"):
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY required for Sonnet generation")
        return "anthropic", key, ""
    if model.startswith("deepseek"):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise ValueError("DEEPSEEK_API_KEY required for DeepSeek generation")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", _DEEPSEEK_BASE_URL)
        return "deepseek", key, base_url
    raise ValueError(f"Unsupported generation model: {model}")


_API_CONCURRENCY_LIMIT = 400


async def generate_content(
    system_prompt: str,
    user_prompt: str,
    model: str = "deepseek-v4-pro",
    max_tokens: int = 8192,
    api_key: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Generate text content via LLM for PDF rendering.

    Returns the raw text response from the model. Retries once if the
    response is empty or too short (<50 chars).
    """
    provider, key, base_url = _resolve_api_key(model, api_key)

    for attempt in range(2):
        if provider == "anthropic":
            result = await _call_anthropic(
                key,
                model,
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
            )
        else:
            result = await _call_deepseek(
                key,
                model,
                base_url,
                system_prompt,
                user_prompt,
                max_tokens,
                temperature,
            )

        if result and len(result.strip()) >= 50:
            return result

        logger.warning(
            "LLM returned empty/short response (attempt %d, got %d chars), retrying",
            attempt + 1,
            len(result.strip()) if result else 0,
        )

    return result or ""


async def _call_anthropic(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    sdk = _try_import("anthropic")
    if sdk is None:
        msg = "anthropic package required for Sonnet generation: pip install anthropic"
        raise ImportError(msg)
    client = sdk.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return response.content[0].text


async def _call_deepseek(
    api_key: str,
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
) -> str:
    sdk = _try_import("openai")
    if sdk is None:
        raise ImportError("openai package required for DeepSeek generation: pip install openai")
    client = sdk.AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=300.0)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


async def generate_document_content(
    document_type: str,
    domain: str,
    sections: list[dict[str, Any]],
    target_pages: int,
    model: str = "deepseek-v4-pro",
    provider_name: str | None = None,
    key_values: dict[str, str] | None = None,
    structure_quality: str = "well_structured",
    api_key: str | None = None,
) -> str:
    """Generate structured document content for PDF rendering.

    Builds a prompt from the document specification and returns markdown content.
    For documents >60 pages, uses sectional generation (outline then per-section).
    """
    target_tokens = target_pages * 800

    if target_pages > 60:
        return await _generate_sectional(
            document_type=document_type,
            domain=domain,
            sections=sections,
            target_pages=target_pages,
            model=model,
            provider_name=provider_name,
            key_values=key_values,
            structure_quality=structure_quality,
            api_key=api_key,
        )

    system_prompt = _build_system_prompt(document_type, domain, provider_name)
    user_prompt = _build_user_prompt(
        sections=sections,
        target_pages=target_pages,
        target_tokens=target_tokens,
        key_values=key_values,
        structure_quality=structure_quality,
    )

    logger.info(
        "Generating %s (%s, ~%d pages, model=%s)",
        document_type,
        provider_name or domain,
        target_pages,
        model,
    )
    return await generate_content(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        model=model,
        max_tokens=min(target_tokens * 2, 16384),
        api_key=api_key,
    )


async def _generate_sectional(
    document_type: str,
    domain: str,
    sections: list[dict[str, Any]],
    target_pages: int,
    model: str,
    provider_name: str | None,
    key_values: dict[str, str] | None,
    structure_quality: str,
    api_key: str | None,
) -> str:
    """Generate long documents (>60 pages) section by section.

    Stage 1: Generate outline with section titles and key data points.
    Stage 2: Generate each section independently (5-10 pages per call).
    """
    system_prompt = _build_system_prompt(document_type, domain, provider_name)

    outline_prompt = (
        f"Create a detailed outline for a {target_pages}-page {document_type} document.\n\n"
        f"The document has {len(sections)} sections:\n"
    )
    for s in sections:
        outline_prompt += f"- {s['title']} ({s.get('target_pages', 5)} pages)\n"
    outline_prompt += (
        "\nFor each section, provide:\n"
        "1. Section title\n"
        "2. Key data points and facts to include\n"
        "3. Target page count\n"
        "4. Content structure (paragraphs, tables, lists)\n\n"
        "Output as a structured outline in markdown."
    )
    if key_values:
        outline_prompt += "\n\nRequired values that MUST appear:\n"
        for k, v in key_values.items():
            outline_prompt += f"- {k}: {v}\n"

    logger.info("Generating outline for %d-page %s", target_pages, document_type)
    outline = await generate_content(
        system_prompt=system_prompt,
        user_prompt=outline_prompt,
        model=model,
        max_tokens=4096,
        api_key=api_key,
    )

    section_texts: list[str] = []
    for i, section in enumerate(sections):
        section_pages = section.get("target_pages", 5)
        section_tokens = section_pages * 800

        section_kv = {}
        if key_values:
            for k, _v in key_values.items():
                if k in section.get("key_values", {}):
                    section_kv[k] = section["key_values"][k]

        section_prompt = (
            f"Write section {i + 1} of a {document_type} document.\n\n"
            f"Section title: {section['title']}\n"
            f"Target length: approximately {section_pages} pages "
            f"(~{section_tokens} tokens)\n\n"
            f"Document outline for context:\n{outline}\n\n"
            f"Write ONLY this section's content in markdown. "
            f"Start with the section heading."
        )
        if section_kv:
            section_prompt += "\n\nRequired values for this section:\n"
            for k, v in section_kv.items():
                section_prompt += f"- {k}: {v}\n"

        if structure_quality == "poorly_structured":
            section_prompt += (
                "\n\nMake this section less formally structured: "
                "skip some heading levels, use inconsistent formatting, "
                "let some paragraphs run long without clear breaks."
            )

        logger.info("Generating section %d/%d: %s", i + 1, len(sections), section["title"])
        text = await generate_content(
            system_prompt=system_prompt,
            user_prompt=section_prompt,
            model=model,
            max_tokens=min(section_tokens * 2, 8192),
            api_key=api_key,
        )
        section_texts.append(text)

    return "\n\n".join(section_texts)


def _build_system_prompt(
    document_type: str,
    domain: str,
    provider_name: str | None,
) -> str:
    """Build the system prompt for document content generation."""
    prompt = (
        f"You are a professional document writer specializing in {domain} documents. "
        f"Generate realistic, substantively correct content for a {document_type} document."
    )
    if provider_name:
        prompt += f" This document is for {provider_name}."
    prompt += (
        "\n\nRequirements:\n"
        "- Content must be factually coherent and internally consistent.\n"
        "- Financial data must add up (totals = sum of parts).\n"
        "- Cross-references must point to real sections within the document.\n"
        "- Use real formats for codes (ICD-10, CPT) even if specific mappings are fictional.\n"
        "- Output clean markdown with proper heading hierarchy.\n"
        "- Do NOT include meta-commentary about the document generation process."
    )
    return prompt


def _build_user_prompt(
    sections: list[dict[str, Any]],
    target_pages: int,
    target_tokens: int,
    key_values: dict[str, str] | None,
    structure_quality: str,
) -> str:
    """Build the user prompt for document content generation."""
    prompt = (
        f"Generate a {target_pages}-page document "
        f"(approximately {target_tokens} tokens of content).\n\n"
        "Document structure:\n"
    )
    for s in sections:
        pages = s.get("target_pages", 3)
        prompt += f"- {s['title']} (~{pages} pages"
        if s.get("content_type") == "table_heavy":
            prompt += ", include data tables"
        elif s.get("content_type") == "mixed":
            prompt += ", mix of text and tables"
        prompt += ")\n"

    if key_values:
        prompt += (
            "\n**CRITICAL — Required values that MUST appear VERBATIM "
            "in the document text. Copy each value exactly as shown. "
            "Do not paraphrase, round, or omit any of them:**\n"
        )
        for k, v in key_values.items():
            prompt += f"- {k}: {v}\n"
        prompt += (
            "\nEvery value listed above must appear word-for-word in the "
            "generated content. This is non-negotiable.\n"
        )

    if structure_quality == "poorly_structured":
        prompt += (
            "\nStructural quality: intentionally messy. "
            "Skip some heading levels (e.g., H2 followed by H4), "
            "use inconsistent formatting, omit some section numbers, "
            "let paragraphs run long, and include footnotes with substantive content. "
            "Do NOT make it unreadable — just less formally structured than typical."
        )
    elif structure_quality == "well_structured":
        prompt += (
            "\nStructural quality: clean and well-organized. "
            "Use consistent heading hierarchy, numbered sections, "
            "clear paragraph breaks, and logical flow."
        )

    prompt += "\n\nOutput the full document content in markdown."
    return prompt


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base as a proxy."""
    try:
        import tiktoken

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except ImportError:
        return len(text) // 4


def compute_generator_hash(generator_path: str) -> str:
    """Compute SHA-256 hash of a generator script for corpus versioning."""
    with open(generator_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def corpus_spec(
    generator_path: str,
    seed: int,
    n_inputs: int,
    tokenizer: str = "cl100k_base",
) -> dict[str, Any]:
    """Build a corpus specification dict for reproducibility tracking."""
    return {
        "generator_version": f"sha256:{compute_generator_hash(generator_path)}",
        "seed": seed,
        "n_inputs": n_inputs,
        "tokenizer": tokenizer,
        "generated_at": datetime.now(UTC).isoformat(),
    }


_DEFAULT_CONCURRENCY = 200


async def run_concurrent(
    tasks: list,
    concurrency: int = _DEFAULT_CONCURRENCY,
) -> list:
    """Run async tasks with bounded concurrency.

    *tasks* is a list of coroutines. Returns results in the same order.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list = [None] * len(tasks)

    async def _run(idx: int, coro: Any) -> None:
        async with semaphore:
            results[idx] = await coro

    await asyncio.gather(*[_run(i, t) for i, t in enumerate(tasks)])
    return results


def generate_content_sync(
    system_prompt: str,
    user_prompt: str,
    model: str = "deepseek-v4-pro",
    max_tokens: int = 8192,
    api_key: str | None = None,
    temperature: float = 0.7,
) -> str:
    """Synchronous wrapper around generate_content."""
    return asyncio.run(
        generate_content(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            api_key=api_key,
            temperature=temperature,
        )
    )
