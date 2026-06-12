#!/usr/bin/env python
"""Build RAG corpus embeddings.json from generated PDFs.

Extracts text from PDFs, chunks it, embeds via OpenAI, and saves as
the JSON manifest format expected by bt_agents/harness/retrieval_sim.py.

Usage::

    python scripts/build_rag_corpus.py --pdf-dir pdfs/generated/profiling/w14_w15_corpus \\
        --output pdfs/w14_corpus/embeddings.json
    python scripts/build_rag_corpus.py --all
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)

_CHUNK_SIZE = 512
_CHUNK_OVERLAP = 64


def _load_dotenv() -> None:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value


def _extract_text_from_pdf(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract text from a PDF, returning pages with text content."""
    pages = []
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": page_num, "text": text.strip()})
    except ImportError:
        logger.warning("pdfplumber not installed — using fallback text extraction")
        pages.append({"page": 1, "text": f"Document: {pdf_path.name}"})
    except Exception as exc:
        logger.warning("Failed to extract text from %s: %s", pdf_path, exc)
    return pages


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping chunks by character count (~4 chars/token)."""
    char_size = chunk_size * 4
    char_overlap = overlap * 4
    chunks = []
    start = 0
    while start < len(text):
        end = start + char_size
        chunks.append(text[start:end])
        start = end - char_overlap
    return [c for c in chunks if c.strip()]


async def _embed_chunks(
    texts: list[str], model: str = "text-embedding-3-small", dry_run: bool = False,
) -> list[list[float]]:
    """Embed a list of texts via OpenAI embeddings API."""
    if dry_run:
        import math
        import random

        rng = random.Random(42)
        dim = 1536
        embeddings = []
        for _ in texts:
            vec = [rng.gauss(0.0, 1.0) for _ in range(dim)]
            norm = math.sqrt(sum(x * x for x in vec))
            embeddings.append([x / norm for x in vec] if norm > 0 else vec)
        return embeddings

    from litellm import aembedding

    embeddings = []
    batch_size = 100
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = await aembedding(model=f"openai/{model}", input=batch)
        for item in resp.data:
            embeddings.append(item["embedding"])
        logger.info("Embedded batch %d-%d of %d", i, i + len(batch), len(texts))

    return embeddings


async def build_corpus(
    pdf_dir: Path,
    output_path: Path,
    dry_run: bool = False,
) -> Path:
    """Build embeddings.json from a directory of PDFs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pdf_files = sorted(pdf_dir.rglob("*.pdf"))
    if not pdf_files:
        click.echo(f"No PDFs found in {pdf_dir}", err=True)
        sys.exit(1)

    click.echo(f"Processing {len(pdf_files)} PDFs from {pdf_dir}...")

    all_chunks: list[dict[str, Any]] = []
    chunk_texts: list[str] = []

    for pdf_path in pdf_files:
        pages = _extract_text_from_pdf(pdf_path)
        doc_name = pdf_path.stem

        for page_info in pages:
            text_chunks = _chunk_text(page_info["text"], _CHUNK_SIZE, _CHUNK_OVERLAP)
            for chunk_idx, chunk_text in enumerate(text_chunks):
                chunk_id = f"{doc_name}_p{page_info['page']}_c{chunk_idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_text,
                    "document_name": doc_name,
                    "page": page_info["page"],
                    "metadata": {"source_pdf": pdf_path.name},
                })
                chunk_texts.append(chunk_text)

    click.echo(f"Extracted {len(all_chunks)} chunks from {len(pdf_files)} PDFs")

    embeddings = await _embed_chunks(chunk_texts, dry_run=dry_run)

    for chunk, embedding in zip(all_chunks, embeddings, strict=True):
        chunk["embedding"] = embedding

    manifest = {"chunks": all_chunks}
    output_path.write_text(json.dumps(manifest, indent=2))
    click.echo(f"Saved corpus: {output_path} ({len(all_chunks)} chunks)")

    return output_path


@click.command()
@click.option("--pdf-dir", type=click.Path(exists=True), default=None)
@click.option("--output", type=click.Path(), default=None)
@click.option("--all", "build_all", is_flag=True, help="Build all RAG corpora.")
@click.option("--dry-run", is_flag=True, help="Use random embeddings instead of API.")
@click.option("-v", "--verbose", is_flag=True)
def main(
    pdf_dir: str | None,
    output: str | None,
    build_all: bool,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Build RAG corpus embeddings from PDF directories."""
    _load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if build_all:
        corpora = [
            ("pdfs/generated/profiling/w14_w15_corpus", "pdfs/w14_corpus/embeddings.json"),
            ("pdfs/generated/profiling/w14_w15_corpus", "pdfs/w15_corpus/embeddings.json"),
            ("pdfs/generated/ground_truth/w14_w15_corpus", "pdfs/w14_corpus/embeddings_gt.json"),
            ("pdfs/generated/ground_truth/w14_w15_corpus", "pdfs/w15_corpus/embeddings_gt.json"),
        ]
        for src, dst in corpora:
            src_path = Path(src)
            if not src_path.exists():
                click.echo(f"Skipping {src} (not found)")
                continue
            asyncio.run(build_corpus(src_path, Path(dst), dry_run=dry_run))
    elif pdf_dir and output:
        asyncio.run(build_corpus(Path(pdf_dir), Path(output), dry_run=dry_run))
    else:
        click.echo("Specify --pdf-dir + --output, or --all.", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
