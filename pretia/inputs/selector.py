"""Auto-detect the best input mode based on available credentials and config."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InputSelection:
    """Result of input mode selection."""

    mode: str
    inputs: list[str]
    message: str


def read_inputs_file(path: str) -> list[str]:
    """Read inputs from a plain-text or JSONL file.

    Plain text: one input per line, blank lines skipped.
    JSONL: each line parsed as JSON — strings kept as-is,
    dicts/lists serialized back to JSON strings.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Inputs file not found: {path}")

    lines = p.read_text().splitlines()
    is_jsonl = p.suffix == ".jsonl"
    inputs: list[str] = []

    for line_num, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if is_jsonl:
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {exc.msg}"
                ) from exc
            if isinstance(parsed, str):
                inputs.append(parsed)
            else:
                inputs.append(json.dumps(parsed))
        else:
            inputs.append(stripped)

    return inputs


def select_input_mode(
    explicit_inputs: list[str] | None = None,
    inputs_file: str | None = None,
    single_input: str | None = None,
    auto_generate: int | None = None,
    from_langfuse: bool = False,
    system_prompt: str | None = None,
) -> InputSelection:
    """Determine the best input mode from the provided arguments and environment.

    Priority: explicit > file > single > langfuse > auto-generate > estimate.
    """
    if explicit_inputs is not None:
        if inputs_file or single_input or auto_generate or from_langfuse:
            logger.warning("Multiple input flags provided; using --input (highest priority).")
        return InputSelection(
            mode="manual",
            inputs=list(explicit_inputs),
            message=f"Using {len(explicit_inputs)} provided inputs.",
        )

    if inputs_file is not None:
        if single_input or auto_generate or from_langfuse:
            logger.warning("Multiple input flags provided; using --inputs-file.")
        inputs = read_inputs_file(inputs_file)
        return InputSelection(
            mode="file",
            inputs=inputs,
            message=(f"Loaded {len(inputs)} inputs from {inputs_file}."),
        )

    if single_input is not None:
        if auto_generate or from_langfuse:
            logger.warning("Multiple input flags provided; using --single-input.")
        return InputSelection(
            mode="single",
            inputs=[single_input],
            message="Single-input mode: one run plus priors.",
        )

    if from_langfuse:
        return InputSelection(
            mode="langfuse",
            inputs=[],
            message=("Langfuse import mode: will pull traces from Langfuse."),
        )

    if auto_generate is not None:
        return InputSelection(
            mode="auto-generate",
            inputs=[],
            message=(
                f"Will auto-generate {auto_generate} synthetic inputs from the system prompt."
            ),
        )

    return _auto_detect(system_prompt)


def _auto_detect(system_prompt: str | None) -> InputSelection:
    has_langfuse = bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )
    if has_langfuse:
        return InputSelection(
            mode="langfuse",
            inputs=[],
            message=("Langfuse credentials detected. Suggesting trace import for best accuracy."),
        )

    has_api_key = bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"))
    if system_prompt or has_api_key:
        return InputSelection(
            mode="auto-generate",
            inputs=[],
            message=("Defaulting to auto-generate 50 synthetic inputs from the system prompt."),
        )

    return InputSelection(
        mode="estimate",
        inputs=[],
        message=(
            "No inputs, credentials, or system prompt available. "
            "Using static estimate (no execution)."
        ),
    )
