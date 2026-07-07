"""Generate diverse synthetic inputs from a workflow's system prompt."""

from __future__ import annotations

import ast
import asyncio
import logging
import os
import re
from typing import Any

logger = logging.getLogger(__name__)


def _extract_workflow_context(source: str) -> str:
    """Extract domain context from workflow source: docstrings and type annotations."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ""

    parts: list[str] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            docstring = ast.get_docstring(node)
            if docstring:
                parts.append(f"{node.name}: {docstring}")

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                annotations = []
                for arg in node.args.args:
                    if arg.annotation:
                        annotations.append(f"{arg.arg}: {ast.unparse(arg.annotation)}")
                if node.returns:
                    annotations.append(f"returns {ast.unparse(node.returns)}")
                if annotations:
                    parts.append(f"{node.name}({', '.join(annotations)})")

    return "\n".join(parts)


_GENERATION_PROMPT_TEMPLATE = """\
Generate exactly {n} diverse test inputs for an AI agent workflow.

The agent's system prompt:
---
{system_prompt}
---
{additional_context}
Generate inputs that a real user would send to this agent. Cover:
- Typical usage (60%): common requests varying in topic and phrasing
- Edge cases (20%): very short inputs, very long inputs, ambiguous requests, \
multi-part questions, misspellings, non-English fragments
- Adversarial/unusual (20%): off-topic requests, attempts to confuse, \
inputs that might trigger loops or retries

Vary the user persona: novice, expert, frustrated, verbose, terse.

Output ONLY the inputs, one per line. No numbering, no explanations, \
no blank lines, no quotes around inputs."""

_PREAMBLE_PATTERNS = re.compile(
    r"^(here are|below are|these (are|inputs)|the following|sure[,!]|"
    r"of course|certainly|i'?ll generate)",
    re.IGNORECASE,
)
_NUMBERED_PREFIX = re.compile(r"^\d+[\.\)]\s*")


def _parse_response(text: str, n: int) -> list[str]:
    """Extract clean inputs from an LLM response."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _PREAMBLE_PATTERNS.search(line):
            continue
        line = _NUMBERED_PREFIX.sub("", line).strip()
        if line:
            lines.append(line)

    if len(lines) < n:
        logger.warning(
            "Requested %d inputs but LLM returned %d",
            n,
            len(lines),
        )
    return lines[:n]


_DASHSCOPE_DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
_DEEPSEEK_DEFAULT_BASE_URL = "https://api.deepseek.com"

_CHEAPEST_MODEL_FOR_PROVIDER: dict[str, str] = {
    "deepseek": "deepseek-v4-flash",
    "dashscope": "qwen-turbo",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-haiku-4-5",
}


def resolve_generator_model(explicit_model: str | None = None) -> str:
    """Pick the cheapest available model for input generation.

    If the user passed --generator-model, use that. Otherwise detect which
    API key is available and pick the cheapest model for that provider.
    Returns the model name (may differ from the default deepseek-v4-flash).
    """
    if explicit_model:
        return explicit_model

    if os.environ.get("DEEPSEEK_API_KEY") and _try_import("openai"):
        return "deepseek-v4-flash"
    if os.environ.get("OPENAI_API_KEY") and _try_import("openai"):
        return "gpt-4o-mini"
    if os.environ.get("ANTHROPIC_API_KEY") and _try_import("anthropic"):
        return "claude-haiku-4-5"
    if os.environ.get("DASHSCOPE_API_KEY") and _try_import("openai"):
        return "qwen-turbo"

    return "deepseek-v4-flash"


def _resolve_provider(
    model: str,
    api_key: str | None,
) -> tuple[str, str, Any]:
    """Return (provider, resolved_api_key, sdk_module).

    Raises ImportError if no SDK is available.
    Raises ValueError if no API key is found.
    """
    want_openai = model.startswith(("gpt-", "o1", "o3", "o4"))
    want_anthropic = model.startswith("claude-")
    want_qwen = model.startswith("qwen")
    want_deepseek = model.startswith("deepseek")

    anthropic_mod = _try_import("anthropic")
    openai_mod = _try_import("openai")

    anthropic_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    openai_key = api_key or os.environ.get("OPENAI_API_KEY")
    dashscope_key = api_key or os.environ.get("DASHSCOPE_API_KEY")
    deepseek_key = api_key or os.environ.get("DEEPSEEK_API_KEY")

    if want_qwen and openai_mod and dashscope_key:
        return "dashscope", dashscope_key, openai_mod
    if want_deepseek and openai_mod and deepseek_key:
        return "deepseek", deepseek_key, openai_mod

    if want_openai and openai_mod and openai_key:
        return "openai", openai_key, openai_mod
    if want_anthropic and anthropic_mod and anthropic_key:
        return "anthropic", anthropic_key, anthropic_mod

    # If a specific provider was requested but can't be satisfied, fail clearly
    if want_deepseek and not openai_mod:
        raise ImportError(
            f"Model '{model}' requires the `openai` package (DeepSeek uses an "
            "OpenAI-compatible API). Install it with: pip install openai"
        )
    if want_deepseek and not deepseek_key:
        raise ValueError(
            f"Model '{model}' requires DEEPSEEK_API_KEY. "
            "Set it in your environment or pass api_key directly."
        )
    if want_qwen and not openai_mod:
        raise ImportError(
            f"Model '{model}' requires the `openai` package. Install it with: pip install openai"
        )
    if want_qwen and not dashscope_key:
        raise ValueError(
            f"Model '{model}' requires DASHSCOPE_API_KEY. "
            "Set it in your environment or pass api_key directly."
        )
    if want_openai and not openai_mod:
        raise ImportError(
            f"Model '{model}' requires the `openai` package. Install it with: pip install openai"
        )
    if want_anthropic and not anthropic_mod:
        raise ImportError(
            f"Model '{model}' requires the `anthropic` package. "
            "Install it with: pip install anthropic"
        )

    # No specific provider requested. Pick the best available.
    if not want_openai and not want_anthropic and not want_qwen and not want_deepseek:
        if anthropic_mod and anthropic_key:
            return "anthropic", anthropic_key, anthropic_mod
        if dashscope_key and openai_mod:
            return "dashscope", dashscope_key, openai_mod
        if deepseek_key and openai_mod:
            return "deepseek", deepseek_key, openai_mod
        if openai_mod and openai_key:
            return "openai", openai_key, openai_mod

    if not anthropic_mod and not openai_mod:
        raise ImportError(
            "Input generation requires either the `anthropic` or `openai` "
            "package. Install one with: pip install anthropic"
            " (or) pip install openai"
        )

    raise ValueError(
        "No API key found. Set ANTHROPIC_API_KEY, OPENAI_API_KEY, "
        "DASHSCOPE_API_KEY, or DEEPSEEK_API_KEY, or pass api_key directly."
    )


def _try_import(name: str) -> Any | None:
    try:
        return __import__(name)
    except ImportError:
        return None


async def _call_anthropic(
    sdk: Any,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    client = sdk.AsyncAnthropic(api_key=api_key)
    response = await client.messages.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


async def _call_openai(
    sdk: Any,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    client = sdk.AsyncOpenAI(api_key=api_key)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _call_dashscope(
    sdk: Any,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    base_url = os.environ.get("DASHSCOPE_BASE_URL", _DASHSCOPE_DEFAULT_BASE_URL)
    client = sdk.AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def _call_deepseek(
    sdk: Any,
    api_key: str,
    model: str,
    prompt: str,
) -> str:
    base_url = os.environ.get("DEEPSEEK_BASE_URL", _DEEPSEEK_DEFAULT_BASE_URL)
    client = sdk.AsyncOpenAI(api_key=api_key, base_url=base_url)
    response = await client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


async def generate_inputs(
    system_prompt: str,
    n: int = 50,
    model: str = "deepseek-v4-flash",
    api_key: str | None = None,
    additional_context: str = "",
) -> list[str]:
    """Generate N diverse synthetic test inputs for an agent workflow.

    Args:
        system_prompt: The agent's system prompt or description.
        n: Number of inputs to generate.
        model: LLM model to use for generation.
        api_key: API key. Falls back to env vars if not provided.
        additional_context: Extra context (type hints, signatures).

    Raises:
        ImportError: If neither anthropic nor openai SDK is installed.
        ValueError: If no API key is available.
    """
    provider, resolved_key, sdk = _resolve_provider(model, api_key)

    ctx_block = ""
    if additional_context:
        ctx_block = f"\nAdditional context about the input format:\n{additional_context}\n"

    truncated_prompt = system_prompt[:2000]
    if len(system_prompt) > 2000:
        logger.warning(
            "System prompt truncated from %d to 2000 chars for input generation.",
            len(system_prompt),
        )
    prompt = _GENERATION_PROMPT_TEMPLATE.format(
        system_prompt=truncated_prompt,
        n=n,
        additional_context=ctx_block,
    )

    if provider == "anthropic":
        text = await _call_anthropic(sdk, resolved_key, model, prompt)
    elif provider == "dashscope":
        if not model.startswith("qwen"):
            model = "qwen-turbo"
        text = await _call_dashscope(sdk, resolved_key, model, prompt)
    elif provider == "deepseek":
        if not model.startswith("deepseek"):
            model = "deepseek-v4-flash"
        text = await _call_deepseek(sdk, resolved_key, model, prompt)
    else:
        if model.startswith("claude-"):
            model = "gpt-4o-mini"
        text = await _call_openai(sdk, resolved_key, model, prompt)

    return _parse_response(text, n)


def generate_inputs_sync(
    system_prompt: str,
    n: int = 50,
    model: str = "deepseek-v4-flash",
    api_key: str | None = None,
    additional_context: str = "",
) -> list[str]:
    """Synchronous wrapper around `generate_inputs()`."""
    return asyncio.run(
        generate_inputs(
            system_prompt,
            n=n,
            model=model,
            api_key=api_key,
            additional_context=additional_context,
        )
    )
