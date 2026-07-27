"""Build tailored wrapper snippets for entrypoint error messages."""

from __future__ import annotations

import inspect
from typing import Any

from pretia.runner import _find_system_prompt_name, _module_uses_sdk

_CLIENT_PARAMS = frozenset({"client", "llm", "model"})
_MESSAGE_PARAMS = frozenset({"messages", "history", "conversation"})


def _find_kickoff_instance(module: Any) -> str | None:
    """Find a module-level object with a real kickoff() method (CrewAI)."""
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if obj is None or isinstance(obj, (str, int, float, bool, list, dict, set, type)):
            continue
        if "kickoff" not in dir(type(obj)) and "kickoff" not in getattr(obj, "__dict__", {}):
            continue
        kickoff = getattr(obj, "kickoff", None)
        if kickoff is not None and callable(kickoff):
            return name
    return None


def _detect_provider(module: Any) -> tuple[str, str] | None:
    """Detect the LLM provider and how it was imported.

    Returns (constructor_expr, import_line) or None.
    """
    if _module_uses_sdk(module, "openai"):
        for obj in vars(module).values():
            cls_name = getattr(type(obj), "__name__", "")
            if cls_name == "OpenAI":
                return "OpenAI()", "from openai import OpenAI"
            if cls_name == "AsyncOpenAI":
                return "AsyncOpenAI()", "from openai import AsyncOpenAI"
        import types

        for obj in vars(module).values():
            if isinstance(obj, type) and getattr(obj, "__name__", "") == "OpenAI":
                return "OpenAI()", "from openai import OpenAI"
            if isinstance(obj, types.ModuleType) and getattr(obj, "__name__", "") == "openai":
                return "openai.OpenAI()", "import openai"
        return "OpenAI()", "from openai import OpenAI"

    if _module_uses_sdk(module, "anthropic"):
        import types

        for obj in vars(module).values():
            if isinstance(obj, type) and getattr(obj, "__name__", "") == "Anthropic":
                return "Anthropic()", "from anthropic import Anthropic"
            if isinstance(obj, types.ModuleType) and getattr(obj, "__name__", "") == "anthropic":
                return "anthropic.Anthropic()", "import anthropic"
        for obj in vars(module).values():
            cls_name = getattr(type(obj), "__name__", "")
            if cls_name == "Anthropic":
                return "Anthropic()", "from anthropic import Anthropic"
        return "Anthropic()", "from anthropic import Anthropic"

    return None


def _find_agent_loop(rejected: list[tuple[str, Any]]) -> tuple[str, list[str]] | None:
    """Find the multi-arg function most likely to be the agent loop.

    Returns (function_name, param_names) or None.
    """
    for name, fn in rejected:
        try:
            sig = inspect.signature(fn)
        except (ValueError, TypeError):
            continue
        param_names = [
            p.name
            for p in sig.parameters.values()
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if set(param_names) & _MESSAGE_PARAMS:
            return name, param_names
    return None


def build_wrapper_snippet(module: Any, rejected: list[tuple[str, Any]]) -> str:
    """Build a deterministic, copy-pasteable wrapper function.

    Inspects the module for provider imports, system prompt variable, and
    agent-loop function to produce a tailored snippet.
    """
    # CrewAI-style: detect a module-level object with a .kickoff() method.
    kickoff_attr = _find_kickoff_instance(module)
    if kickoff_attr:
        return (
            f"def workflow(user_input: str):\n"
            f'    return {kickoff_attr}.kickoff(inputs={{"input": user_input}})\n'
            f"\n"
            f"# Adjust the inputs dict keys to match your Crew's template variables.\n"
            f"# Pretia captures LLM calls automatically."
        )

    provider = _detect_provider(module)
    prompt_var = _find_system_prompt_name(module)
    agent_loop = _find_agent_loop(rejected)

    if not provider and not agent_loop:
        return (
            "def workflow(user_input: str):\n"
            "    # Replace with your agent invocation\n"
            "    return your_agent(user_input)\n"
            "\n"
            "# Adjust to your code — pretia captures LLM calls automatically,\n"
            "# so the return value doesn't need a specific shape."
        )

    lines: list[str] = []
    constructor = provider[0] if provider else "YourClient()"

    lines.append("def workflow(user_input: str):")
    lines.append(f"    client = {constructor}")

    prompt_ref = prompt_var if prompt_var else '"You are a helpful assistant."'
    lines.append("    messages = [")
    lines.append(f'        {{"role": "system", "content": {prompt_ref}}},')
    lines.append('        {"role": "user", "content": user_input},')
    lines.append("    ]")

    if agent_loop:
        fn_name, params = agent_loop
        call_args: list[str] = []
        for p in params:
            if p in _CLIENT_PARAMS:
                call_args.append("client")
            elif p in _MESSAGE_PARAMS:
                call_args.append("messages")
            elif p in ("user_input", "input", "query", "prompt"):
                call_args.append("user_input")
            else:
                call_args.append("...")
        lines.append(f"    return {fn_name}({', '.join(call_args)})")
    else:
        lines.append("    # Replace with your agent call")
        lines.append("    return client.chat.completions.create(messages=messages)")

    lines.append("")
    lines.append("# Adjust to your code — pretia captures LLM calls automatically,")
    lines.append("# so the return value doesn't need a specific shape.")

    return "\n".join(lines)
