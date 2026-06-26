"""Zero-cost static analysis of workflow files for cost estimation."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MODEL_KWARGS = frozenset(
    {
        "model",
        "alternate_model",
        "classifier_model",
    }
)

_SYSTEM_PROMPT_KWARGS = frozenset(
    {
        "system_prompt",
        "system_message",
        "instructions",
    }
)

_DEFAULT_INPUT_TOKENS = 700
_DEFAULT_OUTPUT_TOKENS = 500

_FRAMEWORK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:from|import)\s+langgraph\b"), "langgraph"),
    (re.compile(r"(?:from|import)\s+(?:agents|openai\.agents)\b"), "openai-agents"),
    (re.compile(r"(?:from|import)\s+qwen_agent\b"), "qwen-agent"),
]


@dataclass(frozen=True, slots=True)
class ModelEstimate:
    """One model reference found in the source."""

    model_name: str
    canonical_name: str | None
    step_name: str | None
    max_tokens: int | None
    input_price_per_m: float | None
    output_price_per_m: float | None


@dataclass(frozen=True, slots=True)
class WorkflowEstimate:
    """Result of static analysis on a workflow file."""

    workflow_path: str
    framework: str | None
    models: list[ModelEstimate]
    estimated_cost_per_run: float
    estimated_steps: int
    estimated_system_prompt_tokens: int
    parse_error: str | None = None


def estimate_workflow(workflow_path: str) -> WorkflowEstimate:
    """Analyze a workflow file and return cost estimates without executing it."""
    path = Path(workflow_path)
    if not path.exists():
        raise FileNotFoundError(f"Workflow file not found: {workflow_path}")

    source = path.read_text(encoding="utf-8")
    framework = _detect_framework(source)
    parse_errors: list[str] = []
    raw_models = _extract_models(source, _parse_errors=parse_errors)

    models: list[ModelEstimate] = []
    unrecognized: list[str] = []
    for rm in raw_models:
        canonical, inp, outp = _resolve_pricing(rm["model_name"])
        if canonical is None:
            unrecognized.append(rm["model_name"])
        models.append(
            ModelEstimate(
                model_name=rm["model_name"],
                canonical_name=canonical,
                step_name=rm.get("step_name"),
                max_tokens=rm.get("max_tokens"),
                input_price_per_m=inp,
                output_price_per_m=outp,
            )
        )

    for name in unrecognized:
        logger.warning(
            "Unrecognized model: '%s'. Pricing unavailable. "
            "Use register_model('%s', input_price=X, output_price=Y) to add pricing.",
            name,
            name,
        )

    prompts = _extract_system_prompts(source)
    sp_tokens = sum(_estimate_tokens(p) for p in prompts) if prompts else 0

    cost = _estimate_cost(models, system_prompt_tokens=sp_tokens or None)

    return WorkflowEstimate(
        workflow_path=workflow_path,
        framework=framework,
        models=models,
        estimated_cost_per_run=cost,
        estimated_steps=len(models) or 1,
        estimated_system_prompt_tokens=sp_tokens,
        parse_error=parse_errors[0] if parse_errors else None,
    )


def _detect_framework(source: str) -> str | None:
    """Detect framework from import statements in the source."""
    for pattern, name in _FRAMEWORK_PATTERNS:
        if pattern.search(source):
            return name
    return "generic"


def _extract_models(
    source: str,
    _parse_errors: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Extract model references from the AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        msg = f"Syntax error on line {exc.lineno}: {exc.msg}"
        logger.warning("Could not parse file — %s", msg)
        if _parse_errors is not None:
            _parse_errors.append(msg)
        return []

    results: list[dict[str, Any]] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for info in _extract_from_call(node):
            if info["model_name"] not in seen:
                results.append(info)
                seen.add(info["model_name"])

    return results


def _extract_from_call(node: ast.Call) -> list[dict[str, Any]]:
    """Extract model info from a function/constructor call node."""
    step_name = None
    max_tokens = None
    model_names: list[str] = []

    for kw in node.keywords:
        if (
            kw.arg in _MODEL_KWARGS
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, str)
        ):
            model_names.append(kw.value.value)
        elif kw.arg == "step_name" and isinstance(kw.value, ast.Constant):
            step_name = kw.value.value
        elif (
            kw.arg == "max_tokens"
            and isinstance(kw.value, ast.Constant)
            and isinstance(kw.value.value, int)
        ):
            max_tokens = kw.value.value

    return [
        {"model_name": name, "step_name": step_name, "max_tokens": max_tokens}
        for name in model_names
    ]


def _extract_system_prompts(source: str) -> list[str]:
    """Extract system prompt strings from AST keyword arguments."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # already warned in _extract_models

    prompts: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for kw in node.keywords:
            if (
                kw.arg in _SYSTEM_PROMPT_KWARGS
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
                and len(kw.value.value) > 20
            ):
                text = kw.value.value
                if text not in seen:
                    prompts.append(text)
                    seen.add(text)

    return prompts


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: words * 1.3, rounded up."""
    return ceil(len(text.split()) * 1.3)


def _resolve_pricing(
    model_name: str,
) -> tuple[str | None, float | None, float | None]:
    """Try to resolve model pricing. Returns (canonical, input_$/M, output_$/M)."""
    try:
        from pretia.pricing.tables import MODEL_PRICING, resolve_model

        canonical = resolve_model(model_name)
        inp, outp = MODEL_PRICING[canonical]
        return canonical, inp, outp
    except (ValueError, KeyError):
        return None, None, None


def _estimate_cost(
    models: list[ModelEstimate],
    system_prompt_tokens: int | None = None,
) -> float:
    """Estimate the cost of a single run from model pricing."""
    total = 0.0
    for m in models:
        if m.input_price_per_m is None or m.output_price_per_m is None:
            continue

        if system_prompt_tokens:
            input_tokens = max(system_prompt_tokens, _DEFAULT_INPUT_TOKENS)
        else:
            input_tokens = _DEFAULT_INPUT_TOKENS
        output_tokens = int(m.max_tokens * 0.5) if m.max_tokens else _DEFAULT_OUTPUT_TOKENS

        input_cost = input_tokens * m.input_price_per_m / 1_000_000
        output_cost = output_tokens * m.output_price_per_m / 1_000_000
        total += input_cost + output_cost

    return round(total, 6)
