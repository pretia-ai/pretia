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
        "system",
        "system_prompt",
        "system_message",
        "instructions",
    }
)

_DEFAULT_INPUT_TOKENS = 700
_DEFAULT_OUTPUT_TOKENS = 500
_CLASSIFICATION_OUTPUT_TOKENS = 30
_MAX_TOKENS_UTILIZATION = 0.6
_CLASSIFICATION_MAX_TOKENS_THRESHOLD = 100
_CLASSIFICATION_PROMPT_WORDS = 30

_FRAMEWORK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?:from|import)\s+langgraph\b"), "langgraph"),
    (re.compile(r"(?:from|import)\s+(?:agents|openai\.agents)\b"), "openai-agents"),
    (re.compile(r"(?:from|import)\s+qwen_agent\b"), "qwen-agent"),
    (re.compile(r"(?:from|import)\s+anthropic\b"), "anthropic"),
    (re.compile(r"(?:from|import)\s+openai\b"), "openai"),
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

    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"'{workflow_path}' is not a valid Python source file (binary or non-UTF-8 content)."
        ) from exc
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

    # Graph-aware step counting
    estimated_steps = len(models) or 1
    active_models = models

    if framework == "langgraph":
        graph_info = _extract_langgraph_structure(source)
        if graph_info is not None:
            estimated_steps = max(1, round(graph_info.expected_steps))
            if len(models) > estimated_steps:
                active_models = models[:estimated_steps]
    elif framework == "openai-agents":
        agent_steps = _extract_openai_agents_structure(source)
        if agent_steps is not None:
            estimated_steps = agent_steps
            if len(models) > estimated_steps:
                active_models = models[:estimated_steps]

    cost = _estimate_cost(
        active_models,
        system_prompt_tokens=sp_tokens or None,
        expected_steps=estimated_steps,
    )

    return WorkflowEstimate(
        workflow_path=workflow_path,
        framework=framework,
        models=models,
        estimated_cost_per_run=cost,
        estimated_steps=estimated_steps,
        estimated_system_prompt_tokens=sp_tokens,
        parse_error=parse_errors[0] if parse_errors else None,
    )


@dataclass(frozen=True, slots=True)
class _GraphInfo:
    """Statically extracted graph structure."""

    nodes: list[str]
    edges: list[tuple[str, str]]
    conditional_edges: list[tuple[str, list[str]]]
    entry_point: str | None
    expected_steps: float


def _extract_langgraph_structure(source: str) -> _GraphInfo | None:
    """Parse StateGraph.add_node/add_edge/add_conditional_edges from AST."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    nodes: list[str] = []
    edges: list[tuple[str, str]] = []
    conditional_edges: list[tuple[str, list[str]]] = []
    entry_point: str | None = None

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        method_name = None
        if isinstance(func, ast.Attribute):
            method_name = func.attr

        if method_name == "add_node" and len(node.args) >= 1:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                nodes.append(arg.value)

        elif method_name == "add_edge" and len(node.args) >= 2:
            src = node.args[0]
            dst = node.args[1]
            if (
                isinstance(src, ast.Constant)
                and isinstance(dst, ast.Constant)
                and isinstance(src.value, str)
                and isinstance(dst.value, str)
            ):
                edges.append((src.value, dst.value))

        elif method_name == "add_conditional_edges" and len(node.args) >= 1:
            src = node.args[0]
            if isinstance(src, ast.Constant) and isinstance(src.value, str):
                targets: list[str] = []
                # Collect targets from dict values (arg 3 or path_map kwarg)
                dicts_to_scan: list[ast.Dict] = []
                if len(node.args) >= 3 and isinstance(node.args[2], ast.Dict):
                    dicts_to_scan.append(node.args[2])
                for kw in node.keywords:
                    if kw.arg == "path_map" and isinstance(kw.value, ast.Dict):
                        dicts_to_scan.append(kw.value)
                for d in dicts_to_scan:
                    for v in d.values:
                        if isinstance(v, ast.Constant) and isinstance(v.value, str):
                            targets.append(v.value)
                        elif isinstance(v, ast.Name) and v.id == "END":
                            targets.append("END")
                        elif isinstance(v, ast.Attribute) and v.attr == "END":
                            targets.append("END")
                conditional_edges.append((src.value, targets))

        elif method_name == "set_entry_point" and len(node.args) >= 1:
            arg = node.args[0]
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                entry_point = arg.value

    if not nodes:
        return None

    # Estimate expected steps per run from the graph structure.
    # Each node on the main path counts as 1 step.
    # Conditional edge targets each count as 0.5 (expected value for binary branch).
    reachable: set[str] = set()
    if entry_point:
        reachable.add(entry_point)
    queue = [entry_point] if entry_point else list(nodes[:1])
    visited: set[str] = set()
    while queue:
        current = queue.pop(0)
        if current in visited or current is None:
            continue
        visited.add(current)
        reachable.add(current)
        for src, dst in edges:
            if src == current and dst not in visited and dst != "END" and dst != "__end__":
                queue.append(dst)
        for src, targets in conditional_edges:
            if src == current:
                for t in targets:
                    if t not in visited and t != "END" and t != "__end__":
                        queue.append(t)

    # Expected steps: deterministic edges = 1.0, conditional targets = 1/len(targets)
    # Count nodes reachable via deterministic edges as 1.0 each.
    # For conditional edges, each branch target gets weight 1/N instead of 1.0.
    deterministic_nodes: set[str] = set()
    conditional_nodes: dict[str, float] = {}

    # Mark all nodes as deterministic first
    for n in reachable:
        deterministic_nodes.add(n)

    # Downweight nodes only reachable via conditional edges
    for src, targets in conditional_edges:
        if src not in reachable:
            continue
        real_targets = [t for t in targets if t in reachable and t != "END" and t != "__end__"]
        n_branches = max(len(targets), 1)
        for t in real_targets:
            if t in deterministic_nodes:
                deterministic_nodes.discard(t)
                conditional_nodes[t] = 1.0 / n_branches

    expected = float(len(deterministic_nodes))
    expected += sum(conditional_nodes.values())

    return _GraphInfo(
        nodes=nodes,
        edges=edges,
        conditional_edges=conditional_edges,
        entry_point=entry_point,
        expected_steps=max(expected, 1.0),
    )


def _extract_openai_agents_structure(source: str) -> int | None:
    """Count OpenAI Agents: only price entry agent unless handoffs detected."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    agent_count = 0
    has_handoffs = False

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func_name = None
        if isinstance(node.func, ast.Name):
            func_name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        if func_name == "Agent":
            agent_count += 1
            for kw in node.keywords:
                if kw.arg == "handoffs":
                    has_handoffs = True

    if agent_count == 0:
        return None
    if has_handoffs:
        return agent_count
    return 1


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

    # Build a map of module-level string constants for variable resolution
    string_vars: dict[str, str] = {}
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (
                isinstance(target, ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
                and len(node.value.value) > 20
            ):
                string_vars[target.id] = node.value.value

    prompts: list[str] = []
    seen: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        for kw in node.keywords:
            if kw.arg not in _SYSTEM_PROMPT_KWARGS:
                continue
            text = None
            if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                text = kw.value.value
            elif isinstance(kw.value, ast.Name) and kw.value.id in string_vars:
                text = string_vars[kw.value.id]
            if text and len(text) > 20 and text not in seen:
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


def _estimate_output_tokens(max_tokens: int | None) -> int:
    """Estimate output tokens from max_tokens setting."""
    if max_tokens is None:
        return _DEFAULT_OUTPUT_TOKENS
    if max_tokens <= _CLASSIFICATION_MAX_TOKENS_THRESHOLD:
        return max(int(max_tokens * 0.8), 5)
    return int(max_tokens * _MAX_TOKENS_UTILIZATION)


_DEFAULT_USER_INPUT_TOKENS = 150


def _estimate_cost(
    models: list[ModelEstimate],
    system_prompt_tokens: int | None = None,
    expected_steps: int | None = None,
) -> float:
    """Estimate the cost of a single run from model pricing."""
    total = 0.0
    for m in models:
        if m.input_price_per_m is None or m.output_price_per_m is None:
            continue

        if system_prompt_tokens:
            input_tokens = system_prompt_tokens + _DEFAULT_USER_INPUT_TOKENS
        else:
            input_tokens = _DEFAULT_INPUT_TOKENS
        output_tokens = _estimate_output_tokens(m.max_tokens)

        input_cost = input_tokens * m.input_price_per_m / 1_000_000
        output_cost = output_tokens * m.output_price_per_m / 1_000_000
        total += input_cost + output_cost

    # Scale for graph-aware estimation: if expected_steps > len(models)
    # (e.g. same model used in multiple nodes), scale proportionally
    if expected_steps and len(models) > 0 and expected_steps > len(models):
        total *= expected_steps / len(models)

    return round(total, 6)
