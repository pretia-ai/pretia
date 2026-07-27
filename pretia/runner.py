"""Orchestrate the end-to-end profiling pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import inspect
import logging
import re
import statistics
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from pretia.collectors.base import BaseCollector, StepRecord
from pretia.collectors.generic import GenericCollector
from pretia.inputs.generator import _extract_workflow_context, generate_inputs
from pretia.inputs.selector import InputSelection, select_input_mode
from pretia.pricing.tables import calculate_cost, model_tier
from pretia.projection.patterns import detect_patterns
from pretia.projection.projector import project
from pretia.projection.stats import compute_stats, percentile
from pretia.store import ProfileStore, ProfilingSession

logger = logging.getLogger(__name__)

_WORKFLOW_ATTR_NAMES = ("graph", "workflow", "agent", "app")
_CALLABLE_ATTR_NAMES = ("run", "call", "process", "execute", "handle", "main")
_GRAPH_BUILDER_NAMES = ("build_graph", "create_graph", "make_graph", "get_graph")
_AGENT_BUILDER_NAMES = ("create_agent", "build_agent", "make_agent", "get_agent")
_WRAPPER_FRAMEWORKS = (
    "pydantic_ai",
    "instructor",
    "mirascope",
    "llama_index",
    "dspy",
    "smolagents",
    "haystack",
)
_INSTANCE_METHOD_NAMES = (
    "run",
    "call",
    "process",
    "execute",
    "handle",
    "chat",
    "ask",
    "respond",
    "answer",
    "query",
    "reply",
)
_SYSTEM_PROMPT_RE = re.compile(
    r"(you are|your role|your task|system)",
    re.IGNORECASE,
)

_LLM_CLASS_NAMES = frozenset(
    {
        "ChatAnthropic",
        "ChatOpenAI",
        "ChatGoogleGenerativeAI",
        "ChatMistralAI",
        "ChatOllama",
        "BaseChatModel",
        "BaseLLM",
    }
)


class EntrypointError(click.UsageError):
    """No usable entrypoint found. Carries rejected candidates and a tailored wrapper."""

    def __init__(
        self,
        message: str,
        *,
        rejected: list[tuple[str, Any]],
        wrapper_snippet: str,
        workflow_path: str | None = None,
    ) -> None:
        super().__init__(message)
        self.rejected = rejected
        self.wrapper_snippet = wrapper_snippet
        self.workflow_path = workflow_path


class AmbiguousEntrypointError(click.UsageError):
    """Multiple valid entrypoint candidates found."""

    def __init__(
        self,
        message: str,
        *,
        candidates: list[tuple[str, str]],
    ) -> None:
        super().__init__(message)
        self.candidates = candidates


def _is_llm_model(obj: Any) -> bool:
    """Return True if obj is a LangChain LLM/chat model (instance or class)."""
    name = type(obj).__name__
    if name in _LLM_CLASS_NAMES:
        return True
    cls_name = getattr(obj, "__name__", "")
    if cls_name in _LLM_CLASS_NAMES:
        return True
    for base in getattr(obj, "__mro__", ()):
        if getattr(base, "__name__", "") in _LLM_CLASS_NAMES:
            return True
    return False


def _is_tool_object(obj: Any) -> bool:
    """Return True if obj is a LangChain tool instance (@tool decorator output)."""
    for base in getattr(type(obj), "__mro__", ()):
        if getattr(base, "__name__", "") == "BaseTool":
            return True
    return False


_CLI_COMMAND_NAMES = frozenset({"BaseCommand", "Command", "Group", "MultiCommand", "Typer"})


def _is_cli_command(obj: Any) -> bool:
    """Return True if obj is a click/typer CLI command (never a workflow)."""
    for base in getattr(type(obj), "__mro__", ()):
        if getattr(base, "__name__", "") in _CLI_COMMAND_NAMES:
            return True
    return False


def _is_workflow_candidate(obj: Any) -> bool:
    if obj is None or isinstance(obj, (str, int, float, bool, list, dict, set, type)):
        return False
    if _is_llm_model(obj) or _is_cli_command(obj):
        return False
    if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
        return True
    if asyncio.iscoroutinefunction(obj) or callable(obj):
        return True
    return False


def _registered_tool_names(module: Any) -> set[str]:
    """Collect names of functions registered as tools in the module.

    Detects two patterns:
    - Dict registries: ``TOOL_DISPATCH = {"fn_name": fn, ...}`` — collects
      ``__name__`` of every callable value.
    - Schema lists: ``TOOLS = [{"function": {"name": "fn_name"}}, ...]`` or
      ``[{"name": "fn_name"}, ...]`` — collects string ``"name"`` values at
      depth <= 3.
    """
    names: set[str] = set()
    for attr_name in dir(module):
        if attr_name.startswith("_"):
            continue
        obj = getattr(module, attr_name, None)
        if isinstance(obj, dict):
            for v in obj.values():
                if callable(v):
                    fn_name = getattr(v, "__name__", None)
                    if fn_name:
                        names.add(fn_name)
        elif isinstance(obj, list):
            _collect_schema_names(obj, names, depth=0)
    return names


def _collect_schema_names(obj: Any, names: set[str], depth: int) -> None:
    if depth > 3:
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "name" and isinstance(v, str):
                names.add(v)
            elif isinstance(v, (dict, list)):
                _collect_schema_names(v, names, depth + 1)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                _collect_schema_names(item, names, depth + 1)


def _find_system_prompt_name(module: Any) -> str | None:
    """Return the attribute name of the system prompt variable, if any."""
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if isinstance(obj, str) and len(obj) > 50 and _SYSTEM_PROMPT_RE.search(obj):
            return name
    return None


def _try_instantiate_class(cls: type) -> Any | None:
    """Instantiate a class with no arguments, returning None on failure.

    Only attempts classes whose constructor has zero required positional params.
    """
    try:
        sig = inspect.signature(cls)
    except (ValueError, TypeError):
        return None
    for p in sig.parameters.values():
        if (
            p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ):
            return None
    try:
        return cls()
    except Exception as exc:
        logger.warning(
            "Class %s() has a zero-arg constructor but instantiation failed (%s: %s).",
            getattr(cls, "__name__", cls),
            type(exc).__name__,
            exc,
        )
        logger.debug("Class %s() instantiation traceback", cls, exc_info=True)
        return None


def _resolve_instance_method(instance: Any) -> tuple[str, Any] | None:
    """Find a single-input method on an instance suitable as a workflow entrypoint.

    Checks __call__ first, then method names from _INSTANCE_METHOD_NAMES.
    Returns (method_name, bound_method_or_instance) or None.
    """
    if callable(instance) and _callable_accepts_single_input(instance):
        return ("__call__", instance)
    for name in _INSTANCE_METHOD_NAMES:
        method = getattr(instance, name, None)
        if method is not None and callable(method) and _callable_accepts_single_input(method):
            return (name, method)
    return None


def _make_entry_wrapper(
    cls: type | None,
    method_name: str,
    fallback_bound: Any,
) -> Any:
    """Build a per-run wrapper that reinstantiates the class for state isolation.

    If cls is provided, each call creates a fresh instance. If cls is None (e.g.
    discovery from a module-level instance whose class needs args), falls back to
    the shared bound method.
    """
    is_async = asyncio.iscoroutinefunction(fallback_bound)

    if cls is None:
        return fallback_bound

    if method_name == "__call__":
        if is_async:

            async def _async_call_wrapper(inp: str) -> Any:
                return await cls()(inp)

            _async_call_wrapper.__name__ = getattr(cls, "__name__", "agent")
            return _async_call_wrapper

        def _sync_call_wrapper(inp: str) -> Any:
            return cls()(inp)

        _sync_call_wrapper.__name__ = getattr(cls, "__name__", "agent")
        return _sync_call_wrapper

    if is_async:

        async def _async_method_wrapper(inp: str) -> Any:
            return await getattr(cls(), method_name)(inp)

        _async_method_wrapper.__name__ = f"{getattr(cls, '__name__', 'agent')}.{method_name}"
        return _async_method_wrapper

    def _sync_method_wrapper(inp: str) -> Any:
        return getattr(cls(), method_name)(inp)

    _sync_method_wrapper.__name__ = f"{getattr(cls, '__name__', 'agent')}.{method_name}"
    return _sync_method_wrapper


def _check_signature_accepts_single(sig: inspect.Signature) -> bool:
    """Check a resolved Signature for the single-input contract."""
    required_positional = 0
    has_positional_slot = False
    for p in sig.parameters.values():
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            has_positional_slot = True
            if p.default is inspect.Parameter.empty:
                required_positional += 1
        elif p.kind is inspect.Parameter.VAR_POSITIONAL:
            has_positional_slot = True
        elif p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is inspect.Parameter.empty:
            return False
    return has_positional_slot and required_positional <= 1


def _callable_accepts_single_input(fn: Any) -> bool:
    """Return True if fn(inp) with one positional string argument is a valid call."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return True
    if _check_signature_accepts_single(sig):
        return True
    if hasattr(fn, "__wrapped__"):
        try:
            unwrapped_sig = inspect.signature(fn, follow_wrapped=False)
        except (ValueError, TypeError):
            return True
        return _check_signature_accepts_single(unwrapped_sig)
    return False


def _signature_str(name: str, fn: Any) -> str:
    """Format a callable's name and signature for error messages."""
    try:
        return f"{name}{inspect.signature(fn)}"
    except (ValueError, TypeError):
        return f"{name}(...)"


def _rejection_reason(fn: Any) -> str:
    """Describe why a callable fails the single-input contract."""
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return "signature not introspectable"
    required = sum(
        1
        for p in sig.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and p.default is inspect.Parameter.empty
    )
    if required == 0:
        return "takes no arguments"
    return f"takes {required} required arguments"


def _set_provenance(prov: dict[str, str] | None, entrypoint: str, rule: str) -> None:
    if prov is not None:
        prov["entrypoint"] = entrypoint
        prov["rule"] = rule


def _find_workflow(
    module: Any,
    entry_point: str | None = None,
    workflow_path: str | None = None,
    provenance: dict[str, str] | None = None,
) -> Any | None:
    if entry_point is not None:
        obj = getattr(module, entry_point, None)
        if obj is not None:
            if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
                _set_provenance(provenance, entry_point, "--entry-point")
                return obj
            if isinstance(obj, type):
                instance = _try_instantiate_class(obj)
                if instance is not None:
                    resolved = _resolve_instance_method(instance)
                    if resolved is not None:
                        method_name, bound = resolved
                        _set_provenance(
                            provenance,
                            f"{entry_point}.{method_name}",
                            "--entry-point",
                        )
                        return _make_entry_wrapper(obj, method_name, bound)
                    raise click.UsageError(
                        f"--entry-point '{entry_point}' is a class but has no "
                        f"single-input method (run, call, process, etc.). "
                        f"Add a run(self, input) method or create an instance at "
                        f"module level and use --entry-point <instance_name>."
                    )
                raise click.UsageError(
                    f"--entry-point '{entry_point}' is a class that requires "
                    f"constructor arguments. Create an instance at module level:\n"
                    f"    agent = {entry_point}(...)\n"
                    f"and re-run, or use --entry-point agent."
                )
            if not callable(obj):
                resolved = _resolve_instance_method(obj)
                if resolved is not None:
                    method_name, bound = resolved
                    cls = type(obj)
                    reinstantiable = _try_instantiate_class(cls) if cls is not type else None
                    _set_provenance(
                        provenance,
                        f"{entry_point}.{method_name}",
                        "--entry-point",
                    )
                    return _make_entry_wrapper(
                        cls if reinstantiable is not None else None,
                        method_name,
                        bound,
                    )
                raise click.UsageError(
                    f"--entry-point '{entry_point}' resolved to {type(obj).__name__} "
                    f"and cannot be profiled. Pretia calls the entrypoint as "
                    f"{entry_point}(input) with a single input string."
                )
            if not _callable_accepts_single_input(obj):
                raise click.UsageError(
                    f"--entry-point '{entry_point}' has signature "
                    f"{_signature_str(entry_point, obj)}, but Pretia calls the "
                    f"entrypoint as {entry_point}(input) with a single input string. "
                    f"Wrap it in a one-argument function or choose another entrypoint."
                )
            _set_provenance(provenance, entry_point, "--entry-point")
            return obj
        raise click.UsageError(
            f"--entry-point '{entry_point}' not found in module. "
            f"Available names: {_list_candidates(module)}"
        )

    tool_names = _registered_tool_names(module)

    # 1a. Prefer canonical names that have ainvoke/invoke (compiled graphs)
    for name in _WORKFLOW_ATTR_NAMES:
        obj = getattr(module, name, None)
        if (
            obj is not None
            and not _is_llm_model(obj)
            and (hasattr(obj, "ainvoke") or hasattr(obj, "invoke"))
        ):
            _set_provenance(provenance, name, f"canonical name '{name}' (invoke object)")
            return obj

    # 1b. Fall back to canonical names: workflow candidates, class instances
    # with agent methods, or callable instances (arity-checked to avoid
    # picking ASGI/WSGI apps like FastAPI or Flask).
    for name in _WORKFLOW_ATTR_NAMES:
        obj = getattr(module, name, None)
        if obj is None or _is_llm_model(obj):
            continue
        if isinstance(obj, type):
            continue
        if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
            continue  # already handled by 1a
        if callable(obj) and _callable_accepts_single_input(obj):
            _set_provenance(provenance, name, f"canonical name '{name}'")
            return obj
        resolved = _resolve_instance_method(obj)
        if resolved is not None:
            method_name, bound = resolved
            cls = type(obj)
            reinstantiable = _try_instantiate_class(cls) if cls is not type else None
            _set_provenance(
                provenance,
                f"{name}.{method_name}",
                f"instance method at canonical name '{name}'",
            )
            return _make_entry_wrapper(
                cls if reinstantiable is not None else None,
                method_name,
                bound,
            )

    # 1c. Try graph/agent builder functions
    for name in (*_GRAPH_BUILDER_NAMES, *_AGENT_BUILDER_NAMES):
        fn = getattr(module, name, None)
        if fn is not None and callable(fn) and not _is_llm_model(fn):
            try:
                result = fn()
                if hasattr(result, "ainvoke") or hasattr(result, "invoke"):
                    _set_provenance(provenance, f"{name}()", f"builder {name}()")
                    return result
                resolved = _resolve_instance_method(result)
                if resolved is not None:
                    method_name, bound = resolved

                    def _factory_wrapper_sync(
                        inp: str, _f: Any = fn, _m: str = method_name
                    ) -> Any:
                        return getattr(_f(), _m)(inp)

                    async def _factory_wrapper_async(
                        inp: str, _f: Any = fn, _m: str = method_name
                    ) -> Any:
                        return await getattr(_f(), _m)(inp)

                    if asyncio.iscoroutinefunction(bound):
                        _factory_wrapper_async.__name__ = f"{name}().{method_name}"
                        _set_provenance(
                            provenance,
                            f"{name}().{method_name}",
                            f"builder {name}()",
                        )
                        return _factory_wrapper_async
                    _factory_wrapper_sync.__name__ = f"{name}().{method_name}"
                    _set_provenance(
                        provenance,
                        f"{name}().{method_name}",
                        f"builder {name}()",
                    )
                    return _factory_wrapper_sync
            except Exception as exc:
                logger.warning(
                    "Found %s() but calling it failed (%s: %s). "
                    "Falling back to other workflow candidates.",
                    name,
                    type(exc).__name__,
                    exc,
                )
                logger.debug("Builder %s() traceback", name, exc_info=True)

    # 1d. Try agent-like classes and module-level instances with agent methods.
    mod_name_1d = getattr(module, "__name__", "")
    class_candidates: list[tuple[str, str, Any]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if obj is None or _is_llm_model(obj):
            continue
        if isinstance(obj, type):
            if getattr(obj, "__module__", None) != mod_name_1d:
                continue
            instance = _try_instantiate_class(obj)
            if instance is None:
                continue
            resolved = _resolve_instance_method(instance)
            if resolved is not None:
                method_name, bound = resolved
                wrapper = _make_entry_wrapper(obj, method_name, bound)
                class_candidates.append((f"{name}.{method_name}", name, wrapper))
        elif not isinstance(obj, (str, int, float, bool, list, dict, set)):
            if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
                continue
            if callable(obj):
                continue
            resolved = _resolve_instance_method(obj)
            if resolved is not None:
                method_name, bound = resolved
                cls = type(obj)
                reinstantiable = _try_instantiate_class(cls) if cls is not type else None
                wrapper = _make_entry_wrapper(
                    cls if reinstantiable is not None else None,
                    method_name,
                    bound,
                )
                class_candidates.append((f"{name}.{method_name}", name, wrapper))

    if len(class_candidates) == 1:
        label, cname, wrapper = class_candidates[0]
        is_class = isinstance(getattr(module, cname, None), type)
        rule = "agent class scan" if is_class else "module instance scan"
        _set_provenance(provenance, label, rule)
        return wrapper

    if len(class_candidates) > 1:
        candidates = [(label, _signature_str(label, w)) for label, _, w in class_candidates]
        names = ", ".join(f"'{label}'" for label, _, _ in class_candidates)
        raise AmbiguousEntrypointError(
            f"Found multiple agent classes/instances in module: {names}. "
            f"Specify which one to profile with --entry-point <name>.",
            candidates=candidates,
        )

    # 2. Check for ainvoke/invoke (framework compiled graphs, skip LLM models).
    # Skip classes: an unbound Cls.ainvoke(payload) puts payload in `self` and
    # fails with "missing 1 required positional argument". Skip tools: @tool
    # objects have ainvoke but are workflow components, not the workflow.
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if (
            isinstance(obj, type)
            or _is_llm_model(obj)
            or _is_tool_object(obj)
            or _is_cli_command(obj)
        ):
            continue
        if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
            _set_provenance(provenance, name, f"invoke object '{name}'")
            return obj

    # 3. Check common callable names (arity-validated).
    # CLI commands (click/typer) are unwrapped to their callback function.
    rejected: list[tuple[str, Any]] = []
    for name in _CALLABLE_ATTR_NAMES:
        obj = getattr(module, name, None)
        if obj is None:
            continue
        if _is_cli_command(obj):
            callback = getattr(obj, "callback", None)
            if callback is not None and _callable_accepts_single_input(callback):
                _set_provenance(
                    provenance,
                    name,
                    f"click command '{name}' (unwrapped callback)",
                )
                return callback
            if callback is not None:
                rejected.append((name, callback))
            continue
        if not _is_workflow_candidate(obj):
            continue
        if _callable_accepts_single_input(obj):
            _set_provenance(provenance, name, f"named function '{name}'")
            return obj
        rejected.append((name, obj))

    # 4. Find any async callable (arity-validated), excluding tool functions
    valid: list[tuple[str, Any]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if not asyncio.iscoroutinefunction(obj):
            continue
        if name in tool_names:
            continue
        if _callable_accepts_single_input(obj):
            valid.append((name, obj))
        else:
            rejected.append((name, obj))

    # Prefer names declared in __all__ to reduce false ambiguity.
    mod_all = getattr(module, "__all__", None)
    if isinstance(mod_all, (list, tuple)) and len(valid) > 1:
        exported = [(n, o) for n, o in valid if n in mod_all]
        if exported:
            valid = exported

    if len(valid) == 1:
        _set_provenance(provenance, valid[0][0], f"sole async function '{valid[0][0]}'")
        return valid[0][1]

    if len(valid) > 1:
        candidates = [(n, _signature_str(n, o)) for n, o in valid]
        names = ", ".join(f"'{n}'" for n, _ in valid)
        raise AmbiguousEntrypointError(
            f"Found multiple async callables in module: {names}. "
            f"Specify which one to profile with --entry-point <name>.",
            candidates=candidates,
        )

    # 5. Find sync callables as last resort (arity-validated), excluding tool
    # functions. Picking the alphabetically-first match silently profiles the
    # wrong function when a script has several single-arg helpers (e.g. tool
    # implementations), so a unique match is required.
    sync_valid: list[tuple[str, Any]] = []
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if not (_is_workflow_candidate(obj) and callable(obj)):
            continue
        if name in tool_names:
            continue
        if _callable_accepts_single_input(obj):
            sync_valid.append((name, obj))
        else:
            rejected.append((name, obj))

    # Prefer functions defined in the workflow file over imports.
    # When local rejected candidates exist (user's real functions that fail the
    # arity contract), don't fall back to imported utilities like load_dotenv.
    mod_name = getattr(module, "__name__", "")
    local_valid = [(n, o) for n, o in sync_valid if getattr(o, "__module__", None) == mod_name]
    if local_valid:
        sync_valid = local_valid
    elif rejected:
        local_rejected = [
            (n, o) for n, o in rejected if getattr(o, "__module__", None) == mod_name
        ]
        if local_rejected:
            sync_valid = []

    if isinstance(mod_all, (list, tuple)) and len(sync_valid) > 1:
        exported = [(n, o) for n, o in sync_valid if n in mod_all]
        if exported:
            sync_valid = exported

    if len(sync_valid) == 1:
        _set_provenance(
            provenance,
            sync_valid[0][0],
            f"sole single-argument function '{sync_valid[0][0]}'",
        )
        return sync_valid[0][1]

    if len(sync_valid) > 1:
        candidates = [(n, _signature_str(n, o)) for n, o in sync_valid]
        lines = [f"  - {sig}" for _, sig in candidates]
        raise AmbiguousEntrypointError(
            "Found multiple single-argument callables and cannot tell which one "
            "is your workflow entrypoint:\n"
            + "\n".join(lines)
            + "\nFix: select one with --entry-point <name>, or add a function "
            "named workflow(user_input) that invokes your agent end-to-end "
            "and returns the raw response.",
            candidates=candidates,
        )

    # No usable entrypoint. Build a tailored wrapper snippet.
    from pretia.wrapper_hint import build_wrapper_snippet

    snippet = build_wrapper_snippet(module, rejected)

    if rejected:
        mod_name = getattr(module, "__name__", "")
        local = [(n, o) for n, o in rejected if getattr(o, "__module__", None) == mod_name]
        shown = local if local else rejected
        seen: set[str] = set()
        lines: list[str] = []
        for name, obj in shown:
            sig_line = _signature_str(name, obj)
            if sig_line in seen:
                continue
            seen.add(sig_line)
            lines.append(f"  - {sig_line} — {_rejection_reason(obj)}")
        raise EntrypointError(
            "No usable entrypoint found in the workflow module. Pretia calls "
            "your workflow as entrypoint(input) with a single input string, "
            "but none of these callables accept exactly one argument:\n" + "\n".join(lines),
            rejected=rejected,
            wrapper_snippet=snippet,
            workflow_path=workflow_path,
        )

    return None


def _list_candidates(module: Any) -> str:
    parts: list[str] = []
    for n in dir(module):
        if n.startswith("_"):
            continue
        obj = getattr(module, n, None)
        if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
            parts.append(f"'{n}'")
        elif _is_workflow_candidate(obj) and callable(obj):
            parts.append(_signature_str(n, obj))
    if not parts:
        return "(none found)"
    return ", ".join(parts)


def _extract_system_prompt(module: Any) -> str:
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if isinstance(obj, str) and len(obj) > 50 and _SYSTEM_PROMPT_RE.search(obj):
            return obj
    return ""


def _detect_graph_input_key(graph: Any) -> str:
    """Detect the input key from a LangGraph state schema.

    Returns "messages" if the state uses the standard LangGraph message pattern,
    otherwise looks for a plain str-typed field, falling back to "input".
    """
    import typing

    schema = None
    builder = getattr(graph, "builder", None)
    if builder is not None:
        schema = getattr(builder, "schema", None)

    if schema is None:
        channels = getattr(graph, "channels", None)
        if channels and isinstance(channels, dict):
            if "messages" in channels:
                return "messages"
            for key in channels:
                if isinstance(key, str):
                    return key

    if schema is not None:
        annotations = getattr(schema, "__annotations__", {})
        if "messages" in annotations:
            return "messages"
        for key, type_hint in annotations.items():
            origin = getattr(type_hint, "__origin__", None)
            if type_hint is str or (origin is None and type_hint is str):
                return key
            try:
                if typing.get_origin(type_hint) is None and issubclass(type_hint, str):
                    return key
            except TypeError:
                continue

    return "input"


def _module_uses_sdk(module: Any, sdk_name: str) -> bool:
    """Check if a loaded module imported a given SDK package (module or symbols)."""
    import types

    prefix = sdk_name + "."
    for obj in vars(module).values():
        if isinstance(obj, types.ModuleType):
            mod_name = getattr(obj, "__name__", "")
            if mod_name == sdk_name or mod_name.startswith(prefix):
                return True
            continue
        obj_module = getattr(obj, "__module__", None) or ""
        if obj_module == sdk_name or obj_module.startswith(prefix):
            return True
    return False


def _load_workflow_module(path: str) -> Any:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    if p.suffix == ".ipynb":
        raise click.UsageError(
            "Jupyter notebooks can't be profiled directly. Export first:\n"
            f"  jupyter nbconvert --to script {path}\n"
            "then run pretia on the resulting .py file."
        )
    if p.suffix and p.suffix != ".py":
        raise click.UsageError(
            f"Cannot load '{path}' — pretia profiles .py files. Got '{p.suffix}' extension."
        )

    # Detect package context: walk up while __init__.py exists.
    package_parts: list[str] = []
    parent = p.parent
    while (parent / "__init__.py").exists():
        package_parts.append(parent.name)
        parent = parent.parent
    package_root = parent
    package_parts.reverse()

    # Ensure the workflow directory (or package root) is importable.
    root_str = str(package_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

    if package_parts:
        dotted = ".".join([*package_parts, p.stem])
    else:
        dotted = p.stem

    mod_name = dotted
    if mod_name in sys.modules:
        mod_name = f"_pretia_workflow_{dotted.replace('.', '_')}"

    # Register parent packages so relative imports resolve.
    if package_parts:
        for i in range(len(package_parts)):
            parent_dotted = ".".join(package_parts[: i + 1])
            if parent_dotted not in sys.modules:
                try:
                    importlib.import_module(parent_dotted)
                except Exception:
                    logger.debug(
                        "Could not import parent package %s",
                        parent_dotted,
                        exc_info=True,
                    )

    spec = importlib.util.spec_from_file_location(
        mod_name,
        str(p),
        submodule_search_locations=([str(p.parent)] if package_parts else None),
    )
    if spec is None or spec.loader is None:
        raise click.UsageError(f"Cannot load module from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    if package_parts:
        module.__package__ = ".".join(package_parts)

    # Register before exec: `from __future__ import annotations` makes all
    # annotations strings, and get_type_hints() (called by LangGraph's
    # StateGraph on TypedDict schemas) resolves them via
    # sys.modules[cls.__module__].__dict__ — without this entry it raises
    # NameError on names the workflow file imported.
    # Guard: Streamlit/Gradio apps cause hangs or confusing errors.
    try:
        source = p.read_text(encoding="utf-8", errors="replace")
    except OSError:
        source = ""
    if re.search(r"^\s*(?:import|from)\s+streamlit\b", source, re.MULTILINE):
        raise click.UsageError(
            f"'{path}' is a Streamlit app. Pretia profiles agent functions, "
            f"not UI scripts. Extract the agent logic into a separate module:\n"
            f"  def workflow(user_input: str): ...\n"
            f"and point pretia at that file."
        )
    if re.search(r"^\s*(?:import|from)\s+gradio\b", source, re.MULTILINE) and re.search(
        r"^\w[^\n]*\.launch\(", source, re.MULTILINE
    ):
        raise click.UsageError(
            f"'{path}' launches a Gradio server at import time. "
            f"Point pretia at the function Gradio wraps (the fn= argument) "
            f"in a separate file instead."
        )

    sys.modules[mod_name] = module
    loaded = False
    try:
        spec.loader.exec_module(module)
        loaded = True
    except ImportError as exc:
        sys.modules.pop(mod_name, None)
        pkg = exc.name or str(exc)
        pkg_as_path = p.parent / pkg
        if (pkg_as_path.with_suffix(".py")).exists() or pkg_as_path.is_dir():
            raise ImportError(
                f"'{path}' tried to import '{pkg}' which exists locally "
                f"but failed to load. Ensure the file is valid Python "
                f"and any dependencies are installed."
            ) from exc
        raise ImportError(
            f"'{path}' requires '{pkg}' which is not installed. Install it with: pip install {pkg}"
        ) from exc
    except SyntaxError as exc:
        sys.modules.pop(mod_name, None)
        raise click.UsageError(
            f"Syntax error in '{path}' on line {exc.lineno}: {exc.msg}"
        ) from exc
    except SystemExit as exc:
        sys.modules.pop(mod_name, None)
        raise click.UsageError(
            f"'{path}' called sys.exit() during import. "
            f"Guard module-level code with: if __name__ == '__main__':"
        ) from exc
    except Exception as exc:
        sys.modules.pop(mod_name, None)
        exc_name = type(exc).__name__
        msg = str(exc)
        api_exc_names = {
            "OpenAIError",
            "AuthenticationError",
            "APIError",
            "APIStatusError",
            "APIConnectionError",
            "AnthropicError",
        }
        if exc_name in api_exc_names or "api_key" in msg.lower():
            raise click.UsageError(
                f"'{path}' failed during import due to an API/auth "
                f"configuration error:\n  {exc_name}: {msg}\n"
                f"Hint: set the required API key environment variable "
                f"(e.g. in .env), or move client initialization inside "
                f"your workflow function."
            ) from exc
        raise click.UsageError(f"Failed to load '{path}': {exc_name}: {exc}") from exc
    finally:
        if not loaded:
            sys.modules.pop(mod_name, None)
    return module


def _build_cost_summary(
    runs: list[list[StepRecord]],
) -> dict[str, Any]:
    step_costs: dict[str, list[dict[str, Any]]] = {}
    run_totals: list[float] = []

    for run in runs:
        run_cost = 0.0
        for rec in run:
            if not rec.model or rec.step_type == "tool":
                cost = 0.0
            else:
                try:
                    cost = calculate_cost(
                        rec.model,
                        rec.input_tokens,
                        rec.output_tokens,
                    )
                except ValueError:
                    cost = 0.0

            entry = {
                "cost": cost,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "duration_ms": rec.duration_ms,
                "iteration": rec.iteration,
            }
            step_costs.setdefault(rec.step_name, []).append(entry)
            run_cost += cost
        run_totals.append(run_cost)

    per_step: dict[str, dict[str, Any]] = {}
    for step_name, entries in step_costs.items():
        costs = [e["cost"] for e in entries]
        in_toks = [e["input_tokens"] for e in entries]
        out_toks = [e["output_tokens"] for e in entries]
        durations = [e["duration_ms"] for e in entries]
        iterations = [e["iteration"] for e in entries]
        sorted_costs = sorted(costs)

        per_step[step_name] = {
            "count": len(entries),
            "cost_mean": statistics.mean(costs),
            "cost_min": min(costs),
            "cost_max": max(costs),
            "cost_p50": percentile(sorted_costs, 50),
            "cost_p95": percentile(sorted_costs, 95),
            "input_tokens_mean": statistics.mean(in_toks),
            "output_tokens_mean": statistics.mean(out_toks),
            "duration_ms_mean": statistics.mean(durations),
            "max_iteration": max(iterations),
        }

    mean_run_cost = statistics.mean(run_totals) if run_totals else 0.0

    return {
        "per_step": per_step,
        "run_totals": run_totals,
        "mean_cost_per_run": mean_run_cost,
        "min_cost_per_run": min(run_totals) if run_totals else 0.0,
        "max_cost_per_run": max(run_totals) if run_totals else 0.0,
        "p95_cost_per_run": percentile(sorted(run_totals), 95),
        "total_session_cost": sum(run_totals),
        "projection_100_monthly": mean_run_cost * 100 * 30,
        "projection_1000_monthly": mean_run_cost * 1000 * 30,
        "projection_10000_monthly": mean_run_cost * 10000 * 30,
    }


def _get_step_model(
    runs: list[list[StepRecord]],
    step_name: str,
) -> str:
    for run in runs:
        for rec in run:
            if rec.step_name == step_name:
                return rec.model
    return ""


def _get_step_type(
    runs: list[list[StepRecord]],
    step_name: str,
) -> str:
    for run in runs:
        for rec in run:
            if rec.step_name == step_name:
                return rec.step_type
    return "llm"


class ProfileRunner:
    """Coordinate the full profiling pipeline."""

    def __init__(
        self,
        workflow_path: str,
        collector: str = "auto",
        auto_generate: int | None = None,
        single_input: str | None = None,
        inputs_file: str | None = None,
        explicit_inputs: list[str] | None = None,
        from_langfuse: bool = False,
        langfuse_last_n: int = 10,
        output_dir: str = ".pretia",
        cache_mode: str = "cold",
        progress_callback: Any | None = None,
        generator_model: str = "deepseek-v4-flash",
        corpus_path: str | None = None,
        entry_point: str | None = None,
        concurrency: int | None = None,
    ) -> None:
        self.workflow_path = workflow_path
        self.collector_name = collector
        self.auto_generate = auto_generate
        self.single_input = single_input
        self.inputs_file = inputs_file
        self.explicit_inputs = explicit_inputs
        self.from_langfuse = from_langfuse
        self.langfuse_last_n = langfuse_last_n
        self.output_dir = output_dir
        self.cache_mode = cache_mode
        self.progress_callback = progress_callback
        self.generator_model = generator_model
        self.corpus_path = corpus_path
        self.entry_point = entry_point
        self.concurrency = concurrency
        self.discovery_info: dict[str, str] | None = None

    def _load_workflow(self) -> tuple[Any, str, Any]:
        module = _load_workflow_module(self.workflow_path)
        prov: dict[str, str] = {}
        workflow = _find_workflow(
            module,
            entry_point=self.entry_point,
            workflow_path=self.workflow_path,
            provenance=prov,
        )
        self.discovery_info = prov if prov else None
        if workflow is None:
            from pretia.wrapper_hint import build_wrapper_snippet

            snippet = build_wrapper_snippet(module, [])
            raise EntrypointError(
                f"Could not find a workflow in '{self.workflow_path}'. "
                f"No variable named graph/workflow/agent/app, no ainvoke/invoke object, "
                f"and no async callable found.",
                rejected=[],
                wrapper_snippet=snippet,
                workflow_path=self.workflow_path,
            )
        system_prompt = _extract_system_prompt(module)
        return workflow, system_prompt, module

    def _select_collector(self, workflow: Any, module: Any = None) -> BaseCollector:
        if self.collector_name == "langgraph":
            from pretia.collectors.langgraph import LangGraphCollector

            return LangGraphCollector()

        if self.collector_name == "generic":
            if module is not None:
                instances = [v for v in vars(module).values() if isinstance(v, GenericCollector)]
                if instances:
                    return instances[0]
            return GenericCollector()

        if self.collector_name == "openai":
            from pretia.collectors.openai_agents import OpenAIAgentsCollector

            return OpenAIAgentsCollector()

        if self.collector_name == "qwen":
            from pretia.collectors.qwen_agent import QwenAgentCollector

            return QwenAgentCollector()

        if self.collector_name == "anthropic":
            from pretia.collectors.anthropic_sdk import AnthropicCollector

            return AnthropicCollector()

        if self.collector_name == "openai-sdk":
            from pretia.collectors.openai_sdk import OpenAISDKCollector

            return OpenAISDKCollector()

        uses_openai_raw = module is not None and _module_uses_sdk(module, "openai")
        uses_langchain_openai = module is not None and _module_uses_sdk(module, "langchain_openai")
        uses_langchain_anthropic = module is not None and _module_uses_sdk(
            module, "langchain_anthropic"
        )
        uses_anthropic_raw = module is not None and _module_uses_sdk(module, "anthropic")

        has_ainvoke = hasattr(workflow, "ainvoke")
        has_nodes = hasattr(workflow, "nodes")
        if has_ainvoke and has_nodes:
            if (
                uses_langchain_openai
                or uses_langchain_anthropic
                or not (uses_openai_raw or uses_anthropic_raw)
            ):
                from pretia.collectors.langgraph import LangGraphCollector

                return LangGraphCollector()

        if hasattr(workflow, "name") and hasattr(workflow, "instructions"):
            from pretia.collectors.openai_agents import OpenAIAgentsCollector

            return OpenAIAgentsCollector()

        if (
            hasattr(workflow, "run")
            and hasattr(workflow, "llm")
            and hasattr(workflow, "system_message")
        ):
            from pretia.collectors.qwen_agent import QwenAgentCollector

            return QwenAgentCollector()

        if uses_anthropic_raw and uses_openai_raw:
            from pretia.collectors.multi_sdk import MultiSDKCollector

            return MultiSDKCollector()

        if uses_anthropic_raw:
            from pretia.collectors.anthropic_sdk import AnthropicCollector

            return AnthropicCollector()

        if uses_openai_raw:
            from pretia.collectors.openai_sdk import OpenAISDKCollector

            return OpenAISDKCollector()

        # Indirect SDK detection: wrapper frameworks (pydantic_ai, instructor,
        # etc.) import openai/anthropic internally. If the workflow module uses
        # one of these frameworks, pick the SDK collector based on what's loaded.
        uses_wrapper = module is not None and any(
            _module_uses_sdk(module, fw) for fw in _WRAPPER_FRAMEWORKS
        )
        if uses_wrapper:
            if "anthropic" in sys.modules and "openai" in sys.modules:
                from pretia.collectors.multi_sdk import MultiSDKCollector

                return MultiSDKCollector()
            if "anthropic" in sys.modules:
                from pretia.collectors.anthropic_sdk import AnthropicCollector

                return AnthropicCollector()
            if "openai" in sys.modules:
                from pretia.collectors.openai_sdk import OpenAISDKCollector

                return OpenAISDKCollector()

        logger.info(
            "Using GenericCollector. Instrument your code with "
            "@collector.step() for per-step data."
        )
        return GenericCollector()

    @staticmethod
    def _detect_framework(collector: BaseCollector) -> str | None:
        """Derive a framework label from the selected collector."""
        mapping = {
            "LangGraphCollector": "langgraph",
            "OpenAIAgentsCollector": "openai-agents",
            "OpenAISDKCollector": "openai",
            "AnthropicCollector": "anthropic",
            "QwenAgentCollector": "qwen-agent",
            "GenericCollector": "generic",
        }
        return mapping.get(type(collector).__name__)

    async def _resolve_inputs(
        self,
        system_prompt: str,
    ) -> tuple[InputSelection, list[str]]:
        selection = select_input_mode(
            explicit_inputs=self.explicit_inputs,
            single_input=self.single_input,
            inputs_file=self.inputs_file,
            auto_generate=self.auto_generate,
            from_langfuse=self.from_langfuse,
            system_prompt=system_prompt or None,
        )

        if selection.mode == "auto-generate":
            n = self.auto_generate or 50

            context_parts: list[str] = []
            try:
                wf_source = Path(self.workflow_path).read_text(encoding="utf-8")
                wf_context = _extract_workflow_context(wf_source)
                if wf_context:
                    context_parts.append(wf_context)
            except OSError:
                pass

            if self.corpus_path:
                from pretia.inputs.corpus import load_corpus_context

                try:
                    corpus_ctx = load_corpus_context(self.corpus_path)
                    if corpus_ctx:
                        context_parts.append(
                            f"Documents in the user's knowledge base:\n{corpus_ctx}"
                        )
                except (FileNotFoundError, OSError) as exc:
                    logging.warning("Could not load corpus: %s", exc)

            inputs = await generate_inputs(
                system_prompt or "General purpose agent.",
                n=n,
                model=self.generator_model,
                additional_context="\n\n".join(context_parts),
            )
            return selection, inputs

        if selection.mode in ("single", "manual", "file"):
            return selection, selection.inputs

        if selection.mode == "langfuse":
            from pretia.inputs.importer import (
                create_langfuse_client,
                extract_inputs,
                fetch_traces,
            )

            client = create_langfuse_client()
            traces = fetch_traces(client, last_n=self.langfuse_last_n)
            inputs = extract_inputs(traces)
            return selection, inputs

        raise NotImplementedError(
            "Static estimation is not yet implemented. Provide an API key for input generation."
        )

    @staticmethod
    def _maybe_wrap_sync(workflow: Any, collector: BaseCollector) -> Any:
        """Wrap workflows so all collectors can call them as async callables."""
        from pretia.collectors.anthropic_sdk import AnthropicCollector
        from pretia.collectors.multi_sdk import MultiSDKCollector
        from pretia.collectors.openai_sdk import OpenAISDKCollector

        needs_callable = isinstance(
            collector,
            (GenericCollector, AnthropicCollector, OpenAISDKCollector, MultiSDKCollector),
        )
        if not needs_callable:
            return workflow

        if hasattr(workflow, "ainvoke"):
            graph = workflow
            input_key = _detect_graph_input_key(graph)

            async def _ainvoke_wrapper(inp: str) -> Any:
                if isinstance(inp, dict):
                    payload: Any = inp
                elif input_key == "messages":
                    from langchain_core.messages import HumanMessage

                    payload = {"messages": [HumanMessage(content=inp)]}
                else:
                    payload = {input_key: inp}
                config = {"configurable": {"thread_id": f"pretia-run-{uuid.uuid4()}"}}
                return await graph.ainvoke(payload, config=config)

            return _ainvoke_wrapper

        if inspect.isasyncgenfunction(workflow):
            gen_fn = workflow

            async def _drain_async_gen(inp: str) -> Any:
                return [chunk async for chunk in gen_fn(inp)]

            _drain_async_gen.__name__ = getattr(workflow, "__name__", "generator")
            logger.info("Wrapped async generator in drain shim for profiling.")
            return _drain_async_gen

        if inspect.isgeneratorfunction(workflow):
            gen_fn = workflow

            async def _drain_sync_gen(inp: str) -> Any:
                return await asyncio.to_thread(lambda: list(gen_fn(inp)))

            _drain_sync_gen.__name__ = getattr(workflow, "__name__", "generator")
            logger.info("Wrapped sync generator in drain shim for profiling.")
            return _drain_sync_gen

        if callable(workflow) and not asyncio.iscoroutinefunction(workflow):
            sync_fn = workflow

            async def _async_wrapper(inp: str) -> Any:
                return await asyncio.to_thread(sync_fn, inp)

            logger.info("Wrapped sync callable in async shim for profiling.")
            return _async_wrapper
        return workflow

    def _post_collect(
        self,
        runs: list[list[StepRecord]],
        *,
        workflow_name: str,
        workflow_hash: str,
        sample_size: int,
        input_mode: str,
        input_source: str,
        workflow_id: str,
        graph_steps: list[str] | None = None,
        framework: str | None = None,
        profiling_cost: float | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> ProfilingSession:
        """Shared post-collection pipeline: stats, patterns, projection, save."""
        cost_summary = _build_cost_summary(runs)

        for step_name in cost_summary["per_step"]:
            model = _get_step_model(runs, step_name)
            step_type = _get_step_type(runs, step_name)
            cost_summary["per_step"][step_name]["model"] = model
            cost_summary["per_step"][step_name]["step_type"] = step_type
            if model:
                try:
                    cost_summary["per_step"][step_name]["tier"] = model_tier(model)
                except (ValueError, KeyError):
                    cost_summary["per_step"][step_name]["tier"] = "unknown"
            else:
                cost_summary["per_step"][step_name]["tier"] = "tool"

        profiling_stats = compute_stats(runs)
        patterns = detect_patterns(runs, profiling_stats, graph_steps=graph_steps)
        projection = project(
            profiling_stats,
            patterns,
            runs=runs,
            input_source=input_source,
        )

        from pretia import __version__

        metadata: dict[str, Any] = {
            "cost_summary": cost_summary,
            "stats": profiling_stats.to_dict(),
            "patterns": [p.to_dict() for p in patterns],
            "projection": projection.to_dict(),
            "confidence": projection.confidence.to_dict(),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        session = ProfilingSession(
            workflow_name=workflow_name,
            workflow_hash=workflow_hash,
            profiled_at=datetime.now(UTC),
            sample_size=sample_size,
            input_mode=input_mode,
            runs=runs,
            metadata=metadata,
            workflow_id=workflow_id,
            run_id=str(uuid.uuid4()),
            framework=framework,
            pretia_version=__version__,
            profiling_cost=(
                profiling_cost
                if profiling_cost is not None
                else cost_summary["total_session_cost"]
            ),
        )

        store = ProfileStore(storage_dir=Path(self.output_dir))
        saved_path = store.save(session)
        session.metadata["saved_path"] = str(saved_path)

        return session

    @staticmethod
    def _validate_inputs(inputs: list[str], selection: InputSelection) -> list[str]:
        """Validate and coerce the resolved input list before profiling runs."""
        coerced: list[str] = []
        for i, item in enumerate(inputs):
            if isinstance(item, (str, dict)):
                coerced.append(item)  # type: ignore[arg-type]
            else:
                logger.warning(
                    "Input %d is %s (expected str), coercing via str().",
                    i,
                    type(item).__name__,
                )
                coerced.append(str(item))

        non_blank = [
            x for x in coerced if isinstance(x, dict) or (isinstance(x, str) and x.strip())
        ]
        if not non_blank:
            hints: dict[str, str] = {
                "file": ("Check that the inputs file contains at least one non-empty line."),
                "langfuse": ("No traces returned — check LANGFUSE_* env vars and --last N."),
                "auto-generate": (
                    "Input generation returned nothing — check the generator model "
                    "API key or pass --input explicitly."
                ),
            }
            hint = hints.get(selection.mode, 'Pass --input "..." to supply one directly.')
            raise ValueError(
                f"Input resolution produced 0 inputs (mode: {selection.mode}); "
                f"nothing to profile. {hint}"
            )
        return coerced

    async def run(self) -> ProfilingSession:
        """Execute the full profiling pipeline."""
        workflow, system_prompt, module = self._load_workflow()
        entry_name = getattr(workflow, "__name__", type(workflow).__name__)
        collector = self._select_collector(workflow, module=module)
        workflow = self._maybe_wrap_sync(workflow, collector)
        selection, inputs = await self._resolve_inputs(system_prompt)
        inputs = self._validate_inputs(inputs, selection)

        total = len(inputs)
        cb = self.progress_callback
        collector_name = type(collector).__name__

        def _preflight_cb(i: int, t: int, recs: list[StepRecord]) -> None:
            if cb is not None:
                cb(0, total, recs)

        def _batch_cb(i: int, t: int, recs: list[StepRecord]) -> None:
            if cb is not None:
                cb(i + 1, total, recs)

        first_runs = await collector.collect(
            workflow, inputs[:1], on_run_complete=_preflight_cb, concurrency=1
        )
        preflight_error = getattr(collector, "last_error", None)
        first = first_runs[0] if first_runs else []

        if not first:
            if isinstance(preflight_error, BaseException):
                exc_type = type(preflight_error).__name__
                remaining = total - 1
                raise ValueError(
                    f"First profiling run failed with {exc_type}: {preflight_error}\n"
                    f"No steps were captured, so the remaining {remaining} runs were "
                    f"skipped. Common fixes: verify your API key/credentials are set; "
                    f"confirm the entrypoint accepts a single input string; "
                    f"run with -v for the full traceback."
                ) from preflight_error
            else:
                remaining = total - 1
                raise ValueError(
                    f"First run executed successfully but captured 0 LLM steps, so the "
                    f"remaining {remaining} runs were skipped.\n"
                    f"Pretia profiled the entrypoint '{entry_name}' using "
                    f"{collector_name}"
                    + (
                        f" (selected via: {self.discovery_info['rule']})"
                        if self.discovery_info
                        else ""
                    )
                    + ".\n"
                    f"Likely causes:\n"
                    f"  - '{entry_name}' is not your agent's real entrypoint (e.g. a "
                    f"helper or tool function was auto-selected). Add a function named "
                    f"workflow(user_input) that runs your agent end-to-end, or pass "
                    f"--entry-point <name>.\n"
                    f"  - The wrong collector was selected. Try --collector langgraph "
                    f"| anthropic | openai-sdk | openai | qwen | generic.\n"
                    f"  - Your workflow returns processed text instead of the raw LLM "
                    f"response object."
                )

        rest = inputs[1:]
        rest_runs: list[list[StepRecord]] = []
        if rest:
            rest_runs = await collector.collect(
                workflow, rest, on_run_complete=_batch_cb, concurrency=self.concurrency
            )
        runs = first_runs + rest_runs
        last_error = getattr(collector, "last_error", None) or preflight_error

        valid_runs = [r for r in runs if r]
        failed_count = len(runs) - len(valid_runs)
        if failed_count:
            logger.warning(
                "%d/%d runs failed and were excluded from statistics.",
                failed_count,
                len(runs),
            )

        total_steps = sum(len(run) for run in valid_runs)
        if total_steps == 0:
            if isinstance(last_error, BaseException):
                exc_type = type(last_error).__name__
                raise ValueError(
                    f"Profiling captured 0 steps across {len(runs)} runs "
                    f"({failed_count} runs raised errors). "
                    f"Last error: {exc_type}: {last_error}\n"
                    f"Run with -v for the full traceback."
                ) from last_error
            raise ValueError(
                f"Profiling captured 0 steps across {len(runs)} runs "
                f"({failed_count} runs raised errors). No LLM calls were recorded. "
                "Common causes: workflow returned response.content instead of the raw "
                "response object, API key is invalid, or the wrong collector was "
                "auto-selected. Try: --collector langgraph (for LangGraph workflows), "
                "or return the raw LLM response object from your function."
            )

        from pretia.validation.data_checks import validate_profiling_data

        data_warnings = validate_profiling_data(valid_runs)
        for w in data_warnings:
            logger.warning(w)

        from pretia.graph.extractor import extract_step_names

        graph_steps = extract_step_names(workflow)

        try:
            workflow_src = Path(self.workflow_path).read_bytes()
        except OSError:
            workflow_src = b""

        session = self._post_collect(
            valid_runs,
            workflow_name=self.workflow_path,
            workflow_hash=hashlib.sha256(workflow_src).hexdigest()[:12],
            sample_size=len(inputs),
            input_mode=selection.mode,
            input_source=selection.mode,
            workflow_id=Path(self.workflow_path).stem,
            graph_steps=graph_steps,
            framework=self._detect_framework(collector),
            extra_metadata={
                "graph_steps": graph_steps,
                "failed_runs": failed_count,
            },
        )

        self._auto_diff_baseline(session)

        return session

    def _auto_diff_baseline(self, session: ProfilingSession) -> None:
        """Show a one-line diff summary if a baseline exists."""
        baseline_path = Path(self.output_dir) / "baseline.json"
        if not baseline_path.exists():
            return
        try:
            from pretia.ci.baseline import load_baseline
            from pretia.ci.diff import diff_baseline

            bl = load_baseline(str(baseline_path))
            result = diff_baseline(bl, session)
            session.metadata["baseline_diff_summary"] = result.summary
        except Exception:
            logger.debug("Auto-diff against baseline failed", exc_info=True)

    def run_sync(self) -> ProfilingSession:
        """Synchronous wrapper around `run()`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run())
        raise RuntimeError(
            "run_sync() cannot be called from an async context. "
            "Use 'await runner.run()' instead, or run from a synchronous entry point."
        )

    def analyze_langfuse(self) -> ProfilingSession:
        """Analyze Langfuse traces without re-executing the workflow."""
        from pretia.inputs.importer import (
            create_langfuse_client,
            fetch_traces,
            traces_to_step_records,
        )

        client = create_langfuse_client()
        traces = fetch_traces(client, last_n=self.langfuse_last_n)
        runs = traces_to_step_records(traces)

        return self._post_collect(
            runs,
            workflow_name=f"langfuse-import ({len(traces)} traces)",
            workflow_hash="langfuse",
            sample_size=len(traces),
            input_mode="langfuse-analyze",
            input_source="langfuse",
            workflow_id="langfuse-import",
            profiling_cost=0.0,
            extra_metadata={"langfuse_trace_count": len(traces)},
        )
