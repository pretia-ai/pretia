"""Combinatorial contract test for _find_workflow.

For any synthetic module built from the component pool, _find_workflow must:
1. Return a valid single-input callable / invoke-object, or None, or
   raise exactly AmbiguousEntrypointError / EntrypointError.
2. Never raise any other exception type.
3. Never return a non-callable, non-invoke object.
"""

from __future__ import annotations

import functools
import itertools
import types

import pytest

from pretia.runner import (
    AmbiguousEntrypointError,
    EntrypointError,
    _callable_accepts_single_input,
    _find_workflow,
)


def _make_module(**attrs: object) -> types.ModuleType:
    mod = types.ModuleType("_inv_test_module")
    mod.__name__ = "_inv_test_module"
    for name, value in attrs.items():
        setattr(mod, name, value)
    return mod


def _zero_arg_fn():
    pass


def _one_arg_fn(x):
    return x


def _two_arg_fn(a, b):
    return a


async def _async_one_arg(x):
    return x


async def _async_zero_arg():
    pass


def _kwarg_only(*, key):
    return key


class _ClassWithRun:
    __module__ = "_inv_test_module"

    def run(self, inp):
        return inp


class _ClassNeedingArgs:
    __module__ = "_inv_test_module"

    def __init__(self, key):
        self.key = key

    def run(self, inp):
        return inp


class _BareClass:
    __module__ = "_inv_test_module"
    pass


class _CallableInstance:
    def __call__(self, inp):
        return inp


class _ASGIInstance:
    def __call__(self, scope, receive, send):
        pass


class _InvokeObject:
    def ainvoke(self, x):
        return x


class _FakeLLM:
    pass


_FakeLLM.__name__ = "BaseChatModel"


def _tool_stub(system_name):
    return system_name


def _sync_gen(msg):
    yield msg


async def _async_gen(msg):
    yield msg


class _BaseCommand:
    pass


_BaseCommand.__name__ = "BaseCommand"


class _FakeCliCommand(_BaseCommand):
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    def callback(inp):
        return inp


_one_arg_fn.__module__ = "_inv_test_module"
_two_arg_fn.__module__ = "_inv_test_module"
_zero_arg_fn.__module__ = "_inv_test_module"
_async_one_arg.__module__ = "_inv_test_module"
_async_zero_arg.__module__ = "_inv_test_module"
_kwarg_only.__module__ = "_inv_test_module"

COMPONENT_POOL: list[tuple[str, object]] = [
    ("zero_fn", _zero_arg_fn),
    ("one_fn", _one_arg_fn),
    ("two_fn", _two_arg_fn),
    ("async_one", _async_one_arg),
    ("async_zero", _async_zero_arg),
    ("kwarg_fn", _kwarg_only),
    ("AgentCls", _ClassWithRun),
    ("NeedArgsCls", _ClassNeedingArgs),
    ("EmptyCls", _BareClass),
    ("callable_inst", _CallableInstance()),
    ("asgi_inst", _ASGIInstance()),
    ("invoke_obj", _InvokeObject()),
    ("fake_llm", _FakeLLM()),
    ("TOOL_DISPATCH", {"tool_stub": _tool_stub}),
    ("TOOLS", [{"type": "function", "function": {"name": "tool_stub"}}]),
    ("VERSION", "1.0"),
    ("COUNT", 42),
    ("sync_gen", _sync_gen),
    ("async_gen", _async_gen),
    ("cli_cmd", _FakeCliCommand()),
]


def _make_wrapped_fn():
    def _multi(a, b, c):
        return a

    @functools.wraps(_multi)
    def _single(x):
        return _multi(x, None, None)

    _single.__module__ = "_inv_test_module"
    return _single


COMPONENT_POOL.append(("wrapped_fn", _make_wrapped_fn()))


def _build_modules():
    """Generate synthetic modules: singles, all pairs, sampled triples."""
    modules = []

    for name, obj in COMPONENT_POOL:
        modules.append((_make_module(**{name: obj}), f"single:{name}"))

    for combo in itertools.combinations(COMPONENT_POOL, 2):
        attrs = {name: obj for name, obj in combo}
        label = "+".join(name for name, _ in combo)
        modules.append((_make_module(**attrs), f"pair:{label}"))

    triples = list(itertools.combinations(range(len(COMPONENT_POOL)), 3))
    step = max(1, len(triples) // 200)
    for idx in range(0, len(triples), step):
        i, j, k = triples[idx]
        combo = [COMPONENT_POOL[i], COMPONENT_POOL[j], COMPONENT_POOL[k]]
        attrs = {name: obj for name, obj in combo}
        label = "+".join(name for name, _ in combo)
        modules.append((_make_module(**attrs), f"triple:{label}"))

    return modules


_SYNTHETIC_MODULES = _build_modules()


@pytest.mark.parametrize(
    "module,label",
    _SYNTHETIC_MODULES,
    ids=[label for _, label in _SYNTHETIC_MODULES],
)
class TestDiscoveryInvariants:
    def test_contract(self, module, label):
        """_find_workflow obeys the three-outcome contract."""
        try:
            prov: dict[str, str] = {}
            result = _find_workflow(module, provenance=prov)
        except AmbiguousEntrypointError as exc:
            assert len(exc.candidates) >= 2, f"{label}: ambiguous with < 2 candidates"
            return
        except EntrypointError as exc:
            assert isinstance(exc.wrapper_snippet, str), f"{label}: wrapper_snippet not a string"
            return
        except Exception as exc:
            pytest.fail(f"{label}: unexpected {type(exc).__name__}: {exc}")

        if result is None:
            return

        has_invoke = hasattr(result, "ainvoke") or hasattr(result, "invoke")
        is_valid_callable = callable(result) and _callable_accepts_single_input(result)
        assert has_invoke or is_valid_callable, (
            f"{label}: returned non-callable, non-invoke object: {result}"
        )

        if prov:
            assert "rule" in prov, f"{label}: provenance missing 'rule'"
            assert "entrypoint" in prov, f"{label}: provenance missing 'entrypoint'"
