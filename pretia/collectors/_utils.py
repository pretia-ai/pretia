"""Shared utilities for SDK collectors."""

from __future__ import annotations

import inspect

_SKIP_MODULES = frozenset(
    {
        "pretia",
        "anthropic",
        "openai",
        "httpx",
        "httpcore",
        "asyncio",
        "contextlib",
    }
)


def get_caller_name(default: str = "llm_call") -> str:
    """Walk the call stack to find the nearest user-code function name.

    Skips frames from SDK internals, Pretia, and standard library async machinery.
    """
    for frame_info in inspect.stack():
        module = frame_info.frame.f_globals.get("__name__", "") or ""
        top_package = module.split(".")[0]
        if top_package in _SKIP_MODULES:
            continue
        if frame_info.filename.startswith("<"):
            continue
        name = frame_info.function
        if name in ("__call__", "<module>", "collect", "wrapper", "async_wrapper"):
            continue
        return name
    return default
