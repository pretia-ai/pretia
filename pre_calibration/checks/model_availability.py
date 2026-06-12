"""Check that all models used by backtesting workflows are accessible."""

from __future__ import annotations

import asyncio
import logging

from pre_calibration.pre_calibration import CheckResult

logger = logging.getLogger(__name__)

BACKTESTING_MODELS = [
    "anthropic/claude-haiku-4-5",
    "anthropic/claude-sonnet-4-6",
    "deepseek/deepseek-chat",
    "openai/gpt-4.1-nano",
    "openai/gpt-4.1",
    "gemini/gemini-2.5-flash",
]


async def check() -> CheckResult:
    """Attempt trivial API call to each model used by backtesting workflows."""
    details: dict = {}
    failed = []
    try:
        import litellm
    except ImportError:
        return CheckResult(
            name="model_availability",
            status="WARN",
            details={"error": "litellm not installed"},
            blocking=True,
        )

    for model in BACKTESTING_MODELS:
        try:
            resp = await asyncio.wait_for(
                litellm.acompletion(
                    model=model,
                    messages=[{"role": "user", "content": "Say hello"}],
                    max_tokens=5,
                ),
                timeout=10.0,
            )
            details[model] = "available"
        except Exception as e:
            details[model] = f"unavailable: {e!s}"
            failed.append(model)

    status = "FAIL" if failed else "PASS"
    return CheckResult(
        name="model_availability",
        status=status,
        details=details,
        blocking=True,
    )
