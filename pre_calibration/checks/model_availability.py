"""Check that all models used by backtesting workflows are accessible."""

from __future__ import annotations

import asyncio
import logging

from pre_calibration.pre_calibration import CheckResult

logger = logging.getLogger(__name__)

BACKTESTING_MODELS = [
    "claude-haiku-3",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
    "gpt-4o-mini",
    "gpt-4o",
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
