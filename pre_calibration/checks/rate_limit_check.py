"""Estimate API call volume and warn about rate limits."""

from __future__ import annotations

from pre_calibration.pre_calibration import CheckResult

ESTIMATED_CALLS_PER_WORKFLOW = {
    "W1": 200,
    "W2": 2000,
    "W4": 1500,
    "W5": 270,
    "W9": 250,
    "W11": 250,
    "W12": 250,
    "W13": 700,
    "W14": 700,
    "W15": 2000,
    "W16": 1200,
    "W17": 2000,
    "W18": 550,
    "W19": 4400,
}


async def check() -> CheckResult:
    """Estimate total API calls for the backtesting pilot and full run."""
    total = sum(ESTIMATED_CALLS_PER_WORKFLOW.values())
    by_provider = {
        "anthropic": sum(
            v
            for k, v in ESTIMATED_CALLS_PER_WORKFLOW.items()
            if k in {"W1", "W2", "W5", "W13", "W14", "W16", "W17"}
        ),
        "openai": sum(
            v
            for k, v in ESTIMATED_CALLS_PER_WORKFLOW.items()
            if k in {"W9", "W14", "W15", "W17"}
        ),
        "deepseek": sum(
            v
            for k, v in ESTIMATED_CALLS_PER_WORKFLOW.items()
            if k in {"W4", "W12", "W15", "W18", "W19"}
        ),
        "qwen": sum(
            v
            for k, v in ESTIMATED_CALLS_PER_WORKFLOW.items()
            if k in {"W4", "W11"}
        ),
    }
    return CheckResult(
        name="rate_limit_headroom",
        status="WARN",
        details={"total_estimated_calls": total, "by_provider": by_provider},
        blocking=False,
    )
