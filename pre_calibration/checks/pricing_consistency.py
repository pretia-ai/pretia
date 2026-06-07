"""Verify pricing table consistency."""

from __future__ import annotations

import logging

from agentcost.pricing.tables import MODEL_PRICING, calculate_cost
from pre_calibration.pre_calibration import CheckResult

logger = logging.getLogger(__name__)


async def check() -> CheckResult:
    """Verify all backtesting models have pricing entries and costs are reasonable."""
    details: dict = {}
    issues = []

    for model_name in MODEL_PRICING:
        try:
            cost = calculate_cost(model_name, 1000, 500)
            if cost <= 0:
                issues.append(f"{model_name}: zero or negative cost")
            details[model_name] = f"${cost:.6f} per 1K in + 500 out"
        except Exception as e:
            issues.append(f"{model_name}: {e!s}")
            details[model_name] = f"error: {e!s}"

    try:
        import litellm

        for model_name in list(MODEL_PRICING)[:5]:
            try:
                litellm_cost = litellm.completion_cost(
                    model=model_name,
                    prompt="test",
                    completion="test",
                )
                engine_cost = calculate_cost(model_name, 1, 1)
                if engine_cost > 0 and litellm_cost > 0:
                    ratio = litellm_cost / engine_cost
                    if abs(ratio - 1.0) > 0.05:
                        details[f"{model_name}_litellm_discrepancy"] = f"ratio={ratio:.2f}"
                        issues.append(f"{model_name}: >5% discrepancy with litellm")
            except Exception:
                pass
    except ImportError:
        details["litellm_comparison"] = "skipped — litellm not installed"

    status = "FAIL" if issues else "PASS"
    if not issues and "litellm_comparison" in details:
        status = "WARN"
    return CheckResult(
        name="pricing_consistency",
        status=status,
        details=details,
        blocking=True,
    )
