"""Verify projection engine configuration."""

from __future__ import annotations

import inspect

from pre_calibration.pre_calibration import CheckResult


async def check() -> CheckResult:
    """Inspect Monte Carlo defaults and report actual values."""
    try:
        from agentcost.projection.montecarlo import simulate

        sig = inspect.signature(simulate)
        params = {
            name: p.default
            for name, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty
        }
        details = {
            "n_simulations": params.get("n_simulations", "not found"),
            "seed": params.get("seed", "not found"),
        }
        return CheckResult(
            name="engine_config",
            status="PASS",
            details=details,
            blocking=True,
        )
    except Exception as e:
        return CheckResult(
            name="engine_config",
            status="FAIL",
            details={"error": str(e)},
            blocking=True,
        )
