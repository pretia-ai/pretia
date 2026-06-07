"""Pre-calibration checks that must pass before the backtesting pilot."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Result of a single pre-calibration check."""

    name: str
    status: str  # "PASS" | "WARN" | "FAIL"
    details: dict[str, Any]
    blocking: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "details": self.details,
            "blocking": self.blocking,
        }


@dataclass(frozen=True, slots=True)
class PreCalibrationReport:
    """Full pre-calibration report."""

    timestamp: str
    checks: dict[str, CheckResult]
    blocking_failures: list[str]
    warnings: list[str]
    proceed_to_pilot: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "checks": {k: v.to_dict() for k, v in self.checks.items()},
            "blocking_failures": self.blocking_failures,
            "warnings": self.warnings,
            "proceed_to_pilot": self.proceed_to_pilot,
        }


async def run_pre_calibration(
    prompts_dir: Path = Path("prompts"),
    inputs_dir: Path = Path("inputs/generated"),
    pdfs_dir: Path = Path("pdfs/generated"),
    output: Path | None = None,
) -> PreCalibrationReport:
    """Run all 9 pre-calibration checks."""
    from pre_calibration.checks.engine_config import check as check_engine
    from pre_calibration.checks.input_inventory import check as check_inputs
    from pre_calibration.checks.pdf_inventory import check as check_pdfs
    from pre_calibration.checks.prompt_inventory import check as check_prompts
    from pre_calibration.checks.schema_compatibility import check as check_schema
    from pre_calibration.checks.workspace_check import check as check_workspace

    checks: dict[str, CheckResult] = {}

    # Group A: Engine Readiness
    # Checks 1 & 2 (model_availability, pricing_consistency) require litellm
    # and network access — run them but gracefully handle import errors
    try:
        from pre_calibration.checks.model_availability import check as check_models

        checks["model_availability"] = await check_models()
    except ImportError:
        checks["model_availability"] = CheckResult(
            name="model_availability",
            status="WARN",
            details={"error": "litellm not installed — skipping model availability check"},
            blocking=True,
        )

    try:
        from pre_calibration.checks.pricing_consistency import check as check_pricing

        checks["pricing_consistency"] = await check_pricing()
    except ImportError:
        checks["pricing_consistency"] = CheckResult(
            name="pricing_consistency",
            status="WARN",
            details={"error": "litellm not installed — skipping pricing consistency check"},
            blocking=True,
        )

    checks["collector_schema"] = await check_schema()
    checks["engine_config"] = await check_engine()

    # Group B: Backtest-Specific
    checks["prompt_inventory"] = await check_prompts(prompts_dir=prompts_dir)
    checks["input_inventory"] = await check_inputs(inputs_dir=inputs_dir)
    checks["pdf_inventory"] = await check_pdfs(pdfs_dir=pdfs_dir)

    try:
        from pre_calibration.checks.rate_limit_check import check as check_rates

        checks["rate_limit_headroom"] = await check_rates()
    except ImportError:
        checks["rate_limit_headroom"] = CheckResult(
            name="rate_limit_headroom",
            status="WARN",
            details={"error": "skipped — no rate limit API available"},
            blocking=False,
        )

    checks["workspace"] = await check_workspace()

    blocking_failures = [
        name for name, result in checks.items() if result.status == "FAIL" and result.blocking
    ]
    warnings = [name for name, result in checks.items() if result.status == "WARN"]

    report = PreCalibrationReport(
        timestamp=datetime.now(UTC).isoformat(),
        checks=checks,
        blocking_failures=blocking_failures,
        warnings=warnings,
        proceed_to_pilot=len(blocking_failures) == 0,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report.to_dict(), indent=2))
        logger.info("Pre-calibration report saved to %s", output)

    return report


@click.command()
@click.option("--prompts-dir", type=click.Path(exists=False), default="prompts/")
@click.option("--inputs-dir", type=click.Path(exists=False), default="inputs/generated/")
@click.option("--pdfs-dir", type=click.Path(exists=False), default="pdfs/generated/")
@click.option("--output", type=click.Path(), default="reports/pre_calibration.json")
def main(prompts_dir: str, inputs_dir: str, pdfs_dir: str, output: str) -> None:
    """Run pre-calibration checks."""
    report = asyncio.run(
        run_pre_calibration(
            prompts_dir=Path(prompts_dir),
            inputs_dir=Path(inputs_dir),
            pdfs_dir=Path(pdfs_dir),
            output=Path(output),
        )
    )
    status = "PROCEED" if report.proceed_to_pilot else "BLOCKED"
    click.echo(f"Pre-calibration: {status}")
    if report.blocking_failures:
        click.echo(f"  Blocking failures: {', '.join(report.blocking_failures)}")
    if report.warnings:
        click.echo(f"  Warnings: {', '.join(report.warnings)}")


if __name__ == "__main__":
    main()
