#!/usr/bin/env python
"""Pilot calibration runner for the AgentCost backtesting suite.

Executes 10 pilot runs per workflow, runs 11 calibration checks (7 infrastructure
+ 4 cost plausibility), generates P1-P6 pilot plots, and produces a pilot report
JSON. Gates on a passing pre-calibration report.

Usage::

    python tests/backtesting/run_pilot.py --all
    python tests/backtesting/run_pilot.py --workflow W1
    python tests/backtesting/run_pilot.py --all --dry-run
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from agentcost.pricing.tables import calculate_cost


def _load_dotenv() -> None:
    """Load .env file from project root if it exists."""
    env_path = Path(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value

logger = logging.getLogger(__name__)

_PILOT_N = 10

_PILOT_ORDER = [
    ["W11", "W12"],
    ["W1", "W9"],
    ["W13"],
    ["W4", "W14", "W16"],
    ["W5"],
    ["W17"],
    ["W18", "W19"],
    ["W2"],
    ["W15"],
]


def _flatten_pilot_order() -> list[str]:
    """Return workflow IDs in cheapest-first pilot execution order."""
    return [wf for group in _PILOT_ORDER for wf in group]


def _load_pre_calibration_report(path: Path) -> dict[str, Any] | None:
    """Load and validate the pre-calibration report JSON."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read pre-calibration report: %s", exc)
        return None


def _check_pre_calibration_gate(report_path: Path) -> bool:
    """Verify pre-calibration passed. Returns False with error message if not."""
    report = _load_pre_calibration_report(report_path)
    if report is None:
        click.echo(
            f"Pre-calibration report not found at {report_path}.\n"
            "Run pre-calibration first:\n"
            "  python -m pre_calibration.pre_calibration --output reports/pre_calibration.json",
            err=True,
        )
        return False

    if not report.get("proceed_to_pilot", False):
        click.echo("Pre-calibration report has blocking failures:", err=True)
        for failure in report.get("blocking_failures", []):
            click.echo(f"  BLOCKED: {failure}", err=True)
        click.echo(
            "Fix pre-calibration issues before running the pilot.", err=True
        )
        return False

    click.echo("Pre-calibration gate: PASSED")
    return True


def _config_for_workflow(workflow_id: str) -> Any:
    """Find the BacktestConfig for a workflow ID."""
    from tests.backtesting.configs import BACKTESTING_CONFIGS

    wf_upper = workflow_id.upper()
    for cfg in BACKTESTING_CONFIGS:
        cfg_wf = cfg.name.split("-")[0].upper()
        if cfg_wf == wf_upper:
            return cfg
    return None


def _generate_pilot_inputs(
    workflow_id: str,
    n: int,
    seed: int = 42,
    dry_run: bool = True,
    results_dir: str = "tests/backtesting/results/pilot",
) -> list[dict[str, Any]]:
    """Generate profiling-distribution inputs using the per-workflow generator."""
    try:
        wf_num = workflow_id.upper().replace("W", "")
        import importlib
        import pkgutil

        import inputs.generators as gen_pkg

        for info in pkgutil.iter_modules(gen_pkg.__path__):
            if info.name.startswith(f"w{wf_num.zfill(2)}_"):
                mod = importlib.import_module(f"inputs.generators.{info.name}")
                from inputs.generators._base import BaseInputGenerator

                for attr_name in dir(mod):
                    attr = getattr(mod, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, BaseInputGenerator)
                        and attr is not BaseInputGenerator
                    ):
                        gen = attr(seed=seed)
                        if hasattr(gen, "dry_run"):
                            gen.dry_run = dry_run
                        if hasattr(gen, "_image_output_dir"):
                            gen._image_output_dir = (
                                f"{results_dir}/images/{workflow_id.lower()}"
                            )
                        batch = gen.generate_batch("profiling", n)
                        return [inp.to_dict() for inp in batch]
    except Exception as exc:
        logger.warning("Input generator not available for %s: %s", workflow_id, exc)

    # Fallback: simple synthetic inputs
    return [
        {"input": f"Pilot input {i} for {workflow_id}", "tier": "easy", "_dry_run": True}
        for i in range(n)
    ]


def _compute_run_costs(all_records: list[list[Any]]) -> list[float]:
    """Compute total cost per run from StepRecords."""
    costs = []
    for run in all_records:
        total = 0.0
        for r in run:
            try:
                total += calculate_cost(r.model, r.input_tokens, r.output_tokens)
            except Exception:
                pass
        costs.append(total)
    return costs


async def _run_pilot_workflow(
    workflow_id: str,
    config: Any,
    results_dir: Path,
    dry_run: bool,
    parallel: int,
) -> dict[str, Any]:
    """Execute pilot for a single workflow and run checks."""
    from bt_agents.harness.run_workflow import (
        load_prompts,
        run_batch,
        save_results,
    )
    from tests.backtesting.pilot_checks import run_pilot_checks

    result: dict[str, Any] = {
        "workflow_id": workflow_id,
        "status": "PENDING",
        "infrastructure_checks": {},
        "cost_plausibility": {},
        "per_run_costs": [],
        "blocked": False,
        "error": None,
    }

    # Load prompts
    try:
        prompts = load_prompts(workflow_id)
    except Exception as exc:
        logger.warning("No prompts for %s: %s", workflow_id, exc)
        prompts = {}

    # Generate inputs
    inputs = _generate_pilot_inputs(
        workflow_id, _PILOT_N,
        dry_run=dry_run, results_dir=str(results_dir),
    )
    if dry_run:
        for inp in inputs:
            if isinstance(inp, dict):
                inp["_dry_run"] = True
            else:
                inp = {"input": str(inp), "_dry_run": True}

    # Execute
    try:
        input_dicts = []
        for inp in inputs:
            if isinstance(inp, dict) and "input_data" in inp:
                d = dict(inp["input_data"])
                d["tier"] = inp.get("tier", "easy")
                d["structural_descriptor"] = inp.get("structural_descriptor", {})
                if dry_run:
                    d["_dry_run"] = True
                input_dicts.append(d)
            elif isinstance(inp, dict):
                input_dicts.append(inp)
            else:
                input_dicts.append({"input": str(inp)})

        all_records = await run_batch(workflow_id, input_dicts, prompts, parallel)
    except Exception as exc:
        result["status"] = "ERROR"
        result["error"] = str(exc)
        logger.error("Pilot execution failed for %s: %s", workflow_id, exc)
        return result

    # Save raw results
    try:
        save_results(
            workflow_id,
            all_records,
            str(results_dir),
            inputs=input_dicts,
            prompts=prompts,
            backtest_profile="pilot",
        )
    except Exception as exc:
        logger.warning("Failed to save pilot results for %s: %s", workflow_id, exc)

    # Compute costs
    run_costs = _compute_run_costs(all_records)
    result["per_run_costs"] = run_costs
    result["total_cost"] = sum(run_costs)

    # Run pilot checks
    try:
        check_results = run_pilot_checks(
            workflow_id, all_records, input_dicts, config
        )

        layer1_results = {}
        layer2_results = {}
        blocked = False

        for cr in check_results:
            entry = {"status": cr.status, "details": cr.details}
            if cr.layer == 1:
                layer1_results[cr.name] = entry
                if cr.status == "FAIL" and cr.blocking:
                    blocked = True
            else:
                layer2_results[cr.name] = entry

        result["infrastructure_checks"] = layer1_results
        result["cost_plausibility"] = layer2_results
        result["blocked"] = blocked
        result["status"] = "BLOCKED" if blocked else "PASS"

    except Exception as exc:
        logger.error("Pilot checks failed for %s: %s", workflow_id, exc)
        result["status"] = "ERROR"
        result["error"] = str(exc)

    return result


@click.command()
@click.option("--workflow", type=str, default=None, help="Single workflow (e.g., 'W1').")
@click.option("--all", "run_all", is_flag=True, default=False, help="Run all 14 workflows.")
@click.option("--dry-run", is_flag=True, default=False, help="Validate without API calls.")
@click.option(
    "--results-dir",
    type=click.Path(),
    default="tests/backtesting/results/pilot",
    help="Output directory.",
)
@click.option(
    "--pre-calibration-report",
    type=click.Path(),
    default="reports/pre_calibration.json",
    help="Path to pre-calibration report JSON.",
)
@click.option("--parallel", type=int, default=1, help="Concurrent runs per workflow.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug logging.")
def main(
    workflow: str | None,
    run_all: bool,
    dry_run: bool,
    results_dir: str,
    pre_calibration_report: str,
    parallel: int,
    verbose: bool,
) -> None:
    """Run pilot calibration for the backtesting suite."""
    _load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not run_all and not workflow:
        click.echo("Specify --workflow W1 or --all. Use --help for details.", err=True)
        sys.exit(1)

    # Gate on pre-calibration
    if not _check_pre_calibration_gate(Path(pre_calibration_report)):
        sys.exit(1)

    # Determine workflow list
    if run_all:
        workflow_ids = _flatten_pilot_order()
    else:
        workflow_ids = [workflow.upper()]

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Execute pilots — grouped by provider to run non-conflicting workflows concurrently
    from tests.backtesting.concurrency import (
        build_concurrent_groups,
        get_parallel_for_workflow,
    )

    per_workflow: dict[str, Any] = {}
    blocked_workflows: list[str] = []
    total_cost = 0.0

    groups = build_concurrent_groups(workflow_ids)

    async def _run_group(group: list[str]) -> list[tuple[str, dict[str, Any]]]:
        tasks = []
        for wf_id in group:
            cfg = _config_for_workflow(wf_id)
            if cfg is None:
                continue
            wf_parallel = parallel if parallel > 1 else get_parallel_for_workflow(wf_id)
            tasks.append((wf_id, cfg, wf_parallel))

        async def _one(wid: str, cfg: Any, par: int) -> tuple[str, dict[str, Any]]:
            return wid, await _run_pilot_workflow(wid, cfg, out_dir, dry_run, par)

        results = await asyncio.gather(
            *[_one(wid, cfg, par) for wid, cfg, par in tasks],
            return_exceptions=True,
        )
        out = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Pilot group error: %s", r)
            else:
                out.append(r)
        return out

    for group_idx, group in enumerate(groups):
        group_names = ", ".join(group)
        click.echo(f"  Group {group_idx + 1}/{len(groups)}: [{group_names}]")
        group_results = asyncio.run(_run_group(group))

        for wf_id, result in group_results:
            per_workflow[wf_id] = result
            total_cost += result.get("total_cost", 0.0)

            if result.get("blocked"):
                blocked_workflows.append(wf_id)
                click.echo(f"    {wf_id}: BLOCKED")
            elif result.get("status") == "ERROR":
                click.echo(f"    {wf_id}: ERROR — {result.get('error', 'unknown')}")
            else:
                n_runs = len(result.get("per_run_costs", []))
                cost = result.get("total_cost", 0)
                click.echo(f"    {wf_id}: PASS — {n_runs} runs, ${cost:.4f}")

    # Generate pilot visualizations
    plots_dir = out_dir / "plots"
    try:
        from visualization.pilot.visualize_pilot import generate_all_pilot_visuals

        paths = generate_all_pilot_visuals(out_dir, plots_dir)
        click.echo(f"  Pilot plots: {len(paths)} files in {plots_dir}")
    except Exception as exc:
        click.echo(f"  Pilot plots failed: {exc}", err=True)

    # Write pilot report
    report = {
        "pilot_date": datetime.now(UTC).isoformat(),
        "pre_calibration_status": "PASSED",
        "per_workflow": per_workflow,
        "blocked_workflows": blocked_workflows,
        "total_pilot_cost": total_cost,
        "proceed_to_backtest": len(blocked_workflows) == 0,
        "dry_run": dry_run,
    }

    report_path = out_dir / "pilot_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    click.echo(f"\nPilot report: {report_path}")

    if blocked_workflows:
        click.echo(
            f"\n{len(blocked_workflows)} workflow(s) blocked: "
            f"{', '.join(blocked_workflows)}"
        )
        click.echo("Fix infrastructure issues before running the backtest.")
        sys.exit(1)

    click.echo(f"\nAll workflows passed. Total pilot cost: ${total_cost:.4f}")
    click.echo("Ready for backtest: python tests/backtesting/run_backtest.py --all")


if __name__ == "__main__":
    main()
