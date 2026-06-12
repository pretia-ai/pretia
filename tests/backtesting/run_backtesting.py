#!/usr/bin/env python
"""AgentCost Backtesting Runner.

Runs the full backtesting protocol across 10 test workflows.
This script makes real API calls. Cost estimates:
  Phase 1 (synthetic, 20+100 runs x 10 workflows): ~$8
  Phase 2 (ground truth, 500 runs x 10 workflows): ~$750-950
  Phase 3 (scoring only, no API calls): free

Usage:
    python run_backtesting.py --phase 1                # Synthetic profiles (~$8)
    python run_backtesting.py --phase 2                # Ground truth (~$750-950)
    python run_backtesting.py --phase 3                # Score existing results
    python run_backtesting.py --all                    # All phases
    python run_backtesting.py --workflow W1 --phase 1  # Single workflow
    python run_backtesting.py --phase 2 --resume       # Resume interrupted run
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

import click

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"
INPUTS_DIR = Path(__file__).parent / "inputs"

_SAMPLE_SIZES = {
    "synth20": 20,
    "synth100": 100,
    "real500": 500,
}


def _result_path(workflow_name: str, profile_key: str) -> Path:
    return RESULTS_DIR / f"{workflow_name}_{profile_key}.json"


def _profile_exists(workflow_name: str, profile_key: str) -> bool:
    return _result_path(workflow_name, profile_key).exists()


def _load_inputs(workflow_name: str, n: int, variant: str = "realistic") -> list[str]:
    """Load the first n inputs from the workflow's JSONL file."""
    prefix = workflow_name.split("-")[0].lower()
    idx = prefix.replace("w", "")
    fname = f"w{idx.zfill(2)}_{variant}.jsonl"
    path = INPUTS_DIR / fname
    if not path.exists():
        # Try alternate naming (e.g., w13_inputs.jsonl)
        alt = INPUTS_DIR / f"w{idx.zfill(2)}_inputs.jsonl"
        if alt.exists():
            path = alt
        else:
            click.echo(f"  Input file not found: {path}", err=True)
            return []
    inputs: list[str] = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            try:
                data = json.loads(line.strip())
                inputs.append(data.get("input", ""))
            except json.JSONDecodeError:
                continue
    return inputs


def _run_profile(
    workflow_path: str,
    inputs: list[str],
    output_path: Path,
) -> None:
    """Profile a workflow with the given inputs and save the session."""
    from agentcost.runner import ProfileRunner

    runner = ProfileRunner(
        workflow_path=workflow_path,
        auto_generate=len(inputs),
        output_dir=str(output_path.parent),
    )
    session = runner.run_sync()
    output_path.write_text(json.dumps(session.to_dict(), indent=2))
    click.echo(f"  Saved: {output_path}")


def _run_phase(
    phase: int,
    configs: list,
    workflow_filter: str | None,
    resume: bool,
) -> None:
    """Execute one backtesting phase."""
    from tests.backtesting.configs import BACKTESTING_CONFIGS

    if not configs:
        configs = BACKTESTING_CONFIGS

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if workflow_filter:
        configs = [c for c in configs if workflow_filter.upper() in c.name.upper()]
        if not configs:
            click.echo(f"No workflow matching '{workflow_filter}'.", err=True)
            sys.exit(1)

    if phase == 1:
        _run_phase_1(configs, resume)
    elif phase == 2:
        _run_phase_2(configs, resume)
    elif phase == 3:
        _run_phase_3(configs)
    elif phase == 4:
        _run_phase_4()
    else:
        click.echo(f"Unknown phase: {phase}", err=True)
        sys.exit(1)


def _run_phase_1(configs: list, resume: bool) -> None:
    """Phase 1: Profile each workflow at 20 and 100 samples."""
    total_cost_est = len(configs) * 2 * 0.4
    click.echo(f"Phase 1: Synthetic profiles for {len(configs)} workflows.")
    click.echo(f"Estimated cost: ~${total_cost_est:.0f}")
    if not click.confirm("Proceed?"):
        click.echo("Cancelled.")
        return

    for cfg in configs:
        for key, n in [("synth20", 20), ("synth100", 100)]:
            out_path = _result_path(cfg.name, key)
            if resume and out_path.exists():
                click.echo(f"  Skipping {cfg.name}/{key} (exists)")
                continue

            click.echo(f"  Profiling {cfg.name} with {n} samples...")
            inputs = _load_inputs(cfg.name, n)
            if not inputs:
                click.echo(f"  No inputs for {cfg.name}, skipping.", err=True)
                continue
            try:
                _run_profile(cfg.workflow_path, inputs, out_path)
            except Exception as exc:
                click.echo(f"  Error profiling {cfg.name}/{key}: {exc}", err=True)
                logger.debug("Profile error", exc_info=True)


def _run_phase_2(configs: list, resume: bool) -> None:
    """Phase 2: Ground truth profiles at 500 samples."""
    total_cost_est = sum(
        (c.expected_cost_range[0] + c.expected_cost_range[1]) / 2 * 500 for c in configs
    )
    click.echo(f"Phase 2: Ground truth profiles for {len(configs)} workflows.")
    click.echo(f"Estimated cost: ~${total_cost_est:.0f}")
    click.echo("This is expensive. Make sure you've reviewed Phase 1 results first.")
    if not click.confirm("Proceed with ground truth profiling?"):
        click.echo("Cancelled.")
        return

    for cfg in configs:
        out_path = _result_path(cfg.name, "real500")
        if resume and out_path.exists():
            click.echo(f"  Skipping {cfg.name}/real500 (exists)")
            continue

        click.echo(f"  Profiling {cfg.name} with 500 samples (ground truth)...")
        inputs = _load_inputs(cfg.name, 500)
        if not inputs:
            click.echo(f"  No inputs for {cfg.name}, skipping.", err=True)
            continue
        try:
            _run_profile(cfg.workflow_path, inputs, out_path)
        except Exception as exc:
            click.echo(f"  Error profiling {cfg.name}/real500: {exc}", err=True)
            logger.debug("Profile error", exc_info=True)


def _run_phase_3(configs: list) -> None:
    """Phase 3: Score existing results (no API calls)."""
    from agentcost.store import ProfilingSession
    from agentcost.validation.suite import format_suite_report, run_backtesting_suite

    profiles: dict[str, dict[str, ProfilingSession]] = {}
    for cfg in configs:
        wf_profiles: dict[str, ProfilingSession] = {}
        for key in ("synth20", "synth100", "real500"):
            path = _result_path(cfg.name, key)
            if path.exists():
                try:
                    data = json.loads(path.read_text())
                    wf_profiles[key] = ProfilingSession.from_dict(data)
                except Exception as exc:
                    click.echo(
                        f"  Error loading {cfg.name}/{key}: {exc}",
                        err=True,
                    )
        if wf_profiles:
            profiles[cfg.name] = wf_profiles

    if not profiles:
        click.echo("No result files found. Run phase 1 and 2 first.")
        return

    result = run_backtesting_suite(profiles, configs)
    report = format_suite_report(result)
    click.echo(report)

    report_path = RESULTS_DIR / f"report_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(json.dumps(result.to_dict(), indent=2))
    click.echo(f"\nResults saved: {report_path}")


def _run_phase_4() -> None:
    """Phase 4: Generate visualizations, dashboard, and narrative report."""
    click.echo("Phase 4: Generating visualizations and reports...")

    plots_dir = RESULTS_DIR / "plots"

    try:
        from visualization.pilot.visualize_pilot import generate_all_pilot_visuals

        paths = generate_all_pilot_visuals(RESULTS_DIR, plots_dir / "pilot")
        click.echo(f"  Pilot plots: {len(paths)} files")
    except Exception as exc:
        click.echo(f"  Pilot plots failed: {exc}", err=True)

    try:
        from visualization.backtest.visualize_backtest import generate_all_backtest_visuals

        paths = generate_all_backtest_visuals(RESULTS_DIR, plots_dir / "backtest")
        click.echo(f"  Backtest plots: {len(paths)} files")
    except Exception as exc:
        click.echo(f"  Backtest plots failed: {exc}", err=True)

    try:
        from visualization.dashboard.generate_dashboard import generate_dashboard

        result = generate_dashboard(RESULTS_DIR, RESULTS_DIR / "dashboard.html")
        if result:
            click.echo(f"  Dashboard: {result}")
        else:
            click.echo("  Dashboard: skipped (plotly not installed)")
    except Exception as exc:
        click.echo(f"  Dashboard failed: {exc}", err=True)

    try:
        from visualization.narrative.generate_narrative import generate_narrative

        output = generate_narrative(RESULTS_DIR, RESULTS_DIR / "narrative.md")
        click.echo(f"  Narrative: {output}")
    except Exception as exc:
        click.echo(f"  Narrative failed: {exc}", err=True)


def _run_pre_calibration() -> bool:
    """Run pre-calibration checks. Returns True if pilot can proceed."""
    import asyncio

    try:
        from pre_calibration.pre_calibration import run_pre_calibration

        report = asyncio.run(run_pre_calibration(output=RESULTS_DIR / "pre_calibration.json"))
        if report.proceed_to_pilot:
            click.echo("Pre-calibration: PASSED")
            return True
        click.echo("Pre-calibration: BLOCKED")
        for f in report.blocking_failures:
            click.echo(f"  Failure: {f}")
        return False
    except Exception as exc:
        click.echo(f"Pre-calibration error: {exc}", err=True)
        return False


@click.command()
@click.option(
    "--phase",
    type=int,
    default=None,
    help="Phase to run: 1 (synthetic), 2 (ground truth), 3 (score), 4 (visualize).",
)
@click.option(
    "--all",
    "run_all",
    is_flag=True,
    default=False,
    help="Run all three phases sequentially.",
)
@click.option(
    "--workflow",
    type=str,
    default=None,
    help="Run only workflows matching this name (e.g., 'W1').",
)
@click.option(
    "--resume",
    is_flag=True,
    default=False,
    help="Skip workflows that already have result files.",
)
@click.option(
    "--pre-calibrate",
    is_flag=True,
    default=False,
    help="Run pre-calibration checks before starting.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def main(
    phase: int | None,
    run_all: bool,
    workflow: str | None,
    resume: bool,
    pre_calibrate: bool,
    verbose: bool,
) -> None:
    """Run the AgentCost backtesting protocol."""
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    from tests.backtesting.configs import BACKTESTING_CONFIGS

    if pre_calibrate:
        if not _run_pre_calibration():
            click.echo("Fix pre-calibration issues before running the pilot.")
            sys.exit(1)

    if run_all:
        for p in (1, 2, 3, 4):
            _run_phase(p, BACKTESTING_CONFIGS, workflow, resume)
    elif phase is not None:
        _run_phase(phase, BACKTESTING_CONFIGS, workflow, resume)
    else:
        click.echo("Specify --phase N or --all. Use --help for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
