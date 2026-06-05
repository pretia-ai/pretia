"""Command-line interface for AgentCost."""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

import click
from rich.console import Console

console = Console()


@click.group()
@click.version_option()
def cli() -> None:
    """Pre-deployment cost intelligence for AI agent workflows."""


@cli.group()
def profile() -> None:
    """Profile agent workflows for cost analysis."""


@profile.command()
@click.argument("workflow_path", type=click.Path(exists=True))
@click.option(
    "--collector",
    type=click.Choice(["auto", "langgraph", "openai", "generic"]),
    default="auto",
    help="Collector to use. Default: auto-detect.",
)
@click.option(
    "--auto-generate",
    type=int,
    default=None,
    help="Generate N synthetic test inputs. Default if no other input mode: 50.",
)
@click.option(
    "--input",
    "single_input",
    type=str,
    default=None,
    help="Single test input string.",
)
@click.option(
    "--inputs",
    "inputs_file",
    type=click.Path(),
    default=None,
    help="Path to inputs file (one per line, or .jsonl).",
)
@click.option(
    "--from-langfuse",
    is_flag=True,
    default=False,
    help="Use production traces from Langfuse as inputs.",
)
@click.option(
    "--last",
    "langfuse_last_n",
    type=int,
    default=10,
    help="Number of recent Langfuse traces to use (default: 10).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=".agentcost",
    help="Directory for output files.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose output with debug logging.",
)
@click.option(
    "--allow-cache",
    is_flag=True,
    default=False,
    help="Allow server-side prompt caching (default: bust cache for cold-start costs).",
)
def run(
    workflow_path: str,
    collector: str,
    auto_generate: int | None,
    single_input: str | None,
    inputs_file: str | None,
    from_langfuse: bool,
    langfuse_last_n: int,
    output_dir: str,
    verbose: bool,
    allow_cache: bool,
) -> None:
    """Profile a workflow and generate a cost report."""
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    if from_langfuse:
        if not os.environ.get("LANGFUSE_SECRET_KEY") or not os.environ.get("LANGFUSE_PUBLIC_KEY"):
            console.print(
                "[red]Error:[/red] Langfuse credentials not found. "
                "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY "
                "environment variables."
            )
            sys.exit(1)
        console.print(
            f"Fetching last {langfuse_last_n} traces from Langfuse...",
        )

    from agentcost.ci.report import format_cli_report
    from agentcost.runner import ProfileRunner

    runner = ProfileRunner(
        workflow_path=workflow_path,
        collector=collector,
        auto_generate=auto_generate,
        single_input=single_input,
        inputs_file=inputs_file,
        from_langfuse=from_langfuse,
        langfuse_last_n=langfuse_last_n,
        output_dir=output_dir,
        cache_mode="warm" if allow_cache else "cold",
    )

    try:
        console.print()
        console.rule(
            f"[bold]AgentCost — Profiling {workflow_path}[/bold]",
        )
        console.print()

        with console.status("[bold green]Running profiler..."):
            session = runner.run_sync()

        saved_path = session.metadata.get("saved_path", "")

        for renderable in format_cli_report(session):
            console.print(renderable)
            console.print()

        if saved_path:
            console.print(
                f"Profile saved to [bold]{saved_path}[/bold]",
            )
        console.print()

    except ImportError as exc:
        console.print(f"[red]Missing dependency:[/red] {exc}")
        sys.exit(1)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except NotImplementedError as exc:
        console.print(f"[yellow]Not yet implemented:[/yellow] {exc}")
        sys.exit(1)
    except click.UsageError:
        raise
    except Exception as exc:
        if verbose:
            console.print(traceback.format_exc())
        else:
            console.print(
                f"[red]Error:[/red] {exc}\nRun with -v for full traceback.",
            )
        sys.exit(1)


@cli.command("report")
@click.argument("profile_path")
@click.option(
    "--traffic",
    type=int,
    default=None,
    help="Runs per day for monthly projection. Default: show 100/1K/10K.",
)
def report_cmd(profile_path: str, traffic: int | None) -> None:
    """Generate a detailed report from a saved profile JSON."""
    from agentcost.ci.report import format_cli_report
    from agentcost.projection.patterns import detect_patterns
    from agentcost.projection.stats import compute_stats
    from agentcost.store import ProfileStore

    store = ProfileStore()

    try:
        if profile_path == "latest":
            sessions = store.list_sessions()
            if not sessions:
                console.print(
                    "[red]Error:[/red] No saved profiles found. "
                    "Run 'agentcost profile run' first.",
                )
                sys.exit(1)
            session = store.load(sessions[0])
        else:
            p = Path(profile_path)
            if not p.exists():
                console.print(
                    f"[red]Error:[/red] Profile not found: {profile_path}",
                )
                sys.exit(1)
            session = store.load(p)
    except Exception as exc:
        console.print(
            f"[red]Error:[/red] Invalid profile: {profile_path} — {exc}",
        )
        sys.exit(1)

    if "stats" not in session.metadata and session.runs:
        profiling_stats = compute_stats(session.runs)
        patterns = detect_patterns(session.runs, profiling_stats)
        session.metadata["stats"] = profiling_stats.to_dict()
        session.metadata["patterns"] = [p.to_dict() for p in patterns]

    for renderable in format_cli_report(session, traffic=traffic):
        console.print(renderable)
        console.print()


@cli.command("analyze")
@click.option(
    "--from-langfuse",
    is_flag=True,
    required=True,
    help="Import and analyze production traces from Langfuse.",
)
@click.option(
    "--last",
    "last_n",
    type=int,
    default=10,
    help="Number of recent traces to import (default: 10).",
)
@click.option(
    "--name",
    type=str,
    default=None,
    help="Filter traces by workflow name in Langfuse.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=".agentcost",
    help="Directory for output files.",
)
@click.option(
    "--traffic",
    type=int,
    default=None,
    help="Runs per day for monthly projection.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose output with debug logging.",
)
def analyze_cmd(
    from_langfuse: bool,
    last_n: int,
    name: str | None,
    output_dir: str,
    traffic: int | None,
    verbose: bool,
) -> None:
    """Analyze production traces without re-executing the workflow."""
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    if not os.environ.get("LANGFUSE_SECRET_KEY") or not os.environ.get("LANGFUSE_PUBLIC_KEY"):
        console.print(
            "[red]Error:[/red] Langfuse credentials not found. "
            "Set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY "
            "environment variables.",
        )
        sys.exit(1)

    try:
        from agentcost.ci.report import format_cli_report
        from agentcost.inputs.importer import (
            create_langfuse_client,
            fetch_traces,
            traces_to_step_records,
        )
        from agentcost.projection.patterns import detect_patterns
        from agentcost.projection.stats import compute_stats
        from agentcost.store import ProfileStore, ProfilingSession

        console.print(f"Fetching last {last_n} traces from Langfuse...")
        client = create_langfuse_client()
        traces = fetch_traces(client, last_n=last_n, name=name)

        if not traces:
            console.print(
                "[red]Error:[/red] No traces found in Langfuse. "
                "Check your LANGFUSE_HOST and trace name filter.",
            )
            sys.exit(1)

        console.print(f"Converting {len(traces)} traces to profiling data...")
        runs = traces_to_step_records(traces)

        profiling_stats = compute_stats(runs)
        patterns = detect_patterns(runs, profiling_stats)

        from datetime import UTC, datetime

        workflow_name = traces[0].name or "langfuse-import"
        session = ProfilingSession(
            workflow_name=workflow_name,
            workflow_hash="langfuse",
            profiled_at=datetime.now(UTC),
            sample_size=len(traces),
            input_mode="langfuse-analyze",
            runs=runs,
            metadata={
                "stats": profiling_stats.to_dict(),
                "patterns": [p.to_dict() for p in patterns],
                "langfuse_trace_count": len(traces),
                "langfuse_trace_ids": [t.trace_id for t in traces],
            },
        )

        store = ProfileStore(storage_dir=Path(output_dir))
        saved_path = store.save(session)
        session.metadata["saved_path"] = str(saved_path)

        console.print()
        for renderable in format_cli_report(session, traffic=traffic):
            console.print(renderable)
            console.print()

        console.print(f"Profile saved to [bold]{saved_path}[/bold]")
        console.print()

    except (PermissionError, ConnectionError, OSError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        if verbose:
            console.print(traceback.format_exc())
        else:
            console.print(
                f"[red]Error:[/red] {exc}\nRun with -v for full traceback.",
            )
        sys.exit(1)


@cli.group()
def baseline() -> None:
    """Manage cost baselines."""


@baseline.command("update")
@click.argument("profile_path")
@click.option(
    "--traffic",
    type=int,
    default=1000,
    help="Daily traffic volume for projection (default: 1000).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default=".agentcost",
    help="Directory for baseline file.",
)
def baseline_update(
    profile_path: str,
    traffic: int,
    output_dir: str,
) -> None:
    """Save current profile as a cost baseline."""
    from agentcost.ci.baseline import create_baseline, save_baseline
    from agentcost.store import ProfileStore

    store = ProfileStore(storage_dir=Path(output_dir))

    try:
        if profile_path == "latest":
            sessions = store.list_sessions()
            if not sessions:
                console.print(
                    "[red]Error:[/red] No saved profiles found. "
                    "Run 'agentcost profile run' first.",
                )
                sys.exit(1)
            session = store.load(sessions[0])
        else:
            p = Path(profile_path)
            if not p.exists():
                console.print(
                    f"[red]Error:[/red] Profile not found: {profile_path}",
                )
                sys.exit(1)
            session = store.load(p)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    try:
        bl = create_baseline(session, traffic=traffic)
        saved_path = save_baseline(bl, output_dir=output_dir)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(f"Baseline saved: [bold]{saved_path}[/bold]")
    console.print()
    console.print("[dim]Assumptions baked into this baseline:[/dim]")
    for assumption in bl.assumptions:
        console.print(f"  [dim]• {assumption}[/dim]")


@cli.command("diff")
@click.argument("baseline_path")
@click.argument("profile_path")
@click.option(
    "--traffic",
    type=int,
    default=None,
    help="Override traffic volume (default: use baseline assumption).",
)
@click.option(
    "--threshold",
    type=int,
    default=None,
    help="Fail if monthly cost increase exceeds this percentage.",
)
def diff_cmd(
    baseline_path: str,
    profile_path: str,
    traffic: int | None,
    threshold: int | None,
) -> None:
    """Compare a baseline to a new profile and show cost deltas."""
    from agentcost.ci.baseline import load_baseline
    from agentcost.ci.diff import diff_baseline, format_diff_report
    from agentcost.projection.patterns import detect_patterns
    from agentcost.projection.stats import compute_stats
    from agentcost.store import ProfileStore

    try:
        bl = load_baseline(baseline_path)
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    store = ProfileStore()
    try:
        if profile_path == "latest":
            sessions = store.list_sessions()
            if not sessions:
                console.print(
                    "[red]Error:[/red] No saved profiles found.",
                )
                sys.exit(1)
            session = store.load(sessions[0])
        else:
            p = Path(profile_path)
            if not p.exists():
                console.print(
                    f"[red]Error:[/red] Profile not found: {profile_path}",
                )
                sys.exit(1)
            session = store.load(p)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if "stats" not in session.metadata and session.runs:
        profiling_stats = compute_stats(session.runs)
        patterns = detect_patterns(session.runs, profiling_stats)
        session.metadata["stats"] = profiling_stats.to_dict()
        session.metadata["patterns"] = [p.to_dict() for p in patterns]

    try:
        result = diff_baseline(bl, session, traffic=traffic)
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print(format_diff_report(result))

    if threshold is not None:
        p50_pct = result.total_monthly_pct_change.get("p50", 0)
        if p50_pct > threshold:
            console.print(
                f"\n[red]Cost increase ({p50_pct:.0f}%) exceeds "
                f"threshold ({threshold}%). Failing.[/red]",
            )
            sys.exit(1)


@cli.command("validate")
@click.argument("workflow_path", type=click.Path(exists=True))
@click.option(
    "--budget",
    type=float,
    default=10.0,
    help="Estimated cost for both profiling runs (default: $10).",
)
@click.option(
    "--small-n",
    type=int,
    default=20,
    help="First sample size (default: 20).",
)
@click.option(
    "--large-n",
    type=int,
    default=100,
    help="Second sample size (default: 100).",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose output with debug logging.",
)
def validate_cmd(
    workflow_path: str,
    budget: float,
    small_n: int,
    large_n: int,
    verbose: bool,
) -> None:
    """Run projection quality check (small-n vs large-n comparison)."""
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    total_runs = small_n + large_n
    if not click.confirm(
        f"This will profile your workflow {total_runs} times. "
        f"Estimated cost: ~${budget:.2f}. Proceed?"
    ):
        console.print("Cancelled.")
        return

    try:
        from agentcost.validation.validate_cmd import (
            format_validate_report,
            run_validation,
        )

        with console.status("[bold green]Running validation..."):
            result = run_validation(
                workflow_path,
                budget=budget,
                small_n=small_n,
                large_n=large_n,
            )

        console.print(format_validate_report(result))

        if result.score.verdict == "FAIL":
            sys.exit(1)

    except Exception as exc:
        if verbose:
            console.print(traceback.format_exc())
        else:
            console.print(
                f"[red]Error:[/red] {exc}\nRun with -v for full traceback.",
            )
        sys.exit(1)


@cli.command("update-pricing")
def update_pricing_cmd() -> None:
    """Update model pricing data."""
    console.print(
        "Pricing update is not yet automated.\n"
        "To update prices manually, edit agentcost/pricing/tables.py.\n"
        "See scripts/pricing_sources.md for current pricing page URLs."
    )


if __name__ == "__main__":
    cli()
