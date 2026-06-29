"""Command-line interface for Pretia."""

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
    "--entry-point",
    type=str,
    default=None,
    help="Name of the workflow variable to profile (e.g. 'pipeline', 'bot'). "
    "Default: auto-detect from graph/workflow/agent/app or async callables.",
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
    default=".pretia",
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
@click.option(
    "--generator-model",
    type=str,
    default=None,
    help="LLM model for synthetic input generation. Default: deepseek-v4-flash.",
)
@click.option(
    "--corpus",
    type=click.Path(),
    default=None,
    help="Path to document corpus (file or directory) for RAG-aware input generation.",
)
@click.option(
    "--no-html",
    is_flag=True,
    default=False,
    help="Skip HTML report generation.",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Don't auto-open the HTML report in a browser.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def run(
    workflow_path: str,
    collector: str,
    entry_point: str | None,
    auto_generate: int | None,
    single_input: str | None,
    inputs_file: str | None,
    from_langfuse: bool,
    langfuse_last_n: int,
    output_dir: str,
    verbose: bool,
    allow_cache: bool,
    generator_model: str | None,
    corpus: str | None,
    no_html: bool,
    no_open: bool,
    yes: bool,
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

    num_runs = _infer_run_count(
        auto_generate=auto_generate,
        single_input=single_input,
        inputs_file=inputs_file,
        from_langfuse=from_langfuse,
        langfuse_last_n=langfuse_last_n,
    )

    if not yes:
        _show_confirmation(workflow_path, num_runs)
        if not click.confirm("Proceed?", default=True):
            console.print("Cancelled.")
            return

    if not corpus and _detect_rag_imports(workflow_path):
        console.print(
            "[yellow]RAG patterns detected.[/yellow] For best results, provide "
            "your document corpus with [bold]--corpus <path>[/bold]. Without it, "
            "generated inputs may not trigger retrieval effectively.",
        )
        console.print()

    import time as _time

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
    )

    from pretia.ci.report import format_cli_report
    from pretia.runner import ProfileRunner

    accumulated_cost = 0.0

    progress = Progress(
        TextColumn("[bold green]Profiling"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("Cost: {task.fields[cost]}"),
        TimeRemainingColumn(),
        console=console,
    )

    def _on_run_done(run_idx: int, total: int, records: list) -> None:
        nonlocal accumulated_cost
        for r in records:
            try:
                from pretia.pricing.tables import calculate_cost

                accumulated_cost += calculate_cost(
                    r.model,
                    r.input_tokens,
                    r.output_tokens,
                )
            except (ValueError, KeyError, AttributeError):
                pass
        progress.update(task_id, advance=1, cost=f"${accumulated_cost:.4f}")

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
        progress_callback=_on_run_done,
        generator_model=generator_model or "deepseek-v4-flash",
        corpus_path=corpus,
        entry_point=entry_point,
    )

    try:
        console.print()
        console.rule(
            f"[bold]Pretia — Profiling {workflow_path}[/bold]",
        )
        console.print()

        t0 = _time.monotonic()
        with progress:
            task_id = progress.add_task("Profiling", total=num_runs, cost="$0.0000")
            session = runner.run_sync()
        elapsed = _time.monotonic() - t0

        _enrich_with_recommendations(session)

        _show_profiling_summary(session, elapsed)

        for renderable in format_cli_report(session):
            console.print(renderable)
            console.print()

        saved_path = session.metadata.get("saved_path", "")
        if saved_path:
            console.print(
                f"Profile saved to [bold]{saved_path}[/bold]",
            )

        if not no_html:
            try:
                from pretia.report import render_and_save

                html_path = render_and_save(
                    session,
                    open_browser=not no_open,
                )
                console.print(
                    f"HTML report: [bold]{html_path}[/bold]",
                )
            except Exception as html_exc:
                logging.debug("HTML report generation failed: %s", html_exc)

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
@click.option(
    "--no-html",
    is_flag=True,
    default=False,
    help="Skip HTML report generation.",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Don't auto-open the HTML report in a browser.",
)
def report_cmd(
    profile_path: str,
    traffic: int | None,
    no_html: bool,
    no_open: bool,
) -> None:
    """Generate a detailed report from a saved profile JSON."""
    from pretia.ci.report import format_cli_report
    from pretia.projection.patterns import detect_patterns
    from pretia.projection.stats import compute_stats
    from pretia.store import ProfileStore

    store = ProfileStore()

    try:
        if profile_path == "latest":
            sessions = store.list_sessions()
            if not sessions:
                console.print(
                    "[red]Error:[/red] No saved profiles found. Run 'pretia profile run' first.",
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

    if "recommendations" not in session.metadata:
        _enrich_with_recommendations(session)

    for renderable in format_cli_report(session, traffic=traffic):
        console.print(renderable)
        console.print()

    if not no_html:
        try:
            from pretia.report import render_and_save

            html_path = render_and_save(session, open_browser=not no_open)
            console.print(f"HTML report: [bold]{html_path}[/bold]")
            console.print()
        except Exception as html_exc:
            logging.debug("HTML report generation failed: %s", html_exc)


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
    default=".pretia",
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
@click.option(
    "--no-html",
    is_flag=True,
    default=False,
    help="Skip HTML report generation.",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Don't auto-open the HTML report in a browser.",
)
def analyze_cmd(
    from_langfuse: bool,
    last_n: int,
    name: str | None,
    output_dir: str,
    traffic: int | None,
    verbose: bool,
    no_html: bool,
    no_open: bool,
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
        from pretia.ci.report import format_cli_report
        from pretia.inputs.importer import (
            create_langfuse_client,
            fetch_traces,
            traces_to_step_records,
        )
        from pretia.projection.patterns import detect_patterns
        from pretia.projection.stats import compute_stats
        from pretia.store import ProfileStore, ProfilingSession

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

        from pretia.projection.projector import project

        projection = project(
            profiling_stats,
            patterns,
            runs=runs,
            input_source="langfuse-analyze",
        )

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
                "projection": projection.to_dict(),
                "confidence": projection.confidence.to_dict(),
                "langfuse_trace_count": len(traces),
                "langfuse_trace_ids": [t.trace_id for t in traces],
            },
        )

        store = ProfileStore(storage_dir=Path(output_dir))
        saved_path = store.save(session)
        session.metadata["saved_path"] = str(saved_path)

        _enrich_with_recommendations(session)

        console.print()
        for renderable in format_cli_report(session, traffic=traffic):
            console.print(renderable)
            console.print()

        console.print(f"Profile saved to [bold]{saved_path}[/bold]")

        if not no_html:
            try:
                from pretia.report import render_and_save

                html_path = render_and_save(session, open_browser=not no_open)
                console.print(f"HTML report: [bold]{html_path}[/bold]")
            except Exception as html_exc:
                logging.debug("HTML report generation failed: %s", html_exc)

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
    default=".pretia",
    help="Directory for baseline file.",
)
def baseline_update(
    profile_path: str,
    traffic: int,
    output_dir: str,
) -> None:
    """Save current profile as a cost baseline."""
    from pretia.ci.baseline import create_baseline, save_baseline
    from pretia.store import ProfileStore

    store = ProfileStore(storage_dir=Path(output_dir))

    try:
        if profile_path == "latest":
            sessions = store.list_sessions()
            if not sessions:
                console.print(
                    "[red]Error:[/red] No saved profiles found. Run 'pretia profile run' first.",
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
    from pretia.ci.baseline import load_baseline
    from pretia.ci.diff import diff_baseline, format_diff_report
    from pretia.projection.patterns import detect_patterns
    from pretia.projection.stats import compute_stats
    from pretia.store import ProfileStore

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
        from pretia.validation.validate_cmd import (
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


_RAG_IMPORTS = frozenset(
    {
        "langchain.vectorstores",
        "langchain_community.vectorstores",
        "chromadb",
        "pinecone",
        "faiss",
        "qdrant_client",
    }
)


def _detect_rag_imports(workflow_path: str) -> bool:
    """Check if a workflow file imports retrieval-related packages."""
    try:
        source = Path(workflow_path).read_text(encoding="utf-8")
    except OSError:
        return False
    return any(pkg in source for pkg in _RAG_IMPORTS)


def _infer_run_count(
    *,
    auto_generate: int | None,
    single_input: str | None,
    inputs_file: str | None,
    from_langfuse: bool,
    langfuse_last_n: int,
) -> int:
    """Infer the number of profiling runs from CLI arguments."""
    if single_input is not None:
        return 1
    if inputs_file is not None:
        try:
            return max(1, sum(1 for _ in open(inputs_file)))  # noqa: SIM115
        except OSError:
            return 1
    if from_langfuse:
        return langfuse_last_n
    return auto_generate or 50


def _show_confirmation(workflow_path: str, num_runs: int) -> None:
    """Display a pre-profiling summary panel."""
    from pretia.ci.report import format_cost

    framework = "unknown"
    est_cost = "unknown"
    try:
        from pretia.estimate import estimate_workflow

        est = estimate_workflow(workflow_path)
        framework = est.framework or "generic"
        if est.estimated_cost_per_run > 0:
            total = est.estimated_cost_per_run * num_runs
            est_cost = f"~{format_cost(total)} (uses your API keys)"
        else:
            est_cost = "unknown — no models detected in source"
    except Exception:
        logging.debug("Static estimate failed", exc_info=True)

    lo = max(1, num_runs * 5)
    hi = num_runs * 15
    if hi < 60:
        time_est = f"~{lo}-{hi} seconds"
    else:
        time_est = f"~{lo // 60}-{hi // 60} minutes"

    from rich.panel import Panel
    from rich.text import Text

    text = Text.assemble(
        ("Profiling Summary\n", "bold"),
        ("  Workflow:   ", "dim"),
        (workflow_path, ""),
        ("\n", ""),
        ("  Framework:  ", "dim"),
        (framework, ""),
        ("\n", ""),
        ("  Runs:       ", "dim"),
        (str(num_runs), ""),
        ("\n", ""),
        ("  Est. cost:  ", "dim"),
        (est_cost, ""),
        ("\n", ""),
        ("  Est. time:  ", "dim"),
        (time_est, ""),
    )
    console.print()
    console.print(Panel(text, expand=False))


def _show_profiling_summary(session: object, elapsed: float) -> None:
    """Display a compact post-profiling summary."""
    from pretia.ci.report import format_cost

    meta = session.metadata or {}
    cost_summary = meta.get("cost_summary", {})
    patterns = meta.get("patterns", [])

    total_cost = cost_summary.get("total_session_cost", 0)
    mean_cost = cost_summary.get("mean_cost_per_run", 0)

    console.print()
    for p in patterns:
        severity = p.get("severity", "warning")
        icon = "[red]![/red]" if severity == "danger" else "[yellow]![/yellow]"
        ptype = p.get("pattern_type", "")
        desc = p.get("description", "")[:80]
        console.print(f"  {icon} {ptype}: {desc}")

    from rich.panel import Panel
    from rich.text import Text

    text = Text.assemble(
        ("Profiling Complete\n", "bold green"),
        ("  Runs:        ", "dim"),
        (str(session.sample_size), ""),
        ("\n", ""),
        ("  Total cost:  ", "dim"),
        (format_cost(total_cost), "bold"),
        ("\n", ""),
        ("  Mean/run:    ", "dim"),
        (format_cost(mean_cost), ""),
        ("\n", ""),
        ("  Patterns:    ", "dim"),
        (f"{len(patterns)} detected" if patterns else "none", ""),
        ("\n", ""),
        ("  Time:        ", "dim"),
        (f"{elapsed:.1f}s", ""),
    )
    console.print(Panel(text, expand=False))
    console.print()


def _enrich_with_recommendations(session: object) -> None:
    """Run the recommendation engine and store results in session metadata."""
    from pretia.recommend import compute_score, generate_recommendations

    recs = generate_recommendations(session)
    session.metadata["recommendations"] = [r.to_dict() for r in recs]

    projected_cost = 0.0
    projection = session.metadata.get("projection", {})
    projs = projection.get("projections", {})
    for vol_data in projs.values():
        monthly = vol_data.get("monthly_cost", {})
        p50 = monthly.get("p50", 0)
        if p50 > projected_cost:
            projected_cost = p50

    if projected_cost == 0.0:
        cost_summary = session.metadata.get("cost_summary", {})
        mean_cost = cost_summary.get("mean_cost_per_run", 0)
        projected_cost = mean_cost * 10_000 * 30

    score = compute_score(recs, projected_cost)
    session.metadata["score"] = score.to_dict()


@cli.command("recommend")
@click.argument("profile_path")
def recommend_cmd(profile_path: str) -> None:
    """Generate optimization recommendations from a saved profile."""
    from pretia.ci.report import (
        _build_recommendations_panel,
        _build_score_panel,
    )
    from pretia.projection.patterns import detect_patterns
    from pretia.projection.stats import compute_stats
    from pretia.store import ProfileStore

    store = ProfileStore()

    try:
        if profile_path == "latest":
            sessions = store.list_sessions()
            if not sessions:
                console.print(
                    "[red]Error:[/red] No saved profiles found. Run 'pretia profile run' first.",
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

    _enrich_with_recommendations(session)

    score_data = session.metadata.get("score", {})
    rec_data = session.metadata.get("recommendations", [])

    console.print()
    console.print(_build_score_panel(score_data))
    console.print()
    console.print(_build_recommendations_panel(rec_data))
    console.print()


@cli.command("estimate")
@click.argument("workflow_path", type=click.Path(exists=True))
@click.option(
    "--traffic",
    type=int,
    default=None,
    help="Runs per day for projection. Default: show 100/1K/10K.",
)
@click.option(
    "-v",
    "--verbose",
    is_flag=True,
    default=False,
    help="Verbose output with debug logging.",
)
def estimate_cmd(workflow_path: str, traffic: int | None, verbose: bool) -> None:
    """Instant cost estimate from code structure (no execution)."""
    from pretia.ci.report import format_cost
    from pretia.estimate import estimate_workflow

    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    try:
        est = estimate_workflow(workflow_path)
    except FileNotFoundError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] Failed to analyze: {exc}")
        sys.exit(1)

    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    console.print()
    header = Text.assemble(
        ("Pretia — Static Estimate\n", "bold"),
        ("Workflow: ", "dim"),
        (workflow_path, ""),
        ("\n", ""),
        ("Framework: ", "dim"),
        (est.framework or "unknown", ""),
        (" | ", "dim"),
        ("Steps: ", "dim"),
        (str(est.estimated_steps), ""),
    )
    if est.estimated_system_prompt_tokens > 0:
        header.append("\n")
        header.append("System prompt tokens: ", style="dim")
        header.append(f"~{est.estimated_system_prompt_tokens:,} (extracted from source)")
    console.print(Panel(header, title="Static Estimate"))

    if est.parse_error:
        console.print(
            f"[yellow]Warning:[/yellow] Could not parse file — {est.parse_error}",
        )
        console.print()

    if est.models:
        table = Table(title="Models Detected", show_lines=False, pad_edge=False)
        table.add_column("Model", style="bold")
        table.add_column("Tier")
        table.add_column("Input $/M", justify="right")
        table.add_column("Output $/M", justify="right")

        unrecognized = []
        for m in est.models:
            tier = ""
            if m.canonical_name:
                try:
                    from pretia.pricing.tables import model_tier

                    tier = model_tier(m.canonical_name)
                except (ValueError, KeyError):
                    pass
            else:
                unrecognized.append(m.model_name)

            table.add_row(
                m.model_name,
                tier,
                f"${m.input_price_per_m:.2f}" if m.input_price_per_m else "—",
                f"${m.output_price_per_m:.2f}" if m.output_price_per_m else "—",
            )

        console.print(table)
        if unrecognized:
            for name in unrecognized:
                console.print(
                    f"[yellow]Warning:[/yellow] Unrecognized model '{name}'. "
                    f"Use register_model('{name}', input_price=X, output_price=Y) "
                    "to add pricing.",
                )
        console.print()

        volumes = [traffic] if traffic else [100, 1_000, 10_000]
        proj_table = Table(
            title="Estimated Monthly Cost",
            show_lines=True,
            pad_edge=True,
        )
        proj_table.add_column("Runs/Day", style="bold")
        proj_table.add_column("Monthly Estimate", justify="right")

        for v in volumes:
            monthly = est.estimated_cost_per_run * v * 30
            proj_table.add_row(f"{v:,}", format_cost(monthly))

        console.print(proj_table)
    else:
        console.print(
            "[yellow]No models detected in source file.[/yellow] "
            "The file may use dynamic model assignment or a framework "
            "not yet supported by static analysis.",
        )

    console.print()
    console.print(
        "[dim italic]This is a rough estimate from static analysis. "
        "Run [bold]pretia profile run[/bold] for distributional "
        "projections and optimization recommendations.[/dim italic]",
    )
    console.print()


@cli.command("update-pricing")
def update_pricing_cmd() -> None:
    """Update model pricing data."""
    console.print(
        "Pricing update is not yet automated.\n"
        "To update prices manually, edit pretia/pricing/tables.py.\n"
        "See scripts/pricing_sources.md for current pricing page URLs."
    )


if __name__ == "__main__":
    cli()
