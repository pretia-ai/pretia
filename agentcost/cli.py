"""Command-line interface for AgentCost."""

from __future__ import annotations

import logging
import sys
import traceback

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
    help="Generate N synthetic test inputs. Default if no other input mode: 20.",
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
    help="Import inputs from Langfuse traces.",
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
def run(
    workflow_path: str,
    collector: str,
    auto_generate: int | None,
    single_input: str | None,
    inputs_file: str | None,
    from_langfuse: bool,
    output_dir: str,
    verbose: bool,
) -> None:
    """Profile a workflow and generate a cost report."""
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=log_level, format="%(message)s")

    from agentcost.ci.report import format_cli_report
    from agentcost.runner import ProfileRunner

    runner = ProfileRunner(
        workflow_path=workflow_path,
        collector=collector,
        auto_generate=auto_generate,
        single_input=single_input,
        inputs_file=inputs_file,
        from_langfuse=from_langfuse,
        output_dir=output_dir,
    )

    try:
        console.print()
        console.rule(
            f"[bold]AgentCost — Profiling {workflow_path}[/bold]",
        )
        console.print()

        with console.status("[bold green]Running profiler..."):
            session = runner.run_sync()

        cost_summary = session.metadata.get("cost_summary", {})
        saved_path = session.metadata.get("saved_path", "")

        for renderable in format_cli_report(session, cost_summary):
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
                f"[red]Error:[/red] {exc}\n"
                "Run with -v for full traceback.",
            )
        sys.exit(1)


if __name__ == "__main__":
    cli()
