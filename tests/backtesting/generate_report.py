#!/usr/bin/env python
"""Report generation CLI for the AgentCost backtesting suite.

Collates pilot and backtest results, then calls the existing visualization
generators: dashboard (HTML), narrative (Markdown), JSON export, and PDF export.

Usage::

    python tests/backtesting/generate_report.py --all
    python tests/backtesting/generate_report.py --format dashboard
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

logger = logging.getLogger(__name__)


def _generate_pilot_plots(pilot_dir: Path, output_dir: Path) -> int:
    """Generate P1-P6 pilot visualizations. Returns count of files generated."""
    try:
        from visualization.pilot.visualize_pilot import generate_all_pilot_visuals

        paths = generate_all_pilot_visuals(pilot_dir, output_dir)
        return len(paths)
    except Exception as exc:
        logger.error("Pilot plots failed: %s", exc)
        return 0


def _generate_backtest_plots(backtest_dir: Path, output_dir: Path) -> int:
    """Generate B1-B10 backtest visualizations. Returns count of files generated."""
    try:
        from visualization.backtest.visualize_backtest import (
            generate_all_backtest_visuals,
        )

        paths = generate_all_backtest_visuals(backtest_dir, output_dir)
        return len(paths)
    except Exception as exc:
        logger.error("Backtest plots failed: %s", exc)
        return 0


def _generate_dashboard(
    backtest_dir: Path,
    output_path: Path,
    pilot_plots_dir: Path | None,
    backtest_plots_dir: Path | None,
) -> bool:
    """Generate interactive HTML dashboard. Returns True on success."""
    try:
        from visualization.dashboard.generate_dashboard import generate_dashboard

        result = generate_dashboard(
            backtest_dir,
            output_path,
            pilot_plots_dir=pilot_plots_dir,
            backtest_plots_dir=backtest_plots_dir,
        )
        return result is not None
    except Exception as exc:
        logger.error("Dashboard generation failed: %s", exc)
        return False


def _generate_narrative(backtest_dir: Path, output_path: Path) -> bool:
    """Generate Markdown narrative report. Returns True on success."""
    try:
        from visualization.narrative.generate_narrative import generate_narrative

        generate_narrative(backtest_dir, output_path)
        return True
    except Exception as exc:
        logger.error("Narrative generation failed: %s", exc)
        return False


def _generate_json_export(backtest_dir: Path, output_path: Path) -> bool:
    """Generate consolidated JSON export. Returns True on success."""
    try:
        from visualization.export import export_analytics_json

        export_analytics_json(backtest_dir, output_path)
        return True
    except Exception as exc:
        logger.error("JSON export failed: %s", exc)
        return False


def _generate_pdf_export(
    backtest_dir: Path,
    pilot_plots_dir: Path | None,
    backtest_plots_dir: Path | None,
    output_path: Path,
) -> bool:
    """Generate PDF export with embedded plots. Returns True on success."""
    try:
        from visualization.export import export_analytics_pdf

        export_analytics_pdf(
            backtest_dir,
            pilot_plots_dir,
            backtest_plots_dir,
            output_path,
        )
        return True
    except Exception as exc:
        logger.error("PDF export failed: %s", exc)
        return False


@click.command()
@click.option(
    "--pilot-dir",
    type=click.Path(),
    default="tests/backtesting/results/pilot",
    help="Pilot results directory.",
)
@click.option(
    "--backtest-dir",
    type=click.Path(),
    default="tests/backtesting/results/backtest",
    help="Backtest results directory.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="reports/backtest",
    help="Report output directory.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["dashboard", "narrative", "json", "pdf", "all"]),
    default="all",
    help="Output format(s) to generate.",
)
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug logging.")
def main(
    pilot_dir: str,
    backtest_dir: str,
    output_dir: str,
    fmt: str,
    verbose: bool,
) -> None:
    """Generate reports from backtesting results."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    pilot_path = Path(pilot_dir)
    backtest_path = Path(backtest_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if not backtest_path.exists():
        click.echo(f"Backtest results not found at {backtest_path}.", err=True)
        sys.exit(1)

    formats = ["dashboard", "narrative", "json", "pdf"] if fmt == "all" else [fmt]

    # Generate plots first (needed by dashboard and PDF)
    pilot_plots_dir = out_path / "plots" / "pilot"
    backtest_plots_dir = out_path / "plots" / "backtest"

    click.echo("Generating reports...")

    if pilot_path.exists():
        n = _generate_pilot_plots(pilot_path, pilot_plots_dir)
        click.echo(f"  Pilot plots: {n} files")
    else:
        click.echo("  Pilot plots: skipped (no pilot dir)")
        pilot_plots_dir = None

    n = _generate_backtest_plots(backtest_path, backtest_plots_dir)
    click.echo(f"  Backtest plots: {n} files")
    if n == 0:
        backtest_plots_dir = None

    for f in formats:
        if f == "dashboard":
            ok = _generate_dashboard(
                backtest_path,
                out_path / "dashboard.html",
                pilot_plots_dir,
                backtest_plots_dir,
            )
            click.echo(f"  Dashboard: {'OK' if ok else 'FAILED'}")

        elif f == "narrative":
            ok = _generate_narrative(backtest_path, out_path / "narrative.md")
            click.echo(f"  Narrative: {'OK' if ok else 'FAILED'}")

        elif f == "json":
            ok = _generate_json_export(backtest_path, out_path / "analytics.json")
            click.echo(f"  JSON export: {'OK' if ok else 'FAILED'}")

        elif f == "pdf":
            ok = _generate_pdf_export(
                backtest_path,
                pilot_plots_dir,
                backtest_plots_dir,
                out_path / "analytics.pdf",
            )
            click.echo(f"  PDF export: {'OK' if ok else 'FAILED'}")

    click.echo(f"\nReports written to {out_path}")


if __name__ == "__main__":
    main()
