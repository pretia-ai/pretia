#!/usr/bin/env python
"""Failure attribution CLI for the Pretia backtesting suite.

Loads backtest comparison results for each workflow and classifies failures
into three buckets using the existing attribute_failure() function:
  Bucket 1: engine/infrastructure problem (Comparison A fails)
  Bucket 2: drift sensitivity, reweighting works (A passes, B fails, C recovers)
  Bucket 3: structural drift (A passes, B fails, C doesn't recover)

Usage::

    python tests/backtesting/attribute_failure.py --results-dir results/backtest/
    python tests/backtesting/attribute_failure.py --workflow W1
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger(__name__)


def _load_workflow_results(results_dir: Path) -> dict[str, dict[str, Any]]:
    """Load per-workflow result JSONs from the results directory."""
    results: dict[str, dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*.json")):
        if "pilot" in path.name or "index" in path.name or "report" in path.name:
            continue
        try:
            data = json.loads(path.read_text())
            wf_name = data.get("workflow_name")
            if wf_name:
                results[wf_name] = data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", path, exc)
    return results


def _extract_score(comparisons: dict[str, Any], key: str) -> Any:
    """Extract a ComparisonScore from the comparisons dict."""
    from pretia.validation.scoring import ComparisonScore

    comp = comparisons.get(key)
    if comp is None:
        return None
    score_data = comp.get("score")
    if score_data is None:
        return None
    return ComparisonScore.from_dict(score_data)


def run_attribution(
    results_dir: Path,
    workflow_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Run failure attribution on all workflows in the results directory."""
    from pretia.validation.suite import _compute_recovery, attribute_failure

    workflow_results = _load_workflow_results(results_dir)
    if not workflow_results:
        logger.warning("No workflow results found in %s", results_dir)
        return []

    attributions: list[dict[str, Any]] = []

    for wf_name, data in sorted(workflow_results.items()):
        if workflow_filter and workflow_filter.upper() not in wf_name.upper():
            continue

        comparisons = data.get("comparisons", {})
        score_a = _extract_score(comparisons, "A")
        score_b = _extract_score(comparisons, "B")
        score_c = _extract_score(comparisons, "C")

        attribution = attribute_failure(wf_name, score_a, score_b, score_c)

        recovery_pct = None
        if score_a and score_b and score_c and not score_b.passes:
            recovery_pct = _compute_recovery(score_a, score_b, score_c)

        entry = {
            "workflow_name": wf_name,
            "score_a_passes": score_a.passes if score_a else None,
            "score_b_passes": score_b.passes if score_b else None,
            "score_c_passes": score_c.passes if score_c else None,
            "recovery_pct": recovery_pct,
        }

        if attribution:
            entry["bucket"] = attribution.bucket
            entry["bucket_label"] = attribution.bucket_label
            entry["explanation"] = attribution.explanation
            entry["recommended_action"] = attribution.recommended_action
        else:
            entry["bucket"] = None
            entry["bucket_label"] = "all_pass"
            entry["explanation"] = "All comparisons passed."
            entry["recommended_action"] = None

        attributions.append(entry)

    return attributions


@click.command()
@click.option(
    "--results-dir",
    type=click.Path(exists=True),
    default="tests/backtesting/results/backtest",
    help="Backtest results directory.",
)
@click.option(
    "--output",
    type=click.Path(),
    default=None,
    help="Output JSON path (default: results-dir/attribution.json).",
)
@click.option("--workflow", type=str, default=None, help="Single workflow to attribute.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug logging.")
def main(
    results_dir: str,
    output: str | None,
    workflow: str | None,
    verbose: bool,
) -> None:
    """Classify backtest failures into actionable buckets."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    rd = Path(results_dir)
    attributions = run_attribution(rd, workflow)

    if not attributions:
        click.echo("No results to attribute.", err=True)
        sys.exit(1)

    # Display summary
    passing = [a for a in attributions if a["bucket"] is None]
    bucket_1 = [a for a in attributions if a["bucket"] == 1]
    bucket_2 = [a for a in attributions if a["bucket"] == 2]
    bucket_3 = [a for a in attributions if a["bucket"] == 3]

    click.echo("Failure Attribution Report")
    click.echo("=" * 50)
    click.echo(f"  All pass: {len(passing)}")
    click.echo(f"  Bucket 1 (engine problem): {len(bucket_1)}")
    click.echo(f"  Bucket 2 (drift, reweighting works): {len(bucket_2)}")
    click.echo(f"  Bucket 3 (structural drift): {len(bucket_3)}")
    click.echo()

    for a in attributions:
        icon = "✓" if a["bucket"] is None else f"B{a['bucket']}"
        recovery = f" (recovery: {a['recovery_pct']:.0f}%)" if a["recovery_pct"] else ""
        click.echo(f"  [{icon}] {a['workflow_name']}: {a['explanation']}{recovery}")

    # Write output
    out_path = Path(output) if output else rd / "attribution.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(attributions, indent=2, default=str))
    click.echo(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
