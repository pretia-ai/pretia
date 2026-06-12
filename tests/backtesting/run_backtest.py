#!/usr/bin/env python
"""Backtest comparison runner for the AgentCost backtesting suite.

Orchestrates the three-comparison protocol (A: no drift, B: drifted, C: reweighted)
for each workflow. Computes accuracy metrics, validates detectors, generates B1-B10
plots, and writes results to the long-term dataset.

Usage::

    python tests/backtesting/run_backtest.py --all
    python tests/backtesting/run_backtest.py --workflow W1 --comparison A
    python tests/backtesting/run_backtest.py --all --dry-run
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import platform
import sys
import uuid
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

_GROUND_TRUTH_N: dict[str, int] = {
    "W1": 200,
    "W2": 300,
    "W4": 500,
    "W5": 220,
    "W9": 200,
    "W11": 200,
    "W12": 200,
    "W13": 300,
    "W14": 300,
    "W15": 300,
    "W16": 300,
    "W17": 300,
    "W18": 300,
    "W19": 300,
}

_PROFILING_N = 50

_CHEAP_WORKFLOWS = {"W1", "W9", "W11", "W12"}


def _wf_id(name: str) -> str:
    """Extract workflow ID from config name (e.g., 'W1-support-simple' → 'W1')."""
    return name.split("-")[0].upper()


def _check_validation_gate(
    report_path: Path | None = None,
    min_stage: int = 3,
) -> bool:
    """Verify scripts/validate.py passed at least the required stage.

    Falls back to legacy pilot_report.json if validation.json is missing.
    """
    if report_path is None:
        report_path = Path(__file__).resolve().parents[2] / "reports" / "validation.json"

    if report_path.exists():
        report = json.loads(report_path.read_text())
        max_stage = report.get("max_passed_stage", 0)
        if max_stage >= min_stage:
            click.echo(f"Validation gate: PASSED (stage {max_stage})")
            return True
        click.echo(
            f"Validation only passed through stage {max_stage} (need {min_stage}+).\n"
            "Fix issues and re-run: python scripts/validate.py",
            err=True,
        )
        for stage_data in report.get("stages", []):
            for fail in stage_data.get("blocking_failures", []):
                click.echo(f"  BLOCKED: Stage {stage_data['stage']} / {fail}", err=True)
        return False

    # Fallback: legacy pilot report
    pilot_path = Path(__file__).resolve().parent / "results" / "pilot" / "pilot_report.json"
    if pilot_path.exists():
        click.echo("Using legacy pilot_report.json (run scripts/validate.py for unified gate).")
        pilot = json.loads(pilot_path.read_text())
        blocked = pilot.get("blocked_workflows", [])
        if blocked:
            click.echo(f"Pilot had blocked workflows: {blocked}", err=True)
            return False
        return True

    click.echo(
        "No validation report found.\n"
        "Run: python scripts/validate.py --output reports/validation.json",
        err=True,
    )
    return False


def _check_pilot_gate(pilot_dir: Path, workflow_id: str) -> bool:
    """Legacy: verify pilot report for a specific workflow. Use _check_validation_gate instead."""
    report_path = pilot_dir / "pilot_report.json"
    if not report_path.exists():
        click.echo(
            f"Pilot report not found at {report_path}.\n"
            "Run the pilot first: python tests/backtesting/run_pilot.py --all",
            err=True,
        )
        return False

    report = json.loads(report_path.read_text())
    wf_data = report.get("per_workflow", {}).get(workflow_id)
    if wf_data is None:
        click.echo(f"No pilot data for {workflow_id}.", err=True)
        return False

    if wf_data.get("blocked"):
        click.echo(
            f"Pilot for {workflow_id} had blocking failures. Fix before backtesting.",
            err=True,
        )
        return False

    return True


def _config_for_workflow(workflow_id: str) -> Any:
    """Find BacktestConfig for a workflow ID."""
    from tests.backtesting.configs import BACKTESTING_CONFIGS

    wf_upper = workflow_id.upper()
    for cfg in BACKTESTING_CONFIGS:
        if cfg.name.split("-")[0].upper() == wf_upper:
            return cfg
    return None


def _generate_inputs(
    workflow_id: str,
    n: int,
    profile: str,
    seed: int = 42,
    dry_run: bool = True,
    results_dir: str = "tests/backtesting/results/backtest",
) -> list[dict[str, Any]]:
    """Generate inputs using the per-workflow generator."""
    try:
        import importlib
        import pkgutil

        import inputs.generators as gen_pkg
        from inputs.generators._base import BaseInputGenerator

        wf_num = workflow_id.upper().replace("W", "").zfill(2)

        for info in pkgutil.iter_modules(gen_pkg.__path__):
            if info.name.startswith(f"w{wf_num}_"):
                mod = importlib.import_module(f"inputs.generators.{info.name}")
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
                            gen._image_output_dir = f"{results_dir}/images/{workflow_id.lower()}"
                        batch = gen.generate_batch(profile, n)
                        return [inp.to_dict() for inp in batch]
    except Exception as exc:
        logger.warning("Input generator failed for %s: %s", workflow_id, exc)

    return [
        {"input": f"Input {i} for {workflow_id}", "tier": "easy", "_dry_run": True}
        for i in range(n)
    ]


def _inputs_to_dicts(inputs: list[dict[str, Any]], dry_run: bool) -> list[dict[str, Any]]:
    """Convert generated inputs to the format expected by run_batch."""
    result = []
    for inp in inputs:
        if "input_data" in inp:
            d = dict(inp["input_data"])
            d["tier"] = inp.get("tier", "easy")
            d["structural_descriptor"] = inp.get("structural_descriptor", {})
        else:
            d = dict(inp)
        if dry_run:
            d["_dry_run"] = True
        result.append(d)
    return result


def _compute_run_costs(all_records: list[list[Any]]) -> list[float]:
    """Compute total cost per run."""
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


def _compute_step_costs(all_records: list[list[Any]]) -> dict[str, float]:
    """Compute mean cost per step name across all runs."""
    step_totals: dict[str, float] = {}
    step_counts: dict[str, int] = {}
    for run in all_records:
        for r in run:
            try:
                cost = calculate_cost(r.model, r.input_tokens, r.output_tokens)
                step_totals[r.step_name] = step_totals.get(r.step_name, 0) + cost
                step_counts[r.step_name] = step_counts.get(r.step_name, 0) + 1
            except Exception:
                pass
    return {name: step_totals[name] / step_counts[name] for name in step_totals}


def _reweight_runs(
    profiling_records: list[list[Any]],
    input_tiers: list[str],
) -> list[list[Any]]:
    """Resample profiling runs according to ground truth tier weights.

    Creates a new list of runs where each run is replicated proportionally to
    GROUND_TRUTH_WEIGHTS[tier] / PROFILING_WEIGHTS[tier]. The "extreme" tier
    (absent from profiling) has its weight redistributed.
    """
    from inputs.generators._base import GROUND_TRUTH_WEIGHTS, PROFILING_WEIGHTS

    tier_runs: dict[str, list[list[Any]]] = {}
    for records, tier in zip(profiling_records, input_tiers, strict=False):
        tier_runs.setdefault(tier, []).append(records)

    # Redistribute extreme tier weight to present tiers
    gt_weights = dict(GROUND_TRUTH_WEIGHTS)
    extreme_weight = gt_weights.pop("extreme", 0.03)
    present_tiers = set(tier_runs.keys())
    if present_tiers:
        present_total = sum(gt_weights.get(t, 0) for t in present_tiers)
        if present_total > 0:
            for t in present_tiers:
                gt_weights[t] = gt_weights.get(t, 0) + extreme_weight * (
                    gt_weights.get(t, 0) / present_total
                )
    if "extreme" not in present_tiers:
        logger.warning(
            "No 'extreme' tier in profiling runs — redistributed %.1f%% weight",
            extreme_weight * 100,
        )

    resampled: list[list[Any]] = []

    for tier, runs in tier_runs.items():
        prof_weight = PROFILING_WEIGHTS.get(tier, 0)
        gt_weight = gt_weights.get(tier, 0)

        if prof_weight > 0:
            ratio = gt_weight / prof_weight
        else:
            ratio = 1.0

        target_count = max(1, round(len(runs) * ratio))
        # Repeat or truncate to hit target
        idx = 0
        for _ in range(target_count):
            resampled.append(runs[idx % len(runs)])
            idx += 1

    return resampled


def _compute_per_step_summary(
    stats: Any,
) -> dict[str, dict[str, Any]]:
    """Extract per-step cost summary from ProfilingStats."""
    summary = {}
    total_cost = sum(ss.cost.mean for ss in stats.step_stats.values())
    for name, ss in stats.step_stats.items():
        summary[name] = {
            "mean": ss.cost.mean,
            "median": ss.cost.p50,
            "p75": ss.cost.p75,
            "p95": ss.cost.p95,
            "model": ss.model,
            "call_count": ss.call_count,
            "cost_fraction": ss.cost.mean / total_cost if total_cost > 0 else 0,
        }
    return summary


def _compute_cost_input_correlations(
    run_costs: list[float],
    inputs: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Compute correlations between input characteristics and run costs."""
    result: dict[str, float | None] = {
        "token_count_vs_cost_r": None,
        "tier_vs_cost_anova_f": None,
    }

    token_counts = []
    for inp in inputs:
        tc = inp.get("token_count", 0)
        if tc == 0 and "input" in inp:
            tc = len(str(inp["input"])) // 4
        token_counts.append(tc)

    n = min(len(run_costs), len(token_counts))
    if n < 3:
        return result

    costs = run_costs[:n]
    tokens = token_counts[:n]

    # Pearson r between token count and cost
    mean_c = sum(costs) / n
    mean_t = sum(tokens) / n
    cov = sum((c - mean_c) * (t - mean_t) for c, t in zip(costs, tokens, strict=True)) / n
    std_c = math.sqrt(sum((c - mean_c) ** 2 for c in costs) / n)
    std_t = math.sqrt(sum((t - mean_t) ** 2 for t in tokens) / n)
    if std_c > 0 and std_t > 0:
        result["token_count_vs_cost_r"] = cov / (std_c * std_t)

    return result


async def _run_comparison(
    workflow_id: str,
    comparison: str,
    config: Any,
    profiling_records: list[list[Any]] | None,
    profiling_inputs: list[dict[str, Any]] | None,
    results_dir: Path,
    dry_run: bool,
    parallel: int,
) -> dict[str, Any]:
    """Execute one comparison (A, B, or C) for a single workflow."""
    from agentcost.projection.patterns import detect_patterns
    from agentcost.projection.stats import compute_stats
    from agentcost.validation.scoring import score_comparison
    from bt_agents.harness.run_workflow import (
        load_prompts,
        run_batch,
        save_results,
    )

    gt_n = _GROUND_TRUTH_N.get(workflow_id, 200)
    result: dict[str, Any] = {"comparison": comparison, "error": None}

    try:
        prompts = load_prompts(workflow_id)
    except Exception:
        prompts = {}

    # --- Profiling runs (shared across A, B, C) ---
    if profiling_records is None:
        prof_raw = _generate_inputs(workflow_id, _PROFILING_N, "profiling", seed=42)
        profiling_inputs = _inputs_to_dicts(prof_raw, dry_run)
        profiling_records = await run_batch(workflow_id, profiling_inputs, prompts, parallel)
        save_results(
            workflow_id,
            profiling_records,
            str(results_dir),
            inputs=profiling_inputs,
            prompts=prompts,
            backtest_profile="profiling",
        )

    profiling_stats = compute_stats(profiling_records)
    profiling_costs = _compute_run_costs(profiling_records)

    # --- Ground truth runs ---
    if comparison == "C":
        # Comparison C: no new API calls — reweight profiling runs
        input_tiers = [inp.get("tier", "easy") for inp in (profiling_inputs or [])]
        resampled = _reweight_runs(profiling_records, input_tiers)
        gt_stats = compute_stats(resampled)
        gt_costs = _compute_run_costs(resampled)
        result["note"] = "No API calls — re-analysis of profiling data with reweighting"
    else:
        profile_mode = "profiling" if comparison == "A" else "ground_truth"
        gt_seed = 1000 if comparison == "A" else 2000
        gt_raw = _generate_inputs(workflow_id, gt_n, profile_mode, seed=gt_seed)
        gt_input_dicts = _inputs_to_dicts(gt_raw, dry_run)
        gt_records = await run_batch(workflow_id, gt_input_dicts, prompts, parallel)
        save_results(
            workflow_id,
            gt_records,
            str(results_dir),
            inputs=gt_input_dicts,
            prompts=prompts,
            backtest_profile=f"ground_truth_{comparison.lower()}",
        )
        gt_stats = compute_stats(gt_records)
        gt_costs = _compute_run_costs(gt_records)

    # --- Score ---
    score = score_comparison(profiling_stats, gt_stats, workflow_id, comparison)
    result["score"] = score.to_dict()
    result["profiling_run_costs"] = profiling_costs
    result["ground_truth_run_costs"] = gt_costs
    result["profiling_n"] = len(profiling_records)
    result["ground_truth_n"] = len(gt_costs)

    # --- Patterns (only on ground truth for B, profiling for A/C) ---
    if comparison == "B":
        patterns = detect_patterns(
            gt_records if comparison != "C" else profiling_records, gt_stats
        )
    else:
        patterns = detect_patterns(profiling_records, profiling_stats)

    result["detected_patterns"] = [p.to_dict() for p in patterns]

    # --- Per-step costs ---
    result["step_costs"] = {name: ss.cost.mean for name, ss in gt_stats.step_stats.items()}

    if comparison == "B" and profiling_inputs:
        result["drift_config"] = {
            "tier_shift": True,
            "style_shift": True,
            "length_stretch": 1.5,
            "structural": None,
        }
    if comparison == "C":
        from inputs.generators._base import GROUND_TRUTH_WEIGHTS

        result["traffic_mix"] = list(GROUND_TRUTH_WEIGHTS.values())

    return result, profiling_records, profiling_inputs, profiling_stats


def _build_workflow_result(
    workflow_id: str,
    comparisons: dict[str, dict[str, Any]],
    profiling_stats: Any,
    profiling_records: list[list[Any]],
) -> dict[str, Any]:
    """Build the per-workflow result JSON compatible with visualization generators."""
    from agentcost.validation.scoring import ComparisonScore
    from agentcost.validation.suite import attribute_failure
    from tests.backtesting.detector_validation import results_to_dict, validate_detectors

    # Build ComparisonScores for failure attribution
    scores = {}
    for comp_key in ("A", "B", "C"):
        comp_data = comparisons.get(comp_key)
        if comp_data and comp_data.get("score"):
            scores[comp_key] = ComparisonScore.from_dict(comp_data["score"])
        else:
            scores[comp_key] = None

    # Failure attribution
    attribution = attribute_failure(
        workflow_id,
        scores.get("A"),
        scores.get("B"),
        scores.get("C"),
    )

    # Recovery percentage
    recovery_pct = None
    if scores.get("A") and scores.get("B") and scores.get("C"):
        from agentcost.validation.suite import _compute_recovery

        if not scores["B"].passes:
            recovery_pct = _compute_recovery(scores["A"], scores["B"], scores["C"])

    # Detector validation
    patterns_for_detection = []
    comp_b = comparisons.get("B", comparisons.get("A", {}))
    if comp_b.get("detected_patterns"):
        from agentcost.projection.patterns import DetectedPattern

        for pd in comp_b["detected_patterns"]:
            try:
                patterns_for_detection.append(
                    DetectedPattern(
                        **{
                            k: v
                            for k, v in pd.items()
                            if k in DetectedPattern.__dataclass_fields__
                        }
                    )
                )
            except Exception:
                pass

    detector_results = validate_detectors(workflow_id, patterns_for_detection)
    detector_dict = results_to_dict(detector_results)

    # Step stats from profiling
    step_stats_dict = {}
    if profiling_stats and profiling_stats.step_stats:
        for name, ss in profiling_stats.step_stats.items():
            step_stats_dict[name] = (
                ss.to_dict()
                if hasattr(ss, "to_dict")
                else {
                    "step_name": ss.step_name,
                    "model": ss.model,
                    "cost": {"mean": ss.cost.mean, "p50": ss.cost.p50, "p95": ss.cost.p95},
                }
            )

    # Step costs from best available comparison
    step_costs = {}
    for comp_key in ("B", "A", "C"):
        if comparisons.get(comp_key, {}).get("step_costs"):
            step_costs = comparisons[comp_key]["step_costs"]
            break

    return {
        "workflow_name": workflow_id,
        "comparisons": {
            k: {
                "score": v.get("score"),
                "profiling_run_costs": v.get("profiling_run_costs", []),
                "ground_truth_run_costs": v.get("ground_truth_run_costs", []),
            }
            for k, v in comparisons.items()
        },
        "detected_patterns": comp_b.get("detected_patterns", []),
        "step_costs": step_costs,
        "step_stats": step_stats_dict,
        "failure_attribution": attribution.to_dict() if attribution else None,
        "recovery_pct": recovery_pct,
        "detector_results": detector_dict,
    }


def _build_dataset_record(
    all_results: dict[str, dict[str, Any]],
    all_comparisons: dict[str, dict[str, dict[str, Any]]],
    all_inputs: dict[str, list[dict[str, Any]]],
    budget: Any,
) -> dict[str, Any]:
    """Build the long-term dataset record."""
    from bt_agents.harness.run_workflow import _pricing_table_hash
    from visualization.colors import WORKFLOW_GROUPS

    workflows_data: dict[str, Any] = {}
    passing_all = []
    need_reweight = []
    unresolved = []

    for wf_id, wf_result in all_results.items():
        attr = wf_result.get("failure_attribution")
        if attr is None:
            passing_all.append(wf_id)
        elif attr.get("bucket") == 2:
            need_reweight.append(wf_id)
        else:
            unresolved.append(wf_id)

        comparisons_data = {}
        for comp_key in ("A", "B", "C"):
            comp = all_comparisons.get(wf_id, {}).get(comp_key)
            if comp:
                comparisons_data[f"comparison_{comp_key.lower()}"] = {
                    "profiling_n": comp.get("profiling_n", _PROFILING_N),
                    "ground_truth_n": comp.get("ground_truth_n", 0),
                    "score": comp.get("score"),
                }

        # Per-run metadata
        per_run_meta = []
        inputs = all_inputs.get(wf_id, [])
        run_costs = []
        for comp_key in ("A", "B"):
            comp = all_comparisons.get(wf_id, {}).get(comp_key)
            if comp:
                run_costs = comp.get("ground_truth_run_costs", [])
                break

        for i, cost in enumerate(run_costs):
            inp = inputs[i] if i < len(inputs) else {}
            per_run_meta.append(
                {
                    "input_tier": inp.get("tier", "unknown"),
                    "token_count": inp.get("token_count", 0),
                    "total_cost": cost,
                }
            )

        workflows_data[wf_id] = {
            "pattern_type": WORKFLOW_GROUPS.get(wf_id, "linear"),
            **comparisons_data,
            "failure_attribution": wf_result.get("failure_attribution"),
            "recovery_pct": wf_result.get("recovery_pct"),
            "detector_results": wf_result.get("detector_results", {}),
            "detected_patterns_raw": wf_result.get("detected_patterns", []),
            "per_step_cost_summary": wf_result.get("step_costs", {}),
            "per_run_metadata": per_run_meta[:50],
        }

    # Detector rates
    from tests.backtesting.detector_validation import (
        DetectorResult,
        compute_detector_rates,
    )

    all_detector_results = []
    for wf_result in all_results.values():
        dr = wf_result.get("detector_results", {})
        for det_name, det_data in dr.items():
            if isinstance(det_data, dict):
                all_detector_results.append(
                    DetectorResult(
                        workflow=wf_result.get("workflow_name", ""),
                        detector=det_name,
                        fired=det_data.get("fired", False),
                        expected=det_data.get("expected", False),
                        classification=det_data.get("classification", "TN"),
                        raw_statistic=det_data.get("raw_statistic"),
                        threshold=det_data.get("threshold"),
                    )
                )
    rates = compute_detector_rates(all_detector_results)

    return {
        "meta": {
            "backtest_id": uuid.uuid4().hex,
            "backtest_version": "1.0",
            "timestamp": datetime.now(UTC).isoformat(),
            "engine_version": "0.1.0",
            "pricing_table_hash": _pricing_table_hash(),
            "input_seed": 42,
            "python_version": platform.python_version(),
        },
        "workflows": workflows_data,
        "aggregate": {
            "total_cost_usd": budget.spent if budget else 0,
            "workflows_passing_all": passing_all,
            "workflows_need_reweight": need_reweight,
            "workflows_unresolved": unresolved,
            "detector_tp_rate": rates.get("tp_rate", 0),
            "detector_fn_rate": rates.get("fn_rate", 0),
            "launch_gate": "PASSED" if not unresolved else "BLOCKED",
        },
    }


@click.command()
@click.option("--workflow", type=str, default=None, help="Single workflow (e.g., 'W1').")
@click.option("--all", "run_all", is_flag=True, default=False, help="Run all 14 workflows.")
@click.option(
    "--comparison",
    type=str,
    default="all",
    help="Which comparison: A, B, C, or all (default: all).",
)
@click.option("--dry-run", is_flag=True, default=False, help="Validate without API calls.")
@click.option(
    "--results-dir",
    type=click.Path(),
    default="tests/backtesting/results/backtest",
    help="Output directory.",
)
@click.option(
    "--pilot-dir",
    type=click.Path(),
    default="tests/backtesting/results/pilot",
    help="Pilot results directory.",
)
@click.option("--budget-limit", type=float, default=500.0, help="Max spend in USD.")
@click.option("--resume", is_flag=True, default=False, help="Skip existing results.")
@click.option("--parallel", type=int, default=1, help="Concurrent runs per workflow.")
@click.option("-v", "--verbose", is_flag=True, default=False, help="Debug logging.")
def main(
    workflow: str | None,
    run_all: bool,
    comparison: str,
    dry_run: bool,
    results_dir: str,
    pilot_dir: str,
    budget_limit: float,
    resume: bool,
    parallel: int,
    verbose: bool,
) -> None:
    """Run the three-comparison backtest protocol."""
    _load_dotenv()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    if not run_all and not workflow:
        click.echo("Specify --workflow W1 or --all.", err=True)
        sys.exit(1)

    from tests.backtesting.budget_tracker import BudgetTracker
    from tests.backtesting.configs import BACKTESTING_CONFIGS

    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine workflows
    if run_all:
        workflow_ids = [_wf_id(c.name) for c in BACKTESTING_CONFIGS]
    else:
        workflow_ids = [workflow.upper()]

    comparisons_to_run = ["A", "B", "C"] if comparison.upper() == "ALL" else [comparison.upper()]

    budget = BudgetTracker(limit=budget_limit)

    all_results: dict[str, dict[str, Any]] = {}
    all_comparisons: dict[str, dict[str, dict[str, Any]]] = {}
    all_inputs: dict[str, list[dict[str, Any]]] = {}

    from tests.backtesting.concurrency import (
        build_concurrent_groups,
        get_parallel_for_workflow,
    )

    # Helper: run a comparison for a group of non-conflicting workflows concurrently
    async def _run_comparison_group(
        group: list[str],
        comp: str,
        prof_data: dict[str, tuple[Any, Any]],
    ) -> list[tuple[str, dict[str, Any], Any, Any, Any]]:
        """Run one comparison for all workflows in a concurrent group."""

        async def _one(wf_id: str) -> tuple[str, dict[str, Any], Any, Any, Any]:
            cfg = _config_for_workflow(wf_id)
            pr, pi = prof_data.get(wf_id, (None, None))
            wf_parallel = parallel if parallel > 1 else get_parallel_for_workflow(wf_id)
            return (
                wf_id,
                *await _run_comparison(
                    wf_id,
                    comp,
                    cfg,
                    pr,
                    pi,
                    out_dir,
                    dry_run,
                    wf_parallel,
                ),
            )

        results = await asyncio.gather(
            *[_one(wf_id) for wf_id in group],
            return_exceptions=True,
        )
        out = []
        for r in results:
            if isinstance(r, Exception):
                logger.error("Comparison %s group error: %s", comp, r)
            else:
                out.append(r)
        return out

    # --- Run Comparison A for all workflows (grouped concurrently) ---
    if "A" in comparisons_to_run:
        click.echo("=== Comparison A (no drift) ===")
        a_scores: dict[str, bool] = {}
        eligible_a = [
            wf
            for wf in workflow_ids
            if _config_for_workflow(wf) is not None
            and not (resume and (out_dir / f"{wf.lower()}_comparison_A.json").exists())
            and (dry_run or _check_validation_gate())
        ]
        groups_a = build_concurrent_groups(eligible_a)

        for group_idx, group in enumerate(groups_a):
            click.echo(f"  Group {group_idx + 1}/{len(groups_a)}: [{', '.join(group)}]")
            group_results = asyncio.run(_run_comparison_group(group, "A", {}))

            for wf_id, comp_result, prof_rec, prof_inp, _prof_st in group_results:
                cost = sum(comp_result.get("profiling_run_costs", [])) + sum(
                    comp_result.get("ground_truth_run_costs", [])
                )
                budget.record(wf_id, "A", cost)
                all_comparisons.setdefault(wf_id, {})["A"] = comp_result
                all_inputs[wf_id] = prof_inp or []
                all_results[wf_id] = prof_rec

                passed = comp_result.get("score", {}).get("passes", False)
                a_scores[wf_id] = passed
                me = comp_result["score"].get("mean_error_pct", 0)
                status = "PASS" if passed else "FAIL"
                click.echo(f"    {wf_id}: {status} (mean_error={me:.1f}%)")

            if budget.check_limit():
                click.echo(f"  Budget limit ${budget_limit:.0f} reached.", err=True)
                break

        if len(a_scores) >= 3:
            should_stop, msg = budget.check_comparison_a_gate(a_scores)
            if should_stop:
                click.echo(f"  STOPPED: {msg}", err=True)
                sys.exit(1)

        click.echo(f"  Comparison A complete. Spent: ${budget.spent:.2f}")

    # --- Run Comparison B for all workflows (grouped concurrently) ---
    if "B" in comparisons_to_run:
        click.echo("\n=== Comparison B (drifted) ===")
        eligible_b = [
            wf
            for wf in workflow_ids
            if _config_for_workflow(wf) is not None
            and not (resume and (out_dir / f"{wf.lower()}_comparison_B.json").exists())
        ]
        groups_b = build_concurrent_groups(eligible_b)
        prof_data_b = {wf: (all_results.get(wf), all_inputs.get(wf)) for wf in eligible_b}

        for group_idx, group in enumerate(groups_b):
            click.echo(f"  Group {group_idx + 1}/{len(groups_b)}: [{', '.join(group)}]")
            group_results = asyncio.run(_run_comparison_group(group, "B", prof_data_b))

            for wf_id, comp_result, *_ in group_results:
                cost = sum(comp_result.get("ground_truth_run_costs", []))
                budget.record(wf_id, "B", cost)
                all_comparisons.setdefault(wf_id, {})["B"] = comp_result

                passed = comp_result.get("score", {}).get("passes", False)
                me = comp_result["score"].get("mean_error_pct", 0)
                status = "PASS" if passed else "FAIL"
                click.echo(f"    {wf_id}: {status} (mean_error={me:.1f}%)")

            if budget.check_limit():
                click.echo("  Budget limit reached.", err=True)
                break

        cheap_scores = {
            wf: all_comparisons.get(wf, {}).get("B", {}).get("score", {}).get("passes", True)
            for wf in _CHEAP_WORKFLOWS
            if wf in all_comparisons
        }
        if cheap_scores:
            should_stop, msg = budget.check_comparison_b_cheap_gate(cheap_scores)
            if should_stop:
                click.echo(f"  WARNING: {msg}", err=True)

        click.echo(f"  Comparison B complete. Spent: ${budget.spent:.2f}")

    # --- Run Comparison C for workflows that failed B (no API calls) ---
    if "C" in comparisons_to_run:
        click.echo("\n=== Comparison C (reweighted, no API calls) ===")
        for wf_id in workflow_ids:
            b_data = all_comparisons.get(wf_id, {}).get("B")
            if b_data is None:
                continue
            if b_data.get("score", {}).get("passes", True):
                click.echo(f"  {wf_id}/C: B passed, skipping")
                continue

            config = _config_for_workflow(wf_id)
            if config is None:
                continue

            click.echo(f"  {wf_id}/C: reweighting (no API calls)")

            comp_result, _, _, _ = asyncio.run(
                _run_comparison(
                    wf_id,
                    "C",
                    config,
                    None,
                    all_inputs.get(wf_id),
                    out_dir,
                    dry_run,
                    parallel,
                )
            )
            # C costs $0
            budget.record(wf_id, "C", 0.0)
            all_comparisons.setdefault(wf_id, {})["C"] = comp_result

            passed = comp_result.get("score", {}).get("passes", False)
            status = "PASS" if passed else "FAIL"
            me = comp_result["score"].get("mean_error_pct", 0)
            click.echo(f"    {status} (mean_error={me:.1f}%)")

        click.echo("  Comparison C complete (cost: $0.00)")

    # --- Build per-workflow result JSONs for visualization ---
    click.echo("\n=== Building results ===")
    for wf_id in workflow_ids:
        wf_comparisons = all_comparisons.get(wf_id, {})
        if not wf_comparisons:
            continue

        wf_result = _build_workflow_result(wf_id, wf_comparisons, None, [])
        all_results[wf_id] = wf_result

        result_path = out_dir / f"{wf_id.lower()}_comparison_all.json"
        result_path.write_text(json.dumps(wf_result, indent=2, default=str))
        click.echo(f"  {wf_id}: {result_path}")

    # --- Generate backtest visualizations ---
    plots_dir = out_dir / "plots"
    try:
        from visualization.backtest.visualize_backtest import generate_all_backtest_visuals

        paths = generate_all_backtest_visuals(out_dir, plots_dir)
        click.echo(f"  Backtest plots: {len(paths)} files")
    except Exception as exc:
        click.echo(f"  Backtest plots failed: {exc}", err=True)

    # --- Write to long-term dataset ---
    try:
        from tests.backtesting.dataset import save_backtest_run

        dataset_record = _build_dataset_record(
            all_results,
            all_comparisons,
            all_inputs,
            budget,
        )
        ds_path = save_backtest_run(dataset_record)
        click.echo(f"  Dataset record: {ds_path}")
    except Exception as exc:
        click.echo(f"  Dataset write failed: {exc}", err=True)

    # --- Summary ---
    passing = [wf for wf, r in all_results.items() if r.get("failure_attribution") is None]
    reweight = [
        wf
        for wf, r in all_results.items()
        if r.get("failure_attribution", {}) and r["failure_attribution"].get("bucket") == 2
    ]
    unresolved = [
        wf
        for wf, r in all_results.items()
        if r.get("failure_attribution", {}) and r["failure_attribution"].get("bucket") in (1, 3)
    ]

    click.echo("\n=== Summary ===")
    click.echo(f"  Passing: {len(passing)} ({', '.join(passing) or 'none'})")
    click.echo(f"  Need reweighting: {len(reweight)} ({', '.join(reweight) or 'none'})")
    click.echo(f"  Unresolved: {len(unresolved)} ({', '.join(unresolved) or 'none'})")
    click.echo(f"  Total spend: ${budget.spent:.2f}")
    click.echo(f"  Launch gate: {'PASSED' if not unresolved else 'BLOCKED'}")


if __name__ == "__main__":
    main()
