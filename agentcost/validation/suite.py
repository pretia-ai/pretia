"""Run backtesting protocol across test workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from agentcost.projection.stats import ProfilingStats
from agentcost.store import ProfilingSession
from agentcost.validation.scoring import (
    _THRESHOLDS,
    CalibrationScore,
    ComparisonScore,
    bootstrap_percentile_ci,
    format_calibration_report,
    score_projection,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    """Description of one test workflow."""

    name: str
    archetype: str
    complexity: str
    workflow_path: str
    description: str
    expected_models: list[str]
    has_loops: bool
    expected_cost_range: tuple[float, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "name": self.name,
            "archetype": self.archetype,
            "complexity": self.complexity,
            "workflow_path": self.workflow_path,
            "description": self.description,
            "expected_models": list(self.expected_models),
            "has_loops": self.has_loops,
            "expected_cost_range": list(self.expected_cost_range),
        }


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Backtesting result for one workflow."""

    config: BacktestConfig
    synth20_score: CalibrationScore | None
    synth100_score: CalibrationScore | None
    ground_truth_stats: ProfilingStats | None
    convergence_20_to_100: float | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "config": self.config.to_dict(),
            "synth20_score": self.synth20_score.to_dict() if self.synth20_score else None,
            "synth100_score": self.synth100_score.to_dict() if self.synth100_score else None,
            "ground_truth_stats": (
                self.ground_truth_stats.to_dict() if self.ground_truth_stats else None
            ),
            "convergence_20_to_100": self.convergence_20_to_100,
        }


@dataclass(frozen=True, slots=True)
class BacktestSuiteResult:
    """Results for the full backtesting suite."""

    results: list[BacktestResult]
    overall_verdict: str
    pass_count: int
    warn_count: int
    fail_count: int
    launch_gate: bool
    timestamp: str
    hard_gate_passed: bool = True
    soft_gate_pass_rates: dict[str, float] = field(default_factory=dict)
    overall_passed: bool = True
    directional_bias: dict[str, Any] | None = None
    ground_truth_cis: dict[str, dict[str, tuple[float, float, float]]] = field(
        default_factory=dict,
    )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "results": [r.to_dict() for r in self.results],
            "overall_verdict": self.overall_verdict,
            "pass_count": self.pass_count,
            "warn_count": self.warn_count,
            "fail_count": self.fail_count,
            "launch_gate": self.launch_gate,
            "timestamp": self.timestamp,
            "hard_gate_passed": self.hard_gate_passed,
            "soft_gate_pass_rates": self.soft_gate_pass_rates,
            "overall_passed": self.overall_passed,
            "directional_bias": self.directional_bias,
            "ground_truth_cis": {
                wf: {k: list(v) for k, v in pcts.items()}
                for wf, pcts in self.ground_truth_cis.items()
            },
        }


@dataclass(frozen=True, slots=True)
class FailureAttribution:
    """Classify why a workflow failed the backtest into actionable buckets."""

    workflow_name: str
    bucket: int
    bucket_label: str
    explanation: str
    recommended_action: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow_name": self.workflow_name,
            "bucket": self.bucket,
            "bucket_label": self.bucket_label,
            "explanation": self.explanation,
            "recommended_action": self.recommended_action,
        }


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    """Backtest results across all three comparisons for one workflow."""

    workflow_name: str
    score_a: ComparisonScore | None
    score_b: ComparisonScore | None
    score_c: ComparisonScore | None
    attribution: FailureAttribution | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "workflow_name": self.workflow_name,
            "score_a": self.score_a.to_dict() if self.score_a else None,
            "score_b": self.score_b.to_dict() if self.score_b else None,
            "score_c": self.score_c.to_dict() if self.score_c else None,
            "attribution": self.attribution.to_dict() if self.attribution else None,
        }


def attribute_failure(
    workflow_name: str,
    score_a: ComparisonScore | None,
    score_b: ComparisonScore | None,
    score_c: ComparisonScore | None,
) -> FailureAttribution | None:
    """Apply the failure attribution flowchart from the backtest protocol.

    Returns None if all comparisons pass (no failure to attribute).
    """
    if score_a is None:
        return FailureAttribution(
            workflow_name=workflow_name,
            bucket=1,
            bucket_label="engine_problem",
            explanation="No Comparison A score available — engine or infrastructure problem.",
            recommended_action="Fix the projection engine or data pipeline.",
        )

    if not score_a.passes:
        return FailureAttribution(
            workflow_name=workflow_name,
            bucket=1,
            bucket_label="engine_problem",
            explanation=(
                f"Comparison A (no drift) failed: {', '.join(score_a.failures)}. "
                "Engine cannot project accurately even without distribution mismatch."
            ),
            recommended_action="Fix the projection engine. Common causes: pricing table error, "
            "token counting bug, detector threshold miscalibration.",
        )

    if score_b is None or score_b.passes:
        return None

    if score_c is not None and score_c.passes:
        recovery_pct = _compute_recovery(score_a, score_b, score_c)
        return FailureAttribution(
            workflow_name=workflow_name,
            bucket=2,
            bucket_label="drift_sensitivity",
            explanation=(
                f"Comparison B (drifted) failed but reweighting recovered "
                f"{recovery_pct:.0f}% of lost accuracy. Drift is distributional, "
                "not structural."
            ),
            recommended_action="Use --traffic-mix to specify production distribution weights.",
        )

    recovery_pct = _compute_recovery(score_a, score_b, score_c) if score_c is not None else 0.0
    if score_c is not None and recovery_pct >= 50.0:
        return FailureAttribution(
            workflow_name=workflow_name,
            bucket=2,
            bucket_label="drift_sensitivity",
            explanation=(
                f"Comparison C recovered {recovery_pct:.0f}% of lost accuracy. "
                "Reweighting partially compensates for drift."
            ),
            recommended_action="Use --traffic-mix to specify production distribution weights.",
        )

    return FailureAttribution(
        workflow_name=workflow_name,
        bucket=3,
        bucket_label="structural_drift",
        explanation=(
            f"Comparison B failed and reweighting recovered only "
            f"{recovery_pct:.0f}% of lost accuracy. "
            "Drift is structural (style, length, modality), not distributional."
        ),
        recommended_action="Re-profile with production-representative inputs. "
        "Investigate per-step cost breakdown to identify which step is misprojected.",
    )


def _compute_recovery(
    score_a: ComparisonScore,
    score_b: ComparisonScore,
    score_c: ComparisonScore,
) -> float:
    """Compute what percentage of accuracy lost to drift was recovered by reweighting."""
    drift_impact = score_b.mean_error_pct - score_a.mean_error_pct
    if drift_impact <= 0:
        return 100.0
    recovery = score_b.mean_error_pct - score_c.mean_error_pct
    return max(0.0, min(100.0, recovery / drift_impact * 100))


def check_directional_bias(
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check for systematic over- or under-estimation across workflows."""
    count_above = 0
    count_below = 0
    per_workflow: list[dict[str, Any]] = []

    for r in results:
        proj = r["projected_p95"]
        gt = r["ground_truth_p95"]
        if proj > gt:
            direction = "above"
            count_above += 1
        else:
            direction = "below"
            count_below += 1
        per_workflow.append(
            {
                "name": r["workflow_name"],
                "projected_p95": proj,
                "ground_truth_p95": gt,
                "direction": direction,
            }
        )

    total = len(results)
    bias: str | None = None
    if total > 0:
        if count_below >= 0.8 * total:
            bias = "underestimation"
        elif count_above >= 0.8 * total:
            bias = "overestimation"

    return {
        "count_above": count_above,
        "count_below": count_below,
        "total": total,
        "bias_detected": bias,
        "per_workflow": per_workflow,
    }


def _extract_stats(session: ProfilingSession) -> ProfilingStats | None:
    """Reconstruct ProfilingStats from session metadata if possible."""
    meta = session.metadata or {}
    stats_dict = meta.get("stats")
    if stats_dict is None:
        return None

    from agentcost.projection.stats import (
        PercentileStats,
        RunStats,
        StepStats,
    )

    step_stats: dict[str, StepStats] = {}
    for name, ss in stats_dict.get("step_stats", {}).items():

        def _ps(d: dict) -> PercentileStats:
            return PercentileStats(
                min=d["min"],
                max=d["max"],
                mean=d["mean"],
                std=d["std"],
                p50=d["p50"],
                p75=d["p75"],
                p90=d["p90"],
                p95=d["p95"],
                p99=d["p99"],
            )

        step_stats[name] = StepStats(
            step_name=ss["step_name"],
            step_type=ss["step_type"],
            model=ss["model"],
            call_count=ss["call_count"],
            runs_present=ss["runs_present"],
            input_tokens=_ps(ss["input_tokens"]),
            output_tokens=_ps(ss["output_tokens"]),
            total_tokens=_ps(ss["total_tokens"]),
            cost=_ps(ss["cost"]),
            duration_ms=_ps(ss["duration_ms"]),
            context_size=_ps(ss["context_size"]),
            iterations_per_run=_ps(ss["iterations_per_run"]),
            mean_iterations=ss["mean_iterations"],
        )

    run_stats_list: list[RunStats] = []
    for rs in stats_dict.get("run_stats", []):
        run_stats_list.append(
            RunStats(
                run_index=rs["run_index"],
                total_cost=rs["total_cost"],
                total_tokens=rs["total_tokens"],
                total_input_tokens=rs["total_input_tokens"],
                total_output_tokens=rs["total_output_tokens"],
                step_count=rs["step_count"],
                duration_ms=rs["duration_ms"],
            )
        )

    cpr_dict = stats_dict.get("cost_per_run")
    tpr_dict = stats_dict.get("tokens_per_run")

    def _maybe_ps(d: dict | None) -> PercentileStats | None:
        if d is None:
            return None
        return PercentileStats(
            min=d["min"],
            max=d["max"],
            mean=d["mean"],
            std=d["std"],
            p50=d["p50"],
            p75=d["p75"],
            p90=d["p90"],
            p95=d["p95"],
            p99=d["p99"],
        )

    return ProfilingStats(
        step_stats=step_stats,
        run_stats=run_stats_list,
        cost_per_run=_maybe_ps(cpr_dict),
        tokens_per_run=_maybe_ps(tpr_dict),
        total_runs=stats_dict.get("total_runs", 0),
        total_steps=stats_dict.get("total_steps", 0),
    )


def run_backtesting_suite(
    profiles: dict[str, dict[str, ProfilingSession]],
    configs: list[BacktestConfig],
    traffic: int = 1000,
) -> BacktestSuiteResult:
    """Run the backtesting protocol on pre-computed profiles."""
    results: list[BacktestResult] = []
    bias_data: list[dict[str, Any]] = []
    gt_cis: dict[str, dict[str, tuple[float, float, float]]] = {}

    for cfg in configs:
        wf_profiles = profiles.get(cfg.name)
        if wf_profiles is None:
            logger.warning("No profiles for workflow %r — skipping", cfg.name)
            continue

        synth20_session = wf_profiles.get("synth20")
        synth100_session = wf_profiles.get("synth100")
        real500_session = wf_profiles.get("real500")

        if real500_session is None:
            logger.warning("No ground truth (real500) for %r — skipping", cfg.name)
            continue

        gt_stats = _extract_stats(real500_session)
        if gt_stats is None:
            logger.warning("Cannot extract stats from real500 for %r", cfg.name)
            continue

        gt_costs = [rs.total_cost for rs in gt_stats.run_stats]

        p50_ci: tuple[float, float] | None = None
        p95_ci: tuple[float, float] | None = None
        if len(gt_costs) >= 15:
            p50_result = bootstrap_percentile_ci(gt_costs, 50)
            p95_result = bootstrap_percentile_ci(gt_costs, 95)
            p50_ci = (p50_result[1], p50_result[2])
            p95_ci = (p95_result[1], p95_result[2])
            gt_cis[cfg.name] = {
                "p50": p50_result,
                "p95": p95_result,
            }

        synth20_score: CalibrationScore | None = None
        synth100_score: CalibrationScore | None = None
        convergence: float | None = None

        if synth20_session is not None:
            s20_stats = _extract_stats(synth20_session)
            if s20_stats is not None:
                synth20_score = score_projection(
                    s20_stats,
                    gt_stats,
                    traffic=traffic,
                    workflow_complexity=cfg.complexity,
                    ground_truth_p50_ci=p50_ci,
                    ground_truth_p95_ci=p95_ci,
                )

        if synth100_session is not None:
            s100_stats = _extract_stats(synth100_session)
            if s100_stats is not None:
                synth100_score = score_projection(
                    s100_stats,
                    gt_stats,
                    traffic=traffic,
                    workflow_complexity=cfg.complexity,
                    ground_truth_p50_ci=p50_ci,
                    ground_truth_p95_ci=p95_ci,
                )

                if synth20_session is not None:
                    s20_stats_conv = _extract_stats(synth20_session)
                    if (
                        s20_stats_conv is not None
                        and s20_stats_conv.cost_per_run is not None
                        and s100_stats.cost_per_run is not None
                        and s100_stats.cost_per_run.p50 > 0
                    ):
                        convergence = (
                            abs(s20_stats_conv.cost_per_run.p50 - s100_stats.cost_per_run.p50)
                            / s100_stats.cost_per_run.p50
                            * 100
                        )

        if synth20_score is not None:
            proj_p95 = (
                s20_stats.cost_per_run.p95
                if s20_stats is not None and s20_stats.cost_per_run is not None
                else 0
            )
            gt_p95 = gt_stats.cost_per_run.p95 if gt_stats.cost_per_run else 0
            bias_data.append(
                {
                    "workflow_name": cfg.name,
                    "projected_p95": proj_p95,
                    "ground_truth_p95": gt_p95,
                }
            )

        results.append(
            BacktestResult(
                config=cfg,
                synth20_score=synth20_score,
                synth100_score=synth100_score,
                ground_truth_stats=gt_stats,
                convergence_20_to_100=convergence,
            )
        )

    scored = [r for r in results if r.synth20_score is not None]
    n_scored = len(scored)

    pass_count = sum(1 for r in scored if r.synth20_score.verdict == "PASS")
    warn_count = sum(1 for r in scored if r.synth20_score.verdict == "WARN")
    fail_count = sum(1 for r in scored if r.synth20_score.verdict == "FAIL")

    # --- Hard gates: p50 ratio + top step must pass for every workflow ---
    hard_gate_passed = (
        all(
            0.7 <= r.synth20_score.p50_ratio <= 2.0 and r.synth20_score.top_step_correct
            for r in scored
        )
        if scored
        else True
    )

    # --- Soft gates: p95 coverage, range ratio, step ranking at ≥80% pass rate ---
    soft_rates: dict[str, float] = {}
    if n_scored > 0:
        p95_pass = 0
        range_pass = 0
        rank_pass = 0
        for r in scored:
            th = _THRESHOLDS.get(r.config.complexity, _THRESHOLDS["simple"])
            if r.synth20_score.p95_coverage >= th["p95_coverage"]:
                p95_pass += 1
            if r.synth20_score.range_ratio < th["range_ratio"]:
                range_pass += 1
            if r.synth20_score.ranking_correlation > 0.7:
                rank_pass += 1
        soft_rates = {
            "p95_coverage": p95_pass / n_scored,
            "range_ratio": range_pass / n_scored,
            "step_ranking": rank_pass / n_scored,
        }

    soft_gates_pass = all(v >= 0.80 for v in soft_rates.values()) if soft_rates else True
    overall_passed = hard_gate_passed and soft_gates_pass

    if fail_count > 0:
        overall = "FAIL"
    elif warn_count > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    dir_bias = check_directional_bias(bias_data) if bias_data else None

    return BacktestSuiteResult(
        results=results,
        overall_verdict=overall,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        launch_gate=overall_passed,
        timestamp=datetime.now(UTC).isoformat(),
        hard_gate_passed=hard_gate_passed,
        soft_gate_pass_rates=soft_rates,
        overall_passed=overall_passed,
        directional_bias=dir_bias,
        ground_truth_cis=gt_cis,
    )


def format_suite_report(suite_result: BacktestSuiteResult) -> str:
    """Format a full backtesting suite report."""
    synth20_scores = [r.synth20_score for r in suite_result.results if r.synth20_score is not None]
    lines: list[str] = []

    if synth20_scores:
        lines.append(format_calibration_report(synth20_scores))

    lines.append("")
    lines.append("20-sample vs 100-sample convergence:")
    for r in suite_result.results:
        name = r.config.name[:30]
        if r.convergence_20_to_100 is not None:
            conv = r.convergence_20_to_100
            flag = " ⚠ (recommend 50+ samples)" if conv > 30 else " (20 samples sufficient)"
            lines.append(f"  {name}: {conv:.0f}% difference{flag}")
        else:
            lines.append(f"  {name}: N/A")

    if suite_result.directional_bias:
        bias = suite_result.directional_bias
        if bias["bias_detected"]:
            lines.append("")
            lines.append(f"⚠ Directional bias: {bias['bias_detected']}")
            lines.append(
                f"  {bias['count_above']} above, {bias['count_below']} below "
                f"(of {bias['total']} workflows)"
            )

    return "\n".join(lines)
