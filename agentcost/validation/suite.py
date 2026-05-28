"""Run backtesting protocol across test workflows."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from agentcost.projection.stats import ProfilingStats
from agentcost.store import ProfilingSession
from agentcost.validation.scoring import (
    CalibrationScore,
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

    for cfg in configs:
        wf_profiles = profiles.get(cfg.name)
        if wf_profiles is None:
            logger.warning("No profiles for workflow %r — skipping", cfg.name)
            continue

        synth20_session = wf_profiles.get("synth20")
        synth100_session = wf_profiles.get("synth100")
        real500_session = wf_profiles.get("real500")

        if real500_session is None:
            logger.warning(
                "No ground truth (real500) for %r — skipping",
                cfg.name,
            )
            continue

        gt_stats = _extract_stats(real500_session)
        if gt_stats is None:
            logger.warning("Cannot extract stats from real500 for %r", cfg.name)
            continue

        synth20_score: CalibrationScore | None = None
        synth100_score: CalibrationScore | None = None
        convergence: float | None = None

        if synth20_session is not None:
            s20_stats = _extract_stats(synth20_session)
            if s20_stats is not None:
                synth20_score = score_projection(s20_stats, gt_stats, traffic=traffic)

        if synth100_session is not None:
            s100_stats = _extract_stats(synth100_session)
            if s100_stats is not None:
                synth100_score = score_projection(
                    s100_stats,
                    gt_stats,
                    traffic=traffic,
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

        results.append(
            BacktestResult(
                config=cfg,
                synth20_score=synth20_score,
                synth100_score=synth100_score,
                ground_truth_stats=gt_stats,
                convergence_20_to_100=convergence,
            )
        )

    pass_count = sum(1 for r in results if r.synth20_score and r.synth20_score.verdict == "PASS")
    warn_count = sum(1 for r in results if r.synth20_score and r.synth20_score.verdict == "WARN")
    fail_count = sum(1 for r in results if r.synth20_score and r.synth20_score.verdict == "FAIL")

    if fail_count > 0:
        overall = "FAIL"
    elif warn_count > 0:
        overall = "WARN"
    else:
        overall = "PASS"

    return BacktestSuiteResult(
        results=results,
        overall_verdict=overall,
        pass_count=pass_count,
        warn_count=warn_count,
        fail_count=fail_count,
        launch_gate=fail_count == 0,
        timestamp=datetime.now(UTC).isoformat(),
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

    return "\n".join(lines)
