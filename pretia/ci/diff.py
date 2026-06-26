"""Compare two cost profiles and compute per-step deltas."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from pretia.ci.baseline import Baseline, create_baseline, parse_traffic
from pretia.ci.report import format_cost
from pretia.store import ProfilingSession


def _normal_cdf(z: float) -> float:
    """Approximate CDF of standard normal (Abramowitz & Stegun 26.2.17)."""
    if z < 0:
        return 1 - _normal_cdf(-z)
    t = 1 / (1 + 0.2316419 * z)
    poly = t * (
        0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429)))
    )
    return 1 - poly * math.exp(-z * z / 2) / math.sqrt(2 * math.pi)


def _rank_values(values: list[float]) -> list[float]:
    """Assign 1-based ranks with tie averaging."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[j]]:
            j += 1
        avg_rank = sum(range(i + 1, j + 2)) / (j - i + 1)
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def mann_whitney_u(x: list[float], y: list[float]) -> float:
    """Compute Mann-Whitney U test p-value (two-tailed, normal approximation)."""
    n1 = len(x)
    n2 = len(y)
    if n1 < 2 or n2 < 2:
        return 1.0

    combined = x + y
    ranks = _rank_values(combined)
    r1 = sum(ranks[i] for i in range(n1))
    u1 = r1 - n1 * (n1 + 1) / 2
    u = min(u1, n1 * n2 - u1)

    mu_u = n1 * n2 / 2
    sigma_u = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    if sigma_u == 0:
        return 1.0

    z = (u - mu_u) / sigma_u
    return 2 * (1 - _normal_cdf(abs(z)))


def significance_label(p_value: float) -> str:
    """Map p-value to human-readable significance label."""
    if p_value < 0.05:
        return "significant"
    if p_value < 0.10:
        return "possibly significant"
    return "not significant"


@dataclass(frozen=True, slots=True)
class ModelChange:
    """A model change detected for one step."""

    step_name: str
    old_model: str
    new_model: str
    cost_impact: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "step_name": self.step_name,
            "old_model": self.old_model,
            "new_model": self.new_model,
            "cost_impact": self.cost_impact,
        }


@dataclass(frozen=True, slots=True)
class PatternChanges:
    """Patterns added, resolved, or unchanged between baseline and new profile."""

    new_patterns: list[str]
    resolved_patterns: list[str]
    unchanged_patterns: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "new_patterns": list(self.new_patterns),
            "resolved_patterns": list(self.resolved_patterns),
            "unchanged_patterns": list(self.unchanged_patterns),
        }


@dataclass(frozen=True, slots=True)
class StepDiff:
    """One step's diff between baseline and new profile."""

    step_name: str
    cost_change_pct: float
    cost_change_abs: float
    token_change_pct: float
    iteration_change: float
    model_changed: bool
    old_model: str | None
    new_model: str | None
    flags: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "step_name": self.step_name,
            "cost_change_pct": self.cost_change_pct,
            "cost_change_abs": self.cost_change_abs,
            "token_change_pct": self.token_change_pct,
            "iteration_change": self.iteration_change,
            "model_changed": self.model_changed,
            "old_model": self.old_model,
            "new_model": self.new_model,
            "flags": list(self.flags),
        }


@dataclass(frozen=True, slots=True)
class DiffResult:
    """Full comparison between a baseline and a new profile."""

    baseline_workflow: str
    baseline_date: str
    new_date: str
    total_monthly_change: dict[str, float]
    total_monthly_pct_change: dict[str, float]
    step_diffs: dict[str, StepDiff]
    new_steps: list[str]
    removed_steps: list[str]
    model_changes: list[ModelChange]
    pattern_changes: PatternChanges
    exceeds_threshold: bool | None
    summary: str
    traffic: int = 1000

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "baseline_workflow": self.baseline_workflow,
            "baseline_date": self.baseline_date,
            "new_date": self.new_date,
            "total_monthly_change": dict(self.total_monthly_change),
            "total_monthly_pct_change": dict(self.total_monthly_pct_change),
            "step_diffs": {k: v.to_dict() for k, v in self.step_diffs.items()},
            "new_steps": list(self.new_steps),
            "removed_steps": list(self.removed_steps),
            "model_changes": [m.to_dict() for m in self.model_changes],
            "pattern_changes": self.pattern_changes.to_dict(),
            "exceeds_threshold": self.exceeds_threshold,
            "summary": self.summary,
            "traffic": self.traffic,
        }


def _pct_change(old: float, new: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else 100.0
    return ((new - old) / abs(old)) * 100


def diff_baseline(
    baseline: Baseline,
    new_session: ProfilingSession,
    traffic: int | None = None,
) -> DiffResult:
    """Compare a baseline against a new profiling session."""
    new_meta = new_session.metadata or {}
    new_stats = new_meta.get("stats")
    if new_stats is None:
        raise ValueError("New session has no stats. Cannot compute diff.")

    if traffic is None:
        traffic = parse_traffic(baseline.traffic_assumption)

    new_baseline = create_baseline(new_session, traffic)

    # Monthly change
    total_monthly_change: dict[str, float] = {}
    total_monthly_pct_change: dict[str, float] = {}
    for pct in ("p50", "p95"):
        old_val = baseline.total_monthly.get(pct, 0)
        new_val = new_baseline.total_monthly.get(pct, 0)
        total_monthly_change[pct] = new_val - old_val
        total_monthly_pct_change[pct] = _pct_change(old_val, new_val)

    # Step diffs
    old_step_names = set(baseline.steps)
    new_step_stats = new_stats.get("step_stats", {})
    new_step_names = set(new_step_stats)
    common_steps = old_step_names & new_step_names

    step_diffs: dict[str, StepDiff] = {}
    model_changes: list[ModelChange] = []

    for name in common_steps:
        old_step = baseline.steps[name]
        ns = new_step_stats[name]

        old_cost_mean = old_step.cost_per_run.get("mean", 0)
        new_cost = ns.get("cost", {})
        new_cost_mean = new_cost.get("mean", 0)
        cost_change_abs = new_cost_mean - old_cost_mean
        cost_change_pct = _pct_change(old_cost_mean, new_cost_mean)

        old_tok_in = old_step.tokens_input.get("p50", 0)
        old_tok_out = old_step.tokens_output.get("p50", 0)
        old_tok_total = old_tok_in + old_tok_out
        new_tok = ns.get("total_tokens", {})
        new_tok_mean = new_tok.get("mean", 0)
        token_change_pct = _pct_change(old_tok_total, new_tok_mean)

        old_iter_mean = old_step.iterations.get("mean", 1.0)
        new_ipr = ns.get("iterations_per_run", {})
        new_iter_mean = new_ipr.get("mean", 1.0)
        iteration_change = new_iter_mean - old_iter_mean

        new_model = ns.get("model", "")
        model_changed = old_step.model != new_model and new_model != ""

        flags: list[str] = []
        if cost_change_pct > 50:
            flags.append("cost_increase > 50%")

        step_diffs[name] = StepDiff(
            step_name=name,
            cost_change_pct=cost_change_pct,
            cost_change_abs=cost_change_abs,
            token_change_pct=token_change_pct,
            iteration_change=iteration_change,
            model_changed=model_changed,
            old_model=old_step.model if model_changed else None,
            new_model=new_model if model_changed else None,
            flags=flags,
        )

        if model_changed:
            model_changes.append(
                ModelChange(
                    step_name=name,
                    old_model=old_step.model,
                    new_model=new_model,
                    cost_impact=cost_change_abs,
                )
            )

    new_steps = sorted(new_step_names - old_step_names)
    removed_steps = sorted(old_step_names - new_step_names)

    # Pattern changes
    old_patterns = set(baseline.patterns)
    new_patterns_raw = new_meta.get("patterns", [])
    new_pattern_types: set[str] = set()
    for p in new_patterns_raw:
        pt = p.get("pattern_type", "") if isinstance(p, dict) else ""
        if pt:
            new_pattern_types.add(pt)

    pattern_changes = PatternChanges(
        new_patterns=sorted(new_pattern_types - old_patterns),
        resolved_patterns=sorted(old_patterns - new_pattern_types),
        unchanged_patterns=sorted(old_patterns & new_pattern_types),
    )

    # Summary
    p50_old = baseline.total_monthly.get("p50", 0)
    p50_new = new_baseline.total_monthly.get("p50", 0)
    p50_pct = total_monthly_pct_change.get("p50", 0)

    if abs(p50_pct) < 5:
        summary = f"Monthly cost unchanged: ~{format_cost(p50_old)} at {traffic:,}/day"
    elif p50_pct > 0:
        summary = (
            f"Monthly cost increased {p50_pct:.0f}%: "
            f"{format_cost(p50_old)} → {format_cost(p50_new)} at {traffic:,}/day"
        )
    else:
        summary = (
            f"Monthly cost decreased {abs(p50_pct):.0f}%: "
            f"{format_cost(p50_old)} → {format_cost(p50_new)} at {traffic:,}/day"
        )

    return DiffResult(
        baseline_workflow=baseline.workflow,
        baseline_date=baseline.profiled_at,
        new_date=new_session.profiled_at.isoformat(),
        total_monthly_change=total_monthly_change,
        total_monthly_pct_change=total_monthly_pct_change,
        step_diffs=step_diffs,
        new_steps=new_steps,
        removed_steps=removed_steps,
        model_changes=model_changes,
        pattern_changes=pattern_changes,
        exceeds_threshold=None,
        summary=summary,
        traffic=traffic,
    )


def format_diff_report(diff: DiffResult) -> str:
    """Produce a terminal-friendly diff report."""
    lines: list[str] = []

    lines.append("Pretia Diff Report")
    lines.append(f"Baseline: {diff.baseline_workflow} (profiled {diff.baseline_date})")
    lines.append(f"Compared: new profile ({diff.new_date})")
    lines.append(f"Summary: {diff.summary}")
    lines.append("")

    # Step comparison table
    sorted_diffs = sorted(
        diff.step_diffs.values(),
        key=lambda d: abs(d.cost_change_abs),
        reverse=True,
    )

    if sorted_diffs:
        lines.append("Step Comparison:")
        lines.append(f"{'Step':<20} {'Before':>10} {'After':>10} {'Change':>10} {'Flag':>5}")
        lines.append("-" * 60)
        for sd in sorted_diffs:
            flag = ""
            if sd.cost_change_pct > 100:
                flag = "🔴"
            elif sd.cost_change_pct > 50:
                flag = "🟡"

            if abs(sd.cost_change_pct) < 1:
                change_str = "—"
            elif sd.cost_change_pct > 0:
                change_str = f"+{sd.cost_change_pct:.0f}%"
            else:
                change_str = f"{sd.cost_change_pct:.0f}%"

            lines.append(
                f"{sd.step_name:<20} "
                f"{format_cost(sd.cost_change_abs):>10} "
                f"{'':>10} "
                f"{change_str:>10} "
                f"{flag:>5}"
            )
        lines.append("")

    # New / removed steps
    if diff.new_steps:
        lines.append(f"New steps: {', '.join(diff.new_steps)}")
    if diff.removed_steps:
        lines.append(f"Removed steps: {', '.join(diff.removed_steps)}")
    if diff.new_steps or diff.removed_steps:
        lines.append("")

    # Model changes
    if diff.model_changes:
        lines.append("Model changes:")
        for mc in diff.model_changes:
            impact = format_cost(abs(mc.cost_impact))
            direction = "saves" if mc.cost_impact < 0 else "costs"
            lines.append(
                f"  {mc.step_name}: {mc.old_model} → {mc.new_model} ({direction} {impact}/run)"
            )
        lines.append("")

    # Pattern changes
    pc = diff.pattern_changes
    if pc.resolved_patterns:
        for p in pc.resolved_patterns:
            lines.append(f"Resolved: {p} (no longer detected)")
    if pc.new_patterns:
        for p in pc.new_patterns:
            lines.append(f"New pattern: {p}")
    if pc.resolved_patterns or pc.new_patterns:
        lines.append("")

    # Monthly projection comparison
    p50_old = diff.total_monthly_change.get("p50", 0)
    p95_old = diff.total_monthly_change.get("p95", 0)
    p50_pct = diff.total_monthly_pct_change.get("p50", 0)
    p95_pct = diff.total_monthly_pct_change.get("p95", 0)

    lines.append("Monthly Projection Change:")

    def _fmt_pct(v: float) -> str:
        if abs(v) < 1:
            return "—"
        return f"+{v:.0f}%" if v > 0 else f"{v:.0f}%"

    lines.append(f"  p50/month: {_fmt_pct(p50_pct)} ({format_cost(p50_old)} change)")
    lines.append(f"  p95/month: {_fmt_pct(p95_pct)} ({format_cost(p95_old)} change)")
    lines.append(f"  Traffic assumption: {diff.traffic:,} runs/day")

    return "\n".join(lines)
