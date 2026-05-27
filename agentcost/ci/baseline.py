"""Save and load cost baselines from .agentcost/baseline.json."""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agentcost.store import ProfilingSession

logger = logging.getLogger(__name__)

_TRAFFIC_RE = re.compile(r"(\d+)")


@dataclass(frozen=True, slots=True)
class BaselineStep:
    """One step's baseline data."""

    model: str
    tokens_input: dict[str, float]
    tokens_output: dict[str, float]
    cost_per_run: dict[str, float]
    iterations: dict[str, float]
    system_prompt_hash: str
    system_prompt_tokens: int
    output_format: str
    flags: list[str]
    task_complexity_tier: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "model": self.model,
            "tokens_input": dict(self.tokens_input),
            "tokens_output": dict(self.tokens_output),
            "cost_per_run": dict(self.cost_per_run),
            "iterations": dict(self.iterations),
            "system_prompt_hash": self.system_prompt_hash,
            "system_prompt_tokens": self.system_prompt_tokens,
            "output_format": self.output_format,
            "flags": list(self.flags),
            "task_complexity_tier": self.task_complexity_tier,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaselineStep:
        """Deserialize from a dict."""
        return cls(
            model=data["model"],
            tokens_input=dict(data["tokens_input"]),
            tokens_output=dict(data["tokens_output"]),
            cost_per_run=dict(data["cost_per_run"]),
            iterations=dict(data["iterations"]),
            system_prompt_hash=data["system_prompt_hash"],
            system_prompt_tokens=data["system_prompt_tokens"],
            output_format=data["output_format"],
            flags=list(data["flags"]),
            task_complexity_tier=data.get("task_complexity_tier"),
        )


@dataclass(frozen=True, slots=True)
class Baseline:
    """A saved cost baseline for a workflow."""

    version: str
    workflow: str
    profiled_at: str
    sample_size: int
    traffic_assumption: str
    input_source: str
    collector_type: str
    confidence_tier: str
    steps: dict[str, BaselineStep]
    total_monthly: dict[str, float]
    patterns: list[str]
    assumptions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "version": self.version,
            "workflow": self.workflow,
            "profiled_at": self.profiled_at,
            "sample_size": self.sample_size,
            "traffic_assumption": self.traffic_assumption,
            "input_source": self.input_source,
            "collector_type": self.collector_type,
            "confidence_tier": self.confidence_tier,
            "steps": {k: v.to_dict() for k, v in self.steps.items()},
            "total_monthly": dict(self.total_monthly),
            "patterns": list(self.patterns),
            "assumptions": list(self.assumptions),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Baseline:
        """Deserialize from a dict."""
        version = data.get("version", "")
        if not version.startswith("1."):
            raise ValueError(
                f"Unsupported baseline version: {version!r}. Expected '1.x'."
            )
        return cls(
            version=data["version"],
            workflow=data["workflow"],
            profiled_at=data["profiled_at"],
            sample_size=data["sample_size"],
            traffic_assumption=data["traffic_assumption"],
            input_source=data["input_source"],
            collector_type=data["collector_type"],
            confidence_tier=data["confidence_tier"],
            steps={k: BaselineStep.from_dict(v) for k, v in data["steps"].items()},
            total_monthly=dict(data["total_monthly"]),
            patterns=list(data["patterns"]),
            assumptions=list(data.get("assumptions", [])),
        )


def _extract_step_record_fields(
    session: ProfilingSession,
    step_name: str,
) -> tuple[str, int, str]:
    """Extract system_prompt_hash, system_prompt_tokens, and output_format from runs."""
    hashes: list[str] = []
    spt: list[int] = []
    formats: list[str] = []
    for run in session.runs:
        for rec in run:
            if rec.step_name == step_name:
                hashes.append(rec.system_prompt_hash)
                spt.append(rec.system_prompt_tokens)
                formats.append(rec.output_format)
    prompt_hash = hashes[0] if hashes else "unknown"
    prompt_tokens = spt[0] if spt else 0
    fmt_counter = Counter(formats)
    output_format = fmt_counter.most_common(1)[0][0] if fmt_counter else "text"
    return prompt_hash, prompt_tokens, output_format


def _infer_collector_type(session: ProfilingSession) -> str:
    """Guess the collector type from session metadata."""
    mode = session.input_mode
    if "langfuse" in mode:
        return "langfuse-import"
    return "auto"


def create_baseline(
    session: ProfilingSession,
    traffic: int = 1000,
) -> Baseline:
    """Build a Baseline from a completed ProfilingSession."""
    meta = session.metadata or {}
    stats_dict = meta.get("stats")
    if stats_dict is None:
        raise ValueError(
            "Session has no stats. "
            "Run 'agentcost profile run' to generate a complete profile."
        )

    patterns_raw = meta.get("patterns", [])
    pattern_types: list[str] = []
    step_pattern_map: dict[str, list[str]] = {}
    for p in patterns_raw:
        pt = p.get("pattern_type", "") if isinstance(p, dict) else ""
        sn = p.get("step_name", "") if isinstance(p, dict) else ""
        if pt:
            if pt not in pattern_types:
                pattern_types.append(pt)
            step_pattern_map.setdefault(sn, [])
            if pt not in step_pattern_map[sn]:
                step_pattern_map[sn].append(pt)

    projection = meta.get("projection", {})
    confidence = meta.get("confidence", {})
    confidence_tier = confidence.get("tier", "MODERATE")

    step_stats_dict = stats_dict.get("step_stats", {})
    steps: dict[str, BaselineStep] = {}
    for name, ss in step_stats_dict.items():
        input_tok = ss.get("input_tokens", {})
        output_tok = ss.get("output_tokens", {})
        cost = ss.get("cost", {})
        ipr = ss.get("iterations_per_run", {})

        prompt_hash, prompt_tokens, output_format = _extract_step_record_fields(
            session, name,
        )

        steps[name] = BaselineStep(
            model=ss.get("model", ""),
            tokens_input={"p50": input_tok.get("p50", 0), "p95": input_tok.get("p95", 0)},
            tokens_output={"p50": output_tok.get("p50", 0), "p95": output_tok.get("p95", 0)},
            cost_per_run={
                "p50": cost.get("p50", 0),
                "p95": cost.get("p95", 0),
                "mean": cost.get("mean", 0),
            },
            iterations={"mean": ipr.get("mean", 1.0), "max": ipr.get("max", 1)},
            system_prompt_hash=prompt_hash,
            system_prompt_tokens=prompt_tokens,
            output_format=output_format,
            flags=step_pattern_map.get(name, []),
            task_complexity_tier=None,
        )

    # Build total_monthly from projection data
    total_monthly: dict[str, float] = {}
    proj_data = projection.get("projections", {})
    vol_key = str(traffic)
    vol_data = proj_data.get(vol_key, proj_data.get(traffic, {}))
    if vol_data:
        mc = vol_data.get("monthly_cost", {})
        total_monthly = {
            "p50": mc.get("p50", 0),
            "p75": mc.get("p75", 0),
            "p90": mc.get("p90", 0),
            "p95": mc.get("p95", 0),
        }
    else:
        cpr = stats_dict.get("cost_per_run", {})
        total_monthly = {
            "p50": cpr.get("p50", 0) * traffic * 30,
            "p75": cpr.get("p75", 0) * traffic * 30,
            "p90": cpr.get("p90", 0) * traffic * 30,
            "p95": cpr.get("p95", 0) * traffic * 30,
        }

    # Build assumptions
    method = projection.get("method", "linear")
    input_source = session.input_mode
    assumptions: list[str] = [
        f"Based on {session.sample_size} profiling runs with {input_source} inputs.",
        f"Traffic assumption: {traffic} runs/day.",
        f"Projection method: {method} (confidence: {confidence_tier}).",
    ]

    for p in patterns_raw:
        if isinstance(p, dict) and p.get("pattern_type") == "context_growth":
            sn = p.get("step_name", "unknown")
            assumptions.append(
                f"Context growth detected at step '{sn}'. Projection assumes "
                "growth continues linearly. If context plateaus in production, "
                "actual costs may be lower."
            )

    if session.sample_size < 20:
        assumptions.append(
            f"Small sample size ({session.sample_size} runs). "
            "Wider confidence intervals than shown."
        )

    if input_source in ("auto-generate", "auto_generate"):
        assumptions.append(
            "Inputs were auto-generated. Production input distribution may differ."
        )

    return Baseline(
        version="1.0",
        workflow=session.workflow_name,
        profiled_at=session.profiled_at.isoformat(),
        sample_size=session.sample_size,
        traffic_assumption=f"{traffic}/day",
        input_source=input_source,
        collector_type=_infer_collector_type(session),
        confidence_tier=confidence_tier,
        steps=steps,
        total_monthly=total_monthly,
        patterns=pattern_types,
        assumptions=assumptions,
    )


def save_baseline(
    baseline: Baseline,
    output_dir: str = ".agentcost",
) -> str:
    """Save a baseline as JSON to {output_dir}/baseline.json."""
    dirpath = Path(output_dir)
    dirpath.mkdir(parents=True, exist_ok=True)
    path = dirpath / "baseline.json"
    path.write_text(json.dumps(baseline.to_dict(), indent=2))
    return str(path)


def load_baseline(path: str) -> Baseline:
    """Load a baseline from a JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Baseline not found at '{path}'. "
            "Run 'agentcost baseline update' to create one."
        )
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed baseline JSON at '{path}': {exc}") from exc
    return Baseline.from_dict(data)


def parse_traffic(assumption: str) -> int:
    """Extract the integer traffic volume from a string like '1000/day'."""
    m = _TRAFFIC_RE.search(assumption)
    if m:
        return int(m.group(1))
    return 1000
