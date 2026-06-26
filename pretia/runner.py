"""Orchestrate the end-to-end profiling pipeline."""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import logging
import re
import statistics
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from pretia.collectors.base import BaseCollector, StepRecord
from pretia.collectors.generic import GenericCollector
from pretia.inputs.generator import _extract_workflow_context, generate_inputs
from pretia.inputs.selector import InputSelection, select_input_mode
from pretia.pricing.tables import calculate_cost, model_tier
from pretia.projection.patterns import detect_patterns
from pretia.projection.projector import project
from pretia.projection.stats import compute_stats
from pretia.store import ProfileStore, ProfilingSession

logger = logging.getLogger(__name__)

_WORKFLOW_ATTR_NAMES = ("graph", "workflow", "agent", "app")
_SYSTEM_PROMPT_RE = re.compile(
    r"(you are|your role|your task|system)",
    re.IGNORECASE,
)


def _find_workflow(module: Any) -> Any:
    for name in _WORKFLOW_ATTR_NAMES:
        obj = getattr(module, name, None)
        if obj is not None:
            return obj

    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if hasattr(obj, "ainvoke") or hasattr(obj, "invoke"):
            return obj

    return None


def _extract_system_prompt(module: Any) -> str:
    for name in dir(module):
        if name.startswith("_"):
            continue
        obj = getattr(module, name, None)
        if isinstance(obj, str) and len(obj) > 50 and _SYSTEM_PROMPT_RE.search(obj):
            return obj
    return ""


def _load_workflow_module(path: str) -> Any:
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Workflow file not found: {path}")

    spec = importlib.util.spec_from_file_location(p.stem, str(p))
    if spec is None or spec.loader is None:
        raise click.UsageError(f"Cannot load module from '{path}'.")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        pkg = exc.name or str(exc)
        raise ImportError(
            f"'{path}' requires '{pkg}' which is not installed. Install it with: pip install {pkg}"
        ) from exc
    return module


def _percentile(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


def _build_cost_summary(
    runs: list[list[StepRecord]],
) -> dict[str, Any]:
    step_costs: dict[str, list[dict[str, Any]]] = {}
    run_totals: list[float] = []

    for run in runs:
        run_cost = 0.0
        for rec in run:
            try:
                cost = calculate_cost(
                    rec.model,
                    rec.input_tokens,
                    rec.output_tokens,
                )
            except ValueError:
                cost = 0.0

            entry = {
                "cost": cost,
                "input_tokens": rec.input_tokens,
                "output_tokens": rec.output_tokens,
                "duration_ms": rec.duration_ms,
                "iteration": rec.iteration,
            }
            step_costs.setdefault(rec.step_name, []).append(entry)
            run_cost += cost
        run_totals.append(run_cost)

    per_step: dict[str, dict[str, Any]] = {}
    for step_name, entries in step_costs.items():
        costs = [e["cost"] for e in entries]
        in_toks = [e["input_tokens"] for e in entries]
        out_toks = [e["output_tokens"] for e in entries]
        durations = [e["duration_ms"] for e in entries]
        iterations = [e["iteration"] for e in entries]

        per_step[step_name] = {
            "count": len(entries),
            "cost_mean": statistics.mean(costs),
            "cost_min": min(costs),
            "cost_max": max(costs),
            "cost_p50": _percentile(costs, 50),
            "cost_p95": _percentile(costs, 95),
            "input_tokens_mean": statistics.mean(in_toks),
            "output_tokens_mean": statistics.mean(out_toks),
            "duration_ms_mean": statistics.mean(durations),
            "max_iteration": max(iterations),
        }

    mean_run_cost = statistics.mean(run_totals) if run_totals else 0.0

    return {
        "per_step": per_step,
        "run_totals": run_totals,
        "mean_cost_per_run": mean_run_cost,
        "min_cost_per_run": min(run_totals) if run_totals else 0.0,
        "max_cost_per_run": max(run_totals) if run_totals else 0.0,
        "p95_cost_per_run": _percentile(run_totals, 95),
        "total_session_cost": sum(run_totals),
        "projection_100_day": mean_run_cost * 100 * 30,
        "projection_1000_day": mean_run_cost * 1000 * 30,
        "projection_10000_day": mean_run_cost * 10000 * 30,
    }


def _get_step_model(
    runs: list[list[StepRecord]],
    step_name: str,
) -> str:
    for run in runs:
        for rec in run:
            if rec.step_name == step_name:
                return rec.model
    return ""


def _get_step_type(
    runs: list[list[StepRecord]],
    step_name: str,
) -> str:
    for run in runs:
        for rec in run:
            if rec.step_name == step_name:
                return rec.step_type
    return "llm"


class ProfileRunner:
    """Coordinate the full profiling pipeline."""

    def __init__(
        self,
        workflow_path: str,
        collector: str = "auto",
        auto_generate: int | None = None,
        single_input: str | None = None,
        inputs_file: str | None = None,
        from_langfuse: bool = False,
        langfuse_last_n: int = 10,
        output_dir: str = ".pretia",
        cache_mode: str = "cold",
        progress_callback: Any | None = None,
        generator_model: str = "deepseek-v4-flash",
        corpus_path: str | None = None,
    ) -> None:
        self.workflow_path = workflow_path
        self.collector_name = collector
        self.auto_generate = auto_generate
        self.single_input = single_input
        self.inputs_file = inputs_file
        self.from_langfuse = from_langfuse
        self.langfuse_last_n = langfuse_last_n
        self.output_dir = output_dir
        self.cache_mode = cache_mode
        self.progress_callback = progress_callback
        self.generator_model = generator_model
        self.corpus_path = corpus_path

    def _load_workflow(self) -> tuple[Any, str]:
        module = _load_workflow_module(self.workflow_path)
        workflow = _find_workflow(module)
        if workflow is None:
            raise click.UsageError(
                f"Could not find a workflow in '{self.workflow_path}'. "
                "Expected a module-level variable named `graph`, "
                "`workflow`, `agent`, or `app`, or an object with "
                "an `ainvoke`/`invoke` method."
            )
        system_prompt = _extract_system_prompt(module)
        return workflow, system_prompt

    def _select_collector(self, workflow: Any) -> BaseCollector:
        if self.collector_name == "langgraph":
            from pretia.collectors.langgraph import LangGraphCollector

            return LangGraphCollector()

        if self.collector_name == "generic":
            return GenericCollector()

        if self.collector_name == "openai":
            from pretia.collectors.openai_agents import OpenAIAgentsCollector

            return OpenAIAgentsCollector()

        if self.collector_name == "qwen":
            from pretia.collectors.qwen_agent import QwenAgentCollector

            return QwenAgentCollector()

        has_ainvoke = hasattr(workflow, "ainvoke")
        has_nodes = hasattr(workflow, "nodes")
        if has_ainvoke and has_nodes:
            from pretia.collectors.langgraph import LangGraphCollector

            return LangGraphCollector()

        if hasattr(workflow, "name") and hasattr(workflow, "instructions"):
            from pretia.collectors.openai_agents import OpenAIAgentsCollector

            return OpenAIAgentsCollector()

        if (
            hasattr(workflow, "run")
            and hasattr(workflow, "llm")
            and hasattr(workflow, "system_message")
        ):
            from pretia.collectors.qwen_agent import QwenAgentCollector

            return QwenAgentCollector()

        logger.info(
            "Using GenericCollector. Instrument your code with "
            "@collector.step() for per-step data."
        )
        return GenericCollector()

    @staticmethod
    def _detect_framework(collector: BaseCollector) -> str | None:
        """Derive a framework label from the selected collector."""
        mapping = {
            "LangGraphCollector": "langgraph",
            "OpenAIAgentsCollector": "openai-agents",
            "QwenAgentCollector": "qwen-agent",
            "GenericCollector": "generic",
        }
        return mapping.get(type(collector).__name__)

    async def _resolve_inputs(
        self,
        system_prompt: str,
    ) -> tuple[InputSelection, list[str]]:
        selection = select_input_mode(
            single_input=self.single_input,
            inputs_file=self.inputs_file,
            auto_generate=self.auto_generate,
            from_langfuse=self.from_langfuse,
            system_prompt=system_prompt or None,
        )

        if selection.mode == "auto-generate":
            n = self.auto_generate or 50

            context_parts: list[str] = []
            try:
                wf_source = Path(self.workflow_path).read_text(encoding="utf-8")
                wf_context = _extract_workflow_context(wf_source)
                if wf_context:
                    context_parts.append(wf_context)
            except OSError:
                pass

            if self.corpus_path:
                from pretia.inputs.corpus import load_corpus_context

                try:
                    corpus_ctx = load_corpus_context(self.corpus_path)
                    if corpus_ctx:
                        context_parts.append(
                            f"Documents in the user's knowledge base:\n{corpus_ctx}"
                        )
                except (FileNotFoundError, OSError) as exc:
                    logging.warning("Could not load corpus: %s", exc)

            inputs = await generate_inputs(
                system_prompt or "General purpose agent.",
                n=n,
                model=self.generator_model,
                additional_context="\n\n".join(context_parts),
            )
            return selection, inputs

        if selection.mode in ("single", "manual", "file"):
            return selection, selection.inputs

        if selection.mode == "langfuse":
            from pretia.inputs.importer import (
                create_langfuse_client,
                extract_inputs,
                fetch_traces,
            )

            client = create_langfuse_client()
            traces = fetch_traces(client, last_n=self.langfuse_last_n)
            inputs = extract_inputs(traces)
            return selection, inputs

        raise NotImplementedError(
            "Static estimation is not yet implemented. Provide an API key for input generation."
        )

    async def run(self) -> ProfilingSession:
        """Execute the full profiling pipeline."""
        workflow, system_prompt = self._load_workflow()
        collector = self._select_collector(workflow)
        selection, inputs = await self._resolve_inputs(system_prompt)
        runs = await collector.collect(
            workflow,
            inputs,
            on_run_complete=self.progress_callback,
        )

        cost_summary = _build_cost_summary(runs)

        for step_name in cost_summary["per_step"]:
            model = _get_step_model(runs, step_name)
            step_type = _get_step_type(runs, step_name)
            cost_summary["per_step"][step_name]["model"] = model
            cost_summary["per_step"][step_name]["step_type"] = step_type
            if model:
                try:
                    cost_summary["per_step"][step_name]["tier"] = model_tier(model)
                except (ValueError, KeyError):
                    cost_summary["per_step"][step_name]["tier"] = "unknown"
            else:
                cost_summary["per_step"][step_name]["tier"] = "tool"

        from pretia.validation.data_checks import validate_profiling_data

        data_warnings = validate_profiling_data(runs)
        for w in data_warnings:
            logger.warning(w)

        profiling_stats = compute_stats(runs)
        patterns = detect_patterns(runs, profiling_stats)
        projection = project(
            profiling_stats,
            patterns,
            runs=runs,
            input_source=selection.mode,
        )

        from pretia import __version__

        workflow_src = Path(self.workflow_path).read_bytes()
        session = ProfilingSession(
            workflow_name=self.workflow_path,
            workflow_hash=hashlib.sha256(workflow_src).hexdigest()[:12],
            profiled_at=datetime.now(UTC),
            sample_size=len(inputs),
            input_mode=selection.mode,
            runs=runs,
            metadata={
                "cost_summary": cost_summary,
                "stats": profiling_stats.to_dict(),
                "patterns": [p.to_dict() for p in patterns],
                "projection": projection.to_dict(),
                "confidence": projection.confidence.to_dict(),
            },
            workflow_id=Path(self.workflow_path).stem,
            run_id=str(uuid.uuid4()),
            framework=self._detect_framework(collector),
            pretia_version=__version__,
            profiling_cost=cost_summary["total_session_cost"],
        )

        store = ProfileStore(storage_dir=Path(self.output_dir))
        saved_path = store.save(session)
        session.metadata["saved_path"] = str(saved_path)

        self._auto_diff_baseline(session)

        return session

    def _auto_diff_baseline(self, session: ProfilingSession) -> None:
        """Show a one-line diff summary if a baseline exists."""
        baseline_path = Path(self.output_dir) / "baseline.json"
        if not baseline_path.exists():
            return
        try:
            from pretia.ci.baseline import load_baseline
            from pretia.ci.diff import diff_baseline

            bl = load_baseline(str(baseline_path))
            result = diff_baseline(bl, session)
            session.metadata["baseline_diff_summary"] = result.summary
        except Exception:
            logger.debug("Auto-diff against baseline failed", exc_info=True)

    def run_sync(self) -> ProfilingSession:
        """Synchronous wrapper around `run()`."""
        return asyncio.run(self.run())

    def analyze_langfuse(self) -> ProfilingSession:
        """Analyze Langfuse traces without re-executing the workflow.

        Used by 'pretia analyze --from-langfuse' (CLI command added in prompt 15).
        """
        from pretia.inputs.importer import (
            create_langfuse_client,
            fetch_traces,
            traces_to_step_records,
        )

        client = create_langfuse_client()
        traces = fetch_traces(client, last_n=self.langfuse_last_n)
        runs = traces_to_step_records(traces)

        cost_summary = _build_cost_summary(runs)

        for step_name in cost_summary["per_step"]:
            model = _get_step_model(runs, step_name)
            step_type = _get_step_type(runs, step_name)
            cost_summary["per_step"][step_name]["model"] = model
            cost_summary["per_step"][step_name]["step_type"] = step_type
            if model:
                try:
                    cost_summary["per_step"][step_name]["tier"] = model_tier(model)
                except (ValueError, KeyError):
                    cost_summary["per_step"][step_name]["tier"] = "unknown"
            else:
                cost_summary["per_step"][step_name]["tier"] = "tool"

        profiling_stats = compute_stats(runs)
        patterns = detect_patterns(runs, profiling_stats)
        projection = project(
            profiling_stats,
            patterns,
            runs=runs,
            input_source="langfuse",
        )

        from pretia import __version__

        session = ProfilingSession(
            workflow_name=f"langfuse-import ({len(traces)} traces)",
            workflow_hash="langfuse",
            profiled_at=datetime.now(UTC),
            sample_size=len(traces),
            input_mode="langfuse-analyze",
            runs=runs,
            metadata={
                "cost_summary": cost_summary,
                "stats": profiling_stats.to_dict(),
                "patterns": [p.to_dict() for p in patterns],
                "langfuse_trace_count": len(traces),
                "projection": projection.to_dict(),
                "confidence": projection.confidence.to_dict(),
            },
            workflow_id="langfuse-import",
            run_id=str(uuid.uuid4()),
            framework=None,
            pretia_version=__version__,
            profiling_cost=0.0,
        )

        store = ProfileStore(storage_dir=Path(self.output_dir))
        saved_path = store.save(session)
        session.metadata["saved_path"] = str(saved_path)

        return session
