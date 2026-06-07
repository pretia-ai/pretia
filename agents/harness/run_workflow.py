"""CLI harness for running workflow agents.

Usage::

    python -m agents.harness.run_workflow --workflow W1 --n 5 \\
        --inputs-dir inputs/ --output-dir results/ --prompts-dir prompts/
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import logging
import platform
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import click

from agentcost.collectors.base import StepRecord
from agentcost.pricing.tables import MODEL_PRICING, calculate_cost

logger = logging.getLogger(__name__)

_WORKFLOW_MODULE_MAP: dict[str, str] = {
    "W1": "agents.workflows.w01",
    "W2": "agents.workflows.w02",
    "W4": "agents.workflows.w04",
    "W5": "agents.workflows.w05",
    "W9": "agents.workflows.w09",
    "W11": "agents.workflows.w11",
    "W12": "agents.workflows.w12",
    "W13": "agents.workflows.w13",
    "W14": "agents.workflows.w14",
    "W15": "agents.workflows.w15",
    "W16": "agents.workflows.w16",
    "W17": "agents.workflows.w17",
    "W18": "agents.workflows.w18",
    "W19": "agents.workflows.w19",
}


def load_agent(workflow_id: str) -> Any:
    """Dynamically import and return the agent for a workflow."""
    module_name = _WORKFLOW_MODULE_MAP.get(workflow_id.upper())
    if module_name is None:
        raise click.UsageError(
            f"Unknown workflow {workflow_id!r}. "
            f"Available: {', '.join(sorted(_WORKFLOW_MODULE_MAP))}"
        )
    mod = importlib.import_module(module_name)
    return mod.agent


def load_prompts(workflow_id: str, prompts_dir: str = "prompts/") -> dict[str, str]:
    """Load system prompt files for a workflow from the manifest."""
    manifest_path = Path(prompts_dir) / "manifest.json"
    if not manifest_path.exists():
        raise click.UsageError(f"Manifest not found: {manifest_path}")

    with open(manifest_path) as f:
        manifest = json.load(f)

    wf_upper = workflow_id.upper()
    wf_padded = f"W{int(wf_upper[1:]):02d}" if wf_upper.startswith("W") else wf_upper
    prompts: dict[str, str] = {}
    for entry in manifest.get("prompts", []):
        entry_id = entry.get("workflow_id", "").upper()
        if entry_id != wf_upper and entry_id != wf_padded:
            continue
        step_name = entry["step_name"]
        file_path = Path(prompts_dir) / entry["file_path"]
        if not file_path.exists():
            logger.warning("Prompt file missing: %s", file_path)
            continue
        prompts[step_name] = file_path.read_text(encoding="utf-8")

    if not prompts:
        raise click.UsageError(f"No prompts found for workflow {workflow_id}")
    return prompts


def load_inputs(
    inputs_path: str, n: int | None = None, seed: int | None = None
) -> list[dict[str, Any]]:
    """Load inputs from a JSONL file."""
    path = Path(inputs_path)
    if not path.exists():
        raise click.UsageError(f"Inputs file not found: {path}")

    inputs: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, str):
                    data = {"input": data}
                inputs.append(data)
            except json.JSONDecodeError:
                inputs.append({"input": line})

    if seed is not None:
        import random

        rng = random.Random(seed)
        rng.shuffle(inputs)

    if n is not None:
        inputs = inputs[:n]

    return inputs


async def run_workflow(
    workflow_id: str,
    input_data: dict[str, Any],
    prompts: dict[str, str],
) -> list[StepRecord]:
    """Run a single workflow agent on one input."""
    agent = load_agent(workflow_id)
    return await agent.execute(input_data, prompts)


async def run_batch(
    workflow_id: str,
    inputs: list[dict[str, Any]],
    prompts: dict[str, str],
    parallel: int = 1,
) -> list[list[StepRecord]]:
    """Run a workflow on multiple inputs, optionally in parallel."""
    agent = load_agent(workflow_id)
    all_records: list[list[StepRecord]] = []

    if parallel <= 1:
        for idx, inp in enumerate(inputs):
            logger.info("Running %s input %d/%d", workflow_id, idx + 1, len(inputs))
            records = await agent.execute(inp, prompts)
            all_records.append(records)
    else:
        for batch_start in range(0, len(inputs), parallel):
            batch = inputs[batch_start : batch_start + parallel]
            tasks = [agent.execute(inp, prompts) for inp in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for result in results:
                if isinstance(result, Exception):
                    logger.error("Workflow run failed: %s", result)
                    all_records.append([])
                else:
                    all_records.append(result)

    return all_records


def _extract_run_metadata(
    input_data: dict[str, Any],
    records: list[StepRecord],
) -> dict[str, Any]:
    """Extract per-run metadata for pattern identification and longitudinal analysis.

    Captures input tier, structural descriptor, routing decisions, iteration
    counts, and which steps executed — everything needed for future change
    recommendation features.
    """
    meta: dict[str, Any] = {}

    meta["input_tier"] = input_data.get("tier", input_data.get("input_tier"))
    meta["structural_descriptor"] = input_data.get("structural_descriptor")

    meta["steps_executed"] = [r.step_name for r in records]
    meta["models_used"] = list({r.model for r in records})
    meta["step_costs"] = {
        r.step_name: calculate_cost(r.model, r.input_tokens, r.output_tokens)
        for r in records
    }

    max_iteration = max((r.iteration for r in records), default=1)
    meta["max_iteration"] = max_iteration

    decisions: dict[str, Any] = {}
    for r in records:
        if "classify" in r.step_name and r.output_format == "json":
            decisions["classifier_step"] = r.step_name
        if "path_a" in r.step_name or "path_b" in r.step_name or "path_c" in r.step_name:
            decisions["routed_to"] = r.step_name
        if r.step_name == "final_review":
            decisions["opus_triggered"] = True
        if r.step_name == "conditional_routing":
            decisions["routing_triggered"] = True
        if r.step_name == "intake_override" and len(records) == 1:
            decisions["short_circuited"] = True

    meta["decisions"] = decisions

    return meta


def _pricing_table_hash() -> str:
    """SHA-256 of the canonical pricing table for drift detection."""
    serialized = json.dumps(
        {k: list(v) for k, v in sorted(MODEL_PRICING.items())},
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()


def _prompt_hashes(prompts: dict[str, str]) -> dict[str, str]:
    """SHA-256 of each prompt template for change detection."""
    return {
        name: hashlib.sha256(text.encode()).hexdigest()
        for name, text in prompts.items()
    }


def save_results(
    workflow_id: str,
    all_records: list[list[StepRecord]],
    output_dir: str,
    *,
    inputs: list[dict[str, Any]] | None = None,
    prompts: dict[str, str] | None = None,
    backtest_profile: str | None = None,
) -> Path:
    """Save batch results as JSON with per-run metadata and backtest fields.

    Args:
        workflow_id: Workflow identifier (e.g., "W1").
        all_records: List of runs, each a list of StepRecords.
        output_dir: Directory to write the result file.
        inputs: Original input dicts (for extracting tier/structural metadata).
        prompts: Prompt templates (for computing prompt hashes).
        backtest_profile: "profiling" or "ground_truth" label for longitudinal tracking.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    filename = f"{workflow_id.lower()}_{ts}.json"
    filepath = out_path / filename

    runs = []
    for run_idx, records in enumerate(all_records):
        input_data = inputs[run_idx] if inputs and run_idx < len(inputs) else {}
        run_meta = _extract_run_metadata(input_data, records)

        runs.append({
            "run_id": f"{workflow_id.lower()}_run_{run_idx:03d}",
            "step_count": len(records),
            "total_cost_usd": sum(
                calculate_cost(r.model, r.input_tokens, r.output_tokens)
                for r in records
            ),
            "steps": [r.to_dict() for r in records],
            "metadata": run_meta,
        })

    detected_patterns = _detect_batch_patterns(all_records)

    result: dict[str, Any] = {
        "workflow_id": workflow_id,
        "total_runs": len(runs),
        "profiled_at": datetime.now(UTC).isoformat(),
        "runs": runs,
        "detected_patterns": detected_patterns,
        "backtest_id": uuid.uuid4().hex,
        "backtest_profile": backtest_profile,
        "pricing_table_hash": _pricing_table_hash(),
        "prompt_hashes": _prompt_hashes(prompts) if prompts else None,
    }

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return filepath


def _detect_batch_patterns(
    all_records: list[list[StepRecord]],
) -> list[dict[str, Any]]:
    """Run pattern detection on batch results, return serializable dicts.

    Falls back to an empty list if pattern detection fails (e.g., all
    dry-run records have zero tokens).
    """
    try:
        from agentcost.projection.patterns import detect_patterns
        from agentcost.projection.stats import compute_stats

        stats = compute_stats(all_records)
        patterns = detect_patterns(all_records, stats)
        return [p.to_dict() for p in patterns]
    except Exception as exc:
        logger.warning("Pattern detection failed (expected for dry-run): %s", exc)
        return []


def save_as_session(
    workflow_id: str,
    all_records: list[list[StepRecord]],
    *,
    prompts: dict[str, str] | None = None,
    backtest_profile: str | None = None,
    storage_dir: str | None = None,
) -> Path:
    """Save batch results as a ProfilingSession compatible with ProfileStore.

    Bridges the agents harness to the existing AgentCost persistence layer,
    enabling results to flow through ProfileStore.load() → compute_stats() →
    detect_patterns() → project().
    """
    from agentcost.store import ProfileStore, ProfilingSession

    now = datetime.now(UTC)

    workflow_hash = hashlib.sha256(
        json.dumps({
            "workflow_id": workflow_id,
            "prompt_hashes": _prompt_hashes(prompts) if prompts else {},
            "pricing_table_hash": _pricing_table_hash(),
        }, sort_keys=True).encode()
    ).hexdigest()

    session = ProfilingSession(
        workflow_name=workflow_id,
        workflow_hash=workflow_hash,
        profiled_at=now,
        sample_size=len(all_records),
        input_mode="backtesting",
        runs=all_records,
        metadata={
            "backtest_profile": backtest_profile,
            "backtest_id": uuid.uuid4().hex,
            "pricing_table_hash": _pricing_table_hash(),
            "prompt_hashes": _prompt_hashes(prompts) if prompts else None,
        },
        python_version=platform.python_version(),
    )

    store = ProfileStore(
        storage_dir=Path(storage_dir) if storage_dir else None
    )
    return store.save(session)


@click.command()
@click.option("--workflow", "-w", required=True, help="Workflow ID (e.g., W1, W13)")
@click.option("--n", type=int, default=None, help="Number of inputs to process")
@click.option("--inputs-dir", type=str, default=None, help="Path to JSONL input file")
@click.option("--output-dir", type=str, default="results/", help="Output directory")
@click.option("--prompts-dir", type=str, default="prompts/", help="Prompts directory")
@click.option("--parallel", type=int, default=1, help="Concurrent runs")
@click.option("--seed", type=int, default=None, help="Random seed for input shuffling")
@click.option("--dry-run", is_flag=True, help="Validate without making API calls")
@click.option("--profile", type=str, default=None, help="Backtest profile: profiling or ground_truth")
@click.option("--save-session", is_flag=True, help="Also save as ProfilingSession to .agentcost/")
@click.option("-v", "--verbose", is_flag=True, help="Enable debug logging")
def main(
    workflow: str,
    n: int | None,
    inputs_dir: str | None,
    output_dir: str,
    prompts_dir: str,
    parallel: int,
    seed: int | None,
    dry_run: bool,
    profile: str | None,
    save_session: bool,
    verbose: bool,
) -> None:
    """Run a backtesting workflow agent."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    prompts = load_prompts(workflow, prompts_dir)
    logger.info("Loaded %d prompts for %s", len(prompts), workflow)

    if inputs_dir:
        inputs = load_inputs(inputs_dir, n=n, seed=seed)
    else:
        inputs = [{"input": "Hello, I need help with my account.", "_dry_run": dry_run}]
        if n:
            inputs = inputs * n

    if dry_run:
        for inp in inputs:
            inp["_dry_run"] = True

    logger.info("Running %s with %d inputs (parallel=%d)", workflow, len(inputs), parallel)
    all_records = asyncio.run(run_batch(workflow, inputs, prompts, parallel))

    total_steps = sum(len(r) for r in all_records)
    logger.info("Completed: %d runs, %d total steps", len(all_records), total_steps)

    filepath = save_results(
        workflow, all_records, output_dir,
        inputs=inputs, prompts=prompts, backtest_profile=profile,
    )
    logger.info("Results saved to %s", filepath)

    if save_session:
        session_path = save_as_session(
            workflow, all_records, prompts=prompts, backtest_profile=profile,
        )
        logger.info("ProfilingSession saved to %s", session_path)


if __name__ == "__main__":
    main()
