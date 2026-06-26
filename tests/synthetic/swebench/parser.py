"""Parse SWE-bench trajectory data and extract per-instance token usage."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_INPUT_RATE = 3.0
_DEFAULT_OUTPUT_RATE = 15.0


@dataclass
class SWEBenchInstance:
    """Parsed data for one SWE-bench instance."""

    instance_id: str
    repo: str
    total_tokens: int
    total_cost: float
    input_tokens: int | None
    output_tokens: int | None
    model: str | None
    num_steps: int | None


def extract_repo(instance_id: str) -> str:
    """Extract repository name from SWE-bench instance ID."""
    parts = instance_id.split("__")
    if len(parts) >= 2:
        repo_and_issue = parts[1]
        segments = repo_and_issue.rsplit("-", 1)
        if len(segments) == 2 and segments[1].isdigit():
            return segments[0]
        return repo_and_issue
    return instance_id


def _estimate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    total_tokens: int,
    model: str | None,
) -> float:
    """Estimate cost from tokens, using pricing table or defaults."""
    try:
        from pretia.pricing.tables import calculate_cost, resolve_model

        if model:
            canonical = resolve_model(model)
            inp = input_tokens or total_tokens
            out = output_tokens or 0
            return calculate_cost(canonical, inp, out)
    except (ValueError, ImportError):
        pass

    inp = input_tokens or int(total_tokens * 0.75)
    out = output_tokens or int(total_tokens * 0.25)
    return (inp / 1_000_000 * _DEFAULT_INPUT_RATE) + (out / 1_000_000 * _DEFAULT_OUTPUT_RATE)


def _parse_instance(raw: dict) -> SWEBenchInstance | None:
    """Parse one raw JSON object into a SWEBenchInstance."""
    instance_id = raw.get("instance_id", "")
    if not instance_id:
        return None

    repo = extract_repo(instance_id)
    model = raw.get("model") or raw.get("model_name_or_path")

    info = raw.get("info", {})
    if isinstance(info, str):
        info = {}

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens = 0
    total_cost: float | None = None

    if "tokens_used" in info:
        total_tokens = int(info["tokens_used"])
    if "input_tokens" in info:
        input_tokens = int(info["input_tokens"])
    if "output_tokens" in info:
        output_tokens = int(info["output_tokens"])
    if "cost" in info:
        total_cost = float(info["cost"])
    elif "cost" in raw:
        total_cost = float(raw["cost"])
    if "api_calls" in info:
        num_steps = int(info["api_calls"])
    else:
        num_steps = None

    traj = raw.get("trajectory", [])
    if isinstance(traj, list) and traj:
        if not total_tokens and not total_cost:
            t_inp = 0
            t_out = 0
            for step in traj:
                if isinstance(step, dict):
                    t_inp += int(step.get("input_tokens", 0))
                    t_out += int(step.get("output_tokens", 0))
            if t_inp + t_out > 0:
                input_tokens = t_inp
                output_tokens = t_out
                total_tokens = t_inp + t_out
        if num_steps is None:
            num_steps = len(traj)

    if input_tokens and output_tokens and not total_tokens:
        total_tokens = input_tokens + output_tokens

    if total_tokens == 0 and total_cost is None:
        return None

    if total_cost is None:
        total_cost = _estimate_cost(input_tokens, output_tokens, total_tokens, model)

    return SWEBenchInstance(
        instance_id=instance_id,
        repo=repo,
        total_tokens=total_tokens,
        total_cost=total_cost,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model=model,
        num_steps=num_steps,
    )


def parse_swebench_data(data_path: str) -> list[SWEBenchInstance]:
    """Parse SWE-bench trajectory data into structured instances."""
    path = Path(data_path)
    text = path.read_text()
    instances: list[SWEBenchInstance] = []
    skipped = 0

    lines = text.strip().splitlines()
    if len(lines) > 1 and lines[0].strip().startswith("{"):
        raw_list = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    raw_list.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1
    elif text.strip().startswith("["):
        raw_list = json.loads(text)
    elif text.strip().startswith("{"):
        obj = json.loads(text)
        if isinstance(obj, dict) and any(isinstance(v, dict) for v in obj.values()):
            raw_list = [{"instance_id": k, **v} for k, v in obj.items() if isinstance(v, dict)]
        else:
            raw_list = [obj]
    else:
        raw_list = []
        for line in lines:
            line = line.strip()
            if line:
                try:
                    raw_list.append(json.loads(line))
                except json.JSONDecodeError:
                    skipped += 1

    for raw in raw_list:
        inst = _parse_instance(raw)
        if inst is not None:
            instances.append(inst)
        else:
            skipped += 1

    if skipped:
        logger.info("Skipped %d unparseable instances", skipped)
    logger.info("Parsed %d SWE-bench instances", len(instances))
    return instances
