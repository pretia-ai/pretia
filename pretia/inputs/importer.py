"""Import production traces from Langfuse for cost analysis."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from pretia.collectors.base import StepRecord
from pretia.pricing.tables import calculate_cost

logger = logging.getLogger(__name__)

_GENERATION_TYPES = frozenset({"GENERATION"})
_TOOL_TYPES = frozenset({"SPAN", "TOOL"})
_SKIP_TYPES = frozenset({"EVENT"})
_DEFAULT_HOST = "https://cloud.langfuse.com"


@dataclass(frozen=True, slots=True)
class LangfuseObservation:
    """One LLM call or tool call within a Langfuse trace."""

    observation_id: str
    name: str
    observation_type: str
    model: str | None
    input_tokens: int
    output_tokens: int
    start_time: datetime | None
    end_time: datetime | None
    duration_ms: int
    parent_observation_id: str | None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "observation_id": self.observation_id,
            "name": self.name,
            "observation_type": self.observation_type,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_ms": self.duration_ms,
            "parent_observation_id": self.parent_observation_id,
        }


@dataclass(frozen=True, slots=True)
class LangfuseTrace:
    """One imported Langfuse trace."""

    trace_id: str
    name: str | None
    input_text: str | None
    timestamp: datetime
    observations: list[LangfuseObservation]
    total_input_tokens: int
    total_output_tokens: int
    total_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "input_text": self.input_text,
            "timestamp": self.timestamp.isoformat(),
            "observations": [o.to_dict() for o in self.observations],
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": self.total_cost,
        }


def _compute_duration_ms(
    start_time: datetime | None,
    end_time: datetime | None,
) -> int:
    if start_time is None or end_time is None:
        return 0
    delta = end_time - start_time
    return max(0, int(delta.total_seconds() * 1000))


def _extract_input_text(raw_input: Any) -> str | None:
    """Extract a usable input string from a Langfuse trace's raw input."""
    if raw_input is None:
        return None
    if isinstance(raw_input, str):
        return raw_input if raw_input.strip() else None
    if isinstance(raw_input, dict):
        messages = raw_input.get("messages", [])
        if messages and isinstance(messages, list):
            first = messages[0]
            if isinstance(first, dict):
                content = first.get("content", "")
                if content:
                    return str(content)
        content = raw_input.get("content", "")
        if content:
            return str(content)
        input_val = raw_input.get("input", "")
        if input_val:
            return str(input_val)
    return str(raw_input) if raw_input else None


def _safe_cost(model: str | None, input_tokens: int, output_tokens: int) -> float:
    if not model:
        return 0.0
    try:
        return calculate_cost(model, input_tokens, output_tokens)
    except (ValueError, KeyError):
        logger.warning("Unknown model %r in Langfuse trace — using $0.00", model)
        return 0.0


def create_langfuse_client() -> Any:
    """Create a Langfuse API client from environment variables."""
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
    host = os.environ.get("LANGFUSE_HOST", _DEFAULT_HOST)

    if not secret_key or not public_key:
        missing = []
        if not secret_key:
            missing.append("LANGFUSE_SECRET_KEY")
        if not public_key:
            missing.append("LANGFUSE_PUBLIC_KEY")
        raise OSError(
            f"Langfuse credentials not found. Set {', '.join(missing)}"
            " and LANGFUSE_HOST environment variables. "
            "See https://langfuse.com/docs/sdk/python"
        )

    try:
        from langfuse.api.client import LangfuseAPI
    except ImportError:
        raise ImportError(
            "langfuse is not installed. Install it with: pip install pretia[langfuse]"
        ) from None

    return LangfuseAPI(
        base_url=host,
        username=public_key,
        password=secret_key,
    )


def _parse_observation(obs: Any) -> LangfuseObservation:
    """Convert a Langfuse ObservationsView into a LangfuseObservation."""
    usage = getattr(obs, "usage", None)
    input_tokens = 0
    output_tokens = 0
    if usage is not None:
        input_tokens = getattr(usage, "input", 0) or 0
        output_tokens = getattr(usage, "output", 0) or 0

    start_time = getattr(obs, "start_time", None)
    end_time = getattr(obs, "end_time", None)

    return LangfuseObservation(
        observation_id=getattr(obs, "id", ""),
        name=getattr(obs, "name", None) or "unknown",
        observation_type=getattr(obs, "type", "SPAN"),
        model=getattr(obs, "model", None),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        start_time=start_time,
        end_time=end_time,
        duration_ms=_compute_duration_ms(start_time, end_time),
        parent_observation_id=getattr(obs, "parent_observation_id", None),
    )


def fetch_traces(
    client: Any,
    last_n: int = 10,
    name: str | None = None,
) -> list[LangfuseTrace]:
    """Fetch the most recent traces from Langfuse with their observations."""
    last_n = min(last_n, 100)

    try:
        kwargs: dict[str, Any] = {"limit": last_n}
        if name is not None:
            kwargs["name"] = name
        traces_response = client.trace.list(**kwargs)
    except Exception as exc:
        exc_str = str(exc).lower()
        if "401" in exc_str or "403" in exc_str or "auth" in exc_str:
            raise PermissionError(
                "Langfuse authentication failed. "
                "Check LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY."
            ) from exc
        raise ConnectionError(f"Failed to connect to Langfuse: {exc}") from exc

    trace_list = getattr(traces_response, "data", []) or []
    results: list[LangfuseTrace] = []

    for trace_summary in trace_list:
        trace_id = getattr(trace_summary, "id", "")

        try:
            full_trace = client.trace.get(trace_id=trace_id)
        except Exception:
            logger.warning("Failed to fetch trace %s, skipping", trace_id)
            continue

        raw_observations = getattr(full_trace, "observations", []) or []
        observations = [_parse_observation(obs) for obs in raw_observations]

        total_in = sum(o.input_tokens for o in observations)
        total_out = sum(o.output_tokens for o in observations)
        total_cost = sum(
            _safe_cost(o.model, o.input_tokens, o.output_tokens) for o in observations
        )

        raw_input = getattr(full_trace, "input", None)
        input_text = _extract_input_text(raw_input)
        timestamp = getattr(full_trace, "timestamp", None) or datetime.now(UTC)

        results.append(
            LangfuseTrace(
                trace_id=trace_id,
                name=getattr(full_trace, "name", None),
                input_text=input_text,
                timestamp=timestamp,
                observations=observations,
                total_input_tokens=total_in,
                total_output_tokens=total_out,
                total_cost=total_cost,
            )
        )

    return results


def traces_to_step_records(
    traces: list[LangfuseTrace],
) -> list[list[StepRecord]]:
    """Convert imported Langfuse traces into the standard list[list[StepRecord]] format."""
    runs: list[list[StepRecord]] = []
    for trace in traces:
        obs_name_map: dict[str, str] = {}
        for obs in trace.observations:
            obs_name_map[obs.observation_id] = obs.name

        iteration_counts: dict[str, int] = {}
        step_records: list[StepRecord] = []

        for obs in trace.observations:
            if obs.observation_type in _SKIP_TYPES:
                continue

            if obs.observation_type in _GENERATION_TYPES:
                step_type = "llm"
            elif "retriev" in obs.name.lower():
                step_type = "retrieval"
            elif obs.observation_type in _TOOL_TYPES:
                step_type = "tool"
            else:
                step_type = "llm"

            count = iteration_counts.get(obs.name, 0) + 1
            iteration_counts[obs.name] = count

            parent_name = None
            if obs.parent_observation_id:
                parent_name = obs_name_map.get(obs.parent_observation_id)

            timestamp = obs.start_time or trace.timestamp

            step_records.append(
                StepRecord(
                    step_name=obs.name,
                    step_type=step_type,
                    model=obs.model or "unknown",
                    input_tokens=obs.input_tokens,
                    output_tokens=obs.output_tokens,
                    context_size=obs.input_tokens,
                    tool_definitions_tokens=0,
                    system_prompt_hash="imported",
                    system_prompt_tokens=0,
                    output_format="text",
                    is_retry=False,
                    iteration=count,
                    parent_step=parent_name,
                    duration_ms=obs.duration_ms,
                    timestamp=timestamp,
                )
            )

        runs.append(step_records)
    return runs


def extract_inputs(traces: list[LangfuseTrace]) -> list[str]:
    """Extract input texts from traces for use as profiling inputs."""
    inputs = [t.input_text for t in traces if t.input_text]
    if len(inputs) < 2:
        raise ValueError(
            f"Only {len(inputs)} of {len(traces)} traces had extractable "
            "input text. Langfuse traces may not store root inputs "
            "for this workflow."
        )
    return inputs
