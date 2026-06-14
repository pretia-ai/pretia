"""Extract DAG structure from LangGraph, OpenAI Agents SDK, and CrewAI workflows."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def extract_step_names(workflow: Any) -> list[str] | None:
    """Return step/node names from a workflow object, or None if unsupported."""
    if workflow is None:
        return None

    # LangGraph: CompiledGraph has a `nodes` dict mapping name -> node
    if hasattr(workflow, "nodes"):
        nodes = workflow.nodes
        if isinstance(nodes, dict):
            return sorted(nodes.keys())
        try:
            return sorted(str(n) for n in nodes)
        except (TypeError, ValueError):
            logger.debug("Could not iterate workflow.nodes")
            return None

    # OpenAI Agents SDK: Agent has `handoffs` list of target agents
    if hasattr(workflow, "handoffs"):
        handoffs = workflow.handoffs
        if not handoffs:
            return None
        names: list[str] = []
        for h in handoffs:
            name = getattr(h, "name", None)
            if name is not None:
                names.append(str(name))
        return sorted(names) if names else None

    return None
