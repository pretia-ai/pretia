"""Rate-limit-aware concurrency for backtesting workflows.

Maps each workflow to its provider(s), defines safe per-provider parallelism
caps, and partitions workflows into concurrent groups that avoid sharing a
provider's rate limit pool.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Safe within-workflow parallelism per provider, derived from official rate limits.
# Anthropic Tier 2: 1,000 RPM per model class → 10 concurrent is safe.
# DeepSeek: 500 concurrent connections (Pro) → 10 is conservative.
# OpenAI Tier 1: 500 RPM but only 30K TPM for GPT-4.1 → 5 is safe.
# Gemini Tier 1: 300 RPM Flash → 10 is safe.
# Qwen Singapore: 240 RPM turbo → 5 is conservative.
PROVIDER_PARALLEL: dict[str, int] = {
    "anthropic": 100,
    "deepseek": 100,
    "openai": 100,
    "gemini": 100,
    "qwen": 10,
}

# Per-workflow overrides for rate-limit-constrained workflows.
WORKFLOW_PARALLEL_OVERRIDE: dict[str, int] = {
    "W9": 15,  # GPT-4.1 has 30K TPM at Tier 1
}

WORKFLOW_PROVIDERS: dict[str, list[str]] = {
    "W1": ["anthropic"],
    "W2": ["deepseek"],
    "W4": ["deepseek", "qwen"],  # excluded from active suite but kept for reference
    "W5": ["anthropic"],
    "W9": ["openai"],
    "W11": ["qwen"],
    "W12": ["deepseek"],
    "W13": ["anthropic"],
    "W14": ["openai", "anthropic"],
    "W15": ["openai", "gemini", "deepseek"],
    "W16": ["anthropic"],
    "W17": ["anthropic", "openai"],
    "W18": ["deepseek"],
    "W19": ["deepseek"],
}


def get_parallel_for_workflow(workflow_id: str) -> int:
    """Return the safe --parallel value for a workflow's most constrained provider."""
    wf = workflow_id.upper().split("-")[0]
    if wf in WORKFLOW_PARALLEL_OVERRIDE:
        return WORKFLOW_PARALLEL_OVERRIDE[wf]
    providers = WORKFLOW_PROVIDERS.get(wf, [])
    if not providers:
        return 5
    return min(PROVIDER_PARALLEL.get(p, 5) for p in providers)


def _workflows_conflict(a: str, b: str) -> bool:
    """Return True if two workflows share any provider."""
    providers_a = set(WORKFLOW_PROVIDERS.get(a, []))
    providers_b = set(WORKFLOW_PROVIDERS.get(b, []))
    return bool(providers_a & providers_b)


def build_concurrent_groups(workflow_ids: list[str]) -> list[list[str]]:
    """Partition workflows into groups that can run concurrently.

    Two workflows conflict if they share any provider. Within each group,
    no two workflows conflict — so the group can run with asyncio.gather().
    Groups are executed sequentially.

    Uses greedy graph coloring: assign each workflow to the first group
    where it has no conflicts.
    """
    normalized = [wf.upper().split("-")[0] for wf in workflow_ids]
    groups: list[list[str]] = []

    for wf in normalized:
        placed = False
        for group in groups:
            if all(not _workflows_conflict(wf, existing) for existing in group):
                group.append(wf)
                placed = True
                break
        if not placed:
            groups.append([wf])

    logger.info(
        "Built %d concurrent groups from %d workflows: %s",
        len(groups),
        len(normalized),
        [f"[{', '.join(g)}]" for g in groups],
    )
    return groups
