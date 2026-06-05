"""Group SWE-bench instances by repository and compute cost distributions."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from dataclasses import dataclass

from tests.synthetic.swebench.parser import SWEBenchInstance


def _percentile(sorted_data: list[float], p: float) -> float:
    n = len(sorted_data)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_data[0]
    k = (n - 1) * p / 100.0
    f = int(k)
    c = f + 1
    if c >= n:
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@dataclass
class RepoGroup:
    """Grouped SWE-bench instances for one repository."""

    repo: str
    instance_count: int
    costs: list[float]
    mean_cost: float
    median_cost: float
    std_cost: float
    p95_cost: float
    model: str | None


def group_by_repo(
    instances: list[SWEBenchInstance],
    min_instances: int = 20,
    max_instances: int = 300,
    seed: int = 42,
) -> list[RepoGroup]:
    """Group instances by repo. Only include repos with >= min_instances."""
    by_repo: dict[str, list[SWEBenchInstance]] = defaultdict(list)
    for inst in instances:
        by_repo[inst.repo].append(inst)

    rng = random.Random(seed)
    groups: list[RepoGroup] = []

    for repo, insts in sorted(by_repo.items(), key=lambda x: -len(x[1])):
        if len(insts) < min_instances:
            continue

        if len(insts) > max_instances:
            insts = rng.sample(insts, max_instances)

        costs = sorted(inst.total_cost for inst in insts)
        n = len(costs)
        mean = sum(costs) / n
        var = sum((c - mean) ** 2 for c in costs) / (n - 1) if n > 1 else 0.0

        models = [i.model for i in insts if i.model]
        most_common: str | None = None
        if models:
            counts: dict[str, int] = defaultdict(int)
            for m in models:
                counts[m] += 1
            most_common = max(counts, key=lambda m: counts[m])

        groups.append(
            RepoGroup(
                repo=repo,
                instance_count=n,
                costs=costs,
                mean_cost=mean,
                median_cost=_percentile(costs, 50),
                std_cost=math.sqrt(var),
                p95_cost=_percentile(costs, 95),
                model=most_common,
            )
        )

    return groups
