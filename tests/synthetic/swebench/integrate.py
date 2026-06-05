"""Convert SWE-bench repo groups into SyntheticWorkflow objects."""

from __future__ import annotations

import random

from tests.synthetic.generators import SyntheticWorkflow
from tests.synthetic.swebench.grouper import RepoGroup


def repo_groups_to_synthetic_workflows(
    groups: list[RepoGroup],
    sample_sizes: list[int] | None = None,
) -> list[SyntheticWorkflow]:
    """Convert SWE-bench repo groups into SyntheticWorkflow objects."""
    if sample_sizes is None:
        sample_sizes = [20, 50, 100]

    workflows: list[SyntheticWorkflow] = []

    for group in groups:
        all_costs = group.costs
        m = len(all_costs)

        true_mean = group.mean_cost
        true_p50 = group.median_cost
        true_p95 = group.p95_cost
        true_std = group.std_cost

        for n in sample_sizes:
            if m < n:
                continue

            rng = random.Random(hash((group.repo, n)) % (2**31))
            subsampled = sorted(rng.sample(all_costs, n))

            workflows.append(
                SyntheticWorkflow(
                    name=f"swebench_{group.repo}_n_{n}",
                    distribution_type="swebench",
                    params={
                        "repo": group.repo,
                        "total_instances": m,
                        "model": group.model,
                    },
                    observed_costs=subsampled,
                    true_p50=true_p50,
                    true_p95=true_p95,
                    true_mean=true_mean,
                    true_std=true_std,
                    sample_size=n,
                )
            )

    return workflows
