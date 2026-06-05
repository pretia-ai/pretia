"""Tests for SWE-bench trajectory parsing, grouping, and integration."""

from __future__ import annotations

import pytest

from tests.synthetic.swebench.grouper import RepoGroup, group_by_repo
from tests.synthetic.swebench.integrate import repo_groups_to_synthetic_workflows
from tests.synthetic.swebench.parser import SWEBenchInstance, extract_repo

# ---------------------------------------------------------------------------
# Repo extraction
# ---------------------------------------------------------------------------


class TestExtractRepoStandard:
    def test_standard(self):
        assert extract_repo("django__django-12345") == "django"


class TestExtractRepoHyphenated:
    def test_hyphenated(self):
        assert extract_repo("scikit-learn__scikit-learn-6789") == "scikit-learn"


class TestExtractRepoPallets:
    def test_pallets(self):
        assert extract_repo("pallets__flask-1234") == "flask"


class TestExtractRepoEdgeCase:
    def test_edge_case(self):
        assert extract_repo("psf__requests-5678") == "requests"


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------


def _mock_instances(repo: str, n: int, base_cost: float = 0.05) -> list[SWEBenchInstance]:
    return [
        SWEBenchInstance(
            instance_id=f"{repo}__{repo}-{i}",
            repo=repo,
            total_tokens=1000 * (i + 1),
            total_cost=base_cost * (i + 1),
            input_tokens=None,
            output_tokens=None,
            model="claude-sonnet-4-6",
            num_steps=5,
        )
        for i in range(n)
    ]


class TestGroupByRepoMinimumFilter:
    def test_minimum_filter(self):
        instances = _mock_instances("django", 25) + _mock_instances("flask", 5)
        groups = group_by_repo(instances, min_instances=20)
        repo_names = [g.repo for g in groups]
        assert "django" in repo_names
        assert "flask" not in repo_names


class TestGroupByRepoStatistics:
    def test_statistics(self):
        instances = [
            SWEBenchInstance(
                instance_id=f"test__test-{i}",
                repo="test",
                total_tokens=1000,
                total_cost=float(i + 1),
                input_tokens=None,
                output_tokens=None,
                model=None,
                num_steps=None,
            )
            for i in range(50)
        ]
        groups = group_by_repo(instances, min_instances=20)
        assert len(groups) == 1
        g = groups[0]
        assert g.mean_cost == pytest.approx(25.5, rel=0.01)
        assert g.median_cost == pytest.approx(25.5, rel=0.05)


class TestGroupByRepoCapAt300:
    def test_cap(self):
        instances = _mock_instances("bigrepo", 500)
        groups = group_by_repo(instances, min_instances=20, max_instances=300)
        assert len(groups) == 1
        assert len(groups[0].costs) == 300


# ---------------------------------------------------------------------------
# Integration
# ---------------------------------------------------------------------------


class TestRepoToSyntheticWorkflowStructure:
    def test_structure(self):
        group = RepoGroup(
            repo="django",
            instance_count=100,
            costs=[float(i) for i in range(1, 101)],
            mean_cost=50.5,
            median_cost=50.5,
            std_cost=29.0,
            p95_cost=95.5,
            model="claude-sonnet-4-6",
        )
        workflows = repo_groups_to_synthetic_workflows([group])
        assert len(workflows) == 3
        for wf in workflows:
            assert wf.distribution_type == "swebench"


class TestRepoToSyntheticSmallGroup:
    def test_small_group(self):
        group = RepoGroup(
            repo="flask",
            instance_count=35,
            costs=[float(i) for i in range(1, 36)],
            mean_cost=18.0,
            median_cost=18.0,
            std_cost=10.2,
            p95_cost=34.0,
            model=None,
        )
        workflows = repo_groups_to_synthetic_workflows([group])
        sizes = [wf.sample_size for wf in workflows]
        assert 20 in sizes
        assert 50 not in sizes
        assert 100 not in sizes


class TestRepoToSyntheticTruePercentilesFromFull:
    def test_true_from_full(self):
        costs = [float(i) for i in range(1, 101)]
        group = RepoGroup(
            repo="django",
            instance_count=100,
            costs=costs,
            mean_cost=50.5,
            median_cost=50.5,
            std_cost=29.0,
            p95_cost=95.5,
            model=None,
        )
        workflows = repo_groups_to_synthetic_workflows([group])
        n20_wf = next(wf for wf in workflows if wf.sample_size == 20)
        assert n20_wf.true_p50 == pytest.approx(50.5, rel=0.01)
        assert len(n20_wf.observed_costs) == 20


class TestSwebenchIntegrationOptional:
    def test_optional(self, monkeypatch):

        monkeypatch.setattr(
            "tests.synthetic.swebench.download.download_swebench_data",
            lambda **kw: (_ for _ in ()).throw(FileNotFoundError("no data")),
        )
        from tests.synthetic.generators import generate_lognormal, generate_uniform

        workflows = [generate_lognormal(0.5, 20, seed=1), generate_uniform(20, seed=2)]
        from tests.synthetic.runner import run_one

        results = [run_one(wf, daily_volume=10) for wf in workflows]
        assert len(results) == 2
