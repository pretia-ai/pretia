"""Test rate-limit-aware concurrency grouping from tests/backtesting/concurrency.py."""

from __future__ import annotations

from tests.backtesting.concurrency import (
    PROVIDER_PARALLEL,
    WORKFLOW_PROVIDERS,
    _workflows_conflict,
    build_concurrent_groups,
    get_parallel_for_workflow,
)


class TestGetParallelForWorkflow:
    def test_anthropic_only(self) -> None:
        assert get_parallel_for_workflow("W1") == PROVIDER_PARALLEL["anthropic"]

    def test_deepseek_only(self) -> None:
        assert get_parallel_for_workflow("W2") == PROVIDER_PARALLEL["deepseek"]

    def test_multi_provider_takes_min(self) -> None:
        result = get_parallel_for_workflow("W14")
        providers = WORKFLOW_PROVIDERS["W14"]
        expected = min(PROVIDER_PARALLEL[p] for p in providers)
        assert result == expected

    def test_w9_uses_override(self) -> None:
        assert get_parallel_for_workflow("W9") == 15  # WORKFLOW_PARALLEL_OVERRIDE

    def test_handles_suffix(self) -> None:
        assert get_parallel_for_workflow("W1-support-simple") == get_parallel_for_workflow("W1")

    def test_unknown_workflow_returns_default(self) -> None:
        assert get_parallel_for_workflow("W99") == 5


class TestWorkflowsConflict:
    def test_same_provider_conflicts(self) -> None:
        assert _workflows_conflict("W1", "W13")  # both Anthropic

    def test_different_providers_no_conflict(self) -> None:
        assert not _workflows_conflict("W1", "W2")  # Anthropic vs DeepSeek

    def test_multi_provider_conflicts_on_shared(self) -> None:
        assert _workflows_conflict("W14", "W17")  # both use Anthropic + OpenAI

    def test_multi_vs_single_conflicts(self) -> None:
        assert _workflows_conflict("W14", "W1")  # W14 uses Anthropic, W1 uses Anthropic

    def test_no_self_conflict_trivially(self) -> None:
        assert _workflows_conflict("W1", "W1")  # same workflow always conflicts

    def test_deepseek_and_qwen_no_conflict_with_anthropic(self) -> None:
        assert not _workflows_conflict("W4", "W1")  # DeepSeek+Qwen vs Anthropic

    def test_w15_conflicts_with_deepseek_workflows(self) -> None:
        assert _workflows_conflict("W15", "W2")  # W15 uses DeepSeek, W2 uses DeepSeek


class TestBuildConcurrentGroups:
    def test_independent_workflows_in_one_group(self) -> None:
        groups = build_concurrent_groups(["W9", "W11", "W12"])
        assert len(groups) == 1
        assert set(groups[0]) == {"W9", "W11", "W12"}

    def test_conflicting_workflows_in_separate_groups(self) -> None:
        groups = build_concurrent_groups(["W1", "W13"])
        assert len(groups) == 2

    def test_single_workflow_one_group(self) -> None:
        groups = build_concurrent_groups(["W1"])
        assert len(groups) == 1
        assert groups[0] == ["W1"]

    def test_all_14_workflows_no_group_has_conflicts(self) -> None:
        all_wfs = list(WORKFLOW_PROVIDERS.keys())
        groups = build_concurrent_groups(all_wfs)
        for group in groups:
            for i, a in enumerate(group):
                for b in group[i + 1 :]:
                    assert not _workflows_conflict(a, b), (
                        f"{a} and {b} conflict but are in the same group"
                    )

    def test_all_workflows_present(self) -> None:
        all_wfs = list(WORKFLOW_PROVIDERS.keys())
        groups = build_concurrent_groups(all_wfs)
        result = [wf for g in groups for wf in g]
        assert sorted(result) == sorted(all_wfs)

    def test_fewer_groups_than_workflows(self) -> None:
        all_wfs = list(WORKFLOW_PROVIDERS.keys())
        groups = build_concurrent_groups(all_wfs)
        assert len(groups) < len(all_wfs)

    def test_empty_input(self) -> None:
        assert build_concurrent_groups([]) == []

    def test_handles_suffixed_names(self) -> None:
        groups = build_concurrent_groups(["W1-support-simple", "W2-support-complex"])
        assert len(groups) == 1

    def test_mixed_providers_maximizes_concurrency(self) -> None:
        groups = build_concurrent_groups(["W9", "W11", "W12", "W1"])
        first_group = groups[0]
        assert len(first_group) >= 3
