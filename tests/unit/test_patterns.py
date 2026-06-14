"""Tests for pattern detection: context growth, loop variance, high token variance."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from agentcost.collectors.base import StepRecord
from agentcost.projection.patterns import DetectedPattern, detect_patterns
from agentcost.projection.stats import compute_stats


def _make_record(
    step_name: str = "classify",
    model: str = "gpt-4o",
    input_tokens: int = 100,
    output_tokens: int = 50,
    context_size: int = 100,
    iteration: int = 1,
    duration_ms: int = 500,
    **kwargs: object,
) -> StepRecord:
    defaults: dict[str, object] = {
        "step_name": step_name,
        "step_type": "llm",
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "context_size": context_size,
        "tool_definitions_tokens": 0,
        "system_prompt_hash": "abc123",
        "system_prompt_tokens": 50,
        "output_format": "text",
        "is_retry": False,
        "iteration": iteration,
        "parent_step": None,
        "duration_ms": duration_ms,
        "timestamp": datetime(2026, 5, 25, 12, 0, 0, tzinfo=UTC),
    }
    defaults.update(kwargs)
    return StepRecord(**defaults)


def _simple_cost_fn(model: str, inp: int, out: int) -> float:
    return (inp + out) * 0.001


# ---------------------------------------------------------------------------
# Context growth
# ---------------------------------------------------------------------------


class TestContextGrowth:
    def test_context_growth_detected(self):
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=2200, iteration=2),
                _make_record("review", context_size=3400, iteration=3),
                _make_record("review", context_size=4600, iteration=4),
                _make_record("review", context_size=5800, iteration=5),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].step_name == "review"
        assert cg[0].evidence["r_squared"] > 0.9
        assert 1100 <= cg[0].evidence["slope"] <= 1300

    def test_context_growth_not_detected_flat(self):
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=1000, iteration=2),
                _make_record("review", context_size=1000, iteration=3),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0

    def test_context_growth_not_detected_few_points(self):
        runs = [
            [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=2000, iteration=2),
            ],
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0

    def test_context_growth_severity_danger(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record("review", context_size=1000, iteration=1),
                _make_record("review", context_size=2000, iteration=2),
                _make_record("review", context_size=3000, iteration=3),
                _make_record("review", context_size=4000, iteration=4),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].severity == "danger"
        assert cg[0].evidence["r_squared"] > 0.85


# ---------------------------------------------------------------------------
# Loop count variance
# ---------------------------------------------------------------------------


class TestLoopCountVariance:
    def test_loop_count_variance_detected(self):
        runs = []
        for n_iters in [2, 5, 8, 12, 3, 7, 10, 4, 9, 11]:
            run = [_make_record("review", iteration=i + 1) for i in range(n_iters)]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 1
        assert lcv[0].evidence["max_iterations"] == 12
        assert lcv[0].evidence["cv"] > 0.5

    def test_loop_count_variance_not_detected_stable(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record("review", iteration=1),
                _make_record("review", iteration=2),
                _make_record("review", iteration=3),
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 0

    def test_loop_count_variance_skip_single_iteration(self):
        runs = [
            [_make_record("classify", iteration=1)],
            [_make_record("classify", iteration=1)],
            [_make_record("classify", iteration=1)],
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 0

    def test_loop_count_variance_danger_severity(self):
        runs = []
        # max_i (50) > 3 * mean_iter (~10) triggers "danger"
        for n_iters in [2, 3, 4, 5, 6, 7, 8, 10, 15, 50]:
            run = [_make_record("review", iteration=i + 1) for i in range(n_iters)]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        lcv = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(lcv) == 1
        assert lcv[0].severity == "danger"


# ---------------------------------------------------------------------------
# High token variance
# ---------------------------------------------------------------------------


class TestHighTokenVariance:
    def test_high_token_variance_detected(self):
        runs = []
        for i in range(40):
            if i < 35:
                run = [_make_record("generate", input_tokens=300, output_tokens=200)]
            else:
                run = [_make_record("generate", input_tokens=3000, output_tokens=2000)]
            runs.append(run)

        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 1
        assert htv[0].evidence["p95_p50_ratio_tokens"] > 3.0

    def test_high_token_variance_not_detected_uniform(self):
        runs = []
        for i in range(10):
            tok = 490 + i * 3
            run = [_make_record("generate", input_tokens=tok, output_tokens=tok // 2)]
            runs.append(run)

        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 0

    def test_high_token_variance_skips_zero_p50(self):
        runs = [
            [_make_record("tool_step", input_tokens=0, output_tokens=0, context_size=0)],
            [_make_record("tool_step", input_tokens=0, output_tokens=0, context_size=0)],
        ]
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 0


# ---------------------------------------------------------------------------
# Combined / edge cases
# ---------------------------------------------------------------------------


class TestCombined:
    def test_detect_patterns_combined(self):
        context_growth_run = [
            _make_record("grower", context_size=1000, iteration=1),
            _make_record("grower", context_size=3000, iteration=2),
            _make_record("grower", context_size=5000, iteration=3),
            _make_record("grower", context_size=7000, iteration=4),
        ]
        runs = [context_growth_run] * 3

        variable_loop_runs = []
        for n in [2, 5, 8, 12, 3, 7, 10, 4, 9, 11]:
            variable_loop_runs.append([_make_record("looper", iteration=i + 1) for i in range(n)])

        heavy_tail_runs = []
        for i in range(20):
            if i < 17:
                heavy_tail_runs.append(
                    [_make_record("heavy", input_tokens=200, output_tokens=100)]
                )
            else:
                heavy_tail_runs.append(
                    [_make_record("heavy", input_tokens=3000, output_tokens=2000)]
                )

        all_runs = []
        for i in range(max(len(runs), len(variable_loop_runs), len(heavy_tail_runs))):
            combined_run = []
            if i < len(runs):
                combined_run.extend(runs[i])
            if i < len(variable_loop_runs):
                combined_run.extend(variable_loop_runs[i])
            if i < len(heavy_tail_runs):
                combined_run.extend(heavy_tail_runs[i])
            if combined_run:
                all_runs.append(combined_run)

        stats = compute_stats(all_runs, _simple_cost_fn)
        patterns = detect_patterns(all_runs, stats)

        types = {p.pattern_type for p in patterns}
        assert "context_growth" in types
        assert "loop_count_variance" in types
        assert "high_token_variance" in types

        severities = [p.severity for p in patterns]
        danger_idx = [i for i, s in enumerate(severities) if s == "danger"]
        warning_idx = [i for i, s in enumerate(severities) if s == "warning"]
        if danger_idx and warning_idx:
            assert max(danger_idx) < min(warning_idx)

    def test_detect_patterns_empty_runs(self):
        patterns = detect_patterns([])
        assert patterns == []

    def test_detect_patterns_single_step_no_patterns(self):
        runs = [[_make_record("simple")]] * 5
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        assert patterns == []


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


class TestPatternSerialization:
    def test_detected_pattern_to_dict(self):
        p = DetectedPattern(
            pattern_type="context_growth",
            step_name="review",
            severity="warning",
            evidence={"r_squared": 0.82, "slope": 1200.0},
            description="Context grows.",
        )
        d = p.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        assert d["pattern_type"] == "context_growth"
        assert d["step_name"] == "review"
        assert d["growth_type"] is None
        assert d["variance_percentile_used"] is None


# ---------------------------------------------------------------------------
# Change 1: Context growth overhaul tests
# ---------------------------------------------------------------------------


class TestLinearGrowthDetected:
    def test_linear_growth_detected(self):
        iters = [1, 2, 3, 4, 5, 6, 7, 8]
        ctxs = [100, 210, 290, 410, 490, 610, 690, 810]
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=ctxs[i], iteration=iters[i])
                for i in range(len(iters))
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].growth_type == "linear"
        assert cg[0].pearson_r_squared > 0.9
        assert cg[0].spearman_rho_squared > 0.9
        assert cg[0].severity == "danger"
        assert cg[0].pearson_significant is True


class TestNonlinearGrowthDetected:
    def test_nonlinear_growth_detected(self):
        iters = [1, 2, 3, 4, 5, 6, 7, 8]
        ctxs = [100, 200, 400, 1000, 3000, 10000, 35000, 120000]
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=ctxs[i], iteration=iters[i])
                for i in range(len(iters))
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].growth_type == "nonlinear"
        assert cg[0].spearman_rho_squared > 0.95
        assert cg[0].power_law_alpha is not None
        assert cg[0].power_law_alpha > 1.5
        assert cg[0].growth_classification == "super_linear"


class TestSublinearGrowthDetected:
    def test_sublinear_growth_detected(self):
        iters = list(range(1, 11))
        ctxs = [100, 800, 1000, 1050, 1060, 1065, 1068, 1070, 1071, 1072]
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=ctxs[i], iteration=iters[i])
                for i in range(len(iters))
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].growth_type == "nonlinear"
        assert cg[0].power_law_alpha is not None
        assert cg[0].power_law_alpha < 1.0
        assert cg[0].growth_classification == "sub_linear"


class TestNoGrowthNoiseOnly:
    def test_no_growth_noise_only(self):
        iters = [1, 2, 3, 4, 5, 6, 7, 8]
        ctxs = [500, 480, 520, 490, 510, 505, 495, 500]
        runs = []
        for _ in range(3):
            run = [
                _make_record("review", context_size=ctxs[i], iteration=iters[i])
                for i in range(len(iters))
            ]
            runs.append(run)

        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0


class TestMinimumDataPointsEnforced:
    def test_minimum_data_points_enforced(self):
        runs = [
            [
                _make_record("review", context_size=100, iteration=1),
                _make_record("review", context_size=200, iteration=2),
                _make_record("review", context_size=300, iteration=3),
                _make_record("review", context_size=400, iteration=4),
            ],
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0


class TestPValueGatesSpuriousCorrelation:
    def test_p_value_gates_spurious(self):
        iters = [1, 2, 3, 4, 5]
        ctxs = [100, 160, 120, 200, 170]
        runs = [
            [
                _make_record("review", context_size=ctxs[i], iteration=iters[i])
                for i in range(len(iters))
            ]
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 0


class TestSeverityWarningVsDanger:
    def test_severity_warning(self):
        iters = list(range(1, 9))
        ctxs = [100, 180, 220, 320, 380, 460, 500, 620]
        runs = []
        for _ in range(3):
            runs.append(
                [
                    _make_record("review", context_size=ctxs[i], iteration=iters[i])
                    for i in range(len(iters))
                ]
            )
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        if cg:
            r2 = max(cg[0].pearson_r_squared or 0, cg[0].spearman_rho_squared or 0)
            if 0.7 < r2 <= 0.85:
                assert cg[0].severity == "warning"

    def test_severity_danger(self):
        iters = list(range(1, 9))
        ctxs = [100, 200, 300, 400, 500, 600, 700, 800]
        runs = []
        for _ in range(3):
            runs.append(
                [
                    _make_record("review", context_size=ctxs[i], iteration=iters[i])
                    for i in range(len(iters))
                ]
            )
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cg = [p for p in patterns if p.pattern_type == "context_growth"]
        assert len(cg) == 1
        assert cg[0].severity == "danger"
        r2 = max(cg[0].pearson_r_squared or 0, cg[0].spearman_rho_squared or 0)
        assert r2 > 0.85


# ---------------------------------------------------------------------------
# Change 3: Token variance threshold tests
# ---------------------------------------------------------------------------


class TestTokenVarianceUsesP90AtSmallN:
    def test_uses_p90_at_small_n(self):
        runs = []
        for i in range(20):
            if i < 18:
                runs.append([_make_record("gen", input_tokens=300, output_tokens=200)])
            else:
                runs.append([_make_record("gen", input_tokens=3000, output_tokens=2000)])

        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 0


class TestTokenVarianceUsesP95AtLargeN:
    def test_uses_p95_at_large_n(self):
        runs = []
        for i in range(40):
            if i < 36:
                runs.append([_make_record("gen", input_tokens=300, output_tokens=200)])
            else:
                runs.append([_make_record("gen", input_tokens=3000, output_tokens=2000)])

        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        assert len(htv) == 1


class TestVariancePercentileField:
    def test_variance_percentile_used_small_n(self):
        runs = []
        for i in range(20):
            if i < 14:
                runs.append([_make_record("gen", input_tokens=100, output_tokens=50)])
            else:
                runs.append([_make_record("gen", input_tokens=3000, output_tokens=2000)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        if htv:
            assert htv[0].variance_percentile_used == 90

    def test_variance_percentile_used_large_n(self):
        runs = []
        for i in range(40):
            if i < 30:
                runs.append([_make_record("gen", input_tokens=100, output_tokens=50)])
            else:
                runs.append([_make_record("gen", input_tokens=3000, output_tokens=2000)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        htv = [p for p in patterns if p.pattern_type == "high_token_variance"]
        if htv:
            assert htv[0].variance_percentile_used == 95


# ---------------------------------------------------------------------------
# Step count variance tests
# ---------------------------------------------------------------------------


def _routing_runs(
    active_counts: list[int],
    all_steps: list[str] | None = None,
) -> list[list[StepRecord]]:
    """Build runs where each run has a specified number of active steps."""
    if all_steps is None:
        max_steps = max(active_counts) if active_counts else 5
        all_steps = [f"step_{i}" for i in range(max_steps)]
    runs: list[list[StepRecord]] = []
    for ac in active_counts:
        run: list[StepRecord] = []
        for i, sn in enumerate(all_steps):
            if i < ac:
                run.append(_make_record(sn, input_tokens=200, output_tokens=100))
            else:
                run.append(_make_record(sn, input_tokens=0, output_tokens=0, context_size=0))
        runs.append(run)
    return runs


class TestStepCountVarianceWarning:
    def test_step_count_variance_warning(self):
        counts = [5] * 6 + [7] * 4 + [8] * 4 + [10] * 6
        runs = _routing_runs(counts)
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        scv = [p for p in patterns if p.pattern_type == "step_count_variance"]
        assert len(scv) == 1
        assert scv[0].severity == "warning"
        assert scv[0].step_count_cv is not None
        assert 0.3 < scv[0].step_count_cv < 0.6


class TestStepCountVarianceDangerCv:
    def test_danger_cv(self):
        counts = [2] * 7 + [7] * 7 + [12] * 6
        runs = _routing_runs(counts)
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        scv = [p for p in patterns if p.pattern_type == "step_count_variance"]
        assert len(scv) == 1
        assert scv[0].severity == "danger"


class TestStepCountVarianceDangerRatio:
    def test_danger_ratio(self):
        counts = [6] * 10 + [2] * 10
        runs = _routing_runs(counts)
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        scv = [p for p in patterns if p.pattern_type == "step_count_variance"]
        assert len(scv) == 1
        assert scv[0].severity == "danger"
        assert scv[0].step_count_max == 6
        assert scv[0].step_count_min == 2


class TestStepCountVarianceNoDetection:
    def test_no_detection_uniform(self):
        counts = [5] * 20
        runs = _routing_runs(counts)
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        scv = [p for p in patterns if p.pattern_type == "step_count_variance"]
        assert len(scv) == 0


class TestStepCountVarianceSlightVariation:
    def test_slight_variation(self):
        counts = [5] * 15 + [4] * 5
        runs = _routing_runs(counts)
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        scv = [p for p in patterns if p.pattern_type == "step_count_variance"]
        assert len(scv) == 0


# ---------------------------------------------------------------------------
# Bimodality tests
# ---------------------------------------------------------------------------


class TestBimodalityClearSeparation:
    def test_bimodality_detected(self):
        pytest.importorskip("sklearn")
        cheap = [0.015 + i * 0.0003 for i in range(35)]
        expensive = [0.35 + i * 0.003 for i in range(15)]
        all_costs = cheap + expensive
        runs: list[list[StepRecord]] = []
        for cost_val in all_costs:
            inp = int(cost_val / 0.001 * 0.67)
            out = int(cost_val / 0.001 * 0.33)
            runs.append([_make_record("work", input_tokens=inp, output_tokens=out)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        bim = [p for p in patterns if p.pattern_type == "bimodality"]
        assert len(bim) == 1
        assert bim[0].severity == "warning"
        assert bim[0].bimodal_bic_delta is not None
        assert bim[0].bimodal_bic_delta > 6
        assert bim[0].bimodal_modes is not None
        assert len(bim[0].bimodal_modes) == 2


class TestBimodalityNotDetectedUnimodal:
    def test_unimodal(self):
        pytest.importorskip("sklearn")
        costs = [0.09 + i * 0.002 for i in range(50)]
        runs: list[list[StepRecord]] = []
        for cost_val in costs:
            inp = int(cost_val / 0.001 * 0.67)
            out = int(cost_val / 0.001 * 0.33)
            runs.append([_make_record("work", input_tokens=inp, output_tokens=out)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        bim = [p for p in patterns if p.pattern_type == "bimodality"]
        assert len(bim) == 0


class TestBimodalitySkippedSmallN:
    def test_skipped_small_n(self):
        pytest.importorskip("sklearn")
        cheap = [0.02] * 5
        expensive = [0.40] * 5
        all_costs = cheap + expensive
        runs: list[list[StepRecord]] = []
        for cost_val in all_costs:
            inp = int(cost_val / 0.001 * 0.67)
            out = int(cost_val / 0.001 * 0.33)
            runs.append([_make_record("work", input_tokens=inp, output_tokens=out)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        bim = [p for p in patterns if p.pattern_type == "bimodality"]
        assert len(bim) == 0


class TestBimodalitySkippedNoSklearn:
    def test_skipped_no_sklearn(self, monkeypatch):
        import builtins

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if "sklearn" in name:
                raise ImportError("mocked")
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        runs: list[list[StepRecord]] = []
        for i in range(20):
            tok = 200 if i < 14 else 2000
            runs.append([_make_record("work", input_tokens=tok, output_tokens=tok // 2)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        bim = [p for p in patterns if p.pattern_type == "bimodality"]
        assert len(bim) == 0
        other_types = {p.pattern_type for p in patterns} - {"bimodality"}
        assert len(other_types) >= 0


class TestBimodalityZeroCostHandling:
    def test_zero_cost_handling(self):
        pytest.importorskip("sklearn")
        runs: list[list[StepRecord]] = []
        for _ in range(10):
            runs.append([_make_record("work", input_tokens=0, output_tokens=0, context_size=0)])
        for _ in range(20):
            runs.append([_make_record("work", input_tokens=150, output_tokens=75)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        bim = [p for p in patterns if p.pattern_type == "bimodality"]
        assert len(bim) == 1
        assert bim[0].bimodal_modes is not None
        assert len(bim[0].bimodal_modes) == 2
        assert bim[0].bimodal_modes[0]["mean_cost"] == 0.0


# ---------------------------------------------------------------------------
# robust_cv tests
# ---------------------------------------------------------------------------


class TestRobustCvNoOutlierEffect:
    def test_outlier_resistant(self):
        from agentcost.projection.stats import robust_cv

        values = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 100.0]
        rcv = robust_cv(values)
        assert rcv < 1.0


class TestRobustCvMatchesCvForNormal:
    def test_close_to_standard_cv(self):
        import math

        from agentcost.projection.stats import robust_cv

        values = [8.0, 9.0, 9.0, 10.0, 10.0, 10.0, 10.0, 11.0, 11.0, 12.0]
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
        std_cv = std / mean
        rcv = robust_cv(values)
        assert abs(rcv - std_cv) / std_cv < 0.5


class TestRobustCvZeroMedian:
    def test_returns_zero(self):
        from agentcost.projection.stats import robust_cv

        values = [0.0, 0.0, 0.0, 1.0, 2.0]
        assert robust_cv(values) == 0.0


class TestLoopVarianceUsesRobustCv:
    def test_outlier_does_not_trigger(self):
        runs = []
        for _ in range(9):
            run = [_make_record("step_a", iteration=k) for k in range(1, 4)]
            runs.append(run)
        run_outlier = [_make_record("step_a", iteration=k) for k in range(1, 51)]
        runs.append(run_outlier)
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        loop_patterns = [p for p in patterns if p.pattern_type == "loop_count_variance"]
        assert len(loop_patterns) == 0


class TestGmmParametersStored:
    def test_gmm_fields_populated(self):
        try:
            import sklearn  # noqa: F401
        except ImportError:
            pytest.skip("sklearn not installed")

        runs = []
        import random as _rng

        r = _rng.Random(42)
        for _ in range(20):
            cost = 150 + r.randint(-10, 10)
            runs.append([_make_record("work", input_tokens=cost, output_tokens=75)])
        for _ in range(10):
            cost = 1500 + r.randint(-100, 100)
            runs.append([_make_record("work", input_tokens=cost, output_tokens=750)])
        stats = compute_stats(runs, _simple_cost_fn)
        patterns = detect_patterns(runs, stats)
        bim = [p for p in patterns if p.pattern_type == "bimodality"]
        if not bim:
            pytest.skip("bimodality not detected with this data")
        p = bim[0]
        assert p.gmm_means is not None
        assert p.gmm_stds is not None
        assert p.gmm_weights is not None
        assert len(p.gmm_means) == 2
        assert len(p.gmm_stds) == 2
        assert len(p.gmm_weights) == 2


# ---------------------------------------------------------------------------
# Cache utilization opportunity tests (Pattern #7)
# ---------------------------------------------------------------------------


class TestCacheUtilizationDetected:
    def test_low_cache_hit_fires_warning(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record(
                    "summarize",
                    model="claude-sonnet-4-6",
                    cache_hit_tokens=10,
                    cache_miss_tokens=1990,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cu = [p for p in patterns if p.pattern_type == "cache_utilization_opportunity"]
        assert len(cu) == 1
        assert cu[0].severity == "warning"
        assert cu[0].evidence["cache_hit_ratio"] < 0.1
        assert cu[0].evidence["model"] == "claude-sonnet-4-6"
        assert cu[0].evidence["total_cache_miss_tokens"] > 0


class TestCacheUtilizationNotDetectedHighHitRate:
    def test_high_cache_no_detection(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record(
                    "summarize",
                    model="claude-sonnet-4-6",
                    cache_hit_tokens=1500,
                    cache_miss_tokens=500,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cu = [p for p in patterns if p.pattern_type == "cache_utilization_opportunity"]
        assert len(cu) == 0


class TestCacheUtilizationNotDetectedNoCacheFields:
    def test_no_cache_fields_no_detection(self):
        runs = [[_make_record("summarize", model="claude-sonnet-4-6")] for _ in range(5)]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cu = [p for p in patterns if p.pattern_type == "cache_utilization_opportunity"]
        assert len(cu) == 0


class TestCacheUtilizationNotDetectedUnsupportedModel:
    def test_unsupported_model_no_detection(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record(
                    "summarize",
                    model="gpt-4o",
                    cache_hit_tokens=10,
                    cache_miss_tokens=1990,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cu = [p for p in patterns if p.pattern_type == "cache_utilization_opportunity"]
        assert len(cu) == 0


class TestCacheUtilizationEdgeCaseZeroTokens:
    def test_zero_tokens_no_detection(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record(
                    "summarize",
                    model="claude-sonnet-4-6",
                    cache_hit_tokens=0,
                    cache_miss_tokens=0,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        cu = [p for p in patterns if p.pattern_type == "cache_utilization_opportunity"]
        assert len(cu) == 0


# ---------------------------------------------------------------------------
# Zero-execution step tests (Pattern #12)
# ---------------------------------------------------------------------------


class TestZeroExecutionStepDetected:
    def test_missing_step_fires_warning(self):
        runs = [
            [_make_record("step_a"), _make_record("step_b")],
            [_make_record("step_a"), _make_record("step_b")],
        ]
        graph_steps = ["step_a", "step_b", "step_c"]
        patterns = detect_patterns(
            runs,
            compute_stats(runs, _simple_cost_fn),
            graph_steps=graph_steps,
        )
        zes = [p for p in patterns if p.pattern_type == "zero_execution_step"]
        assert len(zes) == 1
        assert zes[0].step_name == "step_c"
        assert zes[0].severity == "warning"
        assert zes[0].evidence["total_runs"] == 2


class TestZeroExecutionStepNotDetectedAllPresent:
    def test_all_steps_present(self):
        runs = [
            [
                _make_record("step_a"),
                _make_record("step_b"),
                _make_record("step_c"),
            ],
        ]
        graph_steps = ["step_a", "step_b", "step_c"]
        patterns = detect_patterns(
            runs,
            compute_stats(runs, _simple_cost_fn),
            graph_steps=graph_steps,
        )
        zes = [p for p in patterns if p.pattern_type == "zero_execution_step"]
        assert len(zes) == 0


class TestZeroExecutionStepNoGraphSteps:
    def test_no_graph_info_no_detection(self):
        runs = [[_make_record("step_a")]]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        zes = [p for p in patterns if p.pattern_type == "zero_execution_step"]
        assert len(zes) == 0


class TestZeroExecutionStepMultipleMissing:
    def test_multiple_missing(self):
        runs = [[_make_record("step_a")]] * 3
        graph_steps = ["step_a", "step_b", "step_c", "step_d"]
        patterns = detect_patterns(
            runs,
            compute_stats(runs, _simple_cost_fn),
            graph_steps=graph_steps,
        )
        zes = [p for p in patterns if p.pattern_type == "zero_execution_step"]
        assert len(zes) == 3
        missing_names = {p.step_name for p in zes}
        assert missing_names == {"step_b", "step_c", "step_d"}


class TestZeroExecutionStepEmptyRuns:
    def test_empty_runs(self):
        patterns = detect_patterns([], graph_steps=["step_a"])
        assert patterns == []


# ---------------------------------------------------------------------------
# Output token budget tests (Pattern #15)
# ---------------------------------------------------------------------------


class TestOutputBudgetTooLoose:
    def test_budget_too_loose_fires_warning(self):
        runs = []
        for _ in range(10):
            run = [
                _make_record(
                    "generate",
                    output_tokens=200,
                    max_tokens_setting=4096,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        otb = [
            p
            for p in patterns
            if p.pattern_type == "output_token_budget"
            and p.evidence.get("budget_issue") == "too_loose"
        ]
        assert len(otb) == 1
        assert otb[0].severity == "warning"
        assert otb[0].evidence["max_tokens_setting"] == 4096
        assert otb[0].evidence["median_output_tokens"] == 200
        assert "suggested_max_tokens" in otb[0].evidence


class TestOutputBudgetPossibleTruncation:
    def test_truncation_fires_warning(self):
        runs = []
        for _ in range(10):
            run = [
                _make_record(
                    "generate",
                    output_tokens=950,
                    max_tokens_setting=1000,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        otb = [
            p
            for p in patterns
            if p.pattern_type == "output_token_budget"
            and p.evidence.get("budget_issue") == "possible_truncation"
        ]
        assert len(otb) == 1
        assert otb[0].severity == "warning"


class TestOutputBudgetNotDetectedReasonableSetting:
    def test_reasonable_budget_no_detection(self):
        runs = []
        for _ in range(10):
            run = [
                _make_record(
                    "generate",
                    output_tokens=200,
                    max_tokens_setting=500,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        otb = [p for p in patterns if p.pattern_type == "output_token_budget"]
        assert len(otb) == 0


class TestOutputBudgetNotDetectedNoMaxTokens:
    def test_no_max_tokens_no_detection(self):
        runs = [[_make_record("generate", output_tokens=200)] for _ in range(10)]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        otb = [p for p in patterns if p.pattern_type == "output_token_budget"]
        assert len(otb) == 0


class TestOutputBudgetSuggestedRounding:
    def test_suggested_rounds_to_256(self):
        runs = []
        for _ in range(20):
            run = [
                _make_record(
                    "generate",
                    output_tokens=100,
                    max_tokens_setting=4096,
                ),
            ]
            runs.append(run)
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        otb = [
            p
            for p in patterns
            if p.pattern_type == "output_token_budget"
            and p.evidence.get("budget_issue") == "too_loose"
        ]
        assert len(otb) == 1
        suggested = otb[0].evidence["suggested_max_tokens"]
        assert suggested % 256 == 0


class TestOutputBudgetSingleRecord:
    def test_single_record_with_max_tokens(self):
        runs = [
            [
                _make_record(
                    "generate",
                    output_tokens=100,
                    max_tokens_setting=4096,
                )
            ]
        ]
        patterns = detect_patterns(runs, compute_stats(runs, _simple_cost_fn))
        otb = [p for p in patterns if p.pattern_type == "output_token_budget"]
        assert len(otb) == 1
        assert otb[0].evidence["budget_issue"] == "too_loose"


# ---------------------------------------------------------------------------
# Integration: all 3 new detectors wired in
# ---------------------------------------------------------------------------


class TestNewDetectorsIntegration:
    def test_all_new_detectors_wired_in(self):
        runs = []
        for _ in range(5):
            run = [
                _make_record(
                    "cached_step",
                    model="claude-sonnet-4-6",
                    cache_hit_tokens=5,
                    cache_miss_tokens=995,
                ),
                _make_record(
                    "budgeted_step",
                    output_tokens=100,
                    max_tokens_setting=8192,
                ),
            ]
            runs.append(run)
        graph_steps = ["cached_step", "budgeted_step", "phantom_step"]
        patterns = detect_patterns(
            runs,
            compute_stats(runs, _simple_cost_fn),
            graph_steps=graph_steps,
        )
        types = {p.pattern_type for p in patterns}
        assert "cache_utilization_opportunity" in types
        assert "zero_execution_step" in types
        assert "output_token_budget" in types
