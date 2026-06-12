"""Tests for the unified validation types and orchestrator."""

from __future__ import annotations

import json
from unittest.mock import patch

from scripts._validation_types import (
    CheckResult,
    CheckStatus,
    StageResult,
    ValidationReport,
)


class TestCheckResult:
    def test_fail_and_blocking_blocks(self):
        cr = CheckResult(
            name="test", status=CheckStatus.FAIL, details={}, blocking=True
        )
        assert cr.status == CheckStatus.FAIL
        assert cr.blocking is True

    def test_warn_does_not_block(self):
        cr = CheckResult(
            name="test", status=CheckStatus.WARN, details={}, blocking=True
        )
        assert cr.status != CheckStatus.FAIL

    def test_fail_non_blocking_does_not_block(self):
        cr = CheckResult(
            name="test", status=CheckStatus.FAIL, details={}, blocking=False
        )
        assert not cr.blocking

    def test_to_dict(self):
        cr = CheckResult(
            name="pricing",
            status=CheckStatus.PASS,
            details={"models": 5},
            blocking=True,
            check_id="2.1",
        )
        d = cr.to_dict()
        assert d["name"] == "pricing"
        assert d["status"] == "PASS"
        assert d["details"] == {"models": 5}
        assert d["blocking"] is True
        assert d["check_id"] == "2.1"

    def test_to_dict_omits_none_check_id(self):
        cr = CheckResult(
            name="test", status=CheckStatus.PASS, details={}, blocking=False
        )
        assert "check_id" not in cr.to_dict()

    def test_frozen(self):
        cr = CheckResult(
            name="test", status=CheckStatus.PASS, details={}, blocking=True
        )
        try:
            cr.name = "other"  # type: ignore[misc]
            raise AssertionError("Should not allow mutation")
        except AttributeError:
            pass


class TestStageResult:
    def test_passed_when_no_blocking_failures(self):
        checks = (
            CheckResult("a", CheckStatus.PASS, {}, blocking=True),
            CheckResult("b", CheckStatus.WARN, {}, blocking=True),
            CheckResult("c", CheckStatus.FAIL, {}, blocking=False),
        )
        sr = StageResult(stage=1, name="Data Readiness", checks=checks, duration_s=1.5)
        assert sr.passed is True

    def test_failed_when_blocking_failure_exists(self):
        checks = (
            CheckResult("a", CheckStatus.PASS, {}, blocking=True),
            CheckResult("b", CheckStatus.FAIL, {}, blocking=True),
        )
        sr = StageResult(stage=1, name="Data Readiness", checks=checks, duration_s=0.5)
        assert sr.passed is False

    def test_blocking_failures_list(self):
        checks = (
            CheckResult("ok", CheckStatus.PASS, {}, blocking=True),
            CheckResult("bad1", CheckStatus.FAIL, {}, blocking=True),
            CheckResult("bad2", CheckStatus.FAIL, {}, blocking=True),
            CheckResult("non_block", CheckStatus.FAIL, {}, blocking=False),
        )
        sr = StageResult(stage=2, name="Infra", checks=checks, duration_s=0.1)
        assert sr.blocking_failures == ["bad1", "bad2"]

    def test_warnings_list(self):
        checks = (
            CheckResult("w", CheckStatus.WARN, {}, blocking=True),
            CheckResult("p", CheckStatus.PASS, {}, blocking=True),
        )
        sr = StageResult(stage=1, name="Data", checks=checks, duration_s=0.1)
        assert sr.warnings == ["w"]

    def test_to_dict(self):
        checks = (
            CheckResult("a", CheckStatus.PASS, {}, blocking=True),
        )
        sr = StageResult(stage=1, name="Data Readiness", checks=checks, duration_s=1.234)
        d = sr.to_dict()
        assert d["stage"] == 1
        assert d["name"] == "Data Readiness"
        assert d["passed"] is True
        assert d["duration_s"] == 1.23
        assert len(d["checks"]) == 1

    def test_empty_checks_passes(self):
        sr = StageResult(stage=3, name="Calibration", checks=(), duration_s=0.0)
        assert sr.passed is True
        assert sr.blocking_failures == []


class TestValidationReport:
    def _make_stage(self, stage: int, passed: bool) -> StageResult:
        status = CheckStatus.PASS if passed else CheckStatus.FAIL
        checks = (CheckResult("check", status, {}, blocking=True),)
        return StageResult(stage=stage, name=f"Stage {stage}", checks=checks, duration_s=0.1)

    def test_all_passed(self):
        stages = (self._make_stage(1, True), self._make_stage(2, True))
        report = ValidationReport(
            timestamp="2026-06-12T00:00:00Z",
            stages=stages,
            total_duration_s=0.5,
        )
        assert report.passed is True
        assert report.max_passed_stage == 2

    def test_partial_failure(self):
        stages = (self._make_stage(1, True), self._make_stage(2, False))
        report = ValidationReport(
            timestamp="2026-06-12T00:00:00Z",
            stages=stages,
            total_duration_s=0.5,
        )
        assert report.passed is False
        assert report.max_passed_stage == 1

    def test_all_failed(self):
        stages = (self._make_stage(1, False),)
        report = ValidationReport(
            timestamp="2026-06-12T00:00:00Z",
            stages=stages,
            total_duration_s=0.5,
        )
        assert report.passed is False
        assert report.max_passed_stage == 0

    def test_to_dict_json_roundtrip(self):
        stages = (self._make_stage(1, True), self._make_stage(2, True))
        report = ValidationReport(
            timestamp="2026-06-12T00:00:00Z",
            stages=stages,
            total_duration_s=1.5,
            api_cost_usd=0.15,
        )
        json_str = report.to_json()
        parsed = json.loads(json_str)
        assert parsed["passed"] is True
        assert parsed["max_passed_stage"] == 2
        assert parsed["api_cost_usd"] == 0.15
        assert len(parsed["stages"]) == 2

    def test_empty_stages(self):
        report = ValidationReport(
            timestamp="2026-06-12T00:00:00Z",
            stages=(),
            total_duration_s=0.0,
        )
        assert report.passed is True
        assert report.max_passed_stage == 0


class TestRunValidation:
    def test_short_circuits_on_blocking_failure(self):
        from scripts.validate import run_validation

        fail_checks = (
            CheckResult("bad", CheckStatus.FAIL, {}, blocking=True),
        )
        fail_stage = StageResult(stage=1, name="Data", checks=fail_checks, duration_s=0.1)
        pass_checks = (
            CheckResult("ok", CheckStatus.PASS, {}, blocking=True),
        )
        pass_stage = StageResult(stage=2, name="Infra", checks=pass_checks, duration_s=0.1)

        with (
            patch("scripts.validate.run_stage_1", return_value=fail_stage),
            patch("scripts.validate.run_stage_2", return_value=pass_stage) as mock_s2,
        ):
            report = run_validation(skip_live=True)
            assert len(report.stages) == 1
            assert report.stages[0].stage == 1
            mock_s2.assert_not_called()

    def test_runs_all_stages_on_success(self):
        from scripts.validate import run_validation

        def _pass_stage(n, name):
            checks = (CheckResult("ok", CheckStatus.PASS, {}, blocking=True),)
            return StageResult(stage=n, name=name, checks=checks, duration_s=0.1)

        with (
            patch("scripts.validate.run_stage_1", return_value=_pass_stage(1, "Data")),
            patch("scripts.validate.run_stage_2", return_value=_pass_stage(2, "Infra")),
            patch("scripts.validate.run_stage_3", return_value=_pass_stage(3, "Calibration")),
        ):
            report = run_validation(skip_live=True)
            assert len(report.stages) == 3
            assert report.passed is True
            assert report.max_passed_stage == 3

    def test_single_stage_mode(self):
        from scripts.validate import run_validation

        def _pass_stage(n, name):
            checks = (CheckResult("ok", CheckStatus.PASS, {}, blocking=True),)
            return StageResult(stage=n, name=name, checks=checks, duration_s=0.1)

        with patch("scripts.validate.run_stage_2", return_value=_pass_stage(2, "Infra")):
            report = run_validation(stages=[2])
            assert len(report.stages) == 1
            assert report.stages[0].stage == 2

    def test_saves_output(self, tmp_path):
        from scripts.validate import run_validation

        def _pass_stage(n, name):
            checks = (CheckResult("ok", CheckStatus.PASS, {}, blocking=True),)
            return StageResult(stage=n, name=name, checks=checks, duration_s=0.1)

        output = tmp_path / "report.json"
        with (
            patch("scripts.validate.run_stage_1", return_value=_pass_stage(1, "Data")),
            patch("scripts.validate.run_stage_2", return_value=_pass_stage(2, "Infra")),
            patch("scripts.validate.run_stage_3", return_value=_pass_stage(3, "Cal")),
        ):
            run_validation(skip_live=True, output=output)

        assert output.exists()
        data = json.loads(output.read_text())
        assert data["passed"] is True
        assert data["max_passed_stage"] == 3
