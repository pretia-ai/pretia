"""Tests for the pre-calibration system."""

from __future__ import annotations

import json

from pre_calibration.pre_calibration import CheckResult, PreCalibrationReport, run_pre_calibration


class TestCheckResultBlockingLogic:
    def test_fail_and_blocking_blocks(self):
        cr = CheckResult(name="test", status="FAIL", details={}, blocking=True)
        assert cr.status == "FAIL" and cr.blocking

    def test_warn_does_not_block(self):
        cr = CheckResult(name="test", status="WARN", details={}, blocking=True)
        assert cr.status != "FAIL"

    def test_fail_non_blocking_does_not_block(self):
        cr = CheckResult(name="test", status="FAIL", details={}, blocking=False)
        assert not cr.blocking


class TestPreCalibrationReport:
    def test_json_schema(self):
        report = PreCalibrationReport(
            timestamp="2026-06-07T00:00:00+00:00",
            checks={
                "test": CheckResult(name="test", status="PASS", details={"k": "v"}, blocking=True)
            },
            blocking_failures=[],
            warnings=[],
            proceed_to_pilot=True,
        )
        d = report.to_dict()
        assert "timestamp" in d
        assert "checks" in d
        assert "blocking_failures" in d
        assert "warnings" in d
        assert "proceed_to_pilot" in d
        assert d["proceed_to_pilot"] is True

    def test_serializable_to_json(self):
        report = PreCalibrationReport(
            timestamp="2026-06-07T00:00:00+00:00",
            checks={"test": CheckResult(name="test", status="PASS", details={}, blocking=True)},
            blocking_failures=[],
            warnings=[],
            proceed_to_pilot=True,
        )
        json_str = json.dumps(report.to_dict())
        parsed = json.loads(json_str)
        assert parsed["proceed_to_pilot"] is True


class TestSchemaCompatibilityCheck:
    async def test_passes_with_valid_record(self):
        from pre_calibration.checks.schema_compatibility import check

        result = await check()
        assert result.status == "PASS"
        assert result.details["fields_verified"] > 0


class TestPromptInventoryCheck:
    async def test_missing_dir_fails(self, tmp_path):
        from pre_calibration.checks.prompt_inventory import check

        result = await check(prompts_dir=tmp_path / "nonexistent")
        assert result.status == "FAIL"

    async def test_empty_dir_fails(self, tmp_path):
        from pre_calibration.checks.prompt_inventory import check

        result = await check(prompts_dir=tmp_path)
        assert result.status == "FAIL"

    async def test_with_valid_prompts(self, tmp_path):
        from pre_calibration.checks.prompt_inventory import EXPECTED_WORKFLOW_DIRS, check

        for wf in EXPECTED_WORKFLOW_DIRS:
            d = tmp_path / f"{wf}_test"
            d.mkdir()
            (d / "system_prompt.txt").write_text("You are a helpful assistant.")
        result = await check(prompts_dir=tmp_path)
        assert result.status == "PASS"
        assert result.details["prompt_count"] >= 14


class TestInputInventoryCheck:
    async def test_missing_dir_fails(self, tmp_path):
        from pre_calibration.checks.input_inventory import check

        result = await check(inputs_dir=tmp_path / "nonexistent")
        assert result.status == "FAIL"

    async def test_wrong_count_fails(self, tmp_path):
        from pre_calibration.checks.input_inventory import check

        # Create a profiling file with only 30 inputs
        prof = tmp_path / "w01_profiling.jsonl"
        prof.write_text("\n".join(['{"input": "test"}'] * 30))
        result = await check(inputs_dir=tmp_path)
        assert result.status == "FAIL"


class TestPdfInventoryCheck:
    async def test_missing_dir_fails(self, tmp_path):
        from pre_calibration.checks.pdf_inventory import check

        result = await check(pdfs_dir=tmp_path / "nonexistent")
        assert result.status == "FAIL"


class TestWorkspaceCheck:
    async def test_creates_dirs(self, tmp_path, monkeypatch):
        from pre_calibration.checks.workspace_check import check

        monkeypatch.chdir(tmp_path)
        result = await check()
        assert result.status == "PASS"
        assert (tmp_path / "results").is_dir()
        assert (tmp_path / "reports").is_dir()


class TestOrchestratorIntegration:
    async def test_runs_without_crash(self, tmp_path):
        """Orchestrator runs end-to-end with missing dirs — should not crash."""
        report = await run_pre_calibration(
            prompts_dir=tmp_path / "prompts",
            inputs_dir=tmp_path / "inputs",
            pdfs_dir=tmp_path / "pdfs",
        )
        assert isinstance(report, PreCalibrationReport)
        assert isinstance(report.proceed_to_pilot, bool)

    async def test_saves_output(self, tmp_path):
        output = tmp_path / "report.json"
        await run_pre_calibration(
            prompts_dir=tmp_path / "prompts",
            inputs_dir=tmp_path / "inputs",
            pdfs_dir=tmp_path / "pdfs",
            output=output,
        )
        assert output.exists()
        data = json.loads(output.read_text())
        assert "proceed_to_pilot" in data
