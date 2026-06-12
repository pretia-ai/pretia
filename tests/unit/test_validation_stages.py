"""Integration tests for validate.py stages with mocked dependencies."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from scripts._validation_types import CheckStatus, StageResult


class TestStage1DataReadiness:
    def test_missing_manifest_fails(self, tmp_path: Path) -> None:
        from scripts.validate import run_stage_1

        result = run_stage_1(
            prompts_dir=tmp_path / "prompts",
            inputs_prof=tmp_path / "inputs_prof",
            inputs_gt=tmp_path / "inputs_gt",
            pdfs_prof=tmp_path / "pdfs_prof",
            pdfs_gt=tmp_path / "pdfs_gt",
        )
        assert isinstance(result, StageResult)
        assert result.stage == 1
        assert result.passed is False
        assert any(c.name == "prompt_manifest" for c in result.checks)

    def test_valid_prompts_pass(self, tmp_path: Path) -> None:
        from scripts.validate import run_stage_1

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        manifest = {"prompts": []}
        (prompts / "manifest.json").write_text(json.dumps(manifest))

        result = run_stage_1(
            prompts_dir=prompts,
            inputs_prof=tmp_path / "inputs_prof",
            inputs_gt=tmp_path / "inputs_gt",
            pdfs_prof=tmp_path / "pdfs_prof",
            pdfs_gt=tmp_path / "pdfs_gt",
        )
        assert isinstance(result, StageResult)
        assert result.stage == 1
        prompt_checks = [c for c in result.checks if c.name == "prompt_file_existence"]
        assert prompt_checks
        assert prompt_checks[0].status == CheckStatus.PASS

    def test_missing_inputs_detected(self, tmp_path: Path) -> None:
        from scripts.validate import run_stage_1

        prompts = tmp_path / "prompts"
        prompts.mkdir()
        (prompts / "manifest.json").write_text(json.dumps({"prompts": []}))

        result = run_stage_1(
            prompts_dir=prompts,
            inputs_prof=tmp_path / "nonexistent",
            inputs_gt=tmp_path / "nonexistent",
            pdfs_prof=tmp_path / "pdfs_prof",
            pdfs_gt=tmp_path / "pdfs_gt",
        )
        input_checks = [c for c in result.checks if c.name == "input_file_counts"]
        assert input_checks
        assert input_checks[0].status == CheckStatus.FAIL


class TestStage2Infrastructure:
    def test_missing_api_key_fails(self) -> None:
        from scripts.validate import run_stage_2

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("scripts.validate._load_dotenv"),
        ):
            result = run_stage_2()

        assert isinstance(result, StageResult)
        assert result.stage == 2
        key_checks = [c for c in result.checks if c.name == "api_keys"]
        assert key_checks
        assert key_checks[0].status == CheckStatus.FAIL


class TestStage3Calibration:
    def test_delegates_to_synthetic_runner(self) -> None:
        from scripts.validate import run_stage_3

        mock_workflows = [MagicMock(name="w1"), MagicMock(name="w2")]
        mock_results = [MagicMock(), MagicMock()]
        mock_report = MagicMock()
        mock_report.p50_calibration_pct = 90.0
        mock_report.p95_coverage_pct = 80.0

        with (
            patch(
                "scripts.validate.generate_all_synthetic_workflows",
                return_value=mock_workflows,
                create=True,
            ),
            patch(
                "tests.synthetic.generators.generate_all_synthetic_workflows",
                return_value=mock_workflows,
            ),
            patch("tests.synthetic.runner.run_synthetic_calibration", return_value=mock_results),
            patch(
                "tests.synthetic.calibration.compute_calibration_report", return_value=mock_report
            ),
        ):
            result = run_stage_3()

        assert isinstance(result, StageResult)
        assert result.stage == 3
        cal_checks = [c for c in result.checks if c.name == "synthetic_calibration"]
        assert cal_checks
        assert cal_checks[0].status == CheckStatus.PASS

    def test_failing_calibration(self) -> None:
        from scripts.validate import run_stage_3

        mock_report = MagicMock()
        mock_report.p50_calibration_pct = 50.0
        mock_report.p95_coverage_pct = 40.0

        with (
            patch("tests.synthetic.generators.generate_all_synthetic_workflows", return_value=[]),
            patch("tests.synthetic.runner.run_synthetic_calibration", return_value=[]),
            patch(
                "tests.synthetic.calibration.compute_calibration_report", return_value=mock_report
            ),
        ):
            result = run_stage_3()

        cal_checks = [c for c in result.checks if c.name == "synthetic_calibration"]
        assert cal_checks
        assert cal_checks[0].status == CheckStatus.FAIL

    def test_import_error_warns(self) -> None:
        from scripts.validate import run_stage_3

        with patch.dict(
            "sys.modules",
            {
                "tests.synthetic.generators": None,
            },
        ):
            with patch("builtins.__import__", side_effect=ImportError("no module")):
                result = run_stage_3()

        cal_checks = [c for c in result.checks if c.name == "synthetic_calibration"]
        assert cal_checks
        assert cal_checks[0].status in (CheckStatus.WARN, CheckStatus.FAIL)


class TestStage4LiveSmoke:
    def test_import_failure_returns_fail(self) -> None:
        from scripts.validate import run_stage_4

        with patch("builtins.__import__", side_effect=ImportError("no bt_agents")):
            result = run_stage_4()

        assert isinstance(result, StageResult)
        assert result.stage == 4
        assert any(c.status == CheckStatus.FAIL for c in result.checks)
