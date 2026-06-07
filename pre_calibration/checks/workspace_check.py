"""Verify workspace directories exist and are writable."""

from __future__ import annotations

from pathlib import Path

from pre_calibration.pre_calibration import CheckResult

REQUIRED_DIRS = ["results", "reports"]


async def check() -> CheckResult:
    """Verify output directories exist or can be created, and are writable."""
    details: dict = {}
    issues = []

    for dir_name in REQUIRED_DIRS:
        d = Path(dir_name)
        try:
            d.mkdir(parents=True, exist_ok=True)
            # Test write permission
            test_file = d / ".pre_calibration_test"
            test_file.write_text("test")
            test_file.unlink()
            details[dir_name] = "writable"
        except Exception as e:
            issues.append(f"{dir_name}: {e!s}")
            details[dir_name] = f"error: {e!s}"

    status = "FAIL" if issues else "PASS"
    return CheckResult(
        name="workspace",
        status=status,
        details=details,
        blocking=True,
    )
