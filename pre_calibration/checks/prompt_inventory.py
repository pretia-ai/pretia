"""Verify prompt files exist and are non-empty."""

from __future__ import annotations

from pathlib import Path

from pre_calibration.pre_calibration import CheckResult

EXPECTED_WORKFLOW_DIRS = [
    "w01",
    "w02",
    "w04",
    "w05",
    "w09",
    "w11",
    "w12",
    "w13",
    "w14",
    "w15",
    "w16",
    "w17",
    "w18",
    "w19",
]


async def check(*, prompts_dir: Path = Path("prompts")) -> CheckResult:
    """Verify prompt directory structure and non-empty files."""
    details: dict = {}
    missing = []
    empty = []

    if not prompts_dir.is_dir():
        return CheckResult(
            name="prompt_inventory",
            status="FAIL",
            details={"error": f"Prompts directory not found: {prompts_dir}"},
            blocking=True,
        )

    found_dirs = [d.name for d in prompts_dir.iterdir() if d.is_dir()]
    prompt_count = 0

    for wf_prefix in EXPECTED_WORKFLOW_DIRS:
        matching = [d for d in found_dirs if d.startswith(wf_prefix)]
        if not matching:
            missing.append(wf_prefix)
            continue
        for wf_dir_name in matching:
            wf_dir = prompts_dir / wf_dir_name
            for f in wf_dir.glob("*.txt"):
                if f.stat().st_size == 0:
                    empty.append(str(f))
                else:
                    prompt_count += 1
            for f in wf_dir.glob("*.md"):
                if f.stat().st_size == 0:
                    empty.append(str(f))
                else:
                    prompt_count += 1

    details["prompt_count"] = prompt_count
    if missing:
        details["missing_workflows"] = missing
    if empty:
        details["empty_files"] = empty

    if missing:
        status = "FAIL"
    elif empty:
        status = "WARN"
    else:
        status = "PASS"

    return CheckResult(
        name="prompt_inventory",
        status=status,
        details=details,
        blocking=True,
    )
