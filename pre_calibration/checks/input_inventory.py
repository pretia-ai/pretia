"""Verify input files exist with correct counts."""

from __future__ import annotations

from pathlib import Path

from pre_calibration.pre_calibration import CheckResult

EXPECTED_PROFILING_COUNT = 50

EXPECTED_GT_COUNTS = {
    "w01": 200,
    "w02": 500,
    "w04": 500,
    "w05": 220,
    "w09": 200,
    "w11": 200,
    "w12": 200,
    "w13": 300,
    "w14": 300,
    "w15": 500,
    "w16": 300,
    "w17": 500,
    "w18": 500,
    "w19": 500,
}


async def check(*, inputs_dir: Path = Path("inputs/generated")) -> CheckResult:
    """Verify profiling and ground truth input sets exist with correct counts."""
    details: dict = {}
    issues = []

    if not inputs_dir.is_dir():
        return CheckResult(
            name="input_inventory",
            status="FAIL",
            details={"error": f"Inputs directory not found: {inputs_dir}"},
            blocking=True,
        )

    profiling_total = 0
    gt_total = 0

    for wf_id, expected_gt in EXPECTED_GT_COUNTS.items():
        # Try directory-of-JSONs layout first (inputs/generated/profiling/w01/*.json)
        wf_short = wf_id.replace("w0", "w")  # w01 -> w1 for alt naming
        prof_dir = inputs_dir / "profiling" / wf_id
        if not prof_dir.is_dir():
            prof_dir = inputs_dir / "profiling" / wf_short
        gt_dir = inputs_dir / "ground_truth" / wf_id
        if not gt_dir.is_dir():
            gt_dir = inputs_dir / "ground_truth" / wf_short

        if prof_dir.is_dir():
            count = len(list(prof_dir.glob("*.json")))
            profiling_total += count
            if count < EXPECTED_PROFILING_COUNT:
                issues.append(
                    f"{wf_id}: profiling has {count}, expected {EXPECTED_PROFILING_COUNT}"
                )
        else:
            # Fallback: JSONL file at root
            prof_file = inputs_dir / f"{wf_id}_profiling.jsonl"
            alt = inputs_dir / f"{wf_id}_realistic.jsonl"
            alt2 = inputs_dir / f"{wf_id}_inputs.jsonl"
            if prof_file.exists():
                count = sum(1 for _ in prof_file.open())
                profiling_total += count
            elif alt.exists():
                count = sum(1 for _ in alt.open())
                profiling_total += count
            elif alt2.exists():
                count = sum(1 for _ in alt2.open())
                profiling_total += count
            else:
                issues.append(f"{wf_id}: no profiling input file found")

        if gt_dir.is_dir():
            count = len(list(gt_dir.glob("*.json")))
            gt_total += count
            if count < expected_gt:
                issues.append(
                    f"{wf_id}: ground truth has {count}, expected {expected_gt}"
                )
        else:
            gt_file = inputs_dir / f"{wf_id}_ground_truth.jsonl"
            if gt_file.exists():
                count = sum(1 for _ in gt_file.open())
                gt_total += count
                if count < expected_gt:
                    issues.append(
                        f"{wf_id}: ground truth has {count}, expected {expected_gt}"
                    )
            else:
                issues.append(f"{wf_id}: no ground truth file found")

    details["profiling_total"] = profiling_total
    details["ground_truth_total"] = gt_total
    if issues:
        details["issues"] = issues

    status = "FAIL" if issues else "PASS"
    return CheckResult(
        name="input_inventory",
        status=status,
        details=details,
        blocking=True,
    )
