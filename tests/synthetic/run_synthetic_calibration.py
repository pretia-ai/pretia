#!/usr/bin/env python3
"""Run synthetic distribution calibration. Zero API cost."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tests.synthetic.calibration import compute_calibration_report, format_report
from tests.synthetic.generators import generate_all_synthetic_workflows
from tests.synthetic.runner import run_synthetic_calibration


def main() -> None:
    print("Generating synthetic workflows...")
    workflows = generate_all_synthetic_workflows(seed_base=42)
    print(f"Generated {len(workflows)} synthetic workflows")

    swebench_workflows = []
    try:
        from tests.synthetic.swebench.download import download_swebench_data
        from tests.synthetic.swebench.grouper import group_by_repo
        from tests.synthetic.swebench.integrate import repo_groups_to_synthetic_workflows
        from tests.synthetic.swebench.parser import parse_swebench_data

        data_path = download_swebench_data()
        instances = parse_swebench_data(data_path)
        groups = group_by_repo(instances)
        swebench_workflows = repo_groups_to_synthetic_workflows(groups)
        print(f"Added {len(swebench_workflows)} SWE-bench workflows from {len(groups)} repos")
    except FileNotFoundError as e:
        print(f"SWE-bench data not available: {e}")
        print("Continuing with synthetic distributions only.")
    except Exception as e:
        print(f"SWE-bench parsing failed: {e}")
        print("Continuing with synthetic distributions only.")

    all_workflows = workflows + swebench_workflows

    print(f"Running projection engine on {len(all_workflows)} workflows...")
    results = run_synthetic_calibration(all_workflows)

    print("Computing calibration metrics...")
    report = compute_calibration_report(results)

    text = format_report(report)
    print()
    print(text)

    report_path = Path(__file__).parent / "calibration_report.md"
    report_path.write_text(text)
    print(f"\nReport saved to {report_path}")

    if report.p50_calibration_rate < 0.85:
        print(f"\nFAIL: p50 calibration {report.p50_calibration_rate:.0%} < 85% target")
        sys.exit(1)
    if report.p95_coverage_rate < 0.70:
        print(f"\nFAIL: p95 coverage {report.p95_coverage_rate:.0%} < 70% target")
        sys.exit(1)

    print("\nPASS: Synthetic calibration targets met.")


if __name__ == "__main__":
    main()
