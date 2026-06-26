"""Generate a Jinja2-rendered markdown narrative report from backtest results."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import jinja2

from pretia.validation.scoring import ComparisonScore
from pretia.validation.suite import FailureAttribution, attribute_failure

logger = logging.getLogger(__name__)

# Detectors expected per workflow (mirrored from colors.py)
_DETECTORS = [
    "context_growth",
    "loop_count_variance",
    "high_token_variance",
    "step_count_variance",
    "bimodality",
]


def _load_workflow_scores(
    results_dir: Path,
) -> dict[str, dict[str, ComparisonScore | None]]:
    """Load ComparisonScores for each workflow from per-workflow JSON files."""
    scores: dict[str, dict[str, ComparisonScore | None]] = {}
    if not results_dir.is_dir():
        return scores
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Skipping %s: %s", f, exc)
            continue
        wf = data.get("workflow_name")
        if not wf:
            continue
        comps = data.get("comparisons", {})
        wf_scores: dict[str, ComparisonScore | None] = {}
        for key in ("A", "B", "C"):
            comp_data = comps.get(key)
            if comp_data is None:
                wf_scores[key] = None
                continue
            score_dict = comp_data.get("score") if isinstance(comp_data, dict) else None
            if score_dict is None:
                wf_scores[key] = None
            else:
                wf_scores[key] = ComparisonScore.from_dict(score_dict)
        scores[wf] = wf_scores
    return scores


def _load_detected_patterns(results_dir: Path) -> dict[str, list[str]]:
    """Load detected patterns for each workflow."""
    patterns: dict[str, list[str]] = {}
    if not results_dir.is_dir():
        return patterns
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        wf = data.get("workflow_name")
        if wf:
            raw = data.get("detected_patterns", [])
            patterns[wf] = [
                p["pattern_type"] if isinstance(p, dict) else p for p in raw
            ]
    return patterns


def _load_step_costs(results_dir: Path) -> dict[str, dict[str, float]]:
    """Load step costs for each workflow."""
    step_costs: dict[str, dict[str, float]] = {}
    if not results_dir.is_dir():
        return step_costs
    for f in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        wf = data.get("workflow_name")
        if wf:
            step_costs[wf] = data.get("step_costs", {})
    return step_costs


def _compute_attributions(
    scores: dict[str, dict[str, ComparisonScore | None]],
) -> dict[str, FailureAttribution | None]:
    """Run failure attribution for each workflow."""
    attributions: dict[str, FailureAttribution | None] = {}
    for wf, wf_scores in scores.items():
        attr = attribute_failure(
            wf,
            wf_scores.get("A"),
            wf_scores.get("B"),
            wf_scores.get("C"),
        )
        attributions[wf] = attr
    return attributions


def _compute_drift_analysis(
    scores: dict[str, dict[str, ComparisonScore | None]],
) -> dict[str, Any]:
    """Analyze drift patterns across workflows."""
    total = len(scores)
    b_failed = []
    b_passed = []
    uniform_degradation = True
    degradation_amounts: list[float] = []

    for wf, wf_scores in scores.items():
        sa = wf_scores.get("A")
        sb = wf_scores.get("B")
        if sa is None or sb is None:
            continue
        if not sb.passes:
            b_failed.append(wf)
            drift = sb.mean_error_pct - sa.mean_error_pct
            degradation_amounts.append(drift)
        else:
            b_passed.append(wf)

    # Check if degradation is uniform (all within 2x of median)
    if degradation_amounts:
        sorted_amounts = sorted(degradation_amounts)
        median = sorted_amounts[len(sorted_amounts) // 2]
        if median > 0:
            for d in degradation_amounts:
                if d / median > 2.0 or d / median < 0.5:
                    uniform_degradation = False
                    break
        else:
            uniform_degradation = True
    else:
        uniform_degradation = True

    # Check if failures are concentrated in one group
    # Import locally to avoid circular deps at module level
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from visualization.colors import WORKFLOW_GROUPS

    group_failures: dict[str, int] = {}
    for wf in b_failed:
        group = WORKFLOW_GROUPS.get(wf, "unknown")
        group_failures[group] = group_failures.get(group, 0) + 1

    concentrated_group = None
    if group_failures:
        max_group = max(group_failures, key=group_failures.get)  # type: ignore[arg-type]
        if group_failures[max_group] == len(b_failed) and len(b_failed) > 1:
            concentrated_group = max_group

    diagnosis = "no_drift"
    if b_failed:
        if uniform_degradation and len(b_failed) >= total * 0.5:
            diagnosis = "tier_weight_shift"
        elif concentrated_group == "loops":
            diagnosis = "style_shift_loops"
        elif concentrated_group:
            diagnosis = f"style_shift_{concentrated_group}"
        else:
            diagnosis = "mixed_drift"

    # Pre-zip for template use (Jinja2 doesn't have zip)
    degradation_details = [
        {"workflow": wf, "amount": amt}
        for wf, amt in zip(b_failed, degradation_amounts)
    ]

    return {
        "total": total,
        "b_failed_count": len(b_failed),
        "b_failed": b_failed,
        "b_passed": b_passed,
        "uniform_degradation": uniform_degradation,
        "concentrated_group": concentrated_group,
        "diagnosis": diagnosis,
        "degradation_amounts": degradation_amounts,
        "degradation_details": degradation_details,
    }


def _compute_detector_assessment(
    results_dir: Path,
) -> dict[str, Any]:
    """Compute TP/FN rates from detected patterns vs expected."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from visualization.colors import EXPECTED_DETECTORS, classify_detector_result

    detected_patterns = _load_detected_patterns(results_dir)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    per_detector: dict[str, dict[str, int]] = {d: {"TP": 0, "TN": 0, "FP": 0, "FN": 0} for d in _DETECTORS}
    per_workflow: dict[str, list[dict[str, str]]] = {}

    for wf in sorted(EXPECTED_DETECTORS.keys()):
        detected = set(detected_patterns.get(wf, []))
        wf_results: list[dict[str, str]] = []
        for det in _DETECTORS:
            fired = det in detected
            classification = classify_detector_result(wf, det, fired)
            if classification == "TP":
                tp += 1
            elif classification == "TN":
                tn += 1
            elif classification == "FP":
                fp += 1
            else:
                fn += 1
            per_detector[det][classification] += 1
            wf_results.append({"detector": det, "classification": classification})
        per_workflow[wf] = wf_results

    total_expected_positive = tp + fn
    total_expected_negative = tn + fp
    tp_rate = tp / total_expected_positive if total_expected_positive > 0 else 1.0
    fn_rate = fn / total_expected_positive if total_expected_positive > 0 else 0.0
    fp_rate = fp / total_expected_negative if total_expected_negative > 0 else 0.0

    # Per-detector rates
    detector_rates: dict[str, dict[str, float]] = {}
    for det in _DETECTORS:
        counts = per_detector[det]
        pos = counts["TP"] + counts["FN"]
        neg = counts["TN"] + counts["FP"]
        detector_rates[det] = {
            "tp_rate": counts["TP"] / pos if pos > 0 else 1.0,
            "fn_rate": counts["FN"] / pos if pos > 0 else 0.0,
            "fp_rate": counts["FP"] / neg if neg > 0 else 0.0,
        }

    # Find any false positives
    false_positives: list[dict[str, str]] = []
    for wf, wf_results in per_workflow.items():
        for r in wf_results:
            if r["classification"] == "FP":
                false_positives.append({"workflow": wf, "detector": r["detector"]})

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp_rate": tp_rate,
        "fn_rate": fn_rate,
        "fp_rate": fp_rate,
        "per_detector": per_detector,
        "detector_rates": detector_rates,
        "false_positives": false_positives,
        "per_workflow": per_workflow,
    }


def generate_narrative(results_dir: Path, output: Path) -> Path:
    """Load results, run attribution, render to markdown."""
    scores = _load_workflow_scores(results_dir)
    attributions = _compute_attributions(scores)
    drift = _compute_drift_analysis(scores)
    detector = _compute_detector_assessment(results_dir)
    step_costs = _load_step_costs(results_dir)

    # Summary counts
    total = len(scores)
    passing_workflows = sorted(wf for wf, a in attributions.items() if a is None)
    passing_count = len(passing_workflows)
    bucket_1 = [wf for wf, a in attributions.items() if a is not None and a.bucket == 1]
    bucket_2 = [wf for wf, a in attributions.items() if a is not None and a.bucket == 2]
    bucket_3 = [wf for wf, a in attributions.items() if a is not None and a.bucket == 3]
    reweight_count = len(bucket_2)
    unresolved_count = len(bucket_1) + len(bucket_3)

    # Failed workflows with their attributions
    failed_workflows: list[dict[str, Any]] = []
    for wf in sorted(attributions.keys()):
        attr = attributions[wf]
        if attr is not None:
            wf_scores = scores.get(wf, {})
            failed_workflows.append({
                "workflow_name": wf,
                "bucket": attr.bucket,
                "bucket_label": attr.bucket_label,
                "explanation": attr.explanation,
                "recommended_action": attr.recommended_action,
                "score_a": wf_scores.get("A"),
                "score_b": wf_scores.get("B"),
                "score_c": wf_scores.get("C"),
                "step_costs": step_costs.get(wf, {}),
            })

    # Recommendations
    recommend_traffic_mix = reweight_count > 0
    needs_reprofile = [fw for fw in failed_workflows if fw["bucket"] == 3]
    needs_engine_fix = [fw for fw in failed_workflows if fw["bucket"] == 1]

    template_dir = Path(__file__).resolve().parent / "templates"
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_dir)),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,
    )
    template = env.get_template("full_report.md.j2")

    rendered = template.render(
        total=total,
        passing_count=passing_count,
        passing_workflows=passing_workflows,
        reweight_count=reweight_count,
        unresolved_count=unresolved_count,
        failed_workflows=failed_workflows,
        bucket_1=bucket_1,
        bucket_2=bucket_2,
        bucket_3=bucket_3,
        drift=drift,
        detector=detector,
        recommend_traffic_mix=recommend_traffic_mix,
        needs_reprofile=needs_reprofile,
        needs_engine_fix=needs_engine_fix,
        scores=scores,
        step_costs=step_costs,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)
    logger.info("Narrative report written to %s", output)
    return output


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate Pretia narrative report")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing per-workflow JSON result files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/narrative.md"),
        help="Output markdown file path",
    )
    args = parser.parse_args()

    result = generate_narrative(args.results_dir, args.output)
    print(f"Narrative report generated: {result}")  # noqa: T201


if __name__ == "__main__":
    main()
