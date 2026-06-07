"""Acceptance criteria checks for generated input sets.

Validate tier distribution, token entropy, range coverage, duplicate
detection, dirty input injection, cross-set drift, and workflow-specific
structural properties.  Importable as a library and runnable as a CLI.

Usage (CLI):
    python -m inputs.validation.verify_inputs \
        --input-dir inputs/generated/profiling/w01/ --profile profiling --workflow W1

    python -m inputs.validation.verify_inputs \
        --profiling-dir inputs/generated/profiling/w01/ \
        --gt-dir inputs/generated/ground_truth/w01/ --workflow W1
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Default tier ranges (generic) — overridden per-workflow via --tier-ranges
# ---------------------------------------------------------------------------
_DEFAULT_TIER_RANGES: dict[str, tuple[int, int]] = {
    "easy": (10, 70),
    "medium": (40, 180),
    "hard": (100, 600),
    "edge": (5, 500),
    "extreme": (200, 1200),
}

# ---------------------------------------------------------------------------
# Default expected weights
# ---------------------------------------------------------------------------
_PROFILING_WEIGHTS: dict[str, float] = {
    "easy": 0.40,
    "medium": 0.35,
    "hard": 0.20,
    "edge": 0.05,
}

_GROUND_TRUTH_WEIGHTS: dict[str, float] = {
    "easy": 0.55,
    "medium": 0.25,
    "hard": 0.12,
    "edge": 0.05,
    "extreme": 0.03,
}

_WORKFLOW_PROFILING_WEIGHTS: dict[str, dict[str, float]] = {
    "W13": {"easy": 0.70, "medium": 0.20, "hard": 0.10},
    "W17": {"easy": 0.15, "medium": 0.40, "hard": 0.30, "edge": 0.15},
}

_WORKFLOW_GT_WEIGHTS: dict[str, dict[str, float]] = {
    "W13": {"easy": 0.55, "medium": 0.25, "hard": 0.15, "edge": 0.05},
    "W17": {"easy": 0.20, "medium": 0.35, "hard": 0.25, "edge": 0.15, "extreme": 0.05},
}

_WORKFLOW_TIER_RANGES: dict[str, dict[str, tuple[int, int]]] = {
    "W01": {"easy": (10, 70), "medium": (40, 180), "hard": (100, 600), "edge": (5, 500), "extreme": (500, 1200)},
    "W02": {"easy": (40, 180), "medium": (80, 350), "hard": (150, 600), "edge": (20, 800), "extreme": (400, 1000)},
    "W4": {"easy": (500, 2500), "medium": (1500, 7000), "hard": (4000, 15000), "edge": (100, 15000), "extreme": (12000, 25000)},
    "W5": {"easy": (300, 1200), "medium": (800, 3500), "hard": (1500, 8000), "edge": (100, 8000), "extreme": (5000, 12000)},
    "W09": {"easy": (100, 400), "medium": (100, 350), "hard": (60, 250), "edge": (30, 300), "extreme": (300, 600)},
    "W11": {"easy": (10, 70), "medium": (40, 180), "hard": (100, 600), "edge": (5, 500), "extreme": (500, 1200)},
    "W12": {"easy": (200, 1000), "medium": (600, 3500), "hard": (2000, 8000), "edge": (100, 8000), "extreme": (6000, 15000)},
    "W13": {"easy": (10, 50), "medium": (30, 140), "hard": (40, 200), "edge": (10, 200)},
    "W14": {"easy": (15, 60), "medium": (30, 130), "hard": (40, 200), "edge": (5, 600), "extreme": (80, 300)},
    "W15": {"easy": (15, 60), "medium": (30, 130), "hard": (40, 200), "edge": (20, 200), "extreme": (80, 300)},
    "W16": {"easy": (1000, 15000), "medium": (5000, 40000), "hard": (15000, 80000), "edge": (500, 100000), "extreme": (50000, 100000)},
    "W17": {"easy": (50, 200), "medium": (80, 300), "hard": (80, 300), "edge": (80, 300), "extreme": (100, 400)},
    "W18": {"easy": (30000, 50000), "medium": (40000, 75000), "hard": (60000, 95000), "edge": (80000, 100000), "extreme": (90000, 100000)},
    "W19": {"easy": (50, 300), "medium": (100, 500), "hard": (200, 1000), "edge": (50, 1000), "extreme": (500, 2000)},
}


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_inputs(input_dir: str) -> list[dict[str, Any]]:
    """Load all .json files from *input_dir* and return parsed dicts."""
    dirpath = Path(input_dir)
    if not dirpath.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    inputs: list[dict[str, Any]] = []
    for filepath in sorted(dirpath.glob("*.json")):
        with filepath.open() as fh:
            inputs.append(json.load(fh))
    return inputs


def extract_text(input_data: dict[str, Any]) -> str:
    """Extract the primary text content from an input_data dict.

    Handle common field names across workflow generators:
    customer_message, document_text, user_query, query, content,
    lead_profile (serialized as JSON string), and turns (joined).
    """
    for key in ("customer_message", "document_text", "user_query", "query", "content", "input"):
        if key in input_data and isinstance(input_data[key], str):
            return input_data[key]

    # lead_profile — serialize nested dict to a string
    if "lead_profile" in input_data:
        val = input_data["lead_profile"]
        if isinstance(val, str):
            return val
        return json.dumps(val, indent=2)

    # turns — join list of strings
    if "turns" in input_data:
        turns = input_data["turns"]
        if isinstance(turns, list):
            return " ".join(str(t) for t in turns)

    # Fallback: serialize the entire input_data
    return json.dumps(input_data, indent=2)


def has_style_artifact(text: str) -> bool:
    """Return True if *text* exhibits at least one style artifact.

    Artifacts checked: all-lowercase, doubled punctuation (``??``, ``..``),
    informal substitutions (``u `` for ``you ``, ``ur `` for ``your ``),
    and trailing whitespace.
    """
    if not text or not text.strip():
        return False

    # All lowercase (only meaningful if the text contains letters)
    alpha_chars = [c for c in text if c.isalpha()]
    if alpha_chars and all(c.islower() for c in alpha_chars):
        return True

    if "??" in text or ".." in text:
        return True

    # Informal substitutions — word-boundary aware
    if re.search(r"\bu\b", text, re.IGNORECASE) and "you" not in text.lower():
        # "u " as a standalone word (not part of another word)
        pass  # handled below more carefully
    if re.search(r"(?<!\w)u(?:\s|$)", text):
        return True
    if re.search(r"(?<!\w)ur(?:\s|$)", text):
        return True

    # Trailing whitespace (beyond a single space)
    if text != text.rstrip() and text.endswith("  "):
        return True

    return False


# ═══════════════════════════════════════════════════════════════════════════
# Per-set checks
# ═══════════════════════════════════════════════════════════════════════════

def check_tier_distribution(
    inputs: list[dict[str, Any]],
    expected_weights: dict[str, float],
    tolerance: int = 2,
) -> tuple[bool, str]:
    """Verify tier counts match *expected_weights* within +/-*tolerance*."""
    n = len(inputs)
    tier_counts = Counter(inp.get("tier", "unknown") for inp in inputs)

    failures: list[str] = []
    for tier, weight in expected_weights.items():
        expected = round(n * weight)
        actual = tier_counts.get(tier, 0)
        if abs(actual - expected) > tolerance:
            failures.append(
                f"  {tier}: expected ~{expected} (±{tolerance}), got {actual}"
            )

    if failures:
        detail = "\n".join(failures)
        return False, f"Tier distribution mismatch:\n{detail}"
    return True, f"Tier distribution OK (n={n}, tiers={dict(tier_counts)})"


def check_token_entropy(
    inputs: list[dict[str, Any]],
    threshold: float = 0.6,
) -> tuple[bool, str]:
    """Verify Shannon entropy of binned token counts exceeds threshold * log(10)."""
    token_counts = [inp.get("token_count", 0) for inp in inputs]
    if not token_counts:
        return False, "No inputs to check token entropy."

    lo, hi = min(token_counts), max(token_counts)
    if lo == hi:
        return False, f"All token counts identical ({lo}); entropy = 0."

    n_bins = 10
    bin_width = (hi - lo) / n_bins
    bins: list[int] = [0] * n_bins
    for tc in token_counts:
        idx = min(int((tc - lo) / bin_width), n_bins - 1)
        bins[idx] += 1

    total = sum(bins)
    entropy = 0.0
    for count in bins:
        if count > 0:
            p = count / total
            entropy -= p * math.log(p)

    max_entropy = math.log(n_bins)
    ratio = entropy / max_entropy if max_entropy > 0 else 0.0

    if ratio < threshold:
        return False, (
            f"Token entropy too low: {ratio:.3f} < {threshold} "
            f"(entropy={entropy:.3f}, max={max_entropy:.3f})"
        )
    return True, f"Token entropy OK: {ratio:.3f} >= {threshold}"


def check_range_coverage(
    inputs: list[dict[str, Any]],
    tier_ranges: dict[str, tuple[int, int]],
    min_coverage: float = 0.30,
) -> tuple[bool, str]:
    """Verify token counts span >= *min_coverage* of each tier's specified range."""
    by_tier: dict[str, list[int]] = {}
    for inp in inputs:
        tier = inp.get("tier", "unknown")
        by_tier.setdefault(tier, []).append(inp.get("token_count", 0))

    failures: list[str] = []
    for tier, (range_lo, range_hi) in tier_ranges.items():
        tokens = by_tier.get(tier)
        if not tokens or len(tokens) < 5:
            continue
        actual_lo, actual_hi = min(tokens), max(tokens)
        span = range_hi - range_lo
        if span <= 0:
            continue
        actual_span = actual_hi - actual_lo
        coverage = actual_span / span
        if coverage < min_coverage:
            failures.append(
                f"  {tier}: coverage {coverage:.2f} < {min_coverage} "
                f"(actual {actual_lo}-{actual_hi}, spec {range_lo}-{range_hi})"
            )

    if failures:
        detail = "\n".join(failures)
        return False, f"Range coverage insufficient:\n{detail}"
    return True, "Range coverage OK for all tiers."


def check_no_duplicates(
    inputs: list[dict[str, Any]],
    max_similarity: float = 0.85,
) -> tuple[bool, str]:
    """Flag pairwise TF-IDF cosine similarity > *max_similarity* within each tier.

    Requires sklearn; skips with a warning if unavailable.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        return True, "SKIP: sklearn not available; duplicate check skipped."

    by_tier: dict[str, list[str]] = {}
    for inp in inputs:
        tier = inp.get("tier", "unknown")
        text = extract_text(inp.get("input_data", {}))
        by_tier.setdefault(tier, []).append(text)

    violations: list[str] = []
    for tier, texts in by_tier.items():
        if len(texts) < 2:
            continue
        # Filter out empty/near-empty texts to avoid TF-IDF issues
        non_empty = [(i, t) for i, t in enumerate(texts) if len(t.strip()) > 5]
        if len(non_empty) < 2:
            continue

        indices, filtered_texts = zip(*non_empty)
        vectorizer = TfidfVectorizer()
        try:
            tfidf_matrix = vectorizer.fit_transform(filtered_texts)
        except ValueError:
            # All texts identical after tokenization
            violations.append(f"  {tier}: all texts appear identical after tokenization")
            continue

        sim_matrix = cosine_similarity(tfidf_matrix)
        n = sim_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i][j] > max_similarity:
                    violations.append(
                        f"  {tier}: pair ({indices[i]}, {indices[j]}) "
                        f"similarity={sim_matrix[i][j]:.3f} > {max_similarity}"
                    )

    if violations:
        detail = "\n".join(violations[:20])  # cap output
        extra = ""
        if len(violations) > 20:
            extra = f"\n  ... and {len(violations) - 20} more violations"
        return False, f"Duplicate/near-duplicate inputs detected:\n{detail}{extra}"
    return True, "No duplicate inputs detected."


def check_dirty_count(
    inputs: list[dict[str, Any]],
    profile: str,
    min_pct: float = 0.03,
    max_pct: float = 0.07,
) -> tuple[bool, str]:
    """Verify ~5% of inputs are dirty, within [min_pct, max_pct] bounds.

    For profiling (n=50): expect 2-3.  For GT (n=500): expect 20-30.
    """
    n = len(inputs)
    dirty_count = sum(1 for inp in inputs if inp.get("is_dirty", False))

    min_expected = max(1, math.floor(n * min_pct))
    max_expected = math.ceil(n * max_pct)

    if dirty_count < min_expected or dirty_count > max_expected:
        return False, (
            f"Dirty count out of range: {dirty_count} "
            f"(expected {min_expected}-{max_expected} for {profile}, n={n})"
        )
    return True, f"Dirty count OK: {dirty_count} ({profile}, n={n})"


def check_dirty_distribution(inputs: list[dict[str, Any]]) -> tuple[bool, str]:
    """Verify dirty inputs appear in at least 2 different tiers."""
    dirty_tiers = set()
    for inp in inputs:
        if inp.get("is_dirty", False):
            dirty_tiers.add(inp.get("tier", "unknown"))

    if not dirty_tiers:
        return True, "No dirty inputs present (nothing to check)."

    if len(dirty_tiers) < 2:
        return False, (
            f"Dirty inputs concentrated in single tier: {dirty_tiers}. "
            f"Must appear in at least 2 tiers."
        )
    return True, f"Dirty distribution OK: spread across {sorted(dirty_tiers)}"


# ═══════════════════════════════════════════════════════════════════════════
# Cross-set checks
# ═══════════════════════════════════════════════════════════════════════════

def check_gt_tier_weights(
    gt_inputs: list[dict[str, Any]],
    expected_weights: dict[str, float],
    tolerance_pct: float = 2.0,
) -> tuple[bool, str]:
    """Verify GT tier distribution matches expected weights within +/-tolerance_pct."""
    n = len(gt_inputs)
    if n == 0:
        return False, "No GT inputs provided."

    tier_counts = Counter(inp.get("tier", "unknown") for inp in gt_inputs)
    failures: list[str] = []

    for tier, weight in expected_weights.items():
        expected_pct = weight * 100
        actual_count = tier_counts.get(tier, 0)
        actual_pct = (actual_count / n) * 100
        if abs(actual_pct - expected_pct) > tolerance_pct:
            failures.append(
                f"  {tier}: expected {expected_pct:.1f}% (±{tolerance_pct}%), "
                f"got {actual_pct:.1f}% ({actual_count}/{n})"
            )

    if failures:
        detail = "\n".join(failures)
        return False, f"GT tier weight mismatch:\n{detail}"
    return True, f"GT tier weights OK (n={n})"


def check_token_stretch(
    profiling_inputs: list[dict[str, Any]],
    gt_inputs: list[dict[str, Any]],
    min_ratio: float = 0.8,
    max_ratio: float = 3.0,
) -> tuple[bool, str]:
    """Verify GT mean token count is 0.8-3.0x profiling mean per tier."""
    def _mean_by_tier(inputs: list[dict[str, Any]]) -> dict[str, float]:
        by_tier: dict[str, list[int]] = {}
        for inp in inputs:
            tier = inp.get("tier", "unknown")
            by_tier.setdefault(tier, []).append(inp.get("token_count", 0))
        return {
            tier: sum(vals) / len(vals) for tier, vals in by_tier.items() if vals
        }

    prof_means = _mean_by_tier(profiling_inputs)
    gt_means = _mean_by_tier(gt_inputs)

    failures: list[str] = []
    for tier in sorted(set(prof_means) & set(gt_means)):
        if tier in ("edge", "extreme"):
            # Edge/extreme have unpredictable token ranges; skip
            continue
        prof_mean = prof_means[tier]
        gt_mean = gt_means[tier]
        if prof_mean == 0:
            continue
        ratio = gt_mean / prof_mean
        if ratio < min_ratio or ratio > max_ratio:
            failures.append(
                f"  {tier}: ratio {ratio:.2f} (prof={prof_mean:.0f}, gt={gt_mean:.0f}), "
                f"expected {min_ratio}-{max_ratio}"
            )

    if failures:
        detail = "\n".join(failures)
        return False, f"Token stretch out of range:\n{detail}"
    return True, "Token stretch OK for all tiers."


def check_style_artifacts(
    profiling_inputs: list[dict[str, Any]],
    gt_inputs: list[dict[str, Any]],
    gt_min: float = 0.60,
    prof_max: float = 0.15,
) -> tuple[bool, str]:
    """Verify style artifact rates: >=60% in GT, <=15% in profiling."""
    def _artifact_rate(inputs: list[dict[str, Any]]) -> float:
        if not inputs:
            return 0.0
        count = sum(
            1 for inp in inputs
            if has_style_artifact(extract_text(inp.get("input_data", {})))
        )
        return count / len(inputs)

    prof_rate = _artifact_rate(profiling_inputs)
    gt_rate = _artifact_rate(gt_inputs)

    failures: list[str] = []
    if gt_rate < gt_min:
        failures.append(
            f"  GT artifact rate too low: {gt_rate:.2%} < {gt_min:.0%}"
        )
    if prof_rate > prof_max:
        failures.append(
            f"  Profiling artifact rate too high: {prof_rate:.2%} > {prof_max:.0%}"
        )

    if failures:
        detail = "\n".join(failures)
        return False, f"Style artifact check failed:\n{detail}"
    return True, (
        f"Style artifacts OK: profiling={prof_rate:.2%} (<={prof_max:.0%}), "
        f"GT={gt_rate:.2%} (>={gt_min:.0%})"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Workflow-specific checks
# ═══════════════════════════════════════════════════════════════════════════

def check_w5_modality(
    inputs: list[dict[str, Any]],
    profile: str,
    text_target: float,
    image_target: float,
    tolerance: float = 0.08,
) -> tuple[bool, str]:
    """Verify W5 modality distribution (text vs image).

    Profiling: 70% text / 30% image (±8%).
    GT: 40% text / 60% image (±8%).
    The *image_target* encompasses both pure-image and mixed modality inputs.
    """
    n = len(inputs)
    if n == 0:
        return False, "No inputs to check modality."

    text_count = 0
    image_count = 0
    for inp in inputs:
        sd = inp.get("structural_descriptor", {})
        modality = sd.get("modality", inp.get("input_data", {}).get("modality", ""))
        if modality == "text":
            text_count += 1
        else:
            # "image" or "mixed" both count as non-text
            image_count += 1

    text_pct = text_count / n
    image_pct = image_count / n

    failures: list[str] = []
    if abs(text_pct - text_target) > tolerance:
        failures.append(
            f"  text: {text_pct:.2%} vs target {text_target:.0%} (±{tolerance:.0%})"
        )
    if abs(image_pct - image_target) > tolerance:
        failures.append(
            f"  image+mixed: {image_pct:.2%} vs target {image_target:.0%} (±{tolerance:.0%})"
        )

    if failures:
        detail = "\n".join(failures)
        return False, f"W5 modality mismatch ({profile}):\n{detail}"
    return True, (
        f"W5 modality OK ({profile}): text={text_pct:.2%}, "
        f"image+mixed={image_pct:.2%}"
    )


def check_w19_session_depth(
    profiling_inputs: list[dict[str, Any]],
    gt_inputs: list[dict[str, Any]],
) -> tuple[bool, str]:
    """Verify W19 session depth drift: profiling mean <= 5.5, GT mean >= 6.5."""
    def _mean_substantive(inputs: list[dict[str, Any]]) -> float:
        counts = [
            inp.get("structural_descriptor", {}).get("substantive_turn_count", 0)
            for inp in inputs
        ]
        return sum(counts) / len(counts) if counts else 0.0

    prof_mean = _mean_substantive(profiling_inputs)
    gt_mean = _mean_substantive(gt_inputs)

    failures: list[str] = []
    if prof_mean > 5.5:
        failures.append(
            f"  Profiling mean substantive turns {prof_mean:.2f} > 5.5"
        )
    if gt_mean < 6.5:
        failures.append(
            f"  GT mean substantive turns {gt_mean:.2f} < 6.5"
        )

    if failures:
        detail = "\n".join(failures)
        return False, f"W19 session depth drift check failed:\n{detail}"
    return True, (
        f"W19 session depth OK: profiling_mean={prof_mean:.2f} (<=5.5), "
        f"gt_mean={gt_mean:.2f} (>=6.5)"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════

def run_all_checks(
    inputs: list[dict[str, Any]],
    profile: str,
    workflow: str | None = None,
    tier_ranges: dict[str, tuple[int, int]] | None = None,
) -> list[tuple[str, bool, str]]:
    """Run all applicable per-set checks. Return list of (name, passed, message)."""
    if tier_ranges is None:
        wf_key = (workflow or "").upper()
        tier_ranges = _WORKFLOW_TIER_RANGES.get(wf_key, _DEFAULT_TIER_RANGES)

    wf_key = (workflow or "").upper()
    if profile == "profiling":
        expected_weights = _WORKFLOW_PROFILING_WEIGHTS.get(wf_key, _PROFILING_WEIGHTS)
    else:
        expected_weights = _WORKFLOW_GT_WEIGHTS.get(wf_key, _GROUND_TRUTH_WEIGHTS)

    results: list[tuple[str, bool, str]] = []

    # 1. Tier distribution
    passed, msg = check_tier_distribution(inputs, expected_weights)
    results.append(("tier_distribution", passed, msg))

    # 2. Token entropy
    passed, msg = check_token_entropy(inputs)
    results.append(("token_entropy", passed, msg))

    # 3. Range coverage
    passed, msg = check_range_coverage(inputs, tier_ranges)
    results.append(("range_coverage", passed, msg))

    # 4. No duplicates
    passed, msg = check_no_duplicates(inputs)
    results.append(("no_duplicates", passed, msg))

    # 5. Dirty count (skip for workflows with no dirty types)
    _NO_DIRTY = {"5", "16"}
    if wf_key.replace("W", "") not in _NO_DIRTY:
        passed, msg = check_dirty_count(inputs, profile)
        results.append(("dirty_count", passed, msg))

        # 6. Dirty distribution
        passed, msg = check_dirty_distribution(inputs)
        results.append(("dirty_distribution", passed, msg))

    # Workflow-specific per-set checks
    wf = (workflow or "").upper().replace("W", "")
    if wf == "5":
        if profile == "profiling":
            passed, msg = check_w5_modality(inputs, profile, 0.70, 0.30)
        else:
            passed, msg = check_w5_modality(inputs, profile, 0.40, 0.60)
        results.append(("w5_modality", passed, msg))

    return results


def run_cross_checks(
    profiling_inputs: list[dict[str, Any]],
    gt_inputs: list[dict[str, Any]],
    workflow: str | None = None,
) -> list[tuple[str, bool, str]]:
    """Run cross-set checks. Return list of (name, passed, message)."""
    results: list[tuple[str, bool, str]] = []
    wf_key = (workflow or "").upper()
    wf = wf_key.replace("W", "")

    # 7. GT tier weights
    gt_weights = _WORKFLOW_GT_WEIGHTS.get(wf_key, _GROUND_TRUTH_WEIGHTS)
    passed, msg = check_gt_tier_weights(gt_inputs, gt_weights)
    results.append(("gt_tier_weights", passed, msg))

    # 8. Token stretch
    passed, msg = check_token_stretch(profiling_inputs, gt_inputs)
    results.append(("token_stretch", passed, msg))

    # 9. Style artifacts (skip for structured data workflows — JSON/PDF, not text)
    _SKIP_STYLE = {"5", "9", "16", "17", "18"}
    if wf not in _SKIP_STYLE:
        passed, msg = check_style_artifacts(profiling_inputs, gt_inputs)
        results.append(("style_artifacts", passed, msg))

    # Workflow-specific cross-set checks
    if wf == "19":
        passed, msg = check_w19_session_depth(profiling_inputs, gt_inputs)
        results.append(("w19_session_depth", passed, msg))

    return results


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _print_results(results: list[tuple[str, bool, str]]) -> int:
    """Print results and return exit code (0 = all pass, 1 = any fail)."""
    any_fail = False
    for name, passed, msg in results:
        status = "PASS" if passed else "FAIL"
        if not passed:
            any_fail = True
        print(f"[{status}] {name}")
        # Indent multi-line messages
        for line in msg.split("\n"):
            print(f"       {line}")
        print()
    return 1 if any_fail else 0


def main() -> None:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate generated input sets against acceptance criteria.",
    )
    parser.add_argument(
        "--input-dir",
        help="Single directory to check (per-set checks only).",
    )
    parser.add_argument(
        "--profiling-dir",
        help="Profiling input directory (for cross-set checks).",
    )
    parser.add_argument(
        "--gt-dir",
        help="Ground truth input directory (for cross-set checks).",
    )
    parser.add_argument(
        "--profile",
        choices=["profiling", "ground_truth"],
        default="profiling",
        help="Profile type for single-dir checks (default: profiling).",
    )
    parser.add_argument(
        "--workflow",
        help="Workflow ID for workflow-specific checks (e.g. W1, W5, W19).",
    )
    parser.add_argument(
        "--tier-ranges",
        help=(
            'JSON string of tier->(min,max) token ranges. '
            'Example: \'{"easy": [10, 40], "medium": [40, 100]}\''
        ),
    )

    args = parser.parse_args()

    # Parse tier ranges
    tier_ranges: dict[str, tuple[int, int]] | None = None
    if args.tier_ranges:
        raw = json.loads(args.tier_ranges)
        tier_ranges = {k: tuple(v) for k, v in raw.items()}

    all_results: list[tuple[str, bool, str]] = []

    if args.input_dir:
        inputs = load_inputs(args.input_dir)
        print(f"Loaded {len(inputs)} inputs from {args.input_dir}")
        print(f"Profile: {args.profile}, Workflow: {args.workflow or 'generic'}")
        print("=" * 60)
        results = run_all_checks(inputs, args.profile, args.workflow, tier_ranges)
        all_results.extend(results)

    if args.profiling_dir and args.gt_dir:
        prof_inputs = load_inputs(args.profiling_dir)
        gt_inputs = load_inputs(args.gt_dir)
        print(f"Loaded {len(prof_inputs)} profiling + {len(gt_inputs)} GT inputs")
        print(f"Workflow: {args.workflow or 'generic'}")
        print("=" * 60)

        # Run per-set checks on both sets
        print("--- Profiling per-set checks ---")
        prof_results = run_all_checks(
            prof_inputs, "profiling", args.workflow, tier_ranges,
        )
        all_results.extend(prof_results)

        print("--- GT per-set checks ---")
        gt_results = run_all_checks(
            gt_inputs, "ground_truth", args.workflow, tier_ranges,
        )
        all_results.extend(gt_results)

        print("--- Cross-set checks ---")
        cross_results = run_cross_checks(prof_inputs, gt_inputs, args.workflow)
        all_results.extend(cross_results)

    if not args.input_dir and not (args.profiling_dir and args.gt_dir):
        parser.error("Provide --input-dir or both --profiling-dir and --gt-dir.")

    exit_code = _print_results(all_results)

    # Summary
    total = len(all_results)
    passed = sum(1 for _, p, _ in all_results if p)
    failed = total - passed
    print("=" * 60)
    print(f"Summary: {passed}/{total} passed, {failed} failed")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
