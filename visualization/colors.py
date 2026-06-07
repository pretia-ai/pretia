from __future__ import annotations
from dataclasses import dataclass

WORKFLOW_GROUPS: dict[str, str] = {
    "W1": "linear", "W5": "linear", "W9": "linear", "W11": "linear",
    "W12": "linear", "W18": "linear",
    "W2": "loops", "W4": "loops", "W15": "loops", "W19": "loops",
    "W13": "routing", "W17": "routing",
    "W14": "rag_pdf", "W16": "rag_pdf",
}

GROUP_COLORS: dict[str, str] = {
    "linear": "#2ecc71",
    "loops": "#9b59b6",
    "routing": "#f39c12",
    "rag_pdf": "#3498db",
}

COMPARISON_COLORS: dict[str, str] = {
    "A": "#27ae60",
    "B": "#e67e22",
    "C": "#2980b9",
}

DETECTOR_MATRIX_COLORS: dict[str, str] = {
    "TP": "#27ae60",
    "TN": "#bdc3c7",
    "FP": "#f1c40f",
    "FN": "#e74c3c",
}

VERDICT_COLORS: dict[str, str] = {
    "PASS": "#27ae60",
    "WARN": "#f39c12",
    "FAIL": "#e74c3c",
}

# Expected detector activations from cross-cutting-robustness.md Section 9
EXPECTED_DETECTORS: dict[str, dict[str, bool]] = {
    "W1":  {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": False, "bimodality": False},
    "W2":  {"context_growth": True,  "loop_count_variance": True,  "high_token_variance": False, "step_count_variance": True,  "bimodality": False},
    "W4":  {"context_growth": True,  "loop_count_variance": True,  "high_token_variance": False, "step_count_variance": False, "bimodality": False},
    "W5":  {"context_growth": False, "loop_count_variance": False, "high_token_variance": True,  "step_count_variance": False, "bimodality": False},
    "W9":  {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": False, "bimodality": False},
    "W11": {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": False, "bimodality": False},
    "W12": {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": False, "bimodality": False},
    "W13": {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": True,  "bimodality": True},
    "W14": {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": False, "bimodality": False},
    "W15": {"context_growth": True,  "loop_count_variance": True,  "high_token_variance": False, "step_count_variance": True,  "bimodality": True},
    "W16": {"context_growth": False, "loop_count_variance": False, "high_token_variance": True,  "step_count_variance": True,  "bimodality": False},
    "W17": {"context_growth": False, "loop_count_variance": False, "high_token_variance": False, "step_count_variance": True,  "bimodality": True},
    "W18": {"context_growth": False, "loop_count_variance": False, "high_token_variance": True,  "step_count_variance": False, "bimodality": False},
    "W19": {"context_growth": True,  "loop_count_variance": False, "high_token_variance": False, "step_count_variance": False, "bimodality": False},
}


def workflow_color(name: str) -> str:
    """Return hex color for a workflow based on its group."""
    group = WORKFLOW_GROUPS.get(name, "linear")
    return GROUP_COLORS[group]


def classify_detector_result(workflow: str, detector: str, fired: bool) -> str:
    """Classify a detector result as TP, TN, FP, or FN."""
    expected = EXPECTED_DETECTORS.get(workflow, {}).get(detector, False)
    if expected and fired:
        return "TP"
    if expected and not fired:
        return "FN"
    if not expected and fired:
        return "FP"
    return "TN"
