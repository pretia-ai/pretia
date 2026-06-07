from __future__ import annotations
import json
import re
from pathlib import Path
from typing import Any


def discover_results(results_dir: Path) -> dict[str, dict[str, Path]]:
    """Auto-discover result JSON files by workflow and comparison.

    Looks for files matching {workflow}_comparison_{A|B|C}.json or
    {workflow}_{synth20|synth100|real500}.json.
    Returns {workflow_name: {comparison_key: path}}.
    """
    results: dict[str, dict[str, Path]] = {}
    if not results_dir.is_dir():
        return results
    for f in sorted(results_dir.glob("*.json")):
        # Try comparison pattern first
        m = re.match(r"(.+)_comparison_([ABC])\.json$", f.name)
        if m:
            wf, comp = m.group(1), m.group(2)
            results.setdefault(wf, {})[comp] = f
            continue
        # Legacy pattern
        m = re.match(r"(.+)_(synth\d+|real\d+)\.json$", f.name)
        if m:
            wf, key = m.group(1), m.group(2)
            results.setdefault(wf, {})[key] = f
    return results


def save_figure(fig: Any, output_dir: Path, name: str, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save a matplotlib figure in multiple formats. Returns saved paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        paths.append(path)
    return paths


def add_caption(fig: Any, text: str) -> None:
    """Add a one-sentence caption below the figure."""
    fig.text(0.5, -0.02, text, ha="center", va="top", fontsize=9, style="italic",
             wrap=True)


def format_workflow_label(name: str) -> str:
    """Shorten workflow names for axis labels."""
    # "W1-support-simple" -> "W1"
    parts = name.split("-")
    if parts and parts[0].startswith("W"):
        return parts[0]
    return name[:6]


def ensure_output_dir(output_dir: Path) -> Path:
    """Create output directory if it doesn't exist."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
