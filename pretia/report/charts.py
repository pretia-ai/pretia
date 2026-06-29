"""Generate inline SVG charts for the HTML report."""

from __future__ import annotations

import math
from dataclasses import dataclass
from html import escape

from pretia.ci.report import format_cost

_CIRCUMFERENCE = 2 * math.pi * 80  # 502.654...
_BAR_HEIGHT = 28
_BAR_GAP = 12
_MAX_BAR_WIDTH = 480
_LABEL_X = 120
_SVG_WIDTH = 700


@dataclass(frozen=True, slots=True)
class StepCostEntry:
    """One step's cost data for the waterfall chart."""

    step_name: str
    mean_cost: float
    share_pct: float


def render_score_ring(score: int, zone_color: str, zone_label: str) -> str:
    """Render the optimization score as an SVG donut ring.

    Returns a complete ``<svg>`` element string.
    """
    score = max(0, min(100, score))
    offset = _CIRCUMFERENCE * (1 - score / 100)

    safe_label = escape(zone_label)
    safe_color = escape(zone_color)

    return (
        '<svg class="score-ring" viewBox="0 0 200 200" width="260" height="260" '
        'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Optimization score: {score} of 100">'
        '<circle cx="100" cy="100" r="80" fill="none" '
        'stroke="#e2e8f0" stroke-width="12"/>'
        f'<circle class="score-ring-progress" cx="100" cy="100" r="80" fill="none" '
        f'stroke="{safe_color}" stroke-width="12" '
        f'stroke-dasharray="{_CIRCUMFERENCE:.2f}" '
        f'stroke-dashoffset="{offset:.2f}" '
        'stroke-linecap="round" '
        'transform="rotate(-90 100 100)"/>'
        f'<text x="100" y="95" text-anchor="middle" '
        f'font-size="42" font-weight="bold" fill="#1a202c">{score}</text>'
        '<text x="100" y="118" text-anchor="middle" '
        'font-size="14" fill="#718096">of 100</text>'
        f'<text x="100" y="145" text-anchor="middle" '
        f'font-size="12" fill="{safe_color}">{safe_label}</text>'
        "</svg>"
    )


def _bar_color(rank: int, total: int) -> str:
    """Return a color for bar at *rank* (0-indexed) out of *total* bars."""
    if total <= 1:
        return "#E53E3E"
    t = rank / (total - 1)
    if t < 0.33:
        return "#E53E3E"
    if t < 0.66:
        return "#DD6B20"
    return "#A0AEC0"


def render_cost_waterfall(steps: list[StepCostEntry]) -> str:
    """Render a horizontal bar chart of per-step costs as SVG.

    *steps* are sorted internally by descending cost.
    Returns a complete ``<svg>`` element string.
    """
    if not steps:
        return (
            f'<svg viewBox="0 0 {_SVG_WIDTH} 60" width="{_SVG_WIDTH}" '
            'height="60" xmlns="http://www.w3.org/2000/svg">'
            '<text x="300" y="35" text-anchor="middle" '
            'font-size="14" fill="#718096">No step cost data</text>'
            "</svg>"
        )

    sorted_steps = sorted(steps, key=lambda s: s.mean_cost, reverse=True)
    max_cost = sorted_steps[0].mean_cost if sorted_steps[0].mean_cost > 0 else 1.0
    total = len(sorted_steps)

    row_height = _BAR_HEIGHT + _BAR_GAP
    svg_height = total * row_height + 10

    parts: list[str] = [
        f'<svg viewBox="0 0 {_SVG_WIDTH} {svg_height}" '
        f'width="{_SVG_WIDTH}" height="{svg_height}" '
        f'xmlns="http://www.w3.org/2000/svg" role="img" '
        f'aria-label="Cost breakdown by step">'
    ]

    for i, entry in enumerate(sorted_steps):
        y = i * row_height + 5
        bar_width = max(2, (entry.mean_cost / max_cost) * _MAX_BAR_WIDTH)
        color = _bar_color(i, total)
        safe_name = escape(entry.step_name)
        cost_label = format_cost(entry.mean_cost)
        pct_label = f"{entry.share_pct:.0f}%"

        parts.append(
            f'<text x="{_LABEL_X - 8}" y="{y + 17}" '
            f'text-anchor="end" font-size="14" font-weight="600" fill="#4a5568">'
            f"{safe_name}</text>"
        )

        parts.append(
            f'<rect class="waterfall-bar" x="{_LABEL_X}" y="{y}" '
            f'width="{bar_width:.1f}" height="{_BAR_HEIGHT}" '
            f'rx="6" fill="{color}" style="animation-delay:{i * 0.08:.2f}s"/>'
        )

        parts.append(
            f'<text x="{_LABEL_X + bar_width + 8}" y="{y + 17}" '
            f'font-size="13" fill="#4a5568">'
            f"{cost_label} ({pct_label})</text>"
        )

    parts.append("</svg>")
    return "".join(parts)
