"""Render charts to PNG or single-page PDF files."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

_VALID_CHART_TYPES = frozenset({"bar", "line", "pie"})


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """Describe a single chart to render."""

    chart_type: str  # "bar" | "line" | "pie"
    title: str
    data: dict[str, list[float]]  # label -> values
    x_labels: list[str] | None = None
    y_label: str | None = None
    figsize: tuple[float, float] = (6.5, 4.0)  # fits within 1" margins on letter

    def __post_init__(self) -> None:
        if self.chart_type not in _VALID_CHART_TYPES:
            raise ValueError(
                f"chart_type must be one of {sorted(_VALID_CHART_TYPES)}, got {self.chart_type!r}"
            )
        if not self.data:
            raise ValueError("data must contain at least one series")


def render_chart_to_png(
    spec: ChartSpec,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Render a chart to a PNG file using matplotlib.

    Return the output path on success. Parent directories are created if missing.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=spec.figsize)

    if spec.chart_type == "bar":
        _draw_bar(ax, spec, np)
    elif spec.chart_type == "line":
        _draw_line(ax, spec)
    elif spec.chart_type == "pie":
        _draw_pie(ax, spec)

    ax.set_title(spec.title, fontsize=11, pad=10)

    if spec.y_label and spec.chart_type != "pie":
        ax.set_ylabel(spec.y_label)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    log.debug("Wrote chart PNG: %s", output_path)
    return output_path


def render_chart_to_pdf_page(
    spec: ChartSpec,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    """Render a chart as a PNG then embed it in a single-page letter-size PDF.

    Return the output path. Uses reportlab for PDF assembly.
    """
    import tempfile

    from reportlab.lib.pagesizes import letter
    from reportlab.lib.units import inch
    from reportlab.pdfgen import canvas

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        png_path = Path(tmp.name)

    render_chart_to_png(spec, png_path, dpi=dpi)

    page_w, page_h = letter
    output_path.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(output_path), pagesize=letter)

    # Scale image to fit within 1" margins on each side.
    max_img_w = page_w - 2 * inch
    max_img_h = page_h - 2 * inch
    img_w = spec.figsize[0] * dpi
    img_h = spec.figsize[1] * dpi

    scale = min(max_img_w / img_w, max_img_h / img_h, 1.0)
    draw_w = img_w * scale
    draw_h = img_h * scale

    # Center on page.
    x = (page_w - draw_w) / 2
    y = (page_h - draw_h) / 2

    c.drawImage(str(png_path), x, y, width=draw_w, height=draw_h)
    c.showPage()
    c.save()

    # Clean up temp PNG.
    png_path.unlink(missing_ok=True)

    log.debug("Wrote chart PDF page: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Internal drawing helpers
# ---------------------------------------------------------------------------


def _draw_bar(ax: object, spec: ChartSpec, np: object) -> None:  # noqa: ANN401
    """Draw grouped bars — one group per x position, one bar per data key."""
    series_names = list(spec.data.keys())
    n_series = len(series_names)
    n_values = len(next(iter(spec.data.values())))

    x = np.arange(n_values)  # type: ignore[attr-defined]
    total_width = 0.8
    bar_width = total_width / n_series

    for i, name in enumerate(series_names):
        offset = (i - (n_series - 1) / 2) * bar_width
        ax.bar(x + offset, spec.data[name], width=bar_width, label=name)  # type: ignore[attr-defined]

    if spec.x_labels:
        ax.set_xticks(x)  # type: ignore[attr-defined]
        ax.set_xticklabels(spec.x_labels)  # type: ignore[attr-defined]

    if n_series > 1:
        ax.legend(fontsize=8)  # type: ignore[attr-defined]


def _draw_line(ax: object, spec: ChartSpec) -> None:  # noqa: ANN401
    """Draw one line per data key."""
    for name, values in spec.data.items():
        if spec.x_labels and len(spec.x_labels) == len(values):
            ax.plot(spec.x_labels, values, marker="o", markersize=4, label=name)  # type: ignore[attr-defined]
        else:
            ax.plot(values, marker="o", markersize=4, label=name)  # type: ignore[attr-defined]

    if len(spec.data) > 1:
        ax.legend(fontsize=8)  # type: ignore[attr-defined]


def _draw_pie(ax: object, spec: ChartSpec) -> None:  # noqa: ANN401
    """Draw a pie chart from the first data series."""
    first_key = next(iter(spec.data))
    values = spec.data[first_key]
    labels = spec.x_labels if spec.x_labels else [str(i) for i in range(len(values))]

    ax.pie(  # type: ignore[attr-defined]
        values,
        labels=labels,
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.set_aspect("equal")  # type: ignore[attr-defined]
