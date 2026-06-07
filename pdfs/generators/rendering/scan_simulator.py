"""Simulate scanned documents by rasterizing PDF pages with controlled degradation.

Only used for W14/W15 ground truth generation.
Uses pypdfium2 (pure Python, no system deps) for PDF-to-image conversion.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ScanParams:
    """Control the degradation applied when simulating a document scan."""

    dpi: int = 175  # 150-200 range
    blur_sigma: float = 0.5  # 0.3-0.8 range
    rotation_degrees: float = 1.0  # +/-0.5-2.0 range
    noise_sigma: float = 5.0  # 3-8 range
    contrast_factor: float = 1.0  # 0.8-1.2 range


def randomize_scan_params(rng: random.Random) -> ScanParams:
    """Generate randomized but realistic scan degradation parameters."""
    return ScanParams(
        dpi=rng.randint(150, 200),
        blur_sigma=round(rng.uniform(0.3, 0.8), 2),
        rotation_degrees=round(rng.uniform(-2.0, 2.0), 2),
        noise_sigma=round(rng.uniform(3.0, 8.0), 2),
        contrast_factor=round(rng.uniform(0.8, 1.2), 2),
    )


def _pdf_page_to_pil(pdf_path: Path, page_index: int, dpi: int) -> Any:
    """Rasterize one PDF page to a PIL Image using pypdfium2."""
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(str(pdf_path))
    if page_index >= len(pdf):
        msg = f"Page index {page_index} out of range for {pdf_path} (has {len(pdf)} pages)"
        raise ValueError(msg)

    page = pdf[page_index]
    scale = dpi / 72.0
    bitmap = page.render(scale=scale)
    pil_image = bitmap.to_pil()
    pdf.close()
    return pil_image


def rasterize_pdf_page(
    pdf_path: Path,
    page_index: int,
    params: ScanParams | None = None,
) -> Any:
    """Rasterize one PDF page and apply scan-like degradation.

    Degradation pipeline: grayscale -> Gaussian blur -> rotate -> Gaussian noise
    -> contrast adjustment.

    Returns a PIL Image.
    """
    if params is None:
        params = ScanParams()

    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter

    img = _pdf_page_to_pil(pdf_path, page_index, params.dpi)

    # 1. Convert to grayscale.
    img = img.convert("L")

    # 2. Gaussian blur.
    if params.blur_sigma > 0:
        radius = max(1, int(params.blur_sigma * 2))
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 3. Rotation.
    if params.rotation_degrees != 0:
        img = img.rotate(
            params.rotation_degrees,
            resample=Image.BICUBIC,
            expand=False,
            fillcolor=255,
        )

    # 4. Gaussian noise.
    if params.noise_sigma > 0:
        arr = np.array(img, dtype=np.float64)
        noise = np.random.default_rng().normal(0, params.noise_sigma, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="L")

    # 5. Contrast adjustment.
    if params.contrast_factor != 1.0:
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(params.contrast_factor)

    log.debug(
        "Rasterized page %d of %s (dpi=%d, blur=%.2f, rot=%.2f, noise=%.2f, contrast=%.2f)",
        page_index,
        pdf_path.name,
        params.dpi,
        params.blur_sigma,
        params.rotation_degrees,
        params.noise_sigma,
        params.contrast_factor,
    )

    return img


def rasterize_pdf_pages(
    pdf_path: Path,
    page_indices: list[int],
    params: ScanParams | None = None,
    rng: random.Random | None = None,
) -> list[Any]:
    """Rasterize multiple PDF pages with scan degradation.

    If *params* is None and *rng* is provided, randomize degradation per page.
    If both are None, use default ScanParams for every page.
    """
    results: list[Any] = []

    for idx in page_indices:
        if params is not None:
            page_params = params
        elif rng is not None:
            page_params = randomize_scan_params(rng)
        else:
            page_params = ScanParams()

        results.append(rasterize_pdf_page(pdf_path, idx, page_params))

    return results
