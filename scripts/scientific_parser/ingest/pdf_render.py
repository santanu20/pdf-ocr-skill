"""PDF → image rendering at controlled DPI.

Used for OCR fallback when native text layer is poor.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional


def render_page(
    pdf_path: str, page_num: int, dpi: int = 150, out_path: Optional[str] = None
) -> str:
    """Render one PDF page to PNG. page_num is 1-indexed.

    Returns path to rendered image (temp file if out_path is None).
    """
    import pymupdf
    import tempfile

    doc = pymupdf.open(pdf_path)
    if page_num < 1 or page_num > len(doc):
        doc.close()
        raise ValueError(f"page {page_num} out of range (1..{len(doc)})")
    page = doc[page_num - 1]
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72))

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix=f"_p{page_num}_{dpi}dpi.png")
        import os

        os.close(fd)
    pix.save(out_path)
    doc.close()
    return out_path


def render_region(
    pdf_path: str,
    page_num: int,
    bbox: tuple,
    dpi: int = 300,
    out_path: Optional[str] = None,
) -> str:
    """Render a sub-region (bbox = x0,y0,x1,y1 in PDF points) to PNG.

    Used for coarse-to-fine: high-res render only of detected regions.
    """
    import pymupdf
    import tempfile

    doc = pymupdf.open(pdf_path)
    page = doc[page_num - 1]
    clip = pymupdf.Rect(bbox)
    pix = page.get_pixmap(matrix=pymupdf.Matrix(dpi / 72, dpi / 72), clip=clip)

    if out_path is None:
        fd, out_path = tempfile.mkstemp(suffix="_region.png")
        import os

        os.close(fd)
    pix.save(out_path)
    doc.close()
    return out_path


def estimate_render_dpi(font_size_pt: float, target_px_per_em: int = 18) -> int:
    """Pick DPI so that typical body text (font_size_pt) renders to target_px_per_em.

    Default: 12pt body → 150 DPI → 25px tall glyphs (readable for most OCR).
    """
    if font_size_pt <= 0:
        return 150
    dpi = int(target_px_per_em / font_size_pt * 72)
    # Clamp to reasonable bounds
    return max(72, min(600, dpi))


__all__ = ["render_page", "render_region", "estimate_render_dpi"]
