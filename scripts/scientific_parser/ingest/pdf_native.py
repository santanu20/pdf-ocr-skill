"""PDF native text/geometry/font extraction.

Wraps PyMuPDF + pypdfium2 + pdfplumber into a unified API.
Per proposal item #8: PDF-native extraction is highest-priority evidence
(E0 native PDF text, E1 native PDF geometry).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class PDFNativeResult:
    """Output of native PDF extraction for one page."""

    page_num: int
    text: str = ""
    char_count: int = 0
    fonts: list = field(default_factory=list)  # list of font names
    has_text_layer: bool = False
    has_invisible_text: bool = False
    font_types: dict = field(default_factory=dict)  # {font_name: count}
    has_math_fonts: bool = False  # CMMI/CMSY/CMEX/MSBM
    blocks: list = field(default_factory=list)  # PyMuPDF blocks raw
    page_width: float = 0.0
    page_height: float = 0.0
    rotation: int = 0


# Math font prefixes (high signal for math content)
_MATH_FONT_PREFIXES = (
    "CMMI",
    "CMSY",
    "CMEX",
    "MSBM",
    "MSAM",
    "EU1",
    "EU2",
    "Frak",
    "cmmi",
    "cmsy",
)


def extract_page_native(pdf_path: str, page_num: int) -> PDFNativeResult:
    """Extract native text + font info from one page via PyMuPDF.

    page_num is 1-indexed.
    """
    import pymupdf

    result = PDFNativeResult(page_num=page_num)
    try:
        doc = pymupdf.open(pdf_path)
        if page_num < 1 or page_num > len(doc):
            return result
        page = doc[page_num - 1]
        result.page_width = page.rect.width
        result.page_height = page.rect.height
        result.rotation = page.rotation

        # Text
        text = page.get_text("text") or ""
        result.text = text
        result.char_count = len(text)
        result.has_text_layer = result.char_count > 10

        # Fonts
        fonts = page.get_fonts(full=True)
        font_names = []
        for f in fonts:
            # f = (xref, ext, type, basefont, name, encoding)
            basefont = f[3] if len(f) > 3 else ""
            if basefont:
                font_names.append(basefont)
                result.font_types[basefont] = result.font_types.get(basefont, 0) + 1
        result.fonts = font_names
        result.has_math_fonts = any(
            any(basefont.startswith(p) for p in _MATH_FONT_PREFIXES)
            for basefont in font_names
        )

        # Invisible text detection
        try:
            blocks_with_flags = page.get_text(
                "dict", flags=pymupdf.TEXT_PRESERVE_LIGATURES
            )
            invisible_count = 0
            total_count = 0
            for block in blocks_with_flags.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        total_count += 1
                        flags = span.get("flags", 0)
                        # bit 2 (value 4) = synthetic/italic; check rendering mode
                        if span.get("text", "").strip() and flags & 0x4:
                            invisible_count += 1
            if total_count > 0:
                result.has_invisible_text = invisible_count / total_count > 0.5
        except Exception as e:
            log.debug("extract_page_native: Exception degraded: %s", e)

        # Blocks
        result.blocks = page.get_text("blocks") or []
        doc.close()
    except Exception as e:
        log.debug("extract_page_native: Exception degraded: %s", e)
    return result


def get_page_count(pdf_path: str) -> int:
    """Return number of pages. 0 on error."""
    try:
        import pymupdf

        doc = pymupdf.open(pdf_path)
        n = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


__all__ = ["PDFNativeResult", "extract_page_native", "get_page_count"]
