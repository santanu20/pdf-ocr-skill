"""Math equation extraction — perception package.

Phase R3 refactoring (2026-08-12): extracted from pdf_ocr.py monolith.
Re-exports all math symbols from _extracted.py for backwards compat.
"""

from .math_extractor import (
    MATH_FONT_PREFIXES,
    UNICODE_TO_LATEX,
    _OCR_LATEX_CMD_CONFUSIONS,
    detect_display_equations,
    wrap_equations_as_latex,
    ocr_display_equations,
    _is_math_font,
    _has_math_content,
    _span_to_latex,
    reconstruct_math_region,
    extract_math_regions,
    replace_math_in_text,
    improve_math_regions,
)

__all__ = [
    "MATH_FONT_PREFIXES",
    "UNICODE_TO_LATEX",
    "_OCR_LATEX_CMD_CONFUSIONS",
    "detect_display_equations",
    "wrap_equations_as_latex",
    "ocr_display_equations",
    "_is_math_font",
    "_has_math_content",
    "_span_to_latex",
    "reconstruct_math_region",
    "extract_math_regions",
    "replace_math_in_text",
    "improve_math_regions",
]
