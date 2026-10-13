"""Coarse-to-fine routing — adaptive compute budget per page.

Per architecture proposal item #22 (2026-08-12):
  "Coarse-to-fine is now mandatory. The 2026 PaddleOCR-VL work explicitly
   focuses on valid-region focusing / coarse-to-fine visual processing,
   because feeding huge high-resolution pages to a VLM wastes visual tokens
   on irrelevant regions.

   300 DPI whole page should not be your default expensive operation.
   Do: low-res page → layout → regions → high-res only where required."

Implementation strategy:
  - Phase 1 (low-res): render at 75-100 DPI, run layout detection
  - Phase 2 (region classification): for each detected region, decide
    whether high-res OCR is needed (text vs formula vs table vs figure)
  - Phase 3 (targeted high-res): only re-render + OCR regions that need it
  - Phase 4 (skip): text regions on born-digital pages use PDF text layer
    directly (already handled by skill's fast path)

For now (P0.9 minimal viable): the router LOGS the recommended plan but
doesn't enforce it. Skill's existing fast/OCR routing continues. Full
enforcement lands in P1 (render-region integration).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


# Page complexity tiers (informed by PaddleOCR-VL routing lessons)
TIER_TRIVIAL = "trivial"  # PDF text layer, no OCR needed (skill fast path)
TIER_SIMPLE = "simple"  # OCR with default DPI, single model
TIER_MEDIUM = "medium"  # OCR + targeted region refinement
TIER_HARD = "hard"  # multi-model, region crops, verifier loop


@dataclass
class PageComplexity:
    """Output of complexity assessment for a single page."""

    tier: str
    text_density: float  # chars / max_chars_expected
    has_math: bool
    has_tables: bool
    has_figures: bool
    has_multicol: bool
    is_scanned: bool
    confidence: float = 1.0
    recommendation: str = ""
    target_regions: list = field(default_factory=list)  # regions needing high-res


def assess_page_complexity(
    pdf_text_len: int,
    has_math_indicators: bool = False,
    has_table_indicators: bool = False,
    has_figure_indicators: bool = False,
    is_multicol: bool = False,
    is_scanned: bool = False,
    font_count: int = 0,
) -> PageComplexity:
    """Decide which compute tier a page needs.

    Args:
        pdf_text_len: chars in PDF text layer for this page (0 if scanned)
        has_math_indicators: True if CMMI/CMSY/etc. math fonts detected
        has_table_indicators: True if pdfplumber/camelot found table hints
        has_figure_indicators: True if raster figure regions detected
        is_multicol: True if 2+ columns detected
        is_scanned: True if page has no text layer (image-only)
        font_count: number of distinct embedded fonts

    Returns:
        PageComplexity with tier + recommendations
    """
    # Trivial: PDF text layer is rich AND no special regions
    if pdf_text_len >= 500 and not is_scanned and not has_math_indicators:
        return PageComplexity(
            tier=TIER_TRIVIAL,
            text_density=1.0,
            has_math=False,
            has_tables=has_table_indicators,
            has_figures=has_figure_indicators,
            has_multicol=is_multicol,
            is_scanned=False,
            recommendation="Use PDF text layer directly (skill fast path). No OCR.",
        )

    # Hard: scanned + math (math OCR is the most expensive operation)
    if is_scanned and has_math_indicators:
        return PageComplexity(
            tier=TIER_HARD,
            text_density=0.0,
            has_math=True,
            has_tables=has_table_indicators,
            has_figures=has_figure_indicators,
            has_multicol=is_multicol,
            is_scanned=True,
            recommendation=(
                "Scanned + math: use glm-ocr (math-aware) at 300 DPI. "
                "Run pykatex verifier on output. Budget: 60s/page."
            ),
        )

    # Hard: scanned + tables (table structure needs high-res)
    if is_scanned and has_table_indicators:
        return PageComplexity(
            tier=TIER_HARD,
            text_density=0.0,
            has_math=False,
            has_tables=True,
            has_figures=False,
            has_multicol=is_multicol,
            is_scanned=True,
            recommendation=(
                "Scanned + table: OCR at 300 DPI, then table structure "
                "detection. Budget: 45s/page."
            ),
        )

    # Medium: scanned, no special regions (plain text)
    if is_scanned:
        return PageComplexity(
            tier=TIER_MEDIUM,
            text_density=0.0,
            has_math=False,
            has_tables=False,
            has_figures=has_figure_indicators,
            has_multicol=is_multicol,
            is_scanned=True,
            recommendation=(
                "Scanned plain text: use qwen2.5vl:3b at 150 DPI "
                "(2026 OCR winner for plain text). Budget: 20s/page."
            ),
        )

    # Medium: born-digital but math/tables present (needs region-level OCR)
    if has_math_indicators or has_table_indicators:
        return PageComplexity(
            tier=TIER_MEDIUM,
            text_density=pdf_text_len / 1500.0 if pdf_text_len else 0,
            has_math=has_math_indicators,
            has_tables=has_table_indicators,
            has_figures=has_figure_indicators,
            has_multicol=is_multicol,
            is_scanned=False,
            recommendation=(
                "Born-digital with special regions: use PDF text for body, "
                "targeted OCR for math/table regions only. Budget: 10s/page."
            ),
        )

    # Simple: born-digital, minimal special content
    return PageComplexity(
        tier=TIER_SIMPLE,
        text_density=pdf_text_len / 1500.0 if pdf_text_len else 0,
        has_math=False,
        has_tables=False,
        has_figures=has_figure_indicators,
        has_multicol=is_multicol,
        is_scanned=False,
        recommendation="Born-digital text: PDF text layer + typography pipeline.",
    )


def recommend_model(complexity: PageComplexity) -> str:
    """Return model name recommended for this complexity tier."""
    if complexity.tier == TIER_TRIVIAL:
        return "pdf_text_layer"  # no OCR
    if complexity.tier == TIER_HARD and complexity.has_math:
        return "glm-ocr"  # math-aware
    if complexity.tier == TIER_MEDIUM and complexity.is_scanned:
        return "qwen2.5vl:3b"  # plain-text scanned
    if complexity.tier == TIER_HARD:
        return "qwen2.5vl:7b"  # heavy non-math
    return "pdf_text_layer"  # born-digital default


def recommend_dpi(complexity: PageComplexity) -> int:
    """Return DPI recommended for this tier."""
    if complexity.tier == TIER_TRIVIAL:
        return 0  # no rendering needed
    if complexity.tier == TIER_HARD:
        return 300  # max detail for math/tables
    if complexity.tier == TIER_MEDIUM:
        return 150  # sufficient for plain text
    return 100  # TIER_SIMPLE


__all__ = [
    "TIER_TRIVIAL",
    "TIER_SIMPLE",
    "TIER_MEDIUM",
    "TIER_HARD",
    "PageComplexity",
    "assess_page_complexity",
    "recommend_model",
    "recommend_dpi",
]
