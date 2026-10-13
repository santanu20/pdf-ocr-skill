"""Complexity assessment — wraps coarse_to_fine for the package.

Per proposal item #22: PaddleOCR-VL coarse-to-fine routing.
"""

from __future__ import annotations
import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")

# Re-export from coarse_to_fine (Phase P0.9 module)
from coarse_to_fine import (
    TIER_TRIVIAL,
    TIER_SIMPLE,
    TIER_MEDIUM,
    TIER_HARD,
    PageComplexity,
    assess_page_complexity,
    recommend_model,
    recommend_dpi,
)


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
