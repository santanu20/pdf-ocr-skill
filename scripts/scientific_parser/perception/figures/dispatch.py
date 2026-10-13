"""Figure type classifier + dispatch (proposal item #19).

Per architecture proposal item #19:
  "Figure parsing: general core, specialist plugins. The general parser
   should detect: photo, plot, chart, diagram, map, schematic, microscopy,
   workflow, unknown.

   Then dispatch. PlotParser / ChartParser / MapParser / DiagramParser /
   ScientificImageParser. Domain packs can then interpret semantics."

Implementation:
  - Classifier uses caption + aspect ratio + content hints (regex on text)
  - Dispatch routes to a handler (currently stubs — to be filled per type)
  - Pure regex/heuristic — no ML
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Figure types per proposal
FIG_PHOTO = "photo"
FIG_PLOT = "plot"
FIG_CHART = "chart"
FIG_DIAGRAM = "diagram"
FIG_MAP = "map"
FIG_SCHEMATIC = "schematic"
FIG_MICROSCOPY = "microscopy"
FIG_WORKFLOW = "workflow"
FIG_UNKNOWN = "unknown"


# Caption patterns → figure type
_CAPTION_PATTERNS = {
    FIG_PHOTO: [
        r"\bphoto(?:graph)?\b",
        r"\bimage\b",
        r"\bfield\s+photo\b",
    ],
    FIG_PLOT: [
        r"\bplot\b",
        r"\bscatter\s+plot\b",
        r"\bline\s+(?:plot|graph)\b",
        r"\bvariogram\b",
        r"\bhistogram\b",
        r"\bdistribution\b",
    ],
    FIG_CHART: [
        r"\bchart\b",
        r"\bbar\s+chart\b",
        r"\bpie\s+chart\b",
        r"\bbubble\s+chart\b",
        r"\bdiagram\s+chart\b",
    ],
    FIG_DIAGRAM: [
        r"\bdiagram\b",
        r"\bsketch\b",
        r"\bschematic\s+diagram\b",
        r"\bcross[- ]section\b",
        r"\bstratigraphic\s+column\b",
    ],
    FIG_MAP: [
        r"\bmap\b",
        r"\blocation\s+map\b",
        r"\bgeological\s+map\b",
        r"\btopographic\s+map\b",
        r"\bsampling\s+map\b",
    ],
    FIG_SCHEMATIC: [
        r"\bschematic\b",
        r"\bcircuit\b",
        r"\bflow\s+sheet\b",
    ],
    FIG_MICROSCOPY: [
        r"\bmicrograph\b",
        r"\bSEM\s+image\b",
        r"\bTEM\s+image\b",
        r"\bAFM\s+image\b",
        r"\bbackscatter(?:ed)?\b",
        r"\bBSE\s+image\b",
        r"\bCL\s+image\b",
        r"\bcathodoluminescence\b",
        r"\bphotomicrograph\b",
        r"\bthin\s+section\b",
    ],
    FIG_WORKFLOW: [
        r"\bworkflow\b",
        r"\bflowchart\b",
        r"\bprocess\s+diagram\b",
        r"\bmethodology\s+diagram\b",
    ],
}


@dataclass
class FigureClassification:
    """One figure's classification result."""

    figure_type: str
    confidence: float
    caption_match: Optional[str] = None
    aspect_ratio: Optional[float] = None
    dispatch_handler: Optional[str] = None


def classify_figure(
    caption: Optional[str] = None,
    aspect_ratio: Optional[float] = None,
    nearby_text: Optional[str] = None,
) -> FigureClassification:
    """Classify figure from caption + aspect ratio + nearby text.

    Args:
        caption: figure caption text (most informative signal)
        aspect_ratio: width / height
        nearby_text: body text near figure (additional signal)

    Returns:
        FigureClassification with type + confidence + dispatch handler.
    """
    text = " ".join(filter(None, [caption, nearby_text])).lower()

    if not text:
        # No text signal — use aspect ratio only
        if aspect_ratio:
            if aspect_ratio > 1.8:
                return FigureClassification(
                    FIG_PLOT, 0.30, aspect_ratio=aspect_ratio, dispatch_handler=None
                )
            elif aspect_ratio < 0.6:
                return FigureClassification(
                    FIG_PLOT, 0.30, aspect_ratio=aspect_ratio, dispatch_handler=None
                )
        return FigureClassification(
            FIG_UNKNOWN, 0.10, aspect_ratio=aspect_ratio, dispatch_handler=None
        )

    # Match caption patterns
    matches = []
    for fig_type, patterns in _CAPTION_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                matches.append((fig_type, m.group(0), pat))

    if matches:
        # First match wins (highest priority order)
        fig_type, matched_text, _ = matches[0]
        return FigureClassification(
            figure_type=fig_type,
            confidence=0.85,
            caption_match=matched_text,
            aspect_ratio=aspect_ratio,
            dispatch_handler=_dispatch_for(fig_type),
        )

    # No match — unknown
    return FigureClassification(
        figure_type=FIG_UNKNOWN,
        confidence=0.20,
        aspect_ratio=aspect_ratio,
        dispatch_handler=None,
    )


def _dispatch_for(fig_type: str) -> Optional[str]:
    """Return handler name for figure type. None = no specialist handler."""
    return {
        FIG_PLOT: "plot_parser",
        FIG_CHART: "chart_parser",
        FIG_MAP: "map_parser",
        FIG_MICROSCOPY: "microscopy_parser",
        FIG_DIAGRAM: "diagram_parser",
    }.get(fig_type)


__all__ = [
    "FIG_PHOTO",
    "FIG_PLOT",
    "FIG_CHART",
    "FIG_DIAGRAM",
    "FIG_MAP",
    "FIG_SCHEMATIC",
    "FIG_MICROSCOPY",
    "FIG_WORKFLOW",
    "FIG_UNKNOWN",
    "FigureClassification",
    "classify_figure",
]
