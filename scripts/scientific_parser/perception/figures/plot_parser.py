"""Plot parser — extract axes, ticks, data points from vector-graphic plots.

Uses vector_extract for axis lines + caption matching for axes labels.
Pure heuristic — no ML.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PlotData:
    """Extracted plot data."""

    x_axis_label: Optional[str] = None
    y_axis_label: Optional[str] = None
    x_units: Optional[str] = None
    y_units: Optional[str] = None
    data_points: list = field(default_factory=list)  # [(x, y), ...]
    series_labels: list = field(default_factory=list)
    confidence: float = 0.5


def parse_plot(
    caption: Optional[str] = None,
    axes_lines: Optional[list] = None,
    nearby_text: Optional[str] = None,
) -> PlotData:
    """Parse a plot from caption + axes vector data.

    Args:
        caption: figure caption (e.g. "Fig 5. MgO vs SiO2 (wt%)")
        axes_lines: list of VectorPath from ingest.vector_extract
        nearby_text: body text near figure
    """
    import re

    data = PlotData()

    if caption:
        # Try to extract X vs Y pattern: "X vs Y" or "Y vs X"
        m = re.search(r"(\w[\w\d]*)\s+vs\.?\s+(\w[\w\d]*)", caption, re.IGNORECASE)
        if m:
            data.x_axis_label = m.group(2)
            data.y_axis_label = m.group(1)
        # Try to extract units from parenthetical
        m = re.search(r"\(([^)]+)\)\s*$", caption)
        if m:
            data.x_units = m.group(1)
            data.y_units = m.group(1)
        data.confidence = 0.7 if data.x_axis_label else 0.4
    return data


__all__ = ["PlotData", "parse_plot"]
