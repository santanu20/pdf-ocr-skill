"""Chart parser — bar/pie/bubble chart data extraction (heuristic)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ChartData:
    chart_type: Optional[str] = None  # "bar" | "pie" | "bubble" | "line"
    categories: list = field(default_factory=list)
    values: list = field(default_factory=list)
    units: Optional[str] = None
    confidence: float = 0.3


def parse_chart(
    caption: Optional[str] = None, nearby_text: Optional[str] = None
) -> ChartData:
    """Parse chart from caption + nearby text. Heuristic only."""
    import re

    data = ChartData()
    text = " ".join(filter(None, [caption, nearby_text])).lower()
    if "bar chart" in text or "bar graph" in text:
        data.chart_type = "bar"
    elif "pie chart" in text:
        data.chart_type = "pie"
    elif "bubble chart" in text:
        data.chart_type = "bubble"
    elif "line chart" in text or "line graph" in text:
        data.chart_type = "line"
    if data.chart_type:
        data.confidence = 0.6
    return data


__all__ = ["ChartData", "parse_chart"]
