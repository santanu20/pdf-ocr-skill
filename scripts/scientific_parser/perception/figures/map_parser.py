"""Map parser — geological/geographic map feature extraction (heuristic)."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MapData:
    map_type: Optional[str] = (
        None  # "geological" | "sampling" | "location" | "topographic"
    )
    legend: list = field(default_factory=list)  # [(color, label), ...]
    scale_bar: Optional[str] = None
    north_arrow: bool = False
    confidence: float = 0.3


def parse_map(
    caption: Optional[str] = None, nearby_text: Optional[str] = None
) -> MapData:
    import re

    data = MapData()
    text = " ".join(filter(None, [caption, nearby_text])).lower()
    if "geological map" in text:
        data.map_type = "geological"
        data.confidence = 0.8
    elif "sampling map" in text or "sample location" in text:
        data.map_type = "sampling"
        data.confidence = 0.7
    elif "topographic" in text:
        data.map_type = "topographic"
        data.confidence = 0.7
    elif "location map" in text or "index map" in text:
        data.map_type = "location"
        data.confidence = 0.6
    if "scale bar" in text or "scale: 1:" in text:
        data.scale_bar = "present"
    if "north arrow" in text:
        data.north_arrow = True
    return data


__all__ = ["MapData", "parse_map"]
