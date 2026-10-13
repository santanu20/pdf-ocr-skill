"""Diagram parser — cross-section, stratigraphic column, sketch."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiagramData:
    diagram_type: Optional[str] = (
        None  # "cross_section" | "stratigraphic_column" | "schematic"
    )
    layers: list = field(
        default_factory=list
    )  # for stratigraphic: [(top_y, bottom_y, label)]
    confidence: float = 0.3


def parse_diagram(
    caption: Optional[str] = None, nearby_text: Optional[str] = None
) -> DiagramData:
    data = DiagramData()
    text = " ".join(filter(None, [caption, nearby_text])).lower()
    if "cross-section" in text or "cross section" in text:
        data.diagram_type = "cross_section"
        data.confidence = 0.8
    elif "stratigraphic column" in text or "stratigraphic log" in text:
        data.diagram_type = "stratigraphic_column"
        data.confidence = 0.8
    elif "schematic" in text:
        data.diagram_type = "schematic"
        data.confidence = 0.7
    return data


__all__ = ["DiagramData", "parse_diagram"]
