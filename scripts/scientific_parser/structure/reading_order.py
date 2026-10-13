"""Reading order orchestrator.

Top-level wrapper that uses xy_cut.reading_order() over regions extracted
by ingest.pdf_native or perception.layout detectors.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
from ..perception.layout.xy_cut import reading_order as _xy_cut_order


@dataclass
class ReadingOrderResult:
    """Output of reading-order computation."""

    ordered_indices: list
    algorithm: str = "xy_cut_pp"
    multicol: bool = False


def compute_reading_order(
    regions: list, prefer_columns: bool = True, algorithm: str = "xy_cut_pp"
) -> ReadingOrderResult:
    """Compute reading order over a list of regions.

    Args:
        regions: list of dicts/dataclasses with x0/y0/x1/y1
        prefer_columns: True for multi-column layouts
        algorithm: "xy_cut_pp" only (future: "layout_reader", "seg2act")

    Returns:
        ReadingOrderResult with ordered region indices.
    """
    if algorithm != "xy_cut_pp":
        raise NotImplementedError(f"algorithm={algorithm} not yet implemented")
    order = _xy_cut_order(regions, prefer_columns=prefer_columns)
    return ReadingOrderResult(
        ordered_indices=order,
        algorithm=algorithm,
        multicol=prefer_columns,
    )


__all__ = ["ReadingOrderResult", "compute_reading_order"]
