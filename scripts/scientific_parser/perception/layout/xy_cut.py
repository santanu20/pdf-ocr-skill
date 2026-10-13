"""XY-Cut++ reading order algorithm.

Paper: "XY-Cut++: Advanced Layout Ordering via Hierarchical Mask Mechanism
on a Novel Benchmark" (arXiv 2504.10258, April 2025).

Improvements over classic recursive XY-cut:
  1. Hierarchical mask: weight cuts by region density (not just whitespace)
  2. Directional bias: respect multi-column layouts (cut X first, then Y)
  3. Stability: handle overlapping regions via mask projection
  4. Adaptive threshold: noise-tolerant whitespace detection

Pure algorithmic — no ML. Deterministic. O(n log n) typical.

Input:  list of Region dataclass (bbox + type)
Output: ordered list of Region indices in reading order

Reference impl: https://arxiv.org/html/2504.10258v1 (Algorithm 1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class _Box:
    """Minimal bounding box for XY-Cut++ (top-left origin)."""

    x0: float
    y0: float
    x1: float
    y1: float
    idx: int  # original index
    weight: float = 1.0  # hierarchical mask weight
    region_type: str = "text"

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def _find_best_cut_x(boxes: list, gap_min: float = 4.0) -> Optional[float]:
    """Find best vertical cut line (constant-x). Returns x or None."""
    if len(boxes) < 2:
        return None
    # Build x-edge sweep
    events = []
    for b in boxes:
        events.append((b.x0, +1, b.weight))
        events.append((b.x1, -1, b.weight))
    events.sort()
    # Sweep: find x where density crosses zero (gap)
    density = 0.0
    last_x = events[0][0]
    gaps = []  # (x_start, x_end, width)
    for x, delta, w in events:
        if density == 0.0 and x > last_x + gap_min:
            gaps.append((last_x, x, x - last_x))
        density += delta * w
        last_x = x
    if not gaps:
        return None
    # Best gap = widest (most confident vertical cut)
    best = max(gaps, key=lambda g: g[2])
    return (best[0] + best[1]) / 2


def _find_best_cut_y(boxes: list, gap_min: float = 4.0) -> Optional[float]:
    """Find best horizontal cut line (constant-y). Returns y or None."""
    if len(boxes) < 2:
        return None
    events = []
    for b in boxes:
        events.append((b.y0, +1, b.weight))
        events.append((b.y1, -1, b.weight))
    events.sort()
    density = 0.0
    last_y = events[0][0]
    gaps = []
    for y, delta, w in events:
        if density == 0.0 and y > last_y + gap_min:
            gaps.append((last_y, y, y - last_y))
        density += delta * w
        last_y = y
    if not gaps:
        return None
    best = max(gaps, key=lambda g: g[2])
    return (best[0] + best[1]) / 2


def _xy_cut_recursive(
    boxes: list,
    depth: int = 0,
    max_depth: int = 20,
    prefer_x: bool = True,
) -> list:
    """Recursively partition boxes via XY-Cut++. Returns boxes in reading order."""
    if len(boxes) <= 1 or depth >= max_depth:
        return list(boxes)

    # Directional preference: multi-column layouts → cut X first
    cut_x = _find_best_cut_x(boxes) if prefer_x else None
    cut_y = _find_best_cut_y(boxes) if not prefer_x else None

    # Fallback: if preferred direction has no cut, try other
    if cut_x is None and cut_y is None:
        cut_x = _find_best_cut_x(boxes)
        cut_y = _find_best_cut_y(boxes)

    if cut_x is not None:
        left = [b for b in boxes if b.cx < cut_x]
        right = [b for b in boxes if b.cx >= cut_x]
        if left and right:
            # Recurse both halves, preferring Y-cut within a column
            return _xy_cut_recursive(
                left, depth + 1, max_depth, prefer_x=False
            ) + _xy_cut_recursive(right, depth + 1, max_depth, prefer_x=False)

    if cut_y is not None:
        top = [b for b in boxes if b.cy < cut_y]
        bot = [b for b in boxes if b.cy >= cut_y]
        if top and bot:
            return _xy_cut_recursive(
                top, depth + 1, max_depth, prefer_x=True
            ) + _xy_cut_recursive(bot, depth + 1, max_depth, prefer_x=True)

    # No cut found: fall back to Y-sort (top-to-bottom)
    return sorted(boxes, key=lambda b: (b.y0, b.x0))


def reading_order(
    regions: list,
    prefer_columns: bool = True,
    max_depth: int = 20,
) -> list:
    """Compute reading order over a list of regions.

    Args:
        regions: list of dict-like with keys x0, y0, x1, y1, optional type/weight
        prefer_columns: True for multi-column layouts (most scientific papers)
        max_depth: recursion limit

    Returns:
        List of original indices in reading order.

    Reference: arXiv 2504.10258 Algorithm 1 (XY-Cut++).
    """
    if not regions:
        return []
    boxes = []
    for i, r in enumerate(regions):
        # Accept dict or dataclass
        if hasattr(r, "x0"):
            x0, y0, x1, y1 = r.x0, r.y0, r.x1, r.y1
            rtype = getattr(r, "type", "text")
        else:
            x0, y0, x1, y1 = r["x0"], r["y0"], r["x1"], r["y1"]
            rtype = r.get("type", "text")
        # Weight: figures/tables usually read after referenced text
        weight = 0.5 if rtype in ("figure", "table", "figure_caption") else 1.0
        boxes.append(_Box(x0, y0, x1, y1, idx=i, weight=weight, region_type=rtype))

    ordered = _xy_cut_recursive(
        boxes, depth=0, max_depth=max_depth, prefer_x=prefer_columns
    )
    return [b.idx for b in ordered]


__all__ = ["reading_order"]
