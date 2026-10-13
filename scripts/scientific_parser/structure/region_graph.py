"""Region graph — adjacency / containment relationships between blocks.

Per proposal: REGION GRAPH layer feeds L4 Semantics. Used to associate
captions with figures, equations with their references, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RegionNode:
    """One region with bbox + type."""

    id: str
    bbox: tuple  # (x0, y0, x1, y1)
    region_type: str  # text|title|table|figure|equation|caption
    page: int
    text: str = ""
    parent_id: Optional[str] = None
    child_ids: list = field(default_factory=list)
    neighbor_ids: list = field(default_factory=list)


def build_region_graph(regions: list) -> dict:
    """Build region graph: returns {region_id: RegionNode}.

    Computes:
      - Containment (parent/child)
      - Horizontal/vertical adjacency (neighbors)
    """
    nodes = {}
    for r in regions:
        # Accept dict or dataclass
        if hasattr(r, "id"):
            rid = r.id
            bbox = r.bbox
            rtype = getattr(r, "type", "text")
            page = getattr(r, "page", 1)
            text = getattr(r, "text", "")
        else:
            rid = r.get("id")
            bbox = r.get("bbox") or (r.get("x0"), r.get("y0"), r.get("x1"), r.get("y1"))
            rtype = r.get("type", "text")
            page = r.get("page", 1)
            text = r.get("text", "")
        nodes[rid] = RegionNode(
            id=rid, bbox=bbox, region_type=rtype, page=page, text=text
        )

    # Containment: a is parent of b if a.bbox contains b.bbox and a is larger
    for a_id, a in nodes.items():
        for b_id, b in nodes.items():
            if a_id == b_id:
                continue
            if _contains(a.bbox, b.bbox) and _area(a.bbox) > _area(b.bbox) * 1.2:
                # b is inside a — but only assign direct parent if no closer
                if b.parent_id is None or _area(nodes[b.parent_id].bbox) > _area(
                    a.bbox
                ):
                    b.parent_id = a_id

    # Set child_ids based on parent_id
    for rid, n in nodes.items():
        if n.parent_id and n.parent_id in nodes:
            nodes[n.parent_id].child_ids.append(rid)

    # Adjacency: neighbors if boxes are within threshold distance
    threshold = 30.0  # PDF points
    for a_id, a in nodes.items():
        for b_id, b in nodes.items():
            if a_id == b_id or b_id in a.neighbor_ids:
                continue
            if _adjacent(a.bbox, b.bbox, threshold):
                a.neighbor_ids.append(b_id)
                b.neighbor_ids.append(a_id)

    return nodes


def _area(bbox: tuple) -> float:
    if not bbox or len(bbox) != 4:
        return 0.0
    return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])


def _contains(outer: tuple, inner: tuple) -> bool:
    if not outer or not inner or len(outer) != 4 or len(inner) != 4:
        return False
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _adjacent(a: tuple, b: tuple, threshold: float) -> bool:
    """Two bboxes are adjacent if their edges are within threshold."""
    if not a or not b:
        return False
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    # Check overlap or nearness on each axis
    h_gap = max(0, max(ax0, bx0) - min(ax1, bx1))
    v_gap = max(0, max(ay0, by0) - min(ay1, by1))
    return h_gap < threshold and v_gap < threshold


def find_caption_for_figure(figure_id: str, nodes: dict) -> Optional[str]:
    """Find caption region most likely associated with a figure.

    Heuristic: closest "caption" region below the figure.
    """
    fig = nodes.get(figure_id)
    if not fig:
        return None
    candidates = []
    for rid, n in nodes.items():
        if n.region_type != "caption" or n.page != fig.page:
            continue
        # Caption must be below figure
        if n.bbox[1] < fig.bbox[3]:
            continue
        # Distance from figure bottom to caption top
        dist = n.bbox[1] - fig.bbox[3]
        if dist < 50:  # within 50 pts below
            candidates.append((dist, rid))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


__all__ = [
    "RegionNode",
    "build_region_graph",
    "find_caption_for_figure",
]
