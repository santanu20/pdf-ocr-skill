"""SVG-first vector extraction (proposal item #20, #21).

Per architecture proposal:
  "One particularly strong idea: SVG-first extraction. Scientific PDFs often
   contain vector graphics. Before using a VLM: PDF → vector extraction →
   lines / polygons / paths / text / markers / axes.

   For plots and diagrams, this can be vastly more accurate than image OCR.
   Then VLM is used only for semantic interpretation."

Per proposal item #21:
  "Tables should also exploit PDF vectors. PDF → vector lines → candidate
   grid → PDF text positions → cell reconstruction."

Implementation:
  - PyMuPDF (fitz) provides vector path access: page.get_drawings()
  - Returns: list of paths (lines, rects, curves, fills)
  - For tables: detect horizontal+vertical line crossings = grid cells
  - For plots: detect axis lines + tick marks
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VectorPath:
    """One vector path from PDF."""

    kind: str  # "line" | "rect" | "curve" | "fill"
    points: list  # [(x,y), ...]
    bbox: tuple  # (x0, y0, x1, y1)
    stroke_color: Optional[tuple] = None  # (r,g,b) 0-1
    fill_color: Optional[tuple] = None
    width: float = 1.0
    closed: bool = False


@dataclass
class TableGrid:
    """Detected table grid from vector lines."""

    rows: list = field(default_factory=list)  # y-coords of horizontal lines
    cols: list = field(default_factory=list)  # x-coords of vertical lines
    bbox: tuple = (0, 0, 0, 0)
    cells: list = field(default_factory=list)  # [(x0,y0,x1,y1), ...]


@dataclass
class PlotAxes:
    """Detected plot axes from vector lines."""

    x_axis: Optional[tuple] = None  # (x0, y0, x1, y1)
    y_axis: Optional[tuple] = None
    tick_marks: list = field(default_factory=list)  # [(x, y, axis_label)]
    bbox: tuple = (0, 0, 0, 0)


def extract_vector_paths(page) -> list[VectorPath]:
    """Extract all vector paths from a PyMuPDF page.

    Args:
        page: pymupdf.Page object (open via pymupdf.open(...)[i])

    Returns:
        List of VectorPath objects.
    """
    paths_out = []
    try:
        drawings = page.get_drawings()
    except Exception:
        return []
    for d in drawings:
        # d is dict with: items (list of tuples), rect, fill, color, width
        items = d.get("items", [])
        bbox = d.get("rect")
        bbox_t = (bbox.x0, bbox.y0, bbox.x1, bbox.y1) if bbox else (0, 0, 0, 0)
        stroke = d.get("color")
        fill = d.get("fill")
        width = d.get("width", 1.0)
        for item in items:
            # item is tuple like ("l", p1, p2) for line, ("re", rect) for rect
            op = item[0]
            if op == "l":  # line
                p1, p2 = item[1], item[2]
                paths_out.append(
                    VectorPath(
                        kind="line",
                        points=[(p1.x, p1.y), (p2.x, p2.y)],
                        bbox=(
                            min(p1.x, p2.x),
                            min(p1.y, p2.y),
                            max(p1.x, p2.x),
                            max(p1.y, p2.y),
                        ),
                        stroke_color=stroke,
                        width=width,
                    )
                )
            elif op == "re":  # rectangle
                r = item[1]
                paths_out.append(
                    VectorPath(
                        kind="rect",
                        points=[(r.x0, r.y0), (r.x1, r.y1)],
                        bbox=(r.x0, r.y0, r.x1, r.y1),
                        stroke_color=stroke,
                        fill_color=fill,
                        width=width,
                        closed=True,
                    )
                )
            elif op == "c":  # Bezier curve
                p1, p2, p3, p4 = item[1], item[2], item[3], item[4]
                pts = [(p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                paths_out.append(
                    VectorPath(
                        kind="curve",
                        points=pts,
                        bbox=(min(xs), min(ys), max(xs), max(ys)),
                        stroke_color=stroke,
                        width=width,
                    )
                )
            elif op == "qu":  # quad
                pass  # skip
    return paths_out


def detect_table_grid(
    paths: list[VectorPath], min_lines: int = 3
) -> Optional[TableGrid]:
    """Detect table grid from horizontal + vertical lines.

    A table requires:
      - ≥3 horizontal lines (rows: top + middle + bottom)
      - ≥2 vertical lines (cols: left + right)
      - Intersecting (form cells)
    """
    h_lines = []  # (y, x_start, x_end)
    v_lines = []  # (x, y_start, y_end)
    for p in paths:
        if p.kind != "line":
            continue
        (x1, y1), (x2, y2) = p.points
        # Horizontal: y1≈y2
        if abs(y1 - y2) < 1.0 and abs(x2 - x1) > 5.0:
            h_lines.append((y1, min(x1, x2), max(x1, x2)))
        # Vertical: x1≈x2
        elif abs(x1 - x2) < 1.0 and abs(y2 - y1) > 5.0:
            v_lines.append((x1, min(y1, y2), max(y1, y2)))

    if len(h_lines) < min_lines or len(v_lines) < 2:
        return None

    # Group horizontal lines by Y (cluster within 3pt)
    h_lines.sort()
    h_clusters = []
    for y, x0, x1 in h_lines:
        if h_clusters and abs(h_clusters[-1][0] - y) < 3.0:
            # Merge with last cluster
            cy, cx0, cx1 = h_clusters[-1]
            h_clusters[-1] = (cy, min(cx0, x0), max(cx1, x1))
        else:
            h_clusters.append((y, x0, x1))

    # Same for vertical
    v_lines.sort()
    v_clusters = []
    for x, y0, y1 in v_lines:
        if v_clusters and abs(v_clusters[-1][0] - x) < 3.0:
            cx, cy0, cy1 = v_clusters[-1]
            v_clusters[-1] = (cx, min(cy0, y0), max(cy1, y1))
        else:
            v_clusters.append((x, y0, y1))

    if len(h_clusters) < min_lines or len(v_clusters) < 2:
        return None

    # Build cells
    h_ys = sorted(c[0] for c in h_clusters)
    v_xs = sorted(c[0] for c in v_clusters)
    cells = []
    for i in range(len(h_ys) - 1):
        for j in range(len(v_xs) - 1):
            cells.append((v_xs[j], h_ys[i], v_xs[j + 1], h_ys[i + 1]))

    return TableGrid(
        rows=h_ys,
        cols=v_xs,
        bbox=(min(v_xs), min(h_ys), max(v_xs), max(h_ys)),
        cells=cells,
    )


def detect_plot_axes(paths: list[VectorPath]) -> Optional[PlotAxes]:
    """Detect plot axes from vector paths.

    Heuristic: find longest horizontal line (X-axis) + longest vertical (Y-axis)
    that share a corner point. Tick marks = short perpendicular lines.
    """
    h_lines = []
    v_lines = []
    for p in paths:
        if p.kind != "line":
            continue
        (x1, y1), (x2, y2) = p.points
        length = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if length < 10:
            continue  # too short for an axis
        if abs(y1 - y2) < 2.0:
            h_lines.append((length, x1, y1, x2, y2))
        elif abs(x1 - x2) < 2.0:
            v_lines.append((length, x1, y1, x2, y2))

    if not h_lines or not v_lines:
        return None

    h_lines.sort(reverse=True)  # longest first
    v_lines.sort(reverse=True)
    longest_h = h_lines[0]
    longest_v = v_lines[0]

    return PlotAxes(
        x_axis=(longest_h[1], longest_h[2], longest_h[3], longest_h[4]),
        y_axis=(longest_v[1], longest_v[2], longest_v[3], longest_v[4]),
        tick_marks=[],  # TODO: extract tick marks
    )


__all__ = [
    "VectorPath",
    "TableGrid",
    "PlotAxes",
    "extract_vector_paths",
    "detect_table_grid",
    "detect_plot_axes",
]
