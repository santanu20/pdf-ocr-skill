"""XY-Cut++ reading order — faithful Python port of ODL's Java sorter.

Source: opendataloader-pdf-core
processors/readingorder/XYCutPlusPlusSorter.java (Hancom Inc.,
Apache-2.0), based on arXiv:2504.10258. Ported line-faithfully; the
engine's other module ``xy_cut.py`` is an earlier paper-inspired sketch,
not this port.

Algorithm (Java line refs):
  1. Pre-mask cross-layout elements: width >= beta * maxWidth AND
     horizontal overlap (ratio >= 0.1 vs smaller box) with >= 2 others
     (:146). NOTE: default beta=2.0 makes this branch unreachable —
     ODL ships cross-layout detection effectively disabled (their own
     comment at :49).
  2. Density ratio = content area / region area; > threshold selects
     preferHorizontalFirst (:121). NOTE: that flag is passed into
     recursiveSegment but never read there — cut direction is decided
     purely by gap size (:346-356). Preserved as-is for fidelity.
  3. Recursive segmentation: projection/edge-based cuts in both axes,
     validity = gap >= 5.0pt, larger gap wins, split by box CENTER,
     single-group results fall back to Y-then-X (:331-373). Vertical
     cut retries without narrow (<10% region width) outliers that can
     bridge column gaps, e.g. page numbers (:406-444).
  4. Cross-layout merge back by topY interleaving (:590).

Coordinate convention: fitz-style boxes (x0, y0, x1, y1), y-down,
y0 < y1. The Java sorter is bottom-left/Y-up, so this port negates y
at the boundary (topY_up = -y0, bottomY_up = -y1); negation is
monotonic, preserving every comparison and gap magnitude.

Input items: any objects with a ``bbox`` attribute or ``["bbox"]`` key
holding a 4-tuple. Output: NEW list of the same objects in reading
order. Deterministic, pure stdlib, no mutation of inputs.
"""

from __future__ import annotations

DEFAULT_BETA = 2.0
DEFAULT_DENSITY_THRESHOLD = 0.9
OVERLAP_THRESHOLD = 0.1
MIN_OVERLAP_COUNT = 2
MIN_GAP_THRESHOLD = 5.0
NARROW_ELEMENT_WIDTH_RATIO = 0.1


class _Box:
    """Y-up box (veraPDF convention) derived from a fitz-style bbox."""

    __slots__ = ("left", "right", "top", "bottom")

    def __init__(self, bbox):
        x0, y0, x1, y1 = bbox
        self.left = float(x0)
        self.right = float(x1)
        self.top = -float(y0)
        self.bottom = -float(y1)

    @property
    def width(self):
        return self.right - self.left

    @property
    def area(self):
        return max(0.0, self.right - self.left) * max(0.0, self.top - self.bottom)

    def union_extend(self, other):
        self.left = min(self.left, other.left)
        self.right = max(self.right, other.right)
        self.top = max(self.top, other.top)
        self.bottom = min(self.bottom, other.bottom)

    def is_empty(self):
        return self.right <= self.left or self.top <= self.bottom


def _get_bbox(item):
    if hasattr(item, "bbox"):
        return item.bbox
    if isinstance(item, dict) and "bbox" in item:
        return item["bbox"]
    return None


def sort_reading_order(items, beta=DEFAULT_BETA, density_threshold=DEFAULT_DENSITY_THRESHOLD):
    """Return items in XY-Cut++ reading order (Java sort(), :94)."""
    valid = [it for it in items if it is not None and _get_bbox(it) is not None]
    if len(valid) <= 1:
        return list(valid)

    cross = _identify_cross_layout(valid, beta)
    if cross:
        remaining = [it for it in valid if id(it) not in {id(c) for c in cross}]
    else:
        remaining = valid

    if not remaining:
        return _sort_by_y_then_x(valid)

    density_ratio = _compute_density_ratio(remaining)
    prefer_horizontal_first = density_ratio > density_threshold

    sorted_main = _recursive_segment(remaining, prefer_horizontal_first)
    return _merge_cross_layout(sorted_main, cross)


# Phase 1: cross-layout detection (:146-186)


def _identify_cross_layout(items, beta):
    if len(items) < 3:
        return []
    boxes = {id(it): _Box(_get_bbox(it)) for it in items}
    max_width = max(b.width for b in boxes.values())
    threshold = beta * max_width
    cross = []
    for it in items:
        box = boxes[id(it)]
        if box.width >= threshold and _has_minimum_overlaps(
            it, items, box, boxes, MIN_OVERLAP_COUNT
        ):
            cross.append(it)
    return cross


def _has_minimum_overlaps(element, items, element_box, boxes, min_count):
    overlap_count = 0
    for other in items:
        if other is element:
            continue
        if _horizontal_overlap_ratio(element_box, boxes[id(other)]) >= OVERLAP_THRESHOLD:
            overlap_count += 1
            if overlap_count >= min_count:
                return True
    return False


def _horizontal_overlap_ratio(box1, box2):
    overlap_left = max(box1.left, box2.left)
    overlap_right = min(box1.right, box2.right)
    overlap_width = overlap_right - overlap_left
    if overlap_width <= 0:
        return 0.0
    smaller_width = min(box1.width, box2.width)
    return overlap_width / smaller_width if smaller_width > 0 else 0.0


# Phase 2: density ratio (:260-278)


def _compute_density_ratio(items):
    if not items:
        return 1.0
    region = _region_bounds(items)
    if region is None or region.area <= 0:
        return 1.0
    content_area = sum(_Box(_get_bbox(it)).area for it in items)
    return min(1.0, content_area / region.area)


def _region_bounds(items):
    boxes = [_Box(_get_bbox(it)) for it in items]
    if not boxes:
        return None
    bounds = _Box((boxes[0].left, -boxes[0].top, boxes[0].right, -boxes[0].bottom))
    for b in boxes[1:]:
        bounds.union_extend(b)
    return None if bounds.is_empty() else bounds


# Phase 3: recursive segmentation (:331-578)


def _recursive_segment(items, prefer_horizontal_first):
    # NOTE: prefer_horizontal_first is unused upstream (Java :329 param,
    # never read); kept for signature fidelity.
    if len(items) <= 1:
        return list(items)

    horizontal_cut = _best_horizontal_cut(items)
    vertical_cut = _best_vertical_cut(items)

    has_horizontal = horizontal_cut[1] >= MIN_GAP_THRESHOLD
    has_vertical = vertical_cut[1] >= MIN_GAP_THRESHOLD

    if has_horizontal and has_vertical:
        use_horizontal = horizontal_cut[1] > vertical_cut[1]
    elif has_horizontal:
        use_horizontal = True
    elif has_vertical:
        use_horizontal = False
    else:
        return _sort_by_y_then_x(items)

    if use_horizontal:
        groups = _split_by_horizontal_cut(items, horizontal_cut[0])
    else:
        groups = _split_by_vertical_cut(items, vertical_cut[0])

    if len(groups) <= 1:
        return _sort_by_y_then_x(items)

    result = []
    for group in groups:
        result.extend(_recursive_segment(group, prefer_horizontal_first))
    return result


def _center_y_up(bbox):
    x0, y0, x1, y1 = bbox
    return -(y0 + y1) / 2.0


def _center_x(bbox):
    return (bbox[0] + bbox[2]) / 2.0


def _best_vertical_cut(items):
    """Largest x-gap via edge sweep, with narrow-outlier retry (:406)."""
    if len(items) < 2:
        return (0.0, 0.0)
    edge_cut = _vertical_cut_by_edges(items)
    if edge_cut[1] >= MIN_GAP_THRESHOLD:
        return edge_cut
    if len(items) >= 3:
        region = _region_bounds(items)
        if region is not None:
            narrow_threshold = region.width * NARROW_ELEMENT_WIDTH_RATIO
            filtered = [it for it in items if _Box(_get_bbox(it)).width >= narrow_threshold]
            if 2 <= len(filtered) < len(items):
                filtered_cut = _vertical_cut_by_edges(filtered)
                if filtered_cut[1] > edge_cut[1] and filtered_cut[1] >= MIN_GAP_THRESHOLD:
                    return filtered_cut
    return edge_cut


def _vertical_cut_by_edges(items):
    sorted_items = sorted(items, key=lambda it: (_get_bbox(it)[0], _get_bbox(it)[2]))
    largest_gap = 0.0
    cut_position = 0.0
    prev_right = None
    for it in sorted_items:
        bbox = _get_bbox(it)
        left, right = float(bbox[0]), float(bbox[2])
        if prev_right is not None and left > prev_right:
            gap = left - prev_right
            if gap > largest_gap:
                largest_gap = gap
                cut_position = (prev_right + left) / 2.0
        prev_right = right if prev_right is None else max(prev_right, right)
    return (cut_position, largest_gap)


def _best_horizontal_cut(items):
    """Largest y-gap via top-down edge sweep (Y-up; :484)."""
    if len(items) < 2:
        return (0.0, 0.0)
    # sort top→bottom: topY_up desc, then bottomY_up desc
    sorted_items = sorted(
        items,
        key=lambda it: (-_Box(_get_bbox(it)).top, -_Box(_get_bbox(it)).bottom),
    )
    largest_gap = 0.0
    cut_position = 0.0
    prev_bottom = None
    for it in sorted_items:
        box = _Box(_get_bbox(it))
        top, bottom = box.top, box.bottom
        if prev_bottom is not None and prev_bottom > top:
            gap = prev_bottom - top
            if gap > largest_gap:
                largest_gap = gap
                cut_position = (prev_bottom + top) / 2.0
        prev_bottom = bottom if prev_bottom is None else min(prev_bottom, bottom)
    return (cut_position, largest_gap)


def _split_by_horizontal_cut(items, cut_y_up):
    above, below = [], []
    for it in items:
        (above if _center_y_up(_get_bbox(it)) > cut_y_up else below).append(it)
    groups = []
    if above:
        groups.append(above)
    if below:
        groups.append(below)
    return groups


def _split_by_vertical_cut(items, cut_x):
    left, right = [], []
    for it in items:
        (left if _center_x(_get_bbox(it)) < cut_x else right).append(it)
    groups = []
    if left:
        groups.append(left)
    if right:
        groups.append(right)
    return groups


# Phase 4: merge (:590-634) + utilities (:644-650)


def _merge_cross_layout(sorted_main, cross_elements):
    if not cross_elements:
        return sorted_main
    if not sorted_main:
        return _sort_by_y_then_x(cross_elements)
    sorted_cross = _sort_by_y_then_x(cross_elements)
    result = []
    main_index = cross_index = 0
    while main_index < len(sorted_main) or cross_index < len(sorted_cross):
        if cross_index >= len(sorted_cross):
            result.append(sorted_main[main_index])
            main_index += 1
        elif main_index >= len(sorted_main):
            result.append(sorted_cross[cross_index])
            cross_index += 1
        else:
            main_top = _Box(_get_bbox(sorted_main[main_index])).top
            cross_top = _Box(_get_bbox(sorted_cross[cross_index])).top
            if cross_top >= main_top:
                result.append(sorted_cross[cross_index])
                cross_index += 1
            else:
                result.append(sorted_main[main_index])
                main_index += 1
    return result


def _sort_by_y_then_x(items):
    return sorted(items, key=lambda it: (-_Box(_get_bbox(it)).top, _get_bbox(it)[0]))
