"""Pins for the ODL Java XY-Cut++ port (xy_cut_pp.py).

Scenarios ported from XYCutPlusPlusSorterTest.java (Apache-2.0 Hancom):
two-column, header/footer, cross-layout detection/merge, narrow-bridge
filter, recursion guards, center-side split guard. Boxes are fitz-style
(x0, y0, x1, y1), y-down, y0 < y1.
"""

import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import pytest

pp = importlib.import_module("scientific_parser.perception.layout.xy_cut_pp")


def blk(x0, y0, x1, y1, tag):
    return {"bbox": (x0, y0, x1, y1), "tag": tag}


def tags(sorted_items):
    return [it["tag"] for it in sorted_items]


# --- passthrough / edge cases (Java: null/empty/single) ---


def test_empty_and_single_passthrough():
    assert pp.sort_reading_order([]) == []
    one = blk(0, 0, 10, 10, "a")
    assert pp.sort_reading_order([one]) == [one]
    none_bbox = {"tag": "x"}
    assert pp.sort_reading_order([none_bbox]) == []


# --- single column top-to-bottom (Java: sort_singleColumn_topToBottom) ---


def test_single_column_top_to_bottom():
    items = [blk(50, 200 + i * 15, 300, 210 + i * 15, f"r{i}") for i in range(4)]
    assert tags(pp.sort_reading_order(items)) == ["r0", "r1", "r2", "r3"]


# --- two columns, left first (Java: sort_twoColumns_leftColumnFirst) ---


def test_two_columns_left_column_first():
    left = [blk(50, 100 + i * 15, 250, 110 + i * 15, f"L{i}") for i in range(3)]
    right = [blk(350, 100 + i * 15, 550, 110 + i * 15, f"R{i}") for i in range(3)]
    out = tags(pp.sort_reading_order(left + right))
    assert out == ["L0", "L1", "L2", "R0", "R1", "R2"]


# --- header above two columns (Java: sort_twoColumnsWithHeader) ---


def test_two_columns_with_header_first():
    header = blk(50, 20, 550, 40, "H")
    left = [blk(50, 100 + i * 15, 250, 110 + i * 15, f"L{i}") for i in range(2)]
    right = [blk(350, 100 + i * 15, 550, 110 + i * 15, f"R{i}") for i in range(2)]
    out = tags(pp.sort_reading_order([*right, header, *left]))
    assert out[0] == "H"
    assert out[1:] == ["L0", "L1", "R0", "R1"]


# --- header and footer positions (Java: sort_headerAndFooter) ---


def test_header_and_footer():
    header = blk(50, 20, 550, 35, "H")
    body = [blk(60, 100 + i * 15, 540, 110 + i * 15, f"B{i}") for i in range(3)]
    footer = blk(50, 760, 550, 775, "F")
    out = tags(pp.sort_reading_order([footer, *body, header]))
    assert out[0] == "H"
    assert out[-1] == "F"
    assert out[1:-1] == ["B0", "B1", "B2"]


# --- horizontal sections: bigger y-gap wins over x-gap ---


def test_horizontal_sections_y_gap_dominates():
    top_section = [blk(50, 50 + i * 10, 550, 58 + i * 10, f"T{i}") for i in range(2)]
    bottom_section = [blk(50, 400 + i * 10, 550, 408 + i * 10, f"B{i}") for i in range(2)]
    out = tags(pp.sort_reading_order([*bottom_section, *top_section]))
    assert out == ["T0", "T1", "B0", "B1"]


# --- cross-layout detection (Java: identifyCrossLayoutElements_*) ---


def test_cross_layout_wide_header_detected_at_low_beta():
    wide = blk(50, 20, 550, 40, "W")  # 500 wide
    cols = [
        blk(50 + c * 300, 100 + i * 15, 250 + c * 300, 110 + i * 15, f"C{c}{i}")
        for c in range(2)
        for i in range(2)
    ]
    cross = pp._identify_cross_layout([wide, *cols], beta=0.7)
    assert [it["tag"] for it in cross] == ["W"]


def test_cross_layout_narrow_not_detected():
    # narrow element can never reach beta*maxWidth; the two WIDE
    # mutually-overlapping ones ARE cross-layout (Java semantics)
    items = [blk(0, 0, 10, 10, "n"), blk(0, 20, 400, 30, "w1"), blk(0, 40, 400, 50, "w2")]
    cross = pp._identify_cross_layout(items, beta=0.7)
    assert sorted(it["tag"] for it in cross) == ["w1", "w2"]
    assert "n" not in [it["tag"] for it in cross]


def test_cross_layout_wide_without_overlaps_not_detected():
    # wide element horizontally disjoint from two narrow ones
    items = [
        blk(0, 0, 300, 10, "w"),
        blk(400, 50, 410, 60, "a"),
        blk(500, 100, 510, 110, "b"),
    ]
    assert pp._identify_cross_layout(items, beta=0.7) == []


# --- overlap ratio helper (Java: hasMinimumOverlaps / ratio) ---


def test_horizontal_overlap_ratio_relative_to_smaller():
    a = pp._Box((0, 0, 100, 10))
    b = pp._Box((90, 50, 120, 60))  # overlap 10, smaller width 30 -> 1/3
    assert abs(pp._horizontal_overlap_ratio(a, b) - 10.0 / 30.0) < 1e-9
    assert pp._horizontal_overlap_ratio(a, pp._Box((200, 0, 300, 10))) == 0.0


# --- density ratio (Java: computeDensityRatio_*) ---


def test_density_ratio_dense_vs_sparse():
    dense = [blk(0, 0, 100, 100, "d")]  # fills its region exactly
    sparse = [blk(0, 0, 10, 10, "s"), blk(90, 90, 100, 100, "s2")]
    assert abs(pp._compute_density_ratio(dense) - 1.0) < 1e-9
    assert pp._compute_density_ratio(sparse) < 0.05
    assert pp._compute_density_ratio([]) == 1.0


# --- split helpers (Java: splitByHorizontalCut / splitByVerticalCut) ---


def test_split_by_cuts_uses_centers():
    items = [blk(0, 0, 10, 10, "a"), blk(0, 20, 10, 30, "b")]
    # y-up centers: a -> -5, b -> -25; cut at -15 separates them
    above, below = pp._split_by_horizontal_cut(items, cut_y_up=-15.0)
    assert tags(above) == ["a"] and tags(below) == ["b"]
    groups = pp._split_by_vertical_cut(items, cut_x=5.5)
    assert [tags(g) for g in groups] == [["a", "b"]]  # empty sides drop (Java :567)


# --- narrow bridge page-number does not block column cut (Java: sort_twoColumnsWithNarrowBridge) ---


def test_narrow_bridge_does_not_merge_columns():
    left = [blk(50, 100 + i * 15, 250, 110 + i * 15, f"L{i}") for i in range(3)]
    right = [blk(350, 100 + i * 15, 550, 110 + i * 15, f"R{i}") for i in range(3)]
    # narrow page number sitting IN the column gap bridges the x-gap
    bridge = blk(298, 120, 305, 130, "P")
    out = tags(pp.sort_reading_order(left + right + [bridge]))
    # columns split despite the bridge; P lands between them at its band
    assert out == ["L0", "L1", "L2", "P", "R0", "R1", "R2"]


# --- recursion guards (Java: noInfiniteRecursion trio + issue179) ---


def test_centers_on_one_side_of_cut_no_infinite_recursion():
    # cut gap found by edge sweep, but every center lies left of it
    items = [
        blk(0, 0, 100, 10, "a"),
        blk(0, 20, 100, 30, "b"),
        blk(300, 25, 400, 35, "c"),  # center x=350, gap 200 at ~200
    ]
    out = pp.sort_reading_order(items)
    assert sorted(tags(out)) == ["a", "b", "c"]


def test_wide_and_narrow_no_infinite_recursion():
    items = [blk(i * 5, i * 5, i * 5 + 3, i * 5 + 4, f"n{i}") for i in range(12)]
    out = pp.sort_reading_order(items)
    assert len(out) == 12


def test_many_small_gaps_falls_back_to_reading_order():
    # all gaps below MIN_GAP_THRESHOLD -> y-then-x
    items = [blk(50, 40 + i * 4, 300, 44 + i * 4, f"r{i}") for i in range(5)]
    assert tags(pp.sort_reading_order(items)) == [f"r{i}" for i in range(5)]


# --- cross-layout merge at low beta (Java: mergeCrossLayoutElements_atTop) ---


def test_low_beta_cross_header_merged_first():
    header = blk(50, 20, 550, 40, "H")
    left = [blk(50, 100 + i * 15, 250, 110 + i * 15, f"L{i}") for i in range(2)]
    right = [blk(350, 100 + i * 15, 550, 110 + i * 15, f"R{i}") for i in range(2)]
    out = pp.sort_reading_order([header, *right, *left], beta=0.7)
    assert out[0]["tag"] == "H"
    assert tags(out[1:]) == ["L0", "L1", "R0", "R1"]


# --- determinism + input non-mutation ---


def test_deterministic_and_non_mutating():
    items = [blk(350, 100, 550, 115, "R0"), blk(50, 100, 250, 115, "L0"), blk(50, 20, 550, 40, "H")]
    snapshot = [dict(it, bbox=tuple(it["bbox"])) for it in items]
    out1 = pp.sort_reading_order(items)
    out2 = pp.sort_reading_order(items)
    assert [id(x) for x in out1] == [id(x) for x in out2]
    assert [it["bbox"] for it in items] == [s["bbox"] for s in snapshot]
