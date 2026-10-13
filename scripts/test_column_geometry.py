"""Column-geometry contracts (layout_gate) — pin registry for structural fixes.

Pin (fixture/invariant, 2026-08-18): left_edge_column_sharpness on a
single-cluster page (all left-edges aligned — left-aligned prose) must
return (1.0, None), not crash. Crash origin: headers_footers bench
fixture 06f8837dfd6ffb7b8b93427f047f0aa456af2e23_page_1 (ValueError at
the top2 2-unpack; 5 fixtures across 3 categories died with ERR in
run 0026).
"""

import sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PDF_OCR_NO_SKILL_VENV"] = "1"

from scientific_parser.routing.layout_gate import (
    left_edge_column_sharpness,
    column_sharp,
)


def test_single_cluster_page_returns_none():
    """All-aligned left edges: one cluster, no second column, no crash."""
    lines = [(90.0, y, 500.0, y + 10) for y in range(0, 400, 20)]
    cov, col2 = left_edge_column_sharpness(lines)
    assert cov == 1.0
    assert col2 is None
    assert column_sharp(lines) is False


def test_two_cluster_paper_returns_right_column():
    """Genuine 2-column page: col2 is the left edge of the right cluster."""
    left_col = [(90.0, y, 290.0, y + 10) for y in range(0, 400, 20)]
    right_col = [(310.0, y, 510.0, y + 10) for y in range(0, 400, 20)]
    _, col2 = left_edge_column_sharpness(left_col + right_col)
    assert col2 == 310.0
    assert column_sharp(left_col + right_col) is True


def test_below_min_lines_returns_none():
    """Fewer than 4 lines: no column decision (coverage 1.0, col None)."""
    lines = [(90.0, 0, 200.0, 10), (90.0, 20, 200.0, 30)]
    cov, col2 = left_edge_column_sharpness(lines)
    assert cov == 1.0
    assert col2 is None


def test_indent_tolerant_right_column():
    """Hanging indents near the right column still count as column-sharp.

    Regression guard for the indent-tolerant coverage (2026-08-18):
    reference entries 20-60pt right of the column edge must not drop
    coverage below the gate.
    """
    deltas = [0, 35, 55, 0, 40, 0, 25, 0]
    right_varied = [
        (310.0 + deltas[i % len(deltas)], y, 510.0, y + 10)
        for i, y in enumerate(range(0, 400, 20))
    ]
    left_col = [(90.0, y, 290.0, y + 10) for y in range(0, 400, 20)]
    _, col2 = left_edge_column_sharpness(left_col + right_varied)
    assert col2 is not None
