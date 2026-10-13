"""Unit tests for the layout skip gate (reviewer #15).

Run: pytest scripts/scientific_parser/tests/test_layout_gate.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.routing.layout_gate import (
    PageProfile,
    count_columns,
    should_run_layout,
    should_skip_layout,
)


def _simple_text(chars=2000):
    return PageProfile(text_chars=chars, n_columns=1)


# === simple pages skip layout ==================================== #


def test_simple_text_page_skips_layout():
    assert should_skip_layout(_simple_text()) is True
    assert should_run_layout(_simple_text()) is False


def test_long_single_column_prose_skips():
    assert should_skip_layout(PageProfile(text_chars=5000, n_columns=1)) is True


# === hard signals force layout =================================== #


def test_multi_column_runs_layout():
    assert should_run_layout(PageProfile(text_chars=5000, n_columns=2)) is True


def test_table_runs_layout():
    assert should_run_layout(PageProfile(text_chars=5000, has_tables=True)) is True


def test_equation_runs_layout():
    assert should_run_layout(PageProfile(text_chars=5000, has_equations=True)) is True


def test_figure_runs_layout():
    assert should_run_layout(PageProfile(text_chars=5000, has_figures=True)) is True


def test_high_image_coverage_runs_layout():
    assert should_run_layout(PageProfile(text_chars=5000, image_coverage=0.6)) is True


# === conservative on ambiguity =================================== #


def test_thin_text_runs_layout_even_single_column():
    # < 200 chars is suspicious (scanned/garbled) -> run layout
    assert should_run_layout(PageProfile(text_chars=50, n_columns=1)) is True


def test_low_text_density_runs_layout():
    assert should_run_layout(PageProfile(text_chars=5000, text_density_ok=False)) is True


def test_default_profile_is_conservative():
    # default: text_chars=0 -> below threshold -> run layout (safe default)
    assert should_run_layout(PageProfile()) is True


# === skip/run are inverses ======================================= #


def test_skip_is_inverse_of_run():
    for p in [
        _simple_text(),
        PageProfile(text_chars=5000, n_columns=2),
        PageProfile(text_chars=100, has_tables=True),
    ]:
        assert should_skip_layout(p) is (not should_run_layout(p))


# === cost-saving property ======================================== #


def test_document_of_simple_pages_needs_zero_layout_runs():
    # Reviewer's thesis: layout model only where it adds value.
    pages = [_simple_text(3000) for _ in range(100)]
    n_layout_runs = sum(1 for p in pages if should_run_layout(p))
    assert n_layout_runs == 0


# === structure_known provenance (scanned / OCR text) ============== #


def test_unknown_structure_forces_layout_even_single_column():
    # OCR-only / scanned page: flat text cannot prove simplicity.
    p = PageProfile(text_chars=5000, n_columns=1, structure_known=False)
    assert should_run_layout(p) is True


def test_unknown_structure_forces_layout_regardless_of_markers():
    # Even "looks simple" by every other measure, unknown => run.
    p = PageProfile(
        text_chars=5000,
        n_columns=1,
        has_tables=False,
        has_equations=False,
        structure_known=False,
    )
    assert should_run_layout(p) is True


def test_known_simple_native_text_still_skips():
    # Geometry-confirmed single column + native text => skip (fast path).
    p = PageProfile(text_chars=5000, n_columns=1, structure_known=True)
    assert should_skip_layout(p) is True


# === count_columns geometry detection ============================== #

# Block tuple shape mirrors PyMuPDF get_text("blocks"): (x0,y0,x1,y1,text,...)
_W = 612.0  # US Letter width in PDF points


def _blk(x0, y0, x1, y1, txt):
    return (x0, y0, x1, y1, txt, 0, 0)


def test_count_columns_single_column_returns_1():
    blocks = [
        _blk(50, 100, 560, 150, "a full width line of body text here"),
        _blk(50, 160, 560, 210, "another full width line of body text"),
        _blk(50, 220, 560, 270, "yet another full width line of text"),
    ]
    assert count_columns(blocks, _W) == 1


def test_count_columns_two_column_returns_2():
    # Classic 2-column academic page: parallel columns at overlapping y.
    blocks = [
        _blk(50, 100, 250, 200, "left column paragraph one text body"),
        _blk(50, 210, 250, 310, "left column paragraph two text body"),
        _blk(320, 100, 560, 200, "right column paragraph one text body"),
        _blk(320, 210, 560, 310, "right column paragraph two text body"),
    ]
    assert count_columns(blocks, _W) == 2


def test_count_columns_empty_blocks_returns_1():
    assert count_columns([], _W) == 1


def test_count_columns_sparse_blocks_returns_1():
    # Too few substantial blocks to infer parallel columns.
    blocks = [
        _blk(50, 100, 250, 200, "only one short line"),
        _blk(320, 210, 560, 310, "and another short line"),
    ]
    assert count_columns(blocks, _W) == 1


def test_count_columns_short_text_ignored():
    # Short strings (page numbers, isolated labels) must not count.
    blocks = [
        _blk(50, 100, 250, 110, "12"),
        _blk(320, 100, 560, 110, "34"),
        _blk(50, 130, 250, 140, "56"),
        _blk(320, 130, 560, 140, "78"),
    ]
    assert count_columns(blocks, _W) == 1


def test_count_columns_non_overlapping_y_returns_1():
    # Left block at top, right block at bottom — sequential, not parallel.
    blocks = [
        _blk(50, 100, 250, 200, "left column paragraph one text body"),
        _blk(50, 210, 250, 310, "left column paragraph two text body"),
        _blk(320, 400, 560, 500, "right column paragraph one text body"),
        _blk(320, 510, 560, 610, "right column paragraph two text body"),
    ]
    assert count_columns(blocks, _W) == 1
