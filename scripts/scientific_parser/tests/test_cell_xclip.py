"""EXP-3 cell x-clip: split_jammed_cells unit tests (033c1efd jam class)."""

import sys
from pathlib import Path

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.perception.tables.detector import (
    _straddle_split_table,
    split_jammed_cells,
)

# Synthetic geometry: two columns (A: x 50-90, B: x 110-150), three clean
# sibling rows + one jam row whose cell-A words straddle the boundary.
# Word tuple: (x0, y0, x1, y1, text, block, line, word)
B = 100.0  # consensus boundary


def _w(x0, x1, y, text):
    return (x0, y, x1, y + 8.0, text, 0, 0, 0)


WORDS = [
    # row 1 (y=50): clean   A: alpha | B: 111
    _w(50, 70, 50, "alpha"),
    _w(110, 130, 50, "111"),
    # row 2 (y=70): clean   A: beta | B: 222
    _w(50, 70, 70, "beta"),
    _w(110, 130, 70, "222"),
    # row 3 (y=90): JAM     A: '9 M-PSI/06' (9 left, M-PSI/06 right) | B: ''
    _w(50, 58, 90, "9"),
    _w(110, 150, 90, "M-PSI/06"),
    # row 4 (y=110): clean  A: gamma | B: 444
    _w(50, 76, 110, "gamma"),
    _w(110, 130, 110, "444"),
]

ROWS_JAM = [
    ["alpha", "111"],
    ["beta", "222"],
    ["9 M-PSI/06", ""],
    ["gamma", "444"],
]


def test_straddle_split_jam_moves_right_words():
    out = _straddle_split_table(list(WORDS), [list(r) for r in ROWS_JAM])
    assert out[2] == ["9", "M-PSI/06"]
    assert out[0] == ["alpha", "111"]  # untouched rows stay intact


def test_straddle_split_left_move_into_empty_left_neighbor():
    # 033c1efd pg5 class: '9 M-PSI/06' BOTH in the LAST col, empty LEFT
    # neighbor — ok siblings carve '6' | 'SECS-P/10'. '9' (right edge
    # 98 < boundary 100) moves LEFT.
    words = [
        _w(50, 90, 50, "alpha"),
        _w(90, 98, 50, "6"),
        _w(110, 150, 50, "SECS-P/10"),
        _w(50, 90, 70, "beta"),
        _w(90, 98, 70, "9"),
        _w(110, 150, 70, "SECS-P/07"),
        _w(90, 98, 90, "9"),
        _w(110, 150, 90, "M-PSI/06"),
        _w(50, 90, 110, "gamma"),
        _w(90, 98, 110, "9"),
        _w(110, 150, 110, "SECS-P/01"),
    ]
    rows = [
        ["alpha", "6", "SECS-P/10"],
        ["beta", "9", "SECS-P/07"],
        ["Conflict Mgmt", "", "9 M-PSI/06"],
        ["gamma", "9", "SECS-P/01"],
    ]
    out = _straddle_split_table(words, rows)
    assert out[2] == ["Conflict Mgmt", "9", "M-PSI/06"]


def test_straddle_split_no_empty_neighbor_no_split():
    rows = [
        ["alpha", "111"],
        ["beta", "222"],
        ["9 M-PSI/06", "occupied"],  # right neighbor filled
        ["gamma", "444"],
    ]
    out = _straddle_split_table(list(WORDS), rows)
    assert out[2] == ["9 M-PSI/06", "occupied"]


def test_straddle_split_no_consensus_untouched():
    # 4 rows, only 2 vote (jam row's col-1 empty; one row desyncs on a
    # token absent from the words) — 2 votes < 0.6*4 = no boundary forms.
    words = [
        _w(50, 70, 50, "alpha"),
        _w(110, 130, 50, "111"),
        _w(50, 58, 90, "9"),
        _w(110, 150, 90, "M-PSI/06"),
        _w(50, 76, 110, "x"),
        _w(110, 130, 110, "y"),
    ]
    rows = [
        ["alpha", "111"],
        ["9 M-PSI/06", ""],
        ["missingtoken", "y"],
        ["x", "y"],
    ]
    out = _straddle_split_table(words, rows)
    assert out[1] == ["9 M-PSI/06", ""]


def test_straddle_split_single_word_cell_untouched():
    # A cell whose words do NOT straddle (all left of boundary) stays.
    rows = [["alpha", "111"], ["beta", "222"], ["gamma delta", ""], ["x1", "444"]]
    words = [
        _w(50, 70, 50, "alpha"),
        _w(110, 130, 50, "111"),
        _w(50, 70, 70, "beta"),
        _w(110, 130, 70, "222"),
        _w(50, 62, 90, "gamma"),
        _w(64, 84, 90, "delta"),
        _w(110, 130, 110, "444"),
    ]
    out = _straddle_split_table(words, rows)
    assert out[2] == ["gamma delta", ""]


def test_split_jammed_cells_on_pdf(tmp_path):
    pdf_path = tmp_path / "jam.pdf"
    doc = fitz.open()
    page = doc.new_page(width=300, height=400)
    # insert words so fitz 'words' geometry matches the synthetic layout
    page.insert_text((50, 58), "alpha", fontsize=8)
    page.insert_text((110, 58), "111", fontsize=8)
    page.insert_text((50, 78), "beta", fontsize=8)
    page.insert_text((110, 78), "222", fontsize=8)
    page.insert_text((50, 98), "9", fontsize=8)
    page.insert_text((110, 98), "M-PSI/06", fontsize=8)
    page.insert_text((50, 118), "gamma", fontsize=8)
    page.insert_text((110, 118), "444", fontsize=8)
    doc.save(str(pdf_path))
    doc.close()

    tables = [
        {
            "bbox": (40.0, 40.0, 200.0, 130.0),
            "rows": [list(r) for r in ROWS_JAM],
            "markdown": "| old | |",
            "strategy": "positional",
        }
    ]
    out = split_jammed_cells(str(pdf_path), 1, tables)
    assert out is tables
    assert out[0]["rows"][2] == ["9", "M-PSI/06"]


def test_split_jammed_cells_missing_pdf_fail_safe(tmp_path):
    tables = [{"bbox": (0, 0, 10, 10), "rows": [["a"]], "markdown": "| a |"}]
    out = split_jammed_cells(str(tmp_path / "nope.pdf"), 1, tables)
    assert out is tables  # fail-soft: unchanged


def test_split_jammed_cells_empty_table_list(tmp_path):
    assert split_jammed_cells("whatever.pdf", 1, []) == []


def test_split_preserves_scrub_invariant():
    # Split output must contain zero control glyphs (EXP-1.5 scrub stays
    # meaningful post-split; tokens are rebuilt from clean words).
    out = _straddle_split_table(list(WORDS), [list(r) for r in ROWS_JAM])
    joined = " ".join(c for r in out for c in r)
    assert not any(ord(ch) < 0x20 and ch not in "\t\n\r" for ch in joined)
