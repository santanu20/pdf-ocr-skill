"""P2.1 recover_header_band — pure-function unit tests.

Fixture words mimic fitz get_text("words") tuples: (x0, y0, x1, y1, text).
"""

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "scientific_parser"))

from scientific_parser.perception.tables.detector import recover_header_band  # noqa: E402

# 4-column grid, data-like first row (numeric majority)
DATA_ROWS = [
    ["1", "2", "3", "4"],
    ["5", "6", "7", "8"],
    ["9", "10", "11", "12"],
]
BBOX = (100.0, 200.0, 300.0, 350.0)


def _w(x0, y0, x1, text, y1=None):
    return (x0, y0, x1, y1 if y1 is not None else y0 + 8, text)


def test_recovers_closest_header_line_above_bbox():
    words = [
        _w(100, 150, 120, "Treatment"),
        _w(150, 150, 170, "Day"),
        _w(200, 150, 220, "Week"),
        _w(250, 150, 270, "Month"),
    ]
    rows, rw = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw is not None
    assert len(rows) == 4
    assert rows[0] == ["Treatment", "Day", "Week", "Month"]
    assert rows[1:] == DATA_ROWS


def test_no_recovery_when_first_row_is_header_like():
    rows = [["Alpha", "Beta", "Gamma", "Delta"], *DATA_ROWS[1:]]
    words = [_w(100, 150, 120, "Treatment")]
    out, rw = recover_header_band(rows, BBOX, words)
    assert rw is None
    assert out is rows


def test_caption_line_rejected():
    words = [_w(100, 150, 180, "Table"), _w(182, 150, 190, "2"), _w(200, 150, 220, "Day")]
    _out, rw = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw is None


def test_pure_numeric_line_rejected():
    words = [_w(100, 150, 120, "1"), _w(150, 150, 170, "2"), _w(200, 150, 220, "3")]
    _out, rw = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw is None


def test_prose_sentence_rejected():
    words = [
        _w(100, 150, 190, "The"),
        _w(195, 150, 260, "following"),
        _w(265, 150, 330, "values"),
        _w(335, 150, 400, "were"),
        _w(405, 150, 470, "measured"),
        _w(475, 150, 550, "during"),
        _w(555, 150, 620, "the"),
        _w(625, 150, 700, "study."),
    ]
    _out, rw = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw is None


def test_ragged_grid_rejected():
    rows = [["1", "2"], ["3", "4", "5", "6"]]
    _out, rw = recover_header_band(rows, BBOX, [_w(100, 150, 120, "Day")])
    assert rw is None


def test_band_window_respected():
    # header line 100pt above the bbox top — outside the 60pt band
    words = [_w(100, 100, 120, "Day"), _w(150, 100, 170, "Week")]
    _out, rw = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw is None


def test_two_line_stacked_header_prepends_both():
    words = [
        _w(100, 145, 220, "Divided"),
        _w(240, 145, 330, "attention"),
        _w(100, 158, 150, "Error"),
        _w(200, 158, 260, "Rate"),
        _w(270, 158, 330, "percent"),
    ]
    rows, rw = recover_header_band(DATA_ROWS, BBOX, words, max_lines=2)
    assert rw is not None and len(rw) == 2
    assert rows[0][1] == "Divided"  # xc 160 -> col 1
    assert rows[0][3] == "attention"  # xc 285 -> col 3
    assert rows[1][0] == "Error"


def test_column_assignment_by_x_center():
    # 4 cols over bbox 100-300 (50pt each): word at x-center 275 -> col 3
    words = [_w(260, 150, 290, "Month")]
    rows, rw = recover_header_band(
        [["a", "1", "2", "3"], ["b", "4", "5", "6"]], BBOX, words + [_w(100, 150, 140, "Label")]
    )
    assert rw is not None
    assert rows[0][3] == "Month"
    assert rows[0][0] == "Label"


def test_narrow_fragment_rejected():
    # line spans 10pt of a 200pt-wide table
    words = [_w(100, 150, 110, "Day")]
    _out, rw = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw is None


def test_idempotent_on_recovered_rows():
    words = [_w(100, 150, 120, "Day"), _w(190, 150, 215, "Week")]
    rows1, rw1 = recover_header_band(DATA_ROWS, BBOX, words)
    assert rw1 is not None
    rows2, rw2 = recover_header_band(rows1, BBOX, words)
    # first row now header-like — no second prepend
    assert rw2 is None
    assert rows2 is rows1


def test_duplicate_band_rejected():
    # 1a67d1114 shape: the year band sits above the bbox AND in row 0 —
    # prepending it would double the header and shatter the columns
    rows = [
        ["2015/16", "2016/17", "Total number", "listed 2016/"],
        ["Q1 Q2", "Q3", "Q4 2016/17", "2015/16"],
        ["1", "2", "3", "4"],
    ]
    words = [
        _w(100, 150, 140, "2015/16"),
        _w(150, 150, 190, "2016/17"),
        _w(200, 150, 230, "Total"),
        _w(235, 150, 265, "number"),
        _w(270, 150, 285, "of"),
        _w(290, 150, 330, "appeals"),
        _w(335, 150, 360, "listed"),
        _w(365, 150, 400, "2016/17"),
    ]
    out, rw = recover_header_band(rows, BBOX, words)
    assert rw is None
    assert out is rows


def test_year_header_row_not_datalike():
    # 1a67d1114 gold: cell 'Q3' right 'Q4' — this row IS the header;
    # year tokens must not classify it as data and demote it
    rows = [
        ["", "Q1", "Q2", "Q3"],
        ["% Allowed", "57", "57", "55"],
        ["% Remitted", "15", "15", "16"],
    ]
    words = [
        _w(100, 150, 140, "2015/16"),
        _w(150, 150, 190, "2016/17"),
        _w(200, 150, 230, "Total"),
        _w(235, 150, 265, "number"),
        _w(290, 150, 330, "appeals"),
    ]
    out, rw = recover_header_band(rows, BBOX, words)
    assert rw is None
    assert out is rows
