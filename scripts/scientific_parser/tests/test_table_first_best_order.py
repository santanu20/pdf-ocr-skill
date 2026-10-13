"""det59 pins: overlap-cluster best-first table emission ordering."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scientific_parser.postproc.text_pipeline import (  # noqa: E402
    _emit_table_blocks,
    _order_tables_first_best,
)


def _t(strategy, bbox, rows):
    return {"strategy": strategy, "bbox": bbox, "rows": rows, "markdown": "| x |"}


def test__overlap_cluster_best_first__lattice_beats_stream_and_positional():
    # odlbench 0180 shape: one physical table, three surviving views.
    stream = _t("camelot_stream", [66, 317, 541, 453], [["T"]] * 9)
    positional = _t("positional", [76, 443, 529, 465], [["a"], ["b"]])
    lattice = _t(
        "camelot_lattice",
        [71, 319, 541, 409],
        [["Version", "Date", "Change", "Sections"], ["1.0", "d1", "c1", "s1"]],
    )
    out = _order_tables_first_best([stream, positional, lattice])
    assert out[0] is lattice, "exact-gold lattice copy must emit first"
    assert {id(out[1]), id(out[2])} == {id(stream), id(positional)}


def test__non_overlapping_tables__keep_reading_order():
    top = _t("positional", [50, 100, 500, 200], [["a"], ["b"]])
    bottom = _t(
        "camelot_lattice",
        [50, 400, 500, 500],
        [["h1", "h2"], ["x", "y"]],
    )
    # Higher-quality table sits LOWER on the page: order must not flip.
    out = _order_tables_first_best([top, bottom])
    assert out == [top, bottom]


def test__no_bbox__anchors_to_preceding_slot():
    anchored = _t("camelot_lattice", [50, 100, 500, 200], [["h", "h"], ["x", "y"]])
    no_bbox = _t("positional", None, [["a"], ["b"]])
    out = _order_tables_first_best([anchored, no_bbox])
    assert out == [anchored, no_bbox]


def test__empty_and_single__passthrough():
    assert _order_tables_first_best([]) == []
    one = _t("positional", [0, 0, 10, 10], [["a"]])
    assert _order_tables_first_best([one]) == [one]


def test__empty_cell_fraction__tiebreak_within_same_strategy():
    # Same strategy, overlapping: fewer empty cells wins.
    sparse = _t("positional", [0, 0, 100, 100], [["a", ""], ["", ""], ["", ""]])
    dense = _t("positional", [10, 10, 90, 90], [["a", "b"], ["c", "d"]])
    out = _order_tables_first_best([sparse, dense])
    assert out[0] is dense


def test__det60__cluster_nonbest_loses_html_twin():
    # det60: overlap-cluster non-best is demoted to pipe-only emission
    # (TEDS forest fuses all html twins; duplicates pollute the score).
    best = _t("camelot_lattice", [70, 319, 541, 409], [["a", "b"], ["c", "d"]])
    twin = _t("camelot_stream", [66, 317, 541, 453], [["T", ""], ["a", "b"], ["c", "d"]])
    lone = _t("positional", [50, 700, 500, 800], [["x", "y"], ["1", "2"]])
    out = _order_tables_first_best([twin, best, lone])
    assert out[0] is best
    assert best.get("_no_html_twin") is None
    assert twin.get("_no_html_twin") is True
    assert lone.get("_no_html_twin") is None  # non-overlapping untouched
    twin["markdown"] = "| a | b |"
    twin["html"] = "<table><tr><td>a</td><td>b</td></tr></table>"
    assert _emit_table_blocks(twin) == ["| a | b |"]
