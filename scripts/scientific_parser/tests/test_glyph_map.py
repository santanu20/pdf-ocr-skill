"""EXP det16: cell-level glyph map — store/take lifecycle + surgical parity.

Pins the producer/consumer contract: the xval crosscheck stores a verified
(frm, to, layer-words) map keyed by (abspath, mtime_ns, page); the tables
lane pops it once and re-applies the identical surgical substitution to
table cells.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.routing.glyph_map import (  # noqa: E402
    _MAPS,
    store_page_glyph_map,
    surgical_word_sub,
    take_page_glyph_map,
)


def _clear() -> None:
    _MAPS.clear()


def test_store_take_roundtrip_pop(tmp_path):
    _clear()
    f = tmp_path / "a.pdf"
    f.write_bytes(b"%PDF-1.4\n")
    store_page_glyph_map(str(f), 3, "w", "f", ("worst", "widow"))
    # stored canonical form is sorted, not insertion/set order
    got = take_page_glyph_map(str(f), 3)
    assert got == ("w", "f", ("widow", "worst"))
    # pop-on-read: second take returns None (single shot per page)
    assert take_page_glyph_map(str(f), 3) is None


def test_store_missing_file_is_fail_soft():
    _clear()
    # nonexistent path: store silently no-ops, take returns None
    store_page_glyph_map("/tmp/nonexistent_b.pdf", 1, "x", "y", ("word",))
    assert take_page_glyph_map("/tmp/nonexistent_b.pdf", 1) is None


def test_take_is_mtime_and_page_keyed(tmp_path):
    _clear()
    f = tmp_path / "b.pdf"
    f.write_bytes(b"%PDF-1.4\n")
    store_page_glyph_map(str(f), 1, "x", "y", ("word",))
    # different page -> miss
    assert take_page_glyph_map(str(f), 2) is None
    assert take_page_glyph_map(str(f), 1) == ("x", "y", ("word",))


def test_store_rejects_empty_map(tmp_path):
    _clear()
    f = tmp_path / "c.pdf"
    f.write_bytes(b"%PDF-1.4\n")
    store_page_glyph_map(str(f), 1, "w", "f", ())
    assert take_page_glyph_map(str(f), 1) is None
    store_page_glyph_map(str(f), 1, "", "f", ("worst",))
    assert take_page_glyph_map(str(f), 1) is None


def test_take_missing_file_returns_none():
    _clear()
    assert take_page_glyph_map("/tmp/definitely_missing_zz.pdf", 1) is None


def test_surgical_word_sub_case_parity():
    text = "The worst case and WORST outcome share widow words."
    out, n = surgical_word_sub(text, ("worst", "WORST", "widow"), "w", "f")
    assert "forst" in out and "FORST" in out
    assert n == 3  # worst, WORST, widow
    assert "worst" not in out and "WORST" not in out


def test_surgical_word_sub_word_boundary_guard():
    # word-boundary regex must not touch the substring inside a longer word
    out, n = surgical_word_sub("unworstly worst", ("worst",), "w", "f")
    assert "unworstly" in out  # untouched
    assert "forst" == out.split()[-1]
    assert n == 1


def test_surgical_word_sub_noop_on_clean():
    out, n = surgical_word_sub("nothing to fix here", ("worst",), "w", "f")
    assert out == "nothing to fix here" and n == 0
