"""Per-page verified glyph-substitution maps (EXP-13 cell-level lane).

Producer: pdf_ocr xval consumer, after tounicode_crosscheck confirms a
systematic substitution ((frm, to) = (layer-wrong char, correct char),
words = fitz-LAYER words carrying it — the same surface the tables lane
reads). Consumer: detect_all_tables tail applies the map to table cells:
verified words often live ONLY in table-cell space (769f619d: 'worst'
inside table geometry), where the flowed-text surgical pass cannot reach.
"""

from __future__ import annotations

import os
import re
import threading

_LOCK = threading.Lock()
# (abspath, mtime_ns, page) -> (frm, to, layer_words tuple)
_MAPS: dict[tuple, tuple] = {}
_MAX_ENTRIES = 512


def _key(pdf_path, page):
    st = os.stat(pdf_path)
    return (os.path.abspath(pdf_path), st.st_mtime_ns, int(page))


def store_page_glyph_map(pdf_path, page, frm, to, words) -> None:
    """Record a verified glyph map for one page (idempotent overwrite).

    Fail-soft on stat errors (vanished/unreadable path): a map that cannot
    be keyed simply never lands — the consumer no-ops.
    """
    if not pdf_path or not frm or not to or not words:
        return
    try:
        key = _key(pdf_path, page)
    except OSError:
        return
    with _LOCK:
        if len(_MAPS) >= _MAX_ENTRIES and key not in _MAPS:
            _MAPS.clear()  # bounded: stale entries are worthless after purge
        # sorted: set iteration order is hash-seed dependent — the
        # stored tuple must roundtrip identically in every process
        _MAPS[key] = (frm, to, tuple(sorted(set(words))))


def take_page_glyph_map(pdf_path, page):
    """Pop the page's glyph map, or None. Single consumer: tables lane."""
    try:
        key = _key(pdf_path, page)
    except OSError:
        return None
    with _LOCK:
        return _MAPS.pop(key, None)


def surgical_word_sub(text, words, frm, to):
    """Apply frm->to inside the given layer words, word-boundary anchored.

    Exact parity with the pdf_ocr xval surgical pass (case handling
    included): only the verified words are rewritten, never a blanket
    character substitution. Returns (new_text, n_replacements).
    """
    if not text or not frm or not to or not words:
        return text, 0
    applied = 0
    for w in sorted(set(words)):
        if frm not in w.lower():
            continue
        # Accumulate case variants: lowercase replace FIRST, then uppercase
        # within the result. (Overwriting from the original word — the
        # det14/15 inline-loop shape — silently reverted the lowercase
        # replacement whenever frm was lowercase, skipping the whole word.)
        nw = w.replace(frm, to)
        if frm.upper() != frm:
            nw = nw.replace(frm.upper(), to.upper())
        if nw == w:
            continue
        nre = re.compile(r"(?<![\w])" + re.escape(w) + r"(?![\w])")
        text, cnt = nre.subn(nw, text)
        applied += cnt
    return text, applied
