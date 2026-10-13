"""Routing robustness tests (audit-2 ROUTING-ROBUST, 2026-08-18).

One test per router signal: firing, not-firing, and probe-failure paths.
The router decides fast-path vs OCR for every page — silent regressions
here misroute whole categories (measured: 24/40 table pages false-routed
by a broken dep probe; 8 body sentences/page eaten by a band bug).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # scripts/

from scientific_parser.routing.layout_gate import (
    column_sharp,
    left_edge_column_sharpness,
)
from scientific_parser.routing.page_classifier import (
    _is_legitimate_short_line_page,
    preclassify_document,
    should_route_to_ocr,
)

# ── Signal 1: low_chars ────────────────────────────────────────────── #


def test_signal1_low_chars_routes_ocr():
    route, reasons = should_route_to_ocr("short text", None, 1, None)
    assert route and any(r.startswith("low_chars") for r in reasons)


def test_signal1_sufficient_text_stays_fast():
    text = "word " * 200
    route, reasons = should_route_to_ocr(text, None, 1, None)
    assert not route, reasons


# ── Signals 3-6: text-quality (alphanum, space, newline, invalid) ──── #


def test_signal3_low_alphanum_routes_ocr():
    text = "!!!! #### ???? $$$$" * 60
    route, reasons = should_route_to_ocr(text, None, 1, None)
    assert route and any(r.startswith("low_alphanum") for r in reasons)


def test_signal9_broken_encoding_routes_ocr():
    text = ("abc\ufffd def\ufffd " * 80).ljust(300, "x")
    route, reasons = should_route_to_ocr(text, None, 1, None)
    assert route and any(r.startswith("broken_encoding") for r in reasons)


# ── EXP-1.5 corroboration: text-quality signals on born-digital layers ── #

_BORN_DI = {"pages_with_fonts": {1}, "invisible_pages": set(), "born_digital": True}


def _prose_pdf(tmp_path, line="Point of contact Bryce Tappan 667 demolitions research page"):
    """Single-page born-digital PDF whose FITZ surface carries >=200
    alphanumerics (the corroboration bar)."""
    import fitz

    pdf = tmp_path / "prose.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i in range(6):  # 6 lines x ~60 alpha clears the 200-alpha bar
        page.insert_text((72, 100 + 14 * i), line)
    doc.save(pdf)
    doc.close()
    return str(pdf)


def test_exp15_born_digital_control_glyphs_stay_fast(tmp_path):
    # Symbol-font control glyphs interleaved with real prose on a fonted
    # page (audit: 081875a1, 1379 ctrl chars, 2832 alpha) — noise, not OCR.
    # EXP-1.5b: corroboration needs BOTH surfaces healthy — the consumed
    # FITZ surface is provided by a real fixture (pdf_path=None = fail-closed).
    pdf = _prose_pdf(tmp_path)
    prose = "Point of contact: Bryce Tappan, 667-522- demolitions research "
    text = prose * 40 + "\x12" * 200 + "\x11" * 150
    route, reasons = should_route_to_ocr(text, pdf, 1, _BORN_DI)
    assert not any(r.startswith(("invalid_chars", "low_alphanum")) for r in reasons)


def test_exp15_born_digital_low_alphanum_table_page_stays_fast(tmp_path):
    # Numeric/table page (audit: 12cf936d, ar=0.20, alpha=523, fonts) —
    # low alpha ratio alone must not route a born-digital layer with
    # substantial clean content on both surfaces.
    pdf = _prose_pdf(tmp_path)
    text = (" 1234 5678 9012 \x07\x16\x17\n" * 60) + "summary note about data "
    route, reasons = should_route_to_ocr(text, pdf, 1, _BORN_DI)
    assert not any(r.startswith("low_alphanum") for r in reasons)


def test_exp15_fitz_garbage_surface_routes_ocr(tmp_path):
    # 05a0e5bc pin (bench6 flip): pdftotext-healthy text arg + fitz surface
    # = symbol-glyph garbage (fixture: near-empty page, alpha<200) — the
    # fast path consumes the FITZ surface, so corroboration must FAIL and
    # invalid_chars must keep routing the page to OCR.
    import fitz

    pdf = tmp_path / "garbage_surface.pdf"
    doc = fitz.open()
    doc.new_page()  # empty page: fitz alpha = 0
    doc.save(pdf)
    doc.close()
    text = ("Real prose here with plenty of alphanumerics to pass pdftotext " * 20) + "\x12" * 300
    route, reasons = should_route_to_ocr(text, str(pdf), 1, _BORN_DI)
    assert route and any(r.startswith("invalid_chars") for r in reasons)


def test_exp15_thin_born_digital_layer_still_routes():
    # <200 alphanumerics + invalid chars = nearly-empty layer: corroboration
    # does NOT apply, page still routes (conservative default unchanged).
    text = "ab cd ef \x01\x02\x03" * 30
    route, reasons = should_route_to_ocr(text, None, 1, _BORN_DI)
    assert route and any(r.startswith("invalid_chars") for r in reasons)


def test_exp15_non_born_digital_low_alphanum_unchanged():
    # No fonts / doc_info None → original semantics: low_alphanum routes.
    text = "!!!! #### ???? $$$$" * 60
    route, reasons = should_route_to_ocr(text, None, 1, None)
    assert route and any(r.startswith("low_alphanum") for r in reasons)


def test_exp15_mojibake_signal_unaffected_by_corroboration():
    # tounicode_mojibake (Signal 11) must keep routing born-digital pages.
    text = ("integral f(x) dx = ´ ¨ ˆ " * 30) + "prose " * 40
    route, reasons = should_route_to_ocr(text, None, 1, _BORN_DI)
    assert route and any(r.startswith("tounicode_mojibake") for r in reasons)


# ── Signal 7b: positioned_glyphs + ToC exclusion (R5) ──────────────── #


def _word_per_line_text(n=200):
    # Signal 7b requires >= ROUTER_SHORT_LINE_MIN_CHARS (500) total chars
    # with one short word per line (avg words/line < ROUTER_MIN_AVG_WPL).
    return "\n".join(f"word{i}" for i in range(n))


def test_signal7b_word_per_line_routes_ocr_when_not_born():
    assert len(_word_per_line_text()) >= 500
    route, reasons = should_route_to_ocr(_word_per_line_text(), None, 1, None)
    assert route and any(r.startswith("positioned_glyphs") for r in reasons)


def test_signal7b_suppressed_on_per_page_born():
    # R6: per-PAGE born gate — page 1 has fonts and no overlay.
    di = {"pages_with_fonts": {1}, "invisible_pages": set(), "born_digital": False}
    route, reasons = should_route_to_ocr(_word_per_line_text(), None, 1, di)
    assert not route, reasons


def test_r6_overlay_page_loses_born_even_in_born_doc():
    # Doc-level born=True, but THIS page is an invisible overlay — the
    # per-page gate must see through the doc flag (audit R6).
    di = {
        "pages_with_fonts": {1, 2},
        "invisible_pages": {2},
        "born_digital": True,
    }
    _, reasons_p2 = should_route_to_ocr(_word_per_line_text(), None, 2, di)
    assert any(r.startswith("positioned_glyphs") for r in reasons_p2)
    route_p1, _ = should_route_to_ocr(_word_per_line_text(), None, 1, di)
    assert not route_p1


def test_toc_dot_leaders_excluded():
    # Short-line fragments of a ToC: dot-leader + page number under the
    # 10-char short-line filter (the exclusion only sees short lines).
    lines = [f"... {i * 3:02d}" for i in range(60)]
    assert _is_legitimate_short_line_page("\n".join(lines))
    route, reasons = should_route_to_ocr("\n".join(lines), None, 1, None)
    assert not any(r.startswith("positioned_glyphs") for r in reasons)


def test_toc_ascending_numbers_excluded():
    # Real ToC fragment: number-ending entries in ASCENDING order (R5).
    lines = [f"p {i * 4 + 1}" for i in range(60)]
    assert _is_legitimate_short_line_page("\n".join(lines))


def test_toc_descending_numbers_not_excluded():
    # Number-ending but NOT ascending — must NOT be classified as ToC
    # (R5 regression guard: equation/measure fragments).
    lines = [f"x {100 - i}" for i in range(60)]
    assert not _is_legitimate_short_line_page("\n".join(lines))


# ── Signal 10: glyph_reread fail-fire (R2) ──────────────────────────── #


def _signal10_doc_info():
    return {
        "pages_with_fonts": {1},
        "invisible_pages": set(),
        "born_digital": True,
        "suspect_fonts": True,
        "page_cols": {1: 2},
    }


def _prose_2col_text():
    para = (
        "The measured atmospheric conditions vary significantly across the "
        "urban environment and require careful instrumentation. "
    )
    return para * 12  # >800 chars, no table markers


def test_signal10_fires_for_suspect_2col_born_page():
    # RETIRED 2026-08-19 (iter-56): signal 10 no longer fires — fast path
    # beats VL on its own profile (measured 12 fixtures: 37/50 vs 30/50).
    # The contract inverts: the same inputs must NOT produce glyph_reread.
    _, reasons = should_route_to_ocr(_prose_2col_text(), None, 1, _signal10_doc_info())
    assert not any(r.startswith("glyph_reread") for r in reasons)


def test_signal10_probe_failure_fires_with_warning():
    # RETIRED 2026-08-19 (iter-56): same inverted contract, probe-failure
    # path included — no glyph_reread reason under any probe outcome.
    _, reasons = should_route_to_ocr(
        _prose_2col_text(), "/nonexistent/probe.pdf", 1, _signal10_doc_info()
    )
    assert not any(r.startswith("glyph_reread") for r in reasons)


def test_signal10_table_marker_suppresses():
    text = _prose_2col_text() + "\nTable 1: measured values\n| a | b |\n|---|---|\n"
    _, reasons = should_route_to_ocr(text, None, 1, _signal10_doc_info())
    assert not any(r.startswith("glyph_reread") for r in reasons)


def test_signal10_not_born_never_fires():
    di = _signal10_doc_info()
    di["pages_with_fonts"] = set()  # page not fonted
    _, reasons = should_route_to_ocr(_prose_2col_text(), None, 1, di)
    assert not any(r.startswith("glyph_reread") for r in reasons)


# ── Shared column geometry (R4) ──────────────────────────────────────── #


def test_column_sharp_two_dominant_clusters():
    lines = [(60, y * 15, 280, y * 15 + 10) for y in range(12)]
    lines += [(320, y * 15, 540, y * 15 + 10) for y in range(12)]
    cov, col2 = left_edge_column_sharpness(lines)
    assert col2 == 320 and cov == 1.0


def test_column_sharp_scattered_rejected():
    lines = [((i * 37) % 400 + 30, i * 15, (i * 37) % 400 + 200, i * 15 + 10) for i in range(14)]
    assert not column_sharp(lines)


def test_column_sharp_indents_not_columns():
    # Paragraph indents (20pt) must not read as a second column.
    lines = [(60, y * 15, 500, y * 15 + 10) for y in range(10)]
    lines += [(80, y * 15 + 200, 500, y * 15 + 210) for y in range(10)]
    assert not column_sharp(lines)


# ── preclassify per-page fonts (R1) ──────────────────────────────────── #


def _patch_preclassify(monkeypatch, fake_stdout):
    import scientific_parser.routing.page_classifier as pc

    class FakeProc:
        stdout = fake_stdout

    monkeypatch.setattr(pc.subprocess, "run", lambda *a, **kw: FakeProc())
    monkeypatch.setattr(pc, "DEPS", {**getattr(pc, "DEPS", {}), "pdffonts": True})
    monkeypatch.setattr(
        "pypdfium2.PdfDocument", lambda *a, **kw: (_ for _ in ()).throw(OSError("skip"))
    )


def test_preclassify_custom_font_flags_suspect(monkeypatch):
    # Custom-encoded font rows set suspect_fonts (ToUnicode-risk family).
    fake_stdout = (
        "name                                 type              encoding         emb sub uni object ID\n"
        "------------------------------------ ----------------- ---------------- --- --- --- ---------\n"
        "NCTLZK+NimbusRomNo9L-Regu            Type 1            Custom           yes yes yes    138  0\n"
    )
    _patch_preclassify(monkeypatch, fake_stdout)
    di = preclassify_document("fake.pdf", [1])
    assert di["suspect_fonts"] is True


def test_preclassify_builtin_fonts_not_suspect(monkeypatch):
    fake_stdout = (
        "name                                 type              encoding         emb sub uni object ID\n"
        "------------------------------------ ----------------- ---------------- --- --- --- ---------\n"
        "HLWYIH+CMR10                         Type 1            Builtin          yes yes yes     15  0\n"
    )
    _patch_preclassify(monkeypatch, fake_stdout)
    di = preclassify_document("fake.pdf", [1])
    assert di["suspect_fonts"] is False


def test_preclassify_per_page_fonts_via_fitz(tmp_path):
    # R1: per-PAGE font truth. Page 1 carries a real text font; page 2 is
    # bare (no fonts) — the old blanket logic certified both.
    import fitz

    pdf = tmp_path / "two_page.pdf"
    doc = fitz.open()
    doc.new_page()  # page 1
    doc[0].insert_text((72, 72), "fonted page")
    doc.new_page()  # page 2 — fontless
    doc.save(str(pdf))
    doc.close()
    di = preclassify_document(str(pdf), [1, 2])
    assert 1 in di["pages_with_fonts"], "fonted page must be marked"
    assert 2 not in di["pages_with_fonts"], "fontless page must NOT be born"


def test_extract_avg_over_actual_pages(monkeypatch):
    import scientific_parser.routing.page_classifier as pc

    class FakeProc:
        stdout = "page one text\n\fpage two text\n\fpage three\n\ffour\n\ffive\n"

    monkeypatch.setattr(pc.subprocess, "run", lambda *a, **kw: FakeProc())
    by_page, avg = pc.extract_text_layer("fake.pdf", [1, 5])  # non-contiguous
    lens = [
        len("page one text"),
        len("page two text"),
        len("page three"),
        len("four"),
        len("five"),
    ]
    assert avg == sum(lens) / 5, "avg must divide by pages PRODUCED, not len(page_list)"


def test_glyph_reread_signal_retired():
    """iter-56 pin: glyph_reread no longer routes born-digital custom-font
    2col pages to VL. Measured 2026-08-19 (12 bench fixtures, post iter-53
    fragment-block assembly): fast 37/50 vs VL 30/50, and the deterministic
    path removes the ±3-test VL wobble. 01566a95 (custom-font born-digital
    2col, the exact signal-10 profile) must route FAST."""
    import glob
    import sys
    import warnings

    warnings.filterwarnings("ignore")
    sys.path.insert(0, "scripts")
    import pdf_ocr as po

    po.DEPS["pdffonts"] = True
    from scientific_parser.routing.page_classifier import (
        preclassify_document,
        should_route_to_ocr,
    )

    pdfs = glob.glob("bench/bench_data/pdfs/multi_column/01566a951fd40f8c1a4b*page_4*.pdf")
    if not pdfs:
        pytest.skip("fixture not present")
    import subprocess

    P = pdfs[0]
    r = subprocess.run(["pdftotext", "-f", "1", "-l", "1", P, "-"], capture_output=True, text=True)
    di = preclassify_document(P, [1])
    ocr, reasons = should_route_to_ocr(r.stdout, pdf_path=P, page_num=1, doc_info=di)
    assert not ocr, f"retired signal fired: {reasons}"


# ── Signal 11: ToUnicode mojibake (R14b, 2026-09-12) ───────────────── #


def test_signal11_mojibake_markers_route_ocr():
    """Isolated ´/¨/ˆ (broken-CMap ∫/hats) route the page to OCR — the
    2503.04488 pg9 signature (20 isolated markers, page 100% failed on
    the fast path)."""
    moji = (
        "We construct the integral operator ´ over the domain, and the "
        "hat-decoration ¨ acts on q; the pairing r´, evaluated at ´ gives "
        "the trace formula; similarly ˆ applied twice returns the unit."
    )
    route, reasons = should_route_to_ocr(moji, None, 1, None)
    assert route, "mojibake marker page must route"
    assert any(r.startswith("tounicode_mojibake") for r in reasons), reasons


def test_signal11_attached_diacritics_stay_fast():
    """Attached diacritics (café, über) are NOT isolated markers — clean
    prose never trips signal 11 (long enough to clear low_chars)."""
    clean = "The café's naïve résumé über Straße and another café review " * 6
    route, reasons = should_route_to_ocr(clean, None, 1, None)
    assert not any(r.startswith("tounicode_mojibake") for r in reasons), reasons


def test_signal11_pua_math_not_mojibake():
    """PUA codepoints (CMEX delimiter pieces) are the born-digital math
    payload the PUA exclusion protects — signal 11 keys on isolated
    Latin-1 diacritics only, never on PUA."""
    pua = "The tall delimiters    assemble env braces " * 8
    route, reasons = should_route_to_ocr(pua, None, 1, None)
    assert not any(r.startswith("tounicode_mojibake") for r in reasons), reasons


def test_signal11_real_fixture_routes():
    """The headline mojibake fixture (2503.04488 pg9) routes via signal 11
    on its real text layer."""
    import fitz

    fixture = (
        Path(__file__).resolve().parents[3]
        / "bench"
        / "data"
        / "bench_data"
        / "pdfs"
        / "arxiv_math"
        / "2503.04488_pg9.pdf"
    )
    if not fixture.exists():
        pytest.skip("fixture not present")
    text = fitz.open(fixture)[0].get_text()
    route, reasons = should_route_to_ocr(text, str(fixture), 1, None)
    assert route and any(r.startswith("tounicode_mojibake") for r in reasons), reasons


# ── det47: raster-table reachability through the router gate ── #


def test_det47_raster_table_page_needs_chain(tmp_path):
    """Prose-heavy page + table-sized raster (200x100pt) must reach the
    table chain — the det42/det43 raster strategies are otherwise
    unreachable (run-52: 3a91fad8 / 587a1f5a / odlbench doc-122)."""
    import fitz

    from scientific_parser.routing.layout_gate import page_needs_table_chain

    pdf = tmp_path / "raster_table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    line = "Point of contact Bryce Tappan 667 demolitions research page"
    for i in range(6):
        page.insert_text((72, 100 + 14 * i), line)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 200))
    page.insert_image(fitz.Rect(150, 300, 350, 400), pixmap=pix)
    doc.save(pdf)
    doc.close()
    assert page_needs_table_chain(str(pdf), 1, line) is True


def test_det47_pure_prose_page_still_skips_chain(tmp_path):
    """Regression: no raster, no text signals → gate stays False.
    Lines must be VARIED prose — a single short line trips
    _suspicion_signals independent of det47 (pre-existing, verified
    via stash A/B)."""
    import fitz

    from scientific_parser.routing.layout_gate import page_needs_table_chain

    lines = [
        "The committee reviewed the annual report of the research division",
        "and found that most field programs had expanded during the year",
        "while administrative overhead remained broadly stable overall",
        "across every regional office that submitted complete documentation",
    ]
    pdf = tmp_path / "prose_varied.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i, l in enumerate(lines):
        page.insert_text((72, 100 + 14 * i), l)
    doc.save(pdf)
    doc.close()
    assert page_needs_table_chain(str(pdf), 1, " ".join(lines)) is False


# ── det48: raster-only evidence scoping ── #


def test_det48_raster_only_evidence_true_on_prose_plus_image(tmp_path):
    """Prose page + table-sized raster + every other probe False →
    raster_only_table_evidence True (detection scopes to raster)."""
    import fitz

    from scientific_parser.routing.layout_gate import raster_only_table_evidence

    lines = [
        "The committee reviewed the annual report of the research division",
        "and found that most field programs had expanded during the year",
        "while administrative overhead remained broadly stable overall",
        "across every regional office that submitted complete documentation",
    ]
    pdf = tmp_path / "prose_image.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i, l in enumerate(lines):
        page.insert_text((72, 100 + 14 * i), l)
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 400, 200))
    page.insert_image(fitz.Rect(150, 300, 350, 400), pixmap=pix)
    doc.save(pdf)
    doc.close()
    assert raster_only_table_evidence(str(pdf), 1, " ".join(lines)) is True


def test_det48_raster_only_evidence_false_when_text_table_present(tmp_path):
    """A page with real TEXT-table evidence must NOT scope to raster-only
    (the full chain keeps its text strategies). Pipes + aligned decimals
    are a strong text-table signal."""
    import fitz

    from scientific_parser.routing.layout_gate import raster_only_table_evidence

    lines = [
        "| Region | Cases | Rate |",
        "| alpha  | 12.5  | 1.2  |",
        "| beta   | 949.9 | 3.4  |",
    ]
    pdf = tmp_path / "text_table.pdf"
    doc = fitz.open()
    page = doc.new_page()
    for i, l in enumerate(lines):
        page.insert_text((72, 100 + 14 * i), l)
    doc.save(pdf)
    doc.close()
    assert raster_only_table_evidence(str(pdf), 1, "\n".join(lines)) is False
