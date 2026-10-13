"""EXP-1 table-evidence gate + native_find_tables memo unit tests.

Run: pytest scripts/scientific_parser/tests/test_table_gate.py -v
"""

import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.perception.tables.detector import (
    _FIND_TABLES_CACHE,
    native_find_tables,
)
from scientific_parser.routing.layout_gate import (
    _suspicion_signals,
    _text_table_signals,
    page_needs_table_chain,
)


def _make_pdf(tmpdir: str, with_ruled_table: bool, name: str = "p.pdf") -> str:
    """Build a minimal born-digital PDF; optional 4x3 ruled table grid."""
    import fitz

    path = str(Path(tmpdir) / name)
    doc = fitz.open()
    page = doc.new_page(width=612, height=792)
    y = 72
    if with_ruled_table:
        # 3 columns x 3 row-rules + 4 vertical rules = grid >=3H & >=3V
        x0, x1 = 72, 500
        cols = (72, 200, 340, 500)
        rows = (72, 150, 230, 320)
        for yv in rows:
            page.draw_line(fitz.Point(x0, yv), fitz.Point(x1, yv))
        for xv in cols:
            page.draw_line(fitz.Point(xv, 72), fitz.Point(xv, 320))
        # find_tables requires cell text to register a table
        for ri in range(len(rows) - 1):
            for ci in range(len(cols) - 1):
                page.insert_text(fitz.Point(cols[ci] + 8, rows[ri] + 22), f"r{ri}c{ci}")
        y = 340
    # Raw layer must also be a clean prose surface (>=200 chars, long
    # lines) — since exp1-gate3 the gate probes the raw page text too,
    # and a thin raw layer legitimately keeps the chain running.
    page.insert_text(
        fitz.Point(72, y + 20),
        "prose line one two three with several more words to carry a long line",
    )
    page.insert_text(
        fitz.Point(72, y + 44),
        "prose line four five six plus additional words so the raw text layer",
    )
    page.insert_text(
        fitz.Point(72, y + 68),
        "exceeds the thin-text floor and keeps long flowing lines throughout",
    )
    page.insert_text(
        fitz.Point(72, y + 92),
        "the page, holding none of the short-run or columnar shapes at all",
    )
    page.insert_text(
        fitz.Point(72, y + 116),
        "so the gate may safely skip the whole detection chain here today",
    )
    doc.save(path)
    doc.close()
    return path


# === text probes (pure, free) ==================================== #


def test_text_probe_pipe_rows():
    text = "intro\n| a | b |\n| --- | --- |\n| 1 | 2 |\n"
    assert _text_table_signals(text) is True


def test_text_probe_caption():
    assert _text_table_signals("see Table 3 for results") is True
    assert _text_table_signals("see Table IV for results") is True


def test_text_probe_numeric_grid():
    text = "\n".join(
        [
            "alpha    1.20    3.40",
            "beta     2.20    4.40",
            "gamma    3.20    5.40",
            "delta    4.20    6.40",
        ]
    )
    assert _text_table_signals(text) is True


def test_text_probe_plain_prose_negative():
    assert _text_table_signals("just a paragraph of running prose text here") is False
    assert _text_table_signals("") is False


# === geometry probes (fixture PDFs) ============================== #


def test_ruled_table_page_has_evidence():
    with tempfile.TemporaryDirectory() as td:
        pdf = _make_pdf(td, with_ruled_table=True)
        assert page_needs_table_chain(pdf, 1, "prose only no markers") is True


def test_prose_page_no_evidence():
    with tempfile.TemporaryDirectory() as td:
        pdf = _make_pdf(td, with_ruled_table=False)
        # long flowing prose (>=200 chars, long lines) — the safe-skip shape
        prose = (
            "This paragraph carries ordinary running prose with sentences that "
            "wrap across many words per line and none of the short-line runs "
            "or columnar shapes that the last-resort strategies feed on, so "
            "the full detection chain is provably wasted on such a page. "
        )
        assert page_needs_table_chain(pdf, 1, prose) is False


def test_missing_pdf_fails_safe():
    # cannot prove absence → gate must let detection run
    assert page_needs_table_chain("/nonexistent/x.pdf", 1, "plain prose") is True


def test_raw_surface_fires_when_stage_text_clean():
    """Surface-independence (exp1-gate3): the pipeline calls the gate with
    STAGE text (band-stripped/dehyphenated), but the shadow census
    validated signals on the RAW layer. A page whose raw layer is thin
    (<200 chars — the band-stripper can remove exactly that much) must
    keep the chain even when the caller's stage text is clean prose."""
    import fitz

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "thin_raw.pdf")
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        page.insert_text(fitz.Point(72, 92), "Fig. 1 caption")  # 14 chars raw
        doc.save(path)
        doc.close()
        clean_stage = (
            "This paragraph carries ordinary running prose with sentences that "
            "wrap across many words per line and none of the short-line runs "
            "or columnar shapes that the last-resort strategies feed on, so "
            "the full detection chain is provably wasted on such a page. "
        )
        # caller surface silent, raw surface thin → chain must run
        assert _suspicion_signals(clean_stage) is False
        assert page_needs_table_chain(path, 1, clean_stage) is True


def test_two_column_page_needs_chain():
    """2-col repair trap (exp1-gate bench): the carve chain doubles as the
    only correct 2-column order reader on interleaved/shattered layers, so
    column geometry alone must keep the chain running — no other signal."""
    import fitz

    with tempfile.TemporaryDirectory() as td:
        path = str(Path(td) / "twocol.pdf")
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        coltext = lambda prefix, i: (
            f"{prefix} column running prose sentence {i} that is long enough"
        )
        # insert_textbox keeps fitz blocks column-major (plain insert_text
        # merges row-major into one block per line); count_columns needs
        # >=2 blocks per side with y-overlap — real 2-col pages have many.
        for i in range(3):
            page.insert_textbox(fitz.Rect(72, 90 + i * 110, 300, 190 + i * 110), coltext("left", i))
            page.insert_textbox(
                fitz.Rect(316, 90 + i * 110, 544, 190 + i * 110), coltext("right", i)
            )
        doc.save(path)
        doc.close()
        prose = (
            " ".join(coltext("left", i) for i in range(3))
            + " "
            + " ".join(coltext("right", i) for i in range(3))
        )
        assert page_needs_table_chain(path, 1, prose) is True


# === suspicion probes (shadow-sweep signals) ====================== #


def test_suspicion_thin_text():
    assert _suspicion_signals("") is True
    assert _suspicion_signals("   ") is True
    assert _suspicion_signals("scanned page with almost no text") is True  # <200ch


def test_suspicion_short_run():
    # >=6 consecutive short lines (roster/grid shape) over a thick page
    lines = ["intro paragraph " + "word " * 30]  # long line first
    lines += [f"name {i} value {i}" for i in range(6)]  # 6 short lines
    lines += ["outro paragraph " + "word " * 30]  # long line last
    text = "\n".join(lines)
    assert len(text) >= 200
    assert _suspicion_signals(text) is True


def test_suspicion_short_majority():
    # >=50% short lines over >=8 lines
    lines = [f"item {i}" if i % 2 == 0 else "sentence " + "word " * 12 for i in range(10)]
    assert _suspicion_signals("\n".join(lines)) is True


def test_suspicion_flowing_prose_negative():
    prose = (
        "This paragraph carries ordinary running prose with sentences that "
        "wrap across many words per line and none of the short-line runs "
        "or columnar shapes that the last-resort strategies feed on, so "
        "the full detection chain is provably wasted on such a page. "
    )
    assert _suspicion_signals(prose) is False


def test_native_find_tables_memo_roundtrip():
    with tempfile.TemporaryDirectory() as td:
        pdf = _make_pdf(td, with_ruled_table=True)
        first = native_find_tables(pdf, 1)
        assert len(first) >= 1
        assert native_find_tables(pdf, 1) is first  # memo hit, identical object


def test_native_find_tables_missing_pdf_empty():
    assert native_find_tables("/nonexistent/y.pdf", 1) == []


# --- EXP-1.6: region_has_table_evidence (model-escalation gate) ---


def test_region_thin_text_is_evidence():
    # Scan crops / sparse tables have little text -> escalate (CNN is the
    # only hope there).
    from scientific_parser.routing.layout_gate import region_has_table_evidence

    assert region_has_table_evidence("/nonexistent/y.pdf", 1, (0, 0, 100, 100), "") is True
    assert region_has_table_evidence("/nonexistent/y.pdf", 1, (0, 0, 100, 100), "Qty 3") is True


def test_region_pipe_rows_is_evidence():
    from scientific_parser.routing.layout_gate import region_has_table_evidence

    text = (
        "Here is a short preamble sentence. \n"
        "| name | qty |\n"
        "|------|-----|\n"
        "| bolt | 12  |\n"
        "| nut  | 7   |\n"
    )
    assert region_has_table_evidence("/nonexistent/y.pdf", 1, (0, 0, 100, 100), text) is True


def test_region_dense_prose_no_evidence():
    # The 0005784d class: dense 2-col prose dispatched as pseudo-table ->
    # no pipe/caption/gap signals, no ruled grid, plenty of text -> skip CNN.
    from scientific_parser.routing.layout_gate import region_has_table_evidence

    prose = (
        "The evaluation framework described in this section combines both "
        "quantitative benchmarks and qualitative review panels across all "
        "participating systems, and every submission was scored twice by "
        "independent annotators before the aggregate results were released. "
    )
    assert region_has_table_evidence("/nonexistent/y.pdf", 1, (0, 0, 100, 100), prose) is False


def test_region_ruled_grid_inside_bbox_is_evidence():
    from scientific_parser.routing.layout_gate import region_has_table_evidence

    with tempfile.TemporaryDirectory() as td:
        pdf = _make_pdf(td, with_ruled_table=True)
        text = "word " * 80  # dense text, no table text signals
        # grid spans x 72-500 (H midpoints x=286), y 72-320 (V midpoints
        # y=196) — this bbox contains >=3 of each
        assert region_has_table_evidence(pdf, 1, (40, 40, 400, 400), text) is True
        # far-away bbox sees none of the grid strokes
        assert region_has_table_evidence(pdf, 1, (400, 400, 500, 500), text) is False


def test_region_probe_failure_dense_prose_skips(tmp_path):
    # Corrupt PDF + dense prose: vector probe fails soft (False, its own
    # except), text probes say nothing table-ish -> no CNN. Correct: the
    # CNN needs a rendered crop anyway, which a corrupt PDF cannot give.
    from scientific_parser.routing.layout_gate import region_has_table_evidence

    bad = str(tmp_path / "definitely-corrupt.pdf")
    Path(bad).write_bytes(b"%PDF-1.4 not really a pdf")
    prose = "ordinary dense prose text " * 20
    assert region_has_table_evidence(bad, 1, (0, 0, 100, 100), prose) is False


def test_region_raster_overlap_is_evidence():
    # EXP-4b pin (9a61fe78 pg2 FUND table): a raster table region has NO
    # text-layer or vector evidence by construction — image coverage is
    # the only signal. bbox is PDF-pt (contract: callers convert).
    from scientific_parser.routing.layout_gate import region_has_table_evidence

    with tempfile.TemporaryDirectory() as td:
        import fitz

        doc = fitz.open()
        page = doc.new_page(width=612, height=792)
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 60, 60))
        pix.clear_with(90)
        page.insert_image(fitz.Rect(100, 200, 400, 500), pixmap=pix)
        pdf = f"{td}/raster.pdf"
        doc.save(pdf)
        doc.close()
        prose = "ordinary dense prose text " * 20
        # region fully over the image -> evidence True
        assert region_has_table_evidence(pdf, 1, (110, 210, 390, 490), prose) is True
        # region over blank page -> no evidence
        assert region_has_table_evidence(pdf, 1, (40, 40, 80, 90), prose) is False
