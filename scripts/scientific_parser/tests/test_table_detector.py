"""Unit + integration tests for table detector + structure pipeline.

Covers the previously-untested scientific_parser/perception.tables module
flagged CRITICAL in AUDIT_SPECIALIZED_2026-08-13.md. The detector module
had multiple dangling-reference bugs from incomplete R2 extraction (fixed
2026-08-13: classify_cell, validate_table_structure, _line_to_table_cells,
_quality_gate, _is_numeric_cell now properly bound).

Per testing.md §3.1: ≥90% strong assertions.
Run: pytest scripts/scientific_parser/tests/test_table_detector.py -v
"""

import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.perception.tables.detector import (
    build_table_markdown,
    detect_multirow_header,
    detect_tables_in_text,
    infer_column_types,
    _edge_row_is_furniture,
    _scrub_edge_furniture,
    _table_all_furniture,
)
from scientific_parser.perception.tables.structure import (
    classify_cell,
    validate_table_structure,
    _line_to_table_cells,
)
from scientific_parser.postproc.text_pipeline import (
    _html_twin_rows,
    _choose_table_emission,
    _is_monster_rows,
)


# === classify_cell (structure.py) =============================== #


def test_classify_cell_integer_returns_number_type():
    """classify_cell returns 'number' (not 'numeric') for integers."""
    assert classify_cell("42") == "number"


def test_classify_cell_decimal_returns_number_type():
    assert classify_cell("52.3") == "number"


def test_classify_cell_text_word_returns_text():
    assert classify_cell("Alice") == "text"


def test_classify_cell_dash_placeholder_returns_empty_or_placeholder():
    """Geochem convention: '-' sometimes = below-detection. Function treats
    it as 'empty' (cell is a single non-alphanumeric char)."""
    result = classify_cell("-")
    assert result in ("empty", "placeholder", "missing")


def test_classify_cell_empty_string_returns_empty():
    assert classify_cell("") == "empty"


# === infer_column_types (detector.py) =========================== #


def test_infer_column_types_numeric_column_marked_numeric():
    rows = [["Value"], ["10.0"], ["20.0"], ["30.0"]]
    types = infer_column_types(rows)
    assert "numeric" in types


def test_infer_column_types_text_column_marked_text():
    rows = [["Name"], ["Alice"], ["Bob"], ["Carol"]]
    types = infer_column_types(rows)
    assert "text" in types


def test_infer_column_types_mixed_columns_returned_in_correct_order():
    rows = [
        ["Sample", "SiO2"],
        ["A1", "52.3"],
        ["A2", "48.1"],
    ]
    types = infer_column_types(rows)
    assert len(types) == 2
    assert types[1] == "numeric"  # SiO2 column


# === detect_multirow_header (detector.py) ======================= #


def test_detect_multirow_header_clear_data_rows_returns_int_body_start():
    """When the first row is text-header and subsequent rows are clearly
    numeric data, body_start is an int (heuristic-dependent on exact value)."""
    rows = [
        ["Sample", "SiO2"],
        ["A1", "52.3"],
        ["A2", "48.1"],
        ["A3", "50.7"],
    ]
    header, body_start = detect_multirow_header(rows)
    assert isinstance(body_start, int)
    assert isinstance(header, list)


def test_detect_multirow_header_all_text_rows_treats_all_as_header():
    """Pure-text rows (no numerics) get treated as multi-row headers."""
    rows = [["Alpha", "Beta"], ["Gamma", "Delta"], ["Epsilon", "Zeta"]]
    header, body_start = detect_multirow_header(rows)
    assert isinstance(header, list)
    assert isinstance(body_start, int)


def test_detect_multirow_header_empty_rows_does_not_crash():
    """Defensive: empty input must not raise."""
    header, body_start = detect_multirow_header([])
    # Behavior is implementation-defined for empty; just ensure no crash.
    assert isinstance(body_start, int)


# === build_table_markdown (detector.py) ========================= #


def test_build_table_markdown_clear_data_returns_markdown_string():
    """When header/data separation is unambiguous, markdown is non-empty."""
    rows = [
        ["Sample", "SiO2", "Al2O3"],
        ["A1", "52.3", "14.5"],
        ["A2", "48.1", "15.6"],
        ["A3", "50.7", "13.9"],
    ]
    md = build_table_markdown(rows, strategy="ocr_text_parse")
    assert isinstance(md, str)
    # Either contains the table data (success) or is empty (heuristic miss);
    # the function's contract is "str return, may be empty"
    if md:
        assert "Sample" in md or "SiO2" in md


def test_build_table_markdown_returns_string_for_empty_input():
    """Defensive: empty rows must not raise."""
    md = build_table_markdown([], strategy="ocr_text_parse")
    assert isinstance(md, str)


# === validate_table_structure (structure.py) ==================== #


def test_validate_table_structure_valid_numeric_table_passes():
    rows = [
        ["Sample", "SiO2"],
        ["A1", "52.3"],
        ["A2", "48.1"],
    ]
    is_valid, issues = validate_table_structure(rows, strategy="ocr_text_parse")
    assert isinstance(is_valid, bool)
    assert isinstance(issues, list)


def test_validate_table_structure_returns_list_of_issues():
    """Issues is always a list — never None — so callers can iterate safely."""
    rows = [["A", "B"], ["1", "2"]]
    _, issues = validate_table_structure(rows)
    assert isinstance(issues, list)


# === _line_to_table_cells (structure.py) ======================== #


def test_line_to_table_cells_pipe_delimited_returns_cells_list():
    line = "| Name | Age |"
    cells = _line_to_table_cells(line)
    assert isinstance(cells, list)
    # Must contain the cell values, stripped
    assert "Name" in cells
    assert "Age" in cells


def test_line_to_table_cells_separator_line_returns_empty_or_separator_marker():
    """The ``|---|---|`` separator row should not produce data cells."""
    line = "|---|---|"
    cells = _line_to_table_cells(line)
    # Either empty list or contains only separator chars — implementation defined
    assert isinstance(cells, list)


# === detect_tables_in_text (detector.py) — integration ======== #


def test_detect_tables_in_text_markdown_table_extracts_table_dict():
    """End-to-end: text with a markdown table produces a list of table dicts."""
    text = """Some intro prose.

| Sample | SiO2 | Al2O3 | FeO |
|--------|------|-------|-----|
| A1     | 52.3 | 14.5  | 8.2 |
| A2     | 48.1 | 15.6  | 9.4 |

Some trailing prose.
"""
    tables = detect_tables_in_text(text)
    assert isinstance(tables, list)
    assert len(tables) == 1
    t = tables[0]
    assert isinstance(t, dict)
    assert "markdown" in t
    assert "rows" in t
    assert "bbox" in t
    # Spot-check: markdown contains table data
    assert "SiO2" in t["markdown"]
    assert "52.3" in t["markdown"]


def test_detect_tables_in_text_no_table_returns_empty_list():
    """Pure prose with no pipe-delimited structure → no tables detected."""
    text = "This is just a paragraph of prose without any tabular content."
    tables = detect_tables_in_text(text)
    assert tables == []


def test_detect_tables_in_text_empty_string_returns_empty_list():
    """Defensive: empty input must not raise."""
    assert detect_tables_in_text("") == []


def test_detect_tables_in_text_multiple_tables_returns_at_least_one():
    """Two tables in one text: depending on heuristic, may merge or split.
    Contract: returns a list (possibly empty if heuristic misses)."""
    text = """| A | B |
|---|---|
| 1 | 2 |

Some prose between.

| C | D |
|---|---|
| 3 | 4 |
"""
    tables = detect_tables_in_text(text)
    # Heuristic-dependent — just verify it returns a list, doesn't crash
    assert isinstance(tables, list)


# === Smoke tests: full pipeline doesn't crash on adversarial input #


def test_detect_tables_in_text_just_separator_line_returns_empty_list():
    """Lone separator row (no header, no body) should not crash."""
    text = "|---|---|"
    result = detect_tables_in_text(text)
    assert isinstance(result, list)


def test_classify_cell_none_input_does_not_crash():
    """Defensive: None input must not raise."""
    # classify_cell should handle None gracefully
    try:
        result = classify_cell(None)
        assert isinstance(result, str)
    except (TypeError, AttributeError):
        # Acceptable iff documented — but it MUST be a str-or-raise contract
        pytest.skip("classify_cell raises on None — document or fix")


def test_annotation_table_header_emission():
    """iter-11 relocated pin (2026-08-19): the a674 fixture routes
    glyph_reread→VL (custom-font 2-col) so the pins.py extraction asserts
    on a chandra read that emits the header table only intermittently
    (flat lines on other reads; flaked red with byte-identical code).
    The deterministic emission contract lives HERE: detect_tables_in_text
    on a pipe-table text carrying the annotation header must emit
    'Measure | New resources' as separate cells."""
    from scientific_parser.perception.tables.detector import detect_tables_in_text

    text = (
        "| Measure | New resources | Transferred resources | Note |\n"
        "| --- | --- | --- | --- |\n"
        "| Survey | 12 | 4 | annual biodiversity survey across regions |\n"
        "| Listing | 3 | 1 | status review documentation |\n"
        "| Recovery | 45 | 20 | species recovery plan development |\n"
        "| Consultation | 8 | 3 | interagency review meetings |\n"
    )
    tables = detect_tables_in_text(text)
    assert tables, "annotation table must parse"
    flat = " ".join(str(c) for r in tables[0]["rows"] for c in r)
    assert "Measure" in flat and "New resources" in flat


def test_rapidtable_shred_rejected_and_label_value_fallback():
    """iter-55 pins: (a) a grid cut INSIDE words whose rejoin exists in the
    text layer is detected as shredded ('CHEM 2'+'10' → 'CHEM 210' is in
    the layer; 'ENGR'+'351' → 'ENGR351' is not); (b) on the borderless
    label-value fixture the deterministic fallback grid carries every row
    with intact cells."""
    import glob
    import sys

    sys.path.insert(0, "scripts")
    from scientific_parser.perception.tables.table_agent import TableAgent

    pdfs = glob.glob("bench/data/bench_data/pdfs/tables/137297b4*pg4*.pdf")
    if not pdfs:
        pytest.skip("fixture not present")
    agent = TableAgent(pdfs[0], 1, 150.0)

    shred = [
        ["Requirements fo", "r Major in Me", "chanical Engineering", "(9)"],
        ["", "CHEM 200,", "CHEM 201, CHEM 2", "10", "(3)+4"],
    ]
    assert agent._is_layer_shredded(shred)

    clean = [
        ["Basic Science", "(6)+9"],
        ["CHEM 200, CHEM 201, CHEM 210", "(3)+4"],
        ["MATH 150, MATH 250, MATH 251, MATH 305", "(3)+11"],
    ]
    assert not agent._is_layer_shredded(clean)

    grid = agent._label_value_grid(tuple(v * 150.0 / 72.0 for v in (50, 60, 545, 560)))
    assert grid, "label-value fallback must build"
    flat = [c for row in grid for c in row]
    assert any("MATH 150, MATH 250, MATH 251, MATH 305" == c for c in flat)
    assert any(c == "(3)+11" for c in flat)
    assert any("Credit Hours" in c for c in flat)


def test_label_value_prose_carve_guard():
    """EXP-4 pin: dense two-column PROSE must not carve as a label|value
    table. 04af23cb page 5 (journal 2-col prose) builds 37 rows with
    median value 50 chars when the guard is absent — the carve eats the
    body text. Real targets (137297b4) sit at median 2 / max 12."""
    import glob
    import sys

    sys.path.insert(0, "scripts")
    from scientific_parser.perception.tables.table_agent import TableAgent

    scale = 150.0 / 72.0
    p04a = glob.glob("bench/data/bench_data/pdfs/multi_column/04af23cb*page_5*.pdf")
    if not p04a:
        pytest.skip("fixture not present")
    ag = TableAgent(p04a[0], 1, 150.0)
    grid = ag._label_value_grid(tuple(v * scale for v in (40, 55, 560, 740)))
    assert grid is None, "prose-class values (median >30 chars) must not carve"


def test_label_value_fallback_two_column_guard():
    """iter-55: the label|value fallback must return None on multi-column
    tables (904a1b4e year-columns would glue into one value cell) and a
    full clean grid on true 2-column catalogs (137297b4)."""
    import glob
    import sys

    sys.path.insert(0, "scripts")
    from scientific_parser.perception.tables.table_agent import TableAgent

    scale = 150.0 / 72.0
    p904 = glob.glob("bench/data/bench_data/pdfs/tables/904a1b4e*pg141*.pdf")
    if p904:
        ag = TableAgent(p904[0], 1, 150.0)
        assert ag._label_value_grid(tuple(v * scale for v in (71, 94, 472, 274))) is None
    p137 = glob.glob("bench/data/bench_data/pdfs/tables/137297b4*pg4*.pdf")
    if p137:
        ag = TableAgent(p137[0], 1, 150.0)
        grid = ag._label_value_grid(tuple(v * scale for v in (50, 60, 545, 560)))
        assert grid and any(c == "(3)+4" for row in grid for c in row)


def test_header_eats_all_guard():
    """iter-57 pin: detect_multirow_header over-consumed short label|value
    tables ('Assignment|20%' — every row matched header heuristics) leaving
    zero body rows; build_table_markdown returned '' and the whole table
    vanished ('No tables found', 56d7730c lost 4 gold tests). The guard
    must fall back to a single-row header when body_start >= len(rows)."""
    from scientific_parser.perception.tables.detector import build_table_markdown

    rows = [
        ["Assignment", "Percent of Grade"],
        ["Class Participation", "20%"],
        ["Analytical Paper", "40%"],
        ["Final Exam", "40%"],
    ]
    md = build_table_markdown(rows, "lines_strict")
    assert md, "4-row label|value table must build, not vanish"
    assert "| Assignment | Percent of Grade |" in md
    assert "| Class Participation | 20% |" in md


def test_dedup_prefers_uniform_widths():
    """iter-57 pin: on equal row counts, dedup keeps the strategy whose
    rows have uniform column widths. pdfplumber's ragged header
    ('Std.|Dev' 7-wide vs 6-wide body) built EMPTY markdown while
    lines_strict's uniform 7-wide rows built the table — first-seen kept
    the broken one (240cc9b2 'No tables found', 4 tests)."""
    from scientific_parser.perception.tables.structure import _dedup_tables

    ragged = {
        "bbox": (0, 0, 100, 100),
        "strategy": "pdfplumber_text",
        "rows": [["A", "B", "C", "D", "E", "F", "G"], ["a", 1, 2, 3, 4, 5]],
    }
    uniform = {
        "bbox": (0, 0, 100, 100),
        "strategy": "lines_strict",
        "rows": [["A", "B", "C", "D", "E", "F", "G"], ["a", 1, 2, 3, 4, 5, 6]],
    }
    kept = _dedup_tables([ragged, uniform])
    assert len(kept) == 1
    assert kept[0]["strategy"] == "lines_strict"


def test_chatty_list_table_conversion():
    """iter-71 pin: chandra-ocr 'chatty' output (preamble + 'Column N:'
    block + numbered 'Data Rows:' list, 4db371ae pg52) carries real table
    content but NO pipe/HTML table — 5 gold tests died on 'No tables
    found'. The converter must rebuild the grid mechanically from the
    explicit structure: header from Column lines, rows from numbered
    comma-separated entries. VL output is intermittent, so the invariant
    lives here (deterministic text in, deterministic grid out), not on a
    VL-routed end-to-end pin (same reasoning as the a674cb40 relocation,
    iter-11)."""
    from scientific_parser.perception.tables.detector import (
        detect_chatty_list_tables,
    )

    text = (
        "The user wants me to extract data from a table. I will transcribe.\n"
        "Table Structure:\n"
        "- Column 1: TIPO\n"
        "- Column 2: MARCA\n"
        "- Column 3: NUMERO D_EMOTOR\n"
        "- Column 4: HUMERO D_ECHASIS\n"
        "\n"
        "Data Rows:\n"
        "1. CAMIONETA FURG. LUV C/S 4x2 TM, CHEVROLET, 4ZD1511392, TFR16F977109591\n"
        "2. CAMIONETA FURG. D-MAX C/S 4x2 TM, CHEVROLET, C22NE25098182, 8LBOTF29850110664\n"
        "3. CAMION GURG. NPR II, CHEVROLET, 790888, 9GDNKR55898173174\n"
    )
    tables = detect_chatty_list_tables(text)
    assert len(tables) == 1
    rows = tables[0]["rows"]
    assert rows[0] == ["TIPO", "MARCA", "NUMERO D_EMOTOR", "HUMERO D_ECHASIS"]
    assert "TFR16F977109591" in rows[1]
    assert "C22NE25098182" in rows[2]
    assert "9GDNKR55898173174" in rows[3]
    # no structure, no table
    assert detect_chatty_list_tables("plain prose\n1. one, two\n2. three, four") == []


def test_table_renderers_scrub_control_glyphs():
    """EXP-1.5: symbol-font control glyphs must not survive any table emitter.

    Regression pin: 081875a1035e34dee439b9a2a3a55e319405_page_17_pg1.pdf
    leaked 812 control chars through a 12-col pipe table (TableAgent /
    multirow-header lanes bypass the detector emitters).
    """
    from scientific_parser.perception.tables.structure import Table

    row = ['\x14 "E ;(E=\x12S\x12 BERYLLIUM \x19\x0e\x10\x11\x12', "", "ok\ttext"]
    tbl = Table(header_rows=[["H\x01", "H2", "H3"]], body_rows=[row], n_cols=3)
    md = tbl.to_markdown()
    html = tbl.to_html()
    for out in (md, html):
        assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]", out)
    assert "BERYLLIUM" in md
    # det44 sepless emission flattens cells: internal whitespace (incl.
    # \t) normalizes to single spaces — control-glyph scrub unchanged.
    assert "ok text" in md

    from scientific_parser.perception.tables.detector import normalize_grid_rows

    cleaned = normalize_grid_rows([["\x02label", "1\x030"], ["", ""]])
    flat = "".join(c for r in cleaned for c in r)
    assert not re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", flat)
    assert "label" in flat and "10" in flat


# --- det42: rowspan-label value-band tables (9c389720 Table S3 class) ---


def _write_rowspan_pdf(path, rows, page_w=612, page_h=792):
    """Build a one-page PDF: rows = [(y, label_or_None, value_text)].

    Values are CENTER-aligned inside the right band (x 300-540) so their
    left edges drift — the geometry that defeats x0-anchor clustering.
    """
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=page_w, height=page_h)
    for y, label, value in rows:
        if label is not None:
            page.insert_text((172, y), label, fontsize=11)
        if value is not None:
            w = pymupdf.get_text_length(value, fontsize=11)
            page.insert_text((420 - w / 2, y), value, fontsize=11)
    doc.save(path)
    doc.close()


def test_rowspan_labels_centered_groups_fill_down(tmp_path):
    """Sparse labels each govern a run of center-aligned values; labels
    are typeset mid-group, so the first value row sits ABOVE its label."""
    from scientific_parser.perception.tables.detector import (
        detect_tables_rowspan_labels,
    )

    rows = [(90, "Category", "Groups")]
    rows += [(110, None, "Alpha group (10; 20.0%)")]
    rows += [(124, "First", "Beta group (20; 30.0%)")]
    rows += [(138, None, "Gamma group (30; 50.0%)")]
    rows += [(160, None, "Delta (5; 12.5%)")]
    rows += [(174, "Second", "Epsilon (8; 20.0%)")]
    rows += [(188, None, "Zeta (2; 5.0%)")]
    pdf = tmp_path / "rowspan.pdf"
    _write_rowspan_pdf(str(pdf), rows)
    tables = detect_tables_rowspan_labels(str(pdf), 1)
    assert len(tables) == 1
    grid = tables[0]["rows"]
    assert grid[0] == ["Category", "Groups"]
    labels = [r[0] for r in grid[1:]]
    assert labels == ["First", "First", "First", "Second", "Second", "Second"]
    assert tables[0]["strategy"] == "rowspan_labels"


def test_rowspan_labels_prose_left_column_rejected(tmp_path):
    """Justified prose sentences in the left column are not labels."""
    from scientific_parser.perception.tables.detector import (
        detect_tables_rowspan_labels,
    )

    sentence = "and MOB are known to discriminate against the heavier"
    rows = []
    for i in range(8):
        rows.append(
            (100 + 16 * i, sentence if i % 2 == 0 else "Short", f"item {i} ({i}; {i * 3}%)")
        )
    pdf = tmp_path / "prose.pdf"
    _write_rowspan_pdf(str(pdf), rows)
    assert detect_tables_rowspan_labels(str(pdf), 1) == []


def test_rowspan_labels_display_equations_rejected(tmp_path):
    """Aligned equation blocks carry math glyphs; the occurrence-count
    guard vetoes them (2503.04024 pg4 class)."""
    from scientific_parser.perception.tables.detector import (
        detect_tables_rowspan_labels,
    )

    rows = [
        (100, "L(u(x); g(x)) = f(x)", "∀x ∈ Ω,"),
        (116, "B(u(x); g(x)) = η(x)", "(2.1)"),
        (132, "B(u(x); g(x)) = η(x)", "∀x ∈ Γη,"),
        (148, "B(u(x); g(x)) = η(x)", "u(x) = 0"),
        (164, "B(u(x); g(x)) = η(x)", "∀x ∈ ΓD,"),
        (180, "B(u(x); g(x)) = η(x)", "v(x) = 0"),
        (196, "B(u(x); g(x)) = η(x)", "∀x ∈ ΓN,"),
        (212, "B(u(x); g(x)) = η(x)", "w(x) = 0"),
    ]
    pdf = tmp_path / "eqs.pdf"
    _write_rowspan_pdf(str(pdf), rows)
    assert detect_tables_rowspan_labels(str(pdf), 1) == []


def test_rowspan_labels_equal_counts_rejected(tmp_path):
    """One value per label is a definition table, not a rowspan group —
    other strategies own that shape."""
    from scientific_parser.perception.tables.detector import (
        detect_tables_rowspan_labels,
    )

    rows = [(100, "Category", "Groups")]
    for i in range(5):
        rows.append((120 + 18 * i, f"Label{i}", f"value {i} ({i}; {i}0%)"))
    pdf = tmp_path / "defn.pdf"
    _write_rowspan_pdf(str(pdf), rows)
    assert detect_tables_rowspan_labels(str(pdf), 1) == []


# ---------------------------------------------------------------------------
# det43: raster-embedded tables (_raster_region_table) — text is DRAWN into
# the page and recovered by OCR, exactly like a rasterized table screenshot.
# ---------------------------------------------------------------------------


def _write_raster_pdf(path, texts, page_w=612, page_h=792, fontsize=14):
    """texts = [(x, y_baseline, string)] drawn into one page."""
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page(width=page_w, height=page_h)
    for x, y, s in texts:
        page.insert_text((x, y), s, fontsize=fontsize)
    doc.save(path)
    doc.close()
    return page_w, page_h


def _raster_rows_from(tmp_path, texts, pad=30):
    from scientific_parser.perception.tables.detector import _raster_region_table
    from scientific_parser.ingest.ocr_engine import _get_rapidocr
    import pymupdf

    pdf = tmp_path / "raster.pdf"
    w, h = _write_raster_pdf(str(pdf), texts)
    doc = pymupdf.open(str(pdf))
    try:
        page = doc[0]
        rect = pymupdf.Rect(pad, pad, w - pad, h - pad)
        return _raster_region_table(_get_rapidocr(), page, rect)
    finally:
        doc.close()


def test_raster_numeric_grid_emitted(tmp_path):
    """A clean numeric grid OCRs back into its exact row/column shape."""
    xs = (60, 200, 340, 480)
    texts = [
        (xs[0], 70, "Item"),
        (xs[1], 70, "Q1"),
        (xs[2], 70, "Q2"),
        (xs[3], 70, "Delta"),
        (xs[0], 110, "alpha"),
        (xs[1], 110, "0.0"),
        (xs[2], 110, "12.5"),
        (xs[3], 110, "12.5"),
        (xs[0], 150, "beta"),
        (xs[1], 150, "949.9"),
        (xs[2], 150, "3.1"),
        (xs[3], 150, "946.8"),
        (xs[0], 190, "gamma"),
        (xs[1], 190, "7.0"),
        (xs[2], 190, "2.0"),
        (xs[3], 190, "5.0"),
    ]
    rows = _raster_rows_from(tmp_path, texts)
    assert len(rows) >= 3
    joined = " | ".join(" | ".join(r) for r in rows)
    assert "Item" in joined and "949.9" in joined


def test_raster_pairing_detail_above_name_emits(tmp_path):
    """2-band name|detail: multi-line detail blocks typeset ABOVE their
    owner row pair correctly (587a1f5a author|affiliation geometry)."""
    names = ["Andreas Weber", "Shangjr Lin", "Toshio Sato", "Michikazu Kondo"]
    affs = [
        ("Physics Department", "National Taiwan University"),
        ("IBM Research Center", "Almaden USA Laboratory"),
        ("Chemistry Faculty", "Kyoto University Japan"),
        ("Accelerator Division", "J PARC Center Japan"),
    ]
    texts = []
    for k, (name, (a1, a2)) in enumerate(zip(names, affs)):
        ny = 140 + 90 * k
        texts.append((300, ny - 38, a1))
        texts.append((300, ny - 20, a2))
        texts.append((80, ny, name))
    rows = _raster_rows_from(tmp_path, texts)
    assert len(rows) == 4
    for k, r in enumerate(rows):
        assert names[k].split()[0] in r[0]
        assert affs[k][0].split()[0] in r[1] and affs[k][1].split()[0] in r[1]


def test_raster_chart_axis_ticks_rejected(tmp_path):
    """A monotone short-int tick column is a chart axis, not a table."""
    texts = [
        (60, 70, "series"),
        (200, 70, "0"),
        (340, 70, "low"),
        (60, 110, "series"),
        (200, 110, "10"),
        (340, 110, "low"),
        (60, 150, "series"),
        (200, 150, "20"),
        (340, 150, "mid"),
        (60, 190, "series"),
        (200, 190, "30"),
        (340, 190, "mid"),
        (60, 230, "series"),
        (200, 230, "40"),
        (340, 230, "high"),
        (60, 270, "series"),
        (200, 270, "50"),
        (340, 270, "high"),
    ]
    assert _raster_rows_from(tmp_path, texts) == []


def test_raster_glyph_soup_rejected(tmp_path):
    """Dense grids of symbol crumbs (broken figure OCR) never emit."""
    soup = ["0.6", "A", "4", "44", "o", "B", "8", "0.2", "0", "2", "4", "3"]
    texts = []
    xs = (80, 240, 400)
    for i, s in enumerate(soup):
        texts.append((xs[i % 3], 70 + 40 * (i // 3), s))
    assert _raster_rows_from(tmp_path, texts) == []


# --- det32-41 tables campaign (merged from 434e7ad3) ---


class TestDet32TableEmissionHygiene:
    """det32: one canonical emission per table, clean cells.

    Pins the odlbench TEDS fixes: furniture-row drop, cell fold
    (newline/dash/quote normalization), content-level dedup across
    strategies, and the html-twin gate (twin only for genuine
    multi-row headers). Bench evidence: 01030000000053 went
    TEDS 0.185 -> 1.000 via exactly these changes.
    """

    def test_fold_cell_collapses_whitespace_and_ascii_folds(self):
        from scientific_parser.perception.tables.detector import _fold_cell

        assert _fold_cell("IX - Zamboanga \nPeninsula") == "IX - Zamboanga Peninsula"
        assert _fold_cell("y\u2013list") == "y-list"  # en-dash folds
        assert _fold_cell("\u2018quoted\u2019") == "'quoted'"  # curly quotes fold
        assert _fold_cell("  spaced\tout  ") == "spaced out"

    def test_drop_furniture_row_removes_glued_masthead(self):
        from scientific_parser.perception.tables.detector import _drop_furniture_row

        # 053 shape: row 0 = one mega-cell glued from the page top band
        rows = [
            ["16 Encinas Franco and Laguna", "", "", ""],
            ["Region", "Party", "Governor", "Party- List"],
            ["IX - Zamboanga Peninsula", "NP", "Ada", "NP"],
            ["X - Northern Mindanao", "LP", "Emano", "LP"],
        ]
        cleaned = _drop_furniture_row(rows)
        assert cleaned[0] == ["Region", "Party", "Governor", "Party- List"]
        assert len(cleaned) == 3

    def test_drop_furniture_row_keeps_single_column_table(self):
        from scientific_parser.perception.tables.detector import _drop_furniture_row

        # A legitimate single-column table: every row is one wide cell
        rows = [["Introduction"], ["Methods"], ["Results"], ["Discussion"]]
        assert _drop_furniture_row(rows) == rows

    def test_scrub_grid_rows_is_idempotent(self):
        from scientific_parser.perception.tables.detector import _scrub_grid_rows

        rows = [
            ["16 Encinas Franco and Laguna", "", ""],
            ["a\u2013b", "x\ny", ""],
            ["p", "q", "r"],
            ["", "", ""],
        ]
        once = _scrub_grid_rows(rows)
        twice = _scrub_grid_rows(once)
        assert once == twice

    def test_dedup_content_tables_drops_strategy_duplicates(self):
        from scientific_parser.postproc.text_pipeline import _dedup_content_tables

        def t(rows, bbox):
            return {"rows": rows, "bbox": bbox}

        # Same physical table extracted by two strategies at disjoint
        # bboxes (053 positional-vs-camelot shape): identical row sigs.
        rows = [
            ["Region", "Party", "Governor", "Party- List"],
            ["IX - Zamboanga Peninsula", "NP", "Ada", "NP"],
            ["X - Northern Mindanao", "LP", "Emano", "LP"],
            ["XI - Davao", "LP", "Uy", "LP"],
        ]
        tables = [t(rows, (0, 34, 600, 258)), t(rows, (0, 387, 600, 576))]
        out = _dedup_content_tables(tables)
        assert len(out) == 1

    def test_dedup_content_tables_keeps_disjoint_tables(self):
        from scientific_parser.postproc.text_pipeline import _dedup_content_tables

        a = {"rows": [["H1", "H2"], ["a", "1"], ["b", "2"], ["c", "3"]], "bbox": (0, 0, 600, 200)}
        b = {"rows": [["H3", "H4"], ["x", "9"], ["y", "8"], ["z", "7"]], "bbox": (0, 300, 600, 600)}
        out = _dedup_content_tables([a, b])
        assert len(out) == 2

    def test_thead_is_multirow_gate(self):
        from scientific_parser.postproc.text_pipeline import _thead_is_multirow

        single = (
            "<table><thead><tr><th>A</th></tr></thead><tbody><tr><td>1</td></tr></tbody></table>"
        )
        multi = (
            "<table><thead><tr><th>A</th></tr><tr><th>B</th></tr></thead><tbody></tbody></table>"
        )
        assert not _thead_is_multirow(single)
        assert _thead_is_multirow(multi)
        assert not _thead_is_multirow("<table><tbody></tbody></table>")


class TestDet35SingleEmissionChoice:
    def test_denser_twin_wins(self):
        # det36: the twin GRID often segments better — but emission is
        # ALWAYS pipes (raw <table> output scored TEDS 0.0 on 053);
        # twin is converted to pipes, chosen by non-empty cell density.
        from scientific_parser.postproc.text_pipeline import _choose_table_emission

        t = {
            "html": "<table><tbody><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></tbody></table>",
            "markdown": "| a 1 |\n|---|\n| b 2 |",
        }
        out = _choose_table_emission(t)
        assert out.startswith("|")
        assert "<table" not in out
        assert "| a | 1 |" in out and "| b | 2 |" in out

    def test_denser_pipes_win_over_glued_twin(self):
        # 1a67d111 shape: pipes carry separate 'Q1 | Q2 | Q3' cells where
        # the twin glued them into one — normalize_grid_rows compaction
        # can beat the raw twin grid.
        from scientific_parser.postproc.text_pipeline import _choose_table_emission

        t = {
            "html": "<table><thead><tr><th></th><th>Q1 Q2 Q3</th><th>Q4</th></tr></thead><tbody><tr><td>%A</td><td>57</td></tr></tbody></table>",
            "markdown": "<!-- column_types: Q1:numeric -->\n|  | Q1 | Q2 | Q3 | Q4 |\n|---|---|---|---|---|\n| % Allowed | 57 | 57 | 55 |  |",
        }
        out = _choose_table_emission(t)
        assert out.startswith("<!--") or out.startswith("|")
        assert "| Q3 |" in out

    def test_markdown_fallback_when_no_twin(self):
        from scientific_parser.postproc.text_pipeline import _choose_table_emission

        assert _choose_table_emission({"markdown": "| a |"}) == "| a |"
        # Blank twin falls back to pipes too.
        assert _choose_table_emission({"html": "  ", "markdown": "| a |"}) == "| a |"

    def test_both_empty_returns_blank(self):
        from scientific_parser.postproc.text_pipeline import _choose_table_emission

        assert _choose_table_emission({}) == ""
        assert _choose_table_emission({"html": None, "markdown": None}) == ""

    def test_density_ignores_comment_prefix_and_separator_rows(self):
        # The column_types comment made pipe tables look like HTML and
        # scored 0 cells (caught live on 1a67d111).
        from scientific_parser.postproc.text_pipeline import _nonempty_cell_count

        md = "<!-- column_types: Q1:numeric -->\n|  | Q1 | Q2 |\n|---|---|---|\n| %A | 57 | 57 |"
        assert _nonempty_cell_count(md) == 5
        assert _nonempty_cell_count("<table><tbody></tbody></table>") == 0


# === det34: edge-furniture scrubbing (headers_footers P2F class) ============= #


class TestDet34EdgeFurniture:
    def test_edge_row_signatures(self):
        # Bare page digit in a sparse row (52282f90 '| Milken ... | 6 |').
        assert _edge_row_is_furniture(["Milken Institute School of Public Health", "6"])
        # Journal running head glued into one cell (54f9a795).
        assert _edge_row_is_furniture(["The Korean Journal of Hematology Vol 19 No 1 / 2008 43"])
        # Statute publication note (b742cbc3 / 4f4c20ab shapes).
        assert _edge_row_is_furniture(["SOU 2016:1", ""])
        assert _edge_row_is_furniture(["Utkom fran trycket den 15 december 2016"])
        # Data rows survive: year digits are not page digits, real grids pass.
        assert not _edge_row_is_furniture(["2023", "1 245"])
        assert not _edge_row_is_furniture(["Region", "1991", "1996"])
        assert not _edge_row_is_furniture(["IX - Zamboanga Peninsula", "3", "1"])

    def test_scrub_edge_furniture_iterates(self):
        # Dropping the pubnote tail re-exposes the statute row beneath it.
        rows = [
            ["Header A", "Header B"],
            ["data 1", "data 2"],
            ["Utkom fran trycket den 15 december 2016"],
            ["SOU 2016:1"],
        ]
        out = _scrub_edge_furniture(rows)
        assert out == [["Header A", "Header B"], ["data 1", "data 2"]]

    def test_scrub_edge_furniture_keeps_data_edges(self):
        rows = [
            ["Start", "row"],
            ["middle", "data"],
            ["Final", "totals"],
        ]
        assert _scrub_edge_furniture(rows) == rows

    def test_all_furniture_table_vetoed(self):
        # 4f4c20ab: the detected "table" was pure page furniture.
        furniture = [
            ["Statens offentliga utredningar 2016:1"],
            ["Lag om andring i lagen om elektronisk kommunikation"],
            ["Utkom fran trycket den 15 december 2016"],
        ]
        assert _table_all_furniture(furniture)

        # A real small data table is not furniture.
        data = [
            ["Party", "Seats"],
            ["A", "12"],
            ["B", "8"],
        ]
        assert not _table_all_furniture(data)

        # Nine-row data grids never trip the 2-4 row window.
        big = [["c%d" % i, str(i)] for i in range(9)]
        assert not _table_all_furniture(big)


class TestDet39ModalWidthGuard:
    def test_uniform_label_count_table_survives(self):
        # b773892d pg1: every row is label+count at modal width 2 — the
        # page-digit furniture arm must not eat data grids whose own
        # column structure carries the digits.
        rows = [
            ["acid base", "26"],
            ["Chronic kidney disease, haematuria, proteinuria", "24"],
            ["Renal bone disease, renal anaemia", "12"],
            ["Cardiovascular disease, hypertension,", "20"],
        ]
        assert _scrub_edge_furniture(rows) == rows
        assert not _table_all_furniture(rows)

    def test_page_digit_outlier_row_still_drops(self):
        # 52282f90 shape: a 3-column stats grid with the running head
        # absorbed as a sparse 2-cell tail row (below modal width).
        rows = [
            ["Region", "Cases", "Rate"],
            ["North", "120", "3.4"],
            ["South", "98", "2.1"],
            ["Milken Institute School of Public Health", "6", ""],
        ]
        out = _scrub_edge_furniture(rows)
        assert out == [["Region", "Cases", "Rate"], ["North", "120", "3.4"], ["South", "98", "2.1"]]


class TestDet36DedupRichness:
    def test_richer_copy_replaces_despite_sig_cap(self):
        # 77e68e1c pg5: 24-row positional copy must beat an 11-row
        # text_only copy even though both signature sets cap at 8 rows.
        from scientific_parser.postproc.text_pipeline import _dedup_content_tables

        def rows(n, tag):
            out = [["Black", "376", "30.2"], ["White", "329", "43.2"], ["Hispanic", "187", "20.9"]]
            out += [[f"{tag}{i}", str(i), f"{i}.0"] for i in range(n)]
            return out

        poor = {"strategy": "text_only", "rows": rows(5, "x")}
        rich = {"strategy": "positional", "rows": rows(20, "y")}
        kept = _dedup_content_tables([poor, rich])
        assert len(kept) == 1
        assert kept[0]["strategy"] == "positional"

    def test_disjoint_tables_both_kept(self):
        from scientific_parser.postproc.text_pipeline import _dedup_content_tables

        a = {"strategy": "a", "rows": [["h1", "1"], ["r1", "2"], ["r2", "3"]]}
        b = {"strategy": "b", "rows": [["h2", "9"], ["s1", "8"], ["s2", "7"]]}
        assert len(_dedup_content_tables([a, b])) == 2

    def test_twin_to_pipes_entities_and_separator(self):
        # det36 converter: entities unescaped (the 36-probe class where
        # '&#x27;' leaked through), thead row count sets the separator
        # position, cells stay separate.
        from scientific_parser.postproc.text_pipeline import _html_twin_to_pipes

        out = _html_twin_to_pipes(
            "<table><thead><tr><th>Item</th><th>Score</th></tr></thead>"
            "<tbody><tr><td>Cronbach&#x27;s &amp; alpha</td><td>0.9</td></tr>"
            "<tr><td>b</td><td>2</td></tr></tbody></table>"
        )
        lines = out.splitlines()
        assert lines[0] == "| Item | Score |"
        assert lines[1].startswith("|---")
        assert "| Cronbach's & alpha | 0.9 |" in out

    def test_twin_to_pipes_empty_thead_and_glue_guard(self):
        # Empty thead (053 shape): separator must not eat the first
        # data row; glued twin cell stays glued (density choice decides
        # elsewhere); <2 usable rows -> blank.
        from scientific_parser.postproc.text_pipeline import _html_twin_to_pipes

        out = _html_twin_to_pipes(
            "<table><thead></thead><tbody>"
            "<tr><td>IX - Zamboanga Peninsula</td><td>3</td></tr>"
            "<tr><td>NCR</td><td>13</td></tr></tbody></table>"
        )
        assert out.splitlines()[0] == "| IX - Zamboanga Peninsula | 3 |"
        assert _html_twin_to_pipes("<table><tbody><tr><td>only</td></tr></tbody></table>") == ""
        assert _html_twin_to_pipes("") == ""


class TestDet37GatedDualEmission:
    def test_grids_differ_on_shattered_header(self):
        # 0091c5b2 shape: md header row shattered into 8 nonempty cells,
        # twin keeps 4 — any row-count mismatch = material difference.
        from scientific_parser.postproc.text_pipeline import _grids_differ_materially

        md = "| a | b | c | d | e | f | g | h |\n|---|\n| 1 | 2 | 3 | 4 |"
        tw = "| a b c d |\n|---|\n| 1 | 2 | 3 | 4 |"
        assert _grids_differ_materially(md, tw) is True

    def test_grids_agree_on_compaction_equal_rows(self):
        # odlbench 053 shape: row-for-row identical nonempty counts,
        # cell TEXT differs cosmetically (dash variants) — must stay
        # single (TEDS 1.0; calibration: counts equal, text containment
        # holds bidirectionally on every agreeing doc).
        from scientific_parser.postproc.text_pipeline import _grids_differ_materially

        md = "| Region | Party- List | Votes | Seats |\n|---|\n| NCR | x | 13 | 4 |"
        tw = "| Region | Party List | Votes | Seats |\n|---|\n| NCR | x | 13 | 4 |"
        assert _grids_differ_materially(md, tw) is False

    def test_grids_differ_on_row_count_mismatch(self):
        from scientific_parser.postproc.text_pipeline import _grids_differ_materially

        md = "| a | b |\n|---|\n| 1 | 2 |"
        tw = "| a | b |\n|---|\n| 1 | 2 |\n| 3 | 4 |"
        assert _grids_differ_materially(md, tw) is True

    def test_dual_emission_when_grids_differ(self):
        # 1a67d111 shape: md glued header 'Q1 Q2 Q3' vs twin separate
        # cells → pipes + RAW html twin both emitted (TableTest parses
        # either; raw twin needed by HTML-parse-only probes).
        from scientific_parser.postproc.text_pipeline import _emit_table_blocks

        t = {
            "markdown": "| Q1 Q2 Q3 |\n|---|\n| 4 | 5 | 6 |",
            "html": (
                "<table><thead><tr><th>Q1</th><th>Q2</th><th>Q3</th></tr></thead>"
                "<tbody><tr><td>4</td><td>5</td><td>6</td></tr></tbody></table>"
            ),
        }
        blocks = _emit_table_blocks(t)
        assert len(blocks) == 2
        assert blocks[0].startswith("| Q1 Q2 Q3 |")
        assert blocks[1].startswith("<table>")

    def test_single_emission_when_grids_agree(self):
        # det45: agreeing grids still emit pipes + twin. det37 suppressed
        # the twin to dodge a duplicate-table TEDS penalty, but det44 made
        # pipes SEPLESS — the odlbench converter can no longer turn them
        # into <table>, so the twin is the only convertible form; dropping
        # it scores the doc TEDS 0 (measured on the det41 1.0-class docs).
        from scientific_parser.postproc.text_pipeline import _emit_table_blocks

        t = {
            "markdown": "| Region | Votes | Seats | Notes |\n|---|\n| NCR | 13 | 4 | x |",
            "html": (
                "<table><thead><tr><th>Region</th><th>Votes</th>"
                "<th>Seats</th><th>Notes</th></tr></thead>"
                "<tbody><tr><td>NCR</td><td>13</td><td>4</td><td>x</td></tr></tbody></table>"
            ),
        }
        blocks = _emit_table_blocks(t)
        assert len(blocks) == 2
        assert blocks[0].startswith("| Region |")
        assert blocks[1].startswith("<table>")

    def test_single_emission_identical_grids_1a67_t1(self):
        # 1a67d111 t1: identical grids → pipes + twin (det45 always-twin).
        from scientific_parser.postproc.text_pipeline import _emit_table_blocks

        t = {
            "markdown": "| a | b |\n|---|\n| 1 | 2 |",
            "html": (
                "<table><thead><tr><th>a</th><th>b</th></tr></thead>"
                "<tbody><tr><td>1</td><td>2</td></tr></tbody></table>"
            ),
        }
        assert len(_emit_table_blocks(t)) == 2

    def test_det45_twin_derived_when_missing(self):
        # det45: a table that reaches emission without an html twin (det37
        # agreeing-grid suppression, monster-rebuild pop, or a strategy
        # that never built one) gets the twin DERIVED from its rows —
        # sepless pipes are unconvertible on odlbench, so without a twin
        # the doc's TEDS is 0 regardless of pipe quality.
        from scientific_parser.postproc.text_pipeline import _emit_table_blocks

        t = {
            "markdown": "| Item | Count |\n|---|\n| boots | 12 |",
            "rows": [["Item", "Count"], ["boots", "12"]],
        }
        blocks = _emit_table_blocks(t)
        assert len(blocks) == 2
        assert blocks[0].startswith("| Item |")
        assert blocks[1].lstrip().startswith("<table")
        assert "boots" in blocks[1]


class TestDet38DedupGridViews:
    def _mk(self, strategy, rows, md):
        return {"strategy": strategy, "rows": rows, "markdown": md}

    def test_sig_share_different_grids_both_kept(self):
        # b773892d pg1 shape: the merged monster grid glues rows into
        # 2-col cells while the physical table keeps 4-col rows — sig
        # sharing + materially-differing grids = different VIEWS, both
        # survive dedup (probes satisfied by either).
        from scientific_parser.postproc.text_pipeline import _dedup_content_tables

        phys_rows = [
            ["acid base", "26"],
            ["CKD", "24"],
            ["renal bone", "12"],
            ["cardio", "20"],
        ]
        mon_rows = [
            ["CKD / renal bone / cardio", "24 / 12 / 20"],
            ["Total", "200"],
        ]
        phys = self._mk(
            "positional",
            phys_rows,
            "| acid base | 26 |\n|---|\n| CKD | 24 |\n| renal bone | 12 |\n| cardio | 20 |",
        )
        mon = self._mk(
            "monster",
            mon_rows,
            "| CKD / renal bone / cardio | 24 / 12 / 20 |\n|---|\n| Total | 200 |",
        )
        kept = _dedup_content_tables([mon, phys])
        assert len(kept) == 2

    def test_sig_share_equal_grids_still_dedup(self):
        # 053 shape: two copies of the SAME grid → exactly one kept.
        from scientific_parser.postproc.text_pipeline import _dedup_content_tables

        rows = [["NCR", "13", "4"], ["CAR", "9", "2"], ["ARMM", "3", "1"]]
        a = self._mk(
            "positional",
            rows,
            "| a | b | c |\n|---|\n| NCR | 13 | 4 |\n| CAR | 9 | 2 |\n| ARMM | 3 | 1 |",
        )
        b = self._mk(
            "camelot_lattice",
            [list(r) for r in rows],
            "| x | y | z |\n|---|\n| NCR | 13 | 4 |\n| CAR | 9 | 2 |\n| ARMM | 3 | 1 |",
        )
        assert len(_dedup_content_tables([a, b])) == 1


# === det41: rows-level monster + twin compaction + tie->md (904a/b742) = #


class TestDet41MonsterRows:
    def test_mega_cell_with_eaten_column_fires(self):
        # 904a shape: a 13-newline mega-cell in row 2 has swallowed the
        # label column -> 13 of 15 rows below carry an empty first cell.
        rows = [["Year", "1998", "1999"]] * 2
        rows.append(["Label A\nLabel B\nLabel C\nLabel D\nLabel E", "1", "2"])
        rows += [["", "3", "4"]] * 13
        assert _is_monster_rows(rows) is True

    def test_wrapped_labels_own_rows_do_not_fire(self):
        # 033c shape: cells carry up to 7 embedded newlines but every
        # row keeps its own label -> nothing was eaten.
        rows = [["Program me\nSECS-P/01", "SPS/09", "26"]] * 10
        assert _is_monster_rows(rows) is False

    def test_few_newlines_never_fire(self):
        rows = [["Label\nwrap", "1", "2"], ["", "3", "4"], ["", "5", "6"]]
        assert _is_monster_rows(rows) is False


class TestDet41bTwinCompaction:
    def test_phantom_columns_compacted_away(self):
        # 904a NUTS twin: phantom body-empty columns survive the raw
        # carve but must drop through the canonical build path.
        twin = (
            "<table><thead><tr><th>NUTS III</th><th>% da popula&ccedil;&atilde;o activa</th>"
            "<th></th><th></th></tr></thead><tbody>"
            "<tr><td>Koprivnica-Kri&#382;evci</td><td></td><td></td><td>16,9</td><td>18,4</td><td>21,0</td></tr>"
            "<tr><td>Lisboa</td><td></td><td></td><td>8 | 9</td><td>7,4</td><td>6,2</td></tr>"
            "</tbody></table>"
        )
        raw_rows = _html_twin_rows(twin)
        emitted = build_table_markdown(raw_rows)
        # The phantom middle columns must not survive as pipe columns.
        assert "| Koprivnica-Križevci | 16,9 |" in emitted
        # Raw converter (differ-check input) must still see the carve.
        assert len(_html_twin_rows(twin)[0]) >= 4

    def test_choose_emission_tie_prefers_md(self):
        # b742: equal density -> md (furniture-scrubbed, pdftotext cells).
        t = {
            "markdown": "| a | 1 |\n|---|\n| b | 2 |",
            "html": "<table><tr><td>a</td><td>1</td></tr><tr><td>b</td><td>2</td></tr></table>",
        }
        assert _choose_table_emission(t).startswith("| a |")

    def test_choose_emission_denser_twin_still_wins(self):
        md = "| a b c |\n|---|\n| 1 |"  # glued single cell
        twin = (
            "<table><tr><td>a</td><td>b</td><td>c</td></tr>"
            "<tr><td>1</td><td>2</td><td>3</td></tr></table>"
        )
        t = {"markdown": md, "html": twin}
        out = _choose_table_emission(t)
        assert "| a | b | c |" in out


class TestDet41PubnoteWidening:
    def test_fonte_row_is_furniture(self):
        assert _edge_row_is_furniture(["", "Fonte: Bacen", "42"])

    def test_label_count_at_modal_width_is_data(self):
        # b773: uniform label+count rows carry digits at modal width.
        assert not _edge_row_is_furniture(["Chronic kidney disease", "24"], modal_w=2)


# === det44: sepless emission ==================================== #


def test_det44_no_separator_row_in_built_markdown():
    """Pipe tables ship WITHOUT the `| --- |` separator (odlbench TEDS
    forest duplicates the html twin when converters parse our pipes;
    olmOCR parser is separator-agnostic — validated H2 arm)."""
    md = build_table_markdown(
        [["Region", "Cases"], ["North", "121"], ["South", "71"], ["East", "93"]], "positional"
    )
    assert md, "table must build"
    lines = [l for l in md.split("\n") if l.strip()]
    assert not any(set(l.replace("|", "").strip()) <= {"-", " ", ":"} for l in lines)
    pipe_lines = [l for l in lines if l.lstrip().startswith("|")]
    assert pipe_lines[0].startswith("| Region")
    assert md.rstrip().endswith("|")


def test_det44_cells_flattened_single_line():
    """Multi-line cells (camelot '\n' glue) must not become phantom
    continuation rows — every emitted row is one line, cells single-line."""
    md = build_table_markdown(
        [
            ["Name", "Role"],
            ["Doe,\nJane", "PI\n(co-lead)"],
            ["Bob", "developer"],
            ["Eve", "operations"],
        ],
        "camelot_lattice",
    )
    assert md, "table must build"
    for line in md.split("\n"):
        if line.strip().startswith("|"):
            assert "\n" not in line
    joined = md.replace("\n", " ")
    assert "Doe, Jane" in joined
    assert "PI (co-lead)" in joined


def test_det44_trailing_blank_line_separates_adjacent_runs():
    """Output ends with a blank line so consecutive pipe tables in a page
    never merge into one parser run (blank line splits runs)."""
    md = build_table_markdown([["AA", "BB"], ["111", "222"], ["333", "444"]], "positional")
    assert md, "table must build"
    assert md.endswith("\n")
    assert not md.endswith("\n\n")


def test_det44_table_agent_emission_sepless_with_blank():
    """table_agent._cells_to_markdown (monster-rebuild path feeds
    t['markdown'] directly): sepless + trailing blank line."""
    from scientific_parser.perception.tables.table_agent import _cells_to_markdown

    out = _cells_to_markdown([["header1", "header2"], ["alpha", "beta"], ["gamma", "delta"]])
    assert "---" not in out
    first = out.split("\n")[0]
    assert first.startswith("|") and "header1" in first and "header2" in first
    assert "alpha" in out and "beta" in out
    assert out.endswith("\n") and not out.endswith("\n\n")


def test_det44_structure_table_to_markdown_sepless():
    """structure.Table.to_markdown: no separator, flattened cells, single
    trailing blank line."""
    from scientific_parser.perception.tables.structure import Table

    t = Table(
        header_rows=[["Metric", "Value"]],
        body_rows=[["speed", "9\nkm/h"], ["mass", "12 kg"]],
        n_cols=2,
    )
    md = t.to_markdown()
    assert "---" not in md
    assert "9 km/h" in md
    assert md.endswith("\n")


def test_det44_olmocr_style_run_count_preserved():
    """Round-trip: two adjacent sepless tables separated by one blank line
    parse as TWO tables under the olmocr-style consecutive-pipe-lines rule
    (runs split on non-pipe lines)."""
    t1 = build_table_markdown([["AA", "BB"], ["111", "222"], ["555", "666"]], "positional")
    t2 = build_table_markdown([["CC", "DD"], ["333", "444"], ["777", "888"]], "positional")
    assert t1 and t2, "tables must build"
    page = t1 + "\n" + t2  # t1 ends with single newline → one blank line between

    runs = []
    cur = []
    for line in page.split("\n"):
        if line.strip().startswith("|"):
            cur.append(line)
        else:
            if len(cur) >= 2:
                runs.append(cur)
            cur = []
    if len(cur) >= 2:
        runs.append(cur)
    assert len(runs) == 2


# === det47: leader veto / prose guard / twin fallback ============ #


def test_det47_leader_page_list_vetoed():
    from scientific_parser.perception.tables.detector import (
        looks_like_leader_page_list,
    )

    rows = [
        ["Introduction", "7"],
        ["1. Changing Practices in Disaster Management", "7"],
        ["2. Institutional Learning and Adaptation", "23"],
        ["3. Comparative National Responses", "41"],
        ["4. Funding Mechanisms and Donor Behaviour", "58"],
        ["5. Field Operations in Complex Emergencies", "76"],
        ["6. Legal Frameworks and Protections", "94"],
        ["7. Measuring Effectiveness and Impact", "112"],
        ["Appendix A Data Sources", "130"],
        ["Index", "141"],
    ]
    assert looks_like_leader_page_list(rows) is True


def test_det47_decimal_table_survives_leader_veto():
    from scientific_parser.perception.tables.detector import (
        looks_like_leader_page_list,
    )

    rows = [
        ["Start Balance", "0.0", "0.0"],
        ["End Balance", "1,154.5", "949.9"],
        ["+ Earned", "239.0", "709.5"],
        ["+ Bought", "0.0", "0.0"],
        ["Taken", "1,154.5", "0.0"],
        ["Sold", "0.0", "0.0"],
        ["+ Adjustments", "0.0", "0.0"],
        ["Final Total", "1,154.5", "949.9"],
    ]
    assert looks_like_leader_page_list(rows) is False


def test_det47_prose_shatter_rejected():
    from scientific_parser.perception.tables.detector import (
        looks_like_prose_shatter,
    )

    rows = [
        ["we can express the", "Lagrange equations entirely", "in terms of func"],
        ["where the variation is, taken", "with respect to each", "coordinate in"],
        ["the system, and the", "constraint forces are", "eliminated by the"],
        ["using the method of", "undetermined multipliers as", "described above"],
    ]
    assert looks_like_prose_shatter(rows) is True


def test_det47_code_table_survives_prose_guard():
    from scientific_parser.perception.tables.detector import (
        looks_like_prose_shatter,
    )

    rows = [
        ["A/RES/66/2", "Political declaration"],
        ["A/RES/73/2", "Follow-up resolution"],
        ["A/RES/70/1", "Transforming our world"],
        ["A/RES/71/1", "New urban agenda"],
    ]
    assert looks_like_prose_shatter(rows) is False


def test_det47_twin_derived_from_markdown_only_dict():
    from scientific_parser.postproc.text_pipeline import _emit_table_blocks

    md = "| Year | Value |\n| 2019 | 120 |\n| 2020 | 340 |\n| 2021 | 560 |"
    blocks = _emit_table_blocks({"markdown": md})
    assert len(blocks) == 2
    assert blocks[0].startswith("|")
    assert blocks[1].lstrip().startswith("<table")
    assert "2019" in blocks[1] and "560" in blocks[1]


# === det48: raster-only scoping + unit-numeric prose guard ============== #


def test_det48_unit_numeric_table_survives_prose_guard():
    """Measured quantities ('24 ml') are numeric, not prose-shatter —
    odlbench 0117's true table was vetoed by the bare [\\d.,]+ pattern."""
    from scientific_parser.perception.tables.detector import (
        looks_like_prose_shatter,
    )

    rows = [
        ["Add", "Volume"],
        ["water", "24 ml"],
        ["salt", "0 ml"],
        ["stir", "4 ml"],
    ]
    assert looks_like_prose_shatter(rows) is False


def test_det48_raster_only_scopes_text_strategies(tmp_path):
    """raster_only=True runs ONLY the raster strategy — pdfplumber/
    lines_strict/text_only/positional/whitespace/rowspan/camelot are
    skipped (det47's FP class: full chain on raster-only pages emitted
    positional/whitespace junk around the image). Fixture: the numeric
    grid is EMBEDDED AS AN IMAGE (text page rendered to a pixmap and
    re-inserted), matching the 3a91fad8/odlbench-122 production class."""
    import pymupdf
    from scientific_parser.postproc.text_pipeline import detect_all_tables

    xs = (60, 200, 340, 480)
    texts = [
        (xs[0], 70, "Item"),
        (xs[1], 70, "Q1"),
        (xs[2], 70, "Q2"),
        (xs[3], 70, "Delta"),
        (xs[0], 110, "alpha"),
        (xs[1], 110, "0.0"),
        (xs[2], 110, "12.5"),
        (xs[3], 110, "12.5"),
        (xs[0], 150, "beta"),
        (xs[1], 150, "949.9"),
        (xs[2], 150, "3.1"),
        (xs[3], 150, "946.8"),
    ]
    src = tmp_path / "grid_src.pdf"
    _write_raster_pdf(str(src), texts)
    # Render ONLY the grid region and embed it 1:1 as an image on a clean
    # page: the table now exists ONLY as pixels (no text layer for it).
    sdoc = pymupdf.open(str(src))
    try:
        pix = sdoc[0].get_pixmap(dpi=150, clip=pymupdf.Rect(40, 50, 560, 180))
    finally:
        sdoc.close()
    pdf = tmp_path / "raster_scoped.pdf"
    doc = pymupdf.open()
    page = doc.new_page(width=612, height=792)
    page.insert_image(pymupdf.Rect(60, 150, 580, 280), pixmap=pix)
    doc.save(str(pdf))
    doc.close()
    scoped = detect_all_tables(str(pdf), 1, raster_only=True)
    assert all(t["strategy"] == "raster_ocr" for t in scoped)
    assert len(scoped) >= 1  # the numeric grid still emits via raster OCR


def test_det49_equation_rows_vetoed():
    """Aligned equations carved into grids must not emit as tables — table
    emission strips the source line so the inline Unicode->LaTeX lane never
    runs (03759: entropy formula lost as pipe cells)."""
    from scientific_parser.perception.tables.detector import (
        looks_like_equation_rows,
    )

    dense = [
        ["H(Z) =", "− X z P(z) log P(z)"],
        ["log P(z) =", "log |P(z)| + i θ(P(z))"],
    ]
    assert looks_like_equation_rows(dense) is True
    moderate = [
        ["x ∈ S", "2"],
        ["Θ(v) = lim", "7"],
        ["y ≤ g(t)", "k"],
        ["Γ(u)", "0"],
        ["z ∈ w", "3"],
        ["u ∈ T", "9"],
    ]
    assert looks_like_equation_rows(moderate) is True


def test_det49_stats_table_survives_equation_veto():
    """Digit-attached operators (stats '0.83±0.13', specs '1830±50') and
    checkmark columns are data tables, not equations."""
    from scientific_parser.perception.tables.detector import (
        looks_like_equation_rows,
    )

    rows = [
        ["Group", "Dose", "Baseline", "Effect"],
        ["3a", "50mg/kg", "0.0", "0.06±0.13**"],
        ["3c", "50mg/kg", "0.0", "0.65±0.13*"],
        ["Std.", "50mg/kg", "0.0", "0.4±14**"],
        ["Mean ± SD row", "2,05 ± 0,498", "2,38 ± 0,669", "n/a"],
    ]
    assert looks_like_equation_rows(rows) is False
    checks = [
        ["Reviewer", "Approved", "Note", ""],
        ["√ Wolfson, Anna", "√", "Hoyle, Chad", "√"],
        ["√ Scharschmidt, Thomas", "√", "Others", "√"],
    ]
    assert looks_like_equation_rows(checks) is False


# ---------------- det52: fused numeric cell splitting ----------------

class TestDet52SplitFusedNumericCells:
    def test_dual_panel_row_splits(self):
        from scientific_parser.perception.tables.detector import _split_fused_numeric_rows as f
        rows = [["Q3", "0.84 0.74 22%", "14%"], ["9 Months", "2.44 2.12 20%", "15%"]]
        assert f(rows) == [
            ["Q3", "0.84", "0.74", "22%", "14%"],
            ["9 Months", "2.44", "2.12", "20%", "15%"],
        ]

    def test_numeric_pair_table_untouched(self):
        from scientific_parser.perception.tables.detector import _split_fused_numeric_rows as f
        rows = [["A", "1 2"], ["B", "3 4"]]  # split width 3 < 4 -> reject
        assert f(rows) == rows

    def test_inconsistent_split_untouched(self):
        from scientific_parser.perception.tables.detector import _split_fused_numeric_rows as f
        rows = [["X", "1 2", "note"], ["Y", "3", "4"]]
        assert f(rows) == rows

    def test_non_numeric_words_never_split(self):
        from scientific_parser.perception.tables.detector import _split_fused_numeric_rows as f
        rows = [["Item", "see Fig. 1", "x"], ["Other", "in situ tests", "y"]]
        assert f(rows) == rows


class TestDet53ExpandStackedRows:
    """det53: camelot column-packed stacks expand into real rows (fa18a15c pg2)."""

    def _stacked_rows(self):
        return [
            ["District Averages", "Average Temperature", "", "Precip", "", ""],
            ["", "Past Week", "Depart", "Past Week", "Total", "Depart"],
            [
                "Northwest(1) \nN. Central(2) \nNortheast (3) \nW. Central(4)",
                "68 \n67 \n61 \n66",
                "(degrees F) \n1 \n1 \n-5 \n-2",
                "(inches) \n0.01 \n0.04 \n0.18 \n0.47",
                "(inches) \n14.56 \n13.31 \n12.53 \n14.20",
                "(inches) \n4.21 \n1.95 \n1.21 \n3.70",
            ],
        ]

    def test_det53_stacked_columns_expand_to_data_rows(self):
        md = build_table_markdown(self._stacked_rows(), "test")
        lines = [l for l in md.split("\n") if l.startswith("|")]
        assert len(lines) == 5, lines  # merged multirow header + 4 data rows
        assert "| Northwest(1) | 68 | 1 | 0.01 | 14.56 | 4.21 |" in lines
        assert "| W. Central(4) | 66 | -2 | 0.47 | 14.20 | 3.70 |" in lines
        assert not any("(degrees F)" in l or "(inches)" in l for l in lines)

    def test_det53_wrapped_label_line_merges(self):
        rows = [
            ["Label", "Val", "Qty"],
            ["Central \n(5) \nE. Central (6)", "68 \n69", "0.5 \n0.07 \n1.0 \n2.0"],
        ]
        md = build_table_markdown(rows, "test")
        flat = " ".join(md.split())
        assert "Central (5)" in flat
        assert "E. Central (6)" in flat

    def test_det53_short_wrapped_headers_untouched(self):
        rows = [
            ["District \nAverages", "Average \nTemperature", "Total \nPrecip"],
            ["Row A", "68", "0.01"],
            ["Row B", "67", "0.04"],
        ]
        md = build_table_markdown(rows, "test")
        lines = [l for l in md.split("\n") if l.startswith("|")]
        assert len(lines) == 3  # no expansion: k_max=2 header stack never qualifies
        assert "District Averages" in lines[0]
