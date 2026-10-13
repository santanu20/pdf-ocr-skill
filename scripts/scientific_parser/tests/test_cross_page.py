"""Unit tests for cross-page entity stitching (reviewer #28).

Run: pytest scripts/scientific_parser/tests/test_cross_page.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.cross_page import (
    build_cross_page_registry,
    cross_page_entities,
    extract_label_occurrences,
)

# === continuation detection ====================================== #


def test_detects_table_continued():
    occ = extract_label_occurrences("Table 2 (continued)\nSiO2 52.3", page=5)
    tables = [o for o in occ if o.kind == "table"]
    assert len(tables) == 1
    assert tables[0].number == "2"
    assert tables[0].is_continuation is True


def test_detects_cont_abbreviation():
    occ = extract_label_occurrences("Table 3 cont.", page=2)
    assert any(o.is_continuation and o.number == "3" for o in occ)


def test_detects_plain_label_not_continuation():
    occ = extract_label_occurrences("As shown in Table 2 ...", page=4)
    t = [o for o in occ if o.kind == "table"]
    assert len(t) == 1
    assert t[0].is_continuation is False


def test_detects_equation_reference():
    occ = extract_label_occurrences("Substituting into Eq. (12) gives", page=9)
    eqs = [o for o in occ if o.kind == "equation"]
    assert len(eqs) == 1
    assert eqs[0].number == "12"


def test_detects_figure_caption():
    occ = extract_label_occurrences("Figure 5. Geological map of the area.", page=3)
    figs = [o for o in occ if o.kind == "figure"]
    assert len(figs) == 1
    assert figs[0].number == "5"


# === cross-page registry ========================================= #


def test_continued_table_links_pages():
    pages = [
        "Table 2. Major element geochemistry.\nSiO2 52.3",
        "Table 2 (continued)\nFeO 8.1",
    ]
    ents = build_cross_page_registry(pages)
    t2 = [e for e in ents if e.kind == "table" and e.number == "2"]
    assert len(t2) == 1
    assert t2[0].is_cross_page
    assert t2[0].has_continuation
    assert t2[0].spans_pages == [1, 2]


def test_back_reference_links_pages():
    pages = [
        "Equation (12) defines the age model.",  # page 1
        "",  # page 2 empty
        "Recall Eq. 12 from earlier.",  # page 3
    ]
    ents = cross_page_entities(pages)
    eq12 = [e for e in ents if e.kind == "equation" and e.number == "12"]
    assert len(eq12) == 1
    assert eq12[0].spans_pages == [1, 3]


def test_single_page_label_not_cross_page():
    pages = ["Figure 1 shows the outcrop."]
    ents = cross_page_entities(pages)
    assert ents == []  # not cross-page


def test_distinct_numbers_are_distinct_entities():
    pages = ["Table 1\nTable 2"]
    ents = build_cross_page_registry(pages)
    nums = sorted(e.number for e in ents if e.kind == "table")
    assert nums == ["1", "2"]


def test_empty_pages_safe():
    assert build_cross_page_registry([]) == []
    assert build_cross_page_registry(["", ""]) == [] or all(
        not e.is_cross_page for e in build_cross_page_registry(["", ""])
    )


def test_kind_normalization():
    pages = ["Fig 1", "Figure 2", "Table 1", "Eq. 3", "Equation 4"]
    ents = build_cross_page_registry(pages)
    kinds = {(e.kind, e.number) for e in ents}
    assert ("figure", "1") in kinds
    assert ("figure", "2") in kinds
    assert ("table", "1") in kinds
    assert ("equation", "3") in kinds
    assert ("equation", "4") in kinds
