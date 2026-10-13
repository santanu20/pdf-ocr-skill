"""Unit tests for the weighted error metric (reviewer #30).

Verifies the metric ranks scientific correctness over raw character
accuracy: a decimal error must outweigh an ordinary char error.

Run: pytest scripts/scientific_parser/tests/test_weighted_error.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from failure_taxonomy import (
    GEOLOGICAL_MEASUREMENT_FACTOR,
    LABEL_SEVERITY,
    OCR_CHAR,
    OCR_FORMULA_KATEX_FAIL,
    OCR_NUMBER_DECIMAL_LOST,
    READING_ORDER,
    TABLE_STRUCTURE,
    UNIT_OUT_OF_RANGE,
    severity_for,
    weighted_error_breakdown,
    weighted_error_rate,
    weighted_error_score,
)

# === severity_for ================================================ #


def test_severity_char_is_lowest():
    assert severity_for(OCR_CHAR) == 1


def test_severity_numeric_beats_formula_beats_table():
    assert severity_for(OCR_NUMBER_DECIMAL_LOST) == 10
    assert severity_for(OCR_FORMULA_KATEX_FAIL) == 8
    assert severity_for(TABLE_STRUCTURE) == 6
    assert severity_for(READING_ORDER) == 4


def test_severity_unit_is_max_base():
    assert severity_for(UNIT_OUT_OF_RANGE) == 10


def test_severity_unknown_label_defaults_to_1():
    assert severity_for("SOME_UNKNOWN_LABEL") == 1


# === weighted_error_score ======================================== #


def test_decimal_error_outweighs_char_error():
    # 1 decimal error (x10) vs 9 char errors (x1) -> decimal still loses
    # the comparison (parser with the decimal error scores worse).
    decimal_doc = {OCR_NUMBER_DECIMAL_LOST: 1}
    char_doc = {OCR_CHAR: 9}
    assert weighted_error_score(decimal_doc) == 10
    assert weighted_error_score(char_doc) == 9
    assert weighted_error_score(decimal_doc) > weighted_error_score(char_doc)


def test_empty_counts_zero():
    assert weighted_error_score({}) == 0


def test_geo_labels_double_numeric_weight():
    # numeric base x10, geo factor x2 -> x20 per reviewer #30
    counts = {OCR_NUMBER_DECIMAL_LOST: 1}
    base = weighted_error_score(counts)
    geo = weighted_error_score(counts, geo_labels={OCR_NUMBER_DECIMAL_LOST})
    assert geo == base * GEOLOGICAL_MEASUREMENT_FACTOR
    assert geo == 20


def test_mixed_document_total():
    counts = {OCR_CHAR: 5, OCR_NUMBER_DECIMAL_LOST: 2, TABLE_STRUCTURE: 1}
    # 5*1 + 2*10 + 1*6 = 31
    assert weighted_error_score(counts) == 31


# === weighted_error_breakdown ==================================== #


def test_breakdown_sums_per_axis():
    counts = {OCR_CHAR: 2, OCR_NUMBER_DECIMAL_LOST: 1, TABLE_STRUCTURE: 1}
    bd = weighted_error_breakdown(counts)
    assert bd["OCR"] == 2 * 1 + 1 * 10  # 12
    assert bd["TABLE"] == 1 * 6  # 6
    assert bd["EQUATION"] == 0


def test_breakdown_all_axes_present():
    bd = weighted_error_breakdown({})
    from failure_taxonomy import AXES

    assert set(bd.keys()) == set(AXES.keys())
    assert all(v == 0 for v in bd.values())


# === weighted_error_rate ========================================= #


def test_rate_normalizes_by_units():
    counts = {OCR_NUMBER_DECIMAL_LOST: 4}  # score 40
    assert weighted_error_rate(counts, 10) == 4.0
    assert weighted_error_rate(counts, 1) == 40.0


def test_rate_zero_units_safe():
    assert weighted_error_rate({OCR_CHAR: 5}, 0) == 0.0


# === LABEL_SEVERITY sanity ======================================= #


def test_all_numeric_labels_have_high_severity():
    from failure_taxonomy import (
        OCR_NUMBER,
        OCR_NUMBER_EXPONENT_FLIP,
        OCR_NUMBER_PERCENT_LOST,
        OCR_NUMBER_SCIENTIFIC_BAD,
        TABLE_VALUE,
    )

    for lbl in (
        OCR_NUMBER,
        OCR_NUMBER_EXPONENT_FLIP,
        OCR_NUMBER_PERCENT_LOST,
        OCR_NUMBER_SCIENTIFIC_BAD,
        TABLE_VALUE,
    ):
        assert LABEL_SEVERITY[lbl] == 10
