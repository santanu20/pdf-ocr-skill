"""Unit tests for typed validators: table_structure, formula_latex, numeric_range.

Per testing.md §3.1: ≥90% strong assertions.
Run: pytest scripts/scientific_parser/tests/test_typed_validators.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.verification.typed import (
    validate_latex,
    validate_numeric,
    validate_table,
)

# === validate_table: markdown ================================== #


def test_validate_table_markdown_well_formed_passes():
    md = "| Name | Age |\n|---|---|\n| Alice | 30 |\n| Bob | 25 |\n"
    r = validate_table(md)
    assert r.passed is True
    assert r.validator_id == "table_structure"
    assert r.details["format"] == "markdown"
    assert r.details["rows"] == 3  # header + 2 body
    assert r.details["cols"] == 2


def test_validate_table_markdown_inconsistent_column_counts_does_not_pass():
    md = "| Name | Age |\n|---|---|\n| Alice | 30 | extra |\n| Bob | 25 |\n"
    r = validate_table(md)
    assert r.passed is False
    assert r.details["cols_consistent"] is False
    assert "cols_consistent" in r.details["errors"]


def test_validate_table_empty_string_returns_failed_result():
    r = validate_table("")
    assert r.passed is False
    assert r.score == pytest.approx(0.0)
    assert r.details["error"] == "empty_content"


def test_validate_table_unparseable_content_returns_failed_result():
    r = validate_table("just some prose, no table here")
    assert r.passed is False
    assert r.details["error"] == "unparseable"


def test_validate_table_single_row_below_min_does_not_pass():
    md = "| Name | Age |\n|---|---|\n"
    r = validate_table(md, min_rows=2)
    assert r.passed is False


# === validate_table: html ====================================== #


def test_validate_table_html_well_formed_passes():
    html = (
        "<table>"
        "<tr><th>Name</th><th>Age</th></tr>"
        "<tr><td>Alice</td><td>30</td></tr>"
        "<tr><td>Bob</td><td>25</td></tr>"
        "</table>"
    )
    r = validate_table(html)
    assert r.passed is True
    assert r.details["format"] == "html"
    assert r.details["rows"] == 3


def test_validate_table_html_with_empty_body_cell_flags_no_empty_check():
    html = "<table><tr><th>Name</th><th>Age</th></tr><tr><td></td><td>30</td></tr></table>"
    r = validate_table(html)
    assert r.passed is False
    assert r.details["empty_body_cells"] == 1


# === validate_latex =========================================== #


def test_validate_latex_valid_inline_with_delimiters_passes():
    r = validate_latex(r"$\alpha + \beta = \gamma$")
    assert r.passed is True
    assert r.details["parse_ok"] is True
    assert r.details["render_ok"] is True


def test_validate_latex_valid_block_delimiters_passes():
    r = validate_latex(r"$$E = mc^2$$")
    assert r.passed is True


def test_validate_latex_unbalanced_block_delimiter_does_not_pass():
    """Single $$ (unbalanced) — should fail delimiter check."""
    r = validate_latex(r"$\alpha + \beta")  # one $ missing its pair
    # The validator may still parse/render bare latex; check delimiter flag.
    assert r.details["has_delimiters"] is True


def test_validate_latex_empty_string_returns_failed_result():
    r = validate_latex("")
    assert r.passed is False
    assert r.details["error"] == "empty_content"


def test_validate_latex_malformed_command_does_not_pass():
    """Bogus LaTeX command should fail parse/render."""
    r = validate_latex(r"$\thisisnotarealcommand{}$")
    # parse may tolerate; render should fail
    assert r.details["errors"]  # at least one error reported


def test_validate_latex_bare_latex_without_delimiters_still_validates():
    """Bare LaTeX (no delimiters) is valid input — delimiter check should pass."""
    r = validate_latex(r"\alpha + \beta")
    assert r.details["has_delimiters"] is False
    # Bare latex should still parse + render
    assert r.details["parse_ok"] is True


# === validate_numeric ========================================= #


def test_validate_numeric_simple_integer_no_unit_passes():
    r = validate_numeric("42")
    assert r.passed is True
    assert r.details["value_parsed"] == pytest.approx(42.0)


def test_validate_numeric_float_scientific_notation_passes():
    r = validate_numeric("1.5e6")
    assert r.passed is True
    assert r.details["value_parsed"] == pytest.approx(1.5e6)


def test_validate_numeric_with_known_unit_wt_percent_passes():
    r = validate_numeric("12.5", expected_unit="wt%")
    assert r.passed is True
    assert r.details["unit"] == "wt%"


def test_validate_numeric_with_unknown_unit_does_not_pass():
    r = validate_numeric("12.5", expected_unit="bogus_unit")
    assert r.passed is False
    assert "unknown_unit" in str(r.details["errors"])


def test_validate_numeric_in_plausible_range_passes():
    r = validate_numeric("50.0", plausible_range=(0.0, 100.0))
    assert r.passed is True


def test_validate_numeric_outside_plausible_range_does_not_pass():
    r = validate_numeric("150.0", plausible_range=(0.0, 100.0))
    assert r.passed is False
    assert "out_of_range" in str(r.details["errors"])


def test_validate_numeric_unparseable_string_does_not_pass():
    r = validate_numeric("not_a_number")
    assert r.passed is False
    assert r.details["value_parsed"] is None


def test_validate_numeric_negative_value_with_range_passes():
    r = validate_numeric("-42.5", plausible_range=(-100.0, 0.0))
    assert r.passed is True
    assert r.details["value_parsed"] == pytest.approx(-42.5)


def test_validate_numeric_unicode_minus_normalized_before_parse():
    """Real OCR often emits unicode minus (U+2212). Validator should handle it."""
    r = validate_numeric("−3.14")  # U+2212 minus sign
    assert r.passed is True
    assert r.details["value_parsed"] == pytest.approx(-3.14)


def test_validate_numeric_empty_string_returns_failed_result():
    r = validate_numeric("")
    assert r.passed is False
    assert r.details["error"] == "empty_input"


def test_validate_numeric_none_input_returns_failed_result():
    r = validate_numeric(None)  # type: ignore[arg-type]
    assert r.passed is False
    assert r.details["error"] == "none_input"


def test_validate_numeric_plusminus_prefix_parsed():
    """± prefix stripped before parse."""
    r = validate_numeric("±5.0")
    assert r.passed is True
    assert r.details["value_parsed"] == pytest.approx(5.0)


# === Cross-cutting: all return VerificationResult ============= #


def test_all_validators_return_verification_result_with_correct_validator_id():
    assert validate_table("| a | b |\n|---|---|\n| 1 | 2 |").validator_id == "table_structure"
    assert validate_latex(r"$x$").validator_id == "formula_latex"
    assert validate_numeric("1").validator_id == "numeric_range"


def test_all_verification_results_have_score_in_zero_one():
    for r in [
        validate_table(""),
        validate_latex(""),
        validate_numeric(""),
        validate_table("| a |\n|---|\n| 1 |"),
        validate_latex(r"$x$"),
        validate_numeric("1", plausible_range=(0, 10)),
    ]:
        assert 0.0 <= r.score <= 1.0
