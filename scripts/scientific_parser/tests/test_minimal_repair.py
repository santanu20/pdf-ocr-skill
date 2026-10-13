"""Unit tests for minimal-token LaTeX repair (reviewer #21).

Run: pytest scripts/scientific_parser/tests/test_minimal_repair.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.perception.equations.minimal_repair import (
    add_confusion,
    available_confusions,
    minimal_repair,
)

# === never touches a passing formula ============================= #


def test_passing_formula_returned_unchanged():
    def renders_ok(s):
        return s == r"\sqrt{x}"

    assert minimal_repair(r"\sqrt{x}", renders_ok) == r"\sqrt{x}"


def test_empty_string_returned_as_is():
    assert minimal_repair("", lambda s: True) == ""


# === single-token repair ======================================== #


def test_sort_repaired_to_sqrt_when_render_fails():
    # \sort fails render; \sqrt passes -> single-token fix applied
    def renders_ok(s):
        return r"\sqrt" in s and r"\sort" not in s

    out = minimal_repair(r"\sort{x}", renders_ok)
    assert out == r"\sqrt{x}"


def test_only_one_token_changed():
    def renders_ok(s):
        # \frace is an undefined command (fails); only \frac{...} parses
        return r"\frac" in s and r"\frace" not in s

    out = minimal_repair(r"\frace{1}{2}", renders_ok)
    assert out == r"\frac{1}{2}"


# === no wholesale rewrite ======================================== #


def test_no_fix_leaves_original_unchanged():
    # unknown failure -> no single-token fix -> original returned (no rewrite)
    def renders_ok(s):
        return False  # nothing renders

    original = r"\bogusmacro{x + y}"
    assert minimal_repair(original, renders_ok) == original


def test_does_not_attempt_multi_token_rewrite():
    # two bad tokens but fixing only one still fails render -> leave unchanged
    def renders_ok(s):
        return False

    out = minimal_repair(r"\sort \frace", renders_ok)
    assert out == r"\sort \frace"  # unchanged


# === render predicate robustness =================================


def test_render_predicate_exception_treated_as_failure():
    def raises(s):
        raise RuntimeError("katex unavailable")

    # should not propagate; falls through to trying fixes (none work) -> original
    original = r"\sqrt{x}"
    assert minimal_repair(original, raises) == original


def test_first_working_fix_wins():
    calls = []

    def renders_ok(s):
        calls.append(s)
        # passes only once \summ -> \sum applied
        return r"\sum" in s and r"\summ" not in s

    out = minimal_repair(r"\summ_{i}", renders_ok)
    assert out == r"\sum_{i}"


# === extensibility =============================================== #


def test_available_confusions_has_sqrt():
    conf = available_confusions()
    assert conf[r"\sort"] == r"\sqrt"


def test_add_confusion_registers_new_token():
    add_confusion(r"\fooo", r"\foo")
    assert available_confusions().get(r"\fooo") == r"\foo"


def test_add_confusion_rejects_non_latex_token():
    import pytest

    with pytest.raises(ValueError):
        add_confusion("sqrt", r"\sqrt")  # ocr token must be backslash-prefixed
