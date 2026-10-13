"""Unit tests for combine_page_confidence (confidence-override fix).

Regression guard for the 2026-08-13 bug where
``_conf = max(_conf, reflexive_meta["confidence"])`` let a model's
self-reported confidence mask validator-detected defects.

Run: pytest scripts/scientific_parser/tests/test_combine_confidence.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.confidence_helpers import combine_page_confidence

# === No warnings: model may raise (corroborating signal) ========= #


def test_no_warnings_model_higher_raises():
    # validator 0.7, model 0.95, no warnings -> max = 0.95
    assert combine_page_confidence(0.7, 0.95, has_warnings=False) == 0.95


def test_no_warnings_model_lower_keeps_validator():
    assert combine_page_confidence(0.9, 0.5, has_warnings=False) == 0.9


# === Warnings present: model must NOT mask validator defects ====== #


def test_warnings_model_higher_capped_at_validator():
    # The core bug: validator says bad (0.4, warnings), model says good (0.95)
    # -> combined must NOT exceed validator score.
    assert combine_page_confidence(0.4, 0.95, has_warnings=True) <= 0.4


def test_warnings_model_higher_equals_validator_exactly():
    # blend(0.4,0.95)=0.675, min(0.4,0.675)=0.4
    assert combine_page_confidence(0.4, 0.95, has_warnings=True) == 0.4


def test_warnings_model_doubt_lowers_score():
    # validator 0.85 (minor warning), model 0.5 (unsure) -> blend=0.675,
    # min(0.85, 0.675) = 0.675. Model doubt is informative.
    assert combine_page_confidence(0.85, 0.5, has_warnings=True) == 0.68


# === Edge cases ================================================== #


def test_reflexive_none_returns_validator():
    assert combine_page_confidence(0.7, None, has_warnings=False) == 0.7
    assert combine_page_confidence(0.4, None, has_warnings=True) == 0.4


def test_clamped_to_min_0p1():
    # validator below floor -> clamped to 0.1
    assert combine_page_confidence(0.0, 0.9, has_warnings=False) == 0.9
    assert combine_page_confidence(0.0, None, has_warnings=True) == 0.1


def test_clamped_to_max_1p0():
    assert combine_page_confidence(1.0, 1.0, has_warnings=False) == 1.0


def test_result_in_valid_range():
    for vc in (0.1, 0.3, 0.5, 0.7, 1.0):
        for rc in (None, 0.0, 0.5, 1.0):
            for w in (True, False):
                r = combine_page_confidence(vc, rc, has_warnings=w)
                assert 0.1 <= r <= 1.0
