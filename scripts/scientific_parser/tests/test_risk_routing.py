"""Unit tests for risk-aware routing (reviewer #24): uncertainty x impact.

Run: pytest scripts/scientific_parser/tests/test_risk_routing.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.routing.risk import (
    DEFAULT_IMPACT,
    risk,
    semantic_impact,
    should_escalate,
)

# === semantic_impact table ======================================= #


def test_impact_text_is_lowest():
    assert semantic_impact("text") == 1
    assert semantic_impact("paragraph") == 1


def test_impact_geological_age_is_highest():
    assert semantic_impact("geological_age") == 5
    assert semantic_impact("u_pb_age") == 5


def test_impact_unknown_defaults_moderate():
    assert semantic_impact("something_new") == DEFAULT_IMPACT
    assert semantic_impact("") == DEFAULT_IMPACT
    assert semantic_impact(None) == DEFAULT_IMPACT


def test_impact_case_insensitive():
    assert semantic_impact("TABLE") == semantic_impact("table")


# === risk() ====================================================== #


def test_risk_low_confidence_text_is_low_risk():
    # The core reviewer point: uncertain paragraph = low risk
    ra = risk(0.9, "text")  # very uncertain, but low impact
    assert ra.risk == pytest.approx(0.9 * 1)
    assert ra.escalate is False


def test_risk_moderate_confidence_geo_age_is_high_risk():
    # uncertain-ish geological age -> high risk (escalate)
    ra = risk(0.4, "geological_age")  # uncertainty 0.4 x impact 5 = 2.0
    assert ra.risk == pytest.approx(2.0)
    assert ra.escalate is True


def test_risk_table_escalates_above_threshold():
    ra = risk(0.5, "table")  # 0.5 x 3 = 1.5
    assert ra.risk == pytest.approx(1.5)
    assert ra.escalate is True  # == ESCALATE_THRESHOLD


def test_risk_high_confidence_no_escalation():
    ra = risk(0.05, "geological_age")  # 0.05 x 5 = 0.25
    assert ra.escalate is False
    assert ra.review is False


def test_risk_clamps_uncertainty_to_unit_interval():
    ra_hi = risk(5.0, "text")  # clamped to 1.0
    ra_lo = risk(-1.0, "text")  # clamped to 0.0
    assert ra_hi.uncertainty == 1.0
    assert ra_lo.uncertainty == 0.0
    assert ra_lo.risk == 0.0


def test_risk_review_flag_independent_of_escalate():
    # uncertainty 0.8 x text(1) = 0.8 -> review True, escalate False
    ra = risk(0.8, "text")
    assert ra.review is True
    assert ra.escalate is False


# === should_escalate convenience ================================= #


def test_should_escalate_matches_risk_object():
    for u in (0.0, 0.3, 0.5, 0.9):
        for rt in ("text", "table", "geological_age", "equation"):
            assert should_escalate(u, rt) == risk(u, rt).escalate


def test_selective_redundancy_cheap_on_easy_pages():
    # Reviewer #6 thesis: second model only where risk justifies it.
    # A whole document of confident paragraphs needs zero second-model runs.
    pages = [("paragraph", 0.05)] * 100
    n_escalations = sum(1 for rt, u in pages if should_escalate(u, rt))
    assert n_escalations == 0


def test_selective_redundancy_on_critical_values():
    # Low-confidence measurements all escalate.
    pages = [("geological_age", 0.5), ("u_pb_age", 0.6), ("measurement", 0.4)]
    assert all(should_escalate(u, rt) for rt, u in pages)
