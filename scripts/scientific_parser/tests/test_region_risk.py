"""Unit tests for risk-aware selective redundancy wiring (reviewer #6/#24).

Tests the pure decision layer (_region_risk) that drives escalation inside
route_page_by_region: uncertainty (from resolver score + disagreement) x
semantic impact. The execute_action I/O path is exercised end-to-end via the
bench harness on real PDFs.

Run: pytest scripts/scientific_parser/tests/test_region_risk.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.candidates import Candidate, CandidateSet
from scientific_parser.core.ir import (
    E0_NATIVE_PDF_TEXT,
    E4_SPECIALIST_OCR,
    Evidence,
    VerificationResult,
)
from scientific_parser.perception.providers.region_dispatcher import _region_risk


def _cand(level, source, conf, verification=None, text="x"):
    return Candidate(
        source_engine=source,
        evidence=Evidence(
            level=level, source=source, value=text, confidence=conf, verification=verification
        ),
        text=text,
    )


# === high-impact + uncertain -> escalate ========================= #


def test_uncertain_equation_escalates():
    # equation with failed verification -> low resolver score -> high uncertainty
    cs = CandidateSet(region_id="p1_r0", region_type="equation")
    cs.add(
        _cand(
            E4_SPECIALIST_OCR,
            "formula-provider",
            0.4,
            VerificationResult(False, 0.1, "formula_latex"),
        )
    )
    ra = _region_risk(cs)
    assert ra is not None
    assert ra.escalate is True  # uncertain equation = high risk


# === high-confidence + low-impact -> no escalate ================= #


def test_confident_text_does_not_escalate():
    cs = CandidateSet(region_id="p1_r1", region_type="text")
    cs.add(_cand(E0_NATIVE_PDF_TEXT, "pdftotext", 1.0))  # perfect native
    ra = _region_risk(cs)
    assert ra is not None
    assert ra.escalate is False  # confident text = low risk


# === high-impact region tolerated when verified & confident ====== #


def test_verified_equation_not_escalated():
    cs = CandidateSet(region_id="p1_r2", region_type="equation")
    cs.add(
        _cand(
            E4_SPECIALIST_OCR,
            "formula-provider",
            0.9,
            VerificationResult(True, 0.95, "formula_latex"),
        )
    )
    ra = _region_risk(cs)
    # resolver score high -> low uncertainty -> below escalate threshold
    assert ra.escalate is False


# === disagreement boosts uncertainty ============================= #


def test_disagreement_boosts_risk():
    # two candidates disagree; even moderate confidence -> higher risk
    cs = CandidateSet(region_id="p1_r3", region_type="numeric")
    cs.add(_cand(E0_NATIVE_PDF_TEXT, "pdftotext", 0.8, text="4.25"))
    cs.add(_cand(E4_SPECIALIST_OCR, "glm-ocr", 0.8, text="4.80"))
    ra = _region_risk(cs)
    # numeric impact = 5; disagreement bumps uncertainty; should escalate
    assert ra.escalate is True


# === empty candidate set -> None ================================= #


def test_empty_set_returns_none():
    cs = CandidateSet(region_id="p1_r4", region_type="text")
    assert _region_risk(cs) is None


# === impact table drives the difference ========================== #


def test_same_uncertainty_higher_impact_escalates_more():
    # text vs geological_age at the same low confidence
    cs_text = CandidateSet(region_id="p1_r5", region_type="text")
    cs_text.add(_cand(E4_SPECIALIST_OCR, "glm", 0.3))
    cs_geo = CandidateSet(region_id="p1_r6", region_type="geological_age")
    cs_geo.add(_cand(E4_SPECIALIST_OCR, "glm", 0.3))
    ra_text = _region_risk(cs_text)
    ra_geo = _region_risk(cs_geo)
    assert ra_geo.risk > ra_text.risk  # same uncertainty, higher impact
    assert ra_geo.escalate is True
