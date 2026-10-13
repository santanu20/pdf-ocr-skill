"""Unit tests for the candidate lattice (reviewer #38/#18/#5).

Verifies Candidate/CandidateSet capture every extraction attempt with full
provenance and that winner() delegates to the weighted EvidenceResolver.

Run: pytest scripts/scientific_parser/tests/test_candidates.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.candidates import Candidate, CandidateSet
from scientific_parser.core.ir import (
    E0_NATIVE_PDF_TEXT,
    E3_CLASSICAL_OCR,
    E4_SPECIALIST_OCR,
    Evidence,
    VerificationResult,
)


def _ev(level, source, conf=1.0, verification=None):
    return Evidence(
        level=level, source=source, value="x", confidence=conf, verification=verification
    )


def _passed(score=0.9):
    return VerificationResult(passed=True, score=score, validator_id="t")


# === Candidate =================================================== #


def test_candidate_to_dict_has_provenance():
    c = Candidate(
        source_engine="formula-provider",
        evidence=_ev(E4_SPECIALIST_OCR, "formula-provider", 0.88, _passed()),
        text="$$E=mc^2$$",
        crop_bbox=(10, 20, 30, 40),
        render_dpi=300.0,
    )
    d = c.to_dict()
    assert d["source_engine"] == "formula-provider"
    assert d["evidence_level"] == E4_SPECIALIST_OCR
    assert d["confidence"] == pytest.approx(0.88)
    assert d["verification"]["score"] == pytest.approx(0.9)
    assert d["text_len"] == len("$$E=mc^2$$")


def test_candidate_no_verification_serializes_none():
    c = Candidate(
        source_engine="native-pdf-text", evidence=_ev(E0_NATIVE_PDF_TEXT, "pdftotext"), text="hi"
    )
    assert c.to_dict()["verification"] is None


# === CandidateSet.winner ========================================= #


def test_empty_set_winner_none():
    cs = CandidateSet(region_id="p1_r0", region_type="text")
    assert cs.winner() is None
    assert cs.resolved_text() == ""


def test_single_candidate_is_winner():
    cs = CandidateSet(region_id="p1_r0", region_type="text")
    c = Candidate("native-pdf-text", _ev(E0_NATIVE_PDF_TEXT, "pdftotext"), "hello")
    cs.add(c)
    assert cs.winner() is c
    assert cs.resolved_text() == "hello"


def test_winner_uses_weighted_resolver_verified_specialist_beats_unverified_native():
    # Native E0 (reliability 1.0, conf 1.0, no verification = neutral 0.5)
    # vs specialist E4 (reliability 0.85, conf 0.9, verification score 0.95).
    # Specialist score: .4*.85 + .3*.9 + .3*.95 = .34+.27+.285 = .895
    # Native score:     .4*1.0 + .3*1.0 + .3*.5  = .4+.3+.15 = .85
    # -> specialist wins because verification corroborates it.
    cs = CandidateSet(region_id="p1_r1", region_type="equation")
    native = Candidate("native-pdf-text", _ev(E0_NATIVE_PDF_TEXT, "pdftotext", 1.0), "garbled")
    specialist = Candidate(
        "formula-provider",
        _ev(E4_SPECIALIST_OCR, "formula-provider", 0.9, _passed(0.95)),
        "$$E=mc^2$$",
    )
    cs.add(native)
    cs.add(specialist)
    assert cs.winner() is specialist
    assert cs.resolved_text() == "$$E=mc^2$$"


def test_winner_native_beats_unverified_low_conf_specialist():
    # Native E0 conf 1.0 vs specialist E4 conf 0.3 no verification.
    cs = CandidateSet(region_id="p1_r2", region_type="text")
    cs.add(Candidate("native-pdf-text", _ev(E0_NATIVE_PDF_TEXT, "pdftotext", 1.0), "clean text"))
    cs.add(Candidate("rapidocr", _ev(E3_CLASSICAL_OCR, "rapidocr", 0.3), "noisy"))
    # native reliability+conf dominates
    assert cs.winner().source_engine == "native-pdf-text"


# === agreement =================================================== #


def test_agreement_none_for_single_candidate():
    cs = CandidateSet(region_id="p1_r3", region_type="text")
    cs.add(Candidate("native-pdf-text", _ev(E0_NATIVE_PDF_TEXT, "pdftotext"), "x"))
    assert cs.agreement() is None


def test_agreement_full_when_all_match():
    cs = CandidateSet(region_id="p1_r4", region_type="numeric")
    cs.add(Candidate("native", _ev(E0_NATIVE_PDF_TEXT, "pdftotext", 0.9), "4.25"))
    cs.add(Candidate("rapidocr", _ev(E3_CLASSICAL_OCR, "rapidocr", 0.9), "4.25"))
    assert cs.agreement() == 1.0


def test_agreement_partial_on_disagreement():
    cs = CandidateSet(region_id="p1_r5", region_type="numeric")
    cs.add(Candidate("a", _ev(E0_NATIVE_PDF_TEXT, "pdftotext", 0.9), "4.25"))
    cs.add(Candidate("b", _ev(E3_CLASSICAL_OCR, "rapidocr", 0.9), "4.25"))
    cs.add(Candidate("c", _ev(E4_SPECIALIST_OCR, "glm", 0.5), "4.80"))
    # winner is a or b (tie, first wins = "a"); 2/3 match the winner
    assert cs.agreement() == pytest.approx(2 / 3)


# === CandidateSet.to_dict ======================================== #


def test_to_dict_round_trip_keys():
    cs = CandidateSet(region_id="p1_r6", region_type="table", bbox=(1, 2, 3, 4))
    cs.add(
        Candidate(
            "table-provider", _ev(E4_SPECIALIST_OCR, "table-provider", 0.9, _passed()), "<table>"
        )
    )
    d = cs.to_dict()
    assert d["region_id"] == "p1_r6"
    assert d["region_type"] == "table"
    assert d["n_candidates"] == 1
    assert d["winner_source"] == "table-provider"
    assert d["agreement"] is None  # only 1 candidate
    assert list(d["bbox"]) == [1, 2, 3, 4]
