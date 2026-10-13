"""Unit tests for EvidenceResolver and VerificationResult.

Per testing.md §3.1: ≥90% strong assertions.
Per testing.md §5.2: float comparison via pytest.approx, never raw ==.

Run: pytest scripts/scientific_parser/tests/test_evidence_resolver.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.evidence import (
    DEFAULT_RESOLVER,
    EVIDENCE_SOURCE_RELIABILITY,
    EvidenceResolver,
)
from scientific_parser.core.ir import (
    E0_NATIVE_PDF_TEXT,
    E3_CLASSICAL_OCR,
    E4_SPECIALIST_OCR,
    E5_VLM,
    BlockIR,
    Evidence,
    VerificationResult,
)

# === VerificationResult construction ============================== #


def test_verification_result_valid_construction_returns_object():
    v = VerificationResult(passed=True, score=0.95, validator_id="formula_latex")
    assert v.passed is True
    assert v.score == pytest.approx(0.95)
    assert v.validator_id == "formula_latex"
    assert v.details == {}


def test_verification_result_score_above_one_raises_value_error():
    with pytest.raises(ValueError, match="score"):
        VerificationResult(passed=True, score=1.5, validator_id="x")


def test_verification_result_score_below_zero_raises_value_error():
    with pytest.raises(ValueError, match="score"):
        VerificationResult(passed=True, score=-0.1, validator_id="x")


def test_verification_result_empty_validator_id_raises_value_error():
    with pytest.raises(ValueError, match="validator_id"):
        VerificationResult(passed=True, score=0.5, validator_id="")


def test_verification_result_to_dict_roundtrip_preserves_fields():
    v = VerificationResult(
        passed=False,
        score=0.2,
        validator_id="table_structure",
        details={"rows": 3, "cols": 4, "errors": ["missing header"]},
    )
    d = v.to_dict()
    assert d == {
        "passed": False,
        "score": 0.2,
        "validator_id": "table_structure",
        "details": {"rows": 3, "cols": 4, "errors": ["missing header"]},
    }


# === EvidenceResolver construction =============================== #


def test_resolver_default_construction_returns_object_with_default_weights():
    r = EvidenceResolver()
    assert r.source_weight == pytest.approx(0.40)
    assert r.conf_weight == pytest.approx(0.30)
    assert r.verif_weight == pytest.approx(0.30)


def test_resolver_negative_weight_raises_value_error():
    with pytest.raises(ValueError, match="non-negative"):
        EvidenceResolver(source_weight=-0.1)


def test_resolver_all_zero_weights_raises_value_error():
    with pytest.raises(ValueError, match="sum > 0"):
        EvidenceResolver(source_weight=0.0, conf_weight=0.0, verif_weight=0.0)


def test_resolver_custom_reliability_override_replaces_defaults():
    custom = {"E0_native_pdf_text": 0.5}
    r = EvidenceResolver(source_reliability=custom)
    assert r.source_reliability == custom
    # Custom does not mutate module-level dict
    assert EVIDENCE_SOURCE_RELIABILITY["E0_native_pdf_text"] == pytest.approx(1.0)


# === Scoring ====================================================== #


def test_score_evidence_with_verification_blends_three_signals():
    """score = w_s*rel + w_c*conf + w_v*verif, normalized."""
    r = EvidenceResolver()  # weights normalized to 0.4/0.3/0.3 = 1.0
    ev = Evidence(
        level=E0_NATIVE_PDF_TEXT,  # rel=1.0
        source="pdftotext",
        value="x",
        confidence=0.8,
        verification=VerificationResult(passed=True, score=0.9, validator_id="t"),
    )
    expected = 0.4 * 1.0 + 0.3 * 0.8 + 0.3 * 0.9
    assert r.score(ev) == pytest.approx(expected)


def test_score_evidence_without_verification_uses_neutral_score():
    r = EvidenceResolver()
    ev = Evidence(
        level=E3_CLASSICAL_OCR,  # rel=0.7
        source="rapidocr",
        value="x",
        confidence=0.6,
        verification=None,
    )
    expected = 0.4 * 0.7 + 0.3 * 0.6 + 0.3 * 0.5  # neutral=0.5
    assert r.score(ev) == pytest.approx(expected)


def test_score_evidence_clamps_confidence_above_one_to_one():
    r = EvidenceResolver()
    ev = Evidence(
        level=E0_NATIVE_PDF_TEXT,
        source="x",
        value="x",
        confidence=2.5,  # out of range
    )
    # Expected uses clamped conf=1.0
    expected = 0.4 * 1.0 + 0.3 * 1.0 + 0.3 * 0.5
    assert r.score(ev) == pytest.approx(expected)


# === resolve() =================================================== #


def test_resolve_empty_evidence_list_returns_none():
    r = EvidenceResolver()
    assert r.resolve([]) is None


def test_resolve_single_evidence_returns_that_evidence():
    r = EvidenceResolver()
    ev = Evidence(level=E0_NATIVE_PDF_TEXT, source="pdftotext", value="hello")
    assert r.resolve([ev]) is ev


def test_resolve_specialist_with_verification_beats_native_without():
    """The KEY test for the new resolver: a verified specialist (E4, score=0.95)
    must beat an unverified native (E0, no verification) despite E0 having
    higher source_reliability."""
    r = EvidenceResolver()
    native = Evidence(
        level=E0_NATIVE_PDF_TEXT,
        source="pdftotext",
        value="garbled",
        confidence=0.6,
    )
    specialist = Evidence(
        level=E4_SPECIALIST_OCR,
        source="glm-ocr",
        value=r"\alpha",
        confidence=0.85,
        verification=VerificationResult(passed=True, score=0.95, validator_id="formula_latex"),
    )
    winner = r.resolve([native, specialist])
    assert winner is specialist
    assert winner.value == r"\alpha"


def test_resolve_at_equal_score_lower_e_level_wins_as_tiebreak():
    """When weighted scores are equal, lower E-number wins (deterministic)."""
    # Use a resolver where source reliability is uniform so the only
    # differentiator is E-level priority.
    r = EvidenceResolver(
        source_weight=0.0,  # ignore source reliability
        conf_weight=1.0,
        verif_weight=0.0,
    )
    e0 = Evidence(level=E0_NATIVE_PDF_TEXT, source="a", value="e0", confidence=0.5)
    e5 = Evidence(level=E5_VLM, source="b", value="e5", confidence=0.5)
    winner = r.resolve([e5, e0])  # input order randomized
    assert winner is e0


def test_resolve_confidence_dominates_when_verification_absent_everywhere():
    """All-legacy path: highest-confidence Evidence wins."""
    r = EvidenceResolver()
    low = Evidence(level=E0_NATIVE_PDF_TEXT, source="a", value="low", confidence=0.3)
    high = Evidence(level=E5_VLM, source="b", value="high", confidence=0.99)
    winner = r.resolve([low, high])
    # Score low:  0.4*1.0 + 0.3*0.3 + 0.3*0.5 = 0.4+0.09+0.15 = 0.64
    # Score high: 0.4*0.6 + 0.3*0.99 + 0.3*0.5 = 0.24+0.297+0.15 = 0.687
    assert winner is high


# === DEFAULT_RESOLVER =========================================== #


def test_default_resolver_is_evidence_resolver_instance():
    assert isinstance(DEFAULT_RESOLVER, EvidenceResolver)


def test_default_resolver_preserves_priority_behavior_for_legacy_evidence():
    """DEFAULT_RESOLVER should preserve the old fixed-priority behavior for
    legacy Evidence (no verification): E0 wins at equal confidence."""
    e0 = Evidence(level=E0_NATIVE_PDF_TEXT, source="a", value="e0", confidence=0.5)
    e5 = Evidence(level=E5_VLM, source="b", value="e5", confidence=0.5)
    # E0 reliability=1.0, E5 reliability=0.6 — E0 wins on source term alone.
    assert DEFAULT_RESOLVER.resolve([e5, e0]) is e0


# === BlockIR.resolve integration =============================== #


def test_blockir_resolve_with_no_evidence_returns_none():
    b = BlockIR(id="x", type="text", bbox=(0, 0, 1, 1), page_num=1)
    assert b.resolve() is None


def test_blockir_resolve_passes_custom_resolver_through():
    """Caller-supplied resolver overrides DEFAULT_RESOLVER."""
    b = BlockIR(
        id="x",
        type="text",
        bbox=(0, 0, 1, 1),
        page_num=1,
        evidence=[
            Evidence(level=E0_NATIVE_PDF_TEXT, source="a", value="low", confidence=0.3),
            Evidence(level=E5_VLM, source="b", value="high", confidence=0.99),
        ],
    )
    # Verif-only resolver (ignore source reliability): high-conf wins
    custom = EvidenceResolver(source_weight=0.0, conf_weight=1.0, verif_weight=0.0)
    winner = b.resolve(custom)
    assert winner.value == "high"


def test_blockir_to_dict_serializes_resolved_value_and_confidence():
    b = BlockIR(
        id="x",
        type="equation",
        bbox=(0, 0, 1, 1),
        page_num=1,
        evidence=[
            Evidence(
                level=E4_SPECIALIST_OCR,
                source="glm-ocr",
                value=r"\alpha",
                confidence=0.9,
                verification=VerificationResult(
                    passed=True, score=0.95, validator_id="formula_latex"
                ),
            ),
        ],
    )
    d = b.to_dict()
    assert d["resolved_value"] == r"\alpha"
    assert d["resolved_confidence"] == pytest.approx(0.9)


def test_blockir_to_dict_with_no_evidence_emits_none_value():
    b = BlockIR(id="x", type="text", bbox=(0, 0, 1, 1), page_num=1)
    d = b.to_dict()
    assert d["resolved_value"] is None
    assert d["resolved_confidence"] == pytest.approx(0.0)


def test_evidence_to_dict_omits_verification_key_when_none():
    """Stable JSON shape: verification key absent when not set."""
    ev = Evidence(level=E0_NATIVE_PDF_TEXT, source="pdftotext", value="x")
    d = ev.to_dict()
    assert "verification" not in d


def test_evidence_to_dict_includes_verification_dict_when_present():
    ev = Evidence(
        level=E0_NATIVE_PDF_TEXT,
        source="pdftotext",
        value="x",
        verification=VerificationResult(passed=True, score=0.9, validator_id="t"),
    )
    d = ev.to_dict()
    assert d["verification"]["passed"] is True
    assert d["verification"]["validator_id"] == "t"
