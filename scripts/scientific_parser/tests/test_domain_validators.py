"""Unit tests for typed domain validators (reviewer #19): chemistry + geology.

Run: pytest scripts/scientific_parser/tests/test_domain_validators.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.verification.typed import (
    validate_formula,
    validate_oxide_total,
    validate_ree_sequence_typed,
    validate_upb_age,
)

# === chemistry: validate_formula ================================= #


def test_formula_valid_fe2o3():
    r = validate_formula("Fe2O3")
    assert r.passed
    assert r.score >= 0.75
    assert r.details["checks"]["mass_computable"]


def test_formula_valid_h2o_mass():
    r = validate_formula("H2O")
    assert r.passed
    assert r.details["checks"]["mass"] == pytest.approx(18.015, abs=0.01)


def test_formula_unknown_element_fails():
    r = validate_formula("Xx2O3")  # Xx is not an element
    assert not r.passed
    assert r.score == 0.0


def test_formula_unparseable_fails():
    r = validate_formula("2+3")
    assert not r.passed


def test_formula_empty_fails():
    assert not validate_formula("").passed
    assert not validate_formula(None).passed


def test_formula_returns_typed_result_id():
    assert validate_formula("SiO2").validator_id == "chemistry_formula"


# === geology: oxide totals ======================================= #


def test_oxide_total_passes_when_no_oxides():
    # Self-gating: no oxide content -> passed, full score
    r = validate_oxide_total("This is ordinary prose with no geochemistry.")
    assert r.passed
    assert r.score == 1.0


def test_oxide_total_flags_bad_sum():
    # Oxide values that do NOT sum near 100
    text = "SiO2 10.0 Al2O3 5.0 Fe2O3 3.0 MgO 2.0 CaO 1.0 Na2O 0.5 K2O 0.5 TiO2 0.1"
    r = validate_oxide_total(text)
    assert not r.passed
    assert r.score < 1.0


def test_oxide_total_validator_id():
    assert validate_oxide_total("prose").validator_id == "geology_oxide_total"


# === geology: U-Pb age =========================================== #


def test_upb_age_passes_when_no_age():
    r = validate_upb_age("No geochronology here.")
    assert r.passed


def test_upb_age_flags_older_than_earth():
    r = validate_upb_age("Sample age = 5.0 ± 0.1 Ga")
    assert not r.passed
    assert any("age" in w for w in r.details["warnings"])


def test_upb_age_flags_too_old_ma():
    # >4600 Ma exceeds Earth age -> flagged
    r = validate_upb_age("age = 5000 ± 50 Ma")
    assert not r.passed


def test_upb_age_accepts_valid_age():
    r = validate_upb_age(" Concordia age = 450 ± 5 Ma")
    assert r.passed


# === geology: REE sequence ======================================= #


def test_ree_sequence_passes_on_prose():
    r = validate_ree_sequence_typed("Some ordinary paragraph without REE.")
    assert r.passed


def test_ree_sequence_returns_typed_id():
    assert validate_ree_sequence_typed("text").validator_id == "geology_ree_sequence"


# === integration: typed results plug into resolver =============== #


def test_chemistry_result_is_verification_result():
    from scientific_parser.core.ir import VerificationResult

    assert isinstance(validate_formula("H2O"), VerificationResult)


def test_geology_results_are_verification_result():
    from scientific_parser.core.ir import VerificationResult

    for fn in (validate_oxide_total, validate_upb_age, validate_ree_sequence_typed):
        assert isinstance(fn("prose"), VerificationResult)
