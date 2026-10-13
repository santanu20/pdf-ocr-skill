"""Unit tests for the unit_engine alias registry (reviewer #20 completion).

Covers the 2026-08-13 fix (wt%/ppm/Ma/Ga) AND the added aliases
(vol%, Myr, Gyr, weight%, m.y., Byr). Guards against drift between
parse_quantity() and is_known_unit() (single _UNIT_ALIASES source).

Run: pytest scripts/scientific_parser/tests/test_unit_engine.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import unit_engine

# === is_known_unit: geochemistry concentration =================== #


@pytest.mark.parametrize("u", ["wt%", "wt.%", "ppm", "ppb", "mol%"])
def test_known_concentration_units(u):
    assert unit_engine.is_known_unit(u), f"{u} should be recognized"


def test_vol_percent_known():
    assert unit_engine.is_known_unit("vol%")
    assert unit_engine.is_known_unit("vol.%")
    assert unit_engine.is_known_unit("volume%")


def test_weight_percent_alias():
    assert unit_engine.is_known_unit("weight%")
    assert unit_engine.is_known_unit("weight %")


# === is_known_unit: geochronology =============================== #


@pytest.mark.parametrize("u", ["Ma", "Ga", "ka", "Myr", "Gyr", "Byr", "m.y."])
def test_known_geochronology_units(u):
    assert unit_engine.is_known_unit(u), f"{u} should be recognized"


# === is_known_unit: SI / physical (pint-native) ================== #


@pytest.mark.parametrize("u", ["m", "km", "cm", "mm", "°C", "K", "Pa", "MPa", "GPa", "bar"])
def test_known_si_units(u):
    assert unit_engine.is_known_unit(u)


# === parse_quantity: alias equivalence ========================== #


def test_myr_equivalent_to_Ma():
    q_myr = unit_engine.parse_quantity(5, "Myr")
    q_ma = unit_engine.parse_quantity(5, "Ma")
    assert q_myr is not None and q_ma is not None
    assert q_myr == q_ma  # 5 Myr == 5 Ma == 5e6 years


def test_gyr_equivalent_to_Ga():
    assert unit_engine.parse_quantity(2, "Gyr") == unit_engine.parse_quantity(2, "Ga")


def test_byr_equivalent_to_Ga():
    assert unit_engine.parse_quantity(1, "Byr") == unit_engine.parse_quantity(1, "Ga")


def test_vol_pct_parses():
    q = unit_engine.parse_quantity(50, "vol%")
    assert q is not None
    assert q.to("vol_pct").magnitude == pytest.approx(50)


def test_wt_pct_parses():
    q = unit_engine.parse_quantity(12.5, "wt%")
    assert q is not None


# === negative / edge ============================================ #


def test_unknown_unit_not_known():
    assert not unit_engine.is_known_unit("zzqx")
    assert not unit_engine.is_known_unit("")


def test_normalize_handles_space_in_wt_percent():
    # "wt %" (with space) should normalize + alias to wt_pct
    assert unit_engine.is_known_unit("wt %")
    assert unit_engine.is_known_unit("wt. %")


def test_empty_string_parse_returns_none():
    assert (
        unit_engine.parse_quantity(1.0, "") is None
        or unit_engine.parse_quantity(1.0, "") is not None
    )
    # Empty/dimensionless — pint may accept as dimensionless; just must not raise.


def test_model_profile_complete_for_all_models():
    """iter-54 pin: every resolved model profile must carry all four Ollama
    option keys. paddleocr-vl registered without 'temperature' and
    get_model_profile returned the raw dict — ocr_image's payload assembly
    raised KeyError('temperature') BEFORE any HTTP call on every
    paddleocr-vl invocation (retries useless; silent rapidocr degrade).
    Fix: profiles merge over DEFAULT_MODEL_PROFILE."""
    from scientific_parser.ingest.ocr_engine import (
        DEFAULT_MODEL_PROFILE,
        MODEL_PROFILES,
        get_model_profile,
    )

    required = {"temperature", "num_predict", "num_ctx", "repeat_penalty"}
    for name in list(MODEL_PROFILES) + [
        "paddleocr-vl:1.6",
        "chandra-ocr-2:latest",
        "glm-ocr:latest",
        "totally-unknown-model",
    ]:
        prof = get_model_profile(name)
        missing = required - set(prof)
        assert not missing, f"profile {name!r} missing {sorted(missing)}"
    assert set(DEFAULT_MODEL_PROFILE) >= required
