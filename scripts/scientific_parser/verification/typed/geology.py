"""Typed geology validators (reviewer #19).

Wrap the existing self-gating geochemistry checks (oxide totals, REE
sequence, U-Pb age) into :class:`~scientific_parser.core.ir.VerificationResult`
so they plug into :class:`~scientific_parser.core.evidence.EvidenceResolver`
via ``Evidence.verification`` — the typed-verification layer the reviewer
asked for. The underlying detection logic lives in
:mod:`scientific_parser.science.geo_validators` and is reused unchanged.

Scoring policy: each warning from the underlying detector lowers the score
by 0.15 (clamped to [0, 1]); ``passed`` requires no warnings. The detectors
are self-gating (they return [] when the content signature is absent), so
these validators are safe to run on any page.
"""

from __future__ import annotations

import hashlib

from ...core.fingerprint_cache import VALIDATOR_CACHE, CacheKey
from ...core.ir import VerificationResult


def _geo():
    """Lazy import of geo_validators (it does ``from shared import *`` at
    module top, which must NOT run during package init — shared defines
    ``_is_numeric_cell`` late and an early star-import creates a cycle)."""
    from ...science import geo_validators

    return geo_validators


_WARN_PENALTY = 0.15


def _fingerprint(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", "replace")).hexdigest()


def _cached_validate(validator_id: str, text: str, fn):
    """Memoize a typed geo-validator by (validator, text fingerprint).

    Validator verdicts are deterministic functions of the input text, so the
    FingerprintCache (reviewer #26) gives a free speedup on repeated regions
    or pages with identical content. A miss is always safe (recompute).
    """
    key = CacheKey(fingerprint=_fingerprint(text), engine=validator_id, version="1")
    cached = VALIDATOR_CACHE.get(key)
    if cached is not None:
        return cached
    result = fn()
    VALIDATOR_CACHE.put(key, result)
    return result


def _score(warnings: list) -> float:
    return round(max(0.0, 1.0 - _WARN_PENALTY * len(warnings)), 3)


def validate_oxide_total(text: str) -> VerificationResult:
    """Validate whole-rock oxide totals sum to ~100 wt% (±5).

    Returns ``VerificationResult`` with ``validator_id="geology_oxide_total"``.
    Self-gating: ``score=1.0, passed=True`` when no oxide table is present.
    """
    return _cached_validate(
        "geology_oxide_total",
        text,
        lambda: _build_oxide_result(text),
    )


def _build_oxide_result(text: str) -> VerificationResult:
    all_warns = _geo().validate_scientific_data(text)
    oxide_warns = [w for w in all_warns if "oxide" in w]
    return VerificationResult(
        passed=len(oxide_warns) == 0,
        score=_score(oxide_warns),
        validator_id="geology_oxide_total",
        details={"warnings": oxide_warns, "n": len(oxide_warns)},
    )


def validate_upb_age(text: str) -> VerificationResult:
    """Validate U-Pb / geochronology ages are physically plausible.

    Flags ages older than Earth (>4.6 Ga / >4600 Ma) or negative ages.
    Returns ``VerificationResult`` with ``validator_id="geology_upb_age"``.
    """
    return _cached_validate(
        "geology_upb_age",
        text,
        lambda: _build_upb_result(text),
    )


def _build_upb_result(text: str) -> VerificationResult:
    all_warns = _geo().validate_scientific_data(text)
    age_warns = [w for w in all_warns if "age" in w]
    return VerificationResult(
        passed=len(age_warns) == 0,
        score=_score(age_warns),
        validator_id="geology_upb_age",
        details={"warnings": age_warns, "n": len(age_warns)},
    )


def validate_ree_sequence_typed(text: str) -> VerificationResult:
    """Validate REE listings are in correct monotonic order with no OCR swaps.

    Fires only on clusters containing a monotonic REE run (the signature of
    a real REE listing). Returns ``VerificationResult`` with
    ``validator_id="geology_ree_sequence"``.
    """
    return _cached_validate(
        "geology_ree_sequence",
        text,
        lambda: _build_ree_result(text),
    )


def _build_ree_result(text: str) -> VerificationResult:
    ree_warns = _geo().validate_ree_sequence(text)
    return VerificationResult(
        passed=len(ree_warns) == 0,
        score=_score(ree_warns),
        validator_id="geology_ree_sequence",
        details={"warnings": ree_warns, "n": len(ree_warns)},
    )


__all__ = [
    "validate_oxide_total",
    "validate_upb_age",
    "validate_ree_sequence_typed",
]
