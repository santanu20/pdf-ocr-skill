"""Typed chemistry validator (reviewer #19).

Validates a chemical formula by structural rules achievable without
ambiguous oxidation-state context:
  - parse success (well-formed element+count sequence)
  - element validity (every symbol is a known element)
  - molecular weight computable (every element has a known atomic mass)
  - stoichiometry sanity (every element count >= 1, formula non-trivial)

Charge balance is intentionally NOT checked here: assigning oxidation states
to elements is context-dependent (Fe2+/Fe3+, S2-/S6+, ...). A wrong guess
would produce false failures. That check belongs in a domain-aware validator
with explicit species context (reviewer #19 lists it; left as a documented
limitation until species context is available).

Returns :class:`~scientific_parser.core.ir.VerificationResult` so it plugs
into :class:`~scientific_parser.core.evidence.EvidenceResolver`.
"""

from __future__ import annotations

import hashlib

from ...core.fingerprint_cache import VALIDATOR_CACHE, CacheKey
from ...core.ir import VerificationResult
from ...science.formulas import formula_mass, parse_formula


def _cached(formula_str, fn):
    """Memoize by formula fingerprint (deterministic -> safe, reviewer #26)."""
    key = CacheKey(
        fingerprint=hashlib.sha256((formula_str or "").encode("utf-8", "replace")).hexdigest(),
        engine="chemistry_formula",
        version="1",
    )
    cached = VALIDATOR_CACHE.get(key)
    if cached is not None:
        return cached
    result = fn()
    VALIDATOR_CACHE.put(key, result)
    return result


def validate_formula(formula_str: str | None) -> VerificationResult:
    """Validate a chemical formula's structural correctness.

    Args:
      formula_str: e.g. ``"Fe2O3"``, ``"Mg2SiO4"``, ``"H2O"``.

    Returns:
      :class:`VerificationResult` with ``validator_id="chemistry_formula"``.
      ``passed=True`` requires: parses, all elements valid, mass computable,
      >=1 element present.
    """
    if not formula_str or not isinstance(formula_str, str):
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="chemistry_formula",
            details={"error": "empty or non-string formula"},
        )

    return _cached(formula_str, lambda: _build_formula_result(formula_str))


def _build_formula_result(formula_str: str) -> VerificationResult:
    f = parse_formula(formula_str)
    checks: dict = {"parse_ok": f.valid}
    score_parts = [1.0 if f.valid else 0.0]

    if not f.valid:
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="chemistry_formula",
            details={"checks": checks, "note": f.note},
        )

    # Element count + non-triviality
    n_elements = len(f.elements)
    checks["n_elements"] = n_elements
    score_parts.append(1.0 if n_elements >= 1 else 0.0)

    # Molecular weight computable (every element has a known atomic mass)
    mass = formula_mass(f)
    checks["mass_computable"] = mass is not None
    if mass is not None:
        checks["mass"] = round(mass, 3)
    score_parts.append(1.0 if mass is not None else 0.0)

    # Stoichiometry sanity: no zero/negative counts (parse_formula guarantees
    # positive counts, but guard against future regressions).
    bad_counts = [e for e, c in f.elements.items() if c < 1]
    checks["stoichiometry_ok"] = not bad_counts
    score_parts.append(1.0 if not bad_counts else 0.0)

    score = sum(score_parts) / len(score_parts)
    return VerificationResult(
        passed=score >= 0.75 and f.valid and mass is not None,
        score=round(score, 3),
        validator_id="chemistry_formula",
        details={"checks": checks, "elements": dict(f.elements)},
    )


__all__ = ["validate_formula"]
