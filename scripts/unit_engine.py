"""Universal unit engine for scientific document parsing.

Wraps `pint` (mature Python units library) with:
  - Singleton UnitRegistry (loaded once per process)
  - Domain-aware unit parsing (geochemistry/physics/etc. units pint doesn't
    know by default: wt%, ppm, ppb, Ma, Ga, etc.)
  - Range validation hooks for NumericGuard

P0.3 of the architecture redesign (2026-08-12). Per proposal item #29:
"Build a universal unit engine — perhaps the best general-science component."

Design:
  - Lazy-initialized singleton (pint registry is ~50ms cold start)
  - Domain units defined inline (no external files)
  - All public API returns pint.Quantity or None (never raises)
"""

from __future__ import annotations

import re

_REGISTRY = None
_UNITS_READY = False

# Single source of truth for scientific unit aliases pint cannot parse
# directly. Used by BOTH parse_quantity() and is_known_unit() so the two
# paths can never drift (the 2026-08-13 wt% bug was exactly this drift).
# Keys are matched against the lowercased, normalized unit string.
_UNIT_ALIASES: dict[str, str] = {
    # Concentration
    "wt%": "wt_pct",
    "wt.%": "wt_pct",
    "weight%": "wt_pct",
    "weight %": "wt_pct",
    "mol%": "mol_pct",
    "vol%": "vol_pct",
    "vol.%": "vol_pct",
    "volume%": "vol_pct",
    "volume %": "vol_pct",
    # Geochronology age aliases (Myr / m.y. -> Ma; Gyr -> Ga)
    "myr": "Ma",
    "m.y.": "Ma",
    "m.y": "Ma",
    "ma.": "Ma",
    "gyr": "Ga",
    "byr": "Ga",
    "b.y.": "Ga",
}


def _build_registry():
    """Build pint registry with scientific domain units added."""
    global _REGISTRY, _UNITS_READY
    if _REGISTRY is not None:
        return _REGISTRY
    try:
        import pint
    except ImportError as e:
        _UNITS_READY = False
        # Loud ONCE: unit validation silently off otherwise (audit M4) —
        # callers treat None registry as "skip checks" and never tell.
        import logging
        import warnings

        logging.getLogger(__name__).warning(
            "pint not installed — unit validation DISABLED: %s", e
        )
        warnings.warn("pint missing: scientific unit validation disabled", stacklevel=2)
        return None
    _REGISTRY = pint.UnitRegistry()
    _UNITS_READY = True

    # Define scientific domain units pint doesn't have by default.
    # These are common in geochemistry/geology/materials/chemistry papers.
    # Use safe `define` calls (don't fail if already exists).
    defs = [
        # Geochemistry concentration
        "ppm = 1e-6 = ppm",  # parts per million (dimensionless fraction)
        "ppb = 1e-9 = ppb",
        "ppt = 1e-12 = ppt",
        "wt_pct = 0.01 = wt%",  # weight percent as fraction
        "mol_pct = 0.01 = mol%",
        "vol_pct = 0.01 = vol%",  # volume percent as fraction
        # Geochronology age units (treat as time-like for range checks)
        "Ma = 1e6 * year = Ma",  # mega-annum
        "Ga = 1e9 * year = Ga",  # giga-annum
        "ka = 1e3 * year = ka",  # kilo-annum
    ]
    for d in defs:
        try:
            _REGISTRY.define(d)
        except (pint.RedefinitionError, pint.DefinitionSyntaxError):
            pass  # already defined
    return _REGISTRY


def get_registry():
    """Return pint registry (lazy-init). None if pint not installed."""
    return _build_registry()


def parse_quantity(value: float, unit_str: str):
    """Return pint.Quantity(value, unit) or None on parse failure.

    Never raises. Used by NumericGuard to validate value ranges.
    """
    reg = _build_registry()
    if reg is None:
        return None
    # Normalize common scientific abbreviations pint doesn't recognize
    u = unit_str.strip()
    u = _UNIT_ALIASES.get(u.lower(), u)
    try:
        return reg.Quantity(value, u)
    except Exception:
        # Try parsing as compound (e.g., "km/s")
        try:
            expr = reg.parse_expression(u)
            return value * expr
        except Exception:
            return None


def check_plausible(value: float, unit_str: str, lo: float, hi: float) -> bool:
    """Return True if `value unit_str` is within [lo, hi] of same unit.

    Pure numeric comparison — both bounds assumed same unit as value.
    """
    try:
        return lo <= value <= hi
    except (TypeError, ValueError):
        return True  # unknown → not flagged


def normalize_unit_text(unit_str: str) -> str:
    """Normalize Unicode variant unit spellings to canonical form.

    Examples:
      "µm" → "μm" (micro sign U+00B5 → Greek mu U+03BC)
      "° C" → "°C" (space removed)
      "wt. %" → "wt%"
    """
    if not unit_str:
        return unit_str
    u = unit_str.strip()
    # Micro sign → Greek mu (pint prefers Greek mu)
    u = u.replace("\u00b5", "\u03bc")  # µ → μ
    # Collapse spaces around degree / percent
    u = re.sub(r"\s*°\s*", "°", u)
    u = re.sub(r"\s*%\s*", "%", u)
    # "wt. %" → "wt%"
    u = re.sub(r"\.\s*", "", u) if "wt" in u.lower() else u
    return u


def is_known_unit(unit_str: str) -> bool:
    """True if pint understands this unit (after normalization)."""
    if not unit_str:
        return False
    u = normalize_unit_text(unit_str)
    u = _UNIT_ALIASES.get(u.lower(), u)
    try:
        get_registry().parse_expression(u)
        return True
    except Exception:
        return False


__all__ = [
    "get_registry",
    "parse_quantity",
    "check_plausible",
    "normalize_unit_text",
    "is_known_unit",
]
