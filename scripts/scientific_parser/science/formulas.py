"""Formula subsystem — chemistry/materials formula parsing + validation.

Per proposal item #28: "General-purpose equation engine."
This module handles chemical formulas (H2O, SiO2, Fe2O3) — separate from
equations.py (which handles math LaTeX).
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional


# Element symbols (Periodic Table 1-118, common names)
_ELEMENTS = {
    "H",
    "He",
    "Li",
    "Be",
    "B",
    "C",
    "N",
    "O",
    "F",
    "Ne",
    "Na",
    "Mg",
    "Al",
    "Si",
    "P",
    "S",
    "Cl",
    "Ar",
    "K",
    "Ca",
    "Sc",
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Fe",
    "Co",
    "Ni",
    "Cu",
    "Zn",
    "Ga",
    "Ge",
    "As",
    "Se",
    "Br",
    "Kr",
    "Rb",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Mo",
    "Tc",
    "Ru",
    "Rh",
    "Pd",
    "Ag",
    "Cd",
    "In",
    "Sn",
    "Sb",
    "Te",
    "I",
    "Xe",
    "Cs",
    "Ba",
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
    "Hf",
    "Ta",
    "W",
    "Re",
    "Os",
    "Ir",
    "Pt",
    "Au",
    "Hg",
    "Tl",
    "Pb",
    "Bi",
    "Po",
    "At",
    "Rn",
    "Fr",
    "Ra",
    "Ac",
    "Th",
    "Pa",
    "U",
    "Np",
    "Pu",
    "Am",
    "Cm",
    "Bk",
    "Cf",
    "Es",
    "Fm",
    "Md",
    "No",
    "Lr",
    "Rf",
    "Db",
    "Sg",
    "Bh",
    "Hs",
    "Mt",
    "Ds",
    "Rg",
    "Cn",
    "Nh",
    "Fl",
    "Mc",
    "Lv",
    "Ts",
    "Og",
}

_FORMULA_RE = re.compile(r"\b((?:[A-Z][a-z]?\d*)+)\b")


@dataclass
class ChemicalFormula:
    """Parsed chemical formula."""

    raw: str
    elements: dict  # {Element: count}
    valid: bool
    note: str = ""


def parse_formula(text: str) -> ChemicalFormula:
    """Parse a chemical formula like 'Fe2O3' into elements dict."""
    raw = text.strip()
    elements = {}
    valid = True
    note = ""

    # Strip charges (+/- suffix)
    clean = re.sub(r"[\+\-]\d*$", "", raw)

    # Parse element+count sequences
    pos = 0
    while pos < len(clean):
        # Element symbol: capital + optional lowercase
        m = re.match(r"([A-Z][a-z]?)", clean[pos:])
        if not m:
            valid = False
            note = f"unparseable at position {pos}: '{clean[pos:]}'"
            break
        element = m.group(1)
        pos += len(element)
        if element not in _ELEMENTS:
            valid = False
            note = f"unknown element: {element}"
            break
        # Optional count
        m2 = re.match(r"\d+", clean[pos:])
        count = int(m2.group(0)) if m2 else 1
        if m2:
            pos += len(m2.group(0))
        elements[element] = elements.get(element, 0) + count

    return ChemicalFormula(raw=raw, elements=elements, valid=valid, note=note)


def is_mineral_formula(text: str) -> bool:
    """True if text looks like a mineral formula (e.g. 'Fe2O3', 'Mg2SiO4')."""
    if not text or len(text) < 2:
        return False
    f = parse_formula(text)
    return f.valid and len(f.elements) >= 2


def formula_mass(formula: ChemicalFormula) -> Optional[float]:
    """Compute molar mass. None if any element unknown to periodictable."""
    try:
        import periodictable

        total = 0.0
        for elem, count in formula.elements.items():
            el = getattr(periodictable, elem, None)
            if el is None:
                return None
            total += el.mass * count
        return total
    except ImportError:
        return None


def find_formulas_in_text(text: str) -> list:
    """Find all chemical formulas in text. Returns list of ChemicalFormula."""
    out = []
    for m in _FORMULA_RE.finditer(text):
        candidate = m.group(1)
        # Filter: must have at least 2 distinct elements OR have subscript digits
        if not any(c.isdigit() for c in candidate) and len(candidate) <= 2:
            continue
        f = parse_formula(candidate)
        if f.valid and len(f.elements) >= 1:
            out.append(f)
    return out


__all__ = [
    "ChemicalFormula",
    "parse_formula",
    "is_mineral_formula",
    "formula_mass",
    "find_formulas_in_text",
]
