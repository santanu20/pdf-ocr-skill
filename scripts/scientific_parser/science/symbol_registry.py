"""Scientific Symbol Registry — central catalog of all scientific symbols.

Per proposal item #30 (2026-08-12):
  "Don't hard-code Unicode normalization rules everywhere. Create a scientific
   symbol registry: Greek / Math / Units / Chemical / Physics / Biology /
   Statistics / Engineering.

   Then μ, µ, u can be treated as possible variants depending on context.
   Likewise ×, x, * must not automatically be normalized without context."

Central registry; lookup by symbol → metadata (canonical form, category,
LaTeX command, Unicode codepoint, context-dependent variants).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SymbolEntry:
    """One symbol entry in the registry."""

    canonical: str  # preferred form (Unicode)
    category: str  # greek|math|unit|chemical|physics|biology|stats|engineering
    latex: Optional[str] = None  # LaTeX command, e.g. "\\alpha"
    codepoint: Optional[str] = None  # U+03BC
    name: str = ""  # human name "Greek small letter mu"
    variants: list = field(default_factory=list)  # other forms [µ, u, μ]
    contexts: list = field(
        default_factory=list
    )  # contexts where used ["physics", "biology"]


# === Greek letters ========================================================= #
# Each: canonical Unicode, LaTeX, common OCR variants
_GREEK_ENTRIES = [
    SymbolEntry(
        "α",
        "greek",
        "\\alpha",
        "U+03B1",
        "alpha",
        variants=["alpha", "a"],
        contexts=["math", "physics", "stats"],
    ),
    SymbolEntry(
        "β",
        "greek",
        "\\beta",
        "U+03B2",
        "beta",
        variants=["beta", "B"],
        contexts=["math", "physics", "biology"],
    ),
    SymbolEntry(
        "γ",
        "greek",
        "\\gamma",
        "U+03B3",
        "gamma",
        variants=["gamma", "y"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "δ",
        "greek",
        "\\delta",
        "U+03B4",
        "delta",
        variants=["delta"],
        contexts=["math", "physics", "geology"],
    ),
    SymbolEntry(
        "ε",
        "greek",
        "\\epsilon",
        "U+03B5",
        "epsilon",
        variants=["epsilon", "e", "ɛ"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "η",
        "greek",
        "\\eta",
        "U+03B7",
        "eta",
        variants=["eta", "n"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "θ",
        "greek",
        "\\theta",
        "U+03B8",
        "theta",
        variants=["theta", "0"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "λ",
        "greek",
        "\\lambda",
        "U+03BB",
        "lambda",
        variants=["lambda"],
        contexts=["math", "physics", "biology"],
    ),
    SymbolEntry(
        "μ",
        "greek",
        "\\mu",
        "U+03BC",
        "mu",
        variants=["µ", "u", "micro"],
        contexts=["physics", "biology", "units"],
    ),
    SymbolEntry(
        "ν",
        "greek",
        "\\nu",
        "U+03BD",
        "nu",
        variants=["nu", "v"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "π",
        "greek",
        "\\pi",
        "U+03C0",
        "pi",
        variants=["pi"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "ρ",
        "greek",
        "\\rho",
        "U+03C1",
        "rho",
        variants=["rho", "p"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "σ",
        "greek",
        "\\sigma",
        "U+03C3",
        "sigma",
        variants=["sigma"],
        contexts=["math", "stats"],
    ),
    SymbolEntry(
        "τ",
        "greek",
        "\\tau",
        "U+03C4",
        "tau",
        variants=["tau"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "φ",
        "greek",
        "\\phi",
        "U+03C6",
        "phi",
        variants=["phi", "ϕ"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "χ",
        "greek",
        "\\chi",
        "U+03C7",
        "chi",
        variants=["chi", "x"],
        contexts=["math", "stats"],
    ),
    SymbolEntry(
        "ψ",
        "greek",
        "\\psi",
        "U+03C8",
        "psi",
        variants=["psi"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "ω",
        "greek",
        "\\omega",
        "U+03C9",
        "omega",
        variants=["omega", "w"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "Δ",
        "greek",
        "\\Delta",
        "U+0394",
        "Delta (uppercase)",
        variants=["delta", "A"],
        contexts=["math", "geology"],
    ),
    SymbolEntry(
        "Σ",
        "greek",
        "\\Sigma",
        "U+03A3",
        "Sigma (uppercase)",
        variants=["sigma"],
        contexts=["math", "stats"],
    ),
    SymbolEntry(
        "Ω",
        "greek",
        "\\Omega",
        "U+03A9",
        "Omega (uppercase)",
        variants=["omega"],
        contexts=["physics", "engineering"],
    ),
    SymbolEntry(
        "Γ",
        "greek",
        "\\Gamma",
        "U+0393",
        "Gamma (uppercase)",
        variants=["gamma"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "Λ",
        "greek",
        "\\Lambda",
        "U+039B",
        "Lambda (uppercase)",
        variants=["lambda"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "Θ",
        "greek",
        "\\Theta",
        "U+0398",
        "Theta (uppercase)",
        variants=["theta"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "Φ",
        "greek",
        "\\Phi",
        "U+03A6",
        "Phi (uppercase)",
        variants=["phi"],
        contexts=["math", "physics"],
    ),
]

# === Math symbols ========================================================= #
_MATH_ENTRIES = [
    SymbolEntry(
        "×",
        "math",
        "\\times",
        "U+00D7",
        "multiplication sign",
        variants=["x", "*", "·", "•"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "·",
        "math",
        "\\cdot",
        "U+00B7",
        "middle dot",
        variants=[".", "*"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "÷",
        "math",
        "\\div",
        "U+00F7",
        "division sign",
        variants=["/", ":"],
        contexts=["math"],
    ),
    SymbolEntry(
        "±",
        "math",
        "\\pm",
        "U+00B1",
        "plus-minus",
        variants=["+-", "+/-"],
        contexts=["math", "stats"],
    ),
    SymbolEntry(
        "∓", "math", "\\mp", "U+2213", "minus-plus", variants=["-+"], contexts=["math"]
    ),
    SymbolEntry(
        "≤",
        "math",
        "\\leq",
        "U+2264",
        "less-than or equal",
        variants=["<=", "≦"],
        contexts=["math"],
    ),
    SymbolEntry(
        "≥",
        "math",
        "\\geq",
        "U+2265",
        "greater-than or equal",
        variants=[">=", "≧"],
        contexts=["math"],
    ),
    SymbolEntry(
        "≠",
        "math",
        "\\neq",
        "U+2260",
        "not equal",
        variants=["!=", "<>"],
        contexts=["math"],
    ),
    SymbolEntry(
        "≈",
        "math",
        "\\approx",
        "U+2248",
        "almost equal",
        variants=["~=", "≈"],
        contexts=["math"],
    ),
    SymbolEntry(
        "∼",
        "math",
        "\\sim",
        "U+223C",
        "tilde operator",
        variants=["~"],
        contexts=["math", "stats"],
    ),
    SymbolEntry(
        "∝",
        "math",
        "\\propto",
        "U+221D",
        "proportional to",
        variants=["~", "α"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "∞",
        "math",
        "\\infty",
        "U+221E",
        "infinity",
        variants=["inf"],
        contexts=["math"],
    ),
    SymbolEntry(
        "∂",
        "math",
        "\\partial",
        "U+2202",
        "partial derivative",
        variants=["d", "δ"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "∇",
        "math",
        "\\nabla",
        "U+2207",
        "nabla (del)",
        variants=["V"],
        contexts=["math", "physics"],
    ),
    SymbolEntry(
        "∫",
        "math",
        "\\int",
        "U+222B",
        "integral",
        variants=["I", "f"],
        contexts=["math"],
    ),
    SymbolEntry(
        "∑",
        "math",
        "\\sum",
        "U+2211",
        "summation",
        variants=["E", "S"],
        contexts=["math", "stats"],
    ),
    SymbolEntry(
        "∏",
        "math",
        "\\prod",
        "U+220F",
        "product",
        variants=["P", "pi"],
        contexts=["math"],
    ),
    SymbolEntry(
        "√",
        "math",
        "\\sqrt",
        "U+221A",
        "square root",
        variants=["V", "sqrt"],
        contexts=["math"],
    ),
    SymbolEntry(
        "→",
        "math",
        "\\to",
        "U+2192",
        "rightwards arrow",
        variants=["->", "→"],
        contexts=["math", "chemistry"],
    ),
    SymbolEntry(
        "←",
        "math",
        "\\leftarrow",
        "U+2190",
        "leftwards arrow",
        variants=["<-"],
        contexts=["math"],
    ),
    SymbolEntry(
        "↔",
        "math",
        "\\leftrightarrow",
        "U+2194",
        "left-right arrow",
        variants=["<->"],
        contexts=["math", "chemistry"],
    ),
    SymbolEntry(
        "⇌",
        "math",
        "\\rightleftharpoons",
        "U+21CC",
        "equilibrium",
        variants=["<>"],
        contexts=["chemistry"],
    ),
    SymbolEntry(
        "∈", "math", "\\in", "U+2208", "element of", variants=["e"], contexts=["math"]
    ),
    SymbolEntry(
        "∉",
        "math",
        "\\notin",
        "U+2209",
        "not element of",
        variants=["!e"],
        contexts=["math"],
    ),
    SymbolEntry(
        "⊂",
        "math",
        "\\subset",
        "U+2282",
        "subset of",
        variants=["c"],
        contexts=["math"],
    ),
    SymbolEntry(
        "⊆",
        "math",
        "\\subseteq",
        "U+2286",
        "subset or equal",
        variants=["<="],
        contexts=["math"],
    ),
    SymbolEntry(
        "∪", "math", "\\cup", "U+222A", "union", variants=["U"], contexts=["math"]
    ),
    SymbolEntry(
        "∩",
        "math",
        "\\cap",
        "U+2229",
        "intersection",
        variants=["n"],
        contexts=["math"],
    ),
    SymbolEntry(
        "∀", "math", "\\forall", "U+2200", "for all", variants=["A"], contexts=["math"]
    ),
    SymbolEntry(
        "∃",
        "math",
        "\\exists",
        "U+2203",
        "there exists",
        variants=["E"],
        contexts=["math"],
    ),
    SymbolEntry(
        "°",
        "math",
        "^{\\circ}",
        "U+00B0",
        "degree",
        variants=["deg", "0"],
        contexts=["math", "physics", "geology"],
    ),
]

# === Build registry ======================================================= #
_ALL_ENTRIES = _GREEK_ENTRIES + _MATH_ENTRIES

# Index by canonical, by LaTeX, by variant, by codepoint
_BY_CANONICAL: dict[str, SymbolEntry] = {}
_BY_LATEX: dict[str, SymbolEntry] = {}
_BY_VARIANT: dict[str, list[SymbolEntry]] = {}
_BY_CODEPOINT: dict[str, SymbolEntry] = {}

for e in _ALL_ENTRIES:
    _BY_CANONICAL[e.canonical] = e
    if e.latex:
        _BY_LATEX[e.latex] = e
    if e.codepoint:
        _BY_CODEPOINT[e.codepoint] = e
    for v in e.variants:
        _BY_VARIANT.setdefault(v, []).append(e)


def lookup(symbol: str) -> Optional[SymbolEntry]:
    """Look up a symbol by any form (canonical/variant/LaTeX/codepoint)."""
    if symbol in _BY_CANONICAL:
        return _BY_CANONICAL[symbol]
    if symbol in _BY_LATEX:
        return _BY_LATEX[symbol]
    if symbol in _BY_CODEPOINT:
        return _BY_CODEPOINT[symbol]
    matches = _BY_VARIANT.get(symbol, [])
    if matches:
        return matches[0]
    return None


def find_variants(symbol: str) -> list:
    """Return all known variant forms of a symbol."""
    entry = lookup(symbol)
    return entry.variants if entry else []


def is_same_symbol(a: str, b: str) -> bool:
    """True if a and b are variants of the same canonical symbol."""
    ea = lookup(a)
    eb = lookup(b)
    if not ea or not eb:
        return a == b
    return ea.canonical == eb.canonical


def canonicalize(symbol: str) -> str:
    """Return canonical form, or input if unknown."""
    entry = lookup(symbol)
    return entry.canonical if entry else symbol


def latex_for(symbol: str) -> Optional[str]:
    """Return LaTeX command for symbol, or None."""
    entry = lookup(symbol)
    return entry.latex if entry else None


def all_by_category(category: str) -> list[SymbolEntry]:
    """Return all entries in a category."""
    return [e for e in _ALL_ENTRIES if e.category == category]


def registry_stats() -> dict:
    """Summary statistics."""
    cats = {}
    for e in _ALL_ENTRIES:
        cats[e.category] = cats.get(e.category, 0) + 1
    return {
        "total_entries": len(_ALL_ENTRIES),
        "by_category": cats,
        "canonical_count": len(_BY_CANONICAL),
        "latex_count": len(_BY_LATEX),
        "variant_count": len(_BY_VARIANT),
    }


__all__ = [
    "SymbolEntry",
    "lookup",
    "find_variants",
    "is_same_symbol",
    "canonicalize",
    "latex_for",
    "all_by_category",
    "registry_stats",
]
