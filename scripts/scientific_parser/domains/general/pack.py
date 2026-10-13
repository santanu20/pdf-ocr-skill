"""General scientific domain pack — common to all scientific documents.

Per proposal item #12:
  "The general pack should contain: academic vocabulary, scientific
   abbreviations, units, SI prefixes, mathematical notation, common Greek
   symbols, measurement terminology, method names, instrument terminology,
   statistical terminology."
"""

from __future__ import annotations

import re
from typing import Optional


# SI prefixes (used in many scientific units)
_SI_PREFIXES = {
    "yotta": 1e24,
    "zetta": 1e21,
    "exa": 1e18,
    "peta": 1e15,
    "tera": 1e12,
    "giga": 1e9,
    "mega": 1e6,
    "kilo": 1e3,
    "hecto": 1e2,
    "deca": 1e1,
    "deci": 1e-1,
    "centi": 1e-2,
    "milli": 1e-3,
    "micro": 1e-6,
    "nano": 1e-9,
    "pico": 1e-12,
    "femto": 1e-15,
    "atto": 1e-18,
    "zepto": 1e-21,
    "yocto": 1e-24,
}

# Common scientific abbreviations (proposal: "academic vocabulary")
_ACADEMIC_ABBREVS = {
    "et al.": 100,
    "i.e.": 100,
    "e.g.": 100,
    "vs.": 100,
    "via": 100,
    "etc.": 100,
    "cf.": 50,
    "approx.": 50,
    "ca.": 50,
    "Fig.": 80,
    "Eq.": 80,
    "Table": 80,
    "Section": 60,
    "Appendix": 50,
    "Ch.": 40,
    "Vol.": 40,
    "No.": 60,
    "pp.": 60,
    "pH": 80,
    "ppm": 80,
    "ppb": 70,
    "wt%": 70,
    "mol%": 60,
    "vol%": 50,
    "DI": 30,
    "DI water": 30,
    "DI-Water": 10,
    "rpm": 70,
    "PSI": 50,
    "HV": 40,
    "LV": 40,
    "DC": 50,
    "AC": 50,
    "UV": 70,
    "IR": 70,
    "NMR": 70,
    "EPR": 50,
    "ESR": 40,
    "XRD": 60,
    "XRF": 50,
    "SEM": 70,
    "TEM": 60,
    "AFM": 50,
    "STM": 30,
    "DSC": 40,
    "TGA": 40,
    "CV": 30,
    "EIS": 30,
    "GC": 40,
    "MS": 60,
    "HPLC": 50,
    "FTIR": 50,
}

# Greek letters commonly used in science (LaTeX name → Unicode)
_GREEK_LATEX = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "omicron": "ο",
    "pi": "π",
    "rho": "ρ",
    "sigma": "σ",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Alpha": "Α",
    "Beta": "Β",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Epsilon": "Ε",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
}

# Statistical/method terminology
_STATISTICS = {
    "mean": 100,
    "median": 80,
    "mode": 50,
    "std": 70,
    "std. dev": 60,
    "variance": 80,
    "p-value": 80,
    "confidence interval": 60,
    "CI": 50,
    "n=": 50,
    "N=": 50,
    "SD": 60,
    "SE": 50,
    "MSE": 40,
    "RMSE": 50,
    "ANOVA": 60,
    "t-test": 50,
    "χ²": 40,
    "regression": 80,
    "correlation": 80,
    "r²": 60,
    "R²": 60,
    "r-value": 30,
    "Pearson": 50,
    "Spearman": 40,
}


class GeneralPack:
    """General scientific domain — applies to ALL scientific documents."""

    name = "general"

    def detect(self, document_text: str) -> float:
        """General pack applies when ANY scientific indicator present."""
        if not document_text:
            return 0.0
        text_lower = document_text.lower()
        indicators = 0
        # SI prefix usage
        if any(p in text_lower for p in _SI_PREFIXES):
            indicators += 1
        # Greek letter usage
        if any(g in document_text for g in _GREEK_LATEX.values()):
            indicators += 1
        # Academic abbreviations
        if any(abbr.lower() in text_lower for abbr in _ACADEMIC_ABBREVS):
            indicators += 1
        # Statistical terms
        if any(s.lower() in text_lower for s in _STATISTICS):
            indicators += 1
        # Citations (Author, Year) pattern
        if re.search(r"\([A-Z][a-z]+,?\s+\d{4}\)", document_text):
            indicators += 1
        # References section
        if "references" in text_lower or "bibliography" in text_lower:
            indicators += 1
        # Score: 1+ indicator → 0.50 base, scaling up to 1.0 at 4+
        return min(1.0, 0.30 + indicators * 0.175)

    def lexicon(self) -> dict:
        """Merge academic + statistics vocab."""
        merged = dict(_ACADEMIC_ABBREVS)
        merged.update(_STATISTICS)
        return merged

    def entity_types(self) -> list:
        return [
            "Quantity",
            "Unit",
            "Measurement",
            "Citation",
            "Reference",
            "Person",
            "Date",
            "Equation",
            "Variable",
        ]

    def validators(self) -> list:
        from scientific_parser.science.validators import UnitValidator, NumericValidator

        return [UnitValidator(), NumericValidator()]

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        """SymSpell-like suggestions: token → known scientific abbreviations."""
        if not token or len(token) < 2:
            return []
        candidates = []
        for word in _ACADEMIC_ABBREVS:
            if word.lower().startswith(token.lower()[:3]):
                candidates.append(word)
        return candidates[:5]


# Singleton
PACK = GeneralPack()


def register():
    """Register the GeneralPack with the global domain registry."""
    from scientific_parser.domains import base

    base.register(PACK)


__all__ = ["GeneralPack", "PACK", "register"]
