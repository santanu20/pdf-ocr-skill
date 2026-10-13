"""Chemistry domain pack — molecules, reactions, concentrations."""

from __future__ import annotations
from typing import Optional

_LEXICON = {
    "reaction": 100,
    "catalyst": 80,
    "reagent": 70,
    "solvent": 70,
    "concentration": 100,
    "molarity": 80,
    "molality": 60,
    "molar": 70,
    "stoichiometry": 70,
    "enthalpy": 80,
    "entropy": 80,
    "Gibbs": 60,
    "oxidation": 90,
    "reduction": 90,
    "redox": 80,
    "titration": 70,
    "synthesis": 80,
    "hydrolysis": 60,
    "polymerization": 60,
    "absorption": 60,
    "adsorption": 60,
    "desorption": 50,
    "infrared": 60,
    "NMR": 70,
    "mass spec": 50,
    "chromatography": 60,
    "pH": 100,
    "pKa": 80,
    "pKb": 60,
    "Ksp": 60,
    "Keq": 60,
    "kinetics": 80,
    "equilibrium": 80,
    "activation energy": 60,
    "Avogadro": 50,
    "Boltzmann": 50,
    "alcohol": 60,
    "aldehyde": 50,
    "ketone": 50,
    "ester": 50,
    "ether": 50,
    "alkane": 50,
    "alkene": 50,
    "alkyne": 50,
    "aromatic": 50,
    "benzene": 60,
}


class ChemistryPack:
    name = "chemistry"

    def detect(self, document_text: str) -> float:
        if not document_text:
            return 0.0
        tl = document_text.lower()
        score = sum(0.10 for w in _LEXICON if w.lower() in tl)
        import re

        if re.search(r"\b[A-Z][a-z]?\d+[A-Z]", document_text):
            score += 0.20
        if "mol/L" in tl or "M (" in tl:
            score += 0.15
        return min(1.0, score)

    def lexicon(self):
        return dict(_LEXICON)

    def entity_types(self):
        return [
            "Molecule",
            "Element",
            "Reaction",
            "Concentration",
            "Solvent",
            "Catalyst",
        ]

    def validators(self):
        return []

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3:
            return []
        tl = token.lower()
        return [w for w in _LEXICON if w.lower().startswith(tl[:3])][:5]


PACK = ChemistryPack()


def register():
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, "scripts")
    from scientific_parser.domains import base

    base.register(PACK)


__all__ = ["ChemistryPack", "PACK", "register"]
