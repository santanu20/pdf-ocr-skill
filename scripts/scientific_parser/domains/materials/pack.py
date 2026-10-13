"""Materials science domain pack."""

from __future__ import annotations
from typing import Optional

_LEXICON = {
    "crystal": 90,
    "crystalline": 80,
    "amorphous": 70,
    "polycrystalline": 60,
    "lattice": 80,
    "unit cell": 70,
    "Miller indices": 50,
    "Bragg": 60,
    "alloy": 80,
    "composite": 70,
    "ceramic": 70,
    "polymer": 80,
    "semiconductor": 80,
    "conductor": 60,
    "insulator": 50,
    "dielectric": 60,
    "tensile": 70,
    "yield strength": 60,
    "hardness": 70,
    "ductility": 50,
    "Young's modulus": 50,
    "bulk modulus": 40,
    "shear modulus": 40,
    "annealing": 60,
    "quenching": 50,
    "tempering": 40,
    "precipitation": 50,
    "microstructure": 70,
    "grain boundary": 60,
    "dislocation": 50,
    "XRD": 70,
    "SEM": 70,
    "TEM": 60,
    "DSC": 40,
    "TGA": 40,
    "thin film": 70,
    "deposition": 50,
    "sputtering": 40,
    "nanoparticle": 70,
    "nanocomposite": 50,
}


class MaterialsPack:
    name = "materials"

    def detect(self, document_text: str) -> float:
        if not document_text:
            return 0.0
        tl = document_text.lower()
        return min(1.0, sum(0.08 for w in _LEXICON if w.lower() in tl))

    def lexicon(self):
        return dict(_LEXICON)

    def entity_types(self):
        return ["Material", "Crystal", "Phase", "Property", "Microstructure"]

    def validators(self):
        return []

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3:
            return []
        return [w for w in _LEXICON if w.lower().startswith(token.lower()[:3])][:5]


PACK = MaterialsPack()


def register():
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, "scripts")
    from scientific_parser.domains import base

    base.register(PACK)


__all__ = ["MaterialsPack", "PACK", "register"]
