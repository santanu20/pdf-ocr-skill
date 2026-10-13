"""Physics domain pack."""

from __future__ import annotations
from typing import Optional

_LEXICON = {
    "particle": 100,
    "electron": 90,
    "proton": 90,
    "neutron": 80,
    "photon": 90,
    "quark": 80,
    "lepton": 70,
    "boson": 70,
    "fermion": 60,
    "energy": 100,
    "momentum": 90,
    "force": 90,
    "torque": 70,
    "velocity": 90,
    "acceleration": 90,
    "frequency": 80,
    "wavelength": 80,
    "quantum": 90,
    "entanglement": 70,
    "superposition": 60,
    "relativity": 80,
    "spacetime": 60,
    "gravity": 80,
    "gravitational": 70,
    "magnetic": 80,
    "electric": 80,
    "electromagnetic": 90,
    "thermodynamics": 90,
    "entropy": 80,
    "enthalpy": 70,
    "cross-section": 70,
    "scattering": 60,
    "MeV": 80,
    "GeV": 80,
    "keV": 60,
    "eV": 70,
    "isotope": 70,
    "decay": 70,
    "half-life": 70,
}


class PhysicsPack:
    name = "physics"

    def detect(self, document_text: str) -> float:
        if not document_text:
            return 0.0
        tl = document_text.lower()
        return min(1.0, sum(0.08 for w in _LEXICON if w.lower() in tl))

    def lexicon(self):
        return dict(_LEXICON)

    def entity_types(self):
        return ["Particle", "Observable", "Unit", "Experiment"]

    def validators(self):
        return []

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3:
            return []
        return [w for w in _LEXICON if w.lower().startswith(token.lower()[:3])][:5]


PACK = PhysicsPack()


def register():
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, "scripts")
    from scientific_parser.domains import base

    base.register(PACK)


__all__ = ["PhysicsPack", "PACK", "register"]
