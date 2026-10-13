"""Engineering domain pack."""
from __future__ import annotations
from typing import Optional

_LEXICON = {
    "stress": 80, "strain": 80, "fatigue": 70, "fracture": 70,
    "load": 60, "bending": 60, "torsion": 60, "buckling": 50,
    "circuit": 80, "voltage": 80, "current": 80, "resistance": 70,
    "capacitance": 60, "inductance": 50, "impedance": 50,
    "signal": 70, "filter": 50, "amplifier": 50, "oscillator": 40,
    "control system": 60, "feedback": 60, "stability": 60,
    "thermodynamic": 70, "cycle": 50, "Carnot": 40,
    "turbine": 60, "compressor": 50, "pump": 50, "valve": 40,
    "welding": 50, "machining": 40, "casting": 40, "forging": 40,
    "specification": 60, "tolerance": 50, "ISO": 40, "ASTM": 50,
    "CAD": 40, "FEA": 40, "CFD": 50, "simulation": 50,
}


class EngineeringPack:
    name = "engineering"

    def detect(self, document_text: str) -> float:
        if not document_text: return 0.0
        tl = document_text.lower()
        score = sum(0.08 for w in _LEXICON if w.lower() in tl)
        return min(1.0, score)

    def lexicon(self): return dict(_LEXICON)
    def entity_types(self): return ["Component", "Material", "Process",
                                     "Specification", "Measurement"]
    def validators(self): return []
    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3: return []
        tl = token.lower()
        return [w for w in _LEXICON if w.lower().startswith(tl[:3])][:5]


PACK = EngineeringPack()
def register():
    import sys
    if 'scripts' not in sys.path: sys.path.insert(0, 'scripts')
    from scientific_parser.domains import base
    base.register(PACK)

__all__ = ["EngineeringPack", "PACK", "register"]
