"""Astronomy / astrophysics domain pack."""
from __future__ import annotations
from typing import Optional

_LEXICON = {
    "galaxy": 90, "galaxies": 80, "quasar": 80, "pulsar": 70, "nebula": 70,
    "star": 70, "stellar": 80, "supernova": 80, "black hole": 90,
    "redshift": 80, "blueshift": 60, "cosmology": 80, "dark matter": 80,
    "dark energy": 80, "CMB": 70, "Hubble": 70, "parsec": 60,
    "light-year": 60, "astronomical unit": 50, "AU": 40,
    "magnitude": 60, "luminosity": 70, "spectral": 60, "photometric": 50,
    "exoplanet": 80, "transit": 60, "radial velocity": 50,
    "telescope": 70, "Hubble": 50, "James Webb": 50, "JWST": 50,
    "right ascension": 50, "declination": 50, "RA": 30, "Dec": 30,
}


class AstronomyPack:
    name = "astronomy"

    def detect(self, document_text: str) -> float:
        if not document_text: return 0.0
        tl = document_text.lower()
        score = sum(0.10 for w in _LEXICON if w.lower() in tl)
        return min(1.0, score)

    def lexicon(self): return dict(_LEXICON)
    def entity_types(self): return ["Star", "Galaxy", "Constellation",
                                     "Coordinate", "Magnitude"]
    def validators(self): return []
    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3: return []
        tl = token.lower()
        return [w for w in _LEXICON if w.lower().startswith(tl[:3])][:5]


PACK = AstronomyPack()
def register():
    import sys
    if 'scripts' not in sys.path: sys.path.insert(0, 'scripts')
    from scientific_parser.domains import base
    base.register(PACK)

__all__ = ["AstronomyPack", "PACK", "register"]
