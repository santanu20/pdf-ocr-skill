"""Geology domain pack — extracted from pdf-ocr monolith (proposal item #12).

Geology lexicon is large (~500+ minerals, rocks, formations). This module
provides the vocabulary; the OCR-correction pipeline (SymSpell) uses
DomainPack.candidates() instead of a hardcoded dict in the monolith.

Detection: counts geo-specific keywords (mineral names, geochemistry units,
rock types) → probability.
"""

from __future__ import annotations

import re
from typing import Optional


# Common rock-forming minerals (silicates + oxides + carbonates)
_MINERALS = {
    "quartz": 100,
    "feldspar": 100,
    "plagioclase": 90,
    "alkali feldspar": 70,
    "orthoclase": 60,
    "albite": 60,
    "anorthite": 60,
    "microcline": 40,
    "mica": 80,
    "muscovite": 70,
    "biotite": 70,
    "phlogopite": 40,
    "amphibole": 80,
    "hornblende": 70,
    "actinolite": 50,
    "tremolite": 50,
    "pyroxene": 80,
    "augite": 60,
    "diopside": 50,
    "enstatite": 50,
    "hypersthene": 40,
    "olivine": 80,
    "forsterite": 50,
    "fayalite": 40,
    "garnet": 70,
    "almandine": 40,
    "pyrope": 40,
    "grossular": 40,
    "spessartine": 30,
    "epidote": 50,
    "zoisite": 30,
    "clinozoisite": 30,
    "chlorite": 60,
    "serpentine": 50,
    "talc": 50,
    "kaolinite": 60,
    "illite": 40,
    "smectite": 40,
    "montmorillonite": 40,
    "vermiculite": 30,
    "calcite": 80,
    "dolomite": 80,
    "aragonite": 50,
    "siderite": 50,
    "magnesite": 40,
    "halite": 50,
    "gypsum": 60,
    "anhydrite": 50,
    "barite": 50,
    "celestine": 30,
    "halides": 30,
    "sulfates": 40,
    "halide": 20,
    "sulfate": 30,
    "carbonate": 50,
    "silicate": 50,
    "oxide": 50,
    "magnetite": 70,
    "hematite": 80,
    "goethite": 50,
    "limonite": 40,
    "ilmenite": 50,
    "rutile": 50,
    "anatase": 30,
    "brookite": 20,
    "spinel": 50,
    "chromite": 50,
    "pyrite": 70,
    "pyrrhotite": 40,
    "chalcopyrite": 50,
    "galena": 50,
    "sphalerite": 50,
    "pentlandite": 30,
    "covellite": 20,
    "bornite": 30,
    "molybdenite": 40,
    "apatite": 60,
    "monazite": 40,
    "xenotime": 30,
    "zircon": 70,
    "titanite": 40,
    "tourmaline": 50,
    "topaz": 30,
    "beryl": 40,
    "cordierite": 40,
    "sillimanite": 40,
    "kyanite": 40,
    "andalusite": 40,
    "staurolite": 40,
    " wollastonite": 20,
    "prehnite": 20,
    "pumpellyite": 20,
}

# Igneous, sedimentary, metamorphic rock types
_ROCKS = {
    "granite": 100,
    "granodiorite": 70,
    "tonalite": 50,
    "diorite": 60,
    "gabbro": 70,
    "syenite": 50,
    "monzonite": 40,
    "peridotite": 60,
    "dunite": 40,
    "pyroxenite": 40,
    "hornblendite": 30,
    "anorthosite": 50,
    "rhyolite": 70,
    "dacite": 50,
    "andesite": 70,
    "basalt": 90,
    "trachyte": 40,
    "phonolite": 30,
    "komatiite": 40,
    "obsidian": 50,
    "pumice": 50,
    "tuff": 60,
    "ignimbrite": 40,
    "volcanic breccia": 30,
    "lapilli": 30,
    "scoria": 30,
    "sandstone": 100,
    "limestone": 100,
    "shale": 90,
    "mudstone": 70,
    "siltstone": 70,
    "conglomerate": 70,
    "breccia": 60,
    "chert": 50,
    "dolostone": 50,
    "chalk": 40,
    "coal": 60,
    "lignite": 40,
    "peat": 30,
    "evaporite": 40,
    "halite rock": 20,
    "marl": 40,
    "argillite": 30,
    "greywacke": 40,
    "arkose": 40,
    "gneiss": 80,
    "schist": 80,
    "slate": 70,
    "phyllite": 50,
    "marble": 70,
    "quartzite": 70,
    "amphibolite": 60,
    "granulite": 40,
    "eclogite": 40,
    "serpentinite": 50,
    "mylonite": 40,
    "migmatite": 40,
    "skarn": 40,
    "hornfels": 40,
}

# Geochemistry units (high-signal detection features)
_GEOCHEM_UNITS = {
    "wt%": 100,
    "wt.%": 90,
    "weight percent": 70,
    "ppm": 100,
    "ppb": 80,
    "ppt": 60,
    "Mg#": 80,
    "Mg number": 70,
    "Mg-number": 60,
    "A/CNK": 50,
    "A/NK": 40,
    "ASI": 40,  # alumina saturation indices
    "Eu/Eu*": 60,
    "La/Yb": 50,
    "Sm/Nd": 50,
    "Rb/Sr": 50,
    "Th/U": 40,
    "δ¹⁸O": 60,
    "δ¹³C": 60,
    "δ³⁴S": 50,
    "δD": 50,
    "δ¹⁵N": 40,
    "εNd": 50,
    "εHf": 40,
    "⁸⁷Sr/⁸⁶Sr": 50,
    "¹⁴³Nd/¹⁴⁴Nd": 40,
    "Ma": 100,
    "Ga": 80,
    "ka": 60,
    "TAS": 50,  # total alkali silica
}

# Geochronology systems
_GEOCHRON = {
    "U-Pb": 80,
    "Pb-Pb": 50,
    "K-Ar": 70,
    "Ar-Ar": 70,
    "⁴⁰Ar/³⁹Ar": 80,
    "Rb-Sr": 60,
    "Sm-Nd": 60,
    "Lu-Hf": 40,
    "Re-Os": 40,
    "La-Ba": 20,
    "fission track": 50,
    "(U-Th)/He": 50,
    "cosmogenic": 50,
    "¹⁴C": 70,
    "zircon": 80,
    "monazite": 50,
    "apatite": 50,
    "titanite": 40,
    "baddeleyite": 30,
    "concordia": 50,
    "discordia": 40,
    "intercept age": 30,
    "isochron": 50,
}

# Geological eras / periods (high-signal detection)
_ERAS = {
    "Archean": 70,
    "Hadean": 50,
    "Proterozoic": 80,
    "Phanerozoic": 70,
    "Cambrian": 70,
    "Ordovician": 70,
    "Silurian": 60,
    "Devonian": 70,
    "Carboniferous": 70,
    "Permian": 70,
    "Triassic": 80,
    "Jurassic": 80,
    "Cretaceous": 90,
    "Paleogene": 70,
    "Neogene": 60,
    "Quaternary": 70,
    "Holocene": 60,
    "Pleistocene": 70,
    "Pliocene": 60,
    "Miocene": 70,
    "Oligocene": 60,
    "Eocene": 70,
    "Paleocene": 60,
    "Cenozoic": 60,
    "Mesozoic": 60,
    "Paleozoic": 60,
}

# Detection keywords (subset used for fast scoring)
_DETECT_KEYWORDS = (
    list(_MINERALS.keys())[:30]
    + list(_ROCKS.keys())[:20]
    + [
        "geology",
        "geological",
        "geochemistry",
        "petrology",
        "mineralogy",
        "tectonic",
        "stratigraphy",
        "sedimentology",
        "volcanic",
        "igneous",
        "metamorphic",
        "mineral",
        "rock",
        "outcrop",
        "formation",
    ]
)


class GeologyPack:
    """Geology / geochemistry / petrology / mineralogy domain."""

    name = "geology"

    def detect(self, document_text: str) -> float:
        """Score by counting geo-specific tokens. Returns probability [0, 1]."""
        if not document_text:
            return 0.0
        text_lower = document_text.lower()
        # Count distinct geo indicators
        minerals_found = sum(1 for m in _MINERALS if m in text_lower)
        rocks_found = sum(1 for r in _ROCKS if r in text_lower)
        units_found = sum(1 for u in _GEOCHEM_UNITS if u.lower() in text_lower)
        geochron_found = sum(1 for g in _GEOCHRON if g.lower() in text_lower)
        eras_found = sum(1 for e in _ERAS if e in document_text)

        score = (
            minerals_found * 0.05
            + rocks_found * 0.05
            + units_found * 0.15
            + geochron_found * 0.20
            + eras_found * 0.10
        )
        return min(1.0, score)

    def lexicon(self) -> dict:
        merged = {}
        merged.update(_MINERALS)
        merged.update(_ROCKS)
        merged.update(_GEOCHEM_UNITS)
        merged.update(_GEOCHRON)
        merged.update(_ERAS)
        return merged

    def entity_types(self) -> list:
        return [
            "Mineral",
            "Rock",
            "Formation",
            "Sample",
            "Isotope",
            "GeochemicalMeasurement",
            "GeochronologyAge",
            "GeologicalEra",
        ]

    def validators(self) -> list:
        """Geology-specific validators."""
        from scientific_parser.science.validators import NumericValidator, UnitValidator

        return [NumericValidator(), UnitValidator()]
        # TODO: MineralFormulaValidator, OxideTotalValidator (sum to 100±1 wt%)

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        """SymSpell-like — fuzzy match against geo lexicon."""
        if not token or len(token) < 3:
            return []
        token_lower = token.lower()
        # Quick prefix match
        matches = []
        for word in _MINERALS:
            if word.startswith(token_lower[:3]) and word not in matches:
                matches.append(word)
        for word in _ROCKS:
            if word.startswith(token_lower[:3]) and word not in matches:
                matches.append(word)
        return matches[:5]


PACK = GeologyPack()


def register():
    from scientific_parser.domains import base

    base.register(PACK)


__all__ = ["GeologyPack", "PACK", "register"]
