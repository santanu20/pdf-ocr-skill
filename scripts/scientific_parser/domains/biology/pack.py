"""Biology domain pack."""

from __future__ import annotations
from typing import Optional

_LEXICON = {
    "gene": 100,
    "genome": 90,
    "genomics": 80,
    "allele": 70,
    "locus": 60,
    "protein": 100,
    "enzyme": 90,
    "peptide": 70,
    "amino acid": 80,
    "transcription": 80,
    "translation": 80,
    "expression": 80,
    "organism": 90,
    "species": 90,
    "genus": 70,
    "family": 60,
    "cell": 90,
    "membrane": 70,
    "nucleus": 70,
    "mitochondria": 70,
    "chromosome": 80,
    "DNA": 100,
    "RNA": 100,
    "mRNA": 80,
    "PCR": 80,
    "electrophoresis": 60,
    "sequencing": 70,
    "cloning": 60,
    "evolution": 80,
    "phylogenetic": 70,
    "mutation": 80,
    "assay": 70,
    "ELISA": 60,
    "MIC": 50,
    "IC50": 60,
    "in vivo": 70,
    "in vitro": 70,
}


class BiologyPack:
    name = "biology"

    def detect(self, document_text: str) -> float:
        if not document_text:
            return 0.0
        tl = document_text.lower()
        return min(1.0, sum(0.08 for w in _LEXICON if w.lower() in tl))

    def lexicon(self):
        return dict(_LEXICON)

    def entity_types(self):
        return ["Gene", "Protein", "Organism", "Species", "Assay"]

    def validators(self):
        return []

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3:
            return []
        return [w for w in _LEXICON if w.lower().startswith(token.lower()[:3])][:5]


PACK = BiologyPack()


def register():
    import sys

    if "scripts" not in sys.path:
        sys.path.insert(0, "scripts")
    from scientific_parser.domains import base

    base.register(PACK)


__all__ = ["BiologyPack", "PACK", "register"]
