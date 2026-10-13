"""Mathematics domain pack."""
from __future__ import annotations
from typing import Optional

_LEXICON = {
    "theorem": 90, "lemma": 80, "corollary": 70, "proof": 90, "axiom": 70,
    "polynomial": 80, "matrix": 80, "vector": 80, "tensor": 70, "eigenvalue": 70,
    "derivative": 80, "integral": 80, "differential": 70, "manifold": 70,
    "topology": 70, "geometry": 70, "algebra": 70, "calculus": 60,
    "probability": 80, "stochastic": 60, "Markov": 60, "Bayesian": 60,
    "regression": 70, "optimization": 70, "convex": 50,
    "series": 60, "sequence": 60, "convergence": 60, "limit": 60,
    "linear": 50, "nonlinear": 50, "differential equation": 70,
    "Fourier": 70, "Laplace": 60, "Gaussian": 60, "Poisson": 50,
    "set theory": 50, "graph theory": 50, "combinatorics": 50,
    "Riemann": 50, "Lebesgue": 40, "Hilbert": 40, "Banach": 40,
}


class MathPack:
    name = "mathematics"

    def detect(self, document_text: str) -> float:
        if not document_text: return 0.0
        tl = document_text.lower()
        score = sum(0.06 for w in _LEXICON if w.lower() in tl)
        # Heavy math indicator: many LaTeX blocks
        import re
        if len(re.findall(r"\$\$.+?\$\$", document_text, re.DOTALL)) > 3:
            score += 0.20
        return min(1.0, score)

    def lexicon(self): return dict(_LEXICON)
    def entity_types(self): return ["Theorem", "Variable", "Function",
                                     "Operator", "Set"]
    def validators(self): return []
    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3: return []
        tl = token.lower()
        return [w for w in _LEXICON if w.lower().startswith(tl[:3])][:5]


PACK = MathPack()
def register():
    import sys
    if 'scripts' not in sys.path: sys.path.insert(0, 'scripts')
    from scientific_parser.domains import base
    base.register(PACK)

__all__ = ["MathPack", "PACK", "register"]
