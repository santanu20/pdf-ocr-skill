"""Medicine / clinical domain pack."""
from __future__ import annotations
from typing import Optional

_LEXICON = {
    "patient": 90, "clinical": 80, "diagnosis": 80, "treatment": 80,
    "symptom": 80, "prognosis": 70, "etiology": 60, "morbidity": 60,
    "mortality": 70, "incidence": 60, "prevalence": 60,
    "dose": 80, "dosage": 70, "mg/kg": 60, "mg/m²": 50,
    "therapeutic": 70, "pharmacokinetic": 60, "half-life": 60,
    "placebo": 70, "randomized": 70, "trial": 70, "cohort": 60,
    "MRI": 70, "CT scan": 60, "ultrasound": 60, "biopsy": 60,
    "blood pressure": 70, "heart rate": 60, "BMI": 50,
    "mmHg": 60, "bpm": 50,
    "inflammation": 60, "infection": 60, "antibody": 60, "antigen": 50,
}


class MedicinePack:
    name = "medicine"

    def detect(self, document_text: str) -> float:
        if not document_text: return 0.0
        tl = document_text.lower()
        score = sum(0.08 for w in _LEXICON if w.lower() in tl)
        return min(1.0, score)

    def lexicon(self): return dict(_LEXICON)
    def entity_types(self): return ["Patient", "Diagnosis", "Treatment",
                                     "Symptom", "Medication", "Trial"]
    def validators(self): return []
    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        if not token or len(token) < 3: return []
        tl = token.lower()
        return [w for w in _LEXICON if w.lower().startswith(tl[:3])][:5]


PACK = MedicinePack()
def register():
    import sys
    if 'scripts' not in sys.path: sys.path.insert(0, 'scripts')
    from scientific_parser.domains import base
    base.register(PACK)

__all__ = ["MedicinePack", "PACK", "register"]
