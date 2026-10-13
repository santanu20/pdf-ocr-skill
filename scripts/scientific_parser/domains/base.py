"""DomainPack Protocol — pluggable domain intelligence.

Per architecture proposal items #3, #4, #13, #14 (2026-08-12):
  "Use DomainPack as a plugin interface. The core should know nothing about
   geology. Domains extend core ontology. Multiple domains can activate
   simultaneously for interdisciplinary papers.

   DomainPack[] not DomainPack."

A DomainPack provides:
  - detect(document) → probability this pack applies
  - lexicon() → domain vocabulary for OCR correction
  - validators() → list of Validator instances for domain-specific checks
  - entity_types() → domain ontology (Mineral, Reaction, Particle, etc.)
  - candidates(token) → SymSpell-like candidate generation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable


@runtime_checkable
class Validator(Protocol):
    """Composable validator protocol (proposal item #15)."""

    name: str

    def validate(self, value: Any, context: Optional[dict] = None) -> tuple[bool, str]:
        """Return (ok, reason). If not ok, reason explains."""
        ...


@runtime_checkable
class DomainPack(Protocol):
    """Plugin interface for domain intelligence (proposal item #4)."""

    name: str

    def detect(self, document_text: str) -> float:
        """Return probability [0, 1] that this domain applies to the document."""
        ...

    def lexicon(self) -> dict:
        """Return {word: frequency} for OCR correction. Empty dict = no lexicon."""
        ...

    def entity_types(self) -> list:
        """Return list of entity type names (e.g. ['Mineral', 'Rock', ...])."""
        ...

    def validators(self) -> list:
        """Return list of Validator instances for domain-specific checks."""
        ...

    def candidates(self, token: str, context: Optional[dict] = None) -> list:
        """Return candidate corrections for a noisy token."""
        ...


# === Registry ============================================================ #

_REGISTRY: dict[str, DomainPack] = {}


def register(pack: DomainPack) -> None:
    """Register a DomainPack instance."""
    _REGISTRY[pack.name] = pack


def get(name: str) -> Optional[DomainPack]:
    return _REGISTRY.get(name)


def all_packs() -> list:
    return list(_REGISTRY.values())


def detect_domains(document_text: str, threshold: float = 0.20) -> list:
    """Return [(pack_name, score)] for all packs above threshold, sorted desc."""
    scored = [(p.name, p.detect(document_text)) for p in _REGISTRY.values()]
    scored = [(n, s) for n, s in scored if s >= threshold]
    scored.sort(key=lambda x: -x[1])
    return scored


def active_lexicon(document_text: str) -> dict:
    """Merge lexicons from all active domain packs. Weighted by detection score."""
    merged: dict[str, float] = {}
    for pack in _REGISTRY.values():
        score = pack.detect(document_text)
        if score < 0.20:
            continue
        for word, freq in pack.lexicon().items():
            # Highest detection-weighted frequency wins
            weight = freq * score
            if weight > merged.get(word, 0):
                merged[word] = weight
    return merged


def active_validators(document_text: str) -> list:
    """Collect validators from all active packs."""
    out = []
    for pack in _REGISTRY.values():
        if pack.detect(document_text) >= 0.20:
            out.extend(pack.validators())
    return out


__all__ = [
    "Validator",
    "DomainPack",
    "register",
    "get",
    "all_packs",
    "detect_domains",
    "active_lexicon",
    "active_validators",
]
