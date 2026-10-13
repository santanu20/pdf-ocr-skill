"""First-class extraction candidates (reviewer #38/#18/#5).

A ``Candidate`` is one extraction attempt for one region from one source
(native PDF text, a specialist provider, an OCR engine). A ``CandidateSet``
holds every attempt for a region; its ``winner()`` is chosen by the existing
weighted :class:`EvidenceResolver` (source reliability + confidence +
verification) — never by a single model's say-so, and never silently
discarded.

Why this matters (reviewer #5): the prior region dispatcher produced ONE
string per region (first-wins). A missed footnote or a failed specialist
attempt left no audit trail. Capturing a ``CandidateSet`` makes every attempt
inspectable, enables abstention, and is the prerequisite for learned routing
and training-data capture.

Design:
  - ``Candidate`` wraps an :class:`Evidence` (level/source/confidence/
    verification) + the extracted text + provenance metadata (crop bbox,
    render DPI, source engine). Immutable.
  - ``CandidateSet.winner()`` delegates to ``EvidenceResolver`` so the
    selection policy lives in exactly one place.
  - ``CandidateSet.resolved_text()`` is the markdown-emitting text of the
    winner (empty when no candidate passed).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scientific_parser.core.evidence import DEFAULT_RESOLVER, EvidenceResolver
from scientific_parser.core.ir import Evidence


@dataclass(frozen=True)
class Candidate:
    """One extraction attempt for one region from one source.

    Attributes:
      source_engine: who produced this (``"native-pdf-text"``,
        ``"formula-provider"``, ``"table-provider"``, ``"rapidocr"``, ...).
      evidence: :class:`Evidence` carrying level/source/confidence/
        verification — the resolver scores this.
      text: the extracted markdown/text for the region.
      crop_bbox: ``(x0, y0, x1, y1)`` of the source crop in render px
        (``(0,0,0,0)`` for full-page / native text layer).
      render_dpi: DPI at which a raster candidate was produced (0 for native).
      metadata: free-form provenance (model version, prompt version, etc.).
    """

    source_engine: str
    evidence: Evidence
    text: str
    crop_bbox: tuple = (0, 0, 0, 0)
    render_dpi: float = 0.0
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "source_engine": self.source_engine,
            "evidence_level": self.evidence.level,
            "confidence": self.evidence.confidence,
            "verification": (
                None
                if self.evidence.verification is None
                else {
                    "score": self.evidence.verification.score,
                    "passed": self.evidence.verification.passed,
                    "validator_id": self.evidence.verification.validator_id,
                }
            ),
            "text_len": len(self.text),
            "crop_bbox": list(self.crop_bbox),
            "render_dpi": self.render_dpi,
        }


@dataclass
class CandidateSet:
    """All extraction attempts for one region.

    Attributes:
      region_id: stable id (e.g. ``"p3_r1"``).
      region_type: ``"equation"``, ``"table"``, ``"text"``, ...
      bbox: region bbox in render px.
      candidates: list of :class:`Candidate` (insertion order preserved).
    """

    region_id: str
    region_type: str
    bbox: tuple = (0, 0, 0, 0)
    candidates: list[Candidate] = field(default_factory=list)

    def add(self, candidate: Candidate) -> None:
        """Record one extraction attempt."""
        self.candidates.append(candidate)

    def winner(self, resolver: EvidenceResolver | None = None) -> Candidate | None:
        """Return the highest-scoring candidate (weighted resolver), or None."""
        if not self.candidates:
            return None
        if len(self.candidates) == 1:
            return self.candidates[0]
        resolver = resolver or DEFAULT_RESOLVER
        ev_winner = resolver.resolve([c.evidence for c in self.candidates])
        # EvidenceResolver.resolve returns the max Evidence by identity; map
        # back to the owning Candidate.
        for c in self.candidates:
            if c.evidence is ev_winner:
                return c
        return self.candidates[0]

    def resolved_text(self, resolver: EvidenceResolver | None = None) -> str:
        """Markdown text of the winning candidate (empty if no candidate)."""
        w = self.winner(resolver)
        return w.text if w else ""

    def agreement(self) -> float | None:
        """Fraction of candidates whose text matches the winner (1.0 = full).

        A low agreement signals cross-source disagreement — the trigger for
        selective redundancy (reviewer #6) and the risk signal for abstention.
        None when fewer than 2 candidates.
        """
        if len(self.candidates) < 2:
            return None
        w = self.winner()
        if w is None:
            return None
        wt = w.text.strip()
        if not wt:
            return None
        matches = sum(1 for c in self.candidates if c.text.strip() == wt)
        return matches / len(self.candidates)

    def to_dict(self) -> dict:
        w = self.winner()
        return {
            "region_id": self.region_id,
            "region_type": self.region_type,
            "bbox": list(self.bbox),
            "n_candidates": len(self.candidates),
            "agreement": self.agreement(),
            "winner_source": w.source_engine if w else None,
            "winner_level": w.evidence.level if w else None,
            "candidates": [c.to_dict() for c in self.candidates],
        }


__all__ = ["Candidate", "CandidateSet"]
