"""Cross-page entity stitching (reviewer #28).

Scientific documents spread one logical entity across pages:
  - a table continues on the next page ("Table 2 (continued)")
  - a figure's caption lands on the following page
  - an equation is defined as "Eq. (12)" and referenced pages later

This module detects those continuation / back-reference markers and links
occurrences of the same label across pages into one logical entity. It is the
coreference-light core of cross-page reasoning: full entity coreference is a
later step, but linking explicit numbered labels is cheap, deterministic, and
high-value (a continued table's rows should merge with its header).

Pure, no I/O. Inputs are per-page text; output is a registry of linked
occurrences.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Continuation markers: "Table 2 (continued)", "Table 2, continued", "cont."
_CONTINUATION_RE = re.compile(
    r"\b(Table|Fig(?:ure)?|Equation|Eq\.?)\s*\.?\s*(\d+[A-Za-z]?)\b"
    r"\s*(?:[\(\[]\s*(?:continued|cont\.?|cont\d)\s*[\)\]]|(?:continued|cont\.?))",
    re.IGNORECASE,
)
# Label definitions / first occurrences: "Table 2", "Figure 5", "Eq. (12)"
_LABEL_RE = re.compile(
    r"\b(Table|Fig(?:ure)?|Equation|Eq\.?)\s*\.?\s*(?:[\(\[])?\s*(\d+[A-Za-z]?)\s*(?:[\)\]])?",
    re.IGNORECASE,
)


@dataclass
class LabelOccurrence:
    """One mention of a numbered label on a page."""

    kind: str  # "table" | "figure" | "equation" (normalized)
    number: str  # "2", "12a"
    page: int
    is_continuation: bool
    text_span: str  # the matched text (for diagnostics)


@dataclass
class CrossPageEntity:
    """All occurrences of one numbered label across the document."""

    kind: str
    number: str
    occurrences: list[LabelOccurrence] = field(default_factory=list)

    @property
    def spans_pages(self) -> list[int]:
        """Distinct pages where this entity appears (sorted)."""
        return sorted({o.page for o in self.occurrences})

    @property
    def is_cross_page(self) -> bool:
        """True if the entity appears on >1 page (incl. via continuation)."""
        return len(self.spans_pages) > 1 or any(o.is_continuation for o in self.occurrences)

    @property
    def has_continuation(self) -> bool:
        return any(o.is_continuation for o in self.occurrences)

    def to_dict(self) -> dict:
        """Serialize including computed properties (spans_pages etc.).

        ``__dict__`` alone misses ``@property`` values, so use this for the
        sidecar JSON (the prior ``vars(e)`` serialization emitted None for
        spans_pages)."""
        return {
            "kind": self.kind,
            "number": self.number,
            "spans_pages": self.spans_pages,
            "is_cross_page": self.is_cross_page,
            "has_continuation": self.has_continuation,
            "n_occurrences": len(self.occurrences),
            "occurrences": [
                {"page": o.page, "is_continuation": o.is_continuation, "text_span": o.text_span}
                for o in self.occurrences
            ],
        }


def _normalize_kind(raw: str) -> str:
    r = raw.lower()
    if r.startswith("table"):
        return "table"
    if r.startswith("eq"):
        return "equation"
    return "figure"  # Fig / Figure


def extract_label_occurrences(text: str, page: int) -> list[LabelOccurrence]:
    """Find all numbered label mentions on a page, marking continuations."""
    out: list[LabelOccurrence] = []
    if not text:
        return out
    # Continuations first (they are a subset of label matches but carry extra
    # signal). Track their spans to avoid double-counting.
    cont_spans: list[tuple[int, int]] = []
    for m in _CONTINUATION_RE.finditer(text):
        cont_spans.append(m.span())
        out.append(
            LabelOccurrence(
                kind=_normalize_kind(m.group(1)),
                number=m.group(2),
                page=page,
                is_continuation=True,
                text_span=m.group(0),
            )
        )
    for m in _LABEL_RE.finditer(text):
        # skip if this match is inside a continuation span already recorded
        if any(s <= m.start() and m.end() <= e for s, e in cont_spans):
            continue
        out.append(
            LabelOccurrence(
                kind=_normalize_kind(m.group(1)),
                number=m.group(2),
                page=page,
                is_continuation=False,
                text_span=m.group(0),
            )
        )
    return out


def build_cross_page_registry(
    pages: list[str],
) -> list[CrossPageEntity]:
    """Link numbered labels across pages into cross-page entities.

    Args:
      pages: list of per-page text (1-indexed conceptually; index 0 = page 1).

    Returns:
      List of :class:`CrossPageEntity`, one per distinct (kind, number).
      Entities that appear on only one page AND are not continuations are
      included (callers can filter on :attr:`CrossPageEntity.is_cross_page`).
    """
    registry: dict[tuple[str, str], CrossPageEntity] = {}
    for i, text in enumerate(pages, start=1):
        for occ in extract_label_occurrences(text or "", i):
            key = (occ.kind, occ.number)
            ent = registry.get(key)
            if ent is None:
                ent = CrossPageEntity(kind=occ.kind, number=occ.number)
                registry[key] = ent
            ent.occurrences.append(occ)
    return list(registry.values())


def cross_page_entities(pages: list[str]) -> list[CrossPageEntity]:
    """Convenience: return only entities that span pages / are continued."""
    return [e for e in build_cross_page_registry(pages) if e.is_cross_page]


__all__ = [
    "LabelOccurrence",
    "CrossPageEntity",
    "extract_label_occurrences",
    "build_cross_page_registry",
    "cross_page_entities",
]
