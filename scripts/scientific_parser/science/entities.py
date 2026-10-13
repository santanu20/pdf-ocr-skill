"""Entity extractor — domain-neutral scientific entities (proposal item #16).

Per architecture proposal item #16:
  "General scientific semantic types. Build a core ontology around:
   Entity, Quantity, Unit, Measurement, Formula, Equation, Variable,
   Procedure, Instrument, Sample, Material, Species, Location, Date, Time,
   Dataset, Reference, Person, Organization, Figure, Table.

   These are domain-neutral. Then domains extend them."

Implementation:
  - Pure regex-based (no ML)
  - Each extractor returns list of (entity_id, type, value, span)
  - Domain packs register additional extractors via DomainPack protocol
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Entity:
    """One extracted entity."""

    id: str
    type: str  # Quantity|Date|Person|Citation|Unit|Sample|...
    value: str  # raw text matched
    normalized: Optional[str] = None  # cleaned form (e.g. "5 km" → "5km")
    page: Optional[int] = None
    bbox: Optional[tuple] = None
    confidence: float = 1.0
    metadata: dict = field(default_factory=dict)


# === Quantity with unit =====================================================
# Pattern: number + optional space + unit suffix
_QUANTITY_RE = re.compile(
    r"""
    (?<![\w.])
    (?P<value>\d+[.]?\d*(?:[eE][+-]?\d+)?|\.\d+)
    \s*
    (?P<unit>
      # chemical concentration
      wt%|wt\.?\s*%|mol%|vol%|wt\.%
      | ppm|ppb|ppt
      # temperature
      | °C|°F|°K|\bK\b|\bC\b|\bF\b
      # pressure
      | GPa|MPa|kPa|bar|atm|Pa
      # age
      | Ma|Ga|ka
      # length
      | μm|nm|mm|cm|dm|m|km|mile|ft|inch
      # mass
      | mg|μg|g|kg|t|lb
      # volume
      | mL|μL|L|gal
      # time
      | s|ms|min|hour|hr|day|week|month|year|yr
      # angle
      | °|deg
      # percentage
      | %|‰
      # electrical
      | V|mV|A|mA|W|mW|Ω
      # frequency
      | Hz|kHz|MHz|GHz
      # energy
      | J|kJ|cal|kcal|eV|keV|MeV|GeV
    )
    (?![\w%°])
    """,
    re.VERBOSE | re.IGNORECASE,
)


# === Date (multiple formats) =====================================================
_DATE_RE = re.compile(
    r"""
    \b
    (?:
      # ISO 8601
      \d{4}-\d{2}-\d{2}
      # Year-Month
      | \d{4}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}
      # Month Day, Year
      | (?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}
      # Year alone in citation context
      | \b(19|20)\d{2}\b
    )
    \b
    """,
    re.VERBOSE,
)


# === Person name (Author, F.) =====================================================
# Conservative: "Smith, J." or "Smith J." or "Smith, J. A."
_PERSON_RE = re.compile(
    r"\b([A-Z][a-z]+(?:[A-Z][a-z]+)?),?\s+([A-Z]\.(?:\s*[A-Z]\.?)*)"
)


# === Citation (Author, Year) =====================================================
_CITATION_RE = re.compile(
    r"\(([A-Z][a-zA-Z]+(?:\s+(?:et al\.?|and|&)\s+[A-Z][a-zA-Z]+)?),?\s+"
    r"(?:19|20)\d{2}[a-z]?\)"
)


# === Sample ID =====================================================
# Common patterns: GR-01, S-1, JGB-15, KTB-92/3
_SAMPLE_RE = re.compile(r"\b([A-Z]{1,5}-?\d{1,4}(?:/\d+)?)\b")


# === DOI =====================================================
_DOI_RE = re.compile(
    r"\b(10\.\d{4,9}/[-._;()/:A-Z0-9]+)\b",
    re.IGNORECASE,
)


# === URL =====================================================
_URL_RE = re.compile(r"\b(https?://[^\s<>\"']+)\b")


def extract_quantities(text: str) -> list[Entity]:
    """Extract all (value, unit) pairs."""
    out = []
    for i, m in enumerate(_QUANTITY_RE.finditer(text)):
        value = m.group("value")
        unit = m.group("unit")
        out.append(
            Entity(
                id=f"qty_{i}",
                type="Quantity",
                value=m.group(0),
                normalized=f"{value} {unit}",
                confidence=0.95,
                metadata={"numeric_value": float(value), "unit": unit},
            )
        )
    return out


def extract_dates(text: str) -> list[Entity]:
    out = []
    for i, m in enumerate(_DATE_RE.finditer(text)):
        out.append(
            Entity(
                id=f"date_{i}",
                type="Date",
                value=m.group(0),
                confidence=0.85,
            )
        )
    return out


def extract_persons(text: str) -> list[Entity]:
    out = []
    for i, m in enumerate(_PERSON_RE.finditer(text)):
        full = f"{m.group(1)}, {m.group(2)}"
        out.append(
            Entity(
                id=f"person_{i}",
                type="Person",
                value=full,
                confidence=0.75,
            )
        )
    return out


def extract_citations(text: str) -> list[Entity]:
    out = []
    for i, m in enumerate(_CITATION_RE.finditer(text)):
        out.append(
            Entity(
                id=f"cite_{i}",
                type="Citation",
                value=m.group(0),
                confidence=0.85,
            )
        )
    return out


def extract_dois(text: str) -> list[Entity]:
    out = []
    for i, m in enumerate(_DOI_RE.finditer(text)):
        out.append(
            Entity(
                id=f"doi_{i}",
                type="DOI",
                value=m.group(1),
                confidence=1.0,
            )
        )
    return out


def extract_all(text: str) -> list[Entity]:
    """Run all extractors. Returns merged list."""
    out = []
    out.extend(extract_quantities(text))
    out.extend(extract_dates(text))
    out.extend(extract_persons(text))
    out.extend(extract_citations(text))
    out.extend(extract_dois(text))
    return out


__all__ = [
    "Entity",
    "extract_quantities",
    "extract_dates",
    "extract_persons",
    "extract_citations",
    "extract_dois",
    "extract_all",
]
