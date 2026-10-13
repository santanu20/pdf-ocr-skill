"""Table schema inference — column-type detection (proposal item #18).

Per architecture proposal item #18 (2026-08-12):
  "Table parsing should become schema inference. Don't just detect cells.
   Infer: column name, data type, unit, semantic role, missing-value
   convention, numeric precision.

   Example:
     SiO2 | wt%     → {column: 'SiO2', semantic_type: 'chemical_measurement',
                       unit: 'wt%'}"

Implementation:
  - Per-column type inference: numeric, text, identifier, date, unit-only
  - Header-row detection: first row OR row with 'unit' suffix
  - Unit extraction from header text
  - Missing-value conventions: 'n.d.', 'b.d.l.', '-', 'NA', '—'
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Missing-value conventions in scientific tables
_MISSING_VALUES = {
    "n.d.",
    "nd",
    "b.d.l.",
    "bdl",
    "n.a.",
    "na",
    "—",
    "-",
    "—",
    "n/d",
    "<",
    "below detection",
}

# Unit patterns in column headers
_UNIT_PATTERN = re.compile(
    r"\s*\((?P<unit>[^)]+)\)\s*$"  # "SiO2 (wt%)" → wt%
    r"|\s*\[(?P<unit2>[^\]]+)\]\s*$"  # "Fe [ppm]" → ppm
    r"|(?P<unit3>\b(?:wt%|wt\.%|mol%|vol%|ppm|ppb|ppt|Ma|Ga|ka|MPa|GPa|bar|atm)\b)\s*$"
)

# Semantic column name patterns
_SEMANTIC_PATTERNS = {
    "sample_id": re.compile(r"^(sample|specimen|id|name|label|no\.?|#)\s*$", re.I),
    "rock_type": re.compile(r"^(rock|lithology|type|formation|unit)\s*$", re.I),
    "location": re.compile(r"^(lat|long|latitude|longitude|location|site)\s*$", re.I),
    "age": re.compile(r"^(age|date|ma|ga)\s*$", re.I),
    "temperature": re.compile(r"^(t|temp|temperature)\s*$", re.I),
    "pressure": re.compile(r"^(p|press|pressure)\s*$", re.I),
    "chemical": re.compile(r"^([A-Z][a-z]?)(\d+[A-Z]?)*$"),  # SiO2, Fe2O3, MgO
    "isotope_ratio": re.compile(r"^(\d+/\d+|\w+/\w+|\w+\s*/\s*\w+)$"),  # 87Sr/86Sr
    "delta_notation": re.compile(r"^[δ∆].*", re.I),
    "epsilon_notation": re.compile(r"^[ε].*", re.I),
}


@dataclass
class ColumnSchema:
    """Inferred schema for one table column."""

    name: str
    cleaned_name: str  # name with unit stripped
    unit: Optional[str] = None
    data_type: str = "unknown"  # numeric|text|identifier|date|missing_only
    semantic_type: str = "unknown"  # sample_id|chemical|temperature|age|...
    missing_count: int = 0
    null_density: float = 0.0
    numeric_stats: dict = field(default_factory=dict)  # min/max/mean/std
    distinct_values: int = 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "cleaned_name": self.cleaned_name,
            "unit": self.unit,
            "data_type": self.data_type,
            "semantic_type": self.semantic_type,
            "missing_count": self.missing_count,
            "null_density": round(self.null_density, 3),
            "numeric_stats": {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in self.numeric_stats.items()
            },
            "distinct_values": self.distinct_values,
        }


def _extract_unit_from_header(header: str) -> tuple[str, Optional[str]]:
    """Split 'SiO2 (wt%)' → ('SiO2', 'wt%'). Returns (clean_name, unit)."""
    if not header:
        return header, None
    m = _UNIT_PATTERN.search(header)
    if not m:
        return header.strip(), None
    unit = m.group("unit") or m.group("unit2") or m.group("unit3")
    if not unit:
        return header.strip(), None
    clean = header[: m.start()].strip()
    return clean or header.strip(), unit.strip()


def _detect_semantic_type(name: str) -> str:
    """Classify column name into semantic role."""
    for sem_name, pat in _SEMANTIC_PATTERNS.items():
        if pat.match(name):
            return sem_name
    return "unknown"


def _detect_data_type(values: list) -> tuple[str, dict]:
    """Infer column data type from cell values. Returns (type, stats)."""
    non_missing = [
        v
        for v in values
        if v and str(v).strip().lower() not in _MISSISSING_VALUES_MISSING
    ]
    if not non_missing:
        return "missing_only", {}
    # Try numeric
    nums = []
    for v in non_missing:
        try:
            # Strip < > ≤ ≥ for numeric parse
            clean = re.sub(r"[<>≤≥]", "", str(v)).strip()
            nums.append(float(clean))
        except (ValueError, TypeError):
            nums = None
            break
    if nums is not None and len(nums) >= len(non_missing) * 0.7:
        stats = {
            "min": min(nums),
            "max": max(nums),
            "mean": sum(nums) / len(nums),
        }
        return "numeric", stats
    # Identifier (all uppercase or all short codes)
    if all(re.match(r"^[A-Z0-9\-_]+$", str(v)) for v in non_missing):
        return "identifier", {}
    return "text", {}


# Workaround for missing alias
_MISSISSING_VALUES_MISSING = _MISSING_VALUES


def infer_column_schema(name: str, values: list) -> ColumnSchema:
    """Infer schema for one column given header name + cell values."""
    clean_name, unit = _extract_unit_from_header(name)
    sem_type = _detect_semantic_type(clean_name)
    data_type, stats = _detect_data_type(values)
    missing_count = sum(
        1 for v in values if not v or str(v).strip().lower() in _MISSING_VALUES
    )
    distinct = len(set(str(v) for v in values if v))
    return ColumnSchema(
        name=name,
        cleaned_name=clean_name,
        unit=unit,
        data_type=data_type,
        semantic_type=sem_type,
        missing_count=missing_count,
        null_density=missing_count / max(len(values), 1),
        numeric_stats=stats,
        distinct_values=distinct,
    )


def infer_table_schema(rows: list, header_row_idx: int = 0) -> dict:
    """Infer schema for full table.

    Args:
        rows: list of rows (each row = list of cell strings)
        header_row_idx: which row is the header (0 = first)

    Returns:
        {
          "columns": [ColumnSchema dict, ...],
          "n_data_rows": int,
          "n_columns": int,
        }
    """
    if not rows or header_row_idx >= len(rows):
        return {"columns": [], "n_data_rows": 0, "n_columns": 0}
    headers = rows[header_row_idx]
    data_rows = rows[header_row_idx + 1 :]
    n_cols = len(headers)
    columns = []
    for col_idx, header in enumerate(headers):
        col_values = [r[col_idx] if col_idx < len(r) else "" for r in data_rows]
        col_schema = infer_column_schema(header, col_values)
        columns.append(col_schema.to_dict())
    return {
        "columns": columns,
        "n_data_rows": len(data_rows),
        "n_columns": n_cols,
    }


__all__ = [
    "ColumnSchema",
    "infer_column_schema",
    "infer_table_schema",
]
