"""Typed validators — per-content-type structural validators.

Each validator returns a :class:`~scientific_parser.core.ir.VerificationResult`,
plugging into :class:`~scientific_parser.core.evidence.EvidenceResolver` via
the ``Evidence.verification`` field.

Modules:
    - ``table_structure``: row/column consistency for HTML/Markdown tables.
    - ``formula_latex``: LaTeX parse + render validity.
    - ``numeric_range``: numeric value/unit plausibility.
"""

from __future__ import annotations

from .chemistry import validate_formula
from .formula_latex import validate_latex
from .geology import (
    validate_oxide_total,
    validate_ree_sequence_typed,
    validate_upb_age,
)
from .numeric_range import validate_numeric
from .table_structure import validate_table

__all__ = [
    "validate_table",
    "validate_latex",
    "validate_numeric",
    "validate_formula",
    "validate_oxide_total",
    "validate_upb_age",
    "validate_ree_sequence_typed",
]
