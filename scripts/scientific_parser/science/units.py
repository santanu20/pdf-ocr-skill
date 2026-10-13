"""Units subsystem — wraps unit_engine for package API."""

from __future__ import annotations
import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")
from unit_engine import (
    get_registry,
    parse_quantity,
    check_plausible,
    normalize_unit_text,
    is_known_unit,
)

__all__ = [
    "get_registry",
    "parse_quantity",
    "check_plausible",
    "normalize_unit_text",
    "is_known_unit",
]
