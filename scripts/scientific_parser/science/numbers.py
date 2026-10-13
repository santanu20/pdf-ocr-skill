"""Numbers subsystem — wraps numeric_guard for package API."""

from __future__ import annotations
import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")
from numeric_guard import (
    NumericFailure,
    NumericFinding,
    analyze_text,
    annotate_text,
)

__all__ = ["NumericFailure", "NumericFinding", "analyze_text", "annotate_text"]
