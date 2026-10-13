"""Composable validators — domain-neutral checks (proposal items #15, #17).

NumericValidator: catches malformed numbers, range violations.
UnitValidator: validates unit strings against pint registry.

Domain packs add their own validators (e.g. MineralFormulaValidator in
geology pack) via DomainPack.validators().
"""

from __future__ import annotations

import re
from typing import Any, Optional
import logging

log = logging.getLogger(__name__)


class NumericValidator:
    """Validates numeric tokens in text — flags decimal corruption, sign flips."""

    name = "NumericValidator"

    def validate(self, value: Any, context: Optional[dict] = None) -> tuple[bool, str]:
        """If value is a string, scan for suspicious numeric patterns.

        Returns (ok, reason). ok=False means: flagged but not blocked
        (conservative — proposal item #25 selective prediction).
        """
        if not isinstance(value, str):
            return True, ""
        # Re-use numeric_guard if available
        try:
            import sys

            if "scripts" not in sys.path:
                sys.path.insert(0, "scripts")
            from numeric_guard import analyze_text

            findings = analyze_text(value)
            flagged = [f for f in findings if f.failure]
            if flagged:
                reasons = "; ".join(f"{f.failure}({f.raw})" for f in flagged[:3])
                return False, f"NumericGuard flagged: {reasons}"
        except ImportError as e:
            log.debug("validate: ImportError degraded: %s", e)
        return True, ""


class UnitValidator:
    """Validates unit strings against pint registry."""

    name = "UnitValidator"

    def validate(self, value: Any, context: Optional[dict] = None) -> tuple[bool, str]:
        """If value looks like a unit string, check pint understands it."""
        if not isinstance(value, str) or not value:
            return True, ""
        # Only test strings that look like units (short, mostly symbols)
        if len(value) > 10 or any(
            c.isupper() and c.isalpha() and i > 0 for i, c in enumerate(value)
        ):
            # likely a name, not a unit
            return True, ""
        try:
            import sys

            if "scripts" not in sys.path:
                sys.path.insert(0, "scripts")
            from unit_engine import is_known_unit

            if is_known_unit(value):
                return True, ""
            # Try as part of "1 {value}"
            return False, f"unknown unit '{value}'"
        except ImportError:
            return True, ""


class FormulaValidator:
    """Validates LaTeX formula via pykatex (proposal item #28)."""

    name = "FormulaValidator"

    def validate(self, value: Any, context: Optional[dict] = None) -> tuple[bool, str]:
        """If value contains $$...$$ or $...$ blocks, validate each via pykatex."""
        if not isinstance(value, str) or "$" not in value:
            return True, ""
        try:
            import pykatex
        except ImportError:
            return True, ""  # skip if pykatex missing
        # Find math blocks
        blocks = re.findall(r"\$\$(.+?)\$\$", value, re.DOTALL)
        blocks += re.findall(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", value)
        for b in blocks:
            try:
                pykatex.renderToString(b, strict="error")
            except Exception as e:
                return False, f"KaTeX parse error: {str(e)[:80]}"
        return True, ""


class CitationValidator:
    """Validates citation format — (Author, Year) or [N] style."""

    name = "CitationValidator"

    PATTERNS = [
        r"\([A-Z][a-zA-Z]+,?\s+\d{4}[a-z]?\)",  # (Smith, 2020)
        r"\([A-Z][a-zA-Z]+\s+(?:et al\.?|and)\s+.+?\d{4}\)",  # (Smith et al. 2020)
        r"\[\d{1,3}\]",  # [12]
        r"\[\d{1,3}(?:[-,]\s*\d{1,3})*\]",  # [12,15-17]
    ]

    def validate(self, value: Any, context: Optional[dict] = None) -> tuple[bool, str]:
        """If value claims to be a citation, check it matches a known pattern."""
        if not isinstance(value, str) or not value:
            return True, ""
        # Only check if value LOOKS like a citation attempt
        looks_like_cite = (value.startswith("(") and value.endswith(")")) or (
            value.startswith("[") and value.endswith("]")
        )
        if not looks_like_cite:
            return True, ""
        for pat in self.PATTERNS:
            if re.match(pat, value):
                return True, ""
        return False, f"citation format unrecognized: {value[:40]}"


__all__ = [
    "NumericValidator",
    "UnitValidator",
    "FormulaValidator",
    "CitationValidator",
]
