"""NumericGuard — high-ROI subsystem for scientific OCR numeric correctness.

Scientific OCR failures in numbers are catastrophic:
  1.5 → 15, 0.05 → 0.5, 10⁻⁶ → 10⁶, 3.2 → 32

This module detects, validates, and flags numeric tokens in extracted text.
Per the architecture proposal (2026-08-12 P0.2), this is one of the highest-
ROI subsystems for scientific documents: a single wrong decimal can invalidate
a paper's reproduction.

Design:
  - Pure detection (regex-based) — no ML, no deps
  - Conservative: only FLAGS suspicious patterns, never silently rewrites
    numbers (would be silent corruption per §5 H2)
  - Output: warning comments in markdown + structured ledger entries
  - Cross-check hooks: takes optional unit context to verify value ranges
    (e.g. "5°C→5000°C" flagged when unit known to be °C)

Failure taxonomy (per P0.8):
  OCR_NUMBER_DECIMAL_LOST  — "1.5" → "15"
  OCR_NUMBER_DECIMAL_ADDED — "15" → "1.5"
  OCR_NUMBER_EXPONENT_FLIP — "10⁻⁶" → "10⁶"
  OCR_NUMBER_RANGE_MALFORMED — "1.5-3.0" → "1.5-3.0-" (trailing)
  OCR_NUMBER_PLUSMINUS_BAD — "4.2 ± 0.1" → "4.2 + 0.1" (lost ±)
  OCR_NUMBER_PERCENT_LOST — "5.5%" → "5.5"
  OCR_NUMBER_SCIENTIFIC_BAD — "1.5e-3" → "1.5e3" (sign flip)
  OCR_NUMBER_OUTLIER — value implausible given nearby unit/context

Integration point: called after OCR correction, before final output.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Failure category labels (per P0.8 failure taxonomy)
class NumericFailure:
    DECIMAL_LOST = "OCR_NUMBER_DECIMAL_LOST"
    DECIMAL_ADDED = "OCR_NUMBER_DECIMAL_ADDED"
    EXPONENT_FLIP = "OCR_NUMBER_EXPONENT_FLIP"
    RANGE_MALFORMED = "OCR_NUMBER_RANGE_MALFORMED"
    PLUSMINUS_BAD = "OCR_NUMBER_PLUSMINUS_BAD"
    PERCENT_LOST = "OCR_NUMBER_PERCENT_LOST"
    SCIENTIFIC_BAD = "OCR_NUMBER_SCIENTIFIC_BAD"
    OUTLIER = "OCR_NUMBER_OUTLIER"


# Plausibility heuristics per unit (kelvin/celsius/ppm/wt%/etc.)
# Source: common physical ranges in geo/chem/physics papers.
# Used ONLY to flag values outside plausible range — never to silently fix.
_UNIT_RANGES = {
    # Temperature
    "°c": (-273.15, 1e6),
    "°f": (-459.67, 1.8e6),
    "k": (0.0, 1e6),
    # Concentration
    "wt%": (0.0, 100.0),
    "wt.%": (0.0, 100.0),
    "mol%": (0.0, 100.0),
    "ppm": (0.0, 1e6),
    "ppb": (0.0, 1e9),
    "ppt": (0.0, 1e12),
    # Pressure
    "gpa": (0.0, 1e6),
    "mpa": (0.0, 1e9),
    "kpa": (0.0, 1e12),
    "bar": (0.0, 1e9),
    "atm": (0.0, 1e9),
    # Age (geochronology)
    "ma": (0.0, 4.54e3),
    "ga": (0.0, 13.8),  # earth age in Ma, universe in Ga
    "ka": (0.0, 4.54e6),
    # Length
    "μm": (0.0, 1e3),
    "mm": (0.0, 1e6),
    "cm": (0.0, 1e5),
    "m": (0.0, 1e3),
    "km": (0.0, 1e0),
    # Mass
    "mg": (0.0, 1e9),
    "g": (0.0, 1e6),
    "kg": (0.0, 1e3),
    "t": (0.0, 1e0),
}


# Numeric token patterns (ordered most-specific → least)
_NUMBER_RE = re.compile(
    r"""
    (?<![\w.])               # left boundary
    (?P<num>
        # scientific notation
        \d+[.]\d+\s*[eE][+-]?\d+
      | \d+\s*[eE][+-]?\d+
        # decimal
      | \d+[.]\d+
        # integer
      | \d+
    )
    # Optional exponent as Unicode superscript
    (?:\s*(?P<supexp>[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+))?
    (?P<suffix>
      \s*[±]\s*\d+[.]?\d*        # ± uncertainty
    | \s*[%‰]                     # percent / permille
    | \s*[–-]\s*\d+[.]?\d*        # range "1.5-3.0"
    )?
    (?![\w.])
    """,
    re.VERBOSE,
)

_SUPERSRIPT_MAP = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")


@dataclass
class NumericFinding:
    """One numeric-token finding from NumericGuard."""

    line: int
    column: int
    raw: str
    value: Optional[float]
    unit: Optional[str]
    failure: Optional[str]
    confidence: float
    note: str = ""


def _to_float(token: str) -> Optional[float]:
    """Parse numeric token (handles scientific notation, Unicode superscript)."""
    t = token.strip()
    # Translate Unicode superscript exponent: "10⁻⁶" → "10e-6"
    if any(c in _SUPERSRIPT_MAP for c in t):
        # find split point
        for i, c in enumerate(t):
            if c in "⁻⁰¹²³⁴⁵⁶⁷⁸⁹":
                base, exp = t[:i], t[i:].translate(_SUPERSRIPT_MAP)
                try:
                    return float(base) * (10 ** int(exp))
                except ValueError:
                    return None
    try:
        return float(t)
    except ValueError:
        return None


def _detect_unit_nearby(text: str, pos: int, window: int = 8) -> Optional[str]:
    """Find a unit token IMMEDIATELY after `pos` (preferred) or just before.

    Conservative: requires the unit to be the very next non-space token, OR
    directly adjacent. Avoids matching "t" inside "0.3" (false positive).
    """
    # Forward: skip whitespace, look for unit at start of remainder
    fwd = text[pos : pos + window].lstrip()
    # Sort by length DESC so longest-match-wins ("wt%" before "t")
    for unit in sorted(_UNIT_RANGES.keys(), key=len, reverse=True):
        if fwd.startswith(unit):
            return unit
    # Backward: only check 1-3 chars before pos (unit directly attached)
    if pos > 0:
        bwd = text[max(0, pos - 4) : pos]
        for unit in sorted(_UNIT_RANGES.keys(), key=len, reverse=True):
            if bwd.endswith(unit):
                return unit
    return None


def analyze_text(text: str, known_unit: Optional[str] = None) -> list[NumericFinding]:
    """Scan text for suspicious numeric patterns. Returns findings list.

    Conservative: returns ALL findings, even low-confidence ones. Caller
    decides what to do with them (annotate, drop, escalate).
    """
    findings: list[NumericFinding] = []
    lines = text.split("\n")
    for line_idx, line in enumerate(lines, start=1):
        for m in _NUMBER_RE.finditer(line):
            raw = m.group(0)
            num_part = m.group("num")
            sup_exp = m.group("supexp")
            suffix = m.group("suffix") or ""
            col = m.start()

            # Parse value
            value: Optional[float] = None
            try:
                if sup_exp:
                    base = float(num_part.replace(" ", ""))
                    exp = int(sup_exp.translate(_SUPERSRIPT_MAP))
                    value = base * (10**exp)
                else:
                    value = float(num_part.replace(" ", ""))
            except (ValueError, TypeError):
                value = None

            # Find nearby unit
            unit = known_unit or _detect_unit_nearby(line, m.end())

            failure: Optional[str] = None
            note = ""
            confidence = 0.5

            # === DETECTION RULES ===

            # Rule 1: suspicious "round" integers near decimal-context
            # (integer-only in a paragraph with other decimals)
            if (
                value is not None
                and "." not in num_part
                and not sup_exp
                and not suffix
                and 10 <= abs(value) <= 10000
                and unit is None
                # check if neighbors in line have decimals
                and ("." in line.replace(raw, "", 1))
                and re.search(r"\d+[.]\d+", line.replace(raw, "X" * len(raw), 1))
            ):
                failure = NumericFailure.DECIMAL_LOST
                note = (
                    f"integer '{raw}' in decimal-context line (possible lost decimal)"
                )
                confidence = 0.35

            # Rule 2: malformed range (trailing dash, double dash)
            if (
                re.search(r"\d\s*[-–]\s*\d\s*[-–]\s*$", raw)
                or raw.count("–") > 1
                or raw.count("-") > 1
            ):
                if suffix and re.search(r"\d\s*[-–]\s*$", suffix):
                    failure = NumericFailure.RANGE_MALFORMED
                    note = f"range '{raw}' has trailing dash"
                    confidence = 0.7

            # Rule 3: ± corrupted to + (lost ± glyph)
            # Only flag if the line context suggests measurement
            if (
                "+" in suffix
                and "±" not in line
                and any(
                    kw in line.lower()
                    for kw in (
                        "mean",
                        "average",
                        "sd",
                        "std",
                        "σ",
                        "error",
                        "uncertainty",
                    )
                )
            ):
                failure = NumericFailure.PLUSMINUS_BAD
                note = f"'+' in measurement-context (possible lost ±): '{raw}'"
                confidence = 0.6

            # Rule 4: exponent sign flip — value 1e+N with N large + scientific term
            if value is not None and abs(value) >= 1e6:
                if any(
                    kw in line.lower()
                    for kw in ("concentration", "trace", "isotope", "ppm", "ppb")
                ):
                    failure = NumericFailure.EXPONENT_FLIP
                    note = f"very large value {value:.3g} in trace-context (possible sign flip)"
                    confidence = 0.4

            # Rule 5: plausibility check against known unit range
            if value is not None and unit and unit in _UNIT_RANGES:
                lo, hi = _UNIT_RANGES[unit]
                if value < lo or value > hi:
                    failure = NumericFailure.OUTLIER
                    note = f"value {value:g} {unit} outside plausible range [{lo:g}, {hi:g}]"
                    confidence = 0.7

            findings.append(
                NumericFinding(
                    line=line_idx,
                    column=col,
                    raw=raw,
                    value=value,
                    unit=unit,
                    failure=failure,
                    confidence=confidence,
                    note=note,
                )
            )

    return findings


def annotate_text(
    text: str, min_confidence: float = 0.5
) -> tuple[str, list[NumericFinding]]:
    """Add inline HTML comments to text marking suspicious numerics.

    Returns (annotated_text, findings). Only findings with confidence >=
    min_confidence are annotated. Others are still returned for the ledger.
    """
    findings = analyze_text(text)
    flagged = [f for f in findings if f.failure and f.confidence >= min_confidence]
    if not flagged:
        return text, findings

    # Apply annotations bottom-up so line/column offsets stay valid
    out_lines = text.split("\n")
    for f in sorted(flagged, key=lambda x: (x.line, -x.column)):
        if 1 <= f.line <= len(out_lines):
            ln = out_lines[f.line - 1]
            tag = (
                f"<!-- NumericGuard: {f.failure} confidence={f.confidence:.2f} "
                f"— {f.note} -->"
            )
            out_lines[f.line - 1] = ln + "  " + tag
    return "\n".join(out_lines), findings


__all__ = [
    "NumericFailure",
    "NumericFinding",
    "analyze_text",
    "annotate_text",
]
