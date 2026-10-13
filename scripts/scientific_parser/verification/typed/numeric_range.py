"""Numeric range validator — value/unit plausibility.

Per architecture proposal item #4 (2026-08-13): typed validator for numeric
quantities catches:
    - Unparseable value strings
    - Unknown units (e.g. "bogus_unit")
    - Values outside plausible physical range

Reuses :func:`scientific_parser.science.units.parse_quantity` and
:func:`scientific_parser.science.units.is_known_unit` — does NOT duplicate
unit registry logic.

For domain-specific ranges (wt%, Ma, °C, kbar, etc.) callers should pass a
``plausible_range`` dict mapping unit → ``(min, max)``.
"""

from __future__ import annotations

import re

from ...core.ir import VerificationResult

_VALUE_RE = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")


def _try_float(s: str) -> tuple[bool, float | None]:
    """Return (ok, value). Never raises."""
    try:
        return True, float(s)
    except (TypeError, ValueError):
        return False, None


def validate_numeric(
    value_str: str,
    expected_unit: str | None = None,
    plausible_range: tuple[float, float] | None = None,
) -> VerificationResult:
    """Validate a numeric value with optional unit + range check.

    Args:
        value_str: The numeric value as a string (e.g. ``"12.5"``, ``"1e6"``,
            ``"-3.14"``). May contain surrounding whitespace.
        expected_unit: Optional unit string (e.g. ``"wt%"``, ``"Ma"``,
            ``"degC"``). When provided, the validator checks
            ``is_known_unit(expected_unit)``.
        plausible_range: Optional ``(min, max)`` tuple. The parsed value must
            satisfy ``min <= value <= max``. When None, range check is skipped.

    Returns:
        :class:`VerificationResult` with ``validator_id="numeric_range"``.

    Scoring (each contributes 0.34/0.33/0.33):
        1. Value parses as float.
        2. Unit (if provided) is known.
        3. Value (if range provided) is within plausible range.
    """
    if value_str is None:
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="numeric_range",
            details={"error": "none_input"},
        )
    s = str(value_str).strip()
    if not s:
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="numeric_range",
            details={"error": "empty_input"},
        )

    errors = []

    # 1. Value parse
    value_ok, value = _try_float(s)
    if not value_ok:
        # Allow scientific notation with unicode minus / ± prefix
        cleaned = s.lstrip("±~ ").replace("−", "-")
        value_ok, value = _try_float(cleaned)
    if not value_ok:
        errors.append(f"unparseable_value: {s!r}")

    # 2. Unit (if provided)
    unit_ok = True
    if expected_unit is not None:
        try:
            from ...science.units import is_known_unit

            unit_ok = bool(is_known_unit(expected_unit))
            if not unit_ok:
                errors.append(f"unknown_unit: {expected_unit!r}")
        except Exception as e:
            unit_ok = False
            errors.append(f"unit_lookup_failed: {type(e).__name__}: {e}")

    # 3. Range (if provided + value parsed)
    range_ok = True
    if plausible_range is not None and value_ok and value is not None:
        try:
            lo, hi = float(plausible_range[0]), float(plausible_range[1])
            range_ok = lo <= value <= hi
            if not range_ok:
                errors.append(f"out_of_range: value={value} range=[{lo}, {hi}]")
        except (TypeError, ValueError, IndexError) as e:
            range_ok = False
            errors.append(f"range_check_invalid: {type(e).__name__}: {e}")

    score = (
        0.34 * (1.0 if value_ok else 0.0)
        + 0.33 * (1.0 if unit_ok else 0.0)
        + 0.33 * (1.0 if range_ok else 0.0)
    )
    passed = value_ok and unit_ok and range_ok

    return VerificationResult(
        passed=passed,
        score=score,
        validator_id="numeric_range",
        details={
            "value_raw": s,
            "value_parsed": value,
            "unit": expected_unit,
            "plausible_range": list(plausible_range) if plausible_range else None,
            "errors": errors,
        },
    )


__all__ = ["validate_numeric"]
