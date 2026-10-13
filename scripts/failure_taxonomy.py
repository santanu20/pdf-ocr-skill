"""Failure taxonomy — 20-category labels for diagnostic dashboards.

Per architecture proposal item #39 (2026-08-12):
  "Build a failure taxonomy. Every failure should get a label.
   Then your dashboard can say:
     Current accuracy: 96.8%
     But:
       numeric errors       1.7%
       table errors         2.9%
       formula errors       3.4%
       reading-order errors 0.8%
   That's infinitely more useful than WER=2.1%."

This module is a constants-only library — no logic. Every component in the
pipeline imports the same labels so failures are aggregatable across runs.

Usage in other modules:
    from failure_taxonomy import Failure
    log.warning(Failure.OCR_NUMBER_DECIMAL_LOST, extra={...})

The skill's summary.json gets a per-document failure breakdown:
    {
      "failures": {
        "OCR_NUMBER_DECIMAL_LOST": 3,
        "OCR_FORMULA_KATEX_FAIL": 5,
        ...
      }
    }
"""

from __future__ import annotations

# Failure categories grouped by axis. Values are short uppercase strings
# suitable for JSON keys + dashboard labels.

# === OCR perception failures ===
OCR_CHAR = "OCR_CHAR"  # character-level error
OCR_NUMBER = "OCR_NUMBER"  # generic numeric (use specific sub below)
OCR_NUMBER_DECIMAL_LOST = "OCR_NUMBER_DECIMAL_LOST"
OCR_NUMBER_DECIMAL_ADDED = "OCR_NUMBER_DECIMAL_ADDED"
OCR_NUMBER_EXPONENT_FLIP = "OCR_NUMBER_EXPONENT_FLIP"
OCR_NUMBER_RANGE_MALFORMED = "OCR_NUMBER_RANGE_MALFORMED"
OCR_NUMBER_PLUSMINUS_BAD = "OCR_NUMBER_PLUSMINUS_BAD"
OCR_NUMBER_PERCENT_LOST = "OCR_NUMBER_PERCENT_LOST"
OCR_NUMBER_SCIENTIFIC_BAD = "OCR_NUMBER_SCIENTIFIC_BAD"
OCR_NUMBER_OUTLIER = "OCR_NUMBER_OUTLIER"
OCR_SYMBOL = "OCR_SYMBOL"  # Greek letter, math symbol, etc.
OCR_FORMULA = "OCR_FORMULA"  # generic formula (use sub below)
OCR_FORMULA_KATEX_FAIL = "OCR_FORMULA_KATEX_FAIL"
OCR_FORMULA_CMD_CONFUSION = "OCR_FORMULA_CMD_CONFUSION"
OCR_FORMULA_TRUNCATED = "OCR_FORMULA_TRUNCATED"
OCR_FORMULA_NOT_FOUND = "OCR_FORMULA_NOT_FOUND"
OCR_REPETITION_LOOP = "OCR_REPETITION_LOOP"
OCR_CJK_HALLUCINATION = "OCR_CJK_HALLUCINATION"

# === Layout / structure failures ===
LAYOUT = "LAYOUT"
READING_ORDER = "READING_ORDER"
TABLE_STRUCTURE = "TABLE_STRUCTURE"
TABLE_VALUE = "TABLE_VALUE"
TABLE_MERGED_CELL = "TABLE_MERGED_CELL"
TABLE_CROSS_PAGE = "TABLE_CROSS_PAGE"

# === Figure / chart failures ===
FIGURE = "FIGURE"
CHART = "CHART"
CHART_AXIS = "CHART_AXIS"
CHART_LEGEND = "CHART_LEGEND"

# === Equation failures ===
EQUATION = "EQUATION"
EQUATION_AST_INVALID = "EQUATION_AST_INVALID"
EQUATION_DIMENSION_MISMATCH = "EQUATION_DIMENSION_MISMATCH"

# === Semantic / domain failures ===
UNIT = "UNIT"
UNIT_UNKNOWN = "UNIT_UNKNOWN"
UNIT_OUT_OF_RANGE = "UNIT_OUT_OF_RANGE"
ENTITY = "ENTITY"
ENTITY_NOT_IN_LEXICON = "ENTITY_NOT_IN_LEXICON"
CITATION = "CITATION"
CITATION_SPLIT = "CITATION_SPLIT"
CITATION_MALFORMED = "CITATION_MALFORMED"

# === PDF encoding failures ===
PDF_ENCODING = "PDF_ENCODING"
PDF_FONT_TYPE3 = "PDF_FONT_TYPE3"
PDF_FONT_EMBED_MISSING = "PDF_FONT_EMBED_MISSING"
PDF_INVISIBLE_TEXT = "PDF_INVISIBLE_TEXT"

# === Scan / image failures ===
SCAN = "SCAN"
SCAN_ROTATION = "SCAN_ROTATION"
SCAN_SKEW = "SCAN_SKEW"
SCAN_DPI_INSUFFICIENT = "SCAN_DPI_INSUFFICIENT"
SCAN_BLUR = "SCAN_BLUR"

# === Language failures ===
LANGUAGE = "LANGUAGE"
LANGUAGE_UNDETECTED = "LANGUAGE_UNDETECTED"
LANGUAGE_MIXED = "LANGUAGE_MIXED"

# === Domain / pipeline failures ===
DOMAIN = "DOMAIN"
DOMAIN_UNASSIGNED = "DOMAIN_UNASSIGNED"
TRIAGE_REVIEW = "TRIAGE_REVIEW"
TRIAGE_REJECT = "TRIAGE_REJECT"
MODEL_DISAGREEMENT = "MODEL_DISAGREEMENT"


# === Aggregator (for dashboards) ========================================= #

# Group labels by axis for rollup summaries
AXES = {
    "OCR": [
        OCR_CHAR,
        OCR_NUMBER,
        OCR_NUMBER_DECIMAL_LOST,
        OCR_NUMBER_DECIMAL_ADDED,
        OCR_NUMBER_EXPONENT_FLIP,
        OCR_NUMBER_RANGE_MALFORMED,
        OCR_NUMBER_PLUSMINUS_BAD,
        OCR_NUMBER_PERCENT_LOST,
        OCR_NUMBER_SCIENTIFIC_BAD,
        OCR_NUMBER_OUTLIER,
        OCR_SYMBOL,
        OCR_FORMULA,
        OCR_FORMULA_KATEX_FAIL,
        OCR_FORMULA_CMD_CONFUSION,
        OCR_FORMULA_TRUNCATED,
        OCR_FORMULA_NOT_FOUND,
        OCR_REPETITION_LOOP,
        OCR_CJK_HALLUCINATION,
    ],
    "STRUCTURE": [LAYOUT, READING_ORDER],
    "TABLE": [TABLE_STRUCTURE, TABLE_VALUE, TABLE_MERGED_CELL, TABLE_CROSS_PAGE],
    "FIGURE": [FIGURE, CHART, CHART_AXIS, CHART_LEGEND],
    "EQUATION": [EQUATION, EQUATION_AST_INVALID, EQUATION_DIMENSION_MISMATCH],
    "SEMANTIC": [
        UNIT,
        UNIT_UNKNOWN,
        UNIT_OUT_OF_RANGE,
        ENTITY,
        ENTITY_NOT_IN_LEXICON,
        CITATION,
        CITATION_SPLIT,
        CITATION_MALFORMED,
    ],
    "PDF": [PDF_ENCODING, PDF_FONT_TYPE3, PDF_FONT_EMBED_MISSING, PDF_INVISIBLE_TEXT],
    "SCAN": [SCAN, SCAN_ROTATION, SCAN_SKEW, SCAN_DPI_INSUFFICIENT, SCAN_BLUR],
    "LANGUAGE": [LANGUAGE, LANGUAGE_UNDETECTED, LANGUAGE_MIXED],
    "DOMAIN": [
        DOMAIN,
        DOMAIN_UNASSIGNED,
        TRIAGE_REVIEW,
        TRIAGE_REJECT,
        MODEL_DISAGREEMENT,
    ],
}


# === Weighted error metric (reviewer #30) ================================ #
# Scientific parsing is not well served by raw accuracy: one decimal error in
# a U-Pb age is far worse than a dropped comma. These severity weights align
# the error metric with real scientific use. Higher weight = more severe.
#
# Weight rationale (reviewer #30):
#   ordinary char error   x1   word/char-level noise
#   reading-order error   x4   breaks document flow
#   table structural      x6   corrupts a whole table
#   formula/equation      x8   breaks downstream math
#   numeric / unit        x10  falsifies a measurement
#   chemical formula      x10  wrong stoichiometry
#   geological measurement x20 highest-stakes numeric (overridden at call site
#                             when region/domain context is known)

AXIS_SEVERITY: dict[str, int] = {
    "OCR": 1,  # base: ordinary character error
    "STRUCTURE": 4,  # reading-order / layout
    "TABLE": 6,  # table structural
    "FIGURE": 3,
    "EQUATION": 8,  # formula
    "SEMANTIC": 5,  # entity / citation (unit labels override to 10)
    "PDF": 2,
    "SCAN": 2,
    "LANGUAGE": 2,
    "DOMAIN": 1,
}

# Label-level overrides for the highest-stakes failure types.
LABEL_SEVERITY: dict[str, int] = {
    # Numeric errors -> x10 (falsify measurements)
    OCR_NUMBER: 10,
    OCR_NUMBER_DECIMAL_LOST: 10,
    OCR_NUMBER_DECIMAL_ADDED: 10,
    OCR_NUMBER_EXPONENT_FLIP: 10,
    OCR_NUMBER_RANGE_MALFORMED: 10,
    OCR_NUMBER_PLUSMINUS_BAD: 10,
    OCR_NUMBER_PERCENT_LOST: 10,
    OCR_NUMBER_SCIENTIFIC_BAD: 10,
    OCR_NUMBER_OUTLIER: 10,
    TABLE_VALUE: 10,  # wrong numeric cell inside an otherwise-intact table
    # Unit errors -> x10
    UNIT: 10,
    UNIT_UNKNOWN: 10,
    UNIT_OUT_OF_RANGE: 10,
    # Formula / equation -> x8
    OCR_FORMULA: 8,
    OCR_FORMULA_KATEX_FAIL: 8,
    OCR_FORMULA_CMD_CONFUSION: 8,
    OCR_FORMULA_TRUNCATED: 8,
    OCR_FORMULA_NOT_FOUND: 8,
    EQUATION: 8,
    EQUATION_AST_INVALID: 8,
    EQUATION_DIMENSION_MISMATCH: 8,
    # Table structural -> x6
    TABLE_STRUCTURE: 6,
    TABLE_MERGED_CELL: 6,
    TABLE_CROSS_PAGE: 6,
    # Reading order / layout -> x4
    READING_ORDER: 4,
    LAYOUT: 4,
    # Citations -> x4
    CITATION: 4,
    CITATION_SPLIT: 4,
    CITATION_MALFORMED: 4,
    # Char-level -> x1-2
    OCR_CHAR: 1,
    OCR_SYMBOL: 2,
}

# Highest-stakes multiplier applied at call sites where the caller knows the
# failed value is a geological measurement (U-Pb/Ar-Ar age, etc.). The label
# alone cannot encode domain context, so this is a caller-supplied factor.
GEOLOGICAL_MEASUREMENT_FACTOR = 2  # x10 numeric base * 2 = x20 per #30


def severity_for(label: str) -> int:
    """Return the severity weight for a failure label (label > axis > 1)."""
    if label in LABEL_SEVERITY:
        return LABEL_SEVERITY[label]
    for axis, labels in AXES.items():
        if label in labels:
            return AXIS_SEVERITY.get(axis, 1)
    return 1


def weighted_error_score(failure_counts: dict, geo_labels: set | None = None) -> int:
    """Total weighted error for a per-document failure-count dict.

    A decimal error contributes 10x an ordinary char error, so the metric
    ranks scientific correctness rather than raw character accuracy.

    Args:
      failure_counts: ``{label: count}`` (e.g. from summary.json).
      geo_labels: optional set of labels whose failures are geological
        measurements; their weight is multiplied by
        :data:`GEOLOGICAL_MEASUREMENT_FACTOR` (numeric x10 -> x20).
    """
    geo_labels = geo_labels or set()
    total = 0
    for label, count in failure_counts.items():
        w = severity_for(label)
        if label in geo_labels:
            w *= GEOLOGICAL_MEASUREMENT_FACTOR
        total += int(count) * w
    return total


def weighted_error_breakdown(failure_counts: dict) -> dict:
    """Per-axis weighted-error breakdown for dashboards.

    Returns ``{axis: weighted_sum}`` so a dashboard can show where the
    weighted error concentrates (e.g. ``{"OCR": 80, "TABLE": 18, ...}``).
    """
    out = dict.fromkeys(AXES, 0)
    for axis, labels in AXES.items():
        out[axis] = sum(int(failure_counts.get(lbl, 0)) * severity_for(lbl) for lbl in labels)
    return out


def weighted_error_rate(failure_counts: dict, n_units: int, geo_labels: set | None = None) -> float:
    """Mean weighted error per unit (page or region).

    Normalizes :func:`weighted_error_score` by document size so it is
    comparable across documents of different length. ``n_units`` is the
    number of pages or regions evaluated.
    """
    if n_units <= 0:
        return 0.0
    return weighted_error_score(failure_counts, geo_labels) / n_units


# Helper: rollup a flat failure-count dict into axes
def rollup_by_axis(failure_counts: dict) -> dict:
    """Aggregate per-failure counts into per-axis totals.

    Example:
      input:  {"OCR_NUMBER_DECIMAL_LOST": 3, "OCR_FORMULA_KATEX_FAIL": 5}
      output: {"OCR": 8, "STRUCTURE": 0, "TABLE": 0, ...}
    """
    out = dict.fromkeys(AXES, 0)
    for axis, labels in AXES.items():
        out[axis] = sum(failure_counts.get(lbl, 0) for lbl in labels)
    return out


def top_failures(failure_counts: dict, n: int = 5) -> list:
    """Return top-N (label, count) tuples sorted by count desc."""
    items = [(k, v) for k, v in failure_counts.items() if v > 0]
    items.sort(key=lambda x: -x[1])
    return items[:n]


__all__ = [
    # expose all category constants
    *[name for name in dir() if name.isupper() and not name.startswith("_")],
    "AXES",
    "rollup_by_axis",
    "top_failures",
    "severity_for",
    "weighted_error_score",
    "weighted_error_breakdown",
    "weighted_error_rate",
]
