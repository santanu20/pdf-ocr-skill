"""Model disagreement routing — uncertainty signal via cross-model comparison.

Per architecture proposal item #24 (2026-08-12):
  "Use model disagreement as a routing signal. This is better than arbitrary
   thresholds.

   OCR-A = 1.25 ± 0.03
   OCR-B = 1.25 ± 0.03   → accept.

   OCR-A = 1.25
   OCR-B = 1.75          → retry.

   This gives you a natural: uncertainty → compute allocation loop."

Implementation:
  - Run 2+ backends on the same region (or use signal-based proxy when
    running only one backend)
  - Compare outputs at multiple levels: char edit-distance, numeric value
    difference, structural (table cells) similarity
  - If disagreement > threshold → flag for retry or REVIEW
  - If agreement → high-confidence ACCEPT

This module is logic-only; orchestration (calling multiple models) is
the caller's responsibility. This keeps the module testable in isolation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional
import logging

log = logging.getLogger(__name__)


@dataclass
class DisagreementFinding:
    """One dimension of disagreement between two backends."""

    dimension: str  # "char_editdist", "numeric_value", "structural"
    severity: str  # "low" / "medium" / "high"
    value_a: str
    value_b: str
    distance: float  # 0.0 (identical) to 1.0 (totally different)
    note: str = ""


@dataclass
class DisagreementReport:
    """Overall cross-model disagreement assessment."""

    findings: list = field(default_factory=list)
    max_distance: float = 0.0
    recommend_retry: bool = False
    recommend_triage: str = "ACCEPT"  # ACCEPT / REVIEW / REJECT
    summary: str = ""


# Thresholds (tunable, conservative defaults)
_CHAR_EDITDIST_REVIEW = 0.15  # 15% chars different
_CHAR_EDITDIST_REJECT = 0.40
_NUMERIC_REL_DIFF_REVIEW = 0.05  # 5% relative difference
_NUMERIC_REL_DIFF_REJECT = 0.50


def _char_editdist_normalized(a: str, b: str) -> float:
    """Levenshtein distance normalized to [0, 1].

    0.0 = identical, 1.0 = totally different.
    """
    if a == b:
        return 0.0
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    # Standard DP Levenshtein
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / max(m, n)


def _extract_numbers(text: str) -> list:
    """Extract all numeric tokens (return list of (value, position, raw))."""
    out = []
    for m in re.finditer(r"(?<![\w.])(\d+[.]?\d*(?:[eE][+-]?\d+)?)(?![\w.])", text):
        raw = m.group(1)
        try:
            val = float(raw)
            out.append((val, m.start(), raw))
        except ValueError as e:
            log.debug("_extract_numbers: ValueError degraded: %s", e)
    return out


def compare_texts(text_a: str, text_b: str) -> DisagreementReport:
    """Compare two OCR outputs. Returns multi-dimensional disagreement report.

    Dimensions:
      - char_editdist: overall text similarity (Levenshtein normalized)
      - numeric_value: max relative difference between matched numeric tokens
      - line_count: structural difference (line counts)

    Recommendation logic:
      - Any REJECT-level finding → REJECT
      - Any REVIEW-level finding → REVIEW
      - Otherwise ACCEPT
    """
    report = DisagreementReport()
    findings = []

    # 1. Char edit distance
    char_dist = _char_editdist_normalized(text_a, text_b)
    if char_dist >= _CHAR_EDITDIST_REJECT:
        severity = "high"
    elif char_dist >= _CHAR_EDITDIST_REVIEW:
        severity = "medium"
    elif char_dist > 0:
        severity = "low"
    else:
        severity = "low"
    findings.append(
        DisagreementFinding(
            dimension="char_editdist",
            severity=severity,
            value_a=text_a[:80] + ("…" if len(text_a) > 80 else ""),
            value_b=text_b[:80] + ("…" if len(text_b) > 80 else ""),
            distance=char_dist,
            note=f"normalized Levenshtein distance",
        )
    )

    # 2. Numeric comparison
    nums_a = _extract_numbers(text_a)
    nums_b = _extract_numbers(text_b)
    max_num_diff = 0.0
    if nums_a and nums_b and len(nums_a) == len(nums_b):
        for (va, _, ra), (vb, _, rb) in zip(nums_a, nums_b):
            denom = max(abs(va), abs(vb), 1e-9)
            rel = abs(va - vb) / denom
            if rel > max_num_diff:
                max_num_diff = rel
        if max_num_diff >= _NUMERIC_REL_DIFF_REJECT:
            sev = "high"
        elif max_num_diff >= _NUMERIC_REL_DIFF_REVIEW:
            sev = "medium"
        else:
            sev = "low"
        findings.append(
            DisagreementFinding(
                dimension="numeric_value",
                severity=sev,
                value_a=f"{len(nums_a)} numbers, max rel diff {max_num_diff:.3f}",
                value_b=f"{len(nums_b)} numbers",
                distance=max_num_diff,
                note="max relative difference between position-matched numbers",
            )
        )
    elif len(nums_a) != len(nums_b):
        findings.append(
            DisagreementFinding(
                dimension="numeric_value",
                severity="medium",
                value_a=f"{len(nums_a)} numbers",
                value_b=f"{len(nums_b)} numbers",
                distance=1.0,
                note="different count of numeric tokens — structural mismatch",
            )
        )

    # 3. Line count
    lines_a = text_a.count("\n") + 1
    lines_b = text_b.count("\n") + 1
    if lines_a != lines_b:
        line_dist = abs(lines_a - lines_b) / max(lines_a, lines_b)
        sev = "high" if line_dist > 0.5 else "medium" if line_dist > 0.2 else "low"
        findings.append(
            DisagreementFinding(
                dimension="line_count",
                severity=sev,
                value_a=f"{lines_a} lines",
                value_b=f"{lines_b} lines",
                distance=line_dist,
                note="line-count structural difference",
            )
        )

    # Aggregate
    report.findings = findings
    report.max_distance = max((f.distance for f in findings), default=0.0)
    if any(f.severity == "high" for f in findings):
        report.recommend_retry = True
        report.recommend_triage = "REJECT"
    elif any(f.severity == "medium" for f in findings):
        report.recommend_retry = True
        report.recommend_triage = "REVIEW"
    else:
        report.recommend_triage = "ACCEPT"
    report.summary = (
        f"max_distance={report.max_distance:.3f}, "
        f"findings={len(findings)}, "
        f"triage={report.recommend_triage}"
    )
    return report


__all__ = [
    "DisagreementFinding",
    "DisagreementReport",
    "compare_texts",
]
