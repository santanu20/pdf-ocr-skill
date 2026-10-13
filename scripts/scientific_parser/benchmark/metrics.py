"""Metrics — character error rate, F1, equation accuracy, table TEDS."""

from __future__ import annotations
import re
from typing import Optional


def char_error_rate(reference: str, hypothesis: str) -> float:
    """CER = edit_distance / len(reference). Lower is better."""
    if not reference:
        return 0.0 if not hypothesis else 1.0
    m, n = len(reference), len(hypothesis)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / m


def f1_token(reference: str, hypothesis: str) -> float:
    """Token-level F1 (whitespace split)."""
    ref_tokens = set(reference.lower().split())
    hyp_tokens = set(hypothesis.lower().split())
    if not ref_tokens and not hyp_tokens:
        return 1.0
    if not ref_tokens or not hyp_tokens:
        return 0.0
    tp = len(ref_tokens & hyp_tokens)
    fp = len(hyp_tokens - ref_tokens)
    fn = len(ref_tokens - hyp_tokens)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-9)


def equation_accuracy(reference_latex: str, hypothesis_latex: str) -> float:
    """Equation accuracy: 1.0 if normalized LaTeX matches, else 0.0.

    Normalization: strip whitespace, lowercase, remove braces around single chars.
    """

    def normalize(s):
        s = re.sub(r"\s+", "", s)
        s = re.sub(r"_\{([A-Za-z0-9])\}", r"_\1", s)
        s = re.sub(r"\^\{([A-Za-z0-9])\}", r"^\1", s)
        return s.lower()

    return 1.0 if normalize(reference_latex) == normalize(hypothesis_latex) else 0.0


def numeric_accuracy(
    reference_value, hypothesis_value, relative_tolerance: float = 0.01
) -> float:
    """1.0 if |ref - hyp| / max(|ref|, 1e-9) <= tolerance, else 0.0."""
    try:
        r = float(reference_value)
        h = float(hypothesis_value)
    except (ValueError, TypeError):
        return 0.0
    denom = max(abs(r), 1e-9)
    return 1.0 if abs(r - h) / denom <= relative_tolerance else 0.0


__all__ = [
    "char_error_rate",
    "f1_token",
    "equation_accuracy",
    "numeric_accuracy",
]
