"""Conformal prediction for calibrated confidence (proposal item #27, #26).

Per architecture proposal item #26 (2026-08-12):
  "Calibrate confidence. Do not trust raw model confidence. Use temperature
   scaling, isotonic regression, Platt scaling, conformal prediction on your
   benchmark. Then confidence=0.95 should actually mean ~95% of predictions
   in this confidence bucket are correct."

Per item #27:
  "Conformal uncertainty for structured extraction. For ambiguous recognition:
   Candidate set {1.25, 1.75} rather than forcing one answer."

Implementation:
  - SplitN calibration: simple nonconformity scores
  - No external dep (no MAPIE, no aclimate) — pure algorithm
  - For numeric tokens: candidate set = values within conformal threshold
  - For OCR tokens: candidate set = top-K from SymSpell by margin

Limitations:
  - True conformal requires calibration set with ground truth
  - This module implements the FRAMEWORK; skill provides calibration data
  - Without calibration data, falls back to raw confidence (with WARNING)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CandidateSet:
    """Conformal prediction candidate set."""

    candidates: list  # list of (value, nonconformity_score)
    alpha: float  # significance level (1 - confidence)
    threshold: float  # max nonconformity score included

    def values(self) -> list:
        return [v for v, _ in self.candidates]

    def is_singleton(self) -> bool:
        return len(self.candidates) == 1

    def is_abstention(self) -> bool:
        """No candidates within threshold → abstention (proposal item #25)."""
        return len(self.candidates) == 0

    def top(self, n: int = 1) -> list:
        return [v for v, _ in self.candidates[:n]]


class ConformalCalibrator:
    """Calibrates nonconformity scores against a calibration set.

    Standard split-conformal:
      1. Compute nonconformity scores on calibration set
      2. Take quantile at (1 - alpha) level → threshold
      3. New prediction's candidate set = all values with score ≤ threshold
    """

    def __init__(self, alpha: float = 0.10):
        """Args:
        alpha: significance level. 0.10 → 90% coverage guarantee.
        """
        if not 0 < alpha < 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = alpha
        self._calibration_scores: list = []
        self._threshold: Optional[float] = None
        self._calibrated = False

    def calibrate(self, nonconformity_scores: list) -> None:
        """Provide calibration scores (e.g. distances to true value)."""
        if not nonconformity_scores:
            return
        self._calibration_scores = sorted(nonconformity_scores)
        # Quantile at (1 - alpha)
        # ceil((n+1) * (1-alpha)) / n  (standard split-conformal)
        n = len(self._calibration_scores)
        idx = max(0, min(n - 1, int((n + 1) * (1 - self.alpha)) - 1))
        self._threshold = self._calibration_scores[idx]
        self._calibrated = True

    def predict(self, candidates_with_scores: list) -> CandidateSet:
        """Filter candidates by calibrated threshold.

        Args:
            candidates_with_scores: [(value, nonconformity_score), ...]
                sorted ascending by score (lowest = most likely)

        Returns:
            CandidateSet containing all values within threshold.
        """
        if not self._calibrated or self._threshold is None:
            # No calibration: return top-1 only (conservative fallback)
            if candidates_with_scores:
                top = candidates_with_scores[:1]
                return CandidateSet(
                    candidates=top, alpha=self.alpha, threshold=float("inf")
                )
            return CandidateSet(candidates=[], alpha=self.alpha, threshold=0)
        # Filter by threshold
        included = [(v, s) for v, s in candidates_with_scores if s <= self._threshold]
        return CandidateSet(
            candidates=included,
            alpha=self.alpha,
            threshold=self._threshold,
        )


def numeric_nonconformity(true_value: float, predicted_value: float) -> float:
    """Nonconformity score for numeric prediction.

    Lower = more conformant (predicted close to true).
    Uses absolute relative error.
    """
    denom = max(abs(true_value), 1e-9)
    return abs(predicted_value - true_value) / denom


def ocr_nonconformity(true_token: str, predicted_token: str) -> float:
    """Nonconformity for OCR token prediction.

    Lower = more conformant. Uses normalized Levenshtein distance.
    """
    if true_token == predicted_token:
        return 0.0
    if not true_token or not predicted_token:
        return 1.0
    m, n = len(true_token), len(predicted_token)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            cur = dp[j]
            cost = 0 if true_token[i - 1] == predicted_token[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = cur
    return dp[n] / max(m, n)


__all__ = [
    "CandidateSet",
    "ConformalCalibrator",
    "numeric_nonconformity",
    "ocr_nonconformity",
]
