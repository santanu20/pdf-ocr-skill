"""Confidence scoring — calibrated vs raw.

Per proposal item #26: don't trust raw model confidence. Use calibration
(temperature scaling, isotonic regression, Platt scaling, conformal).

This module wraps:
  - raw_confidence → calibrated_confidence via calibrators
  - per-evidence confidence aggregation across an evidence chain
  - document-level confidence summary
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


# Temperature scaling (simple post-hoc calibration)
@dataclass
class TemperatureScaler:
    """T(z) = z/T — soft-max temperature. T>1 = more uniform, T<1 = sharper."""

    temperature: float = 1.0

    def scale(self, raw_confidence: float) -> float:
        if self.temperature <= 0:
            return raw_confidence
        # Apply softmax-style temperature scaling to a single confidence value
        # (interpreted as one element of a 2-class softmax)
        import math

        z = raw_confidence / self.temperature
        z_neg = (1.0 - raw_confidence) / self.temperature
        # Numerically stable softmax for 2 classes
        m = max(z, z_neg)
        ez = math.exp(z - m)
        en = math.exp(z_neg - m)
        return ez / (ez + en)


# Isotonic regression (piecewise monotone)
@dataclass
class IsotonicMapper:
    """Pre-computed isotonic mapping: raw_conf → calibrated_conf."""

    breakpoints: list  # list of (raw, calibrated) sorted by raw ascending

    def map(self, raw_confidence: float) -> float:
        if not self.breakpoints:
            return raw_confidence
        # Find bracketing breakpoints
        for i in range(len(self.breakpoints) - 1):
            r0, c0 = self.breakpoints[i]
            r1, c1 = self.breakpoints[i + 1]
            if r0 <= raw_confidence <= r1:
                # Linear interpolation
                t = (raw_confidence - r0) / max(r1 - r0, 1e-9)
                return c0 + t * (c1 - c0)
        # Out of range — clamp to nearest
        if raw_confidence < self.breakpoints[0][0]:
            return self.breakpoints[0][1]
        return self.breakpoints[-1][1]


def aggregate_evidence_confidence(
    confidences: list, weights: Optional[list] = None
) -> float:
    """Aggregate confidence across multi-level evidence chain.

    Args:
        confidences: list of per-evidence confidence values [0,1]
        weights: optional weights per evidence (default: equal)

    Returns:
        Blended confidence in [0, 1].
    """
    if not confidences:
        return 0.0
    if weights is None:
        weights = [1.0] * len(confidences)
    total_w = sum(weights)
    if total_w == 0:
        return 0.0
    return sum(c * w for c, w in zip(confidences, weights)) / total_w


def document_confidence_summary(page_confidences: list) -> dict:
    """Summarize confidence across document pages.

    Returns dict with: min, max, mean, median, p25, p75, p95.
    """
    if not page_confidences:
        return {
            "min": 0,
            "max": 0,
            "mean": 0,
            "median": 0,
            "p25": 0,
            "p75": 0,
            "p95": 0,
            "n": 0,
        }
    sorted_c = sorted(page_confidences)
    n = len(sorted_c)

    def percentile(p):
        idx = max(0, min(n - 1, int(p * n)))
        return sorted_c[idx]

    return {
        "min": sorted_c[0],
        "max": sorted_c[-1],
        "mean": sum(sorted_c) / n,
        "median": percentile(0.50),
        "p25": percentile(0.25),
        "p75": percentile(0.75),
        "p95": percentile(0.95),
        "n": n,
    }


__all__ = [
    "TemperatureScaler",
    "IsotonicMapper",
    "aggregate_evidence_confidence",
    "document_confidence_summary",
]
