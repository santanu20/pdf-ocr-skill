"""Consensus — multi-model agreement verification (wraps model_disagreement)."""

from __future__ import annotations
import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")

from model_disagreement import (
    DisagreementFinding,
    DisagreementReport,
    compare_texts,
)


def consensus_value(*candidates) -> tuple:
    """Pick consensus value from multiple candidates.

    Args:
        *candidates: 2+ text outputs from different backends

    Returns:
        (consensus_text, confidence, n_agreed)
    """
    if not candidates:
        return ("", 0.0, 0)
    if len(candidates) == 1:
        return (candidates[0], 0.5, 1)
    # All-equal case
    if all(c == candidates[0] for c in candidates):
        return (candidates[0], 1.0, len(candidates))
    # Pairwise compare; pick candidate with highest agreement
    best = candidates[0]
    best_score = -1
    for c in candidates:
        scores = []
        for other in candidates:
            if other is c:
                continue
            r = compare_texts(c, other)
            # Higher = better (less disagreement)
            scores.append(1.0 - r.max_distance)
        avg = sum(scores) / max(len(scores), 1)
        if avg > best_score:
            best_score = avg
            best = c
    n_agreed = sum(1 for c in candidates if c == best)
    return (best, best_score, n_agreed)


__all__ = [
    "consensus_value",
    "DisagreementFinding",
    "DisagreementReport",
    "compare_texts",
]
