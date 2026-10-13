"""Abstention — wraps selective_prediction for verification namespace."""

from __future__ import annotations
import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")

from selective_prediction import (
    ACCEPT,
    REVIEW,
    REJECT,
    TriageResult,
    triage,
    annotate_with_triage,
)


def should_abstain(triage_result: TriageResult) -> bool:
    """True if pipeline should ABSTAIN (refuse to output)."""
    return triage_result.state == REJECT


__all__ = [
    "ACCEPT",
    "REVIEW",
    "REJECT",
    "TriageResult",
    "triage",
    "annotate_with_triage",
    "should_abstain",
]
