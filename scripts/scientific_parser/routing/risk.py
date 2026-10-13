"""Risk-aware routing: uncertainty x semantic impact (reviewer #24).

Per architecture proposal:
  "Use 'impact x uncertainty', not uncertainty alone.
   A low-confidence paragraph is usually low risk.
   A low-confidence geological age is high risk.
   risk = uncertainty x semantic_impact"

Compute is spent where the probability x consequence of error is high.
This drives selective redundancy (reviewer #6): a second model runs only
when ``risk`` exceeds a threshold, not on every page.

``semantic_impact`` maps a region/value type to a consequence weight in
[1, 5]: ordinary text = 1 (low cost of an error), a numeric measurement /
geological age / chemical formula = 5 (a wrong value falsifies the science).
``uncertainty`` is in [0, 1] (1 - confidence). ``risk`` is their product,
in [0, 5].
"""

from __future__ import annotations

from dataclasses import dataclass

# Semantic impact per region/value type (reviewer #24 table).
# Higher = an error here is more costly to the downstream scientist.
SEMANTIC_IMPACT: dict[str, int] = {
    "text": 1,
    "paragraph": 1,
    "heading": 1,
    "title": 1,
    "list": 1,
    "caption": 2,
    "figure": 2,
    "table_text": 3,
    "table": 3,
    "key_value": 5,
    "citation": 4,
    "doi": 4,
    "equation": 4,
    "formula": 5,
    "chemical_formula": 5,
    "numeric": 5,
    "measurement": 5,
    "geological_age": 5,
    "u_pb_age": 5,
    "ar_ar_age": 5,
}

DEFAULT_IMPACT = 2  # unknown region type — treat as moderate consequence

# Default thresholds for selective-redundancy decisions (reviewer #6).
# A second model / independent validator runs when risk >= ESCALATE_THRESHOLD.
ESCALATE_THRESHOLD = 1.5  # e.g. uncertainty 0.3 x impact 5 = 1.5
REVIEW_THRESHOLD = 0.75  # flag for human review (uncertainty 0.75 x text 1)


@dataclass(frozen=True)
class RiskAssessment:
    """Outcome of a risk evaluation for one region/value.

    Attributes:
      uncertainty: 1 - confidence, in [0, 1].
      impact: semantic consequence weight in [1, 5].
      risk: uncertainty * impact, in [0, 5].
      escalate: True iff a second model / independent validator should run
        (selective redundancy per reviewer #6).
      review: True iff the region should be flagged for human review.
    """

    uncertainty: float
    impact: int
    risk: float
    escalate: bool
    review: bool


def semantic_impact(region_type: str | None) -> int:
    """Return the consequence weight for a region/value type (default 2)."""
    return SEMANTIC_IMPACT.get((region_type or "").lower(), DEFAULT_IMPACT)


def risk(uncertainty: float, region_type: str) -> RiskAssessment:
    """Compute risk = uncertainty x semantic_impact.

    Args:
      uncertainty: 1 - confidence, clamped to [0, 1]. Higher = less certain.
      region_type: one of the keys in :data:`SEMANTIC_IMPACT` (e.g.
        ``"text"``, ``"table"``, ``"geological_age"``).

    Returns:
      :class:`RiskAssessment` with the risk score and escalation flags.
    """
    u = max(0.0, min(1.0, float(uncertainty)))
    impact = semantic_impact(region_type)
    r = u * impact
    return RiskAssessment(
        uncertainty=u,
        impact=impact,
        risk=r,
        escalate=r >= ESCALATE_THRESHOLD,
        review=r >= REVIEW_THRESHOLD,
    )


def should_escalate(uncertainty: float, region_type: str) -> bool:
    """Convenience: True iff selective redundancy (second model) is warranted."""
    return risk(uncertainty, region_type).escalate


__all__ = [
    "SEMANTIC_IMPACT",
    "DEFAULT_IMPACT",
    "ESCALATE_THRESHOLD",
    "REVIEW_THRESHOLD",
    "RiskAssessment",
    "semantic_impact",
    "risk",
    "should_escalate",
]
