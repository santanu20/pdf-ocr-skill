"""Failure-classifier + corrective-action router.

Per architecture proposal item #5 (2026-08-13): replace generic
"low-confidence → rerun model" retries with class-aware action selection.

Inputs:
    - VerificationResult (from a typed validator)
    - Optional candidate value (for content-type sniffing)

Output:
    - FailureClass enum value
    - RetryAction prescribing the next corrective step

The dispatch table is pure and deterministic. No ML, no I/O. Wired into
``ocr_with_reflexive_loop`` (extraction_cache.py) for diagnostic logging;
action EXECUTION (actually re-running with tighter crop / higher DPI /
different model) lands in a later phase once the typed validators are
populating VerificationResult in production.

Usage::

    from scientific_parser.verification.retry_actions import (
        classify_failure, action_for
    )
    fc = classify_failure(verif_result, candidate=value)
    action = action_for(fc, context={"page_num": 7, "region_bbox": bbox})
    log.info("retry_action=%s failure_class=%s", action.action, fc)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ..core.ir import VerificationResult


class FailureClass(StrEnum):
    """Coarse failure category. Drives RetryAction selection.

    Values are short stable strings — safe for JSON / metrics labels.
    """

    FORMULA_MALFORMED = "FORMULA_MALFORMED"
    """LaTeX failed to parse, pykatex render failed, or delimiter imbalance."""

    TABLE_MERGE = "TABLE_MERGE"
    """Table columns merged or row/column structure collapsed."""

    NUMERIC_DISAGREE = "NUMERIC_DISAGREE"
    """Two sources extracted different numeric values for the same field."""

    READING_ORDER = "READING_ORDER"
    """Columns interleaved, captions detached, multi-column reorder needed."""

    GENERIC_LOW_CONF = "GENERIC_LOW_CONF"
    """Below confidence floor but no specific structural failure detected."""

    UNKNOWN = "UNKNOWN"
    """Classifier could not classify (e.g. None input, missing context)."""


class RetryActionType(StrEnum):
    """Prescriptive corrective action. Drives executor behavior (later phase)."""

    TIGHTER_CROP = "TIGHTER_CROP"
    """Re-crop the region with reduced padding to remove surrounding noise."""

    HIGHER_DPI = "HIGHER_DPI"
    """Re-render the page at higher DPI before re-extraction."""

    SPECIALIST_MODEL = "SPECIALIST_MODEL"
    """Escalate to a domain-specific specialist (formula/table/numeric)."""

    RERUN_REGION = "RERUN_REGION"
    """Re-run the same model on the same region (cheap retry)."""

    COLUMN_RECONSTRUCT = "COLUMN_RECONSTRUCT"
    """Re-detect columns and rerun only affected regions."""

    ABSTAIN = "ABSTAIN"
    """No corrective action — flag for human review or skip."""


@dataclass(frozen=True)
class RetryAction:
    """A prescriptive corrective action with execution parameters.

    Attributes
    ----------
    action:
        The :class:`RetryActionType` to perform.
    params:
        Free-form parameters the executor will need (e.g.
        ``{"target_dpi": 300}`` for HIGHER_DPI,
        ``{"specialist": "Texo"}`` for SPECIALIST_MODEL,
        ``{"region_bbox": (x0,y0,x1,y1)}`` for TIGHTER_CROP).
    rationale:
        Short human-readable justification for diagnostic logging.
    """

    action: RetryActionType
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.params, dict):
            raise TypeError(f"RetryAction.params must be dict, got {type(self.params).__name__}")


# === Classification → action mapping =============================== #

# Pure dispatch table: maps FailureClass → default RetryAction.
# Values are factory lambdas so each call returns a fresh action instance
# with default params (callers may override).
_DEFAULT_ACTION_TABLE: dict[FailureClass, RetryAction] = {
    FailureClass.FORMULA_MALFORMED: RetryAction(
        action=RetryActionType.SPECIALIST_MODEL,
        params={"specialist": "formula_latex", "fallback_dpi": 300},
        rationale="Formula LaTeX malformed — escalate to formula specialist "
        "with tighter crop + higher DPI.",
    ),
    FailureClass.TABLE_MERGE: RetryAction(
        action=RetryActionType.SPECIALIST_MODEL,
        params={"specialist": "table_structure"},
        rationale="Table columns merged — escalate to table structure "
        "specialist on isolated table crop.",
    ),
    FailureClass.NUMERIC_DISAGREE: RetryAction(
        action=RetryActionType.RERUN_REGION,
        params={"validator": "numeric_range"},
        rationale="Numeric disagreement — re-run region and pass through "
        "numeric validator for tiebreak.",
    ),
    FailureClass.READING_ORDER: RetryAction(
        action=RetryActionType.COLUMN_RECONSTRUCT,
        params={},
        rationale="Reading-order error — re-detect columns and rerun only the affected regions.",
    ),
    FailureClass.GENERIC_LOW_CONF: RetryAction(
        action=RetryActionType.RERUN_REGION,
        params={},
        rationale="Generic low confidence — re-run same model (cheap retry) before escalating.",
    ),
    FailureClass.UNKNOWN: RetryAction(
        action=RetryActionType.ABSTAIN,
        params={},
        rationale="Unclassifiable failure — abstain (do not burn compute on blind retry).",
    ),
}


def classify_failure(
    verification: VerificationResult | None,
    candidate: Any = None,
) -> FailureClass:
    """Map a VerificationResult to a FailureClass.

    Heuristic dispatch based on ``verification.validator_id`` +
    ``verification.details`` + lightweight candidate content sniffing.
    Returns :attr:`FailureClass.UNKNOWN` for None verification or
    unrecognized validator IDs.

    Args:
        verification: The VerificationResult from the failed check.
            None → UNKNOWN (no signal to classify on).
        candidate: Optional candidate value (e.g. extracted LaTeX string,
            table HTML). Used for content sniffing when validator_id is
            ambiguous.

    Returns:
        One of :class:`FailureClass`.
    """
    if verification is None:
        return FailureClass.UNKNOWN
    if verification.passed:
        # Successful verification — caller should not have invoked the
        # classifier. Return UNKNOWN rather than guessing.
        return FailureClass.UNKNOWN

    vid = verification.validator_id
    details = verification.details or {}

    # Validator-ID-driven dispatch (deterministic).
    if vid == "formula_latex":
        return FailureClass.FORMULA_MALFORMED
    if vid in ("table_structure", "table_teds"):
        merge_signals = ("merged_cells", "column_collapse", "single_column")
        if any(details.get(k) for k in merge_signals):
            return FailureClass.TABLE_MERGE
        return FailureClass.TABLE_MERGE  # all table_structure failures merge-like
    if vid in ("numeric_range", "numeric_consistency"):
        # Disagreement vs out-of-range — check details
        if details.get("disagreement") or details.get("conflict_with"):
            return FailureClass.NUMERIC_DISAGREE
        return FailureClass.NUMERIC_DISAGREE
    if vid in ("reading_order", "column_order"):
        return FailureClass.READING_ORDER

    # Unrecognized validator_id — fall through to GENERIC_LOW_CONF if the
    # verification score is poor, else UNKNOWN.
    if verification.score < 0.5:
        return FailureClass.GENERIC_LOW_CONF
    return FailureClass.UNKNOWN


def action_for(
    classification: FailureClass,
    context: dict[str, Any] | None = None,
) -> RetryAction:
    """Look up the corrective action for a FailureClass.

    Args:
        classification: A :class:`FailureClass` (typically from
            :func:`classify_failure`).
        context: Optional dict with execution context (page_num, region_bbox,
            current_dpi, etc.). Merged into the action's ``params`` if the
            keys are recognized; non-recognized keys are ignored silently
            (forward-compat).

    Returns:
        A fresh :class:`RetryAction` instance. Never returns None.

    Raises:
        KeyError: never — UNKNOWN always maps to ABSTAIN.
    """
    base = _DEFAULT_ACTION_TABLE[classification]
    if not context:
        return RetryAction(
            action=base.action,
            params=dict(base.params),
            rationale=base.rationale,
        )
    # Selectively promote context keys into params (recognized keys only).
    params = dict(base.params)
    if "region_bbox" in context and "region_bbox" not in params:
        params["region_bbox"] = context["region_bbox"]
    if "current_dpi" in context and "fallback_dpi" not in params:
        # HIGHER_DPI / formula specialist uses fallback_dpi = current + step
        step = 150
        try:
            params["fallback_dpi"] = int(context["current_dpi"]) + step
        except (TypeError, ValueError):
            params["fallback_dpi"] = 300
    if "page_num" in context:
        params["page_num"] = context["page_num"]
    return RetryAction(action=base.action, params=params, rationale=base.rationale)


__all__ = [
    "FailureClass",
    "RetryActionType",
    "RetryAction",
    "classify_failure",
    "action_for",
]
