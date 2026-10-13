"""Unit tests for retry_actions: classify_failure + action_for.

Per testing.md §3.1: ≥90% strong assertions.
Run: pytest scripts/scientific_parser/tests/test_retry_actions.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.ir import VerificationResult
from scientific_parser.verification.retry_actions import (
    FailureClass,
    RetryAction,
    RetryActionType,
    action_for,
    classify_failure,
)

# === classify_failure: None / passed inputs ===================== #


def test_classify_failure_none_verification_returns_unknown():
    assert classify_failure(None) is FailureClass.UNKNOWN


def test_classify_failure_passed_verification_returns_unknown():
    """Caller misuse: classifier invoked on a passing result."""
    v = VerificationResult(passed=True, score=0.95, validator_id="formula_latex")
    assert classify_failure(v) is FailureClass.UNKNOWN


# === classify_failure: formula ================================= #


def test_classify_failure_formula_latex_validator_returns_formula_malformed():
    v = VerificationResult(
        passed=False,
        score=0.2,
        validator_id="formula_latex",
        details={"error": "unbalanced delimiters"},
    )
    assert classify_failure(v) is FailureClass.FORMULA_MALFORMED


# === classify_failure: table =================================== #


def test_classify_failure_table_structure_returns_table_merge():
    v = VerificationResult(
        passed=False,
        score=0.4,
        validator_id="table_structure",
        details={"merged_cells": 2},
    )
    assert classify_failure(v) is FailureClass.TABLE_MERGE


def test_classify_failure_table_teds_validator_returns_table_merge():
    v = VerificationResult(passed=False, score=0.3, validator_id="table_teds")
    assert classify_failure(v) is FailureClass.TABLE_MERGE


# === classify_failure: numeric ================================= #


def test_classify_failure_numeric_range_with_disagreement_returns_numeric_disagree():
    v = VerificationResult(
        passed=False,
        score=0.3,
        validator_id="numeric_range",
        details={"disagreement": True, "expected": 12.5, "got": 125.0},
    )
    assert classify_failure(v) is FailureClass.NUMERIC_DISAGREE


def test_classify_failure_numeric_consistency_returns_numeric_disagree():
    v = VerificationResult(passed=False, score=0.4, validator_id="numeric_consistency")
    assert classify_failure(v) is FailureClass.NUMERIC_DISAGREE


# === classify_failure: reading order =========================== #


def test_classify_failure_reading_order_returns_reading_order():
    v = VerificationResult(
        passed=False,
        score=0.3,
        validator_id="reading_order",
        details={"columns_interleaved": True},
    )
    assert classify_failure(v) is FailureClass.READING_ORDER


def test_classify_failure_column_order_returns_reading_order():
    v = VerificationResult(passed=False, score=0.4, validator_id="column_order")
    assert classify_failure(v) is FailureClass.READING_ORDER


# === classify_failure: unrecognized validator ================== #


def test_classify_failure_unknown_validator_low_score_returns_generic_low_conf():
    v = VerificationResult(passed=False, score=0.2, validator_id="some_new_validator")
    assert classify_failure(v) is FailureClass.GENERIC_LOW_CONF


def test_classify_failure_unknown_validator_high_score_returns_unknown():
    """Failed verification with score >= 0.5 from unrecognized validator —
    can't classify confidently. Return UNKNOWN (safer than guessing)."""
    v = VerificationResult(passed=False, score=0.6, validator_id="future_validator")
    assert classify_failure(v) is FailureClass.UNKNOWN


# === action_for: default mapping ============================== #


def test_action_for_formula_malformed_returns_specialist_model_action():
    action = action_for(FailureClass.FORMULA_MALFORMED)
    assert action.action is RetryActionType.SPECIALIST_MODEL
    assert action.params["specialist"] == "formula_latex"


def test_action_for_table_merge_returns_specialist_model_table_action():
    action = action_for(FailureClass.TABLE_MERGE)
    assert action.action is RetryActionType.SPECIALIST_MODEL
    assert action.params["specialist"] == "table_structure"


def test_action_for_numeric_disagree_returns_rerun_region_action():
    action = action_for(FailureClass.NUMERIC_DISAGREE)
    assert action.action is RetryActionType.RERUN_REGION
    assert action.params["validator"] == "numeric_range"


def test_action_for_reading_order_returns_column_reconstruct_action():
    action = action_for(FailureClass.READING_ORDER)
    assert action.action is RetryActionType.COLUMN_RECONSTRUCT


def test_action_for_generic_low_conf_returns_rerun_region_action():
    action = action_for(FailureClass.GENERIC_LOW_CONF)
    assert action.action is RetryActionType.RERUN_REGION


def test_action_for_unknown_returns_abstain_action():
    """Fail-fast: unclassifiable failure → abstain, no blind retry."""
    action = action_for(FailureClass.UNKNOWN)
    assert action.action is RetryActionType.ABSTAIN


# === action_for: context promotion ============================ #


def test_action_for_context_with_region_bbox_promotes_into_params():
    bbox = (100, 200, 500, 800)
    action = action_for(
        FailureClass.FORMULA_MALFORMED,
        context={"region_bbox": bbox, "page_num": 7},
    )
    assert action.params["region_bbox"] == bbox
    assert action.params["page_num"] == 7


def test_action_for_context_with_current_dpi_promotes_fallback_dpi():
    """For HIGHER_DPI/specialist actions, current_dpi in context becomes
    fallback_dpi in params (current + 150 step)."""
    action = action_for(
        FailureClass.FORMULA_MALFORMED,
        context={"current_dpi": 150},
    )
    assert action.params["fallback_dpi"] == 300  # 150 + 150


def test_action_for_context_with_invalid_current_dpi_falls_back_to_300():
    action = action_for(
        FailureClass.FORMULA_MALFORMED,
        context={"current_dpi": "not_a_number"},
    )
    assert action.params["fallback_dpi"] == 300


def test_action_for_empty_context_returns_default_params():
    action = action_for(FailureClass.GENERIC_LOW_CONF, context={})
    assert action.action is RetryActionType.RERUN_REGION
    # Default params for RERUN_REGION = empty dict
    assert action.params == {}


def test_action_for_none_context_returns_default_params():
    action = action_for(FailureClass.NUMERIC_DISAGREE, context=None)
    assert action.params["validator"] == "numeric_range"


# === RetryAction dataclass validation ======================== #


def test_retry_action_construct_valid_object_with_params():
    a = RetryAction(
        action=RetryActionType.ABSTAIN,
        params={"reason": "test"},
        rationale="testing",
    )
    assert a.action is RetryActionType.ABSTAIN
    assert a.params == {"reason": "test"}
    assert a.rationale == "testing"


def test_retry_action_non_dict_params_raises_type_error():
    with pytest.raises(TypeError, match="dict"):
        RetryAction(action=RetryActionType.ABSTAIN, params=["not", "a", "dict"])  # type: ignore[arg-type]


def test_retry_action_default_params_is_empty_dict():
    a = RetryAction(action=RetryActionType.RERUN_REGION)
    assert a.params == {}
    assert a.rationale == ""


# === End-to-end: classify → action ========================== #


def test_end_to_end_formula_failure_classifies_and_actions_correctly():
    v = VerificationResult(
        passed=False,
        score=0.1,
        validator_id="formula_latex",
        details={"error": "pykatex render failed"},
    )
    fc = classify_failure(v)
    action = action_for(fc, context={"region_bbox": (10, 20, 100, 200), "current_dpi": 150})
    assert fc is FailureClass.FORMULA_MALFORMED
    assert action.action is RetryActionType.SPECIALIST_MODEL
    assert action.params["specialist"] == "formula_latex"
    assert action.params["fallback_dpi"] == 300
    assert action.params["region_bbox"] == (10, 20, 100, 200)


def test_end_to_end_unknown_failure_abstains_without_burning_compute():
    """Critical contract: unclassifiable failure → ABSTAIN. Never blindly rerun."""
    v = None
    fc = classify_failure(v)
    action = action_for(fc)
    assert fc is FailureClass.UNKNOWN
    assert action.action is RetryActionType.ABSTAIN
    assert action.params == {}
