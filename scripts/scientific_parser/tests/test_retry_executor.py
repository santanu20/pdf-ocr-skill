"""Unit tests for the retry executor (reviewer #22): execute_action.

Covers the pure / deterministic paths of execute_action (ABSTAIN,
missing-context error handling) and the _crop_with_padding helper against a
synthetic image. The I/O paths (HIGHER_DPI page render, RERUN_REGION OCR)
require a PDF + model and are exercised end-to-end elsewhere.

Run: pytest scripts/scientific_parser/tests/test_retry_executor.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.verification.retry_actions import (
    FailureClass,
    RetryAction,
    RetryActionType,
    action_for,
)
from scientific_parser.verification.retry_executor import (
    RetryOutcome,
    _crop_with_padding,
    execute_action,
)

# === ABSTAIN ===================================================== #


def test_abstain_succeeds_with_no_artifact():
    a = action_for(FailureClass.UNKNOWN)  # -> ABSTAIN
    out = execute_action(a, context={})
    assert isinstance(out, RetryOutcome)
    assert out.success is True
    assert out.action_type is RetryActionType.ABSTAIN
    assert out.output_path is None
    assert "abstain" in out.message.lower()


# === TIGHTER_CROP: missing context -> graceful failure =========== #


def test_tighter_crop_missing_context_fails_gracefully():
    a = RetryAction(action=RetryActionType.TIGHTER_CROP, params={})
    out = execute_action(a, context={})  # no img_path / region_bbox
    assert out.success is False
    assert out.action_type is RetryActionType.TIGHTER_CROP
    assert "missing" in out.message.lower() or "img_path" in out.message


def test_tighter_crop_missing_bbox_fails_gracefully():
    a = RetryAction(action=RetryActionType.TIGHTER_CROP, params={})
    out = execute_action(a, context={"img_path": "/tmp/x.png"})  # no bbox
    assert out.success is False


# === _crop_with_padding on a synthetic image ===================== #


def test_crop_with_padding_synthetic_image(tmp_path):
    try:
        from PIL import Image
    except ImportError:
        return  # Pillow optional in some envs
    img = tmp_path / "page.png"
    Image.new("RGB", (200, 300), (255, 255, 255)).save(img)
    out = _crop_with_padding(str(img), (50.0, 50.0, 150.0, 200.0), pad_px=2)
    assert out is not None
    assert Path(out).exists()


def test_crop_with_padding_bad_bbox_returns_none(tmp_path):
    from PIL import Image

    img = tmp_path / "page.png"
    Image.new("RGB", (200, 300), (255, 255, 255)).save(img)
    # inverted bbox (x1 < x0)
    assert _crop_with_padding(str(img), (150.0, 200.0, 50.0, 50.0), pad_px=2) is None


# === execute_action never raises ================================= #


def test_execute_action_never_raises_on_bad_input():
    a = RetryAction(action=RetryActionType.HIGHER_DPI, params={})
    out = execute_action(a, context={})  # missing pdf_path etc.
    assert isinstance(out, RetryOutcome)
    # Either it failed gracefully or succeeded; never raised.


# === full pure chain: classify -> action -> execute (ABSTAIN) ==== #


def test_full_chain_unknown_abstains():
    from scientific_parser.verification.retry_actions import classify_failure

    fc = classify_failure(None)  # None verification -> UNKNOWN
    assert fc == FailureClass.UNKNOWN
    a = action_for(fc)
    out = execute_action(a, context={})
    assert out.action_type is RetryActionType.ABSTAIN
