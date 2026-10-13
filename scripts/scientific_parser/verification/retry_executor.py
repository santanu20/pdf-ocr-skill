"""Retry-action executor — actually perform corrective actions.

Companion to ``retry_actions.py`` (which classifies failures and prescribes
actions). This module EXECUTES those actions and reports the outcome.

Supported actions (per architecture proposal item #5, 2026-08-13):
    - TIGHTER_CROP: re-crop region with reduced padding
    - HIGHER_DPI: re-render page at higher DPI then re-crop
    - RERUN_REGION: re-call the same OCR model on the same input
    - ABSTAIN: no-op (return original input unchanged)

Closed loop since 2026-08-15 (was deferred):
    - SPECIALIST_MODEL: formula_latex → RapidLaTeX on the crop;
      table_structure → RapidTable on the crop. Needs img_path +
      region_bbox in context.
    - COLUMN_RECONSTRUCT: column-aware reading order (recursive XY-cut)
      over the page text layer. Needs pdf_path + page_num in context.

The executor NEVER raises — every action returns a :class:`RetryOutcome`
with ``success: bool`` and diagnostic info. Callers can chain outcomes
through the resolver via ``outcome.verification``.
"""

from __future__ import annotations

import logging
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any

from ..core.ir import VerificationResult
from .retry_actions import RetryAction, RetryActionType

log = logging.getLogger(__name__)


@dataclass
class RetryOutcome:
    """Result of executing a :class:`RetryAction`.

    Attributes
    ----------
    success:
        True iff the action executed without internal error AND produced
        a new artifact (cropped image, re-rendered page, re-OCR'd text).
    action_type:
        The :class:`RetryActionType` that was attempted.
    output_path:
        Path to the new artifact (cropped image / re-rendered page) when
        applicable. None for actions that don't produce a file.
    output_text:
        Re-extracted text from a RERUN_REGION. None for actions that
        don't produce text.
    verification:
        Optional post-action verification result. None if not run.
    details:
        Diagnostic dict (e.g. ``{"render_dpi": 300, "crop_pad_px": 5}``).
    message:
        Human-readable status message for logging.
    """

    success: bool
    action_type: RetryActionType
    output_path: str | None = None
    output_text: str | None = None
    verification: VerificationResult | None = None
    details: dict = field(default_factory=dict)
    message: str = ""


def _crop_with_padding(
    img_path: str,
    bbox_px: tuple[float, float, float, float],
    pad_px: int,
) -> str | None:
    """Crop image to bbox + ``pad_px`` margin. Returns new path or None."""
    try:
        from PIL import Image

        img = Image.open(img_path)
        x0, y0, x1, y1 = bbox_px
        x0 = max(0, int(x0) - pad_px)
        y0 = max(0, int(y0) - pad_px)
        x1 = min(img.width, int(x1) + pad_px)
        y1 = min(img.height, int(y1) + pad_px)
        if x1 <= x0 or y1 <= y0:
            return None
        cropped = img.crop((x0, y0, x1, y1))
        suffix = f".tight_{x0}_{y0}_{x1}_{y1}.png"
        out_path = img_path + suffix
        cropped.save(out_path)
        return out_path
    except Exception as e:
        log.debug("tighter crop failed: %s", e)
        return None


def _render_page_at_dpi(
    pdf_path: str,
    page_num: int,
    render_dpi: float,
    out_dir: str | None = None,
) -> str | None:
    """Render page at ``render_dpi`` using PyMuPDF. Returns PNG path or None."""
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        zoom = render_dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix)
        out_dir = out_dir or tempfile.gettempdir()
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(
            out_dir, f"{os.path.basename(pdf_path)}_p{page_num}_dpi{int(render_dpi)}.png"
        )
        pix.save(out_path)
        doc.close()
        return out_path
    except Exception as e:
        log.debug("render at dpi=%s failed: %s", render_dpi, e)
        return None


def execute_action(
    action: RetryAction,
    context: dict[str, Any],
) -> RetryOutcome:
    """Execute a retry action.

    Args:
        action: A :class:`RetryAction` (typically from
            :func:`scientific_parser.verification.retry_actions.action_for`).
        context: Execution context. Recognized keys:
            - ``img_path``: rendered page PNG (for TIGHTER_CROP)
            - ``region_bbox``: (x0, y0, x1, y1) bbox in render px
            - ``pdf_path``: PDF file path (for HIGHER_DPI)
            - ``page_num``: 1-indexed page number (for HIGHER_DPI)
            - ``current_dpi``: DPI at which ``img_path`` was rendered
            - ``ocr_fn``: callable(img_path, model) -> text (for RERUN_REGION)
            - ``model``: OCR model name (for RERUN_REGION)

    Returns:
        A :class:`RetryOutcome`. Never raises.
    """
    atype = action.action
    details = dict(action.params)

    # --- ABSTAIN ------------------------------------------------------ #
    if atype is RetryActionType.ABSTAIN:
        return RetryOutcome(
            success=True,
            action_type=atype,
            details=details,
            message="abstained (no corrective action)",
        )

    # --- TIGHTER_CROP ------------------------------------------------- #
    if atype is RetryActionType.TIGHTER_CROP:
        img_path = context.get("img_path")
        bbox = context.get("region_bbox")
        if not img_path or not bbox or len(bbox) != 4:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="missing img_path or region_bbox in context",
            )
        pad = int(details.get("pad_px", 2))
        out_path = _crop_with_padding(img_path, bbox, pad_px=pad)
        if out_path is None:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="crop produced empty region",
            )
        details["pad_px"] = pad
        details["out_path"] = out_path
        return RetryOutcome(
            success=True,
            action_type=atype,
            output_path=out_path,
            details=details,
            message=f"tighter crop → {out_path}",
        )

    # --- HIGHER_DPI --------------------------------------------------- #
    if atype is RetryActionType.HIGHER_DPI:
        pdf_path = context.get("pdf_path")
        page_num = context.get("page_num")
        target_dpi = float(details.get("fallback_dpi", 300))
        if not pdf_path or not page_num:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="missing pdf_path or page_num in context",
            )
        out_path = _render_page_at_dpi(pdf_path, int(page_num), target_dpi)
        if out_path is None:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message=f"render at {target_dpi} DPI failed",
            )
        details["render_dpi"] = target_dpi
        details["out_path"] = out_path
        return RetryOutcome(
            success=True,
            action_type=atype,
            output_path=out_path,
            details=details,
            message=f"re-rendered at {target_dpi} DPI → {out_path}",
        )

    # --- RERUN_REGION ------------------------------------------------- #
    if atype is RetryActionType.RERUN_REGION:
        ocr_fn = context.get("ocr_fn")
        img_path = context.get("img_path")
        model = context.get("model", "glm-ocr")
        if not callable(ocr_fn) or not img_path:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="missing callable ocr_fn or img_path in context",
            )
        try:
            text = str(ocr_fn(img_path, model) or "")
            return RetryOutcome(
                success=bool(text),
                action_type=atype,
                output_text=text,
                details=details,
                message=f"re-OCR with {model} → {len(text)} chars",
            )
        except Exception as e:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message=f"re-OCR error: {type(e).__name__}: {e}",
            )

    # --- SPECIALIST_MODEL (closed loop 2026-08-15) --------------------- #
    # Was deferred (log + fall through). Now dispatches to the installed
    # ONNX specialists: formula_latex → RapidLaTeX, table_structure →
    # RapidTable. Requires img_path + region_bbox in context (same
    # contract as TIGHTER_CROP — the classifier emits both together).
    if atype is RetryActionType.SPECIALIST_MODEL:
        specialist = details.get("specialist", "unknown")
        img_path = context.get("img_path")
        bbox = context.get("region_bbox")
        if not img_path or not bbox or len(bbox) != 4:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="SPECIALIST_MODEL needs img_path + region_bbox in context",
            )
        crop = _crop_with_padding(img_path, tuple(bbox), pad_px=int(details.get("pad_px", 5)))
        if crop is None:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="specialist crop produced empty region",
            )
        if specialist == "formula_latex":
            try:
                from ..science.equation_cascade import _clean_latex_shell, _get_rapidlatex

                latex, _conf = _get_rapidlatex().recognize(crop)
                latex = _clean_latex_shell(latex or "")
                if not latex:
                    raise RuntimeError("empty RapidLaTeX output")
                details["out_path"] = crop
                return RetryOutcome(
                    success=True,
                    action_type=atype,
                    output_path=crop,
                    output_text=f"$${latex}$$",
                    details=details,
                    message=f"formula specialist (RapidLaTeX) → {len(latex)} chars",
                )
            except Exception as e:
                return RetryOutcome(
                    success=False,
                    action_type=atype,
                    details=details,
                    message=f"formula specialist failed: {type(e).__name__}: {e}",
                )
        if specialist == "table_structure":
            try:
                from ..perception.providers.table_recognizer import get_table_recognizer

                rec = get_table_recognizer("default")
                if rec is None or not rec.available():
                    raise RuntimeError("rapid-table not installed")
                html, conf = rec.extract_table(crop)
                if not html:
                    raise RuntimeError("empty RapidTable output")
                details["out_path"] = crop
                details["confidence"] = conf
                return RetryOutcome(
                    success=True,
                    action_type=atype,
                    output_path=crop,
                    output_text=html,
                    details=details,
                    message=f"table specialist (RapidTable) → {len(html)} chars",
                )
            except Exception as e:
                return RetryOutcome(
                    success=False,
                    action_type=atype,
                    details=details,
                    message=f"table specialist failed: {type(e).__name__}: {e}",
                )
        return RetryOutcome(
            success=False,
            action_type=atype,
            details=details,
            message=f"unknown specialist: {specialist}",
        )

    # --- COLUMN_RECONSTRUCT (closed loop 2026-08-15) ------------------- #
    # Re-run column-aware reading order (recursive XY-cut) on the page's
    # text-layer blocks. Requires pdf_path + page_num.
    if atype is RetryActionType.COLUMN_RECONSTRUCT:
        pdf_path = context.get("pdf_path")
        page_num = context.get("page_num")
        if not pdf_path or not page_num:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message="COLUMN_RECONSTRUCT needs pdf_path + page_num in context",
            )
        try:
            from ..structure.column_reading_order import column_aware_reading_order

            text = column_aware_reading_order(pdf_path, int(page_num))
            if not text or not text.strip():
                raise RuntimeError("xy-cut returned empty ordering")
            return RetryOutcome(
                success=True,
                action_type=atype,
                output_text=text,
                details=details,
                message=f"column reconstruct → {len(text)} chars",
            )
        except Exception as e:
            return RetryOutcome(
                success=False,
                action_type=atype,
                details=details,
                message=f"column reconstruct failed: {type(e).__name__}: {e}",
            )

    # --- Unknown action ------------------------------------------------ #
    return RetryOutcome(
        success=False,
        action_type=atype,
        details=details,
        message=f"unknown action type: {atype}",
    )


__all__ = [
    "RetryOutcome",
    "execute_action",
]
