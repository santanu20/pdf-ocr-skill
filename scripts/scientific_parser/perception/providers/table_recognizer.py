"""RapidTableProvider — ONNX table structure recognition.

Extracted from scripts/pdf_ocr.py during Phase R4 refactoring (2026-08-12).
Pure ONNX via RapidAI family. ~30MB model auto-downloads on first use.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_RAPIDTABLE_INSTANCE = None


class RapidTableProvider:
    """ONNX table structure recognition via RapidTable.

    Lazy-loaded. Auto-activates when installed (CORE dep).
    """

    PROVIDES = "rapidtable"

    def __init__(self):
        self._engine = None

    def _ensure_loaded(self):
        global _RAPIDTABLE_INSTANCE
        if _RAPIDTABLE_INSTANCE is not None:
            self._engine = _RAPIDTABLE_INSTANCE
            return
        try:
            from rapid_table import RapidTable

            _RAPIDTABLE_INSTANCE = RapidTable()
            self._engine = _RAPIDTABLE_INSTANCE
            log.info("RapidTable loaded (ONNX)")
        except ImportError as e:
            raise RuntimeError(
                f"rapid-table not installed. Install: uv pip install rapid-table. ({e})"
            )
        except Exception as e:
            raise RuntimeError(f"RapidTable init failed: {e}")

    def available(self) -> bool:
        try:
            import rapid_table  # noqa: F401

            return True
        except ImportError:
            return False

    def extract_table(self, img_path, ocr_result=None) -> tuple[str, float]:
        """Extract table from image. Returns (html, confidence).

        If ocr_result is provided (from rapidocr), used to populate cells.
        Else, RapidTable will run its own OCR (slower).
        """
        self._ensure_loaded()
        if self._engine is None:
            return "", 0.0
        try:
            # Engine expects ocr_results as a List[Tuple] (one per image);
            # extract_table handles a single image, so wrap into a 1-list.
            ocr_list = [ocr_result] if ocr_result is not None else None
            result = self._engine(img_path, ocr_results=ocr_list)
            # RapidTable returns RapidTableOutput.pred_htmls — a list of HTML
            # strings (one per input image). Read it first; fall back to the
            # legacy "html" attr / dict shape for older API versions.
            htmls = getattr(result, "pred_htmls", None) or []
            html = htmls[0] if htmls else ""
            if not html:
                html = getattr(result, "html", "") or (
                    result.get("html", "") if isinstance(result, dict) else ""
                )
            # RapidTable doesn't expose confidence; use 0.85 if html present
            conf = 0.85 if html else 0.0
            return html, conf
        except Exception as e:
            log.warning("RapidTable failed: %s", e)
            return "", 0.0


def get_table_recognizer(backend: str = "default"):
    """Factory: table recognizer by backend name.

    backends:
      - "default" / "rapidtable": RapidTable ONNX (auto-activates)
      - "off" / other: returns None
    """
    if backend in ("rapidtable", "default"):
        try:
            return RapidTableProvider()
        except Exception as e:
            log.warning("RapidTable backend unavailable: %s", e)
            return None
    return None


__all__ = ["RapidTableProvider", "get_table_recognizer"]
