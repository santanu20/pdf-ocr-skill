"""RapidOCR (PP-OCRv6) in-process backend.

v3 API: RapidOCR() → returns RapidOCROutput with .txts/.boxes/.scores.
"""

from __future__ import annotations
from typing import Optional
from .interface import OCRBackend, OCRResult, CAP_TEXT, CAP_MULTILINGUAL


class RapidOCRBackend(OCRBackend):
    """PP-OCRv6 small (default) via RapidOCR v3.x."""

    name = "rapidocr"
    capabilities = frozenset({CAP_TEXT, CAP_MULTILINGUAL})

    def __init__(self):
        self._engine = None

    def _get_engine(self):
        if self._engine is None:
            from rapidocr import RapidOCR

            self._engine = RapidOCR()
        return self._engine

    def available(self) -> bool:
        try:
            import rapidocr  # noqa

            return True
        except ImportError:
            return False

    def ocr(self, image_path: str, task: Optional[str] = None) -> OCRResult:
        """OCR image via PP-OCRv6. Returns unified OCRResult."""
        import time

        t0 = time.monotonic()
        try:
            engine = self._get_engine()
            output = engine(image_path)
            elapsed = time.monotonic() - t0

            # v3 API
            if hasattr(output, "txts"):
                # v3 returns tuples. Use len() checks (NOT truthiness —
                # boxes is tuple-of-numpy-arrays, `if output.boxes:` raises
                # "truth value of array ambiguous").
                txts = (
                    list(output.txts)
                    if (output.txts is not None and len(output.txts) > 0)
                    else []
                )
                scores = (
                    list(output.scores)
                    if (output.scores is not None and len(output.scores) > 0)
                    else []
                )
                boxes = (
                    list(output.boxes)
                    if (output.boxes is not None and len(output.boxes) > 0)
                    else []
                )
                if scores and len(txts) == len(scores):
                    kept = []
                    for t, s in zip(txts, scores):
                        try:
                            if float(s) >= 0.5:
                                kept.append(t)
                        except (TypeError, ValueError):
                            kept.append(t)
                    text = "\n".join(str(t) for t in kept)
                    confs = [float(s) for s in scores]
                    avg_conf = sum(confs) / max(len(confs), 1) if confs else 1.0
                else:
                    text = "\n".join(str(t) for t in txts)
                    avg_conf = 1.0
                regions = []
                if len(boxes) > 0:
                    for b, t, s in zip(boxes, txts, scores):
                        try:
                            bbox = (
                                list(b[0]) if (b is not None and len(b) > 0) else None
                            )
                        except Exception:
                            bbox = None
                        try:
                            conf = float(s) if s is not None else 0.0
                        except (TypeError, ValueError):
                            conf = 0.0
                        regions.append(
                            {"bbox": bbox, "text": str(t), "confidence": conf}
                        )
                return OCRResult(
                    text=text,
                    confidence=avg_conf,
                    backend=self.name,
                    regions=regions,
                    metadata={"elapsed_s": round(elapsed, 2)},
                )
            # Legacy v1.4.4 tuple API
            if isinstance(output, tuple) and output[0]:
                result = output[0]
                text_parts = [t for (_, t, c) in result if c >= 0.5]
                avg = sum(c for *_, c in result) / max(len(result), 1)
                return OCRResult(
                    text="\n".join(text_parts),
                    confidence=avg,
                    backend=self.name,
                    metadata={"elapsed_s": round(elapsed, 2)},
                )
            return OCRResult(text="", confidence=0.0, backend=self.name)
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                backend=self.name,
                metadata={"error": str(e)[:200]},
            )


__all__ = ["RapidOCRBackend"]
