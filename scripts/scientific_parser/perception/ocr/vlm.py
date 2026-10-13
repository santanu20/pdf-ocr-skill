"""VLM-backed OCR — generic vision-language model adapter.

For models not directly OCR-trained but capable of OCR via prompting
(GPT-4V, Claude Vision, MiniCPM-V, etc.).
"""

from __future__ import annotations
from typing import Optional, Callable
from .interface import OCRBackend, OCRResult, CAP_TEXT, CAP_FORMULA


class VLMBackend(OCRBackend):
    """Adapter for arbitrary VLM with OCR capability.

    Pass a callable: `fn(image_path, prompt) -> str`.
    """

    name = "vlm"

    def __init__(
        self,
        call_fn: Callable[[str, str], str],
        name: str = "vlm",
        capabilities: frozenset = frozenset({CAP_TEXT}),
    ):
        self._call = call_fn
        self.name = name
        self.capabilities = capabilities

    def available(self) -> bool:
        return self._call is not None

    def ocr(self, image_path: str, task: Optional[str] = None) -> OCRResult:
        import time

        t0 = time.monotonic()
        try:
            text = self._call(image_path, task or "Transcribe this image.")
            elapsed = time.monotonic() - t0
            return OCRResult(
                text=text,
                confidence=0.80,
                backend=self.name,
                metadata={"elapsed_s": round(elapsed, 2)},
            )
        except Exception as e:
            return OCRResult(
                text="",
                confidence=0.0,
                backend=self.name,
                metadata={"error": str(e)[:200]},
            )


__all__ = ["VLMBackend"]
