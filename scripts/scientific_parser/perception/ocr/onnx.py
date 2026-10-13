"""Generic ONNX backend — load any ONNX model with manifest.

Per proposal item #36: "Model artifact should be self-describing."
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional
from .interface import OCRBackend, OCRResult, CAP_TEXT
import logging

log = logging.getLogger(__name__)


class OnnxBackend(OCRBackend):
    """Generic ONNX OCR backend driven by ModelManifest.

    Loads model + dictionary from manifest, runs inference via onnxruntime.
    Pre/post-processing defined in manifest JSON.
    """

    name = "onnx"

    def __init__(self, manifest_path: str):
        self.manifest_path = Path(manifest_path)
        self._manifest = None
        self._session = None
        try:
            import sys

            if "scripts" not in sys.path:
                sys.path.insert(0, "scripts")
            from model_manifest import ModelManifest

            self._manifest = ModelManifest.from_json(
                self.manifest_path / "manifest.json"
            )
            self.name = self._manifest.name
            self.capabilities = frozenset(self._manifest.capabilities or [CAP_TEXT])
        except Exception as e:
            log.debug("__init__: Exception degraded: %s", e)

    def available(self) -> bool:
        try:
            import onnxruntime  # noqa

            return self._manifest is not None
        except ImportError:
            return False

    def ocr(self, image_path: str, task: Optional[str] = None) -> OCRResult:
        """Run inference. Returns OCRResult.

        NOTE: This is a generic stub — concrete OCR (CTC decode, etc.) requires
        per-model postprocessing defined in manifest. For full impl, use
        RapidOCRBackend or OllamaBackend instead.
        """
        return OCRResult(
            text="",
            confidence=0.0,
            backend=self.name,
            metadata={
                "error": "OnnxBackend requires per-model postprocessing — use RapidOCRBackend"
            },
        )


__all__ = ["OnnxBackend"]
