"""OCRBackend interface + capabilities (proposal item #34).

Capability-based routing:
    backend.supports("formula")   # not:  if model == "glm-ocr"
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


# Capability tokens
CAP_TEXT = "text"
CAP_MULTILINGUAL = "multilingual"
CAP_FORMULA = "formula"
CAP_TABLE = "table"
CAP_CHART = "chart"
CAP_LAYOUT = "layout"
CAP_HANDWRITING = "handwriting"
CAP_VERTICAL_TEXT = "vertical_text"


@dataclass
class OCRResult:
    """Unified output of any OCR backend."""

    text: str = ""
    confidence: float = 1.0
    backend: str = ""
    regions: list = field(default_factory=list)  # list of (bbox, text, conf)
    metadata: dict = field(default_factory=dict)


class OCRBackend(ABC):
    """Abstract OCR backend. Concrete backends override."""

    name: str = "abstract"
    capabilities: frozenset = frozenset({CAP_TEXT})

    @abstractmethod
    def ocr(self, image_path: str, task: Optional[str] = None) -> OCRResult: ...

    @abstractmethod
    def available(self) -> bool: ...

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities


__all__ = [
    "CAP_TEXT",
    "CAP_MULTILINGUAL",
    "CAP_FORMULA",
    "CAP_TABLE",
    "CAP_CHART",
    "CAP_LAYOUT",
    "CAP_HANDWRITING",
    "CAP_VERTICAL_TEXT",
    "OCRResult",
    "OCRBackend",
]
