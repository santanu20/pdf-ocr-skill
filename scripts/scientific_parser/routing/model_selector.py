"""Capability-based model selector (proposal item #34).

Per architecture proposal:
  "Make OCR engines interchangeable. Routing becomes capability-based:
      backend.supports('formula')   # not:  if model == 'glm-ocr'
   That will save you from another rewrite when PP-OCRv7 inevitably appears."

Wraps the OCREngine ABC + capabilities + manifest registry. Selects backend
based on required capabilities (e.g. user wants 'formula' support → glm-ocr).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import sys

if "scripts" not in sys.path:
    sys.path.insert(0, "scripts")


# Static capability → backend mapping (built from MODEL_PROFILES + manifests)
# This avoids importing concrete backends (which may need heavy deps).
_CAPABILITY_BACKENDS = {
    "text": ["rapidocr", "qwen2.5vl:3b", "glm-ocr"],  # ordered by speed
    "multilingual": ["qwen2.5vl:3b", "rapidocr", "glm-ocr"],
    "formula": ["rapidlatex", "texo", "unimernet", "glm-ocr"],
    "table": ["rapidtable", "glm-ocr", "qwen2.5vl:7b"],
    "chemical_structure": ["decimer", "molscribe"],  # OCSR (reviewer #5)
    "chart": [],  # not yet supported in any backend
    "layout": ["doclayout-yolo", "rapid-layout", "qwen2.5vl:3b"],
    "handwriting": [],  # not yet supported
    "vertical_text": [],
}


@dataclass
class ModelSelection:
    """Result of capability-based model selection."""

    backend_name: str
    capabilities_provided: list
    capabilities_missing: list
    fallback_used: bool = False
    note: str = ""


def select_model(
    required_capabilities: list,
    available_backends: Optional[list] = None,
    prefer_speed: bool = False,
) -> Optional[ModelSelection]:
    """Pick backend that satisfies required capabilities.

    Args:
        required_capabilities: e.g. ["text", "formula"]
        available_backends: list of backend names installed. If None, assume all.
        prefer_speed: if True, prefer faster backends (rapidocr > qwen > glm-ocr)

    Returns:
        ModelSelection or None if no backend satisfies requirements.
    """
    if not required_capabilities:
        # Default: fastest text backend
        return ModelSelection(
            backend_name="rapidocr",
            capabilities_provided=["text"],
            capabilities_missing=[],
            note="default fallback (no capabilities requested)",
        )

    available = available_backends or list(
        {b for caps in _CAPABILITY_BACKENDS.values() for b in caps}
    )

    # Score each backend: covers how many required capabilities?
    candidates = []
    for backend in available:
        provided = []
        missing = []
        for cap in required_capabilities:
            if backend in _CAPABILITY_BACKENDS.get(cap, []):
                provided.append(cap)
            else:
                missing.append(cap)
        candidates.append((backend, provided, missing))

    # Prefer backends that cover ALL requirements
    full_match = [c for c in candidates if not c[2]]
    if full_match:
        # Sort by preference order
        if prefer_speed:
            speed_order = {
                "rapidocr": 0,
                "qwen2.5vl:3b": 1,
                "qwen2.5vl:7b": 2,
                "glm-ocr": 3,
            }
            full_match.sort(key=lambda c: speed_order.get(c[0], 99))
        else:
            # Accuracy preference
            acc_order = {
                "qwen2.5vl:7b": 0,
                "qwen2.5vl:3b": 1,
                "glm-ocr": 2,
                "rapidocr": 3,
            }
            full_match.sort(key=lambda c: acc_order.get(c[0], 99))
        backend, provided, missing = full_match[0]
        return ModelSelection(
            backend_name=backend,
            capabilities_provided=provided,
            capabilities_missing=missing,
            note="full capability match",
        )

    # No full match — pick best partial
    candidates.sort(key=lambda c: -len(c[1]))
    if candidates and candidates[0][1]:
        backend, provided, missing = candidates[0]
        return ModelSelection(
            backend_name=backend,
            capabilities_provided=provided,
            capabilities_missing=missing,
            fallback_used=True,
            note=f"partial match — missing: {missing}",
        )

    return None


def list_capabilities() -> dict:
    """Return current capability → backend mapping."""
    return dict(_CAPABILITY_BACKENDS)


__all__ = ["ModelSelection", "select_model", "list_capabilities"]
