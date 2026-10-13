"""SOTA specialist providers for per-region extraction.

Two tiers (both CORE, both auto-activate on content detection):
  - Tier 1 (ONNX, lightweight): RapidLaTeX / RapidTable / RapidLayout — no torch.
  - Tier 2 (torch VLMs, research-grade): UniMERNet (equations),
    DECIMER / MolScribe (chemical structures), DocLayout-YOLO (layout).

All graceful-degrade — ``.available()`` gates whether the model runs; the
dispatcher unconditionally TRIES them, so they activate the moment the dep
+ weights are present. Extracted from scripts/pdf_ocr.py during Phase R4
refactoring (2026-08-12); Tier-2 adapters added 2026-08-13 (reviewer #5).
"""

from .chemical_recognizer import (
    DecimerProvider,
    MolScribeProvider,
    get_chemical_recognizer,
)
from .formula_ocr import (
    RapidLaTeXProvider,
    UniMERNetProvider,
    get_formula_recognizer,
)
from .layout_detector import (
    DocLayoutYOLOOnnxProvider,
    DocLayoutYOLOProvider,
    RapidLayoutProvider,
    get_layout_detector,
)
from .region_dispatcher import (
    _crop_image,
    _extract_text_from_bbox,
    _sort_regions_reading_order,
    route_page_by_region,
)
from .table_recognizer import RapidTableProvider, get_table_recognizer

__all__ = [
    "RapidLaTeXProvider",
    "UniMERNetProvider",
    "get_formula_recognizer",
    "RapidLayoutProvider",
    "DocLayoutYOLOProvider",
    "DocLayoutYOLOOnnxProvider",
    "get_layout_detector",
    "RapidTableProvider",
    "get_table_recognizer",
    "DecimerProvider",
    "MolScribeProvider",
    "get_chemical_recognizer",
    "route_page_by_region",
    "_crop_image",
    "_sort_regions_reading_order",
    "_extract_text_from_bbox",
]
