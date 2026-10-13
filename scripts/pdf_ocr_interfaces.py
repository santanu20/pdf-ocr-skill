"""Provider interface contracts for the pdf-ocr skill.

These ABCs define stable contracts that backend providers implement.
The current skill script (pdf_ocr.py) contains heuristic implementations
of all these interfaces; future ML/VLM backends implement the same
contracts and are selected via flags (e.g. --layout-backend, --math-backend).

Design principles:
  - **Inputs/outputs are plain dataclasses / dicts** — no PDF object handles
    passed across the boundary, so backends can run in subprocesses if needed.
  - **Each provider is composable** — a Chunker consumes output from any
    LayoutDetector + TableExtractor; an OCREngine is independent of layout.
  - **Failure modes are explicit** — providers return (result, confidence,
    warnings) tuples so callers (TierRouter, ConfidenceAggregator) can route
    low-quality output to a stronger backend.
  - **Lazy registration** — providers register via __init_subclass__ so the
    registry auto-populates without central editing; plugins (Phase 9) use
    the same mechanism.

All providers MUST be import-safe: importing them must not require any
optional ML dependencies (torch, transformers, etc.). Heavy imports belong
inside __init__ methods or lazy properties, not at module top.

Reference: pdf-ocr-sota-agentic-redesign plan, yggdrasil session
dp-20260811-pdf-ocr-sota-agentic-redesign.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


# =========================================================================== #
# Data containers — plain data, no handles, JSON-serializable where possible.
# =========================================================================== #


@dataclass
class BBox:
    """Axis-aligned bounding box in PDF points (top-left origin)."""

    x: float
    y: float
    w: float
    h: float

    @property
    def x1(self) -> float:
        return self.x + self.w

    @property
    def y1(self) -> float:
        return self.y + self.h


@dataclass
class Region:
    """A detected layout region on a single page.

    type ∈ {text, title, list, table, figure, figure_caption, footer,
            header, page_number, equation, equation_number, code, unknown}
    """

    page: int  # 1-indexed
    bbox: BBox
    type: str
    text: str = ""
    confidence: float = 1.0
    children: list["Region"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Table:
    """A recognized table."""

    page: int
    bbox: BBox
    rows: list[list[str]]  # cells as strings; merged cells appear once
    header_rows: int = 1
    confidence: float = 1.0
    html: Optional[str] = None  # optional HTML form (preserves rowspan/colspan)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        if not self.rows:
            return ""
        n_cols = max(len(r) for r in self.rows) if self.rows else 0
        norm = [r + [""] * (n_cols - len(r)) for r in self.rows]
        out = []
        for row in norm:
            # det44: sepless emission, single-line cells
            out.append("| " + " | ".join(" ".join(str(c).split()) for c in row) + " |")
        return "\n".join(out) + "\n"


@dataclass
class PageResult:
    """Output of processing one page through any backend."""

    page: int  # 1-indexed
    markdown: str
    regions: list[Region] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    source: str = "unknown"  # pdftotext | ocr | ml | vlm
    confidence: float = 1.0
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


# =========================================================================== #
# Provider registry — auto-populates via __init_subclass__.
# =========================================================================== #


_REGISTRY: dict[str, dict[str, type]] = {}


def register_provider(interface: str, name: str, cls: type) -> None:
    """Manually register a provider class. Useful for testing."""
    _REGISTRY.setdefault(interface, {})[name] = cls


def get_provider(interface: str, name: str) -> type:
    """Look up a registered provider class by interface + name."""
    if interface not in _REGISTRY:
        raise KeyError(
            f"No providers registered for interface '{interface}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )
    if name not in _REGISTRY[interface]:
        available = list(_REGISTRY[interface].keys())
        raise KeyError(f"No provider '{name}' for interface '{interface}'. Available: {available}")
    return _REGISTRY[interface][name]


def list_providers(interface: str) -> list[str]:
    """List registered provider names for an interface."""
    return sorted(_REGISTRY.get(interface, {}).keys())


class _ProviderMeta(type(ABC)):
    """Metaclass that auto-registers concrete provider subclasses.

    A subclass is registered when:
      - It's a concrete class (no abstract methods left)
      - It defines class attributes ``INTERFACE`` and ``NAME``
      - ``NAME`` is not "default" OR no other default is registered yet
    """

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        iface = namespace.get("INTERFACE") or getattr(cls, "INTERFACE", None)
        pname = namespace.get("NAME") or getattr(cls, "NAME", None)
        if iface and pname and not getattr(cls, "__abstractmethods__", None):
            register_provider(iface, pname, cls)


# =========================================================================== #
# Interface 1: LayoutDetector
#   Detects layout regions (text/title/table/figure/equation/...) on a page
#   and returns them in reading order.
# =========================================================================== #


class LayoutDetector(ABC, metaclass=_ProviderMeta):
    """Detect layout regions on a page and return reading order.

    Providers:
      - "default" (heuristic, current): PyMuPDF block sort + classify_text_block
      - "doclayout-yolo" (Phase 2): DocLayout-YOLO via ultralytics
    """

    INTERFACE = "layout_detector"

    @abstractmethod
    def detect_regions(self, pdf_path: str, page_num: int) -> list[Region]:
        """Return regions in reading order. page_num is 1-indexed."""
        ...

    @abstractmethod
    def available(self) -> bool:
        """True if this backend's deps + weights are ready."""
        ...


# =========================================================================== #
# Interface 2: TableExtractor
#   Detects table regions and reconstructs their structure.
# =========================================================================== #


class TableExtractor(ABC, metaclass=_ProviderMeta):
    """Extract tables from a page.

    Providers:
      - "default" (heuristic, current): 6-strategy detector
        (pdfplumber + lines_strict + text_only_pattern + ocr_text_parse +
        camelot_lattice + camelot_stream)
      - "tableformer" (Phase 4): IBM TableFormer via docling_contrib
    """

    INTERFACE = "table_extractor"

    @abstractmethod
    def extract_tables(self, pdf_path: str, page_num: int) -> list[Table]:
        """Return all tables found on the page. page_num is 1-indexed."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 3: MathOCR
#   Recognize mathematical expressions and emit LaTeX.
# =========================================================================== #


class MathOCR(ABC, metaclass=_ProviderMeta):
    """OCR math regions to LaTeX.

    Providers:
      - "default" (heuristic, current): font-based detection (CMMI/CMSY/
        CMEX/MSBM/MSAM/EU1/EU2/Frak) + Unicode-to-LaTeX mapping table
      - "unimernet" (Phase 3): UniMERNet Swin+LAM+mBART (CVPR 2026)
    """

    INTERFACE = "math_ocr"

    @abstractmethod
    def recognize(self, image_path_or_bbox: str | tuple) -> str:
        """Return LaTeX for an equation image or region. Inline `$...$`."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 4: FigureUnderstanding
#   Classify figure regions; optionally extract chart data.
# =========================================================================== #


class FigureUnderstanding(ABC, metaclass=_ProviderMeta):
    """Analyze figure/chart regions.

    Providers:
      - "default" (heuristic, current): classify_text_block distinguishes
        BODY vs FIGURE_GLYPH/FIGURE_TICK/FIGURE_LABEL_BLOCK/AXIS_TITLE;
        figure content preserved verbatim.
      - "tinychart" (Phase 5): TinyChart 3.1B VLM for chart captioning +
        data extraction
    """

    INTERFACE = "figure_understanding"

    @abstractmethod
    def analyze_figure(self, pdf_path: str, page_num: int, bbox: BBox) -> dict[str, Any]:
        """Return {caption, data_table, chart_type, confidence}."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 5: OCREngine
#   Image-to-text via a vision model or OCR engine.
# =========================================================================== #


class OCREngine(ABC, metaclass=_ProviderMeta):
    """OCR an image to text.

    Providers:
      - "ollama" (default, current): Ollama vision model (glm-ocr / ovisocr2 /
        any model in MODEL_PROFILES). Zero data egress.
      - "qwen2.5vl" / "qwen2.5vl:3b" / "qwen2.5vl:7b": Ollama via /api/chat
        endpoint (auto-detected). 2026 OCR winner for plain text.
      - "rapidocr" (Phase 12 plugin): RapidOCR v3 = PP-OCRv6 small. CPU, in-process,
        12s/pg. Speed tier.
      - "tesseract" (Phase 12 plugin): Tesseract 5, CPU, 100+ langs
      - "paddle" (Phase 12 plugin): PaddleOCR / RapidOCR for CJK

    Capabilities (per proposal item #34): each provider declares which
    capability tokens it supports. Routing is capability-based, not name-based:
        backend.supports("formula")  # not: if model == "glm-ocr"

    Valid capability tokens:
      - "text" — plain Latin-script text recognition
      - "multilingual" — non-Latin scripts (CJK, Arabic, Cyrillic, etc.)
      - "formula" — mathematical equation recognition (LaTeX output)
      - "table" — table structure recognition
      - "chart" — chart/figure semantic interpretation
      - "layout" — layout/region detection
      - "handwriting" — handwriting recognition
      - "vertical_text" — vertical-script support
    """

    INTERFACE = "ocr_engine"

    #: Capabilities supported by this backend (subclasses override).
    #: Used by capability-based routing (proposal item #34).
    CAPABILITIES: frozenset = frozenset({"text"})

    @abstractmethod
    def ocr(self, image_path: str, task: Optional[str] = None) -> str:
        """Return text transcription. ``task`` is an optional prompt."""
        ...

    def supports(self, capability: str) -> bool:
        """Capability-based routing check."""
        return capability in self.CAPABILITIES

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 6: PostProcessor
#   Apply scientific/typographic cleanup to extracted text.
# =========================================================================== #


class PostProcessor(ABC, metaclass=_ProviderMeta):
    """Apply a specific cleanup/validation pass to text.

    Providers (Phase 9 plugin discovery):
      - "permille_fix", "chemical_notation", "formula_validator",
        "ree_validator", "mineral_abbrevs", "phantom_decimal_sanitizer",
        "dehyphenation", "citation_repair", "doi_preservation",
        "bibliography_mode", "oxide_total_validator", "age_validator",
        "degree_symbol", "ocr_correction", "scientific_symbols",
        "subscript_restore", "unicode_restore", "allcaps_promotion",
        "typography_preserve", "header_strip", "footnote_reconstruct",
        ... (current 12 + Phase 11 expansion)
    """

    INTERFACE = "post_processor"

    @abstractmethod
    def process(self, text: str, context: Optional[dict[str, Any]] = None) -> tuple[str, list[str]]:
        """Return (processed_text, warnings_emitted)."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 7: PageRouter
#   Decide per-page whether to use fast path (text layer) or OCR.
# =========================================================================== #


class PageRouter(ABC, metaclass=_ProviderMeta):
    """Route a page to the right backend.

    Providers:
      - "default" (heuristic, current): 7-signal quality scoring
        (alphanum_ratio, space_ratio, newline_ratio, invalid_char_count,
        multi_space_run_ratio, page_has_invisible_text,
        page_has_embedded_fonts) + adaptive DPI
      - Future: tier-aware router with confidence-based escalation
    """

    INTERFACE = "page_router"

    @abstractmethod
    def route(self, pdf_path: str, page_num: int, doc_info: dict[str, Any]) -> dict[str, Any]:
        """Return {backend: 'fast'|'ocr'|'vlm', dpi: int, reason: str, ...}."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 8: Chunker (Phase 1 — agent output layer)
#   Slice a document into agent-friendly chunks for RAG / retrieval.
# =========================================================================== #


class Chunker(ABC, metaclass=_ProviderMeta):
    """Produce chunks for downstream RAG / agent retrieval.

    Providers (Phase 1):
      - "fixed": fixed-size token chunks (baseline)
      - "semantic": sentence-boundary chunks via embedding similarity
      - "hierarchical": section-aware chunks with parent-child breadcrumbs
      - "contextual": Anthropic Contextual Retrieval — LLM-generated context
        prefix per chunk
    """

    INTERFACE = "chunker"

    @abstractmethod
    def chunk(self, pages: list[PageResult], target_tokens: int = 512) -> list[dict[str, Any]]:
        """Return list of {text, page_refs, section_path, tokens, metadata}."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Interface 9: ConfidenceScorer (Phase 7 — self-correction loop)
#   Per-page / per-region quality metric in [0, 1].
# =========================================================================== #


class ConfidenceScorer(ABC, metaclass=_ProviderMeta):
    """Score page / region quality for the self-correction loop.

    Providers:
      - "default" (Phase 7): weighted blend of (text_density,
        validator_warning_count, ocr_repetition_ratio, table_confidence,
        math_recognition_rate)
    """

    INTERFACE = "confidence_scorer"

    @abstractmethod
    def score_page(self, page: PageResult) -> float:
        """Return confidence in [0, 1]. Below threshold triggers escalation."""
        ...

    @abstractmethod
    def available(self) -> bool: ...


# =========================================================================== #
# Public API
# =========================================================================== #

__all__ = [
    # Data containers
    "BBox",
    "Region",
    "Table",
    "PageResult",
    # Interfaces
    "LayoutDetector",
    "TableExtractor",
    "MathOCR",
    "FigureUnderstanding",
    "OCREngine",
    "PostProcessor",
    "PageRouter",
    "Chunker",
    "ConfidenceScorer",
    # Registry
    "register_provider",
    "get_provider",
    "list_providers",
]
