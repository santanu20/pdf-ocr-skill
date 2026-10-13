"""Core IR: Document / Page / Block + Evidence Hierarchy E0-E6.

Per architecture proposal items #6, #9 (2026-08-12) and the DocTrace paper
(arXiv 2608.03292, Aug 2026):
  "Hierarchical framework that progressively performs evidence localization,
   structured document parsing, and evidence graph reasoning to enable
   explicit evidence provenance."

Evidence Hierarchy (proposal item #9):
  E0 = native PDF text           (highest priority — use directly)
  E1 = native PDF geometry       (font, position, vector)
  E2 = deterministic reconstruction  (regex normalization, dictionary lookup)
  E3 = classical OCR             (Tesseract, PP-OCRv6, RapidOCR)
  E4 = specialist OCR            (math-aware: glm-ocr, UniMERNet)
  E5 = VLM interpretation        (qwen2.5vl, Marker, MinerU)
  E6 = inferred semantic value   (Quantity, Entity, Measurement)

Lower levels are NEVER overwritten by higher levels — they coexist.
Corrections are reversible because original evidence is retained.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Runtime import is lazy inside DocumentBlock.resolve() — evidence.py
    # imports from this module (avoid circularity).
    from .evidence import EvidenceResolver

# === Evidence Hierarchy =========================================== #

E0_NATIVE_PDF_TEXT = "E0_native_pdf_text"
E1_NATIVE_PDF_GEOM = "E1_native_pdf_geometry"
E2_DETERMINISTIC = "E2_deterministic_reconstruction"
E3_CLASSICAL_OCR = "E3_classical_ocr"
E4_SPECIALIST_OCR = "E4_specialist_ocr"
E5_VLM = "E5_vlm"
E6_SEMANTIC = "E6_semantic"

EVIDENCE_PRIORITY = [
    E0_NATIVE_PDF_TEXT,
    E1_NATIVE_PDF_GEOM,
    E2_DETERMINISTIC,
    E3_CLASSICAL_OCR,
    E4_SPECIALIST_OCR,
    E5_VLM,
    E6_SEMANTIC,
]


@dataclass
class VerificationResult:
    """Outcome of a typed validator on one candidate value.

    Per architecture proposal item #1 (2026-08-13, evidence-weighted
    resolution): the resolver needs per-candidate verification signal, not
    just model confidence. Each typed validator (table / formula / numeric /
    reading-order) returns one of these.

    Attributes
    ----------
    passed:
        True iff the candidate passed ALL validator checks.
    score:
        Continuous quality score in ``[0.0, 1.0]``. ``passed=True`` should
        imply ``score >= 0.5`` but the two fields are independent so a
        validator can express "passes minimally but with low margin".
    validator_id:
        Short identifier of the validator that produced this result
        (e.g. ``"table_structure"``, ``"formula_latex"``, ``"numeric_range"``).
    details:
        Free-form dict for diagnostic info (which checks ran, error messages,
        cell counts, LaTeX parse errors, etc.). Never contains secrets/PII.
    """

    passed: bool
    score: float
    validator_id: str
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"VerificationResult.score must be in [0.0, 1.0], got {self.score}")
        if not self.validator_id:
            raise ValueError("VerificationResult.validator_id must be non-empty")

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "score": self.score,
            "validator_id": self.validator_id,
            "details": self.details,
        }


@dataclass
class Evidence:
    """Single piece of evidence attached to a block.

    Multiple evidences (at different E-levels) can attach to the same block,
    representing the layered recognition pipeline (raw → cleaned → semantic).
    """

    level: str  # one of E0-E6
    source: str  # "pdftotext" | "pymupdf" | "glm-ocr" | "rapidocr" | "regex.fix_latex" | ...
    value: Any  # the actual content (str, float, dict)
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    verification: VerificationResult | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        # asdict recurses; verification is already a dict via its own to_dict
        # only when explicitly constructed. Clean up: if verification is None
        # remove the key from serialized form to keep output stable.
        if self.verification is None:
            d.pop("verification", None)
        return d


# === Block / Page / Document IR ================================== #


@dataclass
class BlockIR:
    """Canonical block representation with multi-level evidence.

    Per DocTrace: each block has provenance + evidence chain.
    Final output value picked by :class:`EvidenceResolver` via :meth:`resolve`.
    """

    id: str  # unique within page, e.g. "page7_block12"
    type: str  # text|title|table|figure|equation|caption|...
    bbox: tuple  # (x0, y0, x1, y1) in PDF points
    page_num: int

    # Multi-level evidence chain. Each E-level has its own value — they
    # DON'T overwrite each other. Final value resolved by EvidenceResolver.
    evidence: list = field(default_factory=list)

    # Cross-references (entities this block mentions, tables it refs, etc.)
    mentions: list = field(default_factory=list)  # list of entity_id
    referenced_by: list = field(default_factory=list)  # list of block_id
    validates_against: list = field(default_factory=list)  # list of validator_id

    def add_evidence(self, ev: Evidence) -> None:
        """Add evidence to chain. Does NOT replace existing."""
        self.evidence.append(ev)

    def resolve(self, resolver: EvidenceResolver | None = None) -> Evidence | None:
        """Pick the winning Evidence using weighted scoring.

        Args:
            resolver: Optional :class:`EvidenceResolver`. If None, uses the
                module-level :data:`DEFAULT_RESOLVER` (priority-fallback
                behavior, equivalent to the old fixed-priority lookup but
                expressed through the resolver interface).

        Returns:
            The winning :class:`Evidence`, or ``None`` if the block has no
            evidence.
        """
        if not self.evidence:
            return None
        if resolver is None:
            from .evidence import DEFAULT_RESOLVER

            resolver = DEFAULT_RESOLVER
        return resolver.resolve(self.evidence)

    def to_dict(self, resolver: EvidenceResolver | None = None) -> dict:
        """Serialize block. Includes resolved value + confidence.

        Args:
            resolver: Optional resolver override (see :meth:`resolve`).
        """
        winner = self.resolve(resolver)
        return {
            "id": self.id,
            "type": self.type,
            "bbox": self.bbox,
            "page_num": self.page_num,
            "evidence": [e.to_dict() for e in self.evidence],
            "mentions": self.mentions,
            "referenced_by": self.referenced_by,
            "resolved_value": winner.value if winner else None,
            "resolved_confidence": winner.confidence if winner else 0.0,
        }


@dataclass
class PageIR:
    """One page: list of blocks in reading order + page-level metadata."""

    page_num: int
    blocks: list = field(default_factory=list)  # list[BlockIR]
    reading_order: list = field(default_factory=list)  # block_id sequence
    page_width: float = 0.0
    page_height: float = 0.0
    rotation: int = 0
    metadata: dict = field(default_factory=dict)

    def add_block(self, block: BlockIR) -> None:
        self.blocks.append(block)
        if block.id not in self.reading_order:
            self.reading_order.append(block.id)

    def to_dict(self) -> dict:
        return {
            "page_num": self.page_num,
            "blocks": [b.to_dict() for b in self.blocks],
            "reading_order": self.reading_order,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "rotation": self.rotation,
            "metadata": self.metadata,
        }


@dataclass
class DocumentIR:
    """Full document IR — list of pages + document-level metadata + entity graph.

    Per DocTrace: evidence graph reasoning requires document-level view.
    Entity graph: entities (Quantity, Mineral, Person, Citation) with
    provenance back to source blocks.
    """

    source_pdf: str
    pages: list = field(default_factory=list)  # list[PageIR]
    entities: dict = field(default_factory=dict)  # entity_id → Entity dict
    domain_assignments: dict = field(default_factory=dict)  # domain → score
    metadata: dict = field(default_factory=dict)
    doc_hash: str = ""

    def __post_init__(self):
        if not self.doc_hash and self.source_pdf:
            try:
                h = hashlib.sha256()
                with open(self.source_pdf, "rb") as f:
                    for chunk in iter(lambda: f.read(1 << 20), b""):
                        h.update(chunk)
                self.doc_hash = h.hexdigest()[:16]
            except Exception:
                self.doc_hash = ""

    def add_page(self, page: PageIR) -> None:
        self.pages.append(page)

    def all_blocks(self):
        """Yield all blocks across all pages."""
        for p in self.pages:
            yield from p.blocks

    def block_by_id(self, block_id: str) -> BlockIR | None:
        for b in self.all_blocks():
            if b.id == block_id:
                return b
        return None

    def add_entity(self, entity_id: str, entity: dict) -> None:
        self.entities[entity_id] = entity

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "doc_hash": self.doc_hash,
            "pages": [p.to_dict() for p in self.pages],
            "entities": self.entities,
            "domain_assignments": self.domain_assignments,
            "metadata": self.metadata,
        }


# === Helper constructors ========================================= #


def make_block(
    page_num: int,
    block_idx: int,
    block_type: str,
    bbox: tuple,
    text: str,
    level: str = E0_NATIVE_PDF_TEXT,
    source: str = "pdftotext",
    confidence: float = 1.0,
) -> BlockIR:
    """Quick constructor for a single-evidence block."""
    block_id = f"page{page_num}_block{block_idx}"
    return BlockIR(
        id=block_id,
        type=block_type,
        bbox=bbox,
        page_num=page_num,
        evidence=[Evidence(level=level, source=source, value=text, confidence=confidence)],
    )


__all__ = [
    # Evidence levels
    "E0_NATIVE_PDF_TEXT",
    "E1_NATIVE_PDF_GEOM",
    "E2_DETERMINISTIC",
    "E3_CLASSICAL_OCR",
    "E4_SPECIALIST_OCR",
    "E5_VLM",
    "E6_SEMANTIC",
    "EVIDENCE_PRIORITY",
    # Data classes
    "VerificationResult",
    "Evidence",
    "BlockIR",
    "PageIR",
    "DocumentIR",
    # Constructors
    "make_block",
]
