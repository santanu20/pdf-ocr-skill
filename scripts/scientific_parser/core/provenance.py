"""Provenance — source tracking for every extracted artifact.

Per DocTrace (arXiv 2608.03292): "explicit evidence provenance."
Each extracted entity/block carries a Provenance record pointing back to:
  - source PDF
  - page number
  - bbox
  - extraction engine (pdftotext, glm-ocr, rapidocr, regex, ...)
  - extraction timestamp
  - transformation chain (which normalizers touched it)
"""

from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Provenance:
    """Source-of-record for one artifact."""

    source_pdf: str = ""
    doc_hash: str = ""
    page: Optional[int] = None
    bbox: Optional[tuple] = None
    engine: str = ""  # "pdftotext" | "glm-ocr" | "rapidocr" | "regex" | ...
    extraction_time: float = field(default_factory=time.time)
    transformation_chain: list = field(default_factory=list)  # list of op names

    def add_transform(self, op_name: str) -> None:
        self.transformation_chain.append(op_name)

    def to_dict(self) -> dict:
        return {
            "source_pdf": self.source_pdf,
            "doc_hash": self.doc_hash,
            "page": self.page,
            "bbox": list(self.bbox) if self.bbox else None,
            "engine": self.engine,
            "extraction_time": self.extraction_time,
            "transformation_chain": self.transformation_chain,
        }


def make_provenance(
    source_pdf: str,
    page: int,
    engine: str,
    bbox: Optional[tuple] = None,
    doc_hash: str = "",
) -> Provenance:
    return Provenance(
        source_pdf=source_pdf, doc_hash=doc_hash, page=page, engine=engine, bbox=bbox
    )


__all__ = ["Provenance", "make_provenance"]
