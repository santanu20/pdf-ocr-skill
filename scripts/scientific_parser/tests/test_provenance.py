"""Unit tests for provenance correctness in build_document_ir.

Regression guard for the 2026-08-13 provenance bug: OCR-derived page text
was labelled E0_native_pdf_text / pdftotext simply because the text string
was non-empty. The real source must drive the evidence level.

Run: pytest scripts/scientific_parser/tests/test_provenance.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.core.ir import (
    E0_NATIVE_PDF_TEXT,
    E3_CLASSICAL_OCR,
    E4_SPECIALIST_OCR,
    E5_VLM,
)
from scientific_parser.skill_adapter import _evidence_for_source, build_document_ir

# === _evidence_for_source: source -> (level, label) ============== #


def test_evidence_native_pdftotext_maps_to_E0():
    level, label = _evidence_for_source("pdftotext", None)
    assert level == E0_NATIVE_PDF_TEXT
    assert label == "pdftotext"


def test_evidence_native_text_layer_alias_maps_to_E0():
    level, label = _evidence_for_source("text_layer", None)
    assert level == E0_NATIVE_PDF_TEXT


def test_evidence_rapidocr_maps_to_E3_classical():
    level, label = _evidence_for_source("ocr", "rapidocr")
    assert level == E3_CLASSICAL_OCR
    assert label == "rapidocr"


def test_evidence_glm_ocr_maps_to_E5_vlm():
    level, label = _evidence_for_source("ocr", "glm-ocr")
    assert level == E5_VLM
    assert label == "glm-ocr"


def test_evidence_vl_suffix_maps_to_E5_vlm():
    level, label = _evidence_for_source("ocr", "qwen2.5vl:3b")
    assert level == E5_VLM


def test_evidence_unknown_model_defaults_to_E4_specialist():
    level, label = _evidence_for_source("ocr", "some-custom-engine")
    assert level == E4_SPECIALIST_OCR


def test_evidence_none_source_defaults_to_ocr_not_native():
    # Regression: must NEVER claim native provenance when source is unknown.
    level, _ = _evidence_for_source(None, None)
    assert level != E0_NATIVE_PDF_TEXT


# === build_document_ir: real source drives evidence level ========= #


def test_build_ir_ocr_page_not_labelled_native():
    """The core regression: OCR text must not be E0_native_pdf_text."""
    doc = build_document_ir(
        "x.pdf",
        ["OCR output text"],
        pages_meta=[{"page": 1, "source": "ocr", "model": "glm-ocr"}],
    )
    block = doc.pages[0].blocks[0]
    assert block.evidence[0].level == E5_VLM
    assert block.evidence[0].source == "glm-ocr"
    assert block.evidence[0].level != E0_NATIVE_PDF_TEXT


def test_build_ir_native_page_labelled_E0():
    doc = build_document_ir(
        "x.pdf",
        ["native text layer content"],
        pages_meta=[{"page": 1, "source": "pdftotext"}],
    )
    block = doc.pages[0].blocks[0]
    assert block.evidence[0].level == E0_NATIVE_PDF_TEXT
    assert block.evidence[0].source == "pdftotext"


def test_build_ir_mixed_sources_per_page():
    doc = build_document_ir(
        "x.pdf",
        ["native", "ocr-glm", "ocr-rapid"],
        pages_meta=[
            {"page": 1, "source": "pdftotext"},
            {"page": 2, "source": "ocr", "model": "glm-ocr"},
            {"page": 3, "source": "ocr", "model": "rapidocr"},
        ],
    )
    levels = [doc.pages[i].blocks[0].evidence[0].level for i in (0, 1, 2)]
    assert levels == [E0_NATIVE_PDF_TEXT, E5_VLM, E3_CLASSICAL_OCR]


def test_build_ir_no_meta_defaults_to_specialist_not_native():
    """Without provenance metadata we must not claim native extraction."""
    doc = build_document_ir("x.pdf", ["some text"], pages_meta=None)
    block = doc.pages[0].blocks[0]
    assert block.evidence[0].level == E4_SPECIALIST_OCR


def test_build_ir_empty_page_has_no_block():
    doc = build_document_ir(
        "x.pdf",
        ["", "text"],
        pages_meta=[{"page": 1, "source": "ocr"}, {"page": 2, "source": "pdftotext"}],
    )
    assert doc.pages[0].blocks == []
    assert len(doc.pages[1].blocks) == 1


# === candidates sidecar emission (item 2) ======================== #


def test_emit_sidecars_writes_candidates_json(tmp_path):
    """emit_sidecars emits .candidates.json when candidates are provided."""
    from scientific_parser.skill_adapter import emit_sidecars

    out_md = tmp_path / "doc.md"
    out_md.write_text("# test")
    cands = [
        {
            "region_id": "p1_r0", "region_type": "equation",
            "bbox": [10, 10, 20, 20], "n_candidates": 1, "agreement": None,
            "winner_source": "formula-provider", "winner_level": "E4_specialist_ocr",
            "candidates": [{"source_engine": "formula-provider",
                            "evidence_level": "E4_specialist_ocr",
                            "confidence": 0.9, "verification": None,
                            "text_len": 9, "crop_bbox": [10, 10, 20, 20],
                            "render_dpi": 150.0}],
        }
    ]
    sidecars = emit_sidecars(
        markdown_text="# test", output_md_path=str(out_md),
        source_pdf="x.pdf", candidates=cands,
    )
    assert "candidates" in sidecars
    import json
    written = json.loads((tmp_path / "doc.candidates.json").read_text())
    assert written[0]["region_id"] == "p1_r0"
    assert written[0]["winner_source"] == "formula-provider"


def test_emit_sidecars_no_candidates_no_sidecar(tmp_path):
    """When candidates is None, no .candidates.json is written."""
    from scientific_parser.skill_adapter import emit_sidecars

    out_md = tmp_path / "doc2.md"
    out_md.write_text("# test")
    sidecars = emit_sidecars(markdown_text="# test", output_md_path=str(out_md))
    assert "candidates" not in sidecars
    assert not (tmp_path / "doc2.candidates.json").exists()
