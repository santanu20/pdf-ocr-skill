"""Tests for HTML block protection in run_unified_postproc.

Specialist providers (RapidTable) emit structured <table> HTML. Post-proc
cleaners target OCR/pdftotext artifacts and must not flatten that structure
back to loose cell text. _protect_html_blocks / _restore_html_blocks guard
<table>...</table> blocks across the cleaner registry.

Run: pytest scripts/scientific_parser/tests/test_html_protection.py -v
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scientific_parser.postproc.text_pipeline import (
    _protect_html_blocks,
    _restore_html_blocks,
    run_unified_postproc,
)


# === protect / restore roundtrip ================================== #


def test_roundtrip_no_tables_is_identity():
    t = "just plain text, no html here"
    protected, blocks = _protect_html_blocks(t)
    assert blocks == []
    assert _restore_html_blocks(protected, blocks) == t


def test_roundtrip_single_table_preserved():
    t = "before\n<table><tr><td>96.04%</td></tr></table>\nafter"
    protected, blocks = _protect_html_blocks(t)
    assert len(blocks) == 1
    assert "<table" not in protected  # table extracted
    assert _restore_html_blocks(protected, blocks) == t


def test_roundtrip_multiple_tables_preserved():
    t = "<table><tr><td>A</td></tr></table>\nmiddle\n<table><tr><td>B</td></tr></table>"
    protected, blocks = _protect_html_blocks(t)
    assert len(blocks) == 2
    assert _restore_html_blocks(protected, blocks) == t


def test_table_with_attributes_protected():
    t = '<table class="x"><tr><td colspan="2">data</td></tr></table>'
    protected, blocks = _protect_html_blocks(t)
    assert len(blocks) == 1
    assert _restore_html_blocks(protected, blocks) == t


# === run_unified_postproc preserves tables ======================== #


def _ctx():
    return {
        "pdf_path": "",
        "page": 1,
        "page_meta": {"page_header": None, "page_number": None},
        "tmp_dir": "/tmp",
        "model": "rapidocr",
        "ocr_text": "",
        "doc_state": {},
        "doc_info": {},
    }


def test_postproc_preserves_table_html():
    sample = (
        "Intro paragraph.\n\n"
        "<table><tr><td>Model</td><td>AUCROC</td></tr>"
        "<tr><td>X</td><td>96.04%</td></tr></table>\n\n"
        "Outro paragraph."
    )
    out, _ = run_unified_postproc(sample, "fast", "universal", _ctx())
    assert "<table" in out
    assert "</table>" in out
    assert "96.04%" in out


def test_postproc_preserves_multiple_tables():
    sample = "<table><tr><td>T1</td></tr></table>\ntext\n<table><tr><td>T2</td></tr></table>"
    out, _ = run_unified_postproc(sample, "fast", "universal", _ctx())
    assert out.count("<table") == 2
    assert out.count("</table>") == 2
