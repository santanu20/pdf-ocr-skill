"""Pins (2026-09-03): the jsonl-parity fix exposed three pre-existing
over-strips on long_tiny_text (bench run hf-mathguards: 0.8348 -> 0.8235).

Root causes, all measured on real bench fixtures:

1. COLUMN EDGE = CLUSTER MIN (layout_gate.left_edge_column_sharpness):
   the right-cluster minimum is snagged by a centered page heading
   ('MARTIAL LAW' x=166 vs true column edge ~213, 13_pg811) or
   right-column indent lines (188 vs edge 210, 13_pg639). Every
   left-column body line then 'crossed the gutter' and the band guard
   ate the left column. Fix: return the cluster MEDIAN.

2. DEEP-TOP POSITION-ALONE ACCEPTANCE (page_metadata): dense book pages
   (body 6.2pt) start body text at 5.3% H — inside the deep band. The
   body-flow guard explicitly excluded deep lines. Fix: deep-TOP lines
   are rescuable by body-flow evidence (deep-BOTTOM stays excluded);
   len floor 55 -> 50 ('The common form of the book of the ancient
   world was' = 52 chars).

3. MID-LINE CUT COLLATERAL (pdf_ocr _strip_hf_lines cut pass): a
   flagged bottom-band sidebar fragment ('up to 3,000' at 90.9% H,
   20_pg46) fuzzy-matched the SAME phrase inside the mid-page body
   sentence ('More expensive designs will have up to 3,000 individual
   coils…') and the cut deleted the body words, leaving 'will have
   individual coils'. Fix: cut pass only cuts line-EDGE spans.

Contracts:
  - 13_pg639: detector flags ONLY the folio '619'; body-start lines
    ('subsequently in the middle ages…', 'The common form…', 'the
    roll, composed…') are never flagged.
  - 13_pg811: column edge >= 200 (median, not the 166 heading snag);
    'the right of issuing commissions…' and 'Legislative enactments…'
    never flagged.
  - 20_pg46: engine .md keeps exactly ONE 'up to 3,000' (the body
    sentence copy) — the sidebar copy is stripped, the body copy
    survives the cut pass.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[2]
BENCH = SCRIPTS.parent / "bench" / "data" / "bench_data" / "pdfs" / "long_tiny_text"
VENV_PY = SCRIPTS.parent / ".venv" / "bin" / "python"

PG639 = BENCH / "13_pg639_pg1.pdf"
PG811 = BENCH / "13_pg811_pg1.pdf"
PG46 = BENCH / "20_pg46_pg1.pdf"

requires_bench = pytest.mark.skipif(
    not BENCH.exists(), reason="olmOCR-bench pdfs not downloaded"
)


def _detector(pdf: Path):
    from scientific_parser.routing import page_metadata as pm

    return pm._geometry_header_footer_lines(str(pdf), 1) or []


def _page_lines(pdf: Path):
    import fitz  # noqa: PLC0415

    doc = fitz.open(str(pdf))
    try:
        return [
            line["bbox"]
            for block in doc[0].get_text("dict")["blocks"]
            if block.get("type", 0) == 0
            for line in block.get("lines", [])
        ]
    finally:
        doc.close()


@requires_bench
class TestColumnEdgeMedian:
    def test_edge_is_median_not_heading_snag(self):
        from scientific_parser.routing.layout_gate import left_edge_column_sharpness

        _, col2 = left_edge_column_sharpness(_page_lines(PG811))
        # True right-column edge ~213; the old cluster-min returned the
        # 'MARTIAL LAW' heading position 166.2, making every left-column
        # line 'cross the gutter'.
        assert col2 is not None and col2 >= 200, col2


@requires_bench
class TestDenseBookBodyStart:
    """Body starts at 5.3% H on these 6.2pt book pages; position-alone
    acceptance must yield to body-flow evidence."""

    def test_pg639_only_folio_flagged(self):
        hf = _detector(PG639)
        assert "619" in hf, hf
        joined = " | ".join(hf)
        for body in (
            "subsequently in the middle ages",
            "common form of the book",
            "roll, composed of one continuous sheet",
        ):
            assert body not in joined, (body, hf)

    def test_pg811_body_lines_not_flagged(self):
        hf = _detector(PG811)
        joined = " | ".join(hf)
        for body in (
            "issuing commissions in war",
            "Legislative enactments",
        ):
            assert body not in joined, (body, hf)


@requires_bench
class TestMidlineCutGuard:
    """Engine-level pin: the flagged sidebar fragment must not be cut out
    of the body sentence that repeats it mid-line."""

    def test_pg46_body_repetition_survives(self, tmp_path):
        out = tmp_path / "pg46"
        r = subprocess.run(
            [
                str(VENV_PY),
                str(SCRIPTS / "pdf_ocr.py"),
                str(PG46),
                str(out),
                "--pages",
                "1",
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert r.returncode == 0, r.stderr[-2000:]
        md = Path(str(out) + ".md").read_text(encoding="utf-8")
        # Body sentence intact (was 'will have  individual coils').
        assert "up to 3,000 individual coils" in md, md[:500]
        # No cut scar: the mid-line cut left a double-space hole where
        # the flagged fragment was removed from the body sentence.
        # (The magazine sidebar legitimately repeats the phrase inside
        # its own table cells — occurrence count is not the contract.)
        assert "have  individual" not in md
