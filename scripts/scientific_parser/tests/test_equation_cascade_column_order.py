"""det50 pins: column-aware cascade assembly (two-column math pages).

Root cause pinned: collect_math_clusters groups spans by baseline y only,
so two-column pages fuse left+right spans per visual row and the final
global y-sort interleaves the columns ("BRM consists of 2 linear feedback
guess the initial state..." on 0853d8a0 pg2). det50 adds gutter splitting
(_split_cluster_at_gutter, page-gated) and column-major emission order
(_column_aware_emitted_order). Single-column pages stay byte-identical
(plain y-sort fallback).

Run: pytest scripts/scientific_parser/tests/test_equation_cascade_column_order.py -v
"""

import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent  # -> scripts/
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

SKILL_ROOT = SCRIPTS_DIR.parent  # -> pdf-ocr/
BENCH_DIR = SKILL_ROOT / "bench" / "data" / "bench_data" / "pdfs"
TWO_COL_PDF = BENCH_DIR / "multi_column" / "0853d8a05fd1e839887d4bc3004a6770cb7b_page_2_pg1.pdf"

pytestmark = pytest.mark.skipif(
    not BENCH_DIR.exists(),
    reason="bench dataset not downloaded",
)

from scientific_parser.science.equation_cascade import (  # noqa: E402
    _column_aware_emitted_order,
    _split_cluster_at_gutter,
)


def _span(x0, x1, y=100.0, text="s"):
    return {"bbox": (x0, y, x1, y + 10), "origin": (x0, y + 8), "text": text}


class TestDet50SplitClusterAtGutter:
    def test_straddling_gap_splits_into_two_clusters(self):
        # page_width 612 -> mid 306; gap 200->320 = 120pt straddles mid.
        cl = {
            "spans": [_span(50, 200), _span(320, 560)],
            "bbox": (50, 100, 560, 110),
        }
        parts = _split_cluster_at_gutter(cl, 612.0)
        assert len(parts) == 2
        assert parts[0]["bbox"][2] == 200  # left cluster ends at left col
        assert parts[1]["bbox"][0] == 320  # right cluster starts at right col
        assert len(parts[0]["spans"]) == 1 and len(parts[1]["spans"]) == 1

    def test_wide_gap_not_straddling_mid_stays_fused(self):
        # gap 30pt (>=6) but both spans sit left of mid: no split. This is
        # the single-column sparse-equation shape (wide inter-token gaps).
        cl = {
            "spans": [_span(50, 200), _span(230, 260)],
            "bbox": (50, 100, 260, 110),
        }
        parts = _split_cluster_at_gutter(cl, 612.0)
        assert len(parts) == 1
        assert parts[0] is cl or parts[0]["spans"] == cl["spans"]


class TestDet50ColumnAwareEmittedOrder:
    def _layout(self):
        # emitted[i][0] = y0 of emission i; exts[i] = (x0, x1).
        emitted = [
            (10.0, "title"),  # 0 full-width
            (100.0, "L1"),  # 1 left
            (120.0, "L2"),  # 2 left
            (140.0, "L3"),  # 3 left
            (100.0, "R1"),  # 4 right
            (120.0, "R2"),  # 5 right
            (140.0, "R3"),  # 6 right
            (170.0, "divider"),  # 7 full-width
            (200.0, "L4"),  # 8 left
            (200.0, "R4"),  # 9 right
        ]
        exts = [
            (50, 550),  # 0 crosses
            (50, 280),  # 1
            (50, 280),  # 2
            (50, 280),  # 3
            (320, 550),  # 4
            (320, 550),  # 5
            (320, 550),  # 6
            (50, 550),  # 7 crosses
            (50, 280),  # 8
            (320, 550),  # 9
        ]
        return emitted, exts

    def test_left_column_reads_before_right_within_bands(self):
        emitted, exts = self._layout()
        order = _column_aware_emitted_order(emitted, exts, 600.0)
        names = [emitted[i][1] for i in order]
        assert names == [
            "title",
            "L1",
            "L2",
            "L3",
            "R1",
            "R2",
            "R3",
            "divider",
            "L4",
            "R4",
        ]

    def test_single_column_falls_back_to_ysort(self):
        # Only one right-side emission -> gate closed -> plain y-sort.
        emitted = [(100.0, "a"), (90.0, "b"), (110.0, "c")]
        exts = [(50, 280), (50, 280), (320, 550)]
        order = _column_aware_emitted_order(emitted, exts, 600.0)
        assert order == [1, 0, 2]  # sorted by y0


class TestDet50CascadeTwoColIntegration:
    def test_0853_pg2_no_column_interleave(self):
        import fitz

        from scientific_parser.science.equation_cascade import (
            extract_text_with_cascade,
        )

        if not TWO_COL_PDF.exists():
            pytest.skip(f"fixture missing: {TWO_COL_PDF}")

        doc = fitz.open(str(TWO_COL_PDF))
        try:
            out = extract_text_with_cascade(doc[0])
        finally:
            doc.close()

        # Clean left-column flow survives end to end.
        assert "BRM consists of 2 linear feedback" in out
        # Fused row-major interleave signatures must be gone.
        assert "feedback guess the initial" not in out
        assert "pseudorandom the corresponding" not in out


class TestDet51ConservationAuditsParts:
    """det51: conservation_pass must audit the split PARTS, not the fused
    clusters — auditing the fused cluster re-injects the other column's
    spans into a part's line (2503.04147 pg3 equation corruption)."""

    PDF = BENCH_DIR / "arxiv_math" / "2503.04147_pg3.pdf"

    def test_04147_pg3_split_lines_not_cross_fused(self):
        import fitz

        from scientific_parser.science.equation_cascade import (
            extract_text_with_cascade,
        )

        if not self.PDF.exists():
            pytest.skip(f"fixture missing: {self.PDF}")

        doc = fitz.open(str(self.PDF))
        try:
            out = extract_text_with_cascade(doc[0])
        finally:
            doc.close()

        # The left-column inline equation survives intact (was the det50
        # P2F: right-column z/C/x spans re-injected into its math).
        flat = " ".join(out.split())
        assert "\\mathcal{N} = \\{ 1 , \\cdots , N \\}" in flat
        # Cross-column fusion signatures must be absent.
        assert "and \\bm{z}}" not in out
        assert "N \\times \\times" not in out


class TestDet55ScriptXGapReassignment:
    """det55 pins: superscript spans rejoin their owning line.

    Root cause pinned: a superscript's y-center rides ABOVE its line's
    baseline band, so band clustering chains it to the PREVIOUS line's
    cluster (y-center nearest; the owning line may not exist yet in
    sort order) and its ^{...} markup is lost -- 2503.02004 pg9 emitted
    "$\\mathcal{O} (2 \\cdot K )$" for O(2^M . K). det55 Stage 1.5 moves
    a script-size span to the same-pitch cluster offering a tight x-gap
    (owning line has spans ending just left / resuming just right),
    vetoed unless the script's origin sits within ~half a body size of
    the candidate's baseline (04147 display-eq scripts must not be
    stolen by the dense line below).
    """

    PDF_02004 = BENCH_DIR / "arxiv_math" / "2503.02004_pg9.pdf"
    PDF_04147 = BENCH_DIR / "arxiv_math" / "2503.04147_pg3.pdf"

    def test_superscript_rejoins_own_line(self):
        import fitz

        from scientific_parser.science.equation_cascade import (
            extract_text_with_cascade,
        )

        if not self.PDF_02004.exists():
            pytest.skip(f"fixture missing: {self.PDF_02004}")

        doc = fitz.open(str(self.PDF_02004))
        try:
            out = extract_text_with_cascade(doc[0])
        finally:
            doc.close()

        # The M span is bound to the equation line as a superscript.
        assert "2^{M}" in out
        # The orphaned-adjacent-prose signature must be gone.
        assert "(2   \\cdot K" not in " ".join(out.split())

    def test_display_scripts_not_stolen_across_pitch(self):
        import fitz

        from scientific_parser.science.equation_cascade import (
            extract_text_with_cascade,
        )

        if not self.PDF_04147.exists():
            pytest.skip(f"fixture missing: {self.PDF_04147}")

        doc = fitz.open(str(self.PDF_04147))
        try:
            out = extract_text_with_cascade(doc[0])
        finally:
            doc.close()

        flat = " ".join(out.split())
        # Display equation tokens stay on their own line.
        assert "J_{N, 1}" in flat or "J_{N,1}" in flat
        # The '2 \\pi ( i - j )' numerator is assembled into the frac
        # rather than orphaned as a separate line (Stage 3 pairing can
        # only see it once spans sit in the right cluster).
        assert "\\frac{2 \\pi ( i - j )}{N - 1}" in flat
