"""Pin: whitespace-align row-merge collision guard (run-10a, 2026-09-10).

Root cause pinned: on compact tables (fa18a15c pg2, h_med=9pt, data-row
pitch 9-10pt) the pure y-distance merge test (dYc <= 1.2*h_med = 10.8pt)
fused EVERY adjacent data row into 2-row logical rows -> cells glued
('2 68' where gold has '2' | '68'). The x-collision guard refuses merges
when the rows' words collide in x-bands (parallel data rows); true
stacked sub-lines are x-complementary and still merge.

Fallback contract pinned (v2, run-11): the public wrapper judges each
table's GRID, not the page table count. Tables whose legacy carving is
pathology-free (no glued-numeric/ragged) pass through untouched — same-
column 2-line stacks (00e980a0 pg1, 1a67d111 pg1) and all seven run-10
regression pages keep the legacy carving; the guarded carve is adopted
only where the legacy grid is pathological AND the iter-58 conservation
gates hold (fa18a15c pg2).

Run: pytest scripts/scientific_parser/tests/test_ws_row_guard.py -v
"""

import re
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent  # -> scripts/
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

BENCH_DIR = SCRIPTS_DIR.parent / "bench" / "data" / "bench_data" / "pdfs" / "tables"

pytestmark = pytest.mark.skipif(
    not BENCH_DIR.exists(),
    reason="bench dataset not downloaded",
)

from scientific_parser.perception.tables.detector import (  # noqa: E402
    _ws_impl,
    detect_tables_whitespace,
)

FA18 = BENCH_DIR / "fa18a15c1dbbfcb71b1f1ea1b8f116e24b8a_pg2_pg1.pdf"
STACKED = [
    BENCH_DIR / "00e980a0c4645fc83f27c467d10bbdeb8661_pg64.pdf",
    BENCH_DIR / "1a67d1114deeff1a4ad4a6bee9422afebbb1_pg1_pg1.pdf",
]

# soil-moisture table region on fa18a15c pg2 (word-geometry census 2026-09-10)
_SOIL_Y0, _SOIL_Y1 = 295.0, 480.0
_TWO_NUM_RE = re.compile(r"^\s*\d+(?:\.\d+)?\s+\d+(?:\.\d+)?\s*$")


def _glued_cells(table):
    return [c for row in table["rows"] for c in row if _TWO_NUM_RE.match(str(c))]


def test_guard_splits_fused_data_rows():
    """Guarded pass keeps compact-table data rows separate: no '2 68' glued
    numeric cells anywhere in the soil-table region."""
    for pdf in (FA18,):
        if not pdf.exists():
            pytest.skip(f"fixture missing: {pdf.name}")
        guarded = _ws_impl(str(pdf), 1, guard=True)
        in_band = [t for t in guarded if t["bbox"][1] <= _SOIL_Y1 and t["bbox"][3] >= _SOIL_Y0]
        assert in_band, "soil table region must be detected (guarded)"
        for t in in_band:
            assert not _glued_cells(t), f"glued numeric cells survived guard: {_glued_cells(t)[:4]}"


def test_legacy_fuses_but_v2_adopts_gated():
    """Legacy merge produces the fused pathology on fa18a15c; v2's public
    wrapper adopts the guarded carve there (pathology + conservation gates
    hold) — no glued numeric cells in the soil-table region."""
    if not FA18.exists():
        pytest.skip(f"fixture missing: {FA18.name}")
    legacy = _ws_impl(str(FA18), 1, guard=False)
    fused = [
        c
        for t in legacy
        for c in [x for row in t["rows"] for x in row]
        if _TWO_NUM_RE.match(str(c))
    ]
    assert fused, "legacy path should show the pinned fused-cell pathology"
    public = detect_tables_whitespace(str(FA18), 1)
    for t in public:
        assert not _glued_cells(t) or t["bbox"][1] > _SOIL_Y1 or t["bbox"][3] < _SOIL_Y0


# Run-10's 17 PASS->FAIL regressions: clean stacked-subline pages whose
# grids the count-tie wrapper kept guard-shredded. v2 must return the
# LEGACY carving verbatim wherever the legacy grid shows no
# glued-numeric/ragged pathology (floor behavior preserved).
RUN10_REGRESSED = [
    BENCH_DIR / "0b100ba02aedbdb366a79ac387b4d6bf1774_pg4.pdf",
    BENCH_DIR / "1651cbb49c05c51fb5a2687130144f94865e_pg2.pdf",
    BENCH_DIR / "2e3b2e7d37fe3ad2c9a541a1546f5b9576b7_pg24.pdf",
    BENCH_DIR / "53d3f304da5d1943dd4f650937f7cdbfda90_pg3.pdf",
    BENCH_DIR / "83f40aac04a692b7aacbd813cf4da85354c8_pg26_pg1.pdf",
    BENCH_DIR / "a45ce384707785ce86e301682e1bae0e34b5_pg7_pg1.pdf",
    BENCH_DIR / "a70c1fc81feccfd878dabffd3dc5617b584c_pg5.pdf",
]


def test_clean_pages_keep_legacy_grids():
    """Pages whose legacy whitespace grids are pathology-free must come
    through the public wrapper byte-identical to the legacy carving."""
    for pdf in RUN10_REGRESSED:
        if not pdf.exists():
            pytest.skip(f"fixture missing: {pdf.name}")
        legacy = _ws_impl(str(pdf), 1, guard=False)
        public = detect_tables_whitespace(str(pdf), 1)
        assert len(public) == len(legacy), (
            f"{pdf.name}: wrapper dropped/added tables ({len(public)} vs {len(legacy)})"
        )
        for pub, leg in zip(public, legacy):
            assert pub["rows"] == leg["rows"], (
                f"{pdf.name}: clean page's grid diverged from legacy carving"
            )


def test_stacked_subline_pages_fall_back():
    """Same-column 2-line-stack tables must still emit: wrapper falls back
    to legacy rows where the guard drops the table (00e980a0, 1a67d111)."""
    for pdf in STACKED:
        if not pdf.exists():
            pytest.skip(f"fixture missing: {pdf.name}")
        tables = detect_tables_whitespace(str(pdf), 1)
        assert tables, f"stacked-subline table lost on {pdf.name}"
