"""R16+R20 pins: img2table last-resort table emission (2026-09-13/14).

R16: fires ONLY when every strategy found nothing (no-table-found
population); measured +5 recoveries (7e850300 x3 + c6673ff6 x2), zero
regressions by construction (empty finals = no arbitration competitors).
R20: scanned pages (all-None df cells) get word-back fill — rapidocr
words assigned to img2table cell bboxes by center containment
(d896f9ae_pg4: 246/370 cells, all 5 gold test cells present).
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))

from scientific_parser.postproc.text_pipeline import detect_all_tables  # noqa: E402

BENCH_DIR = ROOT / "bench/data/bench_data/pdfs/tables"
HIT = BENCH_DIR / "7e850300cc7a63925214ffea6856992511f5_pg7_pg1.pdf"
CONTROL = BENCH_DIR / "fa18a15c1dbbfcb71b1f1ea1b8f116e24b8a_pg2_pg1.pdf"
SCANNED = BENCH_DIR / "d896f9ae5405c8d6b147ae9c9c20a0abb6c4_pg4_pg1.pdf"


def test_lastresort_fires_on_empty_finals():
    if not HIT.exists():
        pytest.skip(f"fixture missing: {HIT.name}")
    tables = detect_all_tables(str(HIT), 1)
    i2t = [t for t in tables if t.get("strategy") == "img2table_lastresort"]
    assert i2t, "last-resort must fire when all strategies find nothing"
    joined = " ".join(c for t in i2t for r in t["rows"] for c in r)
    assert "0.1872" in joined, "chemistry stats values must survive df->rows"


def test_lastresort_does_not_fire_when_engine_finds_tables():
    if not CONTROL.exists():
        pytest.skip(f"fixture missing: {CONTROL.name}")
    tables = detect_all_tables(str(CONTROL), 1)
    assert tables, "control page has real engine tables"
    assert not any(t.get("strategy") == "img2table_lastresort" for t in tables), (
        "last-resort must never displace engine-found tables"
    )


def test_wordback_fills_scanned_cells():
    """R20: scanned page (all-None df) gets rapidocr word-back fill —
    gold test cells ('1.4' under Total DDT, 'Los Angeles' row label)
    must survive into the emitted grid."""
    if not SCANNED.exists():
        pytest.skip(f"fixture missing: {SCANNED.name}")
    tables = detect_all_tables(str(SCANNED), 1)
    i2t = [t for t in tables if t.get("strategy") == "img2table_lastresort"]
    assert i2t, "scanned fixture must fire the last-resort path"
    joined = " ".join(c for t in i2t for r in t["rows"] for c in r)
    assert "1.4" in joined, "gold cell '1.4' (Total DDT column) must be present"
    assert "Los Angeles" in joined, "row label must be word-backed"


def test_wordback_does_not_disturb_df_path():
    """R20: born-digital df path (populated cells) must never trigger the
    word-back branch (<0.2 nonempty heuristic) — values come from the
    text layer, deterministic."""
    if not HIT.exists():
        pytest.skip(f"fixture missing: {HIT.name}")
    tables = detect_all_tables(str(HIT), 1)
    i2t = [t for t in tables if t.get("strategy") == "img2table_lastresort"]
    assert i2t, "df-path fixture still fires last-resort"
    joined = " ".join(c for t in i2t for r in t["rows"] for c in r)
    assert "0.1872" in joined


def test_wordback_graceful_when_ocr_raises(monkeypatch):
    """R20: rapidocr failure inside word-back must degrade to the df rows
    (all-None on scanned) without crashing detect_all_tables."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("rapidocr"):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    if not SCANNED.exists():
        pytest.skip(f"fixture missing: {SCANNED.name}")
    tables = detect_all_tables(str(SCANNED), 1)  # must not raise
    assert isinstance(tables, list)


def test_lastresort_graceful_without_img2table(monkeypatch):
    """Dependency missing -> detect_all_tables returns incumbent result
    (empty list on this fixture), never raises."""
    import builtins

    real_import = builtins.__import__

    def blocked(name, *a, **k):
        if name.startswith("img2table"):
            raise ImportError("blocked for test")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", blocked)
    if not HIT.exists():
        pytest.skip(f"fixture missing: {HIT.name}")
    tables = detect_all_tables(str(HIT), 1)
    assert tables == [], "graceful skip must return the empty incumbent result"
