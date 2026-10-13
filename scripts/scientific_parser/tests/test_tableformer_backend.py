"""det61 pins: TableFormer fallback backend.

Pure-function tests run WITHOUT the tableformer extra (no torch import
at module load — that is itself a pinned contract); orchestrator paths
are tested via monkeypatched internals.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from scientific_parser.perception.tables import tableformer_backend as tb


def _resp(cells, matches):
    return {"tf_responses": cells, "predict_details": {"matches": matches}}


def test_module_import_does_not_pull_torch():
    """Environments without the tableformer extra must stay functional."""
    code = (
        "import sys, json; sys.path.insert(0, %r); "
        "import scientific_parser.perception.tables.tableformer_backend; "
        "print(json.dumps(sorted(m for m in sys.modules "
        "if m.split('.')[0] in ('torch', 'docling_ibm_models', 'numpy'))))"
    ) % str(REPO_ROOT / "scripts")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().endswith("[]"), out.stdout


def test_grid_assembly_plain_and_span():
    cells = [
        {  # exclusive ends: single cell (0,0) -> end (1,1), docling OTSL
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "end_row_offset_idx": 1,
            "end_col_offset_idx": 1,
        },
        {
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 1,
            "end_row_offset_idx": 1,
            "end_col_offset_idx": 2,
        },
        {
            "start_row_offset_idx": 1,
            "start_col_offset_idx": 0,
            "end_row_offset_idx": 3,  # rowspan 2: rows 1..2 under exclusive end
            "end_col_offset_idx": 1,
        },
        {
            "start_row_offset_idx": 1,
            "start_col_offset_idx": 1,
            "end_row_offset_idx": 2,
            "end_col_offset_idx": 2,
        },
    ]

    # tokens_by_id: id -> (x, y, text, bbox); region covers x 0..100
    def _tok(x, y, t):
        return (x, y, t, (x, y, x + 5.0, y + 5.0))

    tokens = {
        0: _tok(0.0, 0.0, "A1"),
        1: _tok(10.0, 0.0, "B1"),
        2: _tok(0.0, 10.0, "spans"),
        3: _tok(50.0, 10.0, "second"),
        4: _tok(10.0, 10.0, "first"),
        5: _tok(500.0, 10.0, "OUTSIDE"),  # center outside region
    }
    matches = {  # STR keys: json round-trip inside TFPredictor (pinned)
        "0": [{"table_cell_id": 0, "iopdf": 1.0}],
        "1": [{"table_cell_id": 1, "iopdf": 1.0}],
        "2": [{"table_cell_id": 2, "iopdf": 0.1}],  # below threshold: dropped
        "3": [{"table_cell_id": 3, "iopdf": 1.0}],
        "4": [{"table_cell_id": 3, "iopdf": 0.9}],
        "5": [{"table_cell_id": 1, "iopdf": 1.0}],  # outside region: dropped
    }
    n_rows, n_cols, grid = tb._grid_from_response(
        _resp(cells, matches), tokens, region=(0.0, 0.0, 100.0, 50.0)
    )
    assert (n_rows, n_cols) == (3, 2)
    assert grid[0] == ["A1", "B1"]  # OUTSIDE never reached cell 1
    assert grid[1][0] == ""  # iopdf 0.1 below min_iopdf
    assert grid[2][0] == ""  # span-covered cell stays empty
    assert grid[1][1] == "first second"  # (y, x) join order


def test_grid_assembly_degenerate_rejected():
    one_cell = [
        {
            "start_row_offset_idx": 0,
            "start_col_offset_idx": 0,
            "end_row_offset_idx": 0,
            "end_col_offset_idx": 0,
        }
    ]
    assert tb._grid_from_response(_resp(one_cell, {}), {}) is None  # 1x1
    empty = _resp([], {})
    assert tb._grid_from_response(empty, {}) is None
    no_coords = _resp([{"start_row_offset_idx": 0}], {})
    assert tb._grid_from_response(no_coords, {}) is None
    blank = _resp(
        [
            {
                "start_row_offset_idx": 0,
                "start_col_offset_idx": 0,
                "end_row_offset_idx": 1,
                "end_col_offset_idx": 0,
            },
            {
                "start_row_offset_idx": 1,
                "start_col_offset_idx": 0,
                "end_row_offset_idx": 1,
                "end_col_offset_idx": 0,
            },
        ],
        {},
    )
    assert tb._grid_from_response(blank, {}) is None  # all cells empty


def test_kill_switch_short_circuits_before_predictor(monkeypatch):
    monkeypatch.setenv("PDF_OCR_TABLEFORMER", "0")

    def _boom():
        raise AssertionError("predictor must not load when disabled")

    monkeypatch.setattr(tb, "_get_predictor", _boom)
    assert tb.tables_for_page("whatever.pdf", 1) == []


def test_fail_open_when_predictor_unavailable(monkeypatch):
    monkeypatch.delenv("PDF_OCR_TABLEFORMER", raising=False)
    monkeypatch.setattr(tb, "_get_predictor", lambda: None)
    assert tb.tables_for_page("whatever.pdf", 1) == []


def test_fail_open_on_orchestrator_error(monkeypatch):
    monkeypatch.delenv("PDF_OCR_TABLEFORMER", raising=False)

    class _FakePred:
        def multi_table_predict(self, *a, **k):
            raise RuntimeError("beam exploded")

    monkeypatch.setattr(tb, "_get_predictor", lambda: _FakePred())

    import types

    fake_fitz = types.ModuleType("fitz")

    class _Doc:
        def __init__(self, *a):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def close(self):
            pass

    fake_fitz.open = _Doc
    monkeypatch.setitem(sys.modules, "fitz", fake_fitz)
    assert tb.tables_for_page("whatever.pdf", 1) == []
