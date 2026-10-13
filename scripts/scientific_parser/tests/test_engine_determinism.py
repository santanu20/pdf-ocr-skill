"""Pin (2026-09-09): engine extraction must be run-to-run deterministic and
cache-replay must equal fresh extraction under the same version key.

Measured on olmOCR-bench tiny/tables (session-2 run 1): flip-diff vs the
session-1 floor ledger showed tiny +2/-5 and tables -2 at UNCHANGED engine
code. A/B evidence (10a/10b/15c_pg1 + d896f9ae..._pg4): fresh --no-cache
x2 text identical, cached replay identical, scorer (evaluate_candidate x2)
deterministic — the ledger itself was recorded from a mixed cache state
(engine feature commits landed between version bumps). The durable guards
for that class of bug are: (1) this determinism pin, (2) cache-replay
fidelity pin, (3) the process rule that SKILL_EXTRACTION_VERSION bumps
with every output-affecting engine edit.

Contract:
  1. Two fresh --no-cache extractions of the same fixture emit identical
     page text, route, and route_reasons (elapsed/confidence may vary).
  2. A cache-enabled run that misses (extracts + stores) and a second
     cache-enabled run that hits (replays) emit identical page text.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2]
ENGINE = SCRIPTS / "pdf_ocr.py"
VENV_PY = SCRIPTS.parent / ".venv" / "bin" / "python"

BODY = "Determinism pin body prose line number three with plain words."


def _make_fixture(path: Path) -> None:
    import fitz  # noqa: PLC0415

    doc = fitz.open()
    page = doc.new_page(width=595, height=794)
    for i in range(10):
        page.insert_text(
            (72, 200 + 14 * i),
            f"{BODY} Sentence {i} carries ordinary prose vocabulary.",
            fontsize=9,
        )
    doc.save(str(path))
    doc.close()


def _run(pdf: Path, out_prefix: Path, cache_dir: Path, no_cache: bool) -> dict:
    env = dict(
        os.environ,
        PDF_OCR_NO_SKILL_VENV="1",
        PDF_OCR_CACHE_DIR=str(cache_dir),
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        NUMEXPR_NUM_THREADS="2",
        OMP_WAIT_POLICY="passive",
    )
    cmd = [str(VENV_PY), str(ENGINE), str(pdf), str(out_prefix), "--json"]
    if no_cache:
        cmd.append("--no-cache")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=240, env=env)
    assert proc.returncode == 0, f"engine rc={proc.returncode}: {proc.stderr[-800:]}"
    jsonl = out_prefix.with_name(out_prefix.name + ".pages.jsonl")
    assert jsonl.exists(), "no .pages.jsonl emitted"
    recs = [json.loads(l) for l in jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert recs, "zero page records"
    return {
        "text": [r.get("text", "") for r in recs],
        "route": [r.get("route") for r in recs],
        "route_reasons": [r.get("route_reasons") for r in recs],
    }


def test_fresh_extraction_deterministic() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        pdf = tdp / "detpin.pdf"
        _make_fixture(pdf)
        a = _run(pdf, tdp / "a", tdp / "cache", no_cache=True)
        b = _run(pdf, tdp / "b", tdp / "cache", no_cache=True)
    assert a == b, "fresh --no-cache extractions differ (engine nondeterminism)"


def test_cache_replay_equals_fresh() -> None:
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        pdf = tdp / "detpin.pdf"
        cache = tdp / "cache"
        _make_fixture(pdf)
        miss = _run(pdf, tdp / "miss", cache, no_cache=False)  # extract + store
        hit = _run(pdf, tdp / "hit", cache, no_cache=False)  # replay
    assert miss["text"] == hit["text"], "cache replay text != fresh extraction text"


if __name__ == "__main__":
    test_fresh_extraction_deterministic()
    test_cache_replay_equals_fresh()
    print("determinism pins: OK")
