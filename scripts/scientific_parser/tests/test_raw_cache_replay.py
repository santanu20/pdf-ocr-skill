"""Pins: raw-cache split (2026-09-10, run-7/run-8 RCA).

Run-7 root cause: the OCR cache stored POST-postprocessor output under a
monolithic SKILL_EXTRACTION_VERSION key. Every postproc-only engine bump
invalidated the cache, forcing scanned pages through a fresh stochastic
chandra-ocr-2 roll (measured: tiny 0.8235 -> 0.8009, 14a_pg1 conf 0.70 ->
0.65 + degraded-rapidocr fallback, similarity 0.056 vs prior cached md).

Architecture now: cache stores PRE-chain raw state keyed by
OCR_PIPELINE_VERSION; the document postprocessor chain
(pdf_ocr._run_document_postchain) re-runs on every replay. These pins hold:

1. lookup returns the raw payload (not final files) and its guards hold
   (ocr-version stale, no-text collapse, missing/corrupt = miss).
2. cache_write persists raw.json ONLY when handed a raw_payload — a replay
   rewrite of the finals must never clobber the pristine pre-chain snapshot
   (page_metrics are chain-mutated by then).
3. _finalize_document(raw_replay=True) rebuilds md + jsonl from the replayed
   state and runs the live chain (a postproc change is visible in output
   without any re-extraction).
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

shared = pytest.importorskip("shared")
ec = pytest.importorskip("scientific_parser.core.extraction_cache")
pdf_ocr = pytest.importorskip("pdf_ocr")


class _Args:
    dpi = None
    max_dim = None
    force_ocr = False
    fast = False
    pages = ""
    crop = ""
    formula_backend = "default"
    table_backend = "default"
    layout_backend = "default"
    no_reflexive = False
    no_cascade = False
    reflexive_floor = shared.REFLEXIVE_CONFIDENCE_FLOOR


def _raw_payload():
    return {
        "ocr_pipeline_version": shared.OCR_PIPELINE_VERSION,
        "postproc_version": shared.POSTPROC_VERSION,
        "extraction_version": shared.SKILL_EXTRACTION_VERSION,
        "raw_md": "<!-- meta -->\n\nPAGE ONE TEXT with a \\leftrightarrow cmd",
        "page_metrics": [
            {
                "page": 1,
                "source": "ocr",
                "model": "chandra-ocr-2",
                "chars": 37,
                "elapsed_s": 1.0,
                "route": "ocr",
                "warnings": [],
                "text": "PAGE ONE TEXT with a \\leftrightarrow cmd",
            }
        ],
        "font_math_pages": [],
        "any_math_font_page": False,
        "path_warnings": [],
        "source": "OCR (chandra-ocr-2)",
        "models": ["chandra-ocr-2"],
        "page_count": 1,
        "elapsed_s": 1.0,
    }


def _seed_cache(tmp_path, monkeypatch, payload):
    monkeypatch.setattr(shared, "DEFAULT_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(ec, "DEFAULT_CACHE_DIR", str(tmp_path / "cache"))
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 pin fixture")
    base = tmp_path / "out" / "doc"
    key = ec._cache_key(str(pdf), ["chandra-ocr-2"], _Args)
    cache_dir = Path(tmp_path / "cache" / key)
    ec.cache_write(
        str(cache_dir),
        str(base),
        {"pages": payload["page_metrics"]},
        "FINAL",
        raw_payload=payload,
    )
    return pdf, base, cache_dir


def test__raw_lookup_hit_and_stale_guards(tmp_path, monkeypatch):
    pdf, base, cache_dir = _seed_cache(tmp_path, monkeypatch, _raw_payload())
    hit = ec.cache_lookup(str(pdf), ["chandra-ocr-2"], _Args, str(base))
    assert hit is not None, "valid raw.json must hit"
    raw, cache_dir_rt = hit
    assert raw["raw_md"].startswith("<!-- meta -->"), "lookup returns RAW payload"
    assert cache_dir_rt.endswith(cache_dir.name)

    # base-independence: a DIFFERENT output_base (same PDF + flags) must
    # hit the same raw — the key identifies the doc, not the caller's path
    alt = tmp_path / "elsewhere" / "renamed-output"
    alt_hit = ec.cache_lookup(str(pdf), ["chandra-ocr-2"], _Args, str(alt))
    assert alt_hit is not None and alt_hit[0]["raw_md"] == raw["raw_md"], (
        "raw payload must be caller-path-independent (doc.raw.json)"
    )

    # stale OCR pipeline version -> miss
    raw_file = cache_dir / "doc.raw.json"
    data = json.loads(raw_file.read_text())
    data["ocr_pipeline_version"] = "1999.01.01.ancient"
    raw_file.write_text(json.dumps(data))
    assert ec.cache_lookup(str(pdf), ["chandra-ocr-2"], _Args, str(base)) is None

    # postproc version drift alone -> STILL a hit (the run-7 contract)
    _seed_cache(tmp_path, monkeypatch, _raw_payload())
    raw_file = cache_dir / "doc.raw.json"
    data = json.loads(raw_file.read_text())
    data["postproc_version"] = "1999.01.01.ancient"
    data["extraction_version"] = "1999.01.01.ancient"
    raw_file.write_text(json.dumps(data))
    assert ec.cache_lookup(str(pdf), ["chandra-ocr-2"], _Args, str(base)) is not None

    # no-text collapse -> miss (retry next run)
    _seed_cache(tmp_path, monkeypatch, _raw_payload())
    data = json.loads(raw_file.read_text())
    for p in data["page_metrics"]:
        p["text"] = "[No text recognized]"
    raw_file.write_text(json.dumps(data))
    assert ec.cache_lookup(str(pdf), ["chandra-ocr-2"], _Args, str(base)) is None


def test__cache_write_never_clobbers_raw_without_payload(tmp_path, monkeypatch):
    pdf, base, cache_dir = _seed_cache(tmp_path, monkeypatch, _raw_payload())
    raw_file = cache_dir / "doc.raw.json"
    before = raw_file.read_text()
    # replay-style rewrite: finals only, no raw_payload
    ec.cache_write(str(cache_dir), str(base), {"pages": []}, "FINAL-v2")
    assert raw_file.read_text() == before, (
        "raw replay must not overwrite raw.json — page_metrics are "
        "chain-mutated by replay time; the snapshot must stay pre-chain"
    )
    # fresh-style write WITH payload rewrites it
    payload = _raw_payload()
    payload["elapsed_s"] = 2.5
    ec.cache_write(
        str(cache_dir),
        str(base),
        {"pages": payload["page_metrics"]},
        "FINAL-v3",
        raw_payload=payload,
    )
    assert json.loads(raw_file.read_text())["elapsed_s"] == 2.5


def test__finalize_document_raw_replay__rebuilds_outputs_runs_chain(tmp_path, monkeypatch):
    """Raw replay must apply the LIVE postproc chain to cached raw state —
    a postproc-only engine change shows up in output with zero extraction.
    Proven via the cmdsplit whitelist: raw contains a bare \\leftrightarrow
    that the OLD chain split into '\\left rightarrow'; the live chain must
    leave it intact."""
    payload = _raw_payload()
    md_path = tmp_path / "out.md"
    jsonl_path = tmp_path / "out.pages.jsonl"
    summary_path = tmp_path / "out.summary.json"
    raw_file = tmp_path / "doc.raw.json"
    raw_file.write_text(json.dumps(payload))
    _extract_calls = []

    def _explode(*a, **k):  # any re-extraction on replay = contract breach
        _extract_calls.append(a)
        raise AssertionError("raw replay must not re-extract")

    monkeypatch.setattr(pdf_ocr, "cache_lookup", _explode)
    monkeypatch.setattr(pdf_ocr, "cache_write", lambda *a, **k: None)
    pdf_ocr._finalize_document(
        {
            "final_md": payload["raw_md"],
            "page_metrics": json.loads(json.dumps(payload["page_metrics"])),
            "page_count": payload["page_count"],
            "models": payload["models"],
            "args": _Args(),
            "pdf_path": str(tmp_path / "doc.pdf"),
            "md_path": str(md_path),
            "jsonl_path": str(jsonl_path),
            "summary_path": str(summary_path),
            "json_mode": True,
            "source_provenance": payload["source"],
            "total_elapsed": payload["elapsed_s"],
            "font_math_pages": set(payload["font_math_pages"]),
            "any_math_font_page": payload["any_math_font_page"],
            "use_cache": True,
            "cache_dir": str(tmp_path),
            "all_region_candidates": None,
            "path_warnings_override": payload["path_warnings"],
            "raw_replay": True,
            "output_base": "out",
        }
    )
    assert not _extract_calls
    out = md_path.read_text()
    assert r"\leftrightarrow" in out, (
        "live chain must apply: cmdsplit whitelist leaves \\leftrightarrow intact"
    )
    recs = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(recs) == 1 and recs[0]["page"] == 1, "jsonl rebuilt from replay state"
    summary = json.loads(summary_path.read_text())
    assert summary["source"] == payload["source"]
    assert summary["path_warnings"] == []
    assert raw_file.read_text() == json.dumps(payload), "replay never touches raw.json"


def test__composite_version_shape():
    assert shared.SKILL_EXTRACTION_VERSION == (
        f"{shared.OCR_PIPELINE_VERSION}+{shared.POSTPROC_VERSION}"
    ), "composite provenance string must stay in lockstep with its parts"
