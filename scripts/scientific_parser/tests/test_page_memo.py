"""Pins for R19 stage-granular cache keys (run-17 architecture).

Contracts pinned:
  1. page_memo_key is STABLE across a DETECTION_VERSION change (detection
     tier must not invalidate the stochastic OCR memo) and DIFFERENT across
     an OCR_CORE_VERSION change (a stochastic change must invalidate it).
  2. store/lookup roundtrip on a temp cache dir round-trips the entry.
  3. Lookup guards: blank text -> None; stale ocr_core_version -> None.
Integration-lite (slow, fixture-gated) lives in run-17 harness validation.
"""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from scientific_parser.core import extraction_cache as ec  # noqa: E402
import shared as shared  # noqa: E402  (scripts/ on sys.path above)


_ARGS = SimpleNamespace(
    dpi=200,
    max_dim=2000,
    force=False,
    fast=False,
    pages=None,
    crop=None,
    formula_backend="default",
    table_backend="default",
    layout_backend="default",
    reflexive=False,
    cascade=True,
    reflexive_floor=2,
)


def _memo_key():
    return ec.page_memo_key("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS)


def test_memo_key_stable_across_detection_bump():
    k1 = _memo_key()
    old = shared.DETECTION_VERSION
    shared.DETECTION_VERSION = old + ".x"
    try:
        # module attribute read at call time in _cache_key -> OCR_PIPELINE_VERSION
        ec.OCR_PIPELINE_VERSION = f"{shared.OCR_CORE_VERSION}+{shared.DETECTION_VERSION}"
        k2 = _memo_key()
    finally:
        shared.DETECTION_VERSION = old
        ec.OCR_PIPELINE_VERSION = f"{shared.OCR_CORE_VERSION}+{shared.DETECTION_VERSION}"
    assert k1 == k2, "detection bump must NOT invalidate the OCR memo key"


def test_memo_key_differs_across_core_bump():
    k1 = _memo_key()
    old = shared.OCR_CORE_VERSION
    shared.OCR_CORE_VERSION = old + ".x"
    try:
        ec.OCR_CORE_VERSION = shared.OCR_CORE_VERSION
        k2 = _memo_key()
    finally:
        shared.OCR_CORE_VERSION = old
        ec.OCR_CORE_VERSION = old
    assert k1 != k2, "core bump MUST invalidate the OCR memo key"


def test_store_lookup_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "DEFAULT_CACHE_DIR", str(tmp_path))
    entry = {
        "text": "page text from chandra",
        "cands": [{"kind": "table", "rows": [["a", "b"]]}],
        "reflexive_meta": {"model": "chandra-ocr-2", "attempts": 1},
        "route_reasons": ["tounicode_mojibake (20 isolated markers)"],
        "elapsed_s": 41.2,
        "path_warnings": [],
        "model_failure_warnings": [],
    }
    ec.page_memo_store("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS, 3, entry)
    got = ec.page_memo_lookup("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS, 3)
    assert got is not None
    assert got["text"] == entry["text"]
    assert got["reflexive_meta"]["model"] == "chandra-ocr-2"
    # absent page -> None
    assert ec.page_memo_lookup("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS, 4) is None


def test_lookup_guards(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "DEFAULT_CACHE_DIR", str(tmp_path))
    key = ec.page_memo_key("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS)
    d = tmp_path / key
    d.mkdir(parents=True)
    # blank text
    d.joinpath("pages.ocr.json").write_text(
        json.dumps(
            {
                "ocr_core_version": shared.OCR_CORE_VERSION,
                "pages": {"1": {"text": "   ", "route": "ocr"}},
            }
        )
    )
    assert ec.page_memo_lookup("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS, 1) is None
    # stale version
    d.joinpath("pages.ocr.json").write_text(
        json.dumps(
            {
                "ocr_core_version": "2000.01.01.stale",
                "pages": {"1": {"text": "real text", "route": "ocr"}},
            }
        )
    )
    assert ec.page_memo_lookup("/tmp/pin/fake.pdf", ["chandra-ocr-2"], _ARGS, 1) is None
