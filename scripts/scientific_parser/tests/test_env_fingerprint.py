"""Pins: env fingerprint must discriminate environments without flapping (P0.3).

Era-delta root cause (run-51 class, 42 unattributed flips): cache keys encoded
engine versions only, so dependency updates or ollama model re-pulls silently
replayed stale raw output under an unchanged version string. These pins hold:
  1. compute_fingerprint is deterministic and reacts to EVERY component
  2. the memoized wrapper is stable in-process (no key flap between PDFs)
  3. ollama outages reuse the persisted registry snapshot (no outage flap)
  4. the cache KEY itself changes when the fingerprint changes, for BOTH the
     raw key and the page-memo key (core-tier override path)
  5. the fingerprint is byte-stable across fresh interpreters on this machine
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

pytest.importorskip("shared")
ef = pytest.importorskip("scientific_parser.core.env_fingerprint")
ec = pytest.importorskip("scientific_parser.core.extraction_cache")


def test__compute_fingerprint__deterministic_and_component_sensitive():
    kw = dict(
        uv_lock="aaaa",
        python_tag="py3.13",
        pdftotext_tag="pdftotext version 24.02.0",
        model_digests="glm-ocr=abc123",
    )
    base = ef.compute_fingerprint(["glm-ocr"], **kw)
    assert base == ef.compute_fingerprint(["glm-ocr"], **kw)
    for comp in ("uv_lock", "python_tag", "pdftotext_tag", "model_digests"):
        drifted = dict(kw, **{comp: kw[comp] + "x"})
        assert base != ef.compute_fingerprint(["glm-ocr"], **drifted), (
            f"fingerprint blind to {comp} — stale replays across env changes"
        )


def test__env_fingerprint__memoized_stable_and_order_insensitive():
    a = ef.env_fingerprint(["glm-ocr", "chandra-ocr-2"])
    b = ef.env_fingerprint(("chandra-ocr-2", "glm-ocr"))  # order-insensitive
    c = ef.env_fingerprint(None)  # empty chain is its own stable value
    assert a == b
    assert a != c or not a  # distinct chain -> distinct fp (empty != non-empty)
    assert ef.env_fingerprint(["glm-ocr", "chandra-ocr-2"]) == a  # memoized


def test__model_digests__reuses_persisted_registry_when_ollama_down(tmp_path, monkeypatch):
    reg_path = tmp_path / "model_registry.json"
    reg_path.write_text(json.dumps({"models": {"glm-ocr:latest": "d1" * 32}}))
    monkeypatch.setattr(ef, "_fetch_registry", lambda *a, **k: None)
    got = ef._model_digests(["glm-ocr"], registry_path=str(reg_path))
    assert got == f"glm-ocr={('d1' * 32)[:16]}", (
        "ollama outage must fall back to the persisted snapshot, not flap to absent"
    )


def test__model_digests__absent_and_alias_matching(tmp_path, monkeypatch):
    reg_path = tmp_path / "model_registry.json"
    monkeypatch.setattr(ef, "_fetch_registry", lambda *a, **k: None)
    # no snapshot at all + registry unreachable -> absent, never raises
    got = ef._model_digests(["ghost-model"], registry_path=str(reg_path))
    assert got == "ghost-model=absent"
    # alias: request "glm-ocr", registry keys "glm-ocr:latest"
    reg_path.write_text(json.dumps({"models": {"glm-ocr:latest": "ab" * 32}}))
    got = ef._model_digests(["glm-ocr"], registry_path=str(reg_path))
    assert got == f"glm-ocr={('ab' * 32)[:16]}"


def test__model_digests__persists_live_registry(tmp_path, monkeypatch):
    reg_path = tmp_path / "model_registry.json"
    monkeypatch.setattr(ef, "_fetch_registry", lambda *a, **k: {"m:latest": "cd" * 32})
    first = ef._model_digests(["m"], registry_path=str(reg_path))
    assert first == f"m={('cd' * 32)[:16]}"
    assert json.loads(reg_path.read_text())["models"] == {"m:latest": "cd" * 32}
    # subsequent outage serves from the fresh snapshot
    monkeypatch.setattr(ef, "_fetch_registry", lambda *a, **k: None)
    assert ef._model_digests(["m"], registry_path=str(reg_path)) == first


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
    reflexive_floor = 0.5


def _bench_pdf():
    data = REPO_ROOT / "bench" / "data" / "bench_data" / "pdfs"
    return next(data.glob("*/*.pdf"), None)


def test__cache_key__env_sensitive_both_tiers(monkeypatch):
    pdf = _bench_pdf()
    if pdf is None:
        pytest.skip("no bench PDF available for key material")
    orig = ec._envfp.env_fingerprint
    try:
        monkeypatch.setattr(ec._envfp, "env_fingerprint", lambda m: "env1-fixed")
        key_raw = ec._cache_key(str(pdf), ["glm-ocr"], _Args)
        key_memo = ec.page_memo_key(str(pdf), ["glm-ocr"], _Args)
        monkeypatch.setattr(ec._envfp, "env_fingerprint", lambda m: "env1-changed")
        key_raw2 = ec._cache_key(str(pdf), ["glm-ocr"], _Args)
        key_memo2 = ec.page_memo_key(str(pdf), ["glm-ocr"], _Args)
    finally:
        monkeypatch.setattr(ec._envfp, "env_fingerprint", orig)
        ec._cache_key.cache_clear() if hasattr(ec._cache_key, "cache_clear") else None
    assert key_raw != key_raw2, "raw cache key blind to env change — stale replay across envs"
    assert key_memo != key_memo2, "page-memo key blind to env change — stale OCR replay"


def test__env_fingerprint__cross_interpreter_byte_stable():
    """Era-delta mini-repro: two fresh interpreters, same machine, same env
    -> byte-identical fingerprint (else every batch worker would key differently)."""
    code = (
        "import sys; sys.path.insert(0, %r); "
        "from scientific_parser.core.env_fingerprint import env_fingerprint; "
        "print(env_fingerprint(('glm-ocr',)))"
    ) % str(REPO_ROOT / "scripts")
    outs = [
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=120
        ).stdout.strip()
        for _ in range(2)
    ]
    assert outs[0] and outs[0] == outs[1], f"fingerprint not stable across interpreters: {outs}"
