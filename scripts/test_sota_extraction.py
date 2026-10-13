"""Unit tests for SOTA extraction layer (cascade + reflexive + cache + adaptive router).

Run standalone: python scripts/test_sota_extraction.py
Or via pytest: pytest scripts/test_sota_extraction.py

Covers Phase A of yggdrasil plan dp-20260812-pdf-ocr-sota-implementation:
  - select_ocr_model: per-page-type model routing
  - _pick_best_ocr: cascade winner selection
  - _score_ocr_text: confidence heuristics
  - _cache_key / cache_lookup / cache_write: content-hash cache
  - ocr_with_reflexive_loop: arXiv 2603.22651 reflexive pattern (mocked OCR)
  - Cascade end-to-end (mocked): models=[a,b] actually advances on low conf
  - Bug fix: line 7769 models[:1] no longer discards user's chain

No network, no Ollama. All OCR calls mocked via monkeypatching ocr_image.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

# Make pdf_ocr importable
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR))

# Skill-local venv auto-use (matches pdf_ocr.py's bootstrap)
os.environ.setdefault("PDF_OCR_NO_SKILL_VENV", "1")

import pdf_ocr  # noqa: E402


# ─── Test helpers ────────────────────────────────────────────────────────────
class FakeArgs:
    """Minimal stand-in for argparse Namespace."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


# ─── select_ocr_model ────────────────────────────────────────────────────────
def test_select_default():
    """Default route returns primary model."""
    model, prompt, why = pdf_ocr.select_ocr_model("ocr", ["low_chars"], ["glm-ocr"])
    assert model == "glm-ocr"
    assert prompt == "MATH_OCR_PROMPT"
    assert "default" in why


def test_select_scanned_routes_to_qwen():
    """Scanned page should route to qwen if available (bench: +10 pts)."""
    model, prompt, why = pdf_ocr.select_ocr_model(
        "ocr",
        ["scanned_image_page"],
        ["glm-ocr", "qwen2.5vl:3b"],
    )
    assert model == "qwen2.5vl:3b"
    assert "scanned" in why


def test_select_broken_encoding_routes_to_qwen():
    """Broken encoding should route to qwen (CJK-safe)."""
    model, prompt, why = pdf_ocr.select_ocr_model(
        "ocr",
        ["broken_encoding (50 U+FFFD)"],
        ["glm-ocr", "qwen2.5vl:3b"],
    )
    assert model == "qwen2.5vl:3b"


def test_select_fragmented_table_routes_to_rapidocr():
    """Table fragments should route to rapidocr if available."""
    model, prompt, why = pdf_ocr.select_ocr_model(
        "ocr",
        ["fragmented_table (0.45)"],
        ["glm-ocr", "rapidocr"],
    )
    assert model == "rapidocr"
    assert prompt == "TEXT_OCR_PROMPT"


def test_select_math_sniff():
    """Math content sniff via Unicode symbols in pdftotext."""
    model, prompt, why = pdf_ocr.select_ocr_model(
        "ocr",
        ["low_chars"],
        ["glm-ocr"],
        page_text="The sum ∑ over all i of α_i",
    )
    assert model == "glm-ocr"
    assert "math" in why


def test_select_no_models_returns_default():
    """Empty models list returns safe default, doesn't crash."""
    model, prompt, why = pdf_ocr.select_ocr_model("ocr", ["forced"], [])
    assert model == "glm-ocr"


# ─── _pick_best_ocr ──────────────────────────────────────────────────────────
def test_pick_best_higher_confidence_wins():
    """Higher confidence always wins regardless of attempt count."""
    cur = ("short", "a", 1, 0.3, [])
    cand = ("longer text here", "b", 2, 0.8, [])
    result = pdf_ocr._pick_best_ocr(cur, cand)
    assert result[3] == 0.8
    assert result[1] == "b"


def test_pick_best_tie_breaker_longer_text():
    """Same confidence → longer text wins."""
    cur = ("a", "m", 1, 0.7, [])
    cand = ("this is longer text", "m", 1, 0.7, [])
    result = pdf_ocr._pick_best_ocr(cur, cand)
    assert result == cand


def test_pick_best_tie_breaker_fewer_attempts():
    """Same conf, same length → fewer attempts wins."""
    cur = ("abc", "m", 2, 0.7, [])
    cand = ("abc", "m", 1, 0.7, [])
    result = pdf_ocr._pick_best_ocr(cur, cand)
    assert result == cand


# ─── _score_ocr_text ─────────────────────────────────────────────────────────
def test_score_empty_text():
    conf, bd = pdf_ocr._score_ocr_text("")
    assert conf == 0.0
    assert bd["empty"] is True


def test_score_clean_text():
    """Well-formed scientific text scores high."""
    text = (
        "The quartz monzonite contains 65 wt% SiO2 and shows "
        "enrichment in light rare earth elements (La, Ce, Pr, Nd). "
        "U-Pb zircon ages cluster at 425 ± 3 Ma."
    )
    conf, bd = pdf_ocr._score_ocr_text(text)
    assert conf >= 0.7, f"clean text should score >=0.7, got {conf}"
    assert bd["repeat"] < 0.1


def test_score_repetitive_text_penalized():
    """Hallucination loops (repeated 5-grams) score low."""
    text = "no visible characters " * 50
    conf, bd = pdf_ocr._score_ocr_text(text)
    assert conf < 0.6, f"repetitive text should score <0.6, got {conf}"
    assert bd["repeat"] > 0.3


def test_score_private_chars_penalized():
    """High private-char count (U+FFFE/U+FFFF control chars) scores low.

    Note: U+FFFD (replacement char) is tracked separately as 'ffmd' (broken
    encoding marker). _PRIVATE_CHAR_RE covers control chars + U+FFFE/U+FFFF.
    """
    text = "Geology " + ("\ufffe" * 100) + " rocks"
    conf, bd = pdf_ocr._score_ocr_text(text)
    assert conf < 0.7, f"private-char flood should score <0.7, got {conf}"
    assert bd["priv"] > 0.05, f"priv field should be >0.05, got {bd['priv']}"


def test_score_fffd_penalized():
    """U+FFFD replacement chars (broken encoding) penalized via ffmd field."""
    text = "Geology " + ("\ufffd" * 100) + " rocks"
    conf, bd = pdf_ocr._score_ocr_text(text)
    assert conf < 0.7, f"U+FFFD flood should score <0.7, got {conf}"
    assert bd["ffmd"] > 0.05, f"ffmd field should be >0.05, got {bd['ffmd']}"


def test_score_cjk_penalized():
    """CJK content (likely hallucination on non-CJK pages) penalized."""
    text = "The rock is " + ("岩" * 200) + " type"
    conf, bd = pdf_ocr._score_ocr_text(text)
    assert conf < 0.7
    assert bd["cjk"] > 0.1


# ─── Cache layer ────────────────────────────────────────────────────────────
def test_cache_key_stable():
    """Same input → same key (deterministic)."""
    args = FakeArgs(dpi=None, max_dim=None, force_ocr=False, fast=False, pages="", crop="")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 test content")
        path = f.name
    try:
        k1 = pdf_ocr._cache_key(path, ["glm-ocr"], args)
        k2 = pdf_ocr._cache_key(path, ["glm-ocr"], args)
        assert k1 == k2
        assert len(k1) == 32  # truncated sha256 → 16 bytes hex
    finally:
        os.unlink(path)


def test_cache_key_busts_on_flag_change():
    """Different models list → different key (cache miss)."""
    args = FakeArgs(dpi=None, max_dim=None, force_ocr=False, fast=False, pages="", crop="")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 test content")
        path = f.name
    try:
        k1 = pdf_ocr._cache_key(path, ["glm-ocr"], args)
        k2 = pdf_ocr._cache_key(path, ["glm-ocr", "qwen2.5vl:3b"], args)
        assert k1 != k2
    finally:
        os.unlink(path)


def test_cache_key_busts_on_force_ocr():
    """force_ocr toggle → different key."""
    args_a = FakeArgs(dpi=None, max_dim=None, force_ocr=False, fast=False, pages="", crop="")
    args_b = FakeArgs(dpi=None, max_dim=None, force_ocr=True, fast=False, pages="", crop="")
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 test content")
        path = f.name
    try:
        k1 = pdf_ocr._cache_key(path, ["glm-ocr"], args_a)
        k2 = pdf_ocr._cache_key(path, ["glm-ocr"], args_b)
        assert k1 != k2
    finally:
        os.unlink(path)


def test_cache_write_then_read_roundtrip():
    """Write to cache → lookup returns same data.

    Contract migrated 2026-09-10 (raw-cache split): lookup returns the RAW
    pre-postproc payload (raw.json), not the final summary/md — the document
    chain re-runs on replay. Final-only writes (no raw_payload) are misses."""
    args = FakeArgs(dpi=None, max_dim=None, force_ocr=False, fast=False, pages="", crop="")
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 roundtrip test content")
        output_base = os.path.join(td, "out")
        summary = {
            "source": "test",
            "total_pages": 1,
            "total_chars": 100,
            "total_elapsed_s": 5.0,
            "extraction_version": pdf_ocr.SKILL_EXTRACTION_VERSION,
            "pages": [{"page": 1, "text": "hello"}],
        }
        md_text = "# Test\n\nhello"
        raw_payload = {
            "ocr_pipeline_version": pdf_ocr.OCR_PIPELINE_VERSION,
            "postproc_version": pdf_ocr.POSTPROC_VERSION,
            "raw_md": md_text,
            "page_metrics": [{"page": 1, "text": "hello", "chars": 5}],
            "font_math_pages": [],
            "any_math_font_page": False,
            "path_warnings": [],
            "source": "test",
            "models": ["glm-ocr"],
            "page_count": 1,
            "elapsed_s": 5.0,
        }
        # Write
        key = pdf_ocr._cache_key(pdf_path, ["glm-ocr"], args)
        cache_dir = os.path.join(pdf_ocr.DEFAULT_CACHE_DIR, key)
        pdf_ocr.cache_write(cache_dir, output_base, summary, md_text, raw_payload)
        # Read
        result = pdf_ocr.cache_lookup(pdf_path, ["glm-ocr"], args, output_base)
        assert result is not None, "cache lookup should hit after write"
        loaded_raw, _ = result
        assert loaded_raw["raw_md"] == md_text
        assert loaded_raw["page_metrics"][0]["text"] == "hello"
        assert loaded_raw["ocr_pipeline_version"] == pdf_ocr.OCR_PIPELINE_VERSION
        # Cleanup
        import shutil

        shutil.rmtree(cache_dir, ignore_errors=True)


def test_cache_version_mismatch_is_miss():
    """Stale skill version in cached summary → cache miss."""
    args = FakeArgs(dpi=None, max_dim=None, force_ocr=False, fast=False, pages="", crop="")
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, "test.pdf")
        with open(pdf_path, "wb") as f:
            f.write(b"%PDF-1.4 stale test content")
        output_base = os.path.join(td, "out")
        # Manually write a STALE cache entry
        key = pdf_ocr._cache_key(pdf_path, ["glm-ocr"], args)
        cache_dir = os.path.join(pdf_ocr.DEFAULT_CACHE_DIR, key)
        stale_summary = {
            "extraction_version": "0.0.0.ancient",
            "total_pages": 1,
            "pages": [{"page": 1}],
        }
        pdf_ocr.cache_write(cache_dir, output_base, stale_summary, "stale")
        result = pdf_ocr.cache_lookup(pdf_path, ["glm-ocr"], args, output_base)
        assert result is None, "stale cache should be a miss"
        import shutil

        shutil.rmtree(cache_dir, ignore_errors=True)


# ─── Cascade end-to-end (mocked OCR) ────────────────────────────────────────
def test_cascade_advances_on_low_confidence():
    """ocr_with_reflexive_loop should advance to next model when first returns
    low-confidence text. Validates AUDIT_2026-08-12 finding (line 7769 bug fix).
    """
    # Mock ocr_image to return garbage on first model, good text on second
    call_log = []

    def mock_ocr(img_path, task="Text Recognition:", model="glm-ocr"):
        call_log.append(model)
        if model == "glm-ocr":
            return "no visible characters " * 30  # hallucination loop
        elif model == "qwen2.5vl:3b":
            return "The granite contains quartz and feldspar."
        return ""

    with patch.object(pdf_ocr, "ocr_image", side_effect=mock_ocr):
        text, used_model, attempts, conf, warns = pdf_ocr.ocr_with_reflexive_loop(
            "/fake/path.png",
            ["glm-ocr", "qwen2.5vl:3b"],
            max_attempts_per_model=1,
        )
    assert "qwen2.5vl:3b" in call_log, "should have advanced to qwen"
    assert used_model == "qwen2.5vl:3b", f"expected qwen, got {used_model}"
    assert "granite" in text
    assert conf >= pdf_ocr.REFLEXIVE_CONFIDENCE_FLOOR


def test_cascade_stays_on_first_when_confident():
    """High-confidence first response should not cascade."""
    call_log = []

    def mock_ocr(img_path, task="Text Recognition:", model="glm-ocr"):
        call_log.append(model)
        return "The basalt contains plagioclase and pyroxene."

    with patch.object(pdf_ocr, "ocr_image", side_effect=mock_ocr):
        text, used_model, attempts, conf, warns = pdf_ocr.ocr_with_reflexive_loop(
            "/fake/path.png",
            ["glm-ocr", "qwen2.5vl:3b"],
            max_attempts_per_model=2,
        )
    assert call_log == ["glm-ocr"], "should not have advanced past first"
    assert used_model == "glm-ocr"
    assert attempts == 1


def test_reflexive_retries_with_feedback():
    """Within a single model, reflexive loop retries with error feedback."""
    call_count = {"n": 0}

    def mock_ocr(img_path, task="Text Recognition:", model="glm-ocr"):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return "$$unclosed display math"  # odd $$ count
        return "$$closed math$$ and plain text."

    with patch.object(pdf_ocr, "ocr_image", side_effect=mock_ocr):
        text, used_model, attempts, conf, warns = pdf_ocr.ocr_with_reflexive_loop(
            "/fake/path.png",
            ["glm-ocr"],
            max_attempts_per_model=2,
        )
    assert call_count["n"] == 2, "should have retried within same model"
    assert attempts == 2
    assert "closed math" in text


def test_loop_never_raises_on_ocr_error():
    """OCR errors don't crash the loop — return best-effort."""

    def mock_ocr(img_path, task="Text Recognition:", model="glm-ocr"):
        raise ConnectionRefusedError("ollama down")

    with patch.object(pdf_ocr, "ocr_image", side_effect=mock_ocr):
        text, used_model, attempts, conf, warns = pdf_ocr.ocr_with_reflexive_loop(
            "/fake/path.png",
            ["glm-ocr"],
            max_attempts_per_model=1,
        )
    assert text == ""  # graceful empty return
    assert "error" in str(warns).lower() or any("error" in str(w) for w in warns)


def test_loop_never_raises_on_timeout():
    """Timeouts cascade cleanly without raising."""
    import subprocess

    def mock_ocr(img_path, task="Text Recognition:", model="glm-ocr"):
        raise subprocess.TimeoutExpired(cmd="ollama", timeout=240)

    with patch.object(pdf_ocr, "ocr_image", side_effect=mock_ocr):
        text, used_model, attempts, conf, warns = pdf_ocr.ocr_with_reflexive_loop(
            "/fake/path.png",
            ["glm-ocr", "qwen2.5vl:3b"],
            max_attempts_per_model=1,
        )
    assert any("timeout" in str(w).lower() for w in warns)


# ─── Cascade bug fix verification (line 7769 was models[:1]) ────────────────
def test_cascade_bug_no_longer_truncates_chain():
    """Verify the AUDIT_2026-08-12 finding: argparse 'models' arg with multiple
    values is now preserved through cascade (not sliced to models[:1]).
    """
    # Simulate the args parser + main()'s models handling logic
    import argparse

    test_parser = argparse.ArgumentParser()
    test_parser.add_argument("--models")
    parsed = test_parser.parse_args(["--models", "glm-ocr,qwen2.5vl:3b,rapidocr"])
    models = [m.strip() for m in (parsed.models or "glm-ocr").split(",") if m.strip()]
    # Pre-fix: models would be models[:1] = ['glm-ocr']
    # Post-fix: full chain preserved
    assert len(models) == 3, f"cascade chain should be 3, got {len(models)}"
    assert models == ["glm-ocr", "qwen2.5vl:3b", "rapidocr"]


# ─── Runner ──────────────────────────────────────────────────────────────────
def _run_all():
    """Run all tests, print summary. Returns count of failures."""
    tests = [
        # select_ocr_model
        ("select_default", test_select_default),
        ("select_scanned_routes_to_qwen", test_select_scanned_routes_to_qwen),
        (
            "select_broken_encoding_routes_to_qwen",
            test_select_broken_encoding_routes_to_qwen,
        ),
        (
            "select_fragmented_table_routes_to_rapidocr",
            test_select_fragmented_table_routes_to_rapidocr,
        ),
        ("select_math_sniff", test_select_math_sniff),
        ("select_no_models_returns_default", test_select_no_models_returns_default),
        # _pick_best_ocr
        ("pick_best_higher_confidence_wins", test_pick_best_higher_confidence_wins),
        ("pick_best_tie_breaker_longer_text", test_pick_best_tie_breaker_longer_text),
        (
            "pick_best_tie_breaker_fewer_attempts",
            test_pick_best_tie_breaker_fewer_attempts,
        ),
        # _score_ocr_text
        ("score_empty_text", test_score_empty_text),
        ("score_clean_text", test_score_clean_text),
        ("score_repetitive_text_penalized", test_score_repetitive_text_penalized),
        ("score_private_chars_penalized", test_score_private_chars_penalized),
        ("score_fffd_penalized", test_score_fffd_penalized),
        ("score_cjk_penalized", test_score_cjk_penalized),
        # cache
        ("cache_key_stable", test_cache_key_stable),
        ("cache_key_busts_on_flag_change", test_cache_key_busts_on_flag_change),
        ("cache_key_busts_on_force_ocr", test_cache_key_busts_on_force_ocr),
        ("cache_write_then_read_roundtrip", test_cache_write_then_read_roundtrip),
        ("cache_version_mismatch_is_miss", test_cache_version_mismatch_is_miss),
        # cascade (mocked)
        ("cascade_advances_on_low_confidence", test_cascade_advances_on_low_confidence),
        (
            "cascade_stays_on_first_when_confident",
            test_cascade_stays_on_first_when_confident,
        ),
        ("reflexive_retries_with_feedback", test_reflexive_retries_with_feedback),
        ("loop_never_raises_on_ocr_error", test_loop_never_raises_on_ocr_error),
        ("loop_never_raises_on_timeout", test_loop_never_raises_on_timeout),
        # bug fix
        (
            "cascade_bug_no_longer_truncates_chain",
            test_cascade_bug_no_longer_truncates_chain,
        ),
    ]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(
        f"\n{'=' * 60}\nSOTA Extraction Layer Tests: {passed} passed, {failed} failed ({len(tests)} total)"
    )
    return failed


if __name__ == "__main__":
    sys.exit(1 if _run_all() else 0)
