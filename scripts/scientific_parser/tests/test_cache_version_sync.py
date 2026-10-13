"""Pin: extraction_cache must not shadow shared.py constants (2026-09-09 RCA).

Run-6 zero-effect root cause: extraction_cache.py redeclared
SKILL_EXTRACTION_VERSION as a local literal frozen at
"2026.09.05.mathml-unwrap". The star-import from shared.py was shadowed, so
every later shared.py version bump was silently excluded from the cache key
and the stale-guard — PDFs extracted in the frozen era replayed stale engine
output on every subsequent run (fresh mtimes, old content). This pin fails if
any shared.py constant is ever redeclared in extraction_cache with a drifting
value. PASS->FAIL on this test means the cache is again version-blind.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

shared = pytest.importorskip("shared")
ec = pytest.importorskip("scientific_parser.core.extraction_cache")


def test__cache_version_single_source_of_truth__matches_shared():
    assert ec.SKILL_EXTRACTION_VERSION == shared.SKILL_EXTRACTION_VERSION, (
        "extraction_cache shadows shared.SKILL_EXTRACTION_VERSION "
        f"({ec.SKILL_EXTRACTION_VERSION!r} != {shared.SKILL_EXTRACTION_VERSION!r}) "
        "— cache key would be version-blind; delete the local literal"
    )


def test__cache_constants_single_source_of_truth__match_shared():
    for name in (
        "REFLEXIVE_CONFIDENCE_FLOOR",
        "REFLEXIVE_MAX_ATTEMPTS_PER_MODEL",
        "OCR_PER_ATTEMPT_TIMEOUT_S",
        "OCR_PIPELINE_VERSION",
        "POSTPROC_VERSION",
    ):
        ec_val = getattr(ec, name, None)
        sh_val = getattr(shared, name, None)
        assert ec_val is not None and ec_val == sh_val, (
            f"extraction_cache.{name} ({ec_val!r}) drifted from shared.{name} ({sh_val!r})"
        )


def test__cache_key_version_sensitive__key_changes_on_ocr_pipeline_bump():
    """Contract migrated 2026-09-10 (raw-cache split): the cache dir is
    keyed on OCR_PIPELINE_VERSION (extraction semantics). A composite
    SKILL_EXTRACTION_VERSION bump alone must NOT invalidate the cache —
    postproc-only changes re-run the chain on raw replay instead."""

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
        reflexive_floor = ec.REFLEXIVE_CONFIDENCE_FLOOR

    pdf = next(
        (REPO_ROOT / "bench" / "data" / "bench_data" / "pdfs").glob("*/*.pdf"),
        None,
    )
    if pdf is None:  # no bench data installed — exercise the hash-free path
        pytest.skip("no bench PDF available for key material")
    key_new = ec._cache_key(str(pdf), [], _Args)
    orig_ocr = ec.OCR_PIPELINE_VERSION
    orig_pp = ec.POSTPROC_VERSION
    try:
        ec.OCR_PIPELINE_VERSION = orig_ocr + ".pin-bump"
        key_ocr_bumped = ec._cache_key(str(pdf), [], _Args)
        ec.OCR_PIPELINE_VERSION = orig_ocr
        ec.POSTPROC_VERSION = orig_pp + ".pin-bump"
        ec.SKILL_EXTRACTION_VERSION = f"{orig_ocr}+{orig_pp}.pin-bump"
        key_pp_only = ec._cache_key(str(pdf), [], _Args)
    finally:
        ec.OCR_PIPELINE_VERSION = orig_ocr
        ec.POSTPROC_VERSION = orig_pp
        ec.SKILL_EXTRACTION_VERSION = f"{orig_ocr}+{orig_pp}"
    assert key_new != key_ocr_bumped, (
        "cache key no longer reacts to OCR_PIPELINE_VERSION — stale raw "
        "entries would replay across extraction-semantics changes"
    )
    assert key_new == key_pp_only, (
        "cache key must NOT react to POSTPROC_VERSION — postproc-only "
        "bumps must replay raw through the chain, never re-extract "
        "(run-7 RCA: postproc bump re-rolled stochastic OCR)"
    )
