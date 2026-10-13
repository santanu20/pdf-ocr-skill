"""SOTA cache + reflexive OCR loop (arXiv 2603.22651 pattern).

Extracted from scripts/pdf_ocr.py during Phase R6 refactoring (2026-08-12).

Contains:
  - Content-hash cache: _cache_key, cache_lookup, cache_write
  - Reflexive loop: ocr_with_reflexive_loop (extract -> verify -> retry)
  - Confidence scoring: _pick_best_ocr, _score_ocr_text
  - LaTeX verifier: _verify_laTeX_quality
  - Constants: REFLEXIVE_CONFIDENCE_FLOOR, SKILL_EXTRACTION_VERSION, etc.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import time

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state

# P0.3 (2026-10-10): environment component of the cache key — uv.lock hash,
# python minor, poppler version, ollama model digests. Imported as a module
# attribute (not star) so tests can monkeypatch the binding on this module.
from . import env_fingerprint as _envfp


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


_CONSTS_LOADED = False


def _load_consts():
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    for n in (
        "_PRIVATE_CHAR_RE",
        "_OCR_LATEX_CMD_CONFUSIONS",
        "MATH_FONT_PREFIXES",
        "UNICODE_TO_LATEX",
        "DEPS",
        "MATH_OCR_PROMPT",
        "ocr_image",
        "fix_latex_math_output",
        "strip_thinking_artifacts",
        "MAX_DIM",
    ):
        if hasattr(p, n):
            g[n] = getattr(p, n)
    _CONSTS_LOADED = True


# REFLEXIVE_CONFIDENCE_FLOOR / REFLEXIVE_MAX_ATTEMPTS_PER_MODEL /
# OCR_PER_ATTEMPT_TIMEOUT_S / SKILL_EXTRACTION_VERSION: supplied by the
# shared.py star-import above. Local literals here SHADOWED the imported
# values — the frozen "2026.09.05.mathml-unwrap" version literal silently
# excluded every later shared.py bump from the cache key, so PDFs extracted
# in that era replayed stale engine output forever (run-6 zero-effect RCA,
# 2026-09-09). Do NOT redeclare shared.py constants in this module.

# DEFAULT_CACHE_DIR is canonical in shared.py (star-imported above) —
# do not redefine here (2026-08-16 audit M2).


def _cache_key(pdf_path: str, models, args_ns, version: str | None = None) -> str:
    """Compute content-addressed cache key for a PDF + flag set.

    Includes: file bytes sha256, skill extraction version, models list,
    dpi, max-dim, force-ocr, fast, pages range. Any change in any of these
    produces a different key (cache miss) — guarantees correctness on flag
    change or skill code change.

    No hidden global state: all flags contribute
    explicitly. No implicit env-var dependence except PDF_OCR_CACHE_DIR
    (location only, not semantics).

    ``version`` (R19, 2026-09-13): override the v= component. The raw-cache
    key uses the full OCR_PIPELINE_VERSION (stochastic+deterministic
    composite); the per-page OCR memo key passes OCR_CORE_VERSION only so
    detection-tier bumps keep memo keys stable (warm replay, zero re-roll).
    """
    _load_consts()  # lazy-load constants
    v = version or OCR_PIPELINE_VERSION
    try:
        h = hashlib.sha256()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        file_hash = h.hexdigest()
    except OSError:
        # Image input or unreadable — fallback to path + mtime hash.
        try:
            st = os.stat(pdf_path)
            file_hash = hashlib.sha256(
                f"{pdf_path}|{st.st_size}|{st.st_mtime_ns}".encode()
            ).hexdigest()
        except OSError:
            file_hash = "000000"  # last resort — cache still works, just weaker

    flags = "|".join(
        [
            # v keys the cache dir on the OCR PIPELINE only (raw extraction
            # semantics). Postproc-chain changes bump POSTPROC_VERSION, which
            # is NOT part of the key — the chain re-runs on replay, so its
            # changes apply without re-extracting (2026-09-10 raw-cache split).
            f"v={v}",
            # P0.3 (2026-10-10): env fingerprint (uv.lock + python minor +
            # poppler + ollama model digests). Keyed OUTSIDE v= so it applies
            # to both the raw key (pipeline version) and the page-memo key
            # (core version): both tiers are environment-sensitive. Known
            # cost, pre-registered: every pre-P0.3 cache entry misses once
            # (one-time migration roll, run-17 precedent).
            f"env={_envfp.env_fingerprint(tuple(models))}",
            f"models={','.join(models)}",
            f"dpi={getattr(args_ns, 'dpi', None)}",
            f"max_dim={getattr(args_ns, 'max_dim', None)}",
            f"force={int(getattr(args_ns, 'force_ocr', False))}",
            f"fast={int(getattr(args_ns, 'fast', False))}",
            f"pages={getattr(args_ns, 'pages', '')}",
            f"crop={getattr(args_ns, 'crop', '')}",
            # Specialist backend flags — MUST be in cache key. Toggling
            # --formula-backend between "default" and "rapidlatex" changes
            # output; cache key must reflect this (else stale-cache returns
            # wrong non-specialist output for a specialist-flagged run).
            f"formula_backend={getattr(args_ns, 'formula_backend', 'default')}",
            f"table_backend={getattr(args_ns, 'table_backend', 'default')}",
            f"layout_backend={getattr(args_ns, 'layout_backend', 'default')}",
            f"reflexive={int(not getattr(args_ns, 'no_reflexive', False))}",
            f"cascade={int(not getattr(args_ns, 'no_cascade', False))}",
            f"reflexive_floor={getattr(args_ns, 'reflexive_floor', REFLEXIVE_CONFIDENCE_FLOOR)}",
        ]
    )
    return hashlib.sha256(f"{file_hash}|{flags}".encode()).hexdigest()[:32]


def cache_lookup(pdf_path: str, models, args_ns, output_base: str):
    """Return the cached RAW extraction payload on hit, else None.

    Cache files live at ~/.cache/pdf-ocr/<key>/doc.raw.json (raw
    pre-postprocessor state). The cache dir is keyed on the OCR pipeline +
    file hash + flags ONLY (see _cache_key) — the key already uniquely
    identifies the PDF, so the raw payload is stored under a
    CALLER-INDEPENDENT name ("doc.raw.json"): any invocation, any output
    path, hits the same raw. Postproc-chain changes bump POSTPROC_VERSION,
    which is NOT part of the key — the chain re-runs on replay, so its
    changes apply without re-extracting (2026-09-10 raw-cache split).

    Legacy <base>.summary.json / <base>.md final files are no longer read
    here (the chain always re-runs); they remain written for diagnostics,
    named per output_base as before.

    Reads atomically — partial writes (mid-process crash) appear as misses.
    """
    _load_consts()  # lazy-load constants
    key = _cache_key(pdf_path, models, args_ns)
    cache_dir = os.path.join(DEFAULT_CACHE_DIR, key)
    raw_path = os.path.join(cache_dir, "doc.raw.json")
    if not os.path.exists(raw_path):
        return None
    try:
        with open(raw_path, encoding="utf-8") as f:
            raw = json.load(f)
        if not raw.get("page_metrics") or not raw.get("raw_md"):
            return None
        # NO-TEXT COLLAPSE GUARD (2026-09-05): a transient OCR failure
        # ('[No text recognized]' marker as every page's text, measured
        # 15c_pg1 + d896f9ae_pg4_pg1 under ollama load) must not be
        # replayed forever as a valid extraction — full-collapse docs are
        # always cache misses so the next run retries them.
        _no_text = ("[No text recognized]", "[No text on page]")
        if all((p.get("text") or "").strip() in _no_text for p in raw["page_metrics"]):
            return None
        # Stale guard: OCR pipeline mismatch = miss (forces regeneration).
        if raw.get("ocr_pipeline_version") != OCR_PIPELINE_VERSION:
            return None
        return raw, cache_dir
    except (OSError, json.JSONDecodeError):
        return None


def page_memo_key(pdf_path: str, models, args_ns) -> str:
    """Core-tier-only cache key for per-page OCR memos (R19, 2026-09-13).

    Same inputs as _cache_key except v=OCR_CORE_VERSION (stochastic tier
    only: routing signals, OCR models/prompts, preprocessing, cascade,
    VL fallbacks). Detection-tier bumps (tables/layout/ordering/assembly)
    keep this key STABLE — routed pages replay their stochastic OCR product
    from pages.ocr.json instead of re-rolling chandra/cascade, and the
    deterministic chain re-runs on the memoized text. Run-16 lesson:
    monolithic bumps re-rolled every chandra page bench-wide, sigma
    +-5-10 tests/cat noise swamping small real gains (-15 roll vs +12 real).
    """
    _load_consts()  # lazy-load constants
    return _cache_key(pdf_path, models, args_ns, version=OCR_CORE_VERSION)


def page_memo_lookup(pdf_path: str, models, args_ns, page_num: int):
    """Return the memoized stochastic OCR product for ONE routed page, or None.

    Payload dict: {text, cands, reflexive_meta, route_reasons, elapsed_s,
    path_warnings, model_failure_warnings} — the page state at the
    post-dispatch seam (after reflexive OCR + region dispatch, before the
    deterministic per-page chain). Guards: missing/blank text = miss;
    core-version mismatch = miss. Any read/parse failure = miss (cache is
    throughput optimization, never correctness — H1).
    """
    _load_consts()  # lazy-load constants
    key = page_memo_key(pdf_path, models, args_ns)
    path = os.path.join(DEFAULT_CACHE_DIR, key, "pages.ocr.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
        if doc.get("ocr_core_version") != OCR_CORE_VERSION:
            return None
        entry = doc.get("pages", {}).get(str(page_num))
        if not entry or not (entry.get("text") or "").strip():
            return None
        return entry
    except (OSError, json.JSONDecodeError):
        return None


def page_memo_store(pdf_path: str, models, args_ns, page_num: int, entry: dict) -> None:
    """Merge one page's stochastic product into pages.ocr.json (atomic).

    One JSON per PDF at <cache>/<core_key>/pages.ocr.json: {ocr_core_version,
    pages: {"<n>": entry}}. Read-merge-write + atomic rename; a colliding
    writer (same PDF re-extracted concurrently) loses at most the OTHER
    page keys it didn't see — per-page merge keeps the common case (harness
    workers on DISTINCT PDFs = distinct keys) contention-free. Failures
    logged + swallowed: memo loss only costs a re-roll, never correctness.
    """
    _load_consts()  # lazy-load constants
    try:
        key = page_memo_key(pdf_path, models, args_ns)
        cache_dir = os.path.join(DEFAULT_CACHE_DIR, key)
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, "pages.ocr.json")
        doc = {"ocr_core_version": OCR_CORE_VERSION, "pages": {}}
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    prev = json.load(f)
                if prev.get("ocr_core_version") == OCR_CORE_VERSION:
                    doc["pages"] = prev.get("pages", {})
            except (OSError, json.JSONDecodeError):
                pass  # corrupt/absent — start fresh
        doc["pages"][str(page_num)] = entry
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as e:
        log.warning("page memo store failed (%s): %s", page_num, e)


def cache_write(
    cache_dir: str,
    output_base: str,
    summary: dict,
    md_text: str,
    raw_payload: dict | None = None,
) -> None:
    """Persist summary.json + .md (+ raw.json when raw_payload given)
    atomically to cache dir.

    raw_payload carries the PRE-postprocessor state (raw_md, page_metrics,
    chain gates) captured before the document chain mutated anything. It is
    written ONLY on fresh extraction — a raw-replay run re-runs the chain
    and rewrites the final files but must NOT overwrite raw.json with
    post-chain state (page_metrics are chain-mutated).

    Atomic via temp-file + rename (POSIX atomic-rename guarantee).
    Failures are logged + swallowed — cache is throughput optimization,
    not a correctness path (H1: never block on cache).
    """
    _load_consts()  # lazy-load constants
    try:
        os.makedirs(cache_dir, exist_ok=True)
        # Write-side twin of the lookup no-text guard: don't persist a
        # full-collapse extraction at all.
        _no_text = ("[No text recognized]", "[No text on page]")
        pages = (raw_payload or {}).get("page_metrics") or summary.get("pages") or []
        if pages and all((p.get("text") or "").strip() in _no_text for p in pages):
            log.warning("cache write skipped: extraction is a no-text collapse")
            return
        base = os.path.basename(output_base)
        items = [
            (base + ".summary.json", json.dumps(summary, ensure_ascii=False, indent=2)),
            (base + ".md", md_text),
        ]
        if raw_payload is not None:
            # Base-independent name: the key fully identifies the PDF +
            # flags, so ANY invocation (any output path) must find the raw
            # payload — not just the one that originally wrote it.
            items.append(("doc.raw.json", json.dumps(raw_payload, ensure_ascii=False, indent=2)))
        for name, content in items:
            final = os.path.join(cache_dir, name)
            tmp = final + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(content)
            os.replace(tmp, final)  # atomic on POSIX + Windows (Py3.3+)
    except OSError as e:
        log.warning("cache write failed (non-fatal, cache is optimization): %s", e)


# Domain routing was REMOVED 2026-08-12. SOTA design: every processor is
# self-gating (each proc checks its own content signature before applying).
# No --domain flag, no detect_domain(), no DOMAIN_*_OK sets, no field hints.
# Universal parser — works correctly for any science field out of the box.
# Audit: validate_ree_sequence requires 3+ REE in monotonic cluster;
# detect_mineral_abbreviations requires rock/mineral context; etc.


# =============================================================================
# SOTA Unified Post-Processor Registry (2026-08-12)
# =============================================================================
# Robust pipeline architecture: declarative registry → domain-aware dispatcher.
# Replaces previously duplicated 25-step chains at lines 8540+ (fast) and
# 8730+ (OCR). Each processor self-declares its domain + source applicability.
# The dispatcher iterates the registry, gating by domain + source — eliminating
# 4-callsite bandaid patches for geo/chem gating.
#
# Architecture properties:
#   - DRY: single source of truth (was ~150 LoC duplicated between fast/OCR)
#   - OCP: new processors added as a new PostProcessor entry, no existing code touched
#   - Domain-aware: 'universal' procs always run; geo/chem gated by detect_domain()
#   - Source-aware: pre-cleanup differs (fast=text-encoding fix; OCR=thinking-strip)
#   - Failure-isolated: per-processor try/except — one crash never blocks pipeline
#   - Order-stable: explicit `order` field, sorted iteration


def _verify_laTeX_quality(text):
    """Run pykatex verifier on extracted text. Returns (cleaned_text, warnings).

    Wraps fix_latex_math_output in defensive try/except — verifier must never
    block extraction (H1: fail fast on real errors, but verifier failure =
    fallback to unverified text, not crash).
    """
    _load_consts()  # lazy-load constants
    try:
        cleaned = _pdf_ocr().fix_latex_math_output(text)
        warnings = []
        # Detect residual LaTeX pathology that fix_latex_math_output couldn't repair
        # (these become feedback for the reflexive retry prompt).
        if cleaned.count("$$") % 2 != 0:
            warnings.append("odd count of $$ delimiters (unclosed display math)")
        # Detect nested $ inside $$ ... $$ (common glm-ocr failure)
        nested = re.search(r"\$\$[^$]*\$[^$]*\$\$", cleaned)
        if nested:
            warnings.append("nested $ inside $$ block")
        # Detect \\sort, \\alf, etc. (misrendered commands fix_latex couldn't guess)
        bad_cmd = re.findall(r"\\(sort|alf|bet|gama|alpa|dlta|Sig|Sqrt)\b", cleaned)
        if bad_cmd:
            warnings.append(f"invalid LaTeX commands: {','.join(bad_cmd[:5])}")
        return cleaned, warnings
    except Exception as e:
        log.warning("pykatex verifier crashed (non-fatal): %s", e)
        return text, [f"verifier_crashed: {e}"]


def ocr_with_reflexive_loop(
    img_path,
    model_chain,
    page_text_hint="",
    max_attempts_per_model=REFLEXIVE_MAX_ATTEMPTS_PER_MODEL,
    timeout_per_attempt=OCR_PER_ATTEMPT_TIMEOUT_S,
    confidence_floor=REFLEXIVE_CONFIDENCE_FLOOR,
    pdf_path=None,
    page_num=None,
    region_bbox=None,
    current_dpi=None,
):
    """Reflexive extract → verify → retry loop (arXiv 2603.22651 pattern).

    For each model in `model_chain`, try up to `max_attempts_per_model`:
      1. Extract via ocr_image (with MATH_OCR_PROMPT + feedback if any)
      2. Apply post-OCR cleanup (collapse repetition, strip thinking)
      3. Verify with pykatex (_verify_laTeX_quality) → warnings
      4. Score with _compute_page_confidence → confidence
      5. If confidence >= floor OR no attempts left: return
      6. Else: classify failure → execute corrective action → re-OCR with
         the new artifact (tighter crop / higher-DPI render) when the
         context allows it; otherwise feed warnings into next attempt's
         prompt as feedback.

    Cascade: if all attempts of model A fail to reach floor, advance to model B.
    Returns on first (text, model, attempts, confidence, warnings) tuple where
    either confidence >= floor OR entire chain exhausted.

    Args:
        img_path: Path to the rendered page PNG (or region crop PNG).
        model_chain: List of OCR model names to try in cascade order.
        page_text_hint: Optional text-layer content as a hint for the model.
        max_attempts_per_model: Hard cap on attempts per model.
        timeout_per_attempt: Subprocess timeout per OCR call (seconds).
        confidence_floor: Minimum confidence to accept an attempt.
        pdf_path: Optional source PDF path — enables HIGHER_DPI action.
        page_num: Optional 1-indexed page number — enables HIGHER_DPI action.
        region_bbox: Optional (x0,y0,x1,y1) bbox of the region in img_path
            render-pixel coordinates — enables TIGHTER_CROP action.
        current_dpi: Optional DPI at which img_path was rendered — informs
            HIGHER_DPI target (defaults to current + 150).

    Guarantees:
      - Hard cap on total OCR calls = len(chain) * max_attempts_per_model
        (+ ≤1 extra per attempt when TIGHTER_CROP / HIGHER_DPI fires)
      - Never raises — on any error, returns best-effort text + low confidence
      - Always returns text (possibly empty) — never fails the page
    """
    _load_consts()  # lazy-load MATH_OCR_PROMPT, ocr_image, etc.
    feedback = ""
    best = ("", model_chain[0] if model_chain else "glm-ocr", 0, 0.0, ["no_model"])
    total_attempts = 0
    _model_texts = {}  # track each model's best text for disagreement check
    # Active OCR target: starts as img_path, may be swapped to a tighter crop
    # or higher-DPI re-render by the retry executor mid-loop.
    active_img_path = img_path

    for model in model_chain:
        # Full-page base prompt: paddleocr-vl rambling-hallucinates with
        # MATH_OCR_PROMPT on non-math pages (measured: invented formulas on
        # a handwritten letter). Region crops (region_bbox set) keep the
        # math prompt — they ARE math regions.
        if "paddleocr" in model.lower() and region_bbox is None:
            _base_prompt = "OCR:"
        else:
            _base_prompt = _pdf_ocr().MATH_OCR_PROMPT
        for attempt in range(1, max_attempts_per_model + 1):
            total_attempts += 1
            try:
                # Compose prompt: base prompt + feedback from prior attempt
                prompt = _base_prompt
                if feedback:
                    # Add concrete error feedback (arXiv 2603.22651 §3.2)
                    feedback_block = (
                        f"\n\n--- PRIOR ATTEMPT ERRORS (fix these) ---\n{feedback}\n"
                        "--- END ERRORS. Re-transcribe the image carefully. ---"
                    )
                    prompt = _base_prompt + feedback_block

                # Extract (subprocess call to Ollama). Time-bounded.
                # Uses `active_img_path` which may be a tighter crop or
                # higher-DPI re-render produced by the retry executor below.
                start = time.time()
                raw_text = _pdf_ocr().ocr_image(active_img_path, task=prompt, model=model)
                elapsed = time.time() - start

                if not raw_text or len(raw_text.strip()) < 2:
                    feedback = "previous attempt returned empty text"
                    best = _pick_best_ocr(best, (raw_text, model, attempt, 0.0, ["empty_output"]))
                    continue

                # Light cleanup (full post-proc happens in main flow after loop)
                text = _pdf_ocr().strip_thinking_artifacts(raw_text)
                text = re.sub(r"^\s*[\-=_:*]{2,}\s*$", "", text, flags=re.MULTILINE)

                # Verify + score
                verified, verify_warnings = _verify_laTeX_quality(text)
                confidence, _ = _score_ocr_text(verified)
                _model_texts[model] = verified  # track for disagreement

                best = _pick_best_ocr(
                    best,
                    (verified, model, attempt, confidence, verify_warnings, elapsed),
                )

                # Success gate: confidence >= floor AND no STRUCTURAL LaTeX
                # warnings. Structural issues (odd $$ count, nested $ inside $$)
                # break downstream rendering — must retry even if text-quality
                # score is otherwise high (e.g. "$$unclosed" scores well by
                # text-quality but is unrenderable).
                structural_warning = any(
                    ("odd count" in w) or ("nested" in w) for w in verify_warnings
                )
                if confidence >= confidence_floor and not structural_warning:
                    return best[:5]  # success — return immediately

                # Build feedback for next attempt — include both kinds of issues
                if verify_warnings:
                    feedback = "; ".join(verify_warnings)
                    # Classify the failure → look up corrective action → execute.
                    # Per architecture proposal item #5 (2026-08-13): targeted
                    # retries (tighter crop, higher DPI) recover accuracy that
                    # generic "rerun same model" misses. The executor swaps
                    # `active_img_path` when it produces a useful artifact.
                    try:
                        from ..core.ir import VerificationResult as _VR
                        from ..verification.retry_actions import (
                            action_for as _action_for,
                        )
                        from ..verification.retry_actions import (
                            classify_failure as _classify,
                        )
                        from ..verification.retry_executor import (
                            execute_action as _execute_action,
                        )

                        _vr = _VR(
                            passed=False,
                            score=float(confidence),
                            validator_id="formula_latex"
                            if any("katex" in w.lower() or "$" in w for w in verify_warnings)
                            else "generic_text",
                            details={"warnings": list(verify_warnings)},
                        )
                        _fc = _classify(_vr)
                        _ctx = {
                            "img_path": active_img_path,
                            "region_bbox": region_bbox,
                            "pdf_path": pdf_path,
                            "page_num": page_num,
                            "current_dpi": current_dpi,
                            "model": model,
                            "ocr_fn": lambda img, mdl: _pdf_ocr().ocr_image(
                                img, task=_pdf_ocr().MATH_OCR_PROMPT, model=mdl
                            ),
                        }
                        _act = _action_for(_fc, context=_ctx)
                        _outcome = _execute_action(_act, context=_ctx)
                        log.info(
                            "retry_action: failure_class=%s action=%s "
                            "executed=%s model=%s attempt=%d conf=%.2f msg=%s",
                            _fc.value,
                            _act.action.value,
                            _outcome.success,
                            model,
                            attempt,
                            confidence,
                            _outcome.message,
                        )
                        # If executor produced a new artifact, use it next iter
                        if _outcome.success and _outcome.output_path:
                            active_img_path = _outcome.output_path
                            feedback += (
                                f" | retry_executor swapped image → "
                                f"{os.path.basename(_outcome.output_path)}"
                            )
                    except Exception as _classify_err:
                        log.debug("retry_classifier skipped: %s", _classify_err)
                else:
                    feedback = f"low confidence ({confidence:.2f} < {confidence_floor})"
            except subprocess.TimeoutExpired:
                log.warning(
                    "OCR timeout (model=%s attempt=%d/%d, %ds) — cascading",
                    model,
                    attempt,
                    max_attempts_per_model,
                    timeout_per_attempt,
                )
                feedback = f"previous attempt timed out at {timeout_per_attempt}s"
                best = _pick_best_ocr(best, ("", model, attempt, 0.0, ["timeout"]))
            except Exception as e:
                # Never raise from reflexive loop — page must not fail
                log.warning(
                    "OCR error (model=%s attempt=%d): %s — cascading",
                    model,
                    attempt,
                    e,
                )
                feedback = f"previous attempt errored: {e}"
                best = _pick_best_ocr(best, ("", model, attempt, 0.0, [f"error: {e}"]))

    # Cross-model disagreement check (uncertainty signal via output comparison)
    if len(_model_texts) >= 2:
        try:
            _sdir = os.path.dirname(os.path.abspath(__file__))
            if _sdir not in sys.path:
                sys.path.insert(0, _sdir)
            from model_disagreement import compare_texts

            _tx = list(_model_texts.values())
            _rpt = compare_texts(_tx[0], _tx[1])
            if _rpt.recommend_triage != "ACCEPT":
                _w = list(best[4]) if best[4] else []
                _w.append(
                    f"model_disagreement: {_rpt.recommend_triage} "
                    f"(distance={_rpt.max_distance:.2f})"
                )
                best = best[:4] + (_w,) + tuple(best[5:])
        except Exception as e:
            log.debug("ocr_with_reflexive_loop: Exception degraded: %s", e)
    # Chain exhausted — return best-effort
    return best[:5]


def _pick_best_ocr(current, candidate):
    """Pick the better of two (text, model, attempt, conf, warnings[, elapsed]) tuples.

    Selection rule: higher confidence wins; tie → longer text; tie → fewer attempts.
    Special case: empty-text placeholder (initial state) always yields to candidate
    so catastrophic-failure warnings propagate through the cascade.
    Pure function — easy unit-test target.
    """
    _load_consts()  # lazy-load constants
    c_text, c_model, c_att, c_conf = current[0], current[1], current[2], current[3]
    n_text, n_model, n_att, n_conf = (
        candidate[0],
        candidate[1],
        candidate[2],
        candidate[3],
    )
    # Empty placeholder → always accept candidate (records real attempt info,
    # including error/timeout warnings that would otherwise be lost).
    if not c_text and c_conf <= 0.0:
        return candidate
    if n_conf > c_conf:
        return candidate
    if n_conf == c_conf and len(n_text) > len(c_text):
        return candidate
    if n_conf == c_conf and len(n_text) == len(c_text) and n_att < c_att:
        return candidate
    return current


def _score_ocr_text(text):
    """Quick confidence score for reflexive loop (lighter than full
    _compute_page_confidence — no warning collectors, just text-quality heuristics).

    Returns (confidence_0_to_1, breakdown_dict).
    Mirrors the signal-set of should_route_to_ocr but scored continuously.

    Calibrated against:
      - Clean scientific prose → 0.85+ (no retry)
      - Hallucination loops ("no visible characters" × 30) → <0.3 (cascade)
      - Private-char floods / U+FFFD encoding breaks → <0.4 (cascade)
      - CJK hallucination on non-CJK pages → <0.3 (cascade)
      - Broken LaTeX ($$ unclosed) → caught by _verify_laTeX_quality, not here
    """
    _load_consts()  # lazy-load constants
    if not text or len(text.strip()) < 2:
        return 0.0, {"empty": True}
    s = text.strip()
    chars = len(s)
    # Alphanum ratio (clean prose: ~0.75; garbage: <0.40)
    alnum = sum(1 for c in s if c.isalnum()) / chars
    # Repetition: count ONLY over-repeats (top-5 5-gram counts minus 1 each).
    # Fixes bug where unique baseline (5 distinct 5-grams) scored as 28% repeat.
    # For 28 repeats of one 5-gram: (28-1) + 0 + 0 + 0 + 0 = 27 excess.
    # For 5 unique 5-grams (clean text): (1-1)*5 = 0 excess.
    words = s.split()
    if len(words) > 20:
        from collections import Counter

        top5 = Counter(tuple(words[i : i + 5]) for i in range(len(words) - 4)).most_common(5)
        repeat_excess = sum(max(0, c - 1) for _, c in top5)
        repeat_ratio = repeat_excess / max(len(words) - 4, 1)
    else:
        repeat_ratio = 0.0
    # Private control chars (U+0000-U+0008, U+000E-U+001F, U+FFFE, U+FFFF)
    priv = len(_pdf_ocr()._PRIVATE_CHAR_RE.findall(s)) / max(chars, 1)
    # U+FFFD replacement chars (broken-encoding marker, separate from above)
    fffd = s.count("\ufffd") / max(chars, 1)
    # CJK hallucination on non-CJK pages
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff") / max(chars, 1)

    # Score: start at 1.0, subtract weighted penalties.
    # Curves tuned so:
    #   - Clean text (alnum 0.75, repeat 0)         → ~0.95
    #   - Hallucination (repeat 0.30+)                → <0.20 (catastrophic)
    #   - Encoding flood (priv/fffd 0.05+)            → <0.40
    #   - CJK flood (cjk 0.30+)                       → <0.30
    score = 1.0
    # Alnum penalty: linear, kicks in below 0.50 (clean ~0.75 = no penalty)
    score -= max(0.0, (0.50 - alnum) * 1.4)
    # Repetition penalty: STEEP (hallucination is #1 OCR failure mode)
    #   repeat 0.30 → -0.84 (catastrophic); 0.10 → -0.28 (heavy); 0.05 → -0.14 (light)
    score -= min(0.95, repeat_ratio * 2.8)
    # Private control chars: very steep (always bad)
    score -= min(0.40, priv * 8.0)
    # U+FFFD replacement chars: steep (broken encoding)
    score -= min(0.40, fffd * 6.0)
    # CJK hallucination: steep
    score -= min(0.40, cjk * 3.0)
    # Very-low char count penalty
    if chars < 50:
        score -= (50 - chars) / 100.0

    return max(0.0, min(1.0, score)), {
        "chars": chars,
        "alnum": round(alnum, 2),
        "repeat": round(repeat_ratio, 2),
        "priv": round(priv, 3),
        "ffmd": round(fffd, 3),
        "cjk": round(cjk, 3),
    }


# Threshold for "text layer present" detection. Below this average chars/page,
# assume the PDF is scanned and requires OCR.

__all__ = [
    "REFLEXIVE_CONFIDENCE_FLOOR",
    "REFLEXIVE_MAX_ATTEMPTS_PER_MODEL",
    "OCR_PER_ATTEMPT_TIMEOUT_S",
    "SKILL_EXTRACTION_VERSION",
    "DEFAULT_CACHE_DIR",
    "_cache_key",
    "cache_lookup",
    "cache_write",
    "_verify_laTeX_quality",
    "ocr_with_reflexive_loop",
    "_pick_best_ocr",
    "_score_ocr_text",
]
