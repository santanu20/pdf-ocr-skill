"""Routing — page classification, OCR routing, model selection.

Extracted from scripts/pdf_ocr.py during Phase R5 refactoring (2026-08-12).

Functions:
  - should_route_to_ocr: 9-signal router (low_chars, scanned_image, etc.)
  - preclassify_document: one-pass doc-level preclassify (eliminates O(N) opens)
  - select_ocr_model: per-page-type model selection (Stage-2 router)
  - is_image_like_page: PDF page has only image content (no text)
  - page_has_invisible_text: PDF has invisible text layer
  - _is_legitimate_short_line_page: refs/bullets heuristic
  - extract_text_layer: pdftotext wrapper for a range of pages
"""

from __future__ import annotations

import re
import os
import subprocess
import logging

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state

# R14b (2026-09-12): broken-ToUnicode mojibake markers. ∫/hat/hyphen glyphs
# with corrupted CMaps extract as ISOLATED acute/diaeresis/circumflex
# (measured 2503.04488 pg9: 'r´, ´sq', ¨q — parens→p/q are valid letters and
# stay invisible to text-level detection). Isolation = NOT adjacent to a
# letter: in clean prose these diacritics attach to letters; standalone
# occurrences are mojibake or poetry, never ≥3 per scientific page.
_MOJIBAKE_MARKER_RE = re.compile(r"(?<![a-zA-Zα-ωΑ-Ω])[´¨ˆ](?![a-zA-Zα-ωΑ-Ω])")


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


_LATE = frozenset(
    {
        "IS_IMAGE",
        "PDF_PATH",
        "LOW_TEXT_DENSITY_THRESHOLD",
        "ROUTER_MIN_CHARS",
        "ROUTER_ALPHANUM_MIN",
        "ROUTER_SPACE_RATIO_MAX",
        "ROUTER_NEWLINE_RATIO_MAX",
        "ROUTER_INVALID_CHAR_THRESHOLD",
        "ROUTER_MULTI_SPACE_RUN_MAX",
        "ROUTER_SHORT_LINE_THRESHOLD",
        "ROUTER_MIN_AVG_WPL",
        "ROUTER_MOJIBAKE_MIN_MARKERS",
        "TEXT_LAYER_CHARS_PER_PAGE_MIN",
        "_PRIVATE_CHAR_RE",
        "MATH_FONT_PREFIXES",
    }
)


def __getattr__(name):
    if name in _LATE:
        return getattr(_pdf_ocr(), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Constants loaded lazily into module globals on first function call.
# Workaround: Python module __getattr__ does NOT resolve global name lookups
# inside function bodies (only attribute access on the module). So we eagerly
# copy the needed constants into our globals at first use.
_CONSTS_LOADED = False


def _load_consts():
    """Populate module globals with constants from pdf_ocr (lazy, idempotent)."""
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    for name in (
        "IS_IMAGE",
        "PDF_PATH",
        "LOW_TEXT_DENSITY_THRESHOLD",
        "ROUTER_MIN_CHARS",
        "ROUTER_ALPHANUM_MIN",
        "ROUTER_SPACE_RATIO_MAX",
        "ROUTER_NEWLINE_RATIO_MAX",
        "ROUTER_INVALID_CHAR_ABS",
        "ROUTER_INVALID_CHAR_PCT",
        "ROUTER_MULTI_SPACE_RUN_MAX",
        "ROUTER_SHORT_LINE_MIN_CHARS",
        "ROUTER_SHORT_LINE_RATIO_MAX",
        "ROUTER_MIN_AVG_WPL",
        "ROUTER_IMAGE_COVERAGE_THRESHOLD",
        "ROUTER_MOJIBAKE_MIN_MARKERS",
        "TEXT_LAYER_CHARS_PER_PAGE_MIN",
        "_PRIVATE_CHAR_RE",
        "MATH_FONT_PREFIXES",
        # Helpers + state used by routing functions
        "DEPS",
        "_REFLINE_RE",
        "_DOI_RE",
        "_BULLET_RE",
        "_CODE_RE",
        "page_has_embedded_fonts",
        "alphanum_ratio",
        "_get_page_dict",
        "space_ratio",
        "newline_ratio",
        "invalid_char_count",
        "multi_space_run_ratio",
        "avg_words_per_line",
        "short_line_ratio",
    ):
        if hasattr(p, name):
            g[name] = getattr(p, name)
    # PERF-1a: _pdfium/_pdfium_c are PEP 562 lazy in shared and are NOT
    # pre-bound here — loading them costs ~118ms of import work that only
    # the invisible-text probe (fontless/suspect pages) actually needs.
    # The probe call sites resolve them via shared attr access at run time.
    _CONSTS_LOADED = True


def _is_legitimate_short_line_page(text):
    """Check if a page's short-line pattern is from refs/bullets/code, not shattered table.

    Excludes: reference lists ([1]...), DOIs, bullet points, indented code.
    Returns True if ≥60% of short lines match legitimate patterns.
    """
    _load_consts()  # lazy-load cross-module constants
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return False
    short_lines = [l for l in lines if len(l.strip()) < 10]
    if not short_lines:
        return False
    # TOC/INDEX pages (2026-08-17): entries are short by nature and end
    # with dot-leaders or a trailing page number ('Chapter ..... 42').
    # Measured long_tiny 20_pg42 (a ToC page) falsely firing
    # positioned_glyphs → 56s/page VL for zero score change.
    # R5 (audit-2, 2026-08-18): the bare number-ending pattern alone
    # ("\s\d{1,3}$") also matches equation/measure fragments ("n = 5").
    # A REAL ToC has page numbers in ASCENDING order — require that
    # structural signature (≥4 entries, ≥60% of consecutive pairs
    # non-decreasing) unless a dot-leader fired (dot-leaders are ToC-only).
    _dot = sum(1 for l in short_lines if re.search(r"\.{3,}\s*\d+\s*$", l))
    _num_lines = [l for l in short_lines if re.search(r"\s\d{1,3}$", l.strip())]
    _is_toc = False
    if _dot / len(short_lines) >= 0.50:
        _is_toc = True
    elif len(_num_lines) >= 4 and len(_num_lines) / len(short_lines) >= 0.50:
        nums = [int(re.search(r"(\d{1,3})$", l.strip()).group(1)) for l in _num_lines]
        pairs = list(zip(nums, nums[1:]))
        ascending = sum(1 for a, b in pairs if b >= a) / max(len(pairs), 1)
        _is_toc = ascending >= 0.60
    if _is_toc:
        return True
    legit = sum(
        1
        for l in short_lines
        if _REFLINE_RE.match(l) or _DOI_RE.search(l) or _BULLET_RE.match(l) or _CODE_RE.match(l)
    )
    return legit / len(short_lines) >= 0.60


def preclassify_document(pdf_path, page_list):
    """One-pass document classification to eliminate per-page PDF opens.

    Opens the PDF ONCE, checks all pages for:
      - Embedded fonts (born-digital signal)
      - Invisible text render mode (OCR overlay signal)

    Returns dict:
      'born_digital': bool — True if ALL pages have fonts and NONE have invisible text.
                             When True, per-page routing skips PDF-open checks entirely.
      'invisible_pages': set of page_nums with invisible text render mode.
      'pages_with_fonts': set of page_nums with at least one embedded font.
    """
    _load_consts()  # lazy-load cross-module constants
    result = {
        "born_digital": False,
        "invisible_pages": set(),
        "pages_with_fonts": set(),
        "suspect_fonts": False,  # Custom-encoded fonts → ToUnicode risk
        "page_cols": {},  # page_num → detected column count (fitz, cheap)
    }

    # One pdffonts call for all requested pages
    if DEPS.get("pdffonts"):
        try:
            proc = subprocess.run(
                [
                    "pdffonts",
                    "-f",
                    str(page_list[0]),
                    "-l",
                    str(page_list[-1]),
                    pdf_path,
                ],
                capture_output=True,
                text=True,
                errors="replace",
                timeout=30,
                check=True,
            )
            # pdffonts output: header + one row per font. NO pages column
            # exists (the header is name/type/encoding/emb/sub/uni/object
            # ID — parts[5] is the `sub` flag; a prior "pages column"
            # parse here was dead code reading "yes"/"no"). pdffonts is
            # used ONLY for suspect-font detection (Custom/Symbol
            # encodings = ToUnicode-risk family, measured 2026-08-15
            # multi_column: ArnoPro/Custom mapped 'g'→'e'). Per-PAGE font
            # truth comes from the fitz scan below (R1).
            for line in proc.stdout.split("\n")[2:]:
                line = line.strip()
                if not line:
                    continue
                if re.search(r"\s(Custom|Symbol)\s", line):
                    result["suspect_fonts"] = True
        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as e:
            log.debug("pdffonts probe degraded: %s", e)

    # One pypdfium2 open for invisible-text check on all pages.
    # pypdfium2 resolves lazily from shared (PEP 562 — PERF-1a): ~118ms of
    # import work is paid ONLY when a page actually lacks embedded fonts.
    import shared as _shared_mod

    try:
        _pdfium, _pdfium_c = _shared_mod._pdfium, _shared_mod._pdfium_c
        doc = _pdfium.PdfDocument(pdf_path)
        try:
            for p in page_list:
                page = doc.get_page(p - 1)
                text_objs = list(page.get_objects(filter=[_pdfium_c.FPDF_PAGEOBJ_TEXT]))
                if text_objs:
                    for obj in text_objs:
                        mode = _pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
                        if mode in (
                            _pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE,
                            _pdfium_c.FPDF_TEXTRENDERMODE_UNKNOWN,
                        ):
                            result["invisible_pages"].add(p)
                            break
        finally:
            doc.close()
    except Exception as e:
        log.warning("pypdfium2 invisible-text probe failed: %s", e)

    # Per-page column count + per-PAGE font truth (one fitz open).
    # R1 (audit-2, 2026-08-18): pages_with_fonts used to be blanket-set
    # from pdffonts "any row exists" — a fontless scanned page inside a
    # fonted doc was certified born-digital, disabling the scanned-page
    # signal exactly where it mattered. fitz get_fonts() gives exact
    # per-page truth for the same single open the column scan needs.
    try:
        import fitz

        from scientific_parser.routing.layout_gate import count_columns

        with fitz.open(pdf_path) as doc:
            for p in page_list:
                if p - 1 >= doc.page_count:
                    continue
                pg = doc[p - 1]
                if pg.get_fonts():
                    result["pages_with_fonts"].add(p)
                if result["suspect_fonts"]:
                    result["page_cols"][p] = count_columns(
                        pg.get_text("blocks"), float(pg.rect.width)
                    )
    except Exception as e:
        log.debug("fitz per-page font/column pre-scan failed: %s", e)

    # born_digital: all pages have fonts AND none have invisible text.
    # R1 ORDERING (2026-08-18): MUST be computed after the fitz scan —
    # pages_with_fonts is populated there. Computing it earlier left
    # born_digital permanently False whenever pdffonts carried no usable
    # per-page truth (measured 0514f667: 10 fitz fonts, born False,
    # signal-10 dead → glyph-only reference sections unrecoverable).
    all_have_fonts = all(p in result["pages_with_fonts"] for p in page_list)
    no_invisible = len(result["invisible_pages"]) == 0
    result["born_digital"] = all_have_fonts and no_invisible

    return result


def page_has_invisible_text(pdf_path, page_num):
    """Detect pages where the text layer is invisible (OCR overlay).

    Marker SOTA (strip_existing_ocr). PDFs OCR'd by Adobe/Acrobat have an
    invisible text layer rendered on top of the scanned image — the text is
    there for search/select, but the render mode is "invisible" so the user
    sees only the image. Quality of such text layers varies wildly and they
    often benefit from re-OCR.

    Uses pypdfium2 to check text object render mode. Returns True if any
    text object uses invisible/unknown render mode.
    """
    _load_consts()  # lazy-load cross-module constants
    import shared as _shared_mod

    try:
        _pdfium, _pdfium_c = _shared_mod._pdfium, _shared_mod._pdfium_c
        doc = _pdfium.PdfDocument(pdf_path)
        try:
            page = doc.get_page(page_num - 1)  # 0-indexed
            text_objs = list(page.get_objects(filter=[_pdfium_c.FPDF_PAGEOBJ_TEXT]))
            if not text_objs:
                return None  # no text objects — let other signals decide
            # Marker's check: any text in invisible or unknown render mode
            for obj in text_objs:
                mode = _pdfium_c.FPDFTextObj_GetTextRenderMode(obj)
                if mode in (
                    _pdfium_c.FPDF_TEXTRENDERMODE_INVISIBLE,
                    _pdfium_c.FPDF_TEXTRENDERMODE_UNKNOWN,
                ):
                    return True
            return False
        finally:
            doc.close()
    except Exception:
        return None  # don't crash on weird PDFs


def is_image_like_page(pdf_path, page_num):
    """Detect scanned pages: no embedded fonts OR large image covers most of page.

    Two-signal composite (Marker SOTA + image-area heuristic):
      1. pdffonts: zero embedded fonts = scanned (STRONGEST signal)
      2. pdfimages: image area / page area >= 0.65 = scanned

    Adapts to available tools (DEPS-aware):
      - pdffonts missing → skip font check, fall back to image-area only
      - pdfimages missing → skip image check, font check only
      - both missing → return False (no signal), routing falls back to text-quality only
    """
    _load_consts()  # lazy-load cross-module constants
    # Signal 1 (strongest): font analysis
    has_fonts = page_has_embedded_fonts(pdf_path, page_num)
    if has_fonts is False:
        return True  # confirmed scanned
    if has_fonts is None and not DEPS.get("pdfimages", False):
        return False  # both signals unavailable — let text-quality signals decide

    if not DEPS.get("pdfimages", False):
        return False  # font check was inconclusive, image check unavailable

    # Signal 2: image coverage ratio
    try:
        info = subprocess.run(
            ["pdfinfo", "-f", str(page_num), "-l", str(page_num), pdf_path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=True,
        )
        page_w = page_h = 0
        for line in info.stdout.split("\n"):
            m = re.match(r"\s*Page\s+(?:\d+\s+)?size:\s*([\d.]+)\s*x\s*([\d.]+)", line)
            if m:
                page_w, page_h = float(m.group(1)), float(m.group(2))
                break
        if page_w == 0 or page_h == 0:
            return False
        page_area = page_w * page_h

        proc = subprocess.run(
            ["pdfimages", "-list", "-f", str(page_num), "-l", str(page_num), pdf_path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=True,
        )
        rows = [l for l in proc.stdout.split("\n")[2:] if l.strip() and re.match(r"^\s*\d+", l)]
        if not rows:
            return False
        total_img_area = 0
        for r in rows:
            parts = r.split()
            try:
                w, h = int(parts[3]), int(parts[4])
                total_img_area += w * h
            except (IndexError, ValueError):
                continue
        return (total_img_area / page_area) >= ROUTER_IMAGE_COVERAGE_THRESHOLD
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return False


def tounicode_crosscheck(pdf_path, page_num, img_path):
    """Detect systematic ToUnicode mis-mappings by render-and-compare.

    Renders the page (caller supplies img_path), rapidocr-reads the GLYPHS,
    and word-aligns them against the text layer. A ToUnicode bug maps the
    same glyph to the same wrong char EVERYWHERE (measured 2026-08-15:
    ArnoPro 'g'→'e' across a Maltese page); OCR reading errors are random
    and never repeat one substitution. Returns (n_systematic, example,
    words) where n_systematic = count of the most-repeated single-char
    substitution across differing word pairs, example = (frm, to) with
    frm/to single chars or '' for delete/insert, words = the layer words
    that contributed the top substitution (surgical-fix surface);
    (0, None, ()) when clean or OCR unavailable.
    """
    import difflib
    from collections import Counter

    try:
        import fitz
    except ImportError:
        return 0, None, ()
    try:
        from scientific_parser.ingest.ocr_engine import ocr_image_rapidocr_full
    except ImportError:
        return 0, None, ()
    try:
        _t, boxes, txts, _s = ocr_image_rapidocr_full(img_path)
    except Exception:
        return 0, None, ()
    if not boxes or not txts:
        return 0, None, ()

    alpha_re = re.compile(r"^[^\W\d_]{4,}$", re.UNICODE)
    try:
        doc = fitz.open(pdf_path)
        try:
            words = doc[page_num - 1].get_text("words")
            page_h = doc[page_num - 1].rect.height
        finally:
            doc.close()
    except Exception:
        return 0, None, ()

    # ocr boxes: 4-point polys in render px; convert to (cx, cy, y0, y1) pt.
    # rapidocr txts are whole LINES — tokenize per box for word-vs-word diff.
    scale = 72.0 / 150.0
    ocr_words = []
    for b, t in zip(boxes, txts):
        try:
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            x0, x1, y0, y1 = min(xs) * scale, max(xs) * scale, min(ys) * scale, max(ys) * scale
            span = x1 - x0
            tokens = [tok for tok in re.split(r"\s+", str(t)) if tok]
            if not tokens:
                continue
            # spread tokens evenly across the box x-span (approx per-token cx)
            for k, tok in enumerate(tokens):
                tcx = x0 + span * (k + 0.5) / len(tokens)
                ocr_words.append((tcx, (y0 + y1) / 2, y0, y1, tok))
        except Exception:
            continue

    subs = Counter()
    contributors = []  # (sub_key, layer_word) — surgical-fix surface
    for w in words:
        layer = w[4]
        if not alpha_re.match(layer):
            continue
        cx, cy = (w[0] + w[2]) / 2, (w[1] + w[3]) / 2
        # nearest OCR token whose band contains the layer word center
        best = None
        for ocx, ocy, oy0, oy1, ot in ocr_words:
            if oy0 - 2 <= cy <= oy1 + 2 and abs(ocx - cx) < 25:
                d = abs(ocx - cx)
                if best is None or d < best[0]:
                    best = (d, ot)
        if best is None:
            continue
        ocr_t = best[1].strip()
        if not alpha_re.match(ocr_t):
            continue
        ratio = difflib.SequenceMatcher(None, layer.lower(), ocr_t.lower()).ratio()
        if 0.6 <= ratio < 0.95 and abs(len(layer) - len(ocr_t)) <= 1:
            sm = difflib.SequenceMatcher(None, layer.lower(), ocr_t.lower())
            for tag, i1, i2, j1, j2 in sm.get_opcodes():
                if tag == "replace" and i2 - i1 == 1 and j2 - j1 == 1:
                    key = (layer[i1].lower(), ocr_t[j1].lower())
                    subs[key] += 1
                    contributors.append((key, layer))
                elif tag == "delete" and i2 - i1 == 1:
                    key = (layer[i1].lower(), "")
                    subs[key] += 1
                    contributors.append((key, layer))
                elif tag == "insert" and j2 - j1 == 1:
                    key = ("", ocr_t[j1].lower())
                    subs[key] += 1
                    contributors.append((key, layer))
    if not subs:
        return 0, None, ()
    (frm, to), n = subs.most_common(1)[0]
    top_words = tuple({w for key, w in contributors if key == (frm, to)})
    return n, (frm, to), top_words


def _fitz_surface_alpha_ok(pdf_path, page_num, min_alpha=200):
    """True when the FITZ text surface (what the fast path consumes) carries
    >= min_alpha alphanumerics. Corroboration helper — fail-closed: any error
    or unreachable page returns False (no corroboration -> signals decide,
    which may route to OCR; never silently trusts one surface)."""
    if not pdf_path or page_num is None or page_num < 1:
        return False
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            if doc.is_encrypted or page_num > doc.page_count:
                return False
            text = doc[page_num - 1].get_text("text")
            return sum(1 for c in text if c.isalnum()) >= min_alpha
    except Exception:
        return False


def should_route_to_ocr(text, pdf_path=None, page_num=None, doc_info=None):
    """SOTA per-page routing. Returns (bool should_ocr, list_of_reasons).

    7-signal composite (validated on 90 measurements across 40 PDFs):
      1. chars < 100 → scanned page or extraction failure
      2. (chars < 100) AND image_like page → confirmed scanned
      3. alphanum_ratio < 0.30 → garbled text (Marker)
      4. space_ratio > 0.70 → fragmented whitespace flood (Marker)
      5. newline_ratio > 0.60 → one-word-per-line OCR hallucination (Marker)
      6. invalid_chars > max(6, 3% of text) → encoding issue (Marker)
      7. multi_space_run_ratio > 0.30 → shattered table (NEW)
      8. shattered_cells: ≥90% lines <10 chars (with ref/code/bullet exclusions)

    R2 fix: invisible_text_layer is a CONTRIBUTING signal, NOT an automatic
    OCR trigger. Adobe Clearscan has invisible text + high quality — we trust
    the text layer unless text-quality signals ALSO fire.

    doc_info: optional cached preclassification from preclassify_document().
    When provided, skips per-page PDF opens (invisible text + font checks use
    cached results → O(1) per page instead of O(N) PDF opens).
    """
    _load_consts()  # lazy-load cross-module constants
    reasons = []
    char_count = len(text.strip())

    # Signal 1: low char count (extraction failed or scanned)
    low_chars = char_count < ROUTER_MIN_CHARS
    if low_chars:
        reasons.append(f"low_chars ({char_count})")

    # Signal 2: image_like — only a confirmation when chars are also low.
    if low_chars and pdf_path is not None and page_num is not None:
        if doc_info is not None:
            # Use cached font info: if page has no fonts, it's scanned
            if page_num not in doc_info.get("pages_with_fonts", set()):
                reasons.append("scanned_image_page")
            elif not doc_info.get("born_digital", False):
                # Uncertain — fall back to image-area check
                if is_image_like_page(pdf_path, page_num):
                    reasons.append("scanned_image_page")
        else:
            if is_image_like_page(pdf_path, page_num):
                reasons.append("scanned_image_page")

    # Early return for truly empty/scanned pages
    if reasons:
        return True, reasons

    # Signal 2b: invisible text layer — CONTRIBUTING signal only (R2 fix).
    # Noted but does NOT trigger OCR by itself. High-quality OCR overlays
    # (Adobe Clearscan) have invisible text + good text quality.
    invisible = False
    if pdf_path is not None and page_num is not None:
        if doc_info is not None:
            invisible = page_num in doc_info.get("invisible_pages", set())
        else:
            inv = page_has_invisible_text(pdf_path, page_num)
            invisible = inv is True

    # Text-quality signals (only meaningful if substantial text exists).
    # BORN-DIGITAL PUA EXCLUSION (2026-08-29, user directive: OCR is last
    # resort): math fonts legitimately emit private-use codepoints (CMEX
    # tall-delimiter pieces the cascade's env merge consumes — measured
    # 2503.03903 pg9: 47 PUA glyphs = tall delimiters, routed to OCR).
    # On a born-digital page PUA is math payload, not corruption; only
    # U+FFFD (broken ToUnicode) and control chars remain true emergencies.
    _born_digital_early = (
        bool(doc_info)
        and page_num in doc_info.get("pages_with_fonts", set())
        and page_num not in doc_info.get("invisible_pages", set())
    )
    if _born_digital_early:
        inv_chars = sum(
            1 for c in text if ord(c) == 0xFFFD or (ord(c) < 32 and c not in "\t\n\r\f")
        )
    else:
        inv_chars = invalid_char_count(text)
    ar = alphanum_ratio(text)
    # EXP-1.5 corroboration (2026-09-16, routing audit): on born-digital
    # pages with substantial clean alpha content, low_alphanum and
    # invalid_chars are NOISE, not corruption — numeric/table pages have
    # legitimately low alpha ratios, and unmapped symbol fonts emit control
    # glyphs (0x07-0x1f bullets, diagram labels) interleaved with real
    # prose. Audit of all 1269 bench pages: the only false-routed pages
    # (12cf936d low_alphanum ar=0.20 alpha=523; 081875a1 invalid_chars=1379
    # alpha=2832, both fonts-present, 0 images, 0 U+FFFD) are this class;
    # the fast path + control-char scrub handles them correctly. Genuine
    # layer corruption on fonted pages still routes via tounicode_mojibake
    # (Signal 11) and broken_encoding U+FFFD (Signal 9), untouched here.
    _alpha_count = sum(1 for c in text if c.isalnum())
    # 05a0e5bc lesson (bench6, 2026-09-16): the router reads the PDFTOTEXT
    # surface but the fast path CONSUMES the FITZ surface. This page reads
    # healthy via pdftotext (alpha=2804) yet its fitz surface is symbol-glyph
    # garbage (alpha=92, ctrl=2953) -> corroboration on the routing surface
    # alone mis-routed it to fast. Corroborate BOTH surfaces.
    _fitz_alpha_ok = _fitz_surface_alpha_ok(pdf_path, page_num)
    _corroborated_layer = not (_born_digital_early and _alpha_count >= 200 and _fitz_alpha_ok)
    if ar < ROUTER_ALPHANUM_MIN and _corroborated_layer:
        reasons.append(f"low_alphanum ({ar:.2f})")
    sr = space_ratio(text)
    if sr > ROUTER_SPACE_RATIO_MAX:
        reasons.append(f"high_space ({sr:.2f})")
    nr = newline_ratio(text)
    if nr > ROUTER_NEWLINE_RATIO_MAX:
        reasons.append(f"high_newline ({nr:.2f})")
    if (
        inv_chars > max(ROUTER_INVALID_CHAR_ABS, len(text) * ROUTER_INVALID_CHAR_PCT)
        and _corroborated_layer
    ):
        reasons.append(f"invalid_chars ({inv_chars})")
    msr = multi_space_run_ratio(text)
    if msr > ROUTER_MULTI_SPACE_RUN_MAX:
        reasons.append(f"fragmented_table ({msr:.2f})")

    # Signal 7b: positioned-glyph detector — avg words per line < 5.
    # Catches PDFs where each word is encoded as a separate positioned
    # span (sakai1968, some old Elsevier scans). pdftotext extracts the
    # text correctly but as one-word-per-line, producing unreadable output.
    # Requires chars ≥ ROUTER_SHORT_LINE_MIN_CHARS to avoid false positives
    # on bulleted lists / short-caption pages.
    #
    # R3 fix (2026-08-15, tables bench): on BORN-DIGITAL docs these layout
    # signals are false positives — pdftotext emits legitimate table cells
    # one-per-line (measured: 8/12 born-digital table pages routed to VL
    # OCR for positioned_glyphs/shattered_cells, then garbaged by the VL
    # chat mode). Born-digital + real fonts = the text layer is exact;
    # the fast path's table specialists consume it natively. Genuinely
    # broken text layers are still caught by the strong signals
    # (low_alphanum, invalid_chars, broken_encoding, fragmented_table).
    # R6 (audit-2, 2026-08-18): per-PAGE born-digital, not doc-level. One
    # invisible-overlay page in a 100-page doc used to un-born every page
    # (re-enabling these false-positive-prone signals document-wide); a
    # fontless scanned page in a fonted doc was blanket-marked fonted.
    # Per-page truth: THIS page has fonts and is not an invisible overlay.
    # (When pdffonts degraded, the sets are empty and doc-level born_digital
    # was necessarily False — so per-page strictly dominates, never loses.)
    _born_digital = (
        bool(doc_info)
        and page_num in doc_info.get("pages_with_fonts", set())
        and page_num not in doc_info.get("invisible_pages", set())
    )
    if char_count >= ROUTER_SHORT_LINE_MIN_CHARS and not _born_digital:
        lines_w = [l for l in text.split("\n") if l.strip()]
        if lines_w:
            avg_wpl = sum(len(l.split()) for l in lines_w) / len(lines_w)
            if avg_wpl < ROUTER_MIN_AVG_WPL:
                # Exclude legitimate short-line pages (bullets, refs, captions)
                if not _is_legitimate_short_line_page(text):
                    reasons.append(f"positioned_glyphs (avg_wpl={avg_wpl:.1f})")

    # Signal 8: shattered-cell detector (with false-positive exclusions).
    # After switching from -layout to default mode, shattered tables present
    # as every cell on its own line. But reference lists, code blocks, and
    # bullet lists ALSO have short lines → exclude them (R1 fix).
    # R3: gated on NOT born-digital (same false-positive family as 7b).
    if len(text) >= ROUTER_SHORT_LINE_MIN_CHARS and not _born_digital:
        lines = [l for l in text.split("\n") if l.strip()]
        if lines:
            short_ratio = sum(1 for l in lines if len(l.strip()) < 10) / len(lines)
            if short_ratio >= ROUTER_SHORT_LINE_RATIO_MAX:
                if not _is_legitimate_short_line_page(text):
                    reasons.append(f"shattered_cells (short={short_ratio:.2f})")

    # Signal 9: broken font encoding — high U+FFFD (replacement char) density.
    # When font CMap is broken, text extraction produces garbage (all U+FFFD).
    # pdftotext -layout might recover some text, but OCR is the reliable path.
    fffd_count = text.count("\ufffd")
    if fffd_count > max(10, len(text) * 0.05):
        reasons.append(
            f"broken_encoding ({fffd_count} U+FFFD, {fffd_count / max(len(text), 1) * 100:.0f}%)"
        )
    # Signal 10: glyph_reread — RETIRED 2026-08-19 (iter-56).
    # Born-digital + custom fonts + 2col prose → VL glyph re-read. The
    # original 2026-08-15 measurement ('semmigli'→'semmieli' ToUnicode
    # errors, VL matches gold) is OBSOLETE: iter-53's fragment-block
    # assembly + baseline-merge fixed fast-path reconstruction for exactly
    # these font families (T11-T18 subsets). Head-to-head on all 12
    # glyph_reread fixtures from the bench sample (2026-08-19):
    # fast 37/50 vs VL 30/50 (+7; sole regression 01566a95 −1) — and the
    # deterministic path also kills the ±3-test VL wobble (server-load
    # nondeterminism, measured runs 48-50). Authentic OCR routes remain:
    # low_chars+scanned (image-only), positioned_glyphs+invisible
    # (hidden-OCR overlays).

    # Signal 11: ToUnicode mojibake (R14b, 2026-09-12) — isolated
    # acute/diaeresis/circumflex markers. Fires on BORN-DIGITAL pages by
    # design: this is exactly the counter-case to the PUA-exclusion
    # philosophy above — the text layer is NOT exact, its CMap is broken
    # (∫→´, hats→¨), and the existing whole-page VL remedies are
    # deliberately excluded on font-math pages (they destroy LaTeX).
    # Routing to the OCR path hands the page to the image→cascade→LaTeX
    # path which reads the CORRECT rendered glyphs. Calibrated full-bench:
    # iso>=3 → 28/522 arxiv (139 gold-test fails live there, most pages
    # 100% failed — fast path ≈ 0, cascade can only improve), 1/266 hf,
    # 0 everywhere else. NOT gated on born_digital: the marker signature
    # is mojibake evidence regardless of page provenance.
    mojibake_markers = len(_MOJIBAKE_MARKER_RE.findall(text))
    if mojibake_markers >= ROUTER_MOJIBAKE_MIN_MARKERS:
        reasons.append(f"tounicode_mojibake ({mojibake_markers} isolated markers)")

    # If text-quality signals fired AND invisible text → add as contributing
    if invisible and reasons:
        reasons.append("invisible_text_layer")

    return bool(reasons), reasons


# =============================================================================
# SOTA Extraction Layer — adaptive model router + reflexive loop + cache
# =============================================================================
# Implements Phase 7 of yggdrasil plan dp-20260811-pdf-ocr-sota-agentic-redesign
# (the unbuilt reflexive self-correction loop). Audit ref: AUDIT_2026-08-12.md §5.
#
# Three components, no new deps:
#   1. select_ocr_model(route, reasons, doc_info, models) — per-page-type model
#      selection. Bench-validated: qwen2.5vl:3b beats glm-ocr by +10 pts on
#      scanned pages; rapidocr wins on speed for tables.
#   2. ocr_with_reflexive_loop(img, model_chain, ...) — extract → verify
#      (pykatex) → score (_compute_page_confidence) → retry next model OR
#      re-prompt with error feedback (arXiv 2603.22651 pattern).
#   3. Content-hash cache (~/.cache/pdf-ocr/<sha256>/) — re-runs of same PDF
#      in <5% original time. Marker/MinerU/Docling all have this.
#
# Cascade policy: models list is honored as ordered fallback chain.
# `--models a,b,c` means try a first, fall back to b on low confidence or
# timeout, then c. Replaces the legacy `models[:1]` slice that silently
# discarded the chain (AUDIT_2026-08-12 finding, pdf_ocr.py:7769 pre-fix).

# Confidence below which reflexive loop retries with next model / feedback.
# Calibrated against _compute_page_confidence: pages scoring >=0.55 are
# consistently well-formed (verified on ballhaus + olmOCR-Bench QC batch).


def select_ocr_model(route, reasons, models, doc_info=None, page_text=""):
    """SOTA Stage-2 router: pick best OCR model per page type.

    Inputs:
      route: ("fast"|"ocr") from should_route_to_ocr (Stage 1)
      reasons: list of routing reasons (signal names)
      models: ordered list of available Ollama models (cascade chain)
      doc_info: optional preclassify_document cache
      page_text: optional pdftotext text (for content-type sniffing)

    Returns: (model_name, prompt_name, rationale_str)
      model_name = one of `models` (or first if no specific match)
      prompt_name = "MATH_OCR_PROMPT" | "TEXT_OCR_PROMPT"

    Heuristic mapping (bench-validated):
      - scanned_image_page        → qwen2.5vl:3b if available (else first)
                                    (qwen2.5vl:3b beats glm-ocr by +10 pts on old_scans)
      - broken_encoding / CJK     → qwen2.5vl:3b if available (glm-ocr hallucinates CJK)
      - fragmented_table          → "rapidocr" if available (speed + table structure)
                                    else first model with MATH_OCR_PROMPT (table-rendering)
      - math-region detection     → primary model with MATH_OCR_PROMPT
                                    (math font spans detected via _PRIVATE_CHAR_RE)
      - default                   → models[0] with MATH_OCR_PROMPT (current behavior)

    Pure function — no side effects. Trivially unit-testable.
    """
    _load_consts()  # lazy-load cross-module constants
    if not models:
        return "glm-ocr", "MATH_OCR_PROMPT", "no models supplied, defaulting"
    primary = models[0]
    has_qwen = any(m.startswith("qwen") for m in models)
    has_rapidocr = any("rapidocr" in m.lower() for m in models)

    # Reason-to-model mapping. First match wins (ordered by signal strength).
    reason_text = " ".join(reasons).lower() if reasons else ""

    # Scanned pages: qwen2.5vl:3b is +10 pts over glm-ocr (bench)
    # (suppressed on low-RAM hosts by the single-model cap above)
    if "scanned_image_page" in reason_text and has_qwen:
        qwen = next(m for m in models if m.startswith("qwen"))
        return qwen, "MATH_OCR_PROMPT", "scanned page → qwen2.5vl (bench: +10 pts)"

    # Broken encoding / CJK hallucination: qwen handles better
    if "broken_encoding" in reason_text and has_qwen:
        qwen = next(m for m in models if m.startswith("qwen"))
        return qwen, "MATH_OCR_PROMPT", "broken encoding → qwen2.5vl (CJK-safe)"

    # Fragmented tables: rapidocr is fast and table-aware
    if "fragmented_table" in reason_text and has_rapidocr:
        rapid = next(m for m in models if "rapidocr" in m.lower())
        return rapid, "TEXT_OCR_PROMPT", "table fragment → rapidocr (table-speed)"

    # Math content sniff: private chars or known math symbols in pdftotext
    if page_text and (
        _PRIVATE_CHAR_RE.search(page_text)
        or any(s in page_text for s in ("∑", "∏", "∫", "√", "α", "β", "γ"))
    ):
        return primary, "MATH_OCR_PROMPT", "math content → primary + math prompt"

    return primary, "MATH_OCR_PROMPT", "default → primary + math prompt"


def extract_text_layer(pdf_path, page_list):
    """Extract text via pdftotext for born-digital PDFs. Returns (text_by_page, avg_chars).

    Returns dict {page_num: text} plus average chars per page. If average is below
    TEXT_LAYER_CHARS_PER_PAGE_MIN, the caller should fall back to OCR.
    """
    _load_consts()  # lazy-load cross-module constants
    proc = subprocess.run(
        [
            "pdftotext",
            # Default mode (no -layout) handles 2-column scientific papers correctly:
            # reads left column fully, then right column. The -layout option INTERLEAVES
            # columns onto the same line, producing garbage text on 82% of papers
            # (empirically measured across reference set). Tables are still detected
            # via multi_space_run_ratio signal + region OCR fallback.
            "-enc",
            "UTF-8",  # standard encoding
            "-eol",
            "unix",  # unix line endings
            "-f",
            str(page_list[0]),
            "-l",
            str(page_list[-1]),
            pdf_path,
            "-",
        ],
        capture_output=True,
        text=True,
        errors="replace",
        check=True,
    )
    # pdftotext concatenates pages with form feed (\f) by default
    raw_pages = proc.stdout.split("\f")
    by_page = {}
    total = 0
    for i, text in enumerate(raw_pages):
        page_num = page_list[0] + i
        # Strip trailing whitespace but preserve internal layout
        text = text.rstrip()
        by_page[page_num] = text
        total += len(text.strip())
    # R9 (audit-2, 2026-08-18): average over the pages pdftotext ACTUALLY
    # produced (raw_pages), not len(page_list). With non-contiguous
    # --pages ("1,5"), pdftotext emits 5 pages while page_list has 2 —
    # dividing by len(page_list) overcounted avg_chars ~2.5x and could
    # clear the doc-level TEXT_LAYER_CHARS_PER_PAGE_MIN gate falsely.
    avg = total / max(len(raw_pages), 1)
    return by_page, avg


def _is_glued_text(text: str) -> bool:
    """True when the text layer lost word boundaries (words glued together).

    pdftotext emits glued words on fonts with broken/missing advances;
    normal English averages ~5 chars/alpha-word, glued pages ~12+.
    """
    words = [w for w in text.split() if w.isalpha()]
    if len(words) < 20:
        return False
    return sum(len(w) for w in words) / len(words) > 8.5


def rescue_glued_pages(pdf_path, page_list, by_page):
    """Re-extract glued pdftotext pages with MuPDF (fitz) word inference.

    pdftotext drops spaces when fonts lack proper glyph advances
    (measured: arxiv_math 2503.08272 — 'Finally,webrieflysummarise…'
    vs fitz 'Finally, we briefly summarise…'). MuPDF infers spaces from
    char geometry, so fitz text is the rescue. In-place: returns by_page.
    """
    for p in page_list:
        t = by_page.get(p, "")
        if not t or not _is_glued_text(t):
            continue
        try:
            import fitz

            with fitz.open(pdf_path) as doc:
                t2 = doc[p - 1].get_text()
            if t2 and not _is_glued_text(t2) and len(t2.strip()) >= 0.5 * len(t.strip()):
                by_page[p] = t2
        except Exception as e:
            log.debug("glued-page fitz rescue failed p%s: %s", p, e)
    return by_page


# Dependency state — populated by check_dependencies(), queried by routing logic.
# Avoids re-running which/subprocess checks per page.

__all__ = [
    "_is_legitimate_short_line_page",
    "preclassify_document",
    "page_has_invisible_text",
    "is_image_like_page",
    "should_route_to_ocr",
    "tounicode_crosscheck",
    "select_ocr_model",
    "extract_text_layer",
    "rescue_glued_pages",
]
