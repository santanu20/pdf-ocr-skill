"""Font-based subscript/superscript detection + math regex builder.

Extracted from pdf_ocr.py during R9 final slim-down (2026-08-12).
"""

from __future__ import annotations

import logging
import re
import subprocess

log = logging.getLogger(__name__)
from scientific_parser.core.path_warnings import (
    SCRIPT_EXTRACTION_FALLBACK,
    record_path_warning,
)
from shared import *  # ALL shared constants, patterns, state

# _is_math_symbol, _pdfium, _pdfium_c: imported from shared via `*` since
# 2026-08-13 (single source of truth). See shared.py for canonical defs.


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
    for n in ("_page_dict_cache", "DEPS", "_PRIVATE_CHAR_RE", "MIN_SPAN_CHARS"):
        if hasattr(p, n):
            g[n] = getattr(p, n)
    _CONSTS_LOADED = True


def detect_subscripts_pymupdf(line_chars):
    """Detect subscripts in PyMuPDF coordinates (y increases DOWN).

    PyMuPDF uses top-left origin where y0=top (smaller) and y1=bottom (larger).
    This is DIFFERENT from assign_scripts_to_chars which assumes PDF coords
    (y increases UP). This function handles the coordinate difference.

    Script signal: digit/math char smaller than median height (<70%) whose
    BOTTOM edge is displaced from the line's median bottom — below it by
    >12% of line height = SUBSCRIPT, above it = SUPERSCRIPT (bottoms are
    ascender-independent; tops-only classification misread raised unit
    superscripts like kg/m² as subscripts, 027c3905).
    """
    if len(line_chars) < 2:
        for c in line_chars:
            c["subscript"] = False
            c["superscript"] = False
        return line_chars

    geo = [c for c in line_chars if c.get("has_geometry", True)]
    if len(geo) < 2:
        for c in line_chars:
            c["subscript"] = False
            c["superscript"] = False
        return line_chars

    # In PyMuPDF coords (y DOWN): y0 = top edge (smaller value = higher on page)
    # BOTTOMS are the stable signal: every glyph's bottom sits at (sub) or
    # near (normal) the baseline regardless of ascenders/descenders. A small
    # digit raised above the line (kg/m²) has its BOTTOM well above the
    # median bottom; a true subscript's bottom sits below it. Tops-only
    # classification called raised superscripts subscripts (their tops sit
    # near the x-height top = median top, 027c3905 'kg/m2').
    bottoms = sorted(c["bbox"][3] for c in geo)  # bbox[3] = y1 = bottom
    heights = sorted(c["bbox"][3] - c["bbox"][1] for c in geo)
    median_bottom = bottoms[len(bottoms) // 2]
    median_height = heights[len(heights) // 2]
    if median_height <= 0:
        for c in line_chars:
            c["subscript"] = False
            c["superscript"] = False
        return line_chars

    drop_threshold = median_height * 0.12

    for c in line_chars:
        c.setdefault("subscript", False)
        c.setdefault("superscript", False)
        if not c.get("has_geometry", True):
            continue
        text = c["text"]
        if not (text.isdigit() or (len(text) == 1 and _is_math_symbol(text))):
            continue
        char_height = c["bbox"][3] - c["bbox"][1]
        drop = c["bbox"][3] - median_bottom
        if char_height < median_height * 0.7:
            if drop > drop_threshold:
                c["subscript"] = True
            elif drop < -drop_threshold:
                c["superscript"] = True

    return line_chars


def assign_scripts_to_chars(chars, height_threshold=0.6, line_distance_threshold=0.1):
    """Detect sub/superscripts using position + content-type heuristics.

    Hybrid of pdftext's assign_scripts (geometry) + content-type filter
    (digits/math only) to avoid false positives from optical sizing and
    descender letters (g, j, p, q, y).

    A char is flagged as subscript/superscript if BOTH:
      1. Its vertical position is offset from the median baseline by >15%
         of line height (sub: below baseline; super: above x-height).
      2. Its text content is a digit OR single math symbol (Unicode 'Sm').
         Letters are excluded because descenders (gjpqy) and optical-sized
         lowercase (aeiou) cause false positives in modern proportional fonts.

    This catches common scientific subscripts (H₂O, NH₄, Fe^{2+}, xₙ) while
    avoiding the false-positive flood of pure-geometry detection.
    """
    if len(chars) < 2:
        for c in chars:
            c.setdefault("subscript", False)
            c.setdefault("superscript", False)
        return chars

    # Compute median baseline (median bottom-y across geometrically-valid chars)
    geo_chars = [c for c in chars if c.get("has_geometry", True)]
    if len(geo_chars) < 2:
        for c in chars:
            c.setdefault("subscript", False)
            c.setdefault("superscript", False)
        return chars

    # Baseline = median of bottom-y values
    bottoms = sorted(c["bbox"][1] for c in geo_chars)
    median_baseline = bottoms[len(bottoms) // 2]
    # Median x-height (top of lowercase letters) = median top-y
    tops = sorted(c["bbox"][3] for c in geo_chars)
    median_top = tops[len(tops) // 2]
    line_height = median_top - median_baseline
    if line_height <= 0:
        for c in chars:
            c.setdefault("subscript", False)
            c.setdefault("superscript", False)
        return chars

    # Threshold: char must be offset by ≥15% of line height from baseline
    sub_threshold = line_height * 0.15
    sup_threshold = line_height * 0.15

    for c in chars:
        c.setdefault("subscript", False)
        c.setdefault("superscript", False)
        if not c.get("has_geometry", True):
            continue
        text = c["text"]
        # Only consider digits and single math symbols as potential sub/superscripts
        if not (text.isdigit() or (len(text) == 1 and _is_math_symbol(text))):
            continue
        # Subscript signal: digit positioned below baseline OR notably smaller
        # than median char height with bottom at/below baseline.
        # The "below baseline" check alone misses cases where the subscript
        # extends to the same bottom but is shorter (common in some PDFs where
        # the subscript is rendered as a smaller font shifted down slightly).
        char_height = c["bbox"][3] - c["bbox"][1]
        median_height = max(median_top - median_baseline, 1)
        is_smaller = char_height < median_height * 0.7  # <70% of normal height
        is_below = c["bbox"][1] < median_baseline - sub_threshold
        is_at_baseline = abs(c["bbox"][1] - median_baseline) < sub_threshold * 2
        # Below baseline (subscript) — position signal OR size+baseline signal
        if is_below or (is_smaller and is_at_baseline):
            c["subscript"] = True
        # Superscript detection disabled — too many false positives from
        # page numbers, running headers, and inline digits at different baselines.
        # Subscript-only detection is reliable for chemistry (H₂O, NH₄, Fe^{2+}).
        # For superscripts (rare in scientific text), users should use --force-ocr.
        # elif c["bbox"][3] > median_top + sup_threshold:
        #     ...disabled...

    return chars


def _extract_scripts_pymupdf(pdf_path, page_num):
    """PyMuPDF-based extraction with subscript detection.

    PyMuPDF (fitz) returns text in correct reading order for multi-column
    documents and provides per-char bbox + font size. This makes it
    dramatically better than pypdfium2 for scientific papers (82% of which
    are 2-column).
    """
    import pymupdf as fitz

    try:
        doc = fitz.open(pdf_path)
        try:
            if page_num < 1 or page_num > len(doc):
                return None
            page = doc[page_num - 1]
            d = page.get_text("rawdict")
            if not d.get("blocks"):
                return ""

            # Gather chars in reading order (PyMuPDF handles column order)
            all_chars = []
            for block in d["blocks"]:
                if block.get("type") != 0:
                    continue  # skip images
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        for char in span.get("chars", []):
                            # PyMuPDF bbox: (x0, y0, x1, y1) top-left origin (y increases DOWN)
                            bbox = char["bbox"]
                            h = bbox[3] - bbox[1]
                            w = bbox[2] - bbox[0]
                            all_chars.append(
                                {
                                    "text": char["c"],
                                    "bbox": (bbox[0], bbox[1], bbox[2], bbox[3]),
                                    "font_size": h if h > 0 else 0,
                                    "has_geometry": h > 0 and w > 0,
                                }
                            )

            if not all_chars:
                return ""

            # Split into lines using \n markers from PyMuPDF block structure.
            # PyMuPDF's rawdict gives lines within blocks — natural line breaks.
            lines = [[]]
            for block in d["blocks"]:
                if block.get("type") != 0:
                    continue
                for line in block.get("lines", []):
                    line_chars = []
                    for span in line.get("spans", []):
                        for char in span.get("chars", []):
                            bbox = char["bbox"]  # PyMuPDF native: y increases DOWN
                            h = bbox[3] - bbox[1]
                            w = bbox[2] - bbox[0]
                            line_chars.append(
                                {
                                    "text": char["c"],
                                    "bbox": bbox,
                                    "font_size": h if h > 0 else 0,
                                    "has_geometry": h > 0 and w > 0,
                                }
                            )
                    if line_chars:
                        lines.append(line_chars)
            lines = [l for l in lines if l]

            output_lines = []
            for line_chars in lines:
                geo_count = sum(1 for c in line_chars if c.get("has_geometry"))
                if geo_count >= 2:
                    line_chars = detect_subscripts_pymupdf(line_chars)
                else:
                    for c in line_chars:
                        c["subscript"] = False
                        c["superscript"] = False

                # Build text with sub/superscript markers
                out = ""
                i = 0
                while i < len(line_chars):
                    c = line_chars[i]
                    if c.get("subscript"):
                        run = c["text"]
                        j = i + 1
                        while j < len(line_chars) and line_chars[j].get("subscript"):
                            run += line_chars[j]["text"]
                            j += 1
                        out += "$_{" + run + "}$"
                        i = j
                    elif c.get("superscript"):
                        run = c["text"]
                        j = i + 1
                        while j < len(line_chars) and line_chars[j].get("superscript"):
                            run += line_chars[j]["text"]
                            j += 1
                        out += "$^{" + run + "}$"
                        i = j
                    else:
                        out += c["text"]
                        i += 1
                output_lines.append(out)

            return "\n".join(output_lines)
        finally:
            doc.close()
    except Exception as e:
        log.warning("post-processor failed: %s", e)
        return None


def _extract_scripts_pypdfium(pdf_path, page_num):
    """pypdfium2 fallback — single-column papers only (sanity check inside)."""
    try:
        # PERF-1: shared._pdfium is PEP 562 lazy — attr access loads it
        # on first use; wildcard no longer exports it (shared.__all__).
        import shared as _shared_mod

        doc = _shared_mod._pdfium.PdfDocument(pdf_path)
        try:
            page = doc.get_page(page_num - 1)
            tp = page.get_textpage()
            n_chars = tp.count_chars()
            if n_chars == 0:
                return ""

            # Gather per-char data. Use full_text once for efficiency.
            full_text = tp.get_text_range()
            if len(full_text) != n_chars:
                # Index mismatch — fall back to no-script output
                record_path_warning(
                    SCRIPT_EXTRACTION_FALLBACK,
                    "pypdfium2 index mismatch — subscript detection dropped, plain text kept",
                )
                return full_text.replace("\r", "")

                record_path_warning(
                    SCRIPT_EXTRACTION_FALLBACK,
                    "pypdfium2 index mismatch — subscript detection dropped, plain text kept",
                )
                return full_text.replace("\r", "")
            all_chars = []
            for i in range(n_chars):
                bbox = tp.get_charbox(i)  # (left, bottom, right, top) in PDF points
                char_text = full_text[i]
                h = bbox[3] - bbox[1]
                w = bbox[2] - bbox[0]
                all_chars.append(
                    {
                        "text": char_text,
                        "bbox": bbox,
                        "font_size": h if h > 0 else 0,
                        "has_geometry": h > 0 and w > 0,
                    }
                )

            # Split into lines using \r\n markers in text (natural break).
            # PDF encodes visual line breaks via these control chars.
            lines = [[]]
            for c in all_chars:
                if c["text"] == "\r":
                    continue  # \r\n pair — \n will trigger break
                if c["text"] == "\n":
                    lines.append([])
                    continue
                lines[-1].append(c)
            lines = [l for l in lines if l]

            # For each line: sort left-to-right, run assign_scripts, build output
            output_lines = []
            for line_chars in lines:
                # Sort left-to-right by x position
                line_chars.sort(key=lambda c: c["bbox"][0])
                # Only run script detection if line has ≥2 chars with geometry
                geo_count = sum(1 for c in line_chars if c.get("has_geometry"))
                if geo_count >= 2:
                    line_chars = assign_scripts_to_chars(line_chars)
                else:
                    for c in line_chars:
                        c["subscript"] = False
                        c["superscript"] = False
                # Build text with sub/superscript markers
                out = ""
                i = 0
                while i < len(line_chars):
                    c = line_chars[i]
                    if c.get("subscript"):
                        # Group consecutive subscripts
                        run = c["text"]
                        j = i + 1
                        while j < len(line_chars) and line_chars[j].get("subscript"):
                            run += line_chars[j]["text"]
                            j += 1
                        out += "$_{" + run + "}$"
                        i = j
                    elif c.get("superscript"):
                        run = c["text"]
                        j = i + 1
                        while j < len(line_chars) and line_chars[j].get("superscript"):
                            run += line_chars[j]["text"]
                            j += 1
                        out += "$^{" + run + "}$"
                        i = j
                    else:
                        out += c["text"]
                        i += 1
                output_lines.append(out)

            # Sanity check: compare to pdftotext output. If they diverge
            # significantly (e.g. 2-column papers where pypdfium2 garbles
            # reading order), fall back to plain pdftotext. Subscript
            # detection is unreliable when char order doesn't match.
            try:
                proc = subprocess.run(
                    [
                        "pdftotext",
                        "-f",
                        str(page_num),
                        "-l",
                        str(page_num),
                        pdf_path,
                        "-",
                    ],
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=10,
                    check=True,
                )
                pdftotext_norm = re.sub(r"\s+", " ", proc.stdout).strip()[:500]
                # Strip our LaTeX markers + normalize whitespace for comparison
                our_norm = re.sub(r"\$_\{[^}]+\}\$|\$\^\{[^}]+\}\$", "", "\n".join(output_lines))
                our_norm = re.sub(r"\s+", " ", our_norm).strip()[:500]
                # If first 200 normalized chars differ by >30%, pypdfium2 garbled it
                from difflib import SequenceMatcher

                ratio = SequenceMatcher(None, pdftotext_norm[:200], our_norm[:200]).ratio()
                if ratio < 0.85:
                    # pypdfium2 reading order diverges from pdftotext — likely
                    # 2-column or complex layout. Bail and return plain text.

                    record_path_warning(
                        SCRIPT_EXTRACTION_FALLBACK,
                        "pypdfium2 reading order diverges from pdftotext "
                        "(>15%) — subscript extraction abandoned, plain "
                        "pdftotext used",
                    )
                    return None
            except Exception:
                pass  # if sanity check fails, proceed with our output

            return "\n".join(output_lines)
        finally:
            doc.close()
    except Exception:
        import traceback

        traceback.print_exc()

        record_path_warning(
            SCRIPT_EXTRACTION_FALLBACK,
            "pypdfium2 script extraction crashed — plain pdftotext used",
        )
        return None


def _build_math_regex():
    """Build comprehensive regex covering math/science Unicode chars.

    Scans these Unicode blocks (1610+ chars total):
      - Greek and Coptic (U+0370-03FF)        — all fields
      - Greek Extended (U+1F00-1FFF)          — classics, linguistics
      - Mathematical Operators (U+2200-22FF)  — math, physics, engg
      - Misc Math Symbols-A (U+27C0-27EF)     — math, CS
      - Misc Math Symbols-B (U+2980-29FF)     — advanced math
      - Supplemental Math Operators (U+2A00-2AFF) — advanced math
      - Arrows (U+2190-21FF)                  — all fields
      - Supplemental Arrows-A/B (U+27F0-297F) — physics, CS
      - Letterlike Symbols (U+2100-214F)      — math constants (ℕℤℚℝℂℏ)
      - Misc Technical (U+2300-23FF)          — engineering
      - Geometric Shapes (U+25A0-25FF)        — logic, chemistry
      - Misc Symbols (U+2600-26FF)            — astronomy (☉⊕♀♂♃♄)
      - Subscripts/Superscripts (U+2070-209F) — chemistry (H₂O, x²)
      - Number Forms (U+2150-218F)            — fractions (½⅓⅔)
      - Currency Symbols (U+20A0-20CF)        — economics
      - Math Alphanumeric (U+1D400-1D7FF)     — bold/italic/script math
    """
    import unicodedata

    chars = set()

    # All Unicode blocks containing math/science notation
    math_blocks = [
        (0x0370, 0x04FF),  # Greek and Coptic
        (0x1F00, 0x2000),  # Greek Extended
        (0x2070, 0x20CF),  # Superscripts/Subscripts + Currency
        (0x2100, 0x2150),  # Letterlike Symbols
        (0x2150, 0x2190),  # Number Forms
        (0x2190, 0x2200),  # Arrows
        (0x2200, 0x2300),  # Mathematical Operators
        (0x2300, 0x2400),  # Misc Technical
        (0x25A0, 0x2600),  # Geometric Shapes
        (0x2600, 0x2700),  # Misc Symbols (astronomical)
        (0x27C0, 0x27F0),  # Misc Math Symbols-A
        (0x27F0, 0x2800),  # Supplemental Arrows-A
        (0x2900, 0x2980),  # Supplemental Arrows-B
        (0x2980, 0x2A00),  # Misc Math Symbols-B
        (0x2A00, 0x2B00),  # Supplemental Math Operators
        (0x1D400, 0x1D800),  # Mathematical Alphanumeric Symbols
    ]

    for start, end in math_blocks:
        for cp in range(start, end):
            try:
                c = chr(cp)
                cat = unicodedata.category(c)
                # Skip control/format chars
                if cat.startswith("C"):
                    continue
                # Include ALL chars from math/science Unicode blocks.
                # These blocks are intentionally Unicode-grouped math content
                # regardless of category (Letter for ℕℤℝ, Symbol for ∑∫,
                # Number for ½⅓⅔). The block membership IS the math signal.
                chars.add(c)
            except ValueError as e:
                log.debug("_build_math_regex: ValueError degraded: %s", e)

    # ASCII math relations — unambiguously math, rare in prose/code
    ascii_math_relations = "≈≠≤≥±÷×"
    # ASCII vulgar fractions (½⅓¼ etc.) — in Latin-1 Supplement, not Number Forms
    ascii_fractions = "½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞"
    # Latin-1 Supplement math chars — superscripts ¹²³, degree °, vulgar fractions
    # These are NOT in the Unicode "Superscripts" block (which starts at U+2070)
    # but in Latin-1 Supplement (U+0080-00FF). Frequently used in scientific text.
    latin1_math = "°±²³·¹×÷¼½¾µ"  # degree, plus-minus, sup2, sup3, middot, sup1, times, divide, fractions, micro
    # NOTE: We deliberately exclude +, -, =, <, >, /, *, (), [] from the regex.
    # These are ambiguous — equally common in prose, code, and math.
    # Unicode math symbols (Greek, operators, arrows) are unambiguous signals
    # that the content is math. ASCII operators are only counted as math in
    # detect_math_regions when ADJACENT to Unicode math symbols.
    for c in ascii_math_relations + ascii_fractions + latin1_math:
        chars.add(c)

    # Build character class — sort by codepoint for stability
    sorted_chars = sorted(chars, key=lambda c: ord(c))
    pattern = "[" + re.escape("".join(sorted_chars)) + "]"
    return re.compile(pattern)


__all__ = [
    "detect_subscripts_pymupdf",
    "assign_scripts_to_chars",
    "_extract_scripts_pymupdf",
    "_extract_scripts_pypdfium",
    "_build_math_regex",
]
