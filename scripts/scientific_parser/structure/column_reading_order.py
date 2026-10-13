"""Layout — column detection, XY-Cut++ reading order, block classification.

Extracted from scripts/pdf_ocr.py during Phase R7 refactoring (2026-08-12).

Functions:
  - _xy_cut_partition: XY-Cut++ recursive bisection (arXiv 2504.10258)
  - classify_text_block: classify a text block as title/caption/body/etc
  - column_aware_body_only: separate body text from figure regions
  - column_aware_reading_order: legacy XY-cut (pre-XY-Cut++ port)
"""

from __future__ import annotations

import re
import logging

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state


def _dehyphenating_flatten(text):
    """Flatten block text, joining hyphenation wraps (iter-60).

    fitz lines carry the ORIGINAL typesetting's hyphenated line breaks
    ('varal-\\nta'); the old ``replace("\\n", " ")`` flatten turned them
    into 'varal- ta', which the downstream dehyphenate (order 230) can
    no longer match (it needs the newline). Gold text carries the
    UNBROKEN form. Join lowercase-'-' / lowercase wraps without the
    hyphen; PRESERVE_HYPHENS compounds and uppercase forms (U-Pb,
    Fe-bearing) keep theirs.
    """
    parts = text.split("\n")
    out = parts[0].rstrip() if parts else ""
    for p in parts[1:]:
        p = p.strip()
        if not p:
            continue
        if out.endswith("-") and p[:1].islower():
            head_word = out[:-1].split()[-1] if out[:-1].split() else ""
            tail_word = p.split()[0] if p.split() else ""
            joined = f"{head_word}{tail_word}"
            hyphened = f"{head_word}-{tail_word}"
            if out[:-1].endswith((" ", "")) or (head_word and head_word[-1:].islower()):
                if hyphened not in PRESERVE_HYPHENS:
                    # COMPOUND-PREFIX rule (iter-110, 068682d5 pg1): a
                    # break after a known compound prefix with a complete
                    # dictionary word following is the compound's OWN
                    # hyphen ('non-sequential', 'single-loop' — TeX breaks
                    # compounds AT the hyphen; gold keeps it, joined
                    # TIGHT), not a syllable hyphenation ('varal-ta' —
                    # tail is not a word; gold carries the unbroken form).
                    if _is_compound_hyphen(head_word, tail_word):
                        out = out + p
                        continue
                    out = out[:-1] + p
                    continue
        out = out + " " + p
    return out


_COMPOUND_PREFIXES = frozenset(
    """
    non multi single semi anti auto co pre post sub super inter intra
    ultra micro macro nano quasi self cross well over under out mid bio
    geo neuro proto pseudo counter hyper hypo ex extra re un up down
    """.split()
)


_WORDLIST_CACHE = None


def _is_compound_hyphen(head_word, tail_word):
    """True when a line-break hyphen is a compound's own hyphen: head is a
    known prefix and tail is a complete dictionary word."""
    global _WORDLIST_CACHE
    if head_word.lower() not in _COMPOUND_PREFIXES or not tail_word:
        return False
    if _WORDLIST_CACHE is None:
        try:
            from scientific_parser.postproc.text_pipeline import _english_wordlist

            _WORDLIST_CACHE = _english_wordlist() or frozenset()
        except Exception:
            _WORDLIST_CACHE = frozenset()
    return tail_word.lower() in _WORDLIST_CACHE


def _join_line_parts(parts):
    """Join same-line block parts, merging a trailing hyphen wrap across
    the block boundary (block A ends 'maahantuo-', block B starts 'daan'
    → 'maahantuodaan'; gold carries unbroken forms)."""
    out = parts[0] if parts else ""
    for p in parts[1:]:
        if out.endswith("-") and p[:1].islower():
            head_word = out[:-1].split()[-1] if out[:-1].split() else ""
            tail_word = p.split()[0] if p.split() else ""
            if (
                head_word
                and head_word[-1:].islower()
                and (f"{head_word}-{tail_word}" not in PRESERVE_HYPHENS)
            ):
                out = out[:-1] + p
                continue
        out = out + " " + p
    return out


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


_CONSTS_LOADED = False


def _load_consts():
    """Lazy-load cross-module constants on first call."""
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    for name in (
        "_page_dict_cache",
        "_get_page_dict",
        "classify_text_block",
        "_PRIVATE_CHAR_RE",
        "MATH_FONT_PREFIXES",
        "IS_IMAGE",
        "PDF_PATH",
        "_FIGURE_TABLE_OCR",
        "detect_tables_in_figure",
    ):
        if hasattr(p, name):
            g[name] = getattr(p, name)
    _CONSTS_LOADED = True


def _xy_cut_partition(blocks, depth=0):
    """Recursive XY-cut with XY-Cut++ (arXiv 2504.10258) improvements.

    Improvements over legacy XY-cut:
      1. **Pre-mask processing**: noise blocks (page numbers, header artifacts,
         tiny isolated spans) are excluded from gap analysis. They pollute
         whitespace detection. (XY-Cut++ §3.1)
      2. **Multi-granularity gap threshold**: try multiple thresholds
         (0.10x / 0.15x / 0.20x median width) and pick the cleanest cut.
         (XY-Cut++ §3.2 multi-granularity segmentation)
      3. **Stability gate**: a cut is accepted only if both sides have ≥2
         blocks AND the cut position is unambiguous (clear gap, not a split
         inside a column). (XY-Cut++ §3.3 hierarchical mask)

    Layout-independent reading order. Discovers column structure from gaps.
    Handles 2-col, 3-col, sidebars, boxed text, full-width elements.
    """
    _load_consts()
    _load_consts()
    if len(blocks) <= 1:
        return list(blocks)

    # ─── Pre-mask: filter noise blocks from gap analysis ──────────────── #
    # Noise = very small area OR very short text. These pollute the gap
    # detection (e.g., a stray page number creates a false column edge).
    # They are still OUTPUT in the result, just not used to find cuts.
    median_area = sorted((b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) for b in blocks)
    median_area = median_area[len(median_area) // 2] if median_area else 100
    area_noise_threshold = median_area * 0.05  # <5% of median area = noise
    text_noise_threshold = 3  # <3 chars = likely page number / artifact

    gap_blocks = [
        b
        for b in blocks
        if (b["x1"] - b["x0"]) * (b["y1"] - b["y0"]) >= area_noise_threshold
        and len(b.get("text", "").strip()) >= text_noise_threshold
    ]
    # If filtering removed too much, fall back to full set
    if len(gap_blocks) < max(2, len(blocks) // 2):
        gap_blocks = blocks

    # ─── Multi-granularity gap threshold ──────────────────────────────── #
    # Try 3 thresholds; pick the one producing the cleanest (most balanced) cut.
    _ws = sorted(b["x1"] - b["x0"] for b in gap_blocks)
    median_w = _ws[len(_ws) // 2] if _ws else 50
    _hs = sorted(b["y1"] - b["y0"] for b in gap_blocks)
    median_h = _hs[len(_hs) // 2] if _hs else 20

    def _find_gaps(intervals_sorted, min_gap):
        """Return list of (gap_size, start, end) for gaps > min_gap."""
        merged = [list(intervals_sorted[0])]
        for x0, x1 in intervals_sorted[1:]:
            if x0 <= merged[-1][1] + 2:
                merged[-1][1] = max(merged[-1][1], x1)
            else:
                merged.append([x0, x1])
        return [
            (merged[i][0] - merged[i - 1][1], merged[i - 1][1], merged[i][0])
            for i in range(1, len(merged))
            if merged[i][0] - merged[i - 1][1] > min_gap
        ]

    # Vertical cut (column separation) — multi-granularity
    x_sorted = sorted((b["x0"], b["x1"]) for b in gap_blocks)
    best_v_cut = None  # (gap_size, cut_x, score)
    for thr_mult in (0.10, 0.15, 0.20):
        gap_threshold = max(median_w * thr_mult, 4.0)
        v_gaps = _find_gaps(x_sorted, gap_threshold)
        if not v_gaps:
            continue
        largest = max(v_gaps)
        # Score = gap size × stability (both sides have ≥3 blocks)
        cut_x = (largest[1] + largest[2]) / 2
        left_n = sum(1 for b in blocks if b["cx"] < cut_x)
        right_n = sum(1 for b in blocks if b["cx"] >= cut_x)
        if left_n >= 2 and right_n >= 2:
            balance = min(left_n, right_n) / max(left_n, right_n)
            score = largest[0] * balance
            if best_v_cut is None or score > best_v_cut[2]:
                best_v_cut = (largest[0], cut_x, score)

    # Horizontal cut (section break) — multi-granularity
    y_sorted = sorted((b["y0"], b["y1"]) for b in gap_blocks)
    best_h_cut = None
    for thr_mult in (0.30, 0.50, 0.70):
        gap_threshold = max(median_h * thr_mult, 10.0)
        h_gaps = _find_gaps(y_sorted, gap_threshold)
        if not h_gaps:
            continue
        largest = max(h_gaps)
        cut_y = (largest[1] + largest[2]) / 2
        top_n = sum(1 for b in blocks if b["y1"] < cut_y + 5)
        bottom_n = sum(1 for b in blocks if b["y0"] >= cut_y - 5)
        if top_n >= 2 and bottom_n >= 2:
            balance = min(top_n, bottom_n) / max(top_n, bottom_n)
            score = largest[0] * balance
            if best_h_cut is None or score > best_h_cut[2]:
                best_h_cut = (largest[0], cut_y, score)

    # ─── Stability gate: pick best cut, fallback to no-cut ────────────── #
    # Vertical preferred if it's competitive with horizontal (columns
    # are more common than horizontal sections in scientific papers).
    if best_v_cut and (best_h_cut is None or best_v_cut[2] >= best_h_cut[2] * 0.7):
        cut_x = best_v_cut[1]
        left = [b for b in blocks if b["cx"] < cut_x]
        right = [b for b in blocks if b["cx"] >= cut_x]
        if left and right:
            return _xy_cut_partition(left, depth + 1) + _xy_cut_partition(right, depth + 1)

    if best_h_cut:
        cut_y = best_h_cut[1]
        top = [b for b in blocks if b["y1"] < cut_y + 5]
        bottom = [b for b in blocks if b["y0"] >= cut_y - 5]
        if top and bottom:
            return _xy_cut_partition(top, depth + 1) + _xy_cut_partition(bottom, depth + 1)

    # No stable cut: single column, sort by Y then X
    return sorted(blocks, key=lambda b: (b["y0"], b["x0"]))


def _merged_lines(lines, col_mid=None):
    """Merge fitz "lines" sharing a visual baseline; return list of dicts.

    Fragment fonts (T11-T18 subset producers) and TEXT_PRESERVE_WHITESPACE
    both make MuPDF emit per-fragment "lines" — kerning fragments ('Exp'
    'erimen' 'ts') or even single words ('This' 'pap' 'er') as separate
    line dicts sharing one baseline. Per-line figure heuristics then read
    the stack as axis labels, and naive joining inserts spaces inside
    words ('plemen t'). Baseline tolerance 2pt; fragment join when x-gap
    < 2pt (measured kerning gaps ≤0.5pt; word gaps ≥2.4pt at 9.5pt body).

    col_mid: when set, a same-baseline gap ≥6pt whose position falls
    within col_mid±20 is a COLUMN GUTTER (measured gutters 8-15pt; full-
    width-title word gaps ≤5.5pt) — force a line break there so columns
    never merge ('IE sys'+'knowledge-based' corruption, 09f90a8f).
    """
    entries = []
    for line in lines:
        text = "".join(s["text"] for s in line.get("spans", []))
        if not text.strip():
            continue
        bb = line.get("bbox")
        if bb:
            entries.append((bb[1], bb[0], bb[2], bb[3], text))
        else:
            entries.append((0.0, 0.0, 0.0, 0.0, text))
    entries.sort(key=lambda e: (e[0], e[1]))
    merged = []
    for y0, x0, x1, y1, text in entries:
        gap = None
        if merged and abs(y0 - merged[-1]["y0"]) <= 2:
            gap = x0 - merged[-1]["x1"]
            gutter = (
                col_mid is not None
                and gap >= 6
                and abs((merged[-1]["x1"] + x0) / 2 - col_mid) <= 20
            )
            if not gutter:
                prev = merged[-1]
                prev["text"] = prev["text"] + text if gap < 2 else prev["text"] + " " + text
                prev["x1"] = max(prev["x1"], x1)
                prev["y1"] = max(prev["y1"], y1)
                continue
        merged.append({"text": text, "x0": x0, "x1": x1, "y0": y0, "y1": y1})
    return merged


def _merge_baseline_lines(lines, col_mid=None):
    """Text form of _merged_lines: one visual line per output line."""
    return "\n".join(m["text"] for m in _merged_lines(lines, col_mid))


def classify_text_block(text):
    """Classify a PDF text block as BODY / CAPTION / figure-region.

    Per-line stats catch multi-line figure-label stacks (axis ticks stacked
    vertically) that whole-block heuristics miss. Returns one of:
    BODY, CAPTION, FIGURE_GLYPH, FIGURE_TICK, FIGURE_LABEL_BLOCK,
    AXIS_TITLE, LEGEND_MARKER, EMPTY.

    Non-ML: pure regex + per-line length/word-count statistics.
    """
    _load_consts()
    _load_consts()
    text = text.strip()
    if not text:
        return "EMPTY"
    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return "EMPTY"
    # Caption: starts with Fig/Figure/Table + number
    if re.match(r"^(Fig\.?|Figure|Table)\s*\d", text, re.IGNORECASE):
        return "CAPTION"
    # Figure glyph: PDF text layer contains line-art rendering
    # (::, ==, ++, .., ~~ sequences = curve/line drawings encoded as text)
    if re.search(r"::{3,}|={3,}|\+{3,}|\.{4,}|~{3,}|={5,}", text):
        return "FIGURE_GLYPH"
    # Per-line statistics
    line_lens = [len(l.strip()) for l in lines]
    avg_len = sum(line_lens) / len(line_lens)
    short_lines = sum(1 for L in line_lens if L <= 5)
    short_ratio = short_lines / len(line_lens)
    # Multi-line stack of short labels (figure axis ticks stacked vertically)
    if len(lines) >= 2 and avg_len <= 8 and short_ratio >= 0.5:
        return "FIGURE_LABEL_BLOCK"
    # Single-line cases
    if len(lines) == 1:
        if re.match(r"^[\-+]?\d+(\.\d+)?[\-+]?$", text):
            return "FIGURE_TICK"
        if len(text) <= 5 and re.match(r"^[\W\d\-+.]+$", text):
            return "FIGURE_TICK"
        if re.match(r"^[\[\]\(\)\+\*x□○◇•]+$", text):
            return "LEGEND_MARKER"
    # Axis title: short, contains math/log/unit-paren, no sentence-ending
    # period. iter-53: the paren probe must be CLOSED and unit-like
    # ('log(x)', 'v (m/s)') — a bare open paren ('combination of (')
    # fired on 54-char prose lines and dumped body text into the
    # figure region (09f90a8f: 'tems successful are replaced by a
    # combination of (' → AXIS_TITLE).
    if (
        len(text) < 60
        and not text.endswith(".")
        and len(text.split()) <= 6
        and re.search(
            r"\blog\b|/\s*[A-Z]\s*\(|[a-z]\s*\(\s*(?=[a-zA-Z/%µ°][a-zA-Z0-9/%µ°.+-]{0,5}\s*\))",
            text,
            re.IGNORECASE,
        )
    ):
        return "AXIS_TITLE"
    # Default: body text (sentence-like)
    return "BODY"


# =============================================================================
# SOTA specialist providers — extracted to scientific_parser/perception/providers
# =============================================================================
# Phase R4 refactoring (2026-08-12): provider classes + region router moved to
# scripts/scientific_parser/perception/providers/. All names re-exported below
# for backwards compatibility (external QC tests import from pdf_ocr directly).


def _assemble_fragment_blocks(blocks, col_mid):
    """Rebuild line-coherent blocks on word-fragment pages.

    Some producers (T11-T18 subset fonts) emit EVERY word — or word
    fragment ('F'+'urthermore', 'metho'+'ds') — as its own fitz BLOCK.
    Per-block merging can't rejoin across blocks, and classify sees
    single tokens ('tems') with no line context. Gate: ≥60% of text
    blocks are tiny (≤15 chars). Assembly: merge same-baseline lines
    page-wide (fragment gap <2pt → concat; word gap → space; gap ≥6pt
    at the column midline → line break), then group consecutive lines
    into column-coherent blocks by x-overlap. Returns synthetic blocks
    dict shape, or None when the page is not a fragment page.
    """
    text_blocks = [b for b in blocks if b.get("type", 0) == 0]
    if len(text_blocks) < 20:
        return None
    # Harvest all lines page-wide (needed by BOTH gates).
    lines = []
    for b in text_blocks:
        for l in b.get("lines", []):
            text = "".join(s["text"] for s in l.get("spans", []))
            if text.strip() and l.get("bbox"):
                lines.append(l)
    tiny = 0
    for b in text_blocks:
        t = "".join(s["text"] for l in b.get("lines", []) for s in l.get("spans", [])).strip()
        if len(t) <= 15:
            tiny += 1
    size_gate = tiny >= 0.6 * len(text_blocks)
    # GEOMETRIC GATE (iter-73, 09f90a8f page_2): block-size fraction misses
    # fragment pages where a large figure shares the page (31% tiny blocks,
    # yet 27% of same-baseline adjacent line pairs TOUCH at <0.45pt —
    # pieces of one word; normal kerned text never sits that close).
    # Gate on the touching fraction instead; either gate suffices.
    by_y = sorted((l["bbox"], "".join(s["text"] for s in l.get("spans", []))) for l in lines)
    touch = total = 0
    for a, b in zip(by_y, by_y[1:]):
        (ax0, ay0, ax1, _), (bx0, by0, _, _) = a[0], b[0]
        if abs(ay0 - by0) > 2 or bx0 < ax1:
            continue
        gap = bx0 - ax1
        if gap < 0.45:
            touch += 1
        total += 1
    geo_gate = total >= 20 and touch >= 0.12 * total
    if not (size_gate or geo_gate):
        return None
    merged = _merged_lines(lines, col_mid=col_mid)
    if not merged:
        return None
    # Group consecutive lines into column-coherent blocks (x-overlap),
    # so multi-line stacks stay visible to classify_text_block.
    synth = []
    cur = None
    for m in merged:
        if cur and min(cur["x1"], m["x1"]) - max(cur["x0"], m["x0"]) > 4:
            cur["lines"].append(m)
            cur["x0"] = min(cur["x0"], m["x0"])
            cur["x1"] = max(cur["x1"], m["x1"])
            cur["y1"] = m["y1"]
        else:
            if cur:
                synth.append(cur)
            cur = {
                "x0": m["x0"],
                "x1": m["x1"],
                "y0": m["y0"],
                "y1": m["y1"],
                "lines": [m],
            }
    if cur:
        synth.append(cur)
    out_blocks = []
    for s in synth:
        out_blocks.append(
            {
                "type": 0,
                "bbox": (s["x0"], s["y0"], s["x1"], s["y1"]),
                "lines": [
                    {"bbox": (m["x0"], m["y0"], m["x1"], m["y1"]), "spans": [{"text": m["text"]}]}
                    for m in s["lines"]
                ],
            }
        )
    return out_blocks


def column_aware_body_only(pdf_path, page_num):
    """Reconstruct reading order separating body text from figure regions.

    Replacement for ``column_aware_reading_order`` on born-digital pages with
    figures. Produces CLEAN body text in correct 2-column reading order, with
    figure axis labels / glyphs / legend markers moved to a separated
    ``<!-- FIGURE_REGION -->`` block at the end (preserved verbatim, nothing
    lost — per no-loss mandate).

    Algorithm (non-ML):
      1. Extract text blocks via PyMuPDF dict.
      2. Classify each block via ``classify_text_block``.
      3. Send BODY + CAPTION blocks through band-based reading-order
         reconstruction (group by 25pt Y-band, within band: full-width →
         left-col → right-col). Within each column slice, JOIN blocks
         whose X-gap < 20pt with a space (handles PDFs that encode each
         word as a separate positioned block — sakai1968, some old Elsevier
         scans, Adobe Clearscan).
      4. Collect non-body blocks (figure labels, glyphs, axis titles, legend
         markers) into a separate figure-region section, sorted by Y then X.
      5. Emit body text first, then ``<!-- FIGURE_REGION -->`` marker with
         the non-body blocks verbatim.

    Returns the reconstructed text (or None to fall back to pdftotext order).
    """
    _load_consts()
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return None
    try:
        page_width = pd["rect"].width
        mid_x = page_width / 2
        body_blocks = []
        figure_blocks = []
        page_blocks = (
            _assemble_fragment_blocks(pd["dict"]["blocks"], col_mid=mid_x) or pd["dict"]["blocks"]
        )
        for b in page_blocks:
            if b.get("type", 0) != 0:
                continue
            block_text = _merge_baseline_lines(b.get("lines", []), col_mid=mid_x).strip()
            if not block_text or len(block_text) < 3:
                continue
            cls = classify_text_block(block_text)
            bbox = b["bbox"]
            # iter-39 (audit-2, 2026-08-18): a fitz BLOCK may span the
            # column gutter with text from BOTH columns (measured
            # 0621a009 pg31: 'des Eag|leman' — left+right reference
            # entries inside one block). Block-level column assignment
            # then interleaves columns. Split such blocks at LINE
            # granularity: a real column line never crosses the gutter,
            # so line bboxes carry the true column identity.
            entries = []
            # iter-39 (audit-2, 2026-08-18): a fitz BLOCK may span the
            # column gutter with text from BOTH columns (measured
            # 0621a009 pg31: 'des Eag|leman' — left+right reference
            # entries inside one block). Split wide multi-line blocks at
            # LINE granularity: a real column line never crosses the
            # gutter, so line bboxes carry the true column identity.
            _gutter_span = bbox[2] - bbox[0] > page_width * 0.6 and len(b.get("lines", [])) > 1
            if _gutter_span and cls in ("BODY", "CAPTION"):
                for line in _merged_lines(b.get("lines", []), col_mid=mid_x):
                    lt = line["text"].strip()
                    if lt and len(lt) >= 3:
                        entries.append(
                            {
                                "text": lt,
                                "cls": cls,
                                "x0": line["x0"],
                                "y0": line["y0"],
                                "x1": line["x1"],
                                "y1": line["y1"],
                                "cx": (line["x0"] + line["x1"]) / 2,
                                "w": line["x1"] - line["x0"],
                            }
                        )
            else:
                _sub_entries = None
                # TABLE-GRID CELL SPLIT (iter-112, 09f90a8f pg7): a
                # text-layer table row arrives as ONE single-line block
                # whose spans glue label and value cells by x-order
                # ('Text-surface-based 77.7% 64.5%'). Split the line at
                # span x-gaps >= 25pt into cell fragments — the grid
                # pass below re-groups them column-major like poppler.
                _sp_all = []
                if cls in ("BODY", "CAPTION"):
                    for _ln in b.get("lines", []):
                        _sp_all.extend(_ln.get("spans", []))
                _rows_tmp = []
                for _s in sorted(_sp_all, key=lambda s: (s["bbox"][1], s["bbox"][0])):
                    if _rows_tmp and abs(_s["bbox"][1] - _rows_tmp[-1][0]) <= 3:
                        _rows_tmp[-1][1].append(_s)
                    else:
                        _rows_tmp.append([_s["bbox"][1], [_s]])
                for _ri, (_ry, _rsp) in enumerate(_rows_tmp):
                    _rsp = sorted(_rsp, key=lambda s: s["bbox"][0])
                    _groups = []
                    for _s in _rsp:
                        if _groups and _s["bbox"][0] - _groups[-1][1] < 25:
                            _groups[-1][1] = max(_groups[-1][1], _s["bbox"][2])
                            _groups[-1][2].append(_s)
                        else:
                            _groups.append([_s["bbox"][0], _s["bbox"][2], [_s]])
                    if len(_groups) < 3:
                        # 2-group rows are prose shapes (label + tail,
                        # 'S' + '2'); a table row carries label + >=2
                        # value cells (09f90a8f: label,77.7%,64.5%).
                        continue
                    if any(_gx1 - _gx0 > page_width * 0.3 for _gx0, _gx1, _g in _groups):
                        # A >=25pt gap between wide groups is the COLUMN
                        # GUTTER of 2-column prose (fragment ~0.34W), not
                        # a table cell — never fragment those rows
                        # (measured 02943986 p5: justified Spanish prose
                        # shredded, grid absorbed body lines, -7 tests).
                        continue
                    if any(
                        len(" ".join(s["text"].strip() for s in _gsp).strip()) < 3
                        for _gx0, _gx1, _gsp in _groups
                    ):
                        # micro-fragments ('S', '2' from 'S.2') are
                        # prose typography, never cells.
                        continue

                    def _join_spans(spans):
                        # TIGHT join: subset-font producers shred words
                        # into spans with sub-pt gaps ('T'+'ext-surfac' —
                        # 0.7pt); only real gaps become spaces.
                        out = ""
                        prev_x1 = None
                        for s in spans:
                            t = _dehyphenating_flatten(s["text"]).strip()
                            if not t:
                                continue
                            if prev_x1 is not None and s["bbox"][0] - prev_x1 >= 2:
                                out += " "
                            out += t
                            prev_x1 = s["bbox"][2]
                        return out

                    for _gx0, _gx1, _gsp in _groups:
                        _gt = _join_spans(_gsp)
                        if not _gt:
                            continue
                        entries.append(
                            {
                                "text": _gt,
                                "cls": cls,
                                "x0": _gx0,
                                "y0": _ry,
                                "x1": _gx1,
                                "y1": max(s["bbox"][3] for s in _gsp),
                                "cx": (_gx0 + _gx1) / 2,
                                "w": _gx1 - _gx0,
                                "_frag": True,
                                "_orig_text": _join_spans(_rsp),
                                "_orig_key": (id(b), _ri),
                            }
                        )
                if not any(e.get("_frag") for e in entries):
                    entries.append(
                        {
                            "text": block_text,
                            "cls": cls,
                            "x0": bbox[0],
                            "y0": bbox[1],
                            "x1": bbox[2],
                            "y1": bbox[3],
                            "cx": (bbox[0] + bbox[2]) / 2,
                            "w": bbox[2] - bbox[0],
                        }
                    )
            for entry in entries:
                if entry["cls"] in ("BODY", "CAPTION"):
                    body_blocks.append(entry)
                else:
                    figure_blocks.append(entry)
        if not body_blocks:
            return None
        # TABLE-GRID COLUMN GROUPING (iter-112, 09f90a8f pg7): poppler
        # reads a text-layer table COLUMN-GROUPED (all row labels, then
        # each value column) and gold follows. Detect aligned narrow
        # columns over >=4 tight y-rows of cell fragments and replace
        # the run with ONE pseudo-entry carrying the column-grouped
        # text, so the general assembly keeps its y-position.
        _frags = [e for e in body_blocks if e.get("_frag")]
        if len(_frags) >= 8:
            # HALF-CLUSTER: run one grid pass per page half — a left-
            # column table and a right-column table share y-rows and x-
            # gaps cannot tell column-gap from table-gap (measured:
            # right-column 'Incorrect answers…' values merged into the
            # left table's run). Straddling fragments (full-width
            # tables) are left alone.
            _clusters = [
                [e for e in _frags if e["cx"] < mid_x],
                [e for e in _frags if e["cx"] >= mid_x],
            ]
            for _cluster in _clusters:
                _cfrags = sorted(_cluster, key=lambda e: (e["y0"], e["x0"]))
                _rows = []
                for e in _cfrags:
                    if _rows and abs(e["y0"] - _rows[-1][0]["y0"]) <= 4:
                        _rows[-1].append(e)
                    else:
                        _rows.append([e])
                # SINGLE-RUN (label blocks interleave data rows at >22pt
                # gaps — 'Question Processing' between row 2 and row 3 —
                # so run-splitting on gap shreds one table into two <4-row
                # runs). The y-extent cap below bounds the grid.
                _run = _rows
                if len(_run) < 4:
                    continue
                _all = [e for _r in _run for e in _r]
                _iv = sorted((e["x0"], e["x1"]) for e in _all)
                _cols = [[_iv[0][0], _iv[0][1], [_iv[0]]]]
                for _x0, _x1 in _iv[1:]:
                    if _x0 - _cols[-1][1] >= 15:
                        _cols.append([_x0, _x1, [(_x0, _x1)]])
                    else:
                        _cols[-1][1] = max(_cols[-1][1], _x1)
                        _cols[-1][2].append((_x0, _x1))
                _cols = [c for c in _cols if len(c[2]) >= 3]
                _yext = _run[-1][0]["y0"] - _run[0][0]["y0"]
                if not (
                    len(_cols) >= 2
                    and all(c[1] - c[0] <= page_width * 0.3 for c in _cols)
                    and _yext <= 0.5 * pd["rect"].height
                    and (max(e["x1"] for e in _all) - min(e["x0"] for e in _all))
                    <= page_width * 0.6
                ):
                    continue
                # ABSORB non-fragment single-line label blocks inside a
                # column's x-window and the grid's y-extent (poppler
                # interleaves them into the label column: 'Text-surface-
                _ylo = min(e["y0"] for e in _all) - 2
                _yhi = max(e["y1"] for e in _all) + 2
                _xlo = min(e["x0"] for e in _all) - 4
                _xhi = max(e["x1"] for e in _all) + 4
                _label_w = (_cols[0][0] - 14, _cols[0][1] + 14)
                _absorbed = list(_all)
                for e in body_blocks:
                    if (
                        e.get("_frag")
                        or e["cls"] != "BODY"
                        or "\n" in e["text"]
                        or not (4 <= len(e["text"]) <= 60)
                        or not (_xlo <= e["cx"] <= _xhi)
                    ):
                        continue
                    # LABEL-STACK TAIL (iter-112): a row label's stacked
                    # continuation lines ('with Answer' / 'Justification'
                    # under the last row's label) run up to ~26pt BELOW
                    # the last value row — same x-window as the label
                    # column. Value-column windows keep the tight bound.
                    _ylimit = _yhi + 26 if _label_w[0] <= e["cx"] <= _label_w[1] else _yhi
                    if not (_ylo <= e["y0"] <= _ylimit):
                        continue
                    for _cx0, _cx1, _civ in _cols:
                        if _cx0 - 12 <= e["cx"] <= _cx1 + 12 and e["w"] <= (_cx1 - _cx0) * 2.2 + 24:
                            _absorbed.append(e)
                            break
                _ids = {id(e) for e in _absorbed}
                _lines_out = []
                for _cx0, _cx1, _civ in _cols:
                    _col_frags = sorted(
                        (e for e in _absorbed if _cx0 <= e["cx"] <= _cx1),
                        key=lambda e: e["y0"],
                    )
                    _lines_out.extend(e["text"] for e in _col_frags)
                if _lines_out:
                    body_blocks = [e for e in body_blocks if id(e) not in _ids]
                    body_blocks.append(
                        {
                            "text": "\n".join(_lines_out),
                            "cls": "BODY",
                            "x0": min(e["x0"] for e in _absorbed),
                            "y0": min(e["y0"] for e in _absorbed),
                            "x1": max(e["x1"] for e in _absorbed),
                            "y1": max(e["y1"] for e in _absorbed),
                            "cx": (
                                min(e["x0"] for e in _absorbed) + max(e["x1"] for e in _absorbed)
                            )
                            / 2,
                            "w": max(e["x1"] for e in _absorbed) - min(e["x0"] for e in _absorbed),
                        }
                    )
        # RESTORE unconsumed fragments (iter-112): fragments whose block
        # was NOT absorbed into a confirmed grid go back to their whole
        # line form — a tentative split must never leave prose shredded.
        _leftover = [e for e in body_blocks if e.get("_frag")]
        if _leftover:
            _by_orig = {}
            for e in _leftover:
                _by_orig.setdefault(e["_orig_key"], []).append(e)
            _restored = [e for e in body_blocks if not e.get("_frag")]
            for _grp in _by_orig.values():
                _grp.sort(key=lambda e: e["x0"])
                _restored.append(
                    {
                        "text": _grp[0]["_orig_text"],
                        "cls": _grp[0]["cls"],
                        "x0": min(e["x0"] for e in _grp),
                        "y0": min(e["y0"] for e in _grp),
                        "x1": max(e["x1"] for e in _grp),
                        "y1": max(e["y1"] for e in _grp),
                        "cx": (min(e["x0"] for e in _grp) + max(e["x1"] for e in _grp)) / 2,
                        "w": max(e["x1"] for e in _grp) - min(e["x0"] for e in _grp),
                    }
                )
            body_blocks = _restored
        # iter-39 (audit-2, 2026-08-18): on COLUMN-SHARP pages the correct
        # reading order is COLUMN-MAJOR (entire left column, then entire
        # right) — band-based row-major interleaving read right-column
        # band-3 text before left-column band-6 text (0621a009 pg31:
        # Incognito@right-y72 before Gruet@left-y139; iter39 pin red).
        try:
            from scientific_parser.routing.layout_gate import column_sharp

            _sharp = column_sharp([(e["x0"], e["y0"], e["x1"], e["y1"]) for e in body_blocks])
        except Exception:
            _sharp = False
        if _sharp:
            _flat = lambda e: _dehyphenating_flatten(e["text"]).strip()
            full = sorted(
                (e for e in body_blocks if e["w"] > page_width * 0.6),
                key=lambda e: (e["y0"], e["x0"]),
            )
            left = sorted(
                (e for e in body_blocks if e["cx"] < mid_x and e["w"] <= page_width * 0.6),
                key=lambda e: (e["y0"], e["x0"]),
            )
            right = sorted(
                (e for e in body_blocks if e["cx"] >= mid_x and e["w"] <= page_width * 0.6),
                key=lambda e: (e["y0"], e["x0"]),
            )
            body_text = "\n".join(_flat(e) for e in (full + left + right) if _flat(e))
            body_text = re.sub(r" {2,}", " ", body_text)
            return _append_figure_region(body_text, figure_blocks, pdf_path, page_num)
        # Band-based reading order (25pt Y-bands). Within each band, split
        # into full-width / left-col / right-col, and JOIN consecutive blocks
        # in the same column whose X-gap < 20pt (handles per-word-block PDFs).
        bands = {}
        for b in body_blocks:
            band_key = round(b["y0"] / 25)
            bands.setdefault(band_key, []).append(b)
        out_lines = []
        for band_key in sorted(bands.keys()):
            bb = sorted(bands[band_key], key=lambda x: (x["x0"], x["y0"]))
            full_width = [b for b in bb if b["w"] > page_width * 0.6]
            left_col = [b for b in bb if b["w"] <= page_width * 0.6 and b["cx"] < mid_x]
            right_col = [b for b in bb if b["w"] <= page_width * 0.6 and b["cx"] >= mid_x]
            for col_slice, guard_mid in ((full_width, False), (left_col, True), (right_col, True)):
                if not col_slice:
                    continue
                # JOIN consecutive blocks with X-gap < 20pt into one line.
                # Blocks with larger gap start a new line. In the column
                # slices, a join may never CROSS the midline: a left-column
                # block ending before mid_x and a right-column block
                # starting after it belong to different columns even when
                # the gutter is < 20pt (09f90a8f: gutter 8-15pt joined
                # 'quest'+'ems successful' into one corrupted line).
                current_line_parts = []
                last_x1 = None
                for b in col_slice:
                    crosses_mid = guard_mid and last_x1 is not None and last_x1 < mid_x <= b["x0"]
                    if last_x1 is not None and (b["x0"] - last_x1 >= 20 or crosses_mid):
                        # X-gap large (or column gutter): flush line, start new
                        if current_line_parts:
                            out_lines.append(_join_line_parts(current_line_parts))
                        current_line_parts = []
                    current_line_parts.append(_dehyphenating_flatten(b["text"]).strip())
                    last_x1 = b["x1"]
                if current_line_parts:
                    out_lines.append(_join_line_parts(current_line_parts))
        body_text = "\n".join(out_lines)
        # Collapse runs of 2+ spaces introduced by joining individually-spaced
        # word-blocks (sakai1968-style positioned-glyph PDFs emit each word
        # with its own trailing space).
        body_text = re.sub(r" {2,}", " ", body_text)
        body_text = _append_figure_region(body_text, figure_blocks, pdf_path, page_num)
        return body_text
    except Exception as e:
        log.warning("column_aware_body_only failed p%s: %s", page_num, e)
        return None


def _append_figure_region(body_text, figure_blocks, pdf_path, page_num):
    """Append separated figure-region blocks (verbatim, nothing lost).

    Shared by both reading-order paths (column-major + band-based), so
    the gated TABLE_FROM_FIGURE probe lives in exactly one place.
    """
    if not figure_blocks:
        return body_text
    figure_text = "\n".join(
        b["text"] for b in sorted(figure_blocks, key=lambda x: (x["y0"], x["x0"]))
    )
    body_text += (
        f"\n\n<!-- FIGURE_REGION: {len(figure_blocks)} block(s) "
        f"(axis labels, glyphs, legend markers — preserved verbatim, "
        f"separated from body for readability) -->\n" + figure_text
    )
    # Image-table extraction (Option C, 2026-07-25): some figures contain
    # tables rendered as raster images. GATED behind --figure-table-ocr
    # (adds 5-15s per figure region; born-digital PDFs carry tables in
    # the text layer).
    if _FIGURE_TABLE_OCR:
        try:
            min_x0 = min(b["x0"] for b in figure_blocks)
            min_y0 = min(b["y0"] for b in figure_blocks)
            max_x1 = max(b["x1"] for b in figure_blocks)
            max_y1 = max(b["y1"] for b in figure_blocks)
            fig_width = max_x1 - min_x0
            fig_height = max_y1 - min_y0
            # Only probe if figure is "table-like": wide AND tall enough
            if fig_width > 200 and fig_height > 100:
                fig_bbox = (min_x0, min_y0, max_x1, max_y1)
                fig_tables = detect_tables_in_figure(pdf_path, page_num, fig_bbox)
                if fig_tables:
                    body_text += (
                        f"\n\n<!-- TABLE_FROM_FIGURE: extracted via table-aware "
                        f"OCR on figure region -->\n"
                    )
                    for t in fig_tables:
                        body_text += t["markdown"] + "\n"
        except Exception as e:
            log.warning("detect_tables_in_figure failed p%s: %s", page_num, e)
    return body_text


def column_aware_reading_order(pdf_path, page_num):
    """Reconstruct reading order via recursive XY-cut partition.

    Layout-independent: discovers structure from whitespace gaps.
    Handles arbitrary multi-column layouts, sidebars, boxed text.

    Returns reordered text, or None to keep pdftotext order.
    """
    _load_consts()
    _load_consts()
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return None
    try:
        text_blocks = []
        for b in pd["dict"]["blocks"]:
            if b.get("type", 0) != 0:
                continue
            bbox = b["bbox"]
            block_text = ""
            for line in b.get("lines", []):
                line_text = "".join(s["text"] for s in line.get("spans", []))
                if line_text.strip():
                    block_text += line_text + "\n"
            block_text = block_text.strip()
            if not block_text or len(block_text) < 3:
                continue
            text_blocks.append(
                {
                    "text": block_text,
                    "x0": bbox[0],
                    "y0": bbox[1],
                    "x1": bbox[2],
                    "y1": bbox[3],
                    "cx": (bbox[0] + bbox[2]) / 2,
                }
            )

        if len(text_blocks) < 4:
            return None

        # det63: XY-Cut++ full port (arXiv 2504.10258, ODL parity) as the
        # primary partitioner; incumbent 115-line _xy_cut_partition stays
        # as the loud fail-open fallback (R17 guard: any port failure
        # keeps the det62-proven ordering, never drops the page).
        ordered = None
        try:
            from scientific_parser.perception.layout.xy_cut_pp import (
                sort_reading_order,
            )

            for _b in text_blocks:
                _b["bbox"] = [_b["x0"], _b["y0"], _b["x1"], _b["y1"]]
            ordered = sort_reading_order(text_blocks)
        except Exception as e:
            log.warning("xy_cut_pp partition failed, falling back: %s", e)
            ordered = None
        if ordered is None:
            ordered = _xy_cut_partition(text_blocks)
        return "\n\n".join(b["text"] for b in ordered)
    except Exception as e:
        log.warning("reading order failed: %s", e)
        return None


# =============================================================================
# P0-2: Table reconstruction via PyMuPDF coordinates
# =============================================================================
# P0-2: Table reconstruction via PyMuPDF coordinates
# =============================================================================


__all__ = [
    "_xy_cut_partition",
    "classify_text_block",
    "column_aware_body_only",
    "column_aware_reading_order",
    "pdfplumber_column_crop",
]


def pdfplumber_column_crop(pdf_path, page_num):
    """Fallback 2-column reading order: crop left/right halves + concatenate.

    Fires when ``column_aware_body_only`` / ``column_aware_reading_order``
    return None (XY-Cut found no clean gutter on dense/complex layouts like
    reference lists). Crops the page at the horizontal midpoint via pdfplumber
    + extracts each half separately → correct column-by-column order.

    2-column detection is word-based (substantial text on both sides of the
    midpoint, few words crossing it) — more robust than block heuristics that
    mis-classify dense pages as single-column.

    Returns left+right text, or None if the page isn't 2-column / fails.
    Never applied to single-column pages (would split full-width lines).
    """
    try:
        import pdfplumber

        with pdfplumber.open(pdf_path) as d:
            pg = d.pages[page_num - 1]
            mid = pg.width / 2
            words = pg.extract_words(x_tolerance=2)
            if not words:
                return None
            # 2-column signal: substantial text purely on each side, few words
            # spanning the midpoint (full-width headers/titles are the crossers).
            left = sum(1 for w in words if w["x1"] < mid)
            right = sum(1 for w in words if w["x0"] > mid)
            crossing = sum(1 for w in words if w["x0"] < mid - 10 and w["x1"] > mid + 10)
            if left < 15 or right < 15 or crossing > len(words) * 0.10:
                return None  # single-column or full-width dominant — crop would split lines
            left_text = pg.crop((0, 0, mid, pg.height)).extract_text() or ""
            right_text = pg.crop((mid, 0, pg.width, pg.height)).extract_text() or ""
        text = (left_text + "\n\n" + right_text).strip()
        return text or None
    except Exception as e:
        log.warning("pdfplumber column crop failed: %s", e)
        return None
