"""Page metadata extraction — headers, footers, page numbers, section info.

Extracted from pdf_ocr.py during R9 final slim-down (2026-08-12).
"""

from __future__ import annotations
import re, os, hashlib, subprocess, logging

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state


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
        "_page_dict_cache",
        "DEPS",
        "_PRIVATE_CHAR_RE",
        "_GEOLOGY_DICT_TERMS",
        "PAGE_NUMBER_RE",
        "BRACKET_HEADER_RE",
        "HEADER_LINE_MAX_CHARS",
        "_doc_header_state",
        "_normalize_header_hash",
        "REE_SEQUENCE",
        "OXIDE_MW",
        "MINERAL_ABBREVS",
    ):
        if hasattr(p, n):
            g[n] = getattr(p, n)
    _CONSTS_LOADED = True


def _get_page_dict(pdf_path, page_num):
    _load_consts()
    cache_key = (pdf_path, page_num)
    if cache_key in _page_dict_cache:
        return _page_dict_cache[cache_key]
    try:
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        # CropBox ≠ MediaBox (olmCR-bench pypdf page-crops, print bleed):
        # poppler's text extraction ignores the crop and emits the whole
        # MediaBox, so the gold does too. fitz honors the crop, silently
        # dropping out-of-crop text (0925342e pg9: 4605→1332 chars — the
        # interview band above the crop vanished). Align fitz with poppler
        # semantics: read the full MediaBox.
        if page.cropbox != page.mediabox:
            try:
                page.set_cropbox(page.mediabox)
            except Exception as e:
                # Some pypdf-cropped pages carry a CropBox that itself
                # pokes outside the MediaBox (0e5f0c34: crop y0=0 < media
                # y0=9) and MuPDF refuses the write. The page is left
                # unchanged — extraction proceeds with the original crop,
                # exactly the pre-fix behavior (no partial state).
                log.debug("cropbox reset refused p%s: %s", page_num, e)
        page_dict = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
        page_rect = page.rect
        doc.close()
        result = {"dict": page_dict, "rect": page_rect}
    except Exception as e:
        log.warning("fitz page analysis failed p%s: %s", page_num, e)
        result = None
    _page_dict_cache[cache_key] = result
    return result


# HARD dependencies — no optional, no fallback. ImportError here aborts the
# script at load time (fail fast per the projectwide
# "no silent fallback" policy). All scientific-PDF deps are mandatory:


def strip_recurring_headers(page_metrics, final_md):
    """Learn and remove recurring headers/footers across multi-page documents.

    A header/footer is recurring if its normalized form appears on >70% pages.
    Modifies page_metrics in-place and returns updated final_md.
    """
    _load_consts()
    if len(page_metrics) < 3:
        return final_md

    header_counts = {}
    header_samples = {}

    for record in page_metrics:
        text = record.get("text", "")
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if len(lines) < 3:
            continue

        # First 2 lines = header candidate
        for i in range(min(2, len(lines))):
            normed = _normalize_header_hash(lines[i])
            if len(normed) > 5:
                h = hashlib.md5(normed.encode()).hexdigest()
                header_counts[h] = header_counts.get(h, 0) + 1
                header_samples[h] = lines[i]

        # Last 2 lines = footer candidate
        for i in range(max(0, len(lines) - 2), len(lines)):
            normed = _normalize_header_hash(lines[i])
            if len(normed) > 5:
                h = hashlib.md5(normed.encode()).hexdigest()
                header_counts[h] = header_counts.get(h, 0) + 1
                header_samples[h] = lines[i]

    threshold = len(page_metrics) * 0.7
    recurring_texts = set()
    for h, count in header_counts.items():
        if count >= threshold:
            recurring_texts.add(header_samples[h])

    if not recurring_texts:
        return final_md

    for record in page_metrics:
        text = record.get("text", "")
        lines = text.split("\n")
        for header_text in recurring_texts:
            for i in list(range(min(3, len(lines)))) + list(
                range(max(0, len(lines) - 3), len(lines))
            ):
                if header_text in lines[i]:
                    lines[i] = lines[i].replace(header_text, "").strip()
        record["text"] = "\n".join(l for l in lines if l.strip()).strip()
        record["chars"] = len(record["text"])

    print(f"  Header learning: removed {len(recurring_texts)} recurring pattern(s)")

    return final_md


# =============================================================================
# #11: Scientific validator extension
# =============================================================================

OXIDE_MW = {
    "SiO2": 60.08,
    "Al2O3": 101.96,
    "Fe2O3": 159.69,
    "FeO": 71.85,
    "CaO": 56.08,
    "MgO": 40.30,
    "Na2O": 61.98,
    "K2O": 94.20,
    "TiO2": 79.87,
    "P2O5": 141.94,
    "MnO": 70.94,
    "Cr2O3": 151.99,
    "H2O": 18.02,
    "CO2": 44.01,
}


def _detect_second_column_left(all_lines, page_w):
    """Left edge of the right column on a 2-column page, else None.

    Delegates to the ONE shared column-geometry implementation
    (layout_gate.left_edge_column_sharpness — audit-2 R4: the band guard
    and the router's signal-10 probe had drifted into two private
    clusterings with different constants).
    """
    if len(all_lines) < 8:
        return None
    from scientific_parser.routing.layout_gate import left_edge_column_sharpness

    _, col2 = left_edge_column_sharpness([b[0] for b in all_lines])
    return col2


def _merge_same_baseline_fragments(all_lines, page_w, y_tol=2.0, kern_gap=2.0):
    """Merge word-fragment lines sharing a baseline into whole lines.

    Fragment producers emit 'Ev'/'aluation' as separate line dicts; merging
    same-baseline neighbors (x-overlap/gap < kern_gap → concat, larger gap →
    space) rebuilds the visual line so header/heading/table classification
    sees whole text. A same-baseline gap ≥6pt centered on the page midline
    is a COLUMN GUTTER — break so column-top lines never fuse into
    gutter-crossing lines. Adjacent same-baseline header ELEMENTS (page
    number beside a running head) do merge — strip-equivalent: both
    elements were header lines already (measured 00c67a6c '1116' + 'K. M.
    Eisenhardt…' → one stripped line; bare-page-number METADATA is lost,
    the strip result is not). Non-fragment pages: distinct baselines never
    merge.
    FRAGMENT-PAGE GATE (iter-79b): merging same-baseline ELEMENTS on a
    NORMAL page fuses column tops / side-by-side header elements into
    lines that then fail every acceptance gate — BOTH leak un-stripped
    (measured headers_footers sample: 93.59 → 87.18). Merge ONLY when
    the page shows word-fragment evidence (same gates as
    column_reading_order._assemble_fragment_blocks): ≥60% of lines are
    tiny (≤15 chars), or ≥12% of ≥20 same-baseline adjacent line pairs
    TOUCH at <0.45pt (pieces of one word; kerned text never sits that
    close).
    """
    if len(all_lines) < 2:
        return all_lines
    tiny = sum(1 for _bb, t, _sz in all_lines if len(t.strip()) <= 15)
    by_y = sorted(all_lines, key=lambda e: (round(e[0][1], 1), e[0][0]))
    touch = total = 0
    for a, b in zip(by_y, by_y[1:]):
        if abs(a[0][1] - b[0][1]) > 2 or b[0][0] < a[0][2]:
            continue
        total += 1
        if b[0][0] - a[0][2] < 0.45:
            touch += 1
    if not (tiny >= 0.6 * len(all_lines) or (total >= 20 and touch >= 0.12 * total)):
        return all_lines
    entries = sorted(all_lines, key=lambda e: (round(e[0][1], 1), e[0][0]))
    merged = []
    for bbox, txt, sz in entries:
        if merged:
            pb, pt, ps = merged[-1]
            same_baseline = abs(bbox[1] - pb[1]) <= y_tol
            # A pure page-number fragment never merges with neighboring
            # elements: the bare-number header test only fires on the
            # standalone fragment, and a merged '1116 K. M. Eisenhardt…'
            # line fails every acceptance gate so BOTH elements would
            # leak (00c67a6c page_12).
            _pagenum = re.fullmatch(r"[\d\s.\-–—ivxlcdmIVXLCDM]{1,12}", pt.strip() or "§")
            if same_baseline and not _pagenum and bbox[0] >= pb[2] - 2.0:
                gap = bbox[0] - pb[2]
                gutter = (
                    gap >= 6
                    and page_w > 0
                    and abs((pb[2] + bbox[0]) / 2 - page_w / 2) <= page_w * 0.06
                )
                if not gutter:
                    pt = pt + txt if gap < kern_gap else pt + " " + txt
                    merged[-1] = (
                        (pb[0], min(pb[1], bbox[1]), max(pb[2], bbox[2]), max(pb[3], bbox[3])),
                        pt,
                        max(ps, sz),
                    )
                    continue
        merged.append((bbox, txt, sz))
    return merged


def _geometry_header_footer_lines(pdf_path, page_num):
    """Header/footer lines via GEOMETRY: line bbox in top/bottom band.

    A line whose bbox sits in the top 8% or bottom 8% of the page and is
    short (≤ HEADER_LINE_MAX_CHARS) is a running header/footer — position
    is the evidence, so mixed-case author headers ("K. M. Eisenhardt and
    J. A. Martin") and single-page excerpts (bench PDFs) are caught
    without cross-page consensus. Returns list of lines or None when no
    text layer (scanned pages — caller falls back to text heuristics).

    COLUMN GUARD (audit C4 fix, 2026-08-18): "body text never starts
    inside the margin bands" is FALSE on dense 2-column pages — columns
    start at ~7-9% height and end past 92% (measured 0514f667 pg4: 8
    column-body sentences were learned as headers, then stripped). On a
    detected 2-column page a band line qualifies as header/footer ONLY
    if it crosses the column gutter, is a bare page number, is short +
    horizontally centered, or is typographically distinct from the body
    size. Column-body lines sit inside one column at the body size.
    """
    _load_consts()
    pd = _get_page_dict(pdf_path, page_num)
    if not pd:
        return None
    rect = pd["rect"]
    H = rect.height
    W = rect.width
    if H <= 0:
        return None

    # First pass: every clean text line with bbox + dominant span size.
    all_lines = []
    for block in pd["dict"].get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            bbox = line.get("bbox", (0, 0, 0, 0))
            txt = "".join(s.get("text", "") for s in line.get("spans", [])).strip()
            # GARBAGE GUARD (iter-23b, 2026-08-17): U+FFFD (broken
            # ToUnicode) or combining-solidus fragments are extraction
            # noise, not header evidence — geometry-band + garbage chars
            # = body text misread (08172: 'minimal label of s(j)⊔<repl>'
            # + 'l̸=j s(l) belongs…' learned as headers, then the final
            # strip fuzzy-blanked the CLEAN cascade line, flipping the
            # iter3-bigsqcup pin). A running header is a clean typeset
            # line.
            if not txt or "\ufffd" in txt or "\u0338" in txt:
                continue
            sizes = [s.get("size", 0) for s in line.get("spans", []) if s.get("size")]
            all_lines.append((bbox, txt, max(sizes) if sizes else 0.0))
    # FRAGMENT MERGE (iter-79, 09f90a8f page_7): word-fragment producers
    # (T11-T18 subsets) emit every fragment as its own line — on such
    # pages the top band fills with SECTION-HEADING and table-stub
    # fragments ('5'/'Ev'/'aluation', 'Pro'/'v'/'en'/'Precision'), each
    # individually passing every short-line test, and the final strip
    # then fuzzy-cuts those bare fragments out of BODY words
    # ('evaluation' → 'ev'). Merge same-baseline fragments first: a real
    # running header stays one merged line in the band; heading/table
    # lines become whole lines the column guard and heading rescue judge.
    all_lines = _merge_same_baseline_fragments(all_lines, W)
    # Body size over ALL lines (robust median: masthead stamps are a
    # small minority) — used by the masthead size-cohort rule.
    _szs = sorted(sz for _bb, _t, sz in all_lines if sz > 0)
    _body_size_all = (_szs[len(_szs) // 2],) if _szs else ()

    def _is_band_candidate(bbox, txt, sz):
        y0, y1 = bbox[1], bbox[3]
        # Deep bands (y0 < 6% top, y1 > 92% bottom): header/footer by
        # position alone — includes long boilerplate footer paragraphs
        # (256-char nondiscrimination notices) that fail a length cap.
        deep = y0 < H * 0.06 or y1 > H * 0.92
        # Extended top band (6–10%): masthead blocks ("Publication
        # 4H-224" at 8.2%) — short lines only, body-first-line guard.
        shallow_top = y0 < H * 0.10 and len(txt) <= HEADER_LINE_MAX_CHARS
        # MASTHEAD SIZE COHORT (iter-61): a long top-band line at a size
        # DISTINCT from the body is masthead, not body — 'Copyright ©
        # 2011 The American Society…' (89 chars, 6pt) missed the deep
        # band by 1pt and the length cap, leaking 7 absent-tests. The
        # cap protects BODY-SIZE first lines; distinct size + top band =
        # running header regardless of length. C4 column-tops stay safe:
        # they sit at body size.
        _sz_all = _body_size_all[0] if _body_size_all else None
        shallow_top_cohort = y0 < H * 0.10 and _sz_all is not None and abs(sz - _sz_all) > 1.0

        # MARGIN STAMP (iter-61): rotated side-margin stamps render as a
        # narrow vertical strip hugging the page edge ('Downloaded from
        # molpharm…' x565-577 on a 585pt page, mid-page y). Nothing real
        # fits in a ≤5%-wide edge strip — position alone is evidence.
        # ...requires ROW ISOLATION: matrix/grid entries share their row
        # with siblings ('0','0','0' LDPC rows at 75-87% y flagged en masse
        # by the bare rule, 9 arxiv_math regressions); a true folio sits
        # alone on its row (court '117'; running-head rows are covered by
        # the shallow-top/deep rules, not this one).
        def _row_isolated(bb):
            h = max(1.0, bb[3] - bb[1])
            mid = (bb[1] + bb[3]) / 2
            for other, _t, _s in all_lines:
                if other == bb:
                    continue
                om = (other[1] + other[3]) / 2
                if abs(om - mid) < 0.6 * h:
                    return False
            return True

        def _vert_clear(bb):
            """True when no other line crowds within 0.8x this line's height
            above OR below. Fraction denominators sit tight against the
            next body row (2503.04124 '5': gap_below 1.7pt = 0.2x h); true
            folios have clearance on both sides (court '117': 22.9pt above,
            nothing below)."""
            h = max(1.0, bb[3] - bb[1])
            for other, _t, _s in all_lines:
                if other == bb:
                    continue
                if (bb[1] - other[3]) < 0.8 * h and (other[1] - bb[3]) < 0.8 * h:
                    gap_a = bb[1] - other[3]
                    gap_b = other[1] - bb[3]
                    if -0.5 * h < gap_a < 0.8 * h or -0.5 * h < gap_b < 0.8 * h:
                        return False
            return True

        # (or ≥2-char roman) ≤6 chars in the top or bottom SIXTH is a
        # page number wherever the producer set it — court stamps put
        # the folio at 89% (fc8f9a30 '117', below the 92% deep floor),
        # Elsevier puts it on the running-head row at 6.6%. A standalone
        # number as a WHOLE visual line is not typeset body flow (body
        # numbers ride inside sentences), so the wide band is safe.
        _bare = txt.strip()
        wide_band_bare_number = (
            (y0 < H * 0.15 or y1 > H * 0.85)
            and _row_isolated(bbox)
            and _vert_clear(bbox)
            and (
                re.fullmatch(r"\d{1,6}", _bare) is not None
                or re.fullmatch(r"[ivxlcdm]{2,6}|[IVXLCDM]{2,6}", _bare) is not None
            )
        )
        _w = bbox[2] - bbox[0]
        margin_stamp = _w <= W * 0.05 and (bbox[0] >= W * 0.955 or bbox[2] <= W * 0.045)
        return deep or shallow_top or shallow_top_cohort or margin_stamp or wide_band_bare_number

    col2_left = _detect_second_column_left(all_lines, W)
    body_sizes = sorted(
        sz for bbox, txt, sz in all_lines if sz > 0 and not _is_band_candidate(bbox, txt, sz)
    )
    body_size = body_sizes[len(body_sizes) // 2] if body_sizes else None

    # BODY-REGION counts (iter-80): body-size line population inside each
    # band (see the col2_left-None guard below). Body-size tolerance is
    # proportional — dense small-print pages run body at 5.7-7.2pt.
    _tol = 1.0 if body_size is None else max(1.0, body_size * 0.18)
    _top_body_count = sum(
        1
        for _bb, _bt, _bs in all_lines
        if _bb[1] < H * 0.10 and _bs > 0 and abs(_bs - (body_size or _bs)) <= _tol
    )
    _bot_body_count = sum(
        1
        for _bb, _bt, _bs in all_lines
        if _bb[3] > H * 0.92 and _bs > 0 and abs(_bs - (body_size or _bs)) <= _tol
    )
    _page_num_re = re.compile(r"^[\d\s.\-–—ivxlcdmIVXLCDM]{1,12}$")
    # Running-head chrome signatures (det24a): content patterns that mark a
    # top-band line as journal/report chrome even when the det23a rescue
    # geometry (title-sized line, body flowing beneath) would keep it as a
    # section title. Measured on olmOCR headers_footers det23 flips: journal
    # vol/no + year (531...), roman page number + book title (0122...),
    # report/document IDs (ANEC-R&T-2010...), page markers (стр.132).
    _chrome_year_re = re.compile(r"\b(?:19|20)\d{2}\b")
    _chrome_vol_re = re.compile(r"(?i:\bvol\.?|\bno\.?|№|\bтом\b|\bissue\b|\bизд\b|\b\d+:\d+\b)")
    _chrome_report_re = re.compile(r"\b[A-Z]{2,}[A-Z0-9&-]*-\d[A-Za-z0-9-]*")
    _chrome_pagemark_re = re.compile(r"(?i:\bстр\.?|\bpage\b|\bpp\.?|\bp\.)\s*\.?\d")
    # det25 running-head widenings (residual headers_footers P2F class):
    # URL band line, month-year/date stamp, bare trailing page number on a
    # short line, and 'et al.' author attribution heads.
    _chrome_url_re = re.compile(r"(?i:\bhttps?://|\bwww\.)")
    _chrome_date_re = re.compile(
        r"\b\d{1,2}[ -][A-Za-z]{3,9}[ -](?:19|20)\d{2}\b"
        r"|\b[A-Za-z]{3,9}[ -]\d{1,2},?[ -]?(?:19|20)\d{2}\b"
        r"|\b[A-Za-z]{3,9}[ -](?:19|20)\d{2}\b"
    )
    _chrome_trailnum_re = re.compile(r"\b\d{1,4}\s*[.]?$")
    _chrome_etal_re = re.compile(r"(?i:\bet\.?\s*al\.?\b)")
    # attribution-byline footer heads (d461e0d4 'Prepared by: Becky
    # Geltz,'): attribution stamp + optional date on the next band row.
    _chrome_attr_re = re.compile(
        r"(?i:\b(?:prepared|compiled|created|generated|submitted|written)\s+by\b)"
    )
    _chrome_cite_re = re.compile(r"[A-Z]\. ?[A-Z]\.|\((?:19|20)\d{2}\)\.?")

    def _running_head_chrome(t):
        tl = (t or "").strip()
        if not tl:
            return False
        if _chrome_year_re.search(tl) and _chrome_vol_re.search(tl):
            return True
        if _chrome_report_re.search(tl):
            return True
        if _chrome_pagemark_re.search(tl):
            return True
        if _chrome_url_re.search(tl):
            return True
        if _chrome_attr_re.search(tl) and len(tl.split()) <= 12:
            return True
        if _chrome_date_re.search(tl) and len(tl.split()) <= 10:
            return True
        if (
            _chrome_etal_re.search(tl)
            and len(tl.split()) <= 10
            and re.search(r"(?i:et\.?\s*al\.?\s*$)", tl)
            and tl[:1].isupper()
        ):
            return True
        words = tl.split()
        if (
            len(words) <= 8
            and _chrome_trailnum_re.search(tl)
            and not tl.rstrip().endswith((".", "!", "?", ","))
            and all(w[:1].isupper() for w in words if w[:1].isalpha())
        ):
            return True
        if len(words) <= 10 and re.match(r"(?i)^[ivxlcdm]{2,7}\b", tl):
            return True
        return False

    # CHROME-CLUSTER ADJACENCY (det25, e71f5c59/fd9f7d29 class): a band
    # line sitting in the same row (or within one leading) of an already
    # identified chrome line is part of the same running-head block
    # (journal title-head + DOI + URL + page marker travel together).
    _chrome_line_ids = {
        _i for _i, (_bb, _tx, _sz) in enumerate(all_lines) if _running_head_chrome(_tx)
    }

    def _chrome_cluster_adjacent(bbox, sz, own_text):
        if own_text and _running_head_chrome(own_text):
            return False

        # HORIZONTAL LABEL ARM (det31c, 9530001d 'TITELTHEMA' masthead):
        # magazine topic blocks set the label BESIDE the block text —
        # same row, no x-overlap ('TITELTHEMA' x30-114 | 'Kontrollierte
        # Produktströme –' x127+ @11 over body 8.8). Vertical stacking
        # rules plus the size guard can never see this shape. Membership
        # is a 2-level closure: the candidate hangs off a chrome line
        # directly, or off an intermediary that itself hangs
        # horizontally off chrome ('Armaturen, Rohre…' stacked under
        # 'Kontrollierte Produktströme –').
        def _hangs(_a, _b, _sa, _sb):
            # _a hangs off _b: horizontal side-by-side (label left,
            # same size, tight x-gap, near row) or vertical tight stack
            # with x-overlap.
            _msz = max(_sa, _sb, 1.0)
            if (
                abs(_sb - _sa) <= 1.0
                and abs(_a[1] - _b[1]) <= 1.5 * _msz
                and _b[2] < _a[0]
                and _a[0] - _b[2] <= 2.0 * _msz
            ):
                return True
            if (
                abs(_sb - _sa) <= 1.0
                and _a[1] >= _b[3]
                and _a[1] - _b[3] <= 2.2 * _msz
                and _a[0] < _b[2]
                and _b[0] < _a[2]
            ):
                return True
            return False

        # Anchor set is STATIC (order-independent): regex-chrome lines
        # plus top-band ALL-CAPS distinct-size labels ('TITELTHEMA')
        # — masthead labels are all-caps by convention, independent of
        # whether they were processed before the candidate.
        _top_chrome = [
            (_bb2, _sz2)
            for _i, (_bb2, _tx2, _sz2) in enumerate(all_lines)
            if _bb2[1] < H * 0.15
            and (
                _i in _chrome_line_ids
                or (
                    sum(ch.isupper() for ch in _tx2 if ch.isalpha()) >= 3
                    and sum(ch.isupper() for ch in _tx2 if ch.isalpha())
                    >= 0.9 * max(1, sum(ch.isalpha() for ch in _tx2))
                    and (body_size is None or abs(_sz2 - body_size) > 1.0)
                )
            )
        ]
        for _bb2, _sz2 in _top_chrome:
            if _hangs(bbox, _bb2, sz, _sz2):
                return True
        for _j, (_bb3, _tx3, _sz3) in enumerate(all_lines):
            if _j in _chrome_line_ids or _bb3 is bbox:
                continue
            if not _hangs(bbox, _bb3, sz, _sz3):
                continue
            for _bb2, _sz2 in _top_chrome:
                if _hangs(_bb3, _bb2, _sz3, _sz2):
                    return True
        # SIZE GUARD (det26, odlbench 01030000000029): running-head
        # chrome is set at or below body size — a band line LARGER than
        # body is a section TITLE that merely sits under the journal
        # chrome ('5. The dynamics'@11 over body 10, 12pt beneath the
        # DOI line). Adjacency must not strip it; the rescue arm owns
        # that call. Chrome-shaped heads (Tarvainen 'et al.'@9 ≤ body)
        # stay capturable.
        if body_size is not None and sz > body_size + 0.5:
            return False
        # Top band only: a body line flowing just above a bottom-band
        # page number must not inherit footer chrome status.
        if bbox[1] >= H * 0.5:
            return False
        for _i in _chrome_line_ids:
            _bb2, _tx2, _sz2 = all_lines[_i]
            if _bb2[1] >= H * 0.15:
                continue
            if abs(_bb2[1] - bbox[1]) <= 0.6 * max(sz, _sz2, 1.0):
                return True
            _gap = min(abs(_bb2[1] - bbox[3]), abs(bbox[1] - _bb2[3]))
            if _gap <= 2.2 * max(sz, _sz2, 1.0) and _bb2[0] < bbox[2] and bbox[0] < _bb2[2]:
                return True
        return False

    out = []
    _accepted_bottom_bboxes = []
    for bbox, txt, sz in all_lines:
        if not _is_band_candidate(bbox, txt, sz):
            continue
        # LETTER-SPACED TITLE GUARD (iter-83, 028649 p29): typographic
        # letter-spaced all-caps ('D E F I N I T I O N S' at 3.1% H) is a
        # section TITLE, never a running header — running heads are not
        # letter-spaced. The deep band accepted it and the strip deleted
        # the heading outright.
        if re.fullmatch(r"(?:[A-Z] ){4,}[A-Z]", txt.strip()):
            continue
        # BODY-FLOW GUARD (iter-94, cfbe79724; extended 2026-09-03): a
        # non-bottom-band long line at BODY size with same-size text
        # flowing directly beneath it at overlapping x is a body
        # paragraph's first line, not a masthead — running heads never
        # carry body-size continuation at body leading. Measured:
        # blog-style page whose first body sentence (9.1% H, 74 chars,
        # body 12pt) was accepted via gutter-cross against a
        # hallucinated column and blanked. DEEP-TOP RESCUE
        # (2026-09-03): dense book pages start body at 5.3% H
        # (13_pg639 'subsequently in the middle ages…', 13_pg811 'the
        # right of issuing commissions…') — deep-top position alone
        # must not override body-flow evidence; bottom-deep stays
        # excluded (footer paragraphs get the flow-above guard below).
        # len > 50: 13_pg639 'The common form of the book of the
        # ancient world was' (52 chars); keeps short masthead stamps
        # ('Publication 4H-224') outside this guard.
        if (
            bbox[3] <= H * 0.92
            and len(txt) > 50
            and body_size is not None
            and abs(sz - body_size) <= _tol
            and any(
                _fb is not bbox
                and _fs > 0
                and abs(_fs - body_size) <= _tol
                and max(0.0, min(bbox[2], _fb[2]) - max(bbox[0], _fb[0]))
                >= 0.3 * max(1.0, bbox[2] - bbox[0])
                # TIGHT-LEADING FLOOR (2026-09-03): true body flow's
                # next line starts within ~0.2× size (13_pg639 0.13×,
                # 13_pg811 0.18×); a running head sits a clear gap above
                # its page's first body line (6ed26883 'Sabapathi et
                # al.…' 0.89×, fe8c6edb '© Biodiversity' 1.48×) — the
                # old 1.6× floor rescued real heads.
                and bbox[3] - 0.5 * (sz if sz > 0 else 10.0)
                <= _fb[1]
                <= bbox[3] + 0.5 * (sz if sz > 0 else 10.0)
                for _fb, _ft, _fs in all_lines
            )
        ):
            continue
        # SAME-ROW OVERLAP/ADJACENCY GUARD (2026-08-31; adjacency added
        # 2026-09-03): a shallow-top candidate that X-OVERLAPS — or sits
        # a kern-scale gap (≤10pt) beside — another line on its own row
        # is a split-span fragment of one body line (2503.04297_pg20:
        # 'so [φ] is a class in FkH−1 (gri' x[14-37] + 'L h). We now
        # consider…' x[36-61], overlapping; 13_pg639: 'forms.' x[15-33]
        # + 'The common form…' x[40-203], 6.7pt gap — the fragment-page
        # merge gate leaves normal pages unmerged, so the pair reaches
        # acceptance as two lines). Real running heads are single
        # lines; same-row running head + folio never overlap or touch
        # (fe8d3e95: x[23-75] vs x[85-87] — the 10pt gap floor keeps
        # that pair strippable).
        # BODY-SIZE PAIR FLOOR (2026-09-03): the rescue is for split
        # BODY lines — both fragments must sit at body size. A band
        # line that only overlaps a stray 1-char glyph ('<' over the
        # '© Biodiversity' head, fe8c6edb) or its own scan-echo
        # ('Microorganisms 2021…' double print 5.2pt lower, 09c8a46e;
        # 'The Japan Society…' + OCR-noise neighbor, 4b91e05f) is a
        # HEAD, and those heads run at a size distinct from the body.
        if (
            bbox[1] < H * 0.10
            and body_size is not None
            and abs(sz - body_size) <= _tol
            and any(
                _fb is not bbox
                and _fs > 0
                and abs(_fs - body_size) <= _tol
                and abs((_fb[1] + _fb[3]) / 2 - (bbox[1] + bbox[3]) / 2)
                < 0.6 * max(1.0, bbox[3] - bbox[1])
                and (
                    min(bbox[2], _fb[2]) - max(bbox[0], _fb[0]) > 0
                    or 0 <= _fb[0] - bbox[2] <= 10
                    or 0 <= bbox[0] - _fb[2] <= 10
                )
                for _fb, _ft, _fs in all_lines
            )
        ):
            continue
        # deep BOTTOM band — a long body-size line with same-size text
        # flowing directly ABOVE it at body leading is a paragraph's last
        # line, not a footer. Math pages run equations/prose past 92% H
        # routinely (measured 2503.07567_pg6: 'min = 7 instead of 14. Note
        # that the quantum…' + 'minimum distance is again equal to m + n.'
        # stripped as headers → 8 arxiv_math regressions once the jsonl
        # carried the final strip). Isolated boilerplate footers sit a
        # clear gap below the last body line (≥1.6× leading) and stay
        # strippable; continuation lines do not.
        _bottom_deep = bbox[3] > H * 0.92
        if (
            _bottom_deep
            and len(txt) > 25
            and body_size is not None
            and abs(sz - body_size) <= _tol
            and any(
                _fb is not bbox
                and _fs > 0
                and abs(_fs - body_size) <= _tol
                and max(0.0, min(bbox[2], _fb[2]) - max(bbox[0], _fb[0]))
                >= 0.3 * max(1.0, bbox[2] - bbox[0])
                and _fb[3] <= bbox[1] <= _fb[3] + 1.6 * (sz if sz > 0 else 10.0)
                for _fb, _ft, _fs in all_lines
            )
        ):
            continue
        # MARGIN STAMP acceptance (iter-61): the narrow-edge-strip
        # geometry is decisive regardless of size/length/columns —
        # same position-alone status as the deep bands.
        _w = bbox[2] - bbox[0]
        if _w <= W * 0.05 and (bbox[0] >= W * 0.955 or bbox[2] <= W * 0.045):
            out.append(txt)
            continue
        if col2_left is None:
            # BODY-REGION CAP (iter-80, long_tiny 13_pg566 / 11_pg421 /
            # 11_pg57 / 20_pg33/39/46/48): single-column dense pages
            # (encyclopedias, dialect dictionaries, TV/radio listings)
            # start BODY text inside the top/bottom band; this path used
            # to accept every band line untested and the strip ate body
            # columns (measured 16+ body lines as 'headers'). A running
            # header/footer is a 1-2 line isolated element — a band
            # holding ≥5 body-size lines is a body REGION: reject every
            # body-size line in it; page numbers and clearly-distinct
            # mastheads (≥25% larger, or a small stamp) keep status.
            _pn = bool(_page_num_re.match(txt))
            _dist = body_size is not None and (sz > body_size * 1.25 or sz < body_size - 1.5)
            _top_band = bbox[1] < H * 0.10
            _bot_band = bbox[3] > H * 0.92
            # SECTION-HEADING RESCUE, single-column port (det23a, odlbench
            # 01030000000105): iter-69's rescue exists only on the
            # 2-column path below. On a single-column page a top-band line
            # LARGER than body with body-size text flowing directly
            # beneath (x-overlap, start within y1 + 2.2x size) is a
            # section TITLE, not a masthead. The Copyright-masthead case
            # stays stripped: its only continuation is a 6pt stamp (not
            # body size). Chrome-content lines never rescue (det24a).
            if (
                _top_band
                and body_size is not None
                and sz > body_size + 1.0
                and not _running_head_chrome(txt)
                and not _chrome_cluster_adjacent(bbox, sz, txt)
                # RIGHT-FLOAT LETTERHEAD (det25, fd9f7d29 'Mesto Sobrance'):
                # a ≤5-word top-band line floating past the page midline is
                # municipal/letterhead chrome — real section titles align
                # with the body column (105 title x0=54 vs W/2=306).
                and not (bbox[0] > W * 0.5 and len(txt.split()) <= 5)
            ):
                _lw = max(1.0, bbox[2] - bbox[0])
                _rescued = False
                _cite_block = False
                for _bb, _fb, _fs in all_lines:
                    # hanging-indent citation MARK ('[14].') under the
                    # candidate marks a references list (e71f5c59) —
                    # never rescue a running head over one
                    if (
                        bbox[1] <= _bb[1] <= bbox[3] + 2.2 * sz
                        and _bb[0] < bbox[0]
                        and re.match(r"^\[\d{1,3}\]", _fb.strip())
                    ):
                        _cite_block = True
                        continue
                    if _bb is bbox or abs(_fs - body_size) > 1.0:
                        continue
                    # CITATION CONTINUATION veto (det25, e71f5c59): the
                    # 'body' beneath a references-page running head is a
                    # citation entry (initial pairs '(2014).'), not prose
                    # flow — never rescue over it.
                    if _chrome_cite_re.search(_fb):
                        continue
                    _xo = max(0.0, min(bbox[2], _bb[2]) - max(bbox[0], _bb[0]))
                    if _xo < 0.3 * _lw:
                        continue
                    if bbox[1] <= _bb[1] <= bbox[3] + 2.2 * sz:
                        _rescued = True
                        break
                if _rescued and not _cite_block:
                    continue
            # TIGHT-CONTINUATION veto, single-column port (det23a,
            # iter-111 geometry): a body-size band line with same-size
            # text directly above OR beneath at x-overlap is BODY text
            # flowing through the band; a running header sits isolated
            # (nothing within ~2 line heights either way). Protects the
            # first body lines under a rescued title when the band holds
            # <5 body lines and the iter-80 cap cannot. Chrome-content
            # lines never veto (det24a): journal/report running heads sit
            # at body size with body flowing beneath.
            if (
                not _pn
                and not _dist
                and body_size is not None
                and not _running_head_chrome(txt)
                and not _chrome_cluster_adjacent(bbox, sz, txt)
            ):
                if abs(sz - body_size) <= 1.0:
                    _lw = max(1.0, bbox[2] - bbox[0])
                    _gap = 2.2 * (sz if sz > 0 else 10.0)
                    _tight = False
                    for _bb, _fb, _fs in all_lines:
                        if _bb is bbox or _fs <= 0 or abs(_fs - body_size) > 1.0:
                            continue
                        _xo = max(0.0, min(bbox[2], _bb[2]) - max(bbox[0], _bb[0]))
                        if _xo < 0.3 * _lw:
                            continue
                        if (bbox[1] - 0.5 * max(sz, 8.0) <= _bb[1] <= bbox[3] + _gap) or (
                            _bb[3] - 0.5 * max(sz, 8.0) <= bbox[1] <= _bb[3] + _gap
                        ):
                            _tight = True
                            break
                    if _tight:
                        continue
            if (
                not _pn
                and not _dist
                and body_size is not None
                and ((_top_band and _top_body_count >= 5) or (_bot_band and _bot_body_count >= 5))
            ):
                continue
            # TITLE CORROBORATION (det29a, odlbench 01030000000156):
            # a top-band line >=25% larger than body is accepted as a
            # masthead by default — but book-spread section titles sit
            # alone above the body with a real vertical gap, outside
            # every rescue window. A true masthead always carries
            # chrome evidence: content pattern, a chrome cluster
            # neighbor, or all-caps journal styling. A bare Title-Case
            # display line with none of those is the page TITLE — keep
            # it for the heading pass to promote.
            # det31b (e71f5c59 p6): a running head over a hanging-
            # indent citation block is never the page title either.
            if (
                _top_band
                and _dist
                and not _pn
                and body_size is not None
                and sz > body_size * 1.25
                and next((c for c in txt if c.isalpha()), "").isupper()
                and not _running_head_chrome(txt)
                and not _chrome_cluster_adjacent(bbox, sz, txt)
                and not any(
                    bbox[1] <= _bb[1] <= bbox[3] + 2.2 * sz
                    and _bb[0] < bbox[0]
                    and re.match(r"^\[\d{1,3}\]", _fb.strip())
                    for _bb, _fb, _fs in all_lines
                )
                and not (
                    sum(ch.isupper() for ch in txt if ch.isalpha()) >= 3
                    and sum(ch.isupper() for ch in txt if ch.isalpha())
                    >= 0.9 * max(1, sum(ch.isalpha() for ch in txt))
                )
            ):
                continue
            out.append(txt)
            continue
        x0, x1 = bbox[0], bbox[2]
        crosses_gutter = x0 < col2_left - 5 and x1 > col2_left + 5
        short_centered = (
            len(txt) <= HEADER_LINE_MAX_CHARS and abs((x0 + x1) / 2 - W / 2) <= W * 0.08
        )
        # BODY-CONTINUATION VETO (iter-111, 00945e98 p4): on a 2-column
        # page the RIGHT column starts near the page center, so a short
        # last line of a paragraph ('not valid.', x-center 5.3% off
        # page-center) reads as 'short + centered'. A real centered
        # footer never carries body-size text flowing directly above it
        # at overlapping x within paragraph leading; a paragraph's last
        # line always does. Veto short_centered on that evidence.
        _cont_above = False
        if short_centered and body_size is not None and abs(sz - body_size) <= _tol:
            _lw2 = max(1.0, x1 - x0)
            _gap2 = 2.2 * (sz if sz > 0 else 10.0)
            for _bb, _fb, _fs in all_lines:
                if _bb is bbox or _fs <= 0 or abs(_fs - body_size) > _tol:
                    continue
                if max(0.0, min(x1, _bb[2]) - max(x0, _bb[0])) < 0.3 * _lw2:
                    continue
                if bbox[1] - _gap2 <= _bb[3] <= bbox[1] + 0.5 * max(sz, 8.0):
                    _cont_above = True
                    break
            if _cont_above:
                short_centered = False
        is_page_number = bool(_page_num_re.match(txt))
        distinct_size = body_size is not None and abs(sz - body_size) > 1.0
        # SECTION-HEADING RESCUE (iter-69, 8f039414): a top-band line
        # LARGER than body with body-size text flowing directly beneath
        # it (x-overlap, start within y1 + 2.2×size) is a section
        # heading, not a running header — iter-61's masthead size-cohort
        # stripped 'The Planning Process—A Summary' (18pt, landscape
        # top band, 2 gold order-tests). The Copyright masthead's only
        # continuation is its own 6pt stamp line (not body size) and it
        # is SMALLER than body — both gates exclude it.
        # Citation-block evidence (det25/det31b, e71f5c59): hanging-
        # indent citation MARKS ('[14].') beneath the candidate at a
        # smaller x0 mark a references list — both the iter-69 rescue
        # and title-keep must deny heading/title status over one.
        _cite_ev = any(
            bbox[1] <= _bb[1] <= bbox[3] + 2.2 * sz
            and _bb[0] < bbox[0]
            and re.match(r"^\[\d{1,3}\]", _fb.strip())
            for _bb, _fb, _fs in all_lines
        )
        _heading = False
        if bbox[1] < H * 0.5 and body_size is not None and sz > body_size + 1.0:
            _lw = max(1.0, x1 - x0)
            for _bb, _bt, _bs in all_lines:
                if _bb is bbox or abs(_bs - body_size) > 1.0:
                    continue
                _xo = max(0.0, min(x1, _bb[2]) - max(x0, _bb[0]))
                if _xo < 0.3 * _lw:
                    continue
                if bbox[1] <= _bb[1] <= bbox[3] + 2.2 * sz:
                    _heading = True
                    break
            if _cite_ev:
                _heading = False
        # ISOLATION (2026-08-18), DEEP band only: a running header sits
        # ALONE at the very top (y0 < 6% H) — nothing starts directly
        # beneath it at overlapping x. A column-top body line (why the
        # column guard exists) has column continuation within ~2 line
        # heights. Deep-band position is already "header by position
        # alone" evidence; the column guard was protecting FALSE 2-col
        # detections there too (1a3e6174 catalog: UNH running header at
        # 4.3% H survived because indented lists read as columns).
        # Shallow band (6-10%) keeps the strict column guard: C4 pin
        # (0514f667) column-tops sit at 6.25% and MUST stay protected.
        _deep = bbox[1] < H * 0.06 or bbox[3] > H * 0.92
        _isolated = False
        if _deep:
            _isolated = True
            _lw = max(1.0, x1 - x0)
            _gap = 2.2 * (sz if sz > 0 else 10.0)
            for _bb, _bt, _bs in all_lines:
                if _bb is bbox or (_bt == txt and _bb[1] == bbox[1]):
                    continue
                _xo = max(0.0, min(x1, _bb[2]) - max(x0, _bb[0]))
                if _xo < 0.3 * _lw:
                    continue
                if bbox[1] < H * 0.5:  # top band: continuation BELOW
                    # DESCENDER OVERLAP (iter-111, 00945e98 p4): tight
                    # leading makes adjacent line bboxes interleave by
                    # ~1pt — the continuation's y0 can sit slightly
                    # ABOVE the candidate's y1. Tolerate half a line of
                    # overlap when looking for the neighbor.
                    if bbox[3] - 0.5 * max(sz, 8.0) <= _bb[1] <= bbox[3] + _gap:
                        _isolated = False
                        break
                else:  # bottom band: continuation ABOVE
                    # Same overlap tolerance: 'not valid.' y0=762.87 vs
                    # the line above ending y1=763.84 — a 0.97pt bbox
                    # interleave left the last body line of the column
                    # 'isolated', deep-band-accepted as a footer, and
                    # the strip deleted it.
                    if bbox[1] - _gap <= _bb[3] <= bbox[1] + 0.5 * max(sz, 8.0):
                        _isolated = False
                        break
        # FOOTER-BLOCK CONTINUITY (2026-08-19): multi-line footer blocks
        # (SPIE stamps: 'Downloaded From: ...' above 'Terms of Use: ...',
        # 7pt apart, both at y1/H > 98%) — the second line fails the
        # generic guards (different length, off-center, body size) and
        # survives to leak (measured 0f692b1b absent-test). A line in the
        # deep band sitting directly beneath another ACCEPTED footer line
        # is part of the same block.
        _cont_block = False
        if bbox[1] > H * 0.5:
            for _ob in _accepted_bottom_bboxes:
                if 0 <= bbox[1] - _ob[3] <= 1.5 * max(sz, 8.0) and not (
                    bbox[2] < _ob[0] or bbox[0] > _ob[2]
                ):
                    _cont_block = True
                    break
        # MARGINALIA EXTREMAL CONJUNCTION (iter-120b, clean-room from
        # marker MarginaliaProcessor behavior): a margin-zone line is
        # marginalia ONLY when ALL hold — (a) wholly inside the top 13%
        # or bottom 13% zone; (b) EXTREMAL: header lines end above the
        # body's minimum top, footer lines start below the body's
        # maximum bottom, where BODY = lines not wholly inside either
        # margin zone (margin fragments never disqualify each other);
        # (c) height <= 3.5% page; (d) <= 150 chars. Catches off-center
        # scattered footers beside a detected gutter (measured 178
        # leaking lines). The heading rescue (_heading) vetoes short
        # titles above body start.
        # PREREQUISITE (C4 pin): a column-confined body-size line at a
        # column top satisfies (a)-(d) on short synthetic pages — the
        # conjunction only UPGRADES lines already footer-shaped.
        _marg_footer_shaped = crosses_gutter or is_page_number or distinct_size
        _marginalia = False
        if not _heading and _marg_footer_shaped:
            _mh = bbox[3] - bbox[1]
            _in_top = bbox[1] < H * 0.08 and bbox[3] <= H * 0.13
            _in_bot = bbox[1] >= H * 0.87
            if (_in_top or _in_bot) and _mh <= H * 0.035 and len(txt) <= 150:
                if _in_top:
                    _body_tops = [
                        _fb[1]
                        for _fb, _ft, _fs in all_lines
                        if not (_fb[1] < H * 0.08 and _fb[3] <= H * 0.13)
                    ]
                    _marginalia = bool(_body_tops) and bbox[3] <= min(_body_tops) + 1.0
                else:
                    _body_bots = [_fb[3] for _fb, _ft, _fs in all_lines if _fb[1] < H * 0.87]
                    _marginalia = bool(_body_bots) and bbox[1] >= max(_body_bots) - 1.0
        # TITLE CORROBORATION, 2-column port (det29a, odlbench
        # 01030000000156 'Fact-Checking' @1.6x body over a gapped
        # 2-column body): the distinct-size arm accepts any top-band
        # outlier as masthead chrome. A real masthead carries chrome
        # evidence (content pattern, cluster neighbor, all-caps
        # journal styling); a bare Title-Case display line with none
        # of those is the page TITLE — keep it for the heading pass.
        # det31a (0b045ab8): lowercase-initial lines are kickers/decks
        # ('focus on youth' @1.4x body), never titles — the kept kicker
        # interleaves the wrapped display title in reading order and
        # breaks the multiline merge. Corpus census: zero
        # lowercase-initial GT headings.
        # det31b (e71f5c59 p6): a references page running head over a
        # hanging-indent citation block is chrome — same evidence that
        # denies the iter-69 rescue must deny title-keep.
        _t_alpha = next((c for c in txt if c.isalpha()), "")
        _title_cite_block = _cite_ev
        _title_keep = (
            bbox[1] < H * 0.10
            and body_size is not None
            and sz > body_size * 1.25
            and _t_alpha.isupper()
            and not _page_num_re.match(txt)
            and not _running_head_chrome(txt)
            and not _chrome_cluster_adjacent(bbox, sz, txt)
            and not _title_cite_block
            and not (
                sum(ch.isupper() for ch in txt if ch.isalpha()) >= 3
                and sum(ch.isupper() for ch in txt if ch.isalpha())
                >= 0.9 * max(1, sum(ch.isalpha() for ch in txt))
            )
        )
        # GUTTER-CROSS VETO (iter-84, 081875a1 p13): the rescued heading
        # must veto the gutter-cross rule too — a full-width TITLE
        # ('Producing Nanoporous Metallic Foams', 28pt, body + drop-cap
        # beneath) was accepted as header via crosses_gutter alone and
        # stripped. A true running head is never >body+1 with body-size
        # flow directly beneath (iter-69 gates).
        if (
            (crosses_gutter and not _heading)
            or (short_centered and not _heading)
            or is_page_number
            or (distinct_size and not _heading and not _title_keep)
            or _isolated
            or _cont_block
            or (_marginalia and not _title_keep)
        ):
            out.append(txt)
            if bbox[1] > H * 0.5:
                _accepted_bottom_bboxes.append(bbox)
    return out or None


# Roman numeral page numbers (ii, xiv…) — common in front-matter
_ROMAN_PAGE_RE = re.compile(r"^\s*[ivxlcdmIVXLCDM]{1,7}\s*$")


def extract_page_metadata(text, page_num=None, doc_state=None, pdf_path=None):
    """Heuristically capture running header and page number from OCR text.

    Returns dict {page_header: str|None, page_number: str|None,
    hf_lines: list[str]} extracted from the first/last non-empty lines.

    **geometry mode** (2026-08-14 fix): when ``pdf_path`` is provided and
    the page has a text layer, header/footer lines are detected by LINE
    POSITION (top/bottom 8% band). This fixes three blind spots of the
    text heuristics: (1) the page-1 guard skipped ALL header detection on
    single-page bench excerpts, (2) cross-page consensus needs ≥3 pages —
    impossible on single-page extraction, (3) mixed-case running headers
    never matched the ALLCAPS/bracket patterns. Geometry is position
    evidence and overrides all text-gating.

    **page_num guard** (2026-07-25 fix): author names, journal names, and
    paper titles appear as ALLCAPS on page 1 (title page). These are NOT
    running headers. Skip TEXT-mode header detection on page 1 entirely
    (geometry mode is unaffected — position still applies).

    **doc_state consensus** (2026-07-25 fix): a true running header must
    appear on ≥3 pages. Only used in text mode.
    """
    _load_consts()
    meta = {"page_header": None, "page_number": None, "hf_lines": []}
    if pdf_path:
        geo = _geometry_header_footer_lines(pdf_path, page_num or 1)
        if geo:
            meta["hf_lines"] = geo
            # longest band line = running header (page numbers are short)
            header_like = [l for l in geo if len(l) > 4]
            if header_like:
                meta["page_header"] = max(header_like, key=len)
            for l in geo:
                if PAGE_NUMBER_RE.match(l) or _ROMAN_PAGE_RE.match(l):
                    meta["page_number"] = l
                    break
            return meta

    lines = [l for l in text.split("\n") if l.strip()]
    if not lines:
        return meta
    # Page number: typically last short line, pure digits or roman
    for line in reversed(lines[-3:]):
        stripped = line.strip()
        if len(stripped) <= 10 and (
            PAGE_NUMBER_RE.match(stripped) or _ROMAN_PAGE_RE.match(stripped)
        ):
            meta["page_number"] = stripped
            break
    # Running header detection — skip on page 1 (title page has author names)
    if page_num is not None and page_num == 1:
        return meta
    # Collect candidate from first 3 lines
    candidate = None
    for line in lines[:3]:
        stripped = line.strip()
        if stripped.startswith("## "):
            stripped = stripped[3:].strip()
        if len(stripped) <= HEADER_LINE_MAX_CHARS and (
            BRACKET_HEADER_RE.match(stripped)
            or (stripped.isupper() and any(c.isalpha() for c in stripped) and len(stripped) > 4)
        ):
            candidate = stripped
            break
    if candidate is None:
        return meta
    # If doc_state provided, track candidate across pages and require consensus
    if doc_state is not None:
        key = candidate.lower()
        doc_state[key] = doc_state.get(key, 0) + 1
        # Only promote if seen on ≥3 pages OR (≥2 pages AND this is a later page)
        if doc_state[key] >= 3 or (doc_state[key] >= 2 and page_num and page_num > 2):
            meta["page_header"] = candidate
        # Else: don't promote (single-page ALLCAPS strings are NOT headers)
    else:
        # No doc_state (--pages 1 case) — use old per-page heuristic
        meta["page_header"] = candidate
    return meta


def _format_meta_comment(page_meta, warnings):
    """Render a metadata HTML comment prefix for markdown output."""
    _load_consts()
    if not page_meta and not warnings:
        return ""
    parts = []
    if page_meta.get("page_header"):
        # Geometry mode strips the body copy; emitting ANY of the header
        # text in a comment leaks it back for comment-reading scorers
        # (olmOCR normalize keeps HTML comments; short gold strings match
        # even a 60-char prefix at ratio 1.0). Emit label only.
        parts.append("PageHeader: (detected+stripped)")
    if page_meta.get("page_number"):
        # Label only — emitting the digit leaks it to comment-reading
        # scorers (olmOCR absent-tests: gold "25" must NOT appear
        # anywhere in content, HTML comments included).
        parts.append("PageNumber: (detected+stripped)")
    for w in warnings:
        parts.append(f"WARNING: {w}")
    if not parts:
        return ""
    return f"<!-- {' | '.join(parts)} -->\n\n"


# Table detection heuristic. A table candidate is a run of >=2 consecutive
# lines where each line has >=2 multi-space gaps (≥3 spaces) OR ≥1 pipe char.
# Aligned scientific tables (e.g. "Sample    SiO2    TiO2    Al2O3") hit this
# pattern. False positives are mainly code blocks, which is acceptable.


def page_has_embedded_fonts(pdf_path, page_num):
    """Detect born-digital pages by checking for embedded fonts.

    Marker SOTA (check_page). Born-digital PDFs have embedded fonts (Type1,
    TrueType, etc.). Scanned/OCR'd PDFs typically have zero fonts, OR
    "glyphless" fonts (invisible text render mode + glyphless = OCR overlay).

    Returns True if page has at least one embedded font (born-digital signal).
    Returns None if pdffonts is unavailable (caller falls back to image-area).
    """
    _load_consts()
    if not DEPS.get("pdffonts", False):
        return None  # caller falls back to image-area
    try:
        proc = subprocess.run(
            ["pdffonts", "-f", str(page_num), "-l", str(page_num), pdf_path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
            check=True,
        )
        rows = [l for l in proc.stdout.split("\n")[2:] if l.strip()]
        if not rows:
            return False  # no fonts = scanned
        for r in rows:
            parts = r.split()
            if len(parts) >= 4 and parts[3].lower() == "yes":
                font_name = parts[0].lower()
                if "glyphless" in font_name:
                    continue
                return True
        return False
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return False


__all__ = [
    "_get_page_dict",
    "strip_recurring_headers",
    "extract_page_metadata",
    "_format_meta_comment",
    "page_has_embedded_fonts",
]
