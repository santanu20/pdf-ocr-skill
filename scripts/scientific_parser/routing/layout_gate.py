"""Layout-model skip gate (reviewer #15).

The layout detector (RapidLayout) is a CNN — running it on every page is
wasteful when a page is obviously simple: a single column of native text with
no tables, equations, figures, or heavy image content. Such pages get nothing
from layout perception; the native text layer is already in reading order.

This gate answers: "should I invoke the layout model on this page?" It returns
False (skip) for simple pages so the caller can take the native-text fast path.
It is conservative — when in doubt it returns True (run layout), because a
missed multi-column split is more costly than one CNN forward pass.

Signals (any one triggers layout):
  - multi-column detected (>1)
  - table / equation / figure content present
  - low text density (likely scanned or image-dominant)
  - high image coverage

Inputs are lightweight signals the page profiler already computes; this gate
is pure, deterministic, and unit-testable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PageProfile:
    """Lightweight page fingerprint used by the layout gate.

    All fields optional — callers provide what they have; missing fields are
    treated conservatively (assume the page is NOT trivially simple).
    """

    text_chars: int = 0
    n_columns: int = 1
    has_tables: bool = False
    has_equations: bool = False
    has_figures: bool = False
    image_coverage: float = 0.0  # fraction of page area covered by images [0,1]
    text_density_ok: bool = True  # native text layer is substantial
    # Geometric table signals (2026-08-15, tables-bench fix): pdftotext
    # carries neither pipe rows nor "Table N" captions on many table pages
    # (single-page excerpts, borderless tables), so the text-marker
    # has_tables misses them and the gate skips layout -> 0 regions ->
    # 0 extracted tables (measured 34 table-test pages, 159 test fails).
    has_native_tables: bool = False  # PyMuPDF find_tables() found >=1
    has_borderless_table: bool = False  # x-recurrence signature on words
    # True ONLY when structural signals (columns/tables) were measured from a
    # reliable source (native text-block geometry). Flat OCR text cannot carry
    # these signals, so an OCR-only page is structure_known=False -> the gate
    # conservatively runs layout rather than trusting a blind "looks simple".
    structure_known: bool = True


# Thresholds (conservative — favour running layout on uncertainty).
_MIN_TEXT_CHARS = 200  # below this, page is suspicious (scanned/garbled)
_MAX_IMAGE_COVERAGE = 0.30  # >30% image -> layout perception likely valuable
# Side-by-side text blocks must share this much vertical extent (PDF points)
# to count as parallel columns, not a stray sidebar or pull-quote.
_COLUMN_Y_OVERLAP = 20.0


def count_columns(blocks, page_width):
    """Detect multi-column layout from native text-block geometry.

    Pure function over PyMuPDF ``get_text("blocks")`` output (no rendering,
    no CNN). Returns 2 only when substantial blocks sit side-by-side (one
    ending left of the page midline, one starting right of it) across an
    overlapping vertical band — the signature of parallel columns.

    A false positive (running layout on a single-column page) is safe and
    cheap; a false negative (the old hardcoded ``n_columns = 1``) is the bug
    this fixes — it skipped layout on multi-column pages whose structure was
    invisible in flat OCR text.
    """
    if not blocks or page_width <= 0:
        return 1
    mid = page_width / 2.0
    tol = page_width * 0.05
    left, right = [], []
    for b in blocks:
        if len(b) < 5:
            continue
        x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
        if len(txt.strip()) < 20:
            continue
        if x1 < mid + tol:
            left.append((y0, y1))
        if x0 > mid - tol:
            right.append((y0, y1))
    if len(left) < 2 or len(right) < 2:
        return 1
    for ly0, ly1 in left:
        for ry0, ry1 in right:
            if min(ly1, ry1) - max(ly0, ry0) >= _COLUMN_Y_OVERLAP:
                return 2
    return 1


def detect_borderless_table(words, min_lines: int = 8) -> bool:
    """x-coordinate-recurrence signature of a borderless text table.

    Pure function over PyMuPDF ``get_text("words")`` output (tuples of
    x0, y0, x1, y1, word, block, line, word_no). Borderless tables that
    find_tables() misses (no ruling lines) still show: many short lines
    (few words each) + >=3 x0 positions recurring across a fifth of lines
    — the column-grid signature. Prose never recurs on exact x positions.
    """
    if not words or len(words) < min_lines * 2:
        return False
    lines: dict[int, list] = {}
    for w in words:
        lines.setdefault(round(w[3]), []).append(w)  # group by bottom-y
    n_lines = len(lines)
    if n_lines < min_lines:
        return False
    short = sum(1 for ws in lines.values() if len(ws) <= 6)
    if short / n_lines < 0.6:
        return False
    from collections import Counter

    # Cluster x0 with ±5pt tolerance before counting recurrence — font
    # metrics/alignment shift column starts a few points per row, so
    # exact-value recurrence undercounts (measured: catalog table page
    # showed recurring_x0=1 with 6 real columns).
    xs = sorted(round(w[0]) for ws in lines.values() for w in ws)
    clusters = []
    for x in xs:
        if clusters and x - clusters[-1][-1] <= 5:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    recurring = sum(1 for cl in clusters if len(cl) >= max(4, n_lines * 0.2))
    return recurring >= 3


def should_run_layout(profile: PageProfile) -> bool:
    """Return True if the layout model should run on this page.

    Returns False (skip) ONLY when the page is confidently simple: a single
    column with substantial native text and no tables/equations/figures and
    low image coverage. Any ambiguity -> True (run layout).
    """
    # Unknown structure (scanned page, no native text layer, or OCR-derived
    # text) -> MUST run layout. Flat text cannot prove a page is simple; the
    # multi-column / table signals that motivate layout are invisible in it.
    if not profile.structure_known:
        return True
    # Hard signals: any of these makes layout perception valuable.
    if profile.n_columns > 1:
        return True
    if profile.has_native_tables or profile.has_borderless_table:
        return True
    if profile.has_tables or profile.has_equations or profile.has_figures:
        return True
    if profile.image_coverage > _MAX_IMAGE_COVERAGE:
        return True
    if not profile.text_density_ok:
        return True
    # Otherwise the page is plausibly simple — but only SKIP when we have
    # enough native text to trust. Thin text -> run layout (may be scanned).
    if profile.text_chars < _MIN_TEXT_CHARS:
        return True
    return False  # simple single-column text page -> skip layout


def should_skip_layout(profile: PageProfile) -> bool:
    """Inverse of :func:`should_run_layout` for readable call sites."""
    return not should_run_layout(profile)


# ── Column geometry (shared: router signal-10 + header band guard) ───────── #

# Lines whose left edges differ by more than this (PDF points) start a new
# left-edge cluster. Paragraph indents are 10-20pt; column starts are 150pt+.
_LEFT_EDGE_CLUSTER_GAP_PT = 15.0
# A page is "column-sharp" when the two largest left-edge clusters cover at
# least this fraction of lines — measured on dense academic 2-col pages
# (0.90+) vs book front-matter/novel pages (scattered, 0.45) 2026-08-17/18.
_COLUMN_SHARP_TOP2_COVERAGE = 0.70
# Two dominant clusters closer than this (pt) are one column (indents).
_COLUMN_MIN_SEPARATION_PT = 40.0
# How far right of a column start a line may sit (hanging indents,
# paragraph starts) and still count as that column (2026-08-18).
_INDENT_TOLERANCE_PT = 60.0
# Minimum lines whose left edges sit at the right-hand cluster start for
# that cluster to be a COLUMN (iter-94): 2 = a stamp/float, 3+ = a column.
_COLUMN_RIGHT_MIN_LINES = 3


def left_edge_column_sharpness(lines, gap_pt=_LEFT_EDGE_CLUSTER_GAP_PT):
    """Cluster line left-edges; return (top2_coverage, col2_left_x).

    Single shared implementation for every "is this really a 2-column
    page?" decision (audit-2 R4: the router's signal-10 probe and the
    header-band column guard had drifted into two private clusterings
    with different constants).

    Args:
      lines: iterable of ``(x0, y0, x1, y1)`` bboxes in PDF points
              (PyMuPDF block or line bboxes).
      gap_pt: left-edge gap that starts a new cluster.

    Returns:
      (top2_coverage, col2_left): fraction of lines in the two largest
      clusters, and the left edge of the right-hand dominant cluster
      (None when the page is not column-sharp / single column).
    """
    lefts = sorted(x0 for x0, *_ in lines)
    if len(lefts) < 4:
        return 1.0, None
    clusters = [[lefts[0]]]
    for x in lefts[1:]:
        if x - clusters[-1][-1] > gap_pt:
            clusters.append([x])
        else:
            clusters[-1].append(x)
    top2 = sorted(clusters, key=len, reverse=True)[:2]
    # Single-cluster page (all left-edges aligned — plain left-aligned
    # prose): no second column exists. Guard the 2-unpack explicitly —
    # crash measured 2026-08-18 bench (ValueError, 5 fixtures across 3
    # categories, e.g. 06f8837d headers_footers).
    if len(top2) < 2:
        return 1.0, None
    a, b = sorted(top2, key=lambda c: c[0])
    if b[0] - a[0] < _COLUMN_MIN_SEPARATION_PT:
        return sum(len(c) for c in top2) / len(lefts), None
    # Indent-tolerant coverage (2026-08-18): real right columns carry
    # hanging indents/paragraph starts 20-60pt right of the column edge
    # (measured 0514f667: column at 302, reference entries 336-363) —
    # exact-cluster top-2 undercounts them as a third "column" (0.59 on
    # a genuine 2-col page). Membership by proximity to a dominant start
    # recovers the true structure (0.86).
    centers = [a[0], b[0]]
    near = sum(1 for x in lefts if any(abs(x - c) <= _INDENT_TOLERANCE_PT for c in centers))
    # RIGHT-CLUSTER POPULATION FLOOR (iter-94, cfbe79724): a lone
    # right-edge element — a 'page 3 / 4' stamp at x=497 on a 595pt
    # page — is its own left-edge cluster, and the coverage test only
    # checks proximity to the two dominant starts, so a 9+1 line page
    # scored coverage 1.0 and hallucinated col2=497. The strict col2
    # acceptance branch then ate the body first line (gutter-cross) and
    # spared the real masthead. A real second column carries a
    # population; one element is a stamp or a float, never a column.
    coverage = near / len(lefts)
    right_pop = sum(1 for x in lefts if abs(x - b[0]) <= _INDENT_TOLERANCE_PT)
    if right_pop < _COLUMN_RIGHT_MIN_LINES:
        return coverage, None
    if coverage < _COLUMN_SHARP_TOP2_COVERAGE:
        return coverage, None
    # COLUMN EDGE = CLUSTER MEDIAN (2026-09-03): the cluster MIN is
    # snagged by elements that merely sit within gap_pt of the true
    # column start — a centered page heading ('MARTIAL LAW' x=166 on a
    # 213-edge column, 13_pg811) or right-column indent lines (188 vs
    # edge 210, 13_pg639). Every left-column line then 'crossed the
    # gutter' (x1 ~204 > col2-5+10) and the header band guard ate the
    # left column's body. The median is the justify edge the column's
    # population actually uses; b is already sorted.
    return coverage, b[len(b) // 2]


def column_sharp(lines):
    """Boolean convenience: do two dominant columns structure this page?"""
    return left_edge_column_sharpness(lines)[1] is not None


__all__ = [
    "PageProfile",
    "should_run_layout",
    "should_skip_layout",
    "count_columns",
    "detect_borderless_table",
    "left_edge_column_sharpness",
    "column_sharp",
]


# ── Table-chain gate (EXP-1, 2026-09-15) ──────────────────────────────────── #
# detect_all_tables runs a 7-strategy chain (pdfplumber + camelot + …) per
# page, ~1.6s on a NO-TABLE page (profiled 0005784d). This gate answers the
# cheaper question first: "should the chain run at all?"
#
# Two signal tiers, OR-combined, cheapest first (ODL TriageProcessor
# philosophy — ported idea, Apache-2.0):
#   EVIDENCE — the page plausibly holds a table: free text probes, native
#     find_tables (memo), x-recurrence words, ruled vector grid.
#   SUSPICION — the page's text SHAPE matches populations where the
#     last-resort strategies (positional, img2table) measurably fire:
#     thin text layers (scanned), long runs of short lines, short-line
#     majorities. These are not table evidence; they stop the gate from
#     skipping pages the chain would still carve (shadow-measured: covers
#     154/235 firings incl. every tables-category page).
#
# Shadow-verified on the full 1269-PDF fast-path bench (run 20): gate-out
# pages where the chain still fires carry NO table tests (all sit in
# arxiv/multicol/hf/tiny); their emissions are last-resort false-positive
# carves over prose. Fail-safe: any probe failure → True (run the chain).
_TABLE_CAPTION_RE = re.compile(r"\bTable\s+(?:[A-Z]{0,3}\d{1,3}|[IVXLCDM]{1,6})\b")
_PIPE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$", re.M)
_GRID_LINE_RE = re.compile(r"\S+\s{3,}\S+.*\s{3,}\S+")
# Suspicion thresholds (calibrated by signal sweep:
# short_run>=6 covers 123/235 shadow firings at 543/1034 non-firing cost —
# combined with the other tier-2 probes the gate leaves only the 81
# prose-page last-resort false positives, which carry no bench tests).
_SHORT_LINE_TOKENS = 4  # a line with <=4 tokens is "short" (cell/label-like)
_SUSPICION_SHORT_RUN = 6  # >=6 consecutive short lines (grid/roster shape)
_SUSPICION_FRAC = 0.5  # short-line majority...
_SUSPICION_MIN_LINES = 8  # ...over at least 8 lines


def _text_table_signals(text) -> bool:
    """Free text probes over the page text already in hand."""
    if not text:
        return False
    if len(_PIPE_ROW_RE.findall(text)) >= 2:
        return True  # pipe rows already present — detection must arbitrate
    if _TABLE_CAPTION_RE.search(text):
        return True
    grid_lines = sum(1 for ln in text.split("\n") if _GRID_LINE_RE.search(ln))
    return grid_lines >= 4  # >=4 lines with >=2 aligned multi-space gaps


def _suspicion_signals(text) -> bool:
    """Text-shape probes for populations where last-resort strategies fire."""
    if not text or not text.strip():
        return True  # thin/empty layer: cannot judge → run the chain
    stripped = text.strip()
    if len(stripped) < _MIN_TEXT_CHARS:
        return True  # scanned/garbled — img2table fires on the RENDER
    lines = [ln for ln in stripped.split("\n") if ln.strip()]
    if len(lines) >= _SUSPICION_MIN_LINES:
        short = [len(ln.split()) <= _SHORT_LINE_TOKENS for ln in lines]
        run = best = 0
        for s in short:
            run = run + 1 if s else 0
            best = max(best, run)
        if best >= _SUSPICION_SHORT_RUN:
            return True
        if sum(short) / len(lines) >= _SUSPICION_FRAC:
            return True
    return False


def _vector_grid_signals(pdf_path: str, page_num: int, bbox=None) -> bool:
    """ODL TriageProcessor vector signals over fitz drawings: a ruled grid
    needs >=3 horizontal AND >=3 vertical strokes (or >=8 total rules).
    With ``bbox`` (x0, y0, x1, y1 in PDF points) only strokes whose midpoint
    lies inside the rect count — region-scoped probe for model escalation."""
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            n_h = n_v = 0
            for dr in page.get_drawings():
                for item in dr["items"]:
                    if item[0] == "l":
                        p1, p2 = item[1], item[2]
                        if bbox is not None:
                            mx, my = (p1.x + p2.x) / 2, (p1.y + p2.y) / 2
                            if not (bbox[0] <= mx <= bbox[2] and bbox[1] <= my <= bbox[3]):
                                continue
                        dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                        if dy <= 1.0 and dx >= 5.0:
                            n_h += 1
                        elif dx <= 1.0 and dy >= 5.0:
                            n_v += 1
            return (n_h >= 3 and n_v >= 3) or (n_h + n_v >= 8)
    except Exception:
        return False  # probe failure handled by the caller's fail-safe


def _raster_overlap_ratio(pdf_path: str, page_num: int, bbox) -> float:
    """Fraction of the bbox covered by raster images (cheap, no OCR).

    Raster tables (tables exported as images) have NO text-layer evidence
    by construction — text probes are structurally blind to them, while
    the RapidTable CNN reads pixels and can extract them.
    """
    try:
        import fitz as _fz

        with _fz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            x0, y0, x1, y1 = bbox
            area = max((x1 - x0) * (y1 - y0), 1.0)
            covered = 0.0
            for info in page.get_image_info():
                r = info.get("bbox")
                if not r:
                    continue
                ix0, iy0, ix1, iy1 = r
                ow = max(0.0, min(x1, ix1) - max(x0, ix0))
                oh = max(0.0, min(y1, iy1) - max(y0, iy0))
                covered += ow * oh
            return min(covered / area, 1.0)
    except Exception:
        return 0.0


def region_has_table_evidence(pdf_path: str, page_num: int, bbox, region_text: str) -> bool:
    """Region-scoped evidence that a dispatched table region deserves the
    model fallback (RapidTable CNN). Pseudo-table regions on prose pages pay
    ~9s of CNN inference for output the accept gates then reject (bench3
    census: 78 of 98 escalations rejected). Evidence ladder, cheapest first:
    thin/absent text (scan crop or sparse borderless table), pipe/caption/
    aligned-gap text signals, a ruled vector grid inside the bbox, or raster
    image coverage >= 0.5 (raster tables carry no text evidence by
    construction — bench10 9a61fe78: raster FUND table inside a tall table
    region whose text layer is only contact-prose).
    Probes fail soft: an unreadable PDF with dense prose returns False (the
    CNN needs a rendered crop, which such a PDF cannot provide)."""
    try:
        stripped = (region_text or "").strip()
        if len(stripped) < _MIN_TEXT_CHARS:
            return True
        if _text_table_signals(stripped):
            return True
        if _vector_grid_signals(pdf_path, page_num, bbox):
            return True
        return _raster_overlap_ratio(pdf_path, page_num, bbox) >= 0.5
    except Exception:
        return True


def _raster_table_likely(page) -> bool:
    """Table-ish raster geometry (det43 filter): >=80x40pt, aspect 0.8-12.

    Measured (run-52): ~15% of pages carry one; the router otherwise
    blocks the whole table chain on prose-heavy raster-table pages whose
    text layer carries only prose + partial table text (3a91fad8 231
    words, 587a1f5a 48 words, odlbench doc-122 266 words) — the raster
    OCR strategies (det42/det43) never fire.
    """
    try:
        for xref in page.get_images(full=True):
            for r in page.get_image_rects(xref[0]):
                w, h = r.width, r.height
                if w >= 80 and h >= 40 and 0.8 <= (w / max(h, 1e-6)) <= 12:
                    return True
    except Exception:
        return True  # fail-safe: cannot prove absence → run the chain
    return False


def raster_only_table_evidence(pdf_path: str, page_num: int, text: str) -> bool:
    """det48: True when the ONLY table evidence is raster geometry.

    On such pages the router opened the full 7-strategy chain (det47),
    and the text-geometry strategies became the FP source — measured
    odlbench det47: 0056/0066/0098/0152 positional/img2table grids on
    chart-bearing non-table pages (GT keeps the text plain). Scope the
    chain to the raster strategy alone: recall (doc-122 raster table)
    without the text-strategy FP flood. Fail-safe False: on any probe
    exception the caller runs the full chain via the normal gate.
    """
    if _text_table_signals(text) or _suspicion_signals(text):
        return False
    try:
        from scientific_parser.perception.tables.detector import native_find_tables

        if native_find_tables(pdf_path, page_num):
            return False
        import fitz

        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            raw_text = page.get_text("text")
            if _text_table_signals(raw_text) or _suspicion_signals(raw_text):
                return False
            if detect_borderless_table(page.get_text("words")):
                return False
            if count_columns(page.get_text("blocks"), page.rect.width) >= 2:
                return False
            if not _raster_table_likely(page):
                return False
        if _vector_grid_signals(pdf_path, page_num):
            return False
        return True
    except Exception:
        return False


def page_needs_table_chain(pdf_path: str, page_num: int, text: str) -> bool:
    """True when the detect_all_tables chain should run on this page.

    False = shadow-verified safe skip: no table evidence AND no
    last-resort firing shape. Fail-safe: on any geometry-probe exception
    returns True (the chain then runs — perf loss bounded, correctness
    preserved).

    SURFACE-INDEPENDENCE (exp1-gate3, 2026-09-15): the caller passes
    STAGE text (post band-strip/dehyphenate/cleanup at order-900), but
    the shadow census validated the gate against the RAW text layer.
    Stripped band lines and repaired hyphens carry exactly the short-run
    /thin-text signatures suspicion feeds on (measured: 53d3f304_pg3
    gate=False in-pipeline on clean stage text while raw fires → 32
    tables PASS->FAIL). The gate therefore probes BOTH surfaces: caller
    stage text AND the raw page text fetched inside the doc-open it
    already does. Any surface firing → True.
    """
    if _text_table_signals(text) or _suspicion_signals(text):
        return True
    try:
        from scientific_parser.perception.tables.detector import native_find_tables

        if native_find_tables(pdf_path, page_num):
            return True
        import fitz

        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            # Raw-surface probes: same text signals on the unmodified
            # layer (see surface-independence note above).
            raw_text = page.get_text("text")
            if _text_table_signals(raw_text) or _suspicion_signals(raw_text):
                return True
            if detect_borderless_table(page.get_text("words")):
                return True
            # 2-COLUMN REPAIR TRAP (exp1-gate bench, 2026-09-15): on
            # two-column pages the positional/img2table carves act as an
            # accidental COLUMN-ORDER REPAIR — the raw body layer
            # interleaves or shatters column text (gold phrases split
            # across fragments, >max_l_dist), while the carved table
            # reads each column intact. No text feature separates repair
            # pages from neutral ones (measured 52-page sweep: flip pages
            # span the same short-frac/run/char ranges as neutrals);
            # column geometry does — all 8 PASS->FAIL flips were
            # count_columns==2, both preserved FAIL->PASS gains were ==1.
            # The chain is the only component that reads 2-col order
            # correctly on these layers, so a 2-col page always needs it.
            if count_columns(page.get_text("blocks"), page.rect.width) >= 2:
                return True
            # RASTER-TABLE REACHABILITY (det47, run-52): a table-sized
            # raster with thin text layer — the only route to the raster
            # OCR strategies (det42/det43) is through this chain.
            if _raster_table_likely(page):
                return True
    except Exception:
        return True  # fail-safe: cannot prove absence → run detection
    return _vector_grid_signals(pdf_path, page_num)


__all__ += [
    "page_needs_table_chain",
    "region_has_table_evidence",
]
