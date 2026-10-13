"""Table detection + reconstruction pipeline.

Extracted from scripts/pdf_ocr.py during Phase R2 refactoring (2026-08-12).
Multi-strategy table detection (pdfplumber, camelot, lattice, text-only,
figure-based) + cross-page merging + markdown rendering.
"""

from __future__ import annotations

import bisect
import re
import os
import logging
import subprocess
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state
# _is_numeric_cell: provided by shared.py (single source of truth).

# Intra-package imports for helpers extracted to sibling modules during R2.
# These were left as dangling references in the original extraction; bound
# explicitly here so the module is self-contained.
from .structure import (
    validate_table_structure,
    _line_to_table_cells,
    classify_cell,
    infer_spanning_cells,
    _CTRL_CELL_SCRUB_RE,
    scrub_cell_text,
)
from ...core.confidence_helpers import _quality_gate
from ...routing.page_metadata import _get_page_dict


def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


_CONSTS_LOADED = False


# ── Native find_tables memo — single source of truth ──────────────────────── #
# Up to 4 consumers opened their own fitz doc and ran find_tables on the
# SAME (pdf, page, strategy) per page (profiled 2026-09-15: 4 calls ≈0.8s on
# a multicol page). All probes now route through this memo; results are
# extracted to plain data immediately (no live doc/page refs held). Cache is
# process-lifetime and keyed on (abspath, mtime_ns, page, strategy) — inputs
# are read-only in bench + single-PDF flows; mtime guards out-of-band edits.
_FIND_TABLES_CACHE: dict[tuple, list] = {}
_FIND_TABLES_CACHE_MAX = 4096

# EXP-1.5 (2026-09-16, routing audit): symbol fonts on born-digital pages
# emit 0x07-0x1f control glyph codes into word extraction; table cells built
# from those words carry the noise into final output. The scrub lives in
# structure.py (lowest tables-lane module) and is inherited here. \t\n\r kept.


# det32 (2026-09-23, odlbench TEDS census): ground-truth table cells are
# ASCII-convention (corpus census: hyphen-after-digit 21, en-dash 0, curly
# quotes 0) while our extractors emit en/em dashes (28 digit-en-dash
# occurrences in predictions) and camelot cells embed raw newlines that
# break the emitted pipe rows. Fold once at the shared scrub point so
# every strategy's markdown AND html agree with the GT convention.
_CELL_FOLD = str.maketrans(
    {
        "\u2013": "-",
        "\u2014": "-",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
    }
)


def _fold_cell(c):
    """Collapse whitespace runs + fold smart punctuation in one table cell."""
    if not isinstance(c, str):
        return c
    return re.sub(r"\s+", " ", c).strip().translate(_CELL_FOLD)


def _drop_furniture_row(rows):
    """Drop a leading furniture mega-cell row above the true grid.

    Strategies whose bbox reaches into the page's top band glue running
    furniture (page number + author line) into row 0 as ONE multi-word
    cell (measured odlbench 01030000000053 positional: '16 Encinas Franco
    and Laguna' became the pipe header, 8 padded columns vs the 4-column
    GT). Signature: exactly one non-empty first-row cell of >=3 words,
    above rows whose modal non-empty width is >=2 — a title/furniture
    line, never data.
    """
    if len(rows) < 3:
        return rows

    def _nonempty(r):
        return [str(c).strip() for c in (r or []) if str(c).strip()]

    first = _nonempty(rows[0])
    if len(first) != 1 or len(first[0].split()) < 3:
        return rows
    widths = [len(_nonempty(r)) for r in rows[1:]]
    modal = max(set(widths), key=widths.count) if widths else 0
    if modal < 2:
        return rows
    return rows[1:]


# det34 (2026-09-24, olmOCR headers_footers P2F class): tables appended at
# document end put their grid tail inside the bench manual probes' last-N
# character window, so page furniture absorbed by the grid (bare page
# numbers, journal running heads, source/publication notes, statute IDs)
# fails the page-number absent probes. Years are excluded from the page-
# digit signature so legit 'Year | 2023' style edge rows survive.
_PAGE_DIGIT_RE = re.compile(r"^(?!19\d\d|20\d\d)\d{1,4}$")
_VOLNO_EDGE_RE = re.compile(r"\bVol\.?\s*\d+\b[^\n]{0,80}?\bNo\.?\s*\d+\b", re.I)
_STATUTE_RE = re.compile(r"\b[A-Z]{2,4}\s?\d{4}:\d+\b")
_PUBNOTE_RE = re.compile(
    r"(?i)\b(utkom|tryckt|printed|published|date of issue|fonte|fuente|quelle|source)\b"
)


def _modal_row_width(ne):
    """Modal nonempty-cell count — the grid's own column structure."""
    if not ne:
        return 0
    counts = {}
    for r in ne:
        counts[len(r)] = counts.get(len(r), 0) + 1
    return max(counts.items(), key=lambda kv: (kv[1], kv[0]))[0]


def _edge_row_is_furniture(cells, modal_w=None):
    """Furniture signature for a leading/trailing grid row.

    Measured shapes (det32 bench P2F docs): '| Milken Institute School of
    Public Health | 6 |' (signature + bare page number), '| Fonte: Bacen |
    ... | 42 |', '| | The Korean Journal ... Vol 19 No 1 / 2008 43 |'
    (running head glued as one cell), '| Lag om ...; | Utkom fran trycket
    ... 2016 |' (statute publication note). The page-digit arm needs a
    multi-word companion cell AND must sit OUTSIDE the grid's own column
    structure (det39: below modal row width) — uniform label+count data
    tables ('acid base | 26', b773892d pg1) have every row at modal width
    and are NOT furniture.
    """
    if not cells:
        return True
    digits = [c for c in cells if _PAGE_DIGIT_RE.match(c)]
    words = [c for c in cells if not _PAGE_DIGIT_RE.match(c)]
    if (
        (modal_w is None or len(cells) < modal_w)
        and len(cells) <= 2
        and digits
        and any(len(w.split()) >= 2 for w in words)
    ):
        return True
    if _VOLNO_EDGE_RE.search(" ".join(cells)):
        return True
    if len(cells) <= 3 and any(_STATUTE_RE.search(c) or _PUBNOTE_RE.search(c) for c in cells):
        return True
    return False


def _scrub_edge_furniture(rows):
    """Drop furniture-shaped rows at grid edges, iterated.

    Dropping a tail row re-exposes the new tail (statute ID beneath the
    publication note), so the sweep repeats until the edges stabilise.
    Bounded to three passes.
    """
    for _ in range(3):
        if len(rows) < 2:
            return rows
        ne = [[str(c).strip() for c in (r or []) if str(c).strip()] for r in rows]
        modal_w = _modal_row_width(ne)
        # All-empty rows are padding, not furniture — the emitters drop
        # them; treating them as furniture makes the scrub non-idempotent.
        if ne[-1] and _edge_row_is_furniture(ne[-1], modal_w):
            rows = rows[:-1]
            continue
        if ne[0] and _edge_row_is_furniture(ne[0], modal_w):
            rows = rows[1:]
            continue
        break
    return rows


def _table_all_furniture(rows):
    """True when a tiny table is nothing but the page's furniture block.

    Measured 4f4c20ab: the detected 'table' was the running furniture
    itself (series title row / statute ID / publication note) — no data
    grid on the page at all. Emitting it puts furniture in the output;
    veto instead.
    """
    ne = [[str(c).strip() for c in (r or []) if str(c).strip()] for r in rows]
    ne = [r for r in ne if r]
    if not 2 <= len(ne) <= 4:
        return False
    modal_w = _modal_row_width(ne)

    def _signed(r):
        joined = " ".join(r)
        digits = [c for c in r if _PAGE_DIGIT_RE.match(c)]
        words = [c for c in r if not _PAGE_DIGIT_RE.match(c)]
        return (
            len(r) == 1
            or (
                (modal_w is None or len(r) < modal_w)
                and len(r) <= 2
                and digits
                and any(len(w.split()) >= 2 for w in words)
            )
            or _VOLNO_EDGE_RE.search(joined) is not None
            or any(_STATUTE_RE.search(c) or _PUBNOTE_RE.search(c) for c in r)
        )

    signed = sum(1 for r in ne if _signed(r))
    return signed >= max(2, len(ne) - 1)


def _scrub_grid_rows(rows):
    """Return rows scrubbed: furniture edge rows dropped, whitespace
    runs collapsed, smart punctuation folded, control-glyph noise stripped.

    Idempotent — build_table_markdown/normalize_grid_rows call it in
    sequence on the same rows.
    """
    rows = _drop_furniture_row(rows)
    rows = _scrub_edge_furniture(rows)
    if _table_all_furniture(rows):
        return []

    def _needs(c):
        return isinstance(c, str) and (
            _CTRL_CELL_SCRUB_RE.search(c)
            or re.search(r"[\n\r\t]| {2,}|[\u2013\u2014\u2018\u2019\u201c\u201d]", c)
        )

    if not any(_needs(c) for r in rows for c in (r or ())):
        return rows

    def _scrub(c):
        if not isinstance(c, str):
            return c
        return _CTRL_CELL_SCRUB_RE.sub("", _fold_cell(c))

    return [[_scrub(c) for c in r] if r else r for r in rows]


def looks_like_leader_page_list(rows) -> bool:
    """2-col [title | page-number] leader list (book TOC/index), not a data table.

    det47 (run-52 census): odlbench doc-016 emits a 29-row TOC as a
    positional 'table' — NID strips it as a table while GT keeps the
    entries as plain text, pure deletion damage. Measured exposure of the
    veto: zero — odlbench GT 42 table-docs carry 0 leader-shaped tables;
    olmOCR 918 table probes have 0 pure-int <=4-digit cell values.
    Decimal tables ('1,154.5') survive \\d{1,4} fullmatch by construction.
    """
    if len(rows) < 8:
        return False
    two_cell = [r for r in rows if len([c for c in r if str(c).strip()]) == 2]
    if len(two_cell) < 0.8 * len(rows):
        return False
    last = [str(r[-1]).strip() for r in two_cell]
    pure_int = [v for v in last if re.fullmatch(r"\d{1,4}", v)]
    if len(pure_int) < 0.8 * len(last):
        return False
    first_words = sorted(len(str(r[0]).split()) for r in two_cell)
    return first_words[len(first_words) // 2] >= 3


_EQUATION_KILL_GLYPHS_RE = re.compile(r"[∑∈∫∂∇∀∃∏⊙⊗⊕∪∩∖ΘΛΩ]")
_EQUATION_OP_BETWEEN_RE = re.compile(r"[A-Za-z\)\]]\s*[±≤≥<>=×÷−–]\s*[A-Za-z\(]")
_EQUATION_FN_RE = re.compile(r"\b[A-Za-z]\([A-Za-z0-9]")
_EQUATION_SUB_RE = re.compile(r"[A-Za-z]_\{?[0-9]")
_EQUATION_TRAIL_OP_RE = re.compile(r"[=<>≤≥≈±−]\s*$")


def looks_like_equation_rows(rows) -> bool:
    """Display/inline equations carved into a grid — not a data table.

    det49 (run-56 census): aligned equations emit as pipe tables, which
    strips the source line from page text so the inline Unicode->LaTeX
    lane never runs — GT LaTeX probes then find nothing (03759: 'H(Z) = |
    X z P(z) log P(z)' killed the entropy formula; up to 267 probes / 84
    PDFs exposed). Two tiers, either fires:
      dense  — >=2 non-empty cells, >=80% carry equation fragments:
               kill-glyphs ([sum in int forall exists Theta Lambda Omega
               ...]), letter-op-letter, fn-call 'P(z)', subscripts 'x_1',
               trailing operator 'H(Z) ='.
      moderate — >=4 cells, >=35% carry kill-glyphs or letter-op-letter.
    Measured on bench md corpora: 31/348 arxiv equation-tables killed,
    0/390 tables-cat, 2/278 multi_column (081875a1 mojibake scan tables —
    order probes unaffected in expectation, bench arbitrates). Stats
    ('0.83±0.13') and spec ('1830±50') digit-attached forms survive.
    """
    cells = [str(c).strip() for r in rows for c in (r or ()) if str(c).strip()]
    if len(cells) < 2:
        return False
    ext = sum(
        1
        for c in cells
        if _EQUATION_KILL_GLYPHS_RE.search(c)
        or _EQUATION_OP_BETWEEN_RE.search(c)
        or _EQUATION_FN_RE.search(c)
        or _EQUATION_SUB_RE.search(c)
        or _EQUATION_TRAIL_OP_RE.search(c)
    )
    if len(cells) >= 2 and ext >= 0.8 * len(cells):
        return True
    mod = sum(
        1 for c in cells if _EQUATION_KILL_GLYPHS_RE.search(c) or _EQUATION_OP_BETWEEN_RE.search(c)
    )
    return len(cells) >= 4 and mod >= 0.35 * len(cells)


def looks_like_prose_shatter(rows) -> bool:
    """Justified-prose fragments carved into a grid — not a data table.

    det47 (run-52 census: odlbench 033/034/036 img2table_lastresort
    FPs). True when no whole-cell numeric value, no code-like cell, and
    >=50% of rows start mid-sentence (leading cell ends ,;: or lowercase
    multi-word). Keeps the measured wins: 7e850300 ('10.9888' numeric
    cells), c6673ff6 ('A/RES/66/2' code-like cells).
    """
    cells = [str(c).strip() for r in rows for c in (r or ()) if str(c).strip()]
    if not cells:
        return False
    if any(re.fullmatch(r"[\d.,]+%?", c) for c in cells):
        return False
    # det48: digit-leading values with a short unit suffix ('24 ml',
    # '5 kg', '10 °C') are MEASURED QUANTITIES — odlbench 0117's true
    # table ('24 ml'/'0 ml'/'4 ml') was vetoed as prose by the bare
    # [\d.,]+ pattern. Digit-first + optional unit keeps word-cell prose
    # (continuation fragments never start with a digit) vetoed.
    if any(re.fullmatch(r"\d[\d.,]*\s*[A-Za-z°%]{0,4}\.?", c) for c in cells):
        return False
    if any(sum(ch.isdigit() for ch in c) >= 2 and ("/" in c or "-" in c) for c in cells):
        return False

    def _cont(r):
        l = str(r[0]).strip() if r else ""
        return bool(l) and (l[-1] in ",;:" or (l[-1].islower() and len(l.split()) > 1))

    return sum(1 for r in rows if _cont(r)) / max(1, len(rows)) >= 0.5


def native_find_tables(pdf_path, page_num, strategy="default"):
    """Memoized PyMuPDF find_tables → list of {"bbox", "rows"} dicts.

    Fail-soft: any error returns [] (callers treat as "no native tables",
    the same semantics each consumer had before).
    """
    key = None
    try:
        abs_path = os.path.abspath(pdf_path)
        key = (abs_path, os.stat(abs_path).st_mtime_ns, page_num, strategy)
        if key in _FIND_TABLES_CACHE:
            return _FIND_TABLES_CACHE[key]
    except OSError:
        key = None
    out = []
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            # "default" = caller wants PyMuPDF's own default strategy
            # ("lines") — the kwarg rejects that literal string.
            if strategy == "default":
                finder = page.find_tables()
            else:
                finder = page.find_tables(strategy=strategy)
            for t in finder.tables:
                extracted = t.extract()
                out.append(
                    {
                        "bbox": tuple(t.bbox),
                        "rows": [[(c or "") for c in r] for r in (extracted or [])],
                    }
                )
    except Exception as e:
        log.debug("native_find_tables failed p%s: %s", page_num, e)
    if key is not None:
        if len(_FIND_TABLES_CACHE) >= _FIND_TABLES_CACHE_MAX:
            _FIND_TABLES_CACHE.clear()
        _FIND_TABLES_CACHE[key] = out
    return out


def _load_consts():
    global _CONSTS_LOADED
    if _CONSTS_LOADED:
        return
    p = _pdf_ocr()
    g = globals()
    for n in (
        "_page_dict_cache",
        "page_has_embedded_fonts",
        "_get_page_dict",
        "DEPS",
        "_PRIVATE_CHAR_RE",
        "IS_IMAGE",
        "PDF_PATH",
        "_OPEN_TABLES_STATE",
        "_FIGURE_TABLE_OCR",
    ):
        if hasattr(p, n):
            g[n] = getattr(p, n)
    _CONSTS_LOADED = True


@dataclass
class Cell:
    """Structured table cell with full metadata.

    Markdown is merely one renderer — Cell carries enough data for
    HTML (rowspan/colspan), XLSX (merged cells), JSON (full export),
    and provenance tracking (source, confidence).
    """

    text: str = ""
    bbox: tuple = (0, 0, 0, 0)  # (x0, y0, x1, y1) in PDF points
    row: int = 0
    col: int = 0
    confidence: float = 1.0  # extraction quality (1.0=fitz, 0.5=OCR)
    font_size: float = 0.0  # median span size in cell
    font_flags: int = 0  # fitz span flags (bold/italic/etc.)
    alignment: str = "left"  # "left" / "center" / "right"
    rowspan: int = 1  # merged cell: spans N rows
    colspan: int = 1  # merged cell: spans N cols
    source: str = "fitz"  # "fitz" / "ocr" / "layout"


def _detect_table_regions_numeric(pd, tbl_bbox_hint=None):
    """Detect table regions using document-aware features (domain-independent).

    Instead of domain-specific regex (SiO2, wt%, ppm), uses structural signals:
    1. Numeric token density — fraction of tokens that are numbers
    2. Token spacing — wide bbox / few tokens = column-aligned
    3. Alignment consistency — similar token counts across consecutive lines
    4. Lexical repetition — low token diversity = tabular structure

    Works for oxide tables, fossil measurements, crystallography, isotope
    ratios, statistical tables, integer counts — any tabular data.

    Returns list of (x0, y0, x1, y1) bboxes for detected table regions.
    """
    if pd is None:
        return []

    # Collect lines with structural features
    data_lines = []
    for block in pd["dict"]["blocks"]:
        if block.get("type") != 0:
            continue
        for line_obj in block.get("lines", []):
            lbbox = line_obj["bbox"]
            text = "".join(s.get("text", "") for s in line_obj.get("spans", []))
            if not text.strip():
                continue

            tokens = text.split()
            if len(tokens) < 2:
                continue

            # Feature 1: numeric token density (domain-independent)
            numeric_tokens = sum(
                1 for t in tokens if re.match(r"^[\d.]+$", t) and re.search(r"\d", t)
            )
            numeric_ratio = numeric_tokens / len(tokens)

            # Feature 2: token spacing (wide bbox, few tokens = column-aligned)
            line_width = lbbox[2] - lbbox[0]
            avg_token_spacing = line_width / max(len(tokens), 1)

            # Feature 3: lexical diversity (low = repetitive = table-like)
            unique = len(set(t.lower() for t in tokens))
            diversity = unique / len(tokens)

            # Composite score: weighted structural features
            score = (
                0.35 * numeric_ratio  # numeric density
                + 0.20 * min(avg_token_spacing / 40.0, 1.0)  # spacing
                + 0.20 * (1.0 - diversity)  # repetition
                + 0.15 * min(len(tokens) / 8.0, 1.0)  # density
                + 0.10 * (1.0 if numeric_tokens >= 3 else 0.0)  # strong numeric
            )

            lcy = (lbbox[1] + lbbox[3]) / 2
            data_lines.append(
                {
                    "bbox": lbbox,
                    "cy": lcy,
                    "score": score,
                    "n_tokens": len(tokens),
                    "n_numeric": numeric_tokens,
                }
            )

    if len(data_lines) < 5:
        return []

    # Feature 4: alignment consistency — check if consecutive lines have
    # similar token counts (tabular structure repeats column pattern)
    data_lines.sort(key=lambda l: l["cy"])
    for i, dl in enumerate(data_lines):
        # Compare with neighbors
        neighbors = [
            data_lines[j] for j in range(max(0, i - 2), min(len(data_lines), i + 3)) if j != i
        ]
        if neighbors:
            avg_nt = sum(n["n_tokens"] for n in neighbors) / len(neighbors)
            consistency = 1.0 - min(abs(dl["n_tokens"] - avg_nt) / max(avg_nt, 1), 1.0)
            dl["score"] += 0.15 * consistency

    # Adaptive data-line threshold from score distribution
    _scores = sorted(l["score"] for l in data_lines)
    _data_thresh = max(_scores[len(_scores) // 3], 0.25) if len(_scores) > 5 else 0.35

    # Sliding window: find regions with ≥5 data lines scoring above threshold
    runs = []
    i = 0
    while i < len(data_lines):
        window = data_lines[i : i + 12]
        data_in_window = [l for l in window if l["score"] >= _data_thresh]
        if len(data_in_window) >= 5:
            run_lines = list(window)
            j = i + len(window)
            median_h = sorted(l["bbox"][3] - l["bbox"][1] for l in data_lines)[len(data_lines) // 2]
            while j < len(data_lines):
                gap = data_lines[j]["cy"] - run_lines[-1]["cy"]
                if gap > median_h * 3:
                    break
                run_lines.append(data_lines[j])
                j += 1
            runs.append(run_lines)
            i = j
        else:
            i += 1

    if not runs:
        return []

    regions = []
    for run in runs:
        x0 = min(l["bbox"][0] for l in run)
        y0 = min(l["bbox"][1] for l in run)
        x1 = max(l["bbox"][2] for l in run)
        y1 = max(l["bbox"][3] for l in run)
        y0 = max(0, y0 - 30)  # include header above data
        regions.append((x0 - 5, y0, x1 + 5, y1 + 5))

    # STACKED-REGION TRIM (iter-113, 904a1b4e pg141): the -30pt header
    # inclusion pokes a later region under the previous table's bottom
    # (region2 y0=276 < region1 y1=279 — table 1's LAST ROW entered
    # region 2 and its 6-col grid shifted the NUTS table's values two
    # columns right). A later region stacked under an earlier one (big
    # x-overlap) starts below the earlier bottom.
    regions.sort(key=lambda r: r[1])
    trimmed = []
    for r in regions:
        x0, y0, x1, y1 = r
        for p in trimmed:
            _ox = max(0, min(x1, p[2]) - max(x0, p[0]))
            if _ox > 0.5 * min(x1 - x0, p[2] - p[0]) and y0 < p[3]:
                y0 = p[3] + 1
        if y0 < y1:
            trimmed.append((x0, y0, x1, y1))
    return trimmed


def _optimize_grid_em(lines, row_bounds, col_bounds, max_iter=10):
    """EM optimization of grid boundaries to maximize span-cell alignment.

    Treats table reconstruction as optimization: find the grid (row + column
    boundaries) that maximizes total bbox overlap between text lines and cells.

    E-step: assign each line to cell with maximum overlap area.
    M-step: reposition each boundary to maximize total alignment.
    Converges to local optimum (initialized by DBSCAN clustering).

    No dependencies — pure Python coordinate optimization.
    """
    # Adaptive resolution from line heights
    _lhs = sorted(l["y1"] - l["y0"] for l in lines) if lines else [12]
    _median_lh = _lhs[len(_lhs) // 2]
    _em_res = max(_median_lh * 0.15, 1.0)  # search step
    _em_tol = max(_median_lh * 0.04, 0.3)  # convergence tolerance

    def _overlap_1d(a0, a1, b0, b1):
        return max(0, min(a1, b1) - max(a0, b0))

    for _iteration in range(max_iter):
        # E-step: assign lines to cells by max overlap
        assignments = []
        for l in lines:
            best_r, best_c, best_ov = 0, 0, 0
            for ri in range(len(row_bounds) - 1):
                ry0, ry1 = row_bounds[ri], row_bounds[ri + 1]
                dy = _overlap_1d(l["y0"], l["y1"], ry0, ry1)
                if dy == 0:
                    continue
                for ci in range(len(col_bounds) - 1):
                    cx0, cx1 = col_bounds[ci], col_bounds[ci + 1]
                    dx = _overlap_1d(l["x0"], l["x1"], cx0, cx1)
                    ov = dx * dy
                    if ov > best_ov:
                        best_ov = ov
                        best_r, best_c = ri, ci
            assignments.append((l, best_r, best_c))

        moved = False

        # M-step: optimize each column boundary
        for bi in range(1, len(col_bounds) - 1):
            left = [l for l, _, c in assignments if c == bi - 1]
            right = [l for l, _, c in assignments if c == bi]
            if not left and not right:
                continue
            lo = col_bounds[bi - 1] + 2
            hi = col_bounds[bi + 1] - 2
            best_x, best_s = col_bounds[bi], -1
            x = lo
            while x <= hi:
                s = sum(_overlap_1d(l["x0"], l["x1"], col_bounds[bi - 1], x) for l in left)
                s += sum(_overlap_1d(l["x0"], l["x1"], x, col_bounds[bi + 1]) for l in right)
                if s > best_s:
                    best_s, best_x = s, x
                x += _em_res
            if abs(best_x - col_bounds[bi]) > _em_tol:
                col_bounds[bi] = best_x
                moved = True

        # M-step: optimize each row boundary
        for bi in range(1, len(row_bounds) - 1):
            top = [l for l, r, _ in assignments if r == bi - 1]
            bot = [l for l, r, _ in assignments if r == bi]
            if not top and not bot:
                continue
            lo = row_bounds[bi - 1] + 2
            hi = row_bounds[bi + 1] - 2
            best_y, best_s = row_bounds[bi], -1
            y = lo
            while y <= hi:
                s = sum(_overlap_1d(l["y0"], l["y1"], row_bounds[bi - 1], y) for l in top)
                s += sum(_overlap_1d(l["y0"], l["y1"], y, row_bounds[bi + 1]) for l in bot)
                if s > best_s:
                    best_s, best_y = s, y
                y += _em_res
            if abs(best_y - row_bounds[bi]) > _em_tol:
                row_bounds[bi] = best_y
                moved = True

        if not moved:
            break

    return col_bounds, row_bounds


def _build_table_grid(pd, tbl_bbox, doc=None, page=None):
    """Build a table grid from span coordinates within a bbox.

    Returns (cells_2d, n_rows, n_cols) or None if grid construction fails.
    Uses alignment graph (primary) + DBSCAN (fallback) for column detection,
    EM optimization for boundary refinement, and Cell objects with confidence.
    """
    import bisect

    # Step 1: Get text lines within table region
    lines_in_table = []
    for block in pd["dict"]["blocks"]:
        if block.get("type") != 0:
            continue
        for line_obj in block.get("lines", []):
            lbbox = line_obj["bbox"]
            lcx = (lbbox[0] + lbbox[2]) / 2
            lcy = (lbbox[1] + lbbox[3]) / 2
            if (
                tbl_bbox[0] - 5 <= lcx <= tbl_bbox[2] + 5
                and tbl_bbox[1] - 5 <= lcy <= tbl_bbox[3] + 5
            ):
                spans = line_obj.get("spans", [])
                text = "".join(s.get("text", "") for s in spans)
                if text.strip():
                    sizes = [s.get("size", 10) for s in spans if s.get("text", "").strip()]
                    flags = [s.get("flags", 0) for s in spans if s.get("text", "").strip()]
                    lines_in_table.append(
                        {
                            "text": text.strip(),
                            "x0": lbbox[0],
                            "y0": lbbox[1],
                            "x1": lbbox[2],
                            "y1": lbbox[3],
                            "cx": lcx,
                            "cy": lcy,
                            "height": lbbox[3] - lbbox[1],
                            "spans": spans,
                            "sizes": sizes,
                            "flags": flags,
                        }
                    )

    if len(lines_in_table) < 4:
        return None

    # Step 2: Cluster lines into rows (adaptive eps, baseline=y0)
    sorted_lines = sorted(lines_in_table, key=lambda l: l["y0"])
    median_height = sorted(l["height"] for l in lines_in_table)[len(lines_in_table) // 2]
    row_eps = max(median_height * 0.5, 3.0)

    row_clusters = [[sorted_lines[0]]]
    for l in sorted_lines[1:]:
        if l["y0"] - row_clusters[-1][-1]["y0"] <= row_eps:
            row_clusters[-1].append(l)
        else:
            row_clusters.append([l])

    if len(row_clusters) < 3:
        return None

    row_boundaries = [tbl_bbox[1]]
    for i in range(1, len(row_clusters)):
        prev_y = row_clusters[i - 1][-1]["y0"]
        curr_y = row_clusters[i][0]["y0"]
        row_boundaries.append((prev_y + curr_y) / 2)
    row_boundaries.append(tbl_bbox[3])

    # Step 3: Column detection via alignment graph (primary) + DBSCAN (fallback)
    word_list = []
    for block in pd["dict"]["blocks"]:
        if block.get("type") != 0:
            continue
        for line_obj in block.get("lines", []):
            lbbox = line_obj["bbox"]
            lcy = (lbbox[1] + lbbox[3]) / 2
            if not (tbl_bbox[1] - 5 <= lcy <= tbl_bbox[3] + 5):
                continue
            raw_spans = [
                (s["bbox"][0], s["bbox"][2], s.get("text", ""), s.get("size", 10))
                for s in line_obj.get("spans", [])
                if s.get("text", "").strip()
            ]
            line_spans = sorted(raw_spans, key=lambda x: x[0])
            if not line_spans:
                continue
            span_sizes = [s[3] for s in line_spans]
            median_fs = sorted(span_sizes)[len(span_sizes) // 2] if span_sizes else 10
            merge_gap = max(3.0, median_fs * 0.25)
            words = [[line_spans[0][0], line_spans[0][1], line_spans[0][2]]]
            for x0, x1, txt, _sz in line_spans[1:]:
                if x0 - words[-1][1] < merge_gap:
                    words[-1][1] = max(words[-1][1], x1)
                    words[-1][2] += txt
                else:
                    words.append([x0, x1, txt])
            for w_x0, w_x1, w_txt in words:
                word_list.append(
                    {
                        "x0": w_x0,
                        "x1": w_x1,
                        "y0": lbbox[1],
                        "y1": lbbox[3],
                        "width": w_x1 - w_x0,
                        "text": w_txt,
                    }
                )

    if len(word_list) < 4:
        return None

    # Primary: Alignment graph
    word_list.sort(key=lambda w: w["y0"])
    line_groups = [[word_list[0]]]
    line_eps = max((word_list[0]["y1"] - word_list[0]["y0"]) * 0.5, 3.0)
    for w in word_list[1:]:
        if w["y0"] - line_groups[-1][-1]["y0"] <= line_eps:
            line_groups[-1].append(w)
        else:
            line_groups.append([w])

    col_bounds = None
    if len(line_groups) >= 3:
        n = len(word_list)
        parent = list(range(n))

        def _find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def _union(a, b):
            ra, rb = _find(a), _find(b)
            if ra != rb:
                parent[ra] = rb

        widx = {id(w): i for i, w in enumerate(word_list)}
        for li in range(len(line_groups)):
            for offset in (1, 2):
                if li + offset >= len(line_groups):
                    break
                for wa in line_groups[li]:
                    for wb in line_groups[li + offset]:
                        ov = min(wa["x1"], wb["x1"]) - max(wa["x0"], wb["x0"])
                        mw = min(wa["width"], wb["width"])
                        if mw > 0 and ov / mw > 0.3:
                            _union(widx[id(wa)], widx[id(wb)])
        comps = {}
        for i, w in enumerate(word_list):
            r = _find(i)
            comps.setdefault(r, []).append(w)
        columns = []
        for members in comps.values():
            if len(members) >= 2:
                columns.append(
                    (
                        min(m["x0"] for m in members),
                        max(m["x1"] for m in members),
                    )
                )
        if len(columns) >= 2:
            columns.sort(key=lambda c: c[0])
            col_bounds = [columns[0][0]]
            for i in range(1, len(columns)):
                col_bounds.append((columns[i - 1][1] + columns[i][0]) / 2)
            col_bounds.append(columns[-1][1])

    # Fallback: DBSCAN
    if col_bounds is None:
        word_cxs = sorted((w["x0"] + w["x1"]) / 2 for w in word_list)
        if len(word_cxs) >= 4:
            cx_gaps = [
                (word_cxs[i + 1] - word_cxs[i], i)
                for i in range(len(word_cxs) - 1)
                if word_cxs[i + 1] - word_cxs[i] > 0.5
            ]
            if cx_gaps:
                gaps_s = sorted(g for g, _ in cx_gaps)
                max_jump = 0
                knee = max(len(gaps_s) // 4, 1)
                for i in range(1, len(gaps_s)):
                    if gaps_s[i] - gaps_s[i - 1] > max_jump:
                        max_jump = gaps_s[i] - gaps_s[i - 1]
                        knee = i
                eps = max(gaps_s[min(knee, len(gaps_s) - 1)], 6)
                eps = min(eps, 25)
                clusters = [[word_cxs[0]]]
                for wc in word_cxs[1:]:
                    if wc - clusters[-1][-1] <= eps:
                        clusters[-1].append(wc)
                    else:
                        clusters.append([wc])
                if len(clusters) >= 2:
                    col_bounds = [tbl_bbox[0]]
                    for i in range(1, len(clusters)):
                        col_bounds.append((clusters[i - 1][-1] + clusters[i][0]) / 2)
                    col_bounds.append(tbl_bbox[2])

    if col_bounds is None or len(col_bounds) < 3:
        return None

    # EM optimization
    col_bounds, row_boundaries = _optimize_grid_em(
        lines_in_table, list(row_boundaries), list(col_bounds), max_iter=10
    )

    # Build grid
    n_rows = len(row_boundaries) - 1
    n_cols = len(col_bounds) - 1
    grid = {}
    for l in lines_in_table:
        ri = max(0, min(bisect.bisect_right(row_boundaries, l["cy"]) - 1, n_rows - 1))
        best_col, best_ov = 0, 0
        for ci in range(n_cols):
            ov = max(0, min(l["x1"], col_bounds[ci + 1]) - max(l["x0"], col_bounds[ci]))
            if ov > best_ov:
                best_ov, best_col = ov, ci
        key = (ri, best_col)
        if key not in grid:
            grid[key] = []
        grid[key].append(l)

    filled = len(grid)
    total = n_rows * n_cols
    if total > 0 and filled / total < 0.05:
        return None

    # Step 4: Build Cell objects with confidence propagation
    cells = []
    for r in range(n_rows):
        row_cells = []
        ry0 = row_boundaries[r] if r < len(row_boundaries) else 0
        ry1 = row_boundaries[r + 1] if r + 1 < len(row_boundaries) else 0
        for c_idx in range(n_cols):
            cx0 = col_bounds[c_idx] if c_idx < len(col_bounds) else 0
            cx1 = col_bounds[c_idx + 1] if c_idx + 1 < len(col_bounds) else 0
            cell_data = grid.get((r, c_idx), [])
            if not cell_data:
                row_cells.append(
                    Cell(
                        text="",
                        bbox=(cx0, ry0, cx1, ry1),
                        row=r,
                        col=c_idx,
                        confidence=0.3,
                    )
                )
                continue
            # Merge line texts
            cell_text = " ".join(d["text"] for d in cell_data)
            cell_text = " ".join(cell_text.split())
            # Compute confidence from text quality
            conf = 1.0
            if "\ufffd" in cell_text:
                conf -= 0.3
            # Compute font metadata
            all_sizes = [s for d in cell_data for s in d.get("sizes", [])]
            all_flags = [f for d in cell_data for f in d.get("flags", [])]
            median_fs = sorted(all_sizes)[len(all_sizes) // 2] if all_sizes else 0
            median_flags = max(all_flags) if all_flags else 0
            # Alignment from bbox
            cell_w = cx1 - cx0
            alignment = "center" if (cell_w > 0 and len(cell_text) * 4 < cell_w * 0.4) else "left"
            row_cells.append(
                Cell(
                    text=cell_text,
                    bbox=(cx0, ry0, cx1, ry1),
                    row=r,
                    col=c_idx,
                    confidence=conf,
                    source="fitz",
                    font_size=median_fs,
                    font_flags=median_flags,
                    alignment=alignment,
                )
            )
        cells.append(row_cells)

    # LABEL-WRAP SPLIT (iter-113, 904a1b4e pg141): a wrapped row label's
    # continuation word lands in the first value column and glues to its
    # number ('anos 31,0') because line-level column assignment puts the
    # short continuation line in the value column. Signature: a data row
    # whose cell i (first non-empty after the label) reads
    # '<lowercase-word> <number>' while every later non-empty cell in the
    # row is a pure number — move the word back onto the label.
    _num = re.compile(r"^[-+\d.,%€$‰]+$")
    for row_cells in cells:
        ne = [(i, c) for i, c in enumerate(row_cells) if c.text.strip()]
        if len(ne) < 3:
            continue
        li, label = ne[0]
        vi, val = ne[1]
        m = re.match(r"^([a-zà-ÿ]{2,})\s+([-+\d.,%€$‰]+)$", val.text.strip())
        if (
            m
            and vi > li
            and all(_num.match(c.text.strip()) for _j, c in ne[2:])
            and not label.text.rstrip().endswith((".", ":"))
        ):
            label.text = f"{label.text.rstrip()} {m.group(1)}"
            val.text = m.group(2)

    return cells, n_rows, n_cols


def detect_multirow_header(rows, max_header_rows=3):
    """Detect multi-row table headers and merge into single markdown header.

    Scientific paper tables often have 2-3 header rows:
      Row 0: Group labels (e.g., "Major elements" spanning multiple cols)
      Row 1: Per-column labels (e.g., "SiO2", "TiO2", "Al2O3")
      Row 2: Units (e.g., "wt%", "ppm", "mol%")

    Markdown tables only support single-row headers. This function:
      1. Scans first max_header_rows for header pattern (mostly non-numeric,
         short cells, often repeated for group labels).
      2. Combines multi-row headers with " / " separator:
         "Major elements / SiO2 / wt%" — readable, preserves structure.
      3. Returns (combined_header, body_rows_start_index).

    A row is a "header candidate" if:
      - ≥50% cells are non-numeric (no decimals)
      - Average cell length ≤10 chars (short labels)
      - Row contains alphabetic chars (not pure numbers)
    """
    if not rows or len(rows) < 2:
        return rows[0] if rows else [], 1
    header_rows = []
    body_start = 1

    # A measurement cell (number ± uncertainty, or a multi-digit value) is
    # DATA, never a header label. Rows containing one are body rows —
    # geochem tables ("1 | O | 5653±154") were falsely merged as headers.
    # No end-anchor: cells carry trailing OCR junk ("5653±154\n. .").
    _MEASUREMENT_RE = re.compile(
        r"^\d[\d.,]*\s*(?:±|\\pm|∓|\+/-)\s*\d[\d.,]*"  # 5653±154
        r"|^\d{3,}[\d.,]*"  # 3+ leading digits = data value
    )

    def _is_header_row(row):
        non_numeric = sum(1 for c in row if c and not _is_numeric_cell(c))
        non_numeric_ratio = non_numeric / max(len([c for c in row if c]), 1)
        avg_len = sum(len(c) for c in row if c) / max(len([c for c in row if c]), 1)
        has_alpha = any(any(ch.isalpha() for ch in c) for c in row if c)
        has_measurement = any(c and _MEASUREMENT_RE.match(str(c).strip()) for c in row)
        return non_numeric_ratio >= 0.5 and avg_len <= 12 and has_alpha and not has_measurement

    for i, row in enumerate(rows[: max_header_rows + 1]):
        if i == 0:
            header_rows.append(row)
            continue
        # Check if this row looks like another header row
        if _is_header_row(row):
            header_rows.append(row)
            body_start = i + 1
            continue
        # (Subscript/unit row merging happens at the GEOMETRY-AWARE site in
        # detect_tables_pdfplumber — x-alignment is required to know which
        # column a '1' belongs to; the " / " join here would corrupt header
        # cells and break exact-match gold tests.)
        break
    if len(header_rows) == 1:
        return header_rows[0], body_start
    # Combine multi-row headers cell-by-cell
    n_cols = max(len(r) for r in header_rows)
    combined = []
    for ci in range(n_cols):
        parts = []
        for hr in header_rows:
            if ci < len(hr) and hr[ci]:
                # Skip duplicates (group label repeats are noise)
                if hr[ci] not in parts:
                    parts.append(hr[ci])
        combined.append(" / ".join(parts))
    return combined, body_start


def infer_column_types(rows):
    """Infer the type of each column from cell content distribution.

    Returns list of types (one per column): 'number', 'percent', 'date',
    'identifier', 'text', 'mixed'.

    A column is 'number' if ≥70% of non-empty cells are numeric.
    'mixed' if no clear type dominates (<50% any single type).
    """
    if not rows:
        return []
    n_cols = max(len(r) for r in rows)
    col_types = []
    for ci in range(n_cols):
        cell_types = []
        for r in rows:
            if ci < len(r):
                t = classify_cell(r[ci])
                if t != "empty":
                    cell_types.append(t)
        if not cell_types:
            col_types.append("text")
            continue
        # Find dominant type
        type_counts = {}
        for t in cell_types:
            type_counts[t] = type_counts.get(t, 0) + 1
        dominant = max(type_counts.items(), key=lambda x: x[1])
        if dominant[1] / len(cell_types) >= 0.70:
            # Group number + scientific + percent as "numeric" for validation
            if dominant[0] in ("number", "scientific", "percent"):
                col_types.append("numeric")
            else:
                col_types.append(dominant[0])
        else:
            # Check if all are numeric-family
            numeric_count = sum(type_counts.get(t, 0) for t in ("number", "scientific", "percent"))
            if numeric_count / len(cell_types) >= 0.60:
                col_types.append("numeric")
            else:
                col_types.append("mixed")
    return col_types


def build_table_html(rows, max_header_rows=3):
    rows = _scrub_grid_rows(rows)
    """Build an HTML table preserving MULTI-ROW header structure.

    The olmOCR table checker parses HTML tables with real <thead>/<th>
    semantics: every header row is registered (markdown parsing marks only
    row 0), so top_heading tests against rows 1-2 of a 3-row header fail on
    markdown but pass on HTML. Emitted alongside the markdown pipe table.
    """
    from html import escape as _html_escape

    if not rows or len(rows) < 2:
        return ""
    # Reuse the same header/body split as build_table_markdown.
    header, body_start = detect_multirow_header(rows, max_header_rows)
    header_rows = rows[:body_start]
    body_rows = rows[body_start:] or [rows[-1]]
    n_cols = max(len(r) for r in rows)

    def _cell(c, tag):
        c = str(c) if c is not None else ""
        return f"<{tag}>{_html_escape(c.strip())}</{tag}>"

    out = ["<table>", "<thead>"]
    for hr in header_rows:
        out.append(
            "<tr>" + "".join(_cell(c, "th") for c in (list(hr) + [""] * n_cols)[:n_cols]) + "</tr>"
        )
    out.append("</thead>")
    out.append("<tbody>")
    for br in body_rows:
        out.append(
            "<tr>" + "".join(_cell(c, "td") for c in (list(br) + [""] * n_cols)[:n_cols]) + "</tr>"
        )
    out.append("</tbody>")
    out.append("</table>")
    return "\n".join(out)


def _defragment_row(cells, col_solid=None):
    """Merge word-fragment cells produced by geometric table slicing.

    Cell-splitting strategies (whitespace cuts, RapidTable word boxes,
    camelot lattice) occasionally break a word across a cell boundary:
    'Shallo|w', 'D|o|c' (measured 2026-08-19 multicol near-miss census:
    165 fails dominated by this family). Rule: a trailing cell of 1-2
    LOWERCASE letters whose left neighbor ends in a letter is a fragment
    — append it to the neighbor. Numeric cells ('1', '5') are legitimate
    data values and never merge. ``col_solid`` (from normalize_grid_rows)
    marks columns that carry SOLID values (>=3 chars or numeric) in any
    row — a lone letter in such a column is a real single-letter column
    ('n', 'm' stat names, 83f40 pg26), never a fragment."""
    if not cells:
        return list(cells)
    out = [str(c) for c in cells]
    i = len(out) - 1
    while i > 0:
        frag = out[i].strip()
        left = out[i - 1].rstrip()
        if (
            1 <= len(frag) <= 2
            and frag.isalpha()
            and frag.islower()
            and left
            and left[-1].isalpha()
            and not (col_solid and i < len(col_solid) and col_solid[i])
        ):
            out[i - 1] = left + frag
            out.pop(i)
        i -= 1
    return out


def _flat_cell(c):
    """Collapse a cell to single-line text (whitespace → single spaces).

    EMISSION CONTRACT (det44, 2026-10-02): pipe tables ship WITHOUT the
    `| --- |` separator row and WITHOUT in-cell newlines. Multi-line
    camelot/pdfplumber cells otherwise become phantom continuation rows
    ('| | | | value |'), and the separator makes downstream markdown→HTML
    converters turn our pipes into a SECOND table alongside the html twin
    (duplicate-table deletions in TEDS forest scoring). Sepless pipes pass
    through as plain text on those converters, leaving the twin as the
    only parsed table. olmOCR's parser (max-over-matches, runs split on
    non-pipe lines) reads sepless tables identically — validated 718/717.
    """
    return " ".join(str(c).split())


def normalize_grid_rows(rows):
    """Shared grid-row normalization (iter-113): defragment word
    fragments, split label-wrap glue, compact body-empty columns."""
    rows = _scrub_grid_rows(rows)
    # Column support (iter-115): a column carrying a SOLID value (>=3
    # chars or numeric) in any row is a real column — lone letters there
    # are single-letter columns ('n','m'), not shred fragments.
    _n = max(len(r) for r in rows) if rows else 0
    _solid = [False] * _n
    for r in rows:
        for ci in range(min(len(r), _n)):
            v = str(r[ci]).strip()
            if len(v) >= 3 or re.fullmatch(r"[-+.\d]+", v):
                _solid[ci] = True
    rows = [_defragment_row(r, _solid) for r in rows]
    # LABEL-WRAP SPLIT (iter-113, 904a1b4e pg141): a wrapped row label's
    # continuation word lands in the first value column glued to its
    # number ('anos 31,0') — line-level column bucketing puts the short
    # continuation line with the value. Signature: data row whose first
    # non-empty cell after the label reads '<lowercase-word> <number>'
    # while every later non-empty cell is a pure number — move the word
    # back onto the label.
    _lw_num = re.compile(r"^[-+\d.,%€$‰]+$")
    _lw_pat = re.compile(r"^([a-zà-ÿ]{2,})\s+([-+\d.,%€$‰]+)$")
    _rows2 = []
    for row in rows:
        r = [str(c) for c in row]
        ne = [(i, c) for i, c in enumerate(r) if c.strip()]
        if len(ne) >= 3:
            li, label = ne[0]
            vi, val = ne[1]
            m = _lw_pat.match(val.strip())
            if (
                m
                and vi > li
                and all(_lw_num.match(c.strip()) for _j, c in ne[2:])
                and not label.rstrip().endswith((".", ":"))
            ):
                r[li] = f"{label.rstrip()} {m.group(1)}"
                r[vi] = m.group(2)
        _rows2.append(r)
    rows = _rows2
    # COLUMN COMPACTION (iter-113, 904a1b4e pg141): a strategy's grid may
    # carry columns that are EMPTY in every body row (lines_strict carves
    # 6 cols for the NUTS table; values sit at cols 3-5 with phantom 1-2
    # between label and values). The scorer's top-heading / up / down
    # lookups scan the cell's OWN column — phantoms starve them. Drop
    # body-empty columns; header texts they carried re-anchor onto the
    # first surviving column to their right (a spanning band header
    # belongs to the values it heads).
    _hdr, _bstart = detect_multirow_header(rows)
    if _bstart >= len(rows):
        _hdr, _bstart = rows[0], 1
    _body = rows[_bstart:]
    if _body:
        _ncols = max(len(r) for r in rows)
        _col_empty = [
            not any((str(r[c]).strip() if c < len(r) else "") for r in _body) for c in range(_ncols)
        ]
        if any(_col_empty) and not all(_col_empty):
            _keep = [c for c in range(_ncols) if not _col_empty[c]]
            _first_kept_right = {}
            for c in range(_ncols):
                if _col_empty[c]:
                    _kr = next((k for k in _keep if k > c), None)
                    _first_kept_right[c] = _kr if _kr is not None else _keep[-1]

            def _compact(r):
                out = {}
                extras = {}
                for c in range(_ncols):
                    v = str(r[c]).strip() if c < len(r) else ""
                    if _col_empty[c]:
                        if v:
                            extras.setdefault(_first_kept_right[c], []).append(v)
                    else:
                        out[c] = v
                res = []
                for k in _keep:
                    txt = out.get(k, "")
                    if k in extras:
                        _x = " ".join(extras[k])
                        txt = f"{_x} {txt}".strip() if txt else _x
                    res.append(txt)
                return res

            rows = [_compact(r) for r in rows]
    # BAND-ROW MERGE (iter-113, 904a1b4e pg141): a spanning band header
    # row ('% da | população | activa') is one phrase sliced across
    # columns by the grid carve — the scorer's down/up lookups meet the
    # FRAGMENT ('activa') instead of the phrase. A body row whose every
    # non-empty cell is non-numeric, short (<=3 words), >=2 non-empty
    # cells, total <=6 words is a band: join into ONE cell at the median
    # fragment column (the value column it heads).
    _bw_num = re.compile(r"^[-+\d.,%€$‰]+$")
    _merged_rows = []
    for _ri2, r in enumerate(rows):
        r = [str(c) for c in r]
        _ne = [(i, c) for i, c in enumerate(r) if c.strip()]
        _is_band = (
            _ri2 > 0
            and len(_ne) >= 2
            and all(not _bw_num.match(c.strip()) for _i, c in _ne)
            and all(len(c.split()) <= 3 for _i, c in _ne)
            and sum(len(c.split()) for _i, c in _ne) <= 6
        )
        if _is_band:
            # NUMERIC-BELOW EVIDENCE (iter-113 fix-up, 8ec7ac2f pg1): the
            # band signature also matches every DATA row of a 2-column
            # label/org roster ('Natalie Chapman | Freight Transport
            # Association', iter-70 pin). A true band HEADS numeric values
            # in the columns it spans; a text-table row has none below.
            _spanned = [_i for _i, _c in _ne]
            _is_band = any(
                _bw_num.match(str(rr[cc]).strip())
                for rr in rows[_ri2 + 1 :]
                for cc in _spanned
                if cc < len(rr)
            )
        if _is_band:
            # LAST fragment column: the band heads the values to its
            # right ('% da população activa' ends at the column its
            # 2001-value sits above — 55,0's down-lookup, 904a1b4e).
            _mid = _ne[-1][0]
            _joined = " ".join(c.strip() for _i, c in _ne)
            r = ["" if _i != _mid else _joined for _i in range(len(r))]
        _merged_rows.append(r)
    rows = _merged_rows
    return rows


def _cell_tokens(value):
    """Normalized token list for a table cell (whitespace-split, casefolded)."""
    return [t for t in re.split(r"\s+", str(value or "").strip().casefold()) if t]


def _straddle_split_pass(words, rows, min_rows=3, consensus=0.6):
    """Single split pass (EXP-3): see _straddle_split_table for contract."""
    if len(rows) < min_rows or not words:
        return rows
    n_cols = max(len(r) for r in rows)
    if n_cols < 2:
        return rows

    # 1) Row-major word walk: match cell tokens to words in reading order,
    # collecting geometry per (row, col). Unmatched tokens are SKIPPED
    # (cell text may be normalized differently than raw words — one bad
    # token must not kill the row's boundary votes); cursor restarts per
    # row because text-equality matching is position-agnostic per column.
    word_x = {}  # (row_idx, col_idx) -> list[(x0, x1, text)]
    for ri, row in enumerate(rows):
        cells = list(row) + [""] * (n_cols - len(row))
        wi = 0
        for ci in range(n_cols):
            for tok in _cell_tokens(cells[ci]):
                for j in range(wi, len(words)):
                    if words[j][4].casefold().strip(".,:;()") == tok:
                        wx0, wy0, wx1, wy1, _txt = words[j][:5]
                        word_x.setdefault((ri, ci), []).append((wx0, wx1, words[j][4]))
                        wi = j + 1
                        break

    # 2) Consensus boundaries between adjacent cols (EDGE-based, 033c1efd
    # geometry: '9' x1=486.7, 'M-PSI/06' x0=492.0 -> boundary 489 == the
    # ruled v-line): each voting row gives the midpoint of (left max-x1,
    # right min-x0). Denominator = rows with words on EITHER side of the
    # pair (VOTING-ELIGIBLE) — real tables carry header/section/empty
    # rows that never vote and must not veto a strong boundary
    # (033c1efd: 16 clean voters / 39 raw rows). Need >=60% of eligible
    # (min 3 votes) and a coherent cluster (IQR <= 6pt).
    boundaries = {}
    for ci in range(n_cols - 1):
        votes = []
        eligible = 0
        for ri in range(len(rows)):
            left = word_x.get((ri, ci))
            right = word_x.get((ri, ci + 1))
            if left and right:
                votes.append((max(x1 for x0, x1, _ in left) + min(x0 for x0, x1, _ in right)) / 2.0)
                eligible += 1
            elif left or right:
                eligible += 1
        if len(votes) >= max(3, consensus * eligible):
            votes.sort()
            q1 = votes[len(votes) // 4]
            q3 = votes[(3 * len(votes)) // 4]
            if q3 - q1 <= 6.0:
                boundaries[ci] = votes[len(votes) // 2]

    if not boundaries:
        return rows

    # 3) Split jammed cells per boundary, DIRECTIONAL (033c1efd pg5:
    # '9 M-PSI/06' both in the LAST col with the empty neighbor LEFT —
    # ok sibling rows carve '9' | 'SECS-P/10'): (a) right-move: words in
    # cell ci at/after boundary with cell ci+1 EMPTY; (b) left-move:
    # words in cell ci+1 before boundary with cell ci EMPTY. Moves only
    # into EMPTY cells — no overwrite, no column-count change.
    changed = False
    new_rows = [list(r) + [""] * (n_cols - len(r)) for r in rows]
    for ri in range(len(rows)):
        for ci in range(n_cols - 1):
            if ci not in boundaries:
                continue
            b = boundaries[ci]
            left_cell_w = word_x.get((ri, ci)) or []
            right_cell_w = word_x.get((ri, ci + 1)) or []
            # (a) right-move out of ci into empty ci+1 (edge test: word
            # entirely at/after boundary moves; a word SPANNING the
            # boundary is a genuine wide word and stays)
            if len(left_cell_w) >= 2 and not str(new_rows[ri][ci + 1]).strip():
                stay = [(x0, x1, t) for x0, x1, t in left_cell_w if x1 <= b]
                move = [(x0, x1, t) for x0, x1, t in left_cell_w if x0 >= b]
                if stay and move and len(stay) + len(move) == len(left_cell_w):
                    new_rows[ri][ci] = " ".join(t for _, _, t in sorted(stay, key=lambda p: p[0]))
                    new_rows[ri][ci + 1] = " ".join(
                        t for _, _, t in sorted(move, key=lambda p: p[0])
                    )
                    changed = True
            # (b) left-move out of ci+1 into empty ci
            elif len(right_cell_w) >= 2 and not str(new_rows[ri][ci]).strip():
                stay = [(x0, x1, t) for x0, x1, t in right_cell_w if x0 >= b]
                move = [(x0, x1, t) for x0, x1, t in right_cell_w if x1 <= b]
                if stay and move and len(stay) + len(move) == len(right_cell_w):
                    new_rows[ri][ci + 1] = " ".join(
                        t for _, _, t in sorted(stay, key=lambda p: p[0])
                    )
                    new_rows[ri][ci] = " ".join(t for _, _, t in sorted(move, key=lambda p: p[0]))
                    changed = True
    return new_rows if changed else rows


def _straddle_split_table(words, rows, min_rows=3, consensus=0.6, max_passes=3):
    """Split jammed cells via column-consensus word geometry (EXP-3).

    A cell whose >=2 words straddle a consensus inter-column boundary is a
    jam ('9 M-PSI/06' in one cell, 033c1efd pg5 class): sibling rows vote
    the boundary (midpoint of left-word max-x vs right-word min-x), and
    the right-side words move over — only when the receiving neighbor cell
    is EMPTY (no overwrite, no column-count change). Fail-soft.

    ITERATES to fixpoint (max_passes): one move fills one empty cell, so a
    multi-column jam ('No Element Weight %' with 2 empty neighbors) needs
    a pass per boundary — a single-pass split left 'Element Weight %'
    jammed and BLOCKED the emission-time header splitter that would have
    carved it (b8d3ad0f pg4 regression). Moves only fill EMPTY cells, so
    each pass strictly fills more cells: monotone, terminates.
    """
    for _ in range(max_passes):
        next_rows = _straddle_split_pass(words, rows, min_rows, consensus)
        if next_rows is rows:
            break
        rows = next_rows
    return rows


def split_jammed_cells(pdf_path, page_num, tables, log_tag="cell_xclip"):
    """EXP-3 generic post-pass: geometry-split jammed cells in final tables.

    Strategy-agnostic (positional/camelot/lines/...): uses fitz word
    geometry + sibling-row column consensus; only fills EMPTY neighbor
    cells. Mutates table dicts' 'rows' in place. Fail-soft per table.
    """
    if not tables:
        return tables
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            if page_num < 1 or page_num > doc.page_count:
                return tables
            words = doc[page_num - 1].get_text("words")
    except Exception as e:
        log.debug("%s: word fetch failed p%s: %s", log_tag, page_num, e)
        return tables
    if not words:
        return tables
    words = sorted(words, key=lambda w: (round(w[1], 1), w[0]))

    n_split = 0
    for t in tables:
        if not isinstance(t, dict):
            continue
        rows = t.get("rows")
        bbox = t.get("bbox")
        if not rows or not bbox:
            continue
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except Exception:
            continue
        inside = [
            w
            for w in words
            if x0 - 2 <= (w[0] + w[1]) / 2 <= x1 + 2 and y0 - 2 <= (w[3] + w[1]) / 2 <= y1 + 2
        ]
        if len(inside) < 4:
            continue
        new_rows = _straddle_split_table(inside, rows)
        if new_rows is not rows:
            t["rows"] = new_rows
            n_split += 1
    if n_split:
        log.info("%s: split %d jammed table(s) on page %s", log_tag, n_split, page_num)
    return tables


def split_header_on_word_gaps(header, header_words):
    """iter-119a: split fused header cells at true column boundaries.

    The header LINE's own word boxes are the column oracle — header
    words are typeset one per column ('Q1' x=158, 'Q2' x=201 …). A
    header cell whose tokens exist as separate page words with
    inter-word gaps >= 1.5x word-space width splits into one cell per
    token; phrases with normal spacing stay whole (validated on
    1a67d1114 pg1: fused 'Q1 Q2 Q3' vs fitz 7-column truth).
    """
    if not header_words:
        return header
    by_text = {}
    for wx0, wx1, wt in header_words:
        by_text.setdefault(wt, (wx0, wx1))
    new_header = []
    changed = False
    for cell in header:
        toks = cell.split() if cell else []
        if len(toks) < 2 or any(t not in by_text for t in toks):
            new_header.append(cell)
            continue
        exts = [by_text[t] for t in toks]
        gaps = [exts[i + 1][0] - exts[i][1] for i in range(len(exts) - 1)]
        space_w = max(2.0, 0.25 * (exts[0][1] - exts[0][0]))
        if any(g >= 1.5 * space_w for g in gaps):
            new_header.extend(toks)
            changed = True
        else:
            new_header.append(cell)
    if not changed:
        return header
    width = len(new_header)
    return new_header


_CAPTION_RE = re.compile(r"^(?:tab(?:le|\.)?|fig(?:ure|\.)?)\s*\d", re.I)
_MEAS_RE = re.compile(r"^\d{3,}|±")
# prose function words — a header band carries labels/units, never these
_STOPWORDS = {
    "the",
    "a",
    "an",
    "this",
    "these",
    "those",
    "for",
    "in",
    "on",
    "of",
    "to",
    "and",
    "with",
    "from",
    "by",
    "at",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "during",
    "following",
    "we",
    "our",
    "above",
}


def _first_row_is_datalike(rows):
    """Inverse of detect_multirow_header's header predicate: the grid's
    first row carries numeric-majority or measurement content — the true
    header band lives ABOVE the detected bbox and never entered the grid
    (P2.1 census: 39 top_heading relation fails with the header text
    sitting in the prose band above the table). Bare 3+-digit runs are
    deliberately NOT measurement evidence here (1a67d1114: '2016/17'
    year tokens demoted the real Q1..Q4 header to body and the width
    truncation destroyed the Q3|Q4 adjacency the gold asserts)."""
    row = [str(c).strip() for c in (rows[0] or []) if str(c).strip()]
    if len(row) < 2:
        return False
    nonnum = sum(1 for c in row if not _is_numeric_cell(c))
    has_uncertainty = any(re.search(r"±|\\pm|∓|\+/-", c) for c in row)
    return nonnum <= len(row) / 2 or has_uncertainty


def recover_header_band(rows, bbox, page_words, max_lines=2, band_pt=60.0):
    """P2.1: pull the header band above the table bbox into the grid.

    When every strategy's rows start at DATA (first row numeric-like), the
    typeset header line(s) above the bbox never entered the grid and every
    column-anchored top_heading probe fails. Group the page words strictly
    above the bbox top into visual lines, gate each line (caption guard,
    alpha guard, prose guard, band-width guard), assign words to grid
    columns by x-center, and prepend as the header row(s) — the existing
    detect_multirow_header + split_header_on_word_gaps chain then treats
    them as row 0. Conservative by design: a wrongly recovered header
    pollutes the scorer's best_match fallback, so every gate rejects.

    Returns (rows, recovered_words) — rows unchanged (same list) when no
    line qualifies; recovered_words are the prepended lines' word tuples
    for the fused-header split at the build site.
    """
    if not rows or len(rows) < 2 or not bbox or not page_words:
        return rows, None
    widths = [len(r) for r in rows if r]
    if len(set(widths)) != 1:  # ragged grid — column anchors unreliable
        return rows, None
    n_cols = widths[0]
    if n_cols < 2 or not _first_row_is_datalike(rows):
        return rows, None
    bx0, by0, bx1, _by1 = bbox
    bw = max(1.0, bx1 - bx0)
    above = [
        w
        for w in page_words
        if w[3] <= by0 + 2
        and w[1] >= by0 - band_pt
        and w[0] < bx1 + 6
        and w[2] > bx0 - 6
        and str(w[4]).strip()
    ]
    if not above:
        return rows, None
    above.sort(key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines, cur, cur_y = [], [], None
    for w in above:
        yc = (w[1] + w[3]) / 2
        if cur_y is None or abs(yc - cur_y) <= 3.0:
            cur.append(w)
            cur_y = yc if cur_y is None else cur_y
        else:
            lines.append(cur)
            cur, cur_y = [w], yc
    if cur:
        lines.append(cur)
    header_rows, recovered_words = [], []
    # token-multiset of the grid's top rows — a band whose text already
    # lives in the grid is a re-carve duplicate, not an absent header
    # (1a67d1114: the year band sits above the bbox AND in row 0;
    # prepending it doubled the header and shattered the columns)
    _top_tokens = {}
    for r in rows[:3]:
        for c in r:
            for tok in str(c).split():
                _top_tokens[tok] = _top_tokens.get(tok, 0) + 1
    for ln in lines[-max_lines:]:
        text = " ".join(str(w[4]) for w in ln)
        toks = [str(w[4]) for w in ln]
        if not any(ch.isalpha() for ch in text):
            continue  # numeric run — a neighbouring table's data line
        if _CAPTION_RE.match(text.strip()):
            continue
        lx0, lx1 = ln[0][0], ln[-1][2]
        if (lx1 - lx0) <= 0.5 * bw:
            continue  # narrow fragment (page furniture), not a band
        if len(toks) > 2.5 * n_cols:
            continue  # prose sentence
        if sum(1 for t in toks if t.lower().strip(".,;:") in _STOPWORDS) >= 2:
            continue  # prose function words — not a header band
        if text.strip().endswith(".") and len(toks) > 6:
            continue
        _have = sum(1 for t in toks if _top_tokens.get(t, 0) > 0)
        if _have >= 0.6 * len(toks):
            continue  # already inside the grid — duplicate, not absent
        cells = [""] * n_cols
        for w in ln:
            xc = (w[0] + w[2]) / 2
            j = int((xc - bx0) / bw * n_cols)
            j = min(max(j, 0), n_cols - 1)
            cells[j] = (cells[j] + " " + str(w[4])).strip()
        header_rows.append(cells)
        recovered_words.append([(w[0], w[2], str(w[4])) for w in ln])
    if not header_rows:
        return rows, None
    return header_rows + list(rows), recovered_words


_FUSED_CELL_RE = re.compile(r"^[\d.,$%()+×\-]+(?:\s+[\d.,$%()+×\-]+){1,4}$")


def _split_fused_numeric_rows(rows):
    """det52: split fused numeric cells that pad rows with trailing empties.

    Measured class (a769f9a3 pg2): dual-panel financial rows come back with
    several values glued into one cell ('2.44 2.12 20%') plus trailing empty
    cells, while sibling rows carry the same values as separate columns
    ('| Q3 | 0.84 | 0.74 | 22% | 14% |'). Split a numeric-token run when the
    row carries trailing empties; accept the row only if the expansion lands
    exactly on the grid width. Non-numeric cells are never split; rows
    without trailing empties never change.
    """
    if not rows:
        return rows
    ncols = max(len(r) for r in rows)
    if ncols < 3 or len(rows) < 2:
        return rows
    split_rows = []
    any_split = False
    for r in rows:
        new = []
        split = False
        for c in r:
            s = str(c).strip()
            if _FUSED_CELL_RE.match(s):
                toks = s.split()
                if 1 < len(toks) <= 4:
                    new.extend(toks)
                    split = True
                    continue
            new.append(c)
        if split:
            any_split = True
        split_rows.append(new)
    if not any_split:
        return rows
    width = len(split_rows[0])
    if width < 4 or any(len(r) != width for r in split_rows):
        return rows
    return split_rows


def _expand_stacked_rows(rows):
    """det53: expand camelot column-packed rows into real data rows.

    Measured class (fa18a15c pg2): camelot_lattice returns the table body
    with each data cell holding a whole COLUMN as newline-stacked lines
    ('68 \n67 \n61 ...' = nine temperatures in one cell). Value cells
    may carry a leading units line ('(degrees F) \n1 \n1 ...') and label
    cells may wrap ('Central' + '(5)'). det44 flattening preserved the
    collapse as one wide row. Canonicalize per cell — strip leading
    bracketed-unit lines, merge wrapped label continuations
    (bare '(5)' joins the previous line) — then expand bottom-aligned
    into one row per line. Header rows with 2-3-line wrapped labels never
    qualify (k >= 4 required). Run-60 sweep: fires on fa18a15c only;
    c8c85869/7d3aedd0 packed rows arrive flattened, not stacked, and are
    untouched.
    """
    if not rows:
        return rows
    unit_re = re.compile(r"^(?:\s*\([^)]{1,18}\)){1,3}$")
    wrap_re = re.compile(r"^\([A-Za-z0-9]{1,4}\)$")
    out = []
    for r in rows:
        cells = [str(c) for c in r]
        nonempty = [c for c in cells if c.strip()]
        if len(nonempty) < 3:
            out.append(r)
            continue
        counts = [len(c.split("\n")) for c in nonempty]
        k_max = max(counts)
        if k_max < 4 or k_max > 60:
            out.append(r)
            continue
        deep = sum(1 for k in counts if k >= k_max - 1)
        if deep < 3:
            out.append(r)
            continue
        cols = []
        for c in cells:
            if not c.strip():
                cols.append(None)
                continue
            lines = [l.strip() for l in c.split("\n")]
            while lines and unit_re.match(lines[0]):
                lines.pop(0)
            merged = []
            for l in lines:
                if merged and wrap_re.match(l):
                    merged[-1] = (merged[-1] + " " + l).strip()
                else:
                    merged.append(l)
            cols.append(merged)
        k = max(len(col) for col in cols if col is not None)
        for i in range(k):
            row_i = [(col[i].strip() if col is not None and i < len(col) else "") for col in cols]
            nonempty_i = [v for v in row_i if v]
            if not nonempty_i:
                continue
            unitish = sum(
                1 for v in nonempty_i if len(v) <= 20 and v.startswith("(") and v.endswith(")")
            )
            if unitish and unitish >= len(nonempty_i) - 1 and unitish >= 2:
                continue
            out.append(row_i)
    return out


def build_table_markdown(rows, strategy="unknown", header_words=None):
    rows = _expand_stacked_rows(rows)  # det53: must run pre-scrub — scrub flattens the \n stacks
    rows = _scrub_grid_rows(rows)
    """Build markdown table from rows with multi-row header support.

    Replaces the inline markdown construction in each detector. Produces
    consistent output across strategies. ``header_words`` (iter-119a):
    [(x0, x1, text)] of the header line's page words — when provided,
    fused header cells split at true column boundaries before emit.

    GARBAGE-OUTPUT GUARD: rejects tables where ≥80% of body rows are empty
    OR contain only whitespace/dashes (signature of failed OCR outputting
    blank rows that the table detector then wraps as a "table").
    Returns "" for garbage tables — caller should skip.
    """
    if not rows or len(rows) < 2:
        return ""
    rows = normalize_grid_rows(rows)
    rows = [[_flat_cell(c) for c in r] for r in rows]  # det44: single-line cells
    rows = _split_fused_numeric_rows(rows)  # det52: split '2.44 2.12 20%' fusions on short rows
    header, body_start = detect_multirow_header(rows)
    # HEADER-EATS-ALL GUARD (2026-08-19, iter-57): on short label|value
    # tables ('Assignment|20%'), every row matches the multi-row-header
    # heuristics (short, non-decimal cells) and body_start reached
    # len(rows) — the whole table became header, body_rows = [], and
    # build returned "" ('No tables found', 56d7730c 4 gold tests).
    # A header by definition leaves at least one body row; fall back to
    # single-row header when the merger over-consumed.
    if body_start >= len(rows):
        header, body_start = rows[0], 1
    if header_words:
        try:
            header = split_header_on_word_gaps(header, header_words)
        except Exception as e:
            log.warning("header word-gap split skipped: %s", e)
    n_cols = len(header)
    # Garbage guard: count "substantive" body rows (≥1 cell with ≥3 chars)
    body_rows = rows[body_start:]
    if not body_rows:
        return ""
    substantive = 0
    for r in body_rows:
        for c in r:
            cs = c.strip() if c else ""
            # Reject if cell is just dashes/whitespace (separator-like)
            if len(cs) >= 3 and not all(ch in "- _:." for ch in cs):
                substantive += 1
                break
    if substantive < len(body_rows) * 0.20:
        # Less than 20% substantive rows = garbage table (OCR failure signature)
        log.info(
            "build_table_markdown: rejected garbage table (strategy=%s, rows=%d, substantive=%d)",
            strategy,
            len(body_rows),
            substantive,
        )
        return ""
    # Structure validation (Option C, 2026-07-25): reject tables that fail
    # column consistency, body-text-as-table, or empty-cell-ratio checks.
    is_valid, issues = validate_table_structure(rows, strategy)
    if not is_valid:
        log.info(
            "build_table_markdown: rejected invalid table (strategy=%s, issues=%s)",
            strategy,
            issues,
        )
        return ""
    # Detect rowspan/colspan patterns for richer markdown (future HTML
    # rendering; for markdown we just log them for now).
    rowspans, colspans = infer_spanning_cells(rows)
    if rowspans or colspans:
        log.info(
            "build_table_markdown: spanning cells detected (rowspans=%d colspans=%d, strategy=%s)",
            len(rowspans),
            len(colspans),
            strategy,
        )
    # Infer column types for metadata annotation
    col_types = infer_column_types(rows)
    md_lines = []
    if col_types and any(t != "text" for t in col_types):
        types_str = ", ".join(f"{h}:{t}" for h, t in zip(header, col_types) if t != "text")
        if types_str:
            md_lines.append(f"<!-- column_types: {types_str} -->")
    md_lines.append("| " + " | ".join(header) + " |")
    for r in rows[body_start:]:
        r_padded = (r + [""] * n_cols)[:n_cols]
        md_lines.append("| " + " | ".join(r_padded) + " |")
    md_lines.append("")
    return "\n".join(md_lines)


# Document-level state for cross-page table merging.
# Tracks "open tables" at end of each page; if next page's first table
# has matching X-extent + column count, rows are appended.


# Numeric-ish token (cell value): digits with , . % ± signs, or short label
_NUMTOK_RE = re.compile(r"^[\d.,%±()\[\]+\-]+$")
_SENT_RE = re.compile(r"[.;:]")


def reconstruct_aligned_numeric_tables(text: str) -> str:
    """Rebuild pipe tables from flat OCR text with aligned numeric rows.

    Scanned tables OCR'd by rapidocr arrive as flat lines:
        1999 2000 2001 2002 2003
        9,123 8,476 9,218 9,996 7,333
        6,755 6,343 6,895 7,123 6,002
    A run of >=3 consecutive lines (single short label line may interleave)
    with consistent token counts and numeric-majority cells = a data grid.
    Emits a markdown pipe table; the run's first line becomes the header.

    Deterministic, pure-text — no geometry needed (works on OCR output).
    """
    lines = text.split("\n")
    out: list[str] = []
    i = 0
    n = len(lines)

    def _row_tokens(ln: str) -> list[str] | None:
        s = ln.strip()
        if not s or s.startswith("|") or s.startswith("#"):
            return None
        if len(s) > 100 or _SENT_RE.search(s.rstrip(" .")):
            return None
        toks = s.split()
        if len(toks) < 2 or len(toks) > 12:
            return None
        num = sum(1 for t in toks if _NUMTOK_RE.match(t))
        return toks if num >= max(2, len(toks) * 0.5) else None

    while i < n:
        toks = _row_tokens(lines[i])
        if toks is None:
            out.append(lines[i])
            i += 1
            continue
        # Collect the run: numeric-majority lines with consistent token
        # counts. ONE short label line ("Passenger Cars") may interleave
        # between data rows — it stays in the text flow (dropped from grid).
        run = [toks]
        j = i + 1
        while j < n:
            t2 = _row_tokens(lines[j])
            if t2 is not None and abs(len(t2) - len(run[-1])) <= 1:
                run.append(t2)
                j += 1
                continue
            s2 = lines[j].strip()
            if (
                s2
                and len(s2) <= 30
                and not _SENT_RE.search(s2)
                and not s2.startswith("|")
                and j + 1 < n
                and _row_tokens(lines[j + 1]) is not None
                and abs(len(_row_tokens(lines[j + 1])) - len(run[-1])) <= 1
            ):
                out.append(lines[j])  # keep label in text flow (no data loss)
                j += 1
                continue
            break
        if len(run) >= 3:
            counts = [len(r) for r in run]
            n_cols = max(counts)
            if n_cols >= 2 and n_cols - min(counts) <= 1:
                rows = [(r + [""] * n_cols)[:n_cols] for r in run]
                out.append("| " + " | ".join(_flat_cell(c) for c in rows[0]) + " |")
                for r in rows[1:]:
                    out.append("| " + " | ".join(_flat_cell(c) for c in r) + " |")
                out.append("")
                i = j
                continue
        out.append(lines[i])
        i += 1
    return "\n".join(out)


def detect_tables_in_text(text):
    """Parse markdown pipe-tables from OCR model output, validate them.

    GLM-OCR often emits its own markdown tables in the OCR text. These need
    to be:
      1. Validated (≥3 rows × ≥2 cols, no body-text-as-table)
      2. Multi-row header merged
      3. Returned in same format as detect_all_tables output

    This runs on the OCR'd TEXT (not PDF page geometry) — catches tables
    that the model identified correctly but the page-geometry detectors
    would miss (image-only tables, complex ruled grids).
    """
    if not text:
        return []
    lines = text.split("\n")
    # Find runs of consecutive pipe-table lines
    tables_found = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line.startswith("|") or not line.endswith("|"):
            i += 1
            continue
        # Collect consecutive table lines
        table_lines = []
        while i < len(lines) and lines[i].strip().startswith("|"):
            table_lines.append(lines[i].strip())
            i += 1
        if len(table_lines) < 3:
            continue
        # Parse rows
        rows = []
        for tl in table_lines:
            # Skip separator line (---|---|---)
            cells = [c.strip() for c in tl.strip("|").split("|")]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            rows.append(cells)
        if len(rows) < 3:
            continue
        # Validate: at least 1 numeric cell OR ≥2 short non-numeric cells per row
        # (filters out body-text-as-table false positives)
        col_counts = [len(r) for r in rows]
        if max(col_counts) < 2:
            continue
        # Quality gate: reject if any cell has >15 words AND <30% decimals
        # (body-text-as-table signature)
        is_valid = True
        for r in rows:
            for c in r:
                words = c.split()
                if len(words) > 15:
                    decimals = sum(1 for w in words if _is_numeric_cell(w))
                    if decimals / max(len(words), 1) < 0.30:
                        is_valid = False
                        break
            if not is_valid:
                break
        if not is_valid:
            continue
        # Build markdown with multi-row header
        markdown = build_table_markdown(rows, strategy="ocr_text_parse")
        if markdown:
            tables_found.append(
                {
                    "bbox": (0, 0, 0, 0),  # No geometry for OCR-text tables
                    "markdown": markdown,
                    "rows": rows,
                    "strategy": "ocr_text_parse",
                }
            )
    return tables_found


def _repair_broken_cmap_cell(cell):
    """Repair broken-CMap glyph substitutions inside a numeric-pair cell.

    Measured family (53d3f304 pg3 render layer): '(' maps to '~', ')' to
    '!', and '-' to '2' — '~20.05,0.10! 0.11' renders as '(-0.05,0.10)
    0.11'. Scoped to tokens BRACKETED by ~…! so a bare digit '2' is never
    touched; '2' → '-' only immediately after '~' or ',' inside a numeric
    token (unambiguous: a second leading digit in a decimal position
    cannot follow the open paren or comma).
    """
    import re as _re

    def _fix(tok):
        if not (tok.startswith("~") and tok.endswith("!")):
            return tok
        inner = tok[1:-1]
        if not _re.match(r"^[\d.,]+$", inner):
            return tok
        inner = _re.sub(r"(^|,)2", r"\g<1>-", inner)
        return "(" + inner + ")"

    return " ".join(_fix(t) for t in cell.split(" "))


def detect_chatty_list_tables(text):
    """Convert VL 'chatty' list output into a table grid (iter-71, 4db371ae).

    chandra-ocr sometimes answers with narration instead of markdown:
    a preamble ('The user wants me to...'), a 'Table Structure:' block of
    'Column N: NAME' lines, then 'Data Rows:' followed by numbered
    comma-separated entries. The CONTENT is a faithful transcription but
    no pipe/HTML table exists, so every gold table test dies on 'No
    tables found'. Mechanical rebuild — no invention: header = the named
    columns, rows = the numbered entries, split on commas.

    Gates (a prose paragraph never qualifies):
      - >=3 'Column N:' lines in a contiguous block
      - a 'Data Rows' marker within 5 lines after the block
      - >=2 numbered rows whose comma-field count equals the column count
        (majority of numbered rows; miscounted rows are padded/truncated
        to the header width)
    """
    if not text or "Column" not in text or "Data Rows" not in text:
        return []
    lines = text.split("\n")
    col_re = re.compile(r"^(?:[-*]\s*)?Column\s+(\d+)\s*[.:]\s*(.+?)\s*$")
    row_re = re.compile(r"^\s*(\d{1,3})\.\s+(.+?)\s*$")
    headers = None
    i = 0
    tables_found = []
    while i < len(lines):
        m = col_re.match(lines[i].strip())
        if not m:
            i += 1
            continue
        block = []
        while i < len(lines):
            mm = col_re.match(lines[i].strip())
            if mm:
                block.append((int(mm.group(1)), mm.group(2)))
                i += 1
            elif (
                lines[i].strip() == "" and i + 1 < len(lines) and col_re.match(lines[i + 1].strip())
            ):
                i += 1  # blank line inside the block
            else:
                break
        if len(block) < 3:
            continue
        # numbered indices must be 1..n ascending (a real column block)
        if [b[0] for b in block] != list(range(1, len(block) + 1)):
            continue
        # 'Data Rows' marker within 5 lines
        j = i
        marker = None
        while j < len(lines) and j - i <= 5:
            if "Data Rows" in lines[j]:
                marker = j
                break
            j += 1
        if marker is None:
            continue
        headers = [b[1] for b in block]
        rows = []
        k = marker + 1
        while k < len(lines):
            rm = row_re.match(lines[k])
            if not rm:
                if lines[k].strip() == "" and rows:
                    k += 1
                    continue
                break
            fields = [f.strip() for f in rm.group(2).split(",")]
            # The model drops commas irregularly at the RAGGED RIGHT
            # ('R_E$GULAR BODEGA AEROPUERTO' = 2 logical fields, one
            # token run). Left-positional assignment keeps the leading
            # columns exact; merged trailing fields stay in the last
            # parsed field. Overflow rows (>n_cols fields) are dropped.
            if len(fields) <= len(headers):
                rows.append(fields)
            k += 1
        if len(rows) < 2:
            continue
        grid = [headers] + [(r + [""] * len(headers))[: len(headers)] for r in rows]
        tables_found.append(
            {
                "bbox": (0, 0, 0, 0),  # no geometry — OCR-text table
                "rows": grid,
                "strategy": "ocr_chatty_list",
            }
        )
        i = k
    return tables_found


def _ws_impl(pdf_path, page_num, min_rows=4, min_cols=3, guard=True):
    """Detect borderless tables by VERTICAL WHITESPACE ALIGNMENT.

    Compact port of img2table's borderless pipeline (MIT): per-row
    whitespace gaps → adjacent-row matching → column sections. Rows whose
    WIDE gaps (>2.5x the page's median word gap) vertically align across
    >=min_rows consecutive rows form a table; the shared column cuts then
    slice every row into cells.

    Closes the glued-header gap (measured 53d3f304 pg3: pdfplumber_text
    emitted '0electrodes' — cell text with NO x-gap boundary, while the
    true gap structure '0  electrodes' is visible in word geometry).
    Pure geometry, deterministic, no ML.
    """
    import fitz

    try:
        doc = fitz.open(pdf_path)
        if page_num - 1 >= len(doc):
            doc.close()
            return []
        page = doc[page_num - 1]
        words = [w for w in page.get_text("words") if w[4].strip()]
        page_w = float(page.rect.width)
        doc.close()
    except Exception as e:
        log.warning("detect_tables_whitespace open failed p%s: %s", page_num, e)
        return []
    if len(words) < min_rows * min_cols:
        return []

    # Word → y-row map
    row_map = {}
    for x0, y0, x1, y1, txt, *_ in words:
        yc = round((y0 + y1) / 2)
        row_map.setdefault(yc, []).append((x0, x1, txt))
    # LOGICAL rows: tables render each cell as 1-2 STACKED lines ('(x,y)'
    # pair line above its r-value line, ~10pt apart). Sub-lines of the
    # SAME cell share a column x-band and merge into one logical row;
    # separate visual rows are >1.2 line-heights apart.
    heights = sorted(w[3] - w[1] for w in words)
    h_med = heights[len(heights) // 2] or 10.0

    def _x_collision_frac(row_a, row_b):
        """Fraction of row_b words whose center falls inside a row_a word's
        x-band (+/- 2.5pt). TRUE sub-line/label pairs are x-COMPLEMENTARY
        (stacked parts of the same visual row occupy different columns);
        parallel data rows collide almost fully (measured fa18a15c pg2:
        sub-line pairs 0/4, 0/1, 1/7 vs data-row pairs 6/6)."""
        bands = [(x0 - 2.5, x1 + 2.5) for x0, x1, _t in row_a]
        if not bands or not row_b:
            return 0.0
        hits = 0
        for x0, x1, _t in row_b:
            cx = (x0 + x1) / 2
            if any(lo <= cx <= hi for lo, hi in bands):
                hits += 1
        return hits / len(row_b)

    logical = []
    for yc in sorted(row_map):
        sub = sorted(row_map[yc], key=lambda t: t[0])
        if logical and yc - logical[-1]["yc"] <= 1.2 * h_med:
            # Row-pitch conflation guard (fa18a15c pg2): on compact tables
            # (h_med=9 -> threshold 10.8pt vs 9pt data-row pitch) the pure
            # y-distance test fuses EVERY adjacent data row. Sub-lines of
            # one visual row are x-complementary; parallel data rows
            # collide. Merge only on complement. The guard is deliberately
            # AGGRESSIVE (v1 semantics, run-11 A/B: refusing non-bare
            # collisions too is what recovers legacy-empty pages like
            # 31d7e5e5 pg162 / a45ce384 pg7) — same-column unit/text
            # stacks it wrongly splits are arbitrated by the
            # detect_tables_whitespace wrapper's bare-pair pathology gate,
            # which keeps the legacy carving there (1651cbb4 pg2 et al.).
            if (not guard) or _x_collision_frac(logical[-1]["words"], sub) < 0.5:
                logical[-1]["words"].extend(row_map[yc])
                logical[-1]["subs"].append(sub)
                continue
        logical.append(
            {
                "yc": yc,
                "words": list(row_map[yc]),
                "subs": [sorted(row_map[yc], key=lambda t: t[0])],
            }
        )
    rows = []
    gaps_all = []
    for r in logical:
        ws = sorted(r["words"], key=lambda t: t[0])
        gaps = [(b[0] - a[1], (a[1] + b[0]) / 2) for a, b in zip(ws, ws[1:])] if len(ws) > 1 else []
        rows.append({"yc": r["yc"], "words": ws, "gaps": gaps, "wide": [], "subs": r["subs"]})
        gaps_all.extend(g for g, _cx in gaps)
    if not gaps_all:
        return []
    # Negative gaps (overlapping word boxes — italic kerning, ~ operator
    # artifacts) must not poison the median: a page whose median gap is
    # -2.2 yields a negative wide-threshold and every gap looks "wide"
    # (measured 53d3f304 pg3).
    pos_gaps = sorted(g for g in gaps_all if g > 0)
    if not pos_gaps:
        return []
    gap_med = pos_gaps[len(pos_gaps) // 2] or 3.0
    wide_thr = max(2.5 * gap_med, gap_med + 2.0)
    # Wide gaps per row (x-center of the gap).
    for r in rows:
        ws = r["words"]
        r["wide"] = [(g, cx) for g, cx in r["gaps"] if g >= wide_thr]
        # WEDGE (iter-90): the LEFT EDGE of the word AFTER each wide gap.
        # Gap CENTERS drift with the left cell's content width (0b100ba
        # Phenotype cut centers {310..363} = 53pt spread); left edges
        # are typeset-fixed ({382,384,385,385} = 3pt). Growth matching
        # that falls back to these edges survives variable-width cells.
        r["wedges"] = [(cx, b[0]) for (g, cx), b in zip(r["gaps"], ws[1:]) if g >= wide_thr]

    # Adjacent-row whitespace MATCHING (img2table matching_whitespaces):
    # two gaps match when their x-centers sit within 0.5 * min width.
    def _match(a, b):
        return abs(a - b) <= max(1.5 * gap_med, 6.0)

    # Slide: find runs of consecutive rows sharing >=min_cols-1 wide gaps.
    # Tolerance is generous (a column HEADER's gaps sit a few pt off the
    # data-column gaps — header cells are wider; measured 53d3f304: data
    # cut at 222 vs header cut 221, but 285 vs 294 = 9pt off). Use one
    x_tol = max(1.5 * gap_med, 0.6 * h_med, 6.0)

    def _match(a, b):
        return abs(a - b) <= x_tol

    n = len(rows)

    tables = []

    def _grow(i):
        """Grow the longest run whose rows all share >=min_cols-1 wide-gap
        alignments. Grows BOTH directions from seed i (a column HEADER's
        gaps misalign from the data columns by up to a char-width —
        measured 53d3f304: header cut 294 vs data cut 285 — so the seed
        must not be the header). Returns (run_indices, shared_cuts).

        EDGE-AWARE (iter-90): a shared entry is (gap_center, next_word_
        left_edge). A row's wide gap matches a shared entry when the
        CENTERS align within x_tol OR the left edges align within
        edge_tol — gap centers drift with the left cell's content width
        (0b100ba LTP region: cut centers spread 53pt while every column
        left edge is stable), so center-only matching truncates runs on
        variable-width columns. Recovery/slicing still derive cuts from
        the clustered edges, so edge-extended rows slice correctly."""
        edge_tol = max(0.5 * h_med, 2.5)
        shared = list(rows[i]["wedges"])
        lo = hi = i

        # LABEL-FRAGMENT PASS-THROUGH (iter-88, ebfe1f9a pg29): row
        # labels often sit on their OWN line above the value line
        # (measured y-offset 12-15pt > 1.2*h_med merge, so they form
        # separate logical rows with ZERO wide gaps). A low-gap row
        # whose words are all confined LEFT of the first shared cut
        # is a label fragment, not a table boundary — grow PAST it
        # and merge its words into the following row at slice time.
        def _is_label_fragment(ri):
            wds = rows[ri]["words"]
            _first_cut = min(c for c, _e in shared) if shared else 0
            return bool(shared) and max(w[1] for w in wds) < _first_cut + x_tol

        def _shared_kept(row):
            """Shared entries this row still supports (>=min_cols-1 to
            keep growing). Center match OR left-edge match."""
            wed = rows[row]["wedges"]
            kept = []
            for cx, ex in shared:
                if any(_match(cx, cxj) or abs(ex - exj) <= edge_tol for cxj, exj in wed):
                    kept.append((cx, ex))
            return kept

        while lo - 1 >= 0:
            if len(rows[lo - 1]["wide"]) < min_cols - 1:
                if _is_label_fragment(lo - 1):
                    lo -= 1
                    continue
                break
            ns = _shared_kept(lo - 1)
            if len(ns) < min_cols - 1:
                if _is_label_fragment(lo - 1):
                    lo -= 1
                    continue
                break
            shared = ns
            lo -= 1
        while hi + 1 < n:
            if len(rows[hi + 1]["wide"]) < min_cols - 1:
                if _is_label_fragment(hi + 1):
                    hi += 1
                    continue
                break
            ns = _shared_kept(hi + 1)
            if len(ns) < min_cols - 1:
                if _is_label_fragment(hi + 1):
                    hi += 1
                    continue
                break
            shared = ns
            hi += 1
        return list(range(lo, hi + 1)), [cx for cx, _e in shared]

    i = 0
    while i < n:
        if len(rows[i]["wide"]) < min_cols - 1:
            i += 1
            continue
        run, shared = _grow(i)
        # HEADER ABSORPTION: the header row's own gaps drift up to a
        # char-width from the data columns (measured 541 cuts {221,294,
        # 360,424} vs data cuts {149,285,351,415}) so _grow excludes it —
        # but it sits directly above the run with the same wide-gap COUNT.
        # Absorb it (and its text) as the first grid row.
        absorbed = False
        if run[0] - 1 >= 0 and len(rows[run[0] - 1]["wide"]) >= min_cols - 1:
            run = [run[0] - 1] + run
            absorbed = True
        if len(run) >= min_rows:
            cuts = sorted(set(round(cx, 1) for cx in shared))
            # LEFT-EDGE CUT RECOVERY: gap-CENTER matching drops cuts
            # whose column's cells vary wildly in width — centers drift
            # with content length and never meet x_tol (0b100ba pg4
            # Phenotype/Age: centers {326,344,358,362} vs tol 6). But
            # the word AFTER the gap starts at the column's fixed
            # typeset left edge in every row ({382,384,385,385}). A
            # tight left-edge cluster backed by >=min_rows rows whose
            # gap is NOT already sliced by an existing cut (that
            # discriminator kills duplicate boundaries inside an
            # already-cut gap) is a real column cut.
            # ENRICH: fold label-fragment rows (grown past in _grow)
            # into the FOLLOWING row — their words join the merged
            # word list (so the label|first-value gap exists in every
            # row for cut recovery) and ride as an extra sub-line
            # (so slicing keeps the label's own line order intact).
            enriched = []
            pend = None
            for r in rows[run[0] : run[-1] + 1]:
                frag = (
                    len(r["wide"]) < min_cols - 1
                    and cuts
                    and max(w[1] for w in r["words"]) < min(cuts) + x_tol
                )
                if frag:
                    fw = sorted(r["words"], key=lambda t: t[0])
                    pend = fw if pend is None else sorted(pend + fw, key=lambda t: t[0])
                    continue
                if pend is not None:
                    if not enriched and len(r["wide"]) >= min_cols - 1:
                        # Fragment sits directly above the COLUMN HEADER
                        # (edge-aware growth now often pulls the header
                        # INTO the run, so 'absorbed' no longer marks
                        # it): caption words ('distance r ~below!.',
                        # 53d3f304) pollute header cells either by
                        # word-merge or by the multi-row header join.
                        # Caption rows carry no column content — DROP.
                        pend = None
                    else:
                        r = dict(
                            r,
                            words=sorted(r["words"] + pend, key=lambda t: t[0]),
                            subs=[pend] + r["subs"],
                        )
                        pend = None
                enriched.append(r)
            if pend is not None:
                enriched.append(
                    {"yc": run[-1], "words": pend, "subs": [pend], "wide": [], "gaps": []}
                )
            data_rows = range(1, len(enriched)) if absorbed else range(len(enriched))
            edge_tol = max(0.5 * h_med, 2.5)
            edges = []
            for ri in data_rows:
                ws = enriched[ri]["words"]
                for a, b in zip(ws, ws[1:]):
                    if b[0] - a[1] >= wide_thr:
                        edges.append((b[0], ri, a[1]))
            if edges:
                edges.sort()
                clusters = [[edges[0]]]
                for e in edges[1:]:
                    if e[0] - clusters[-1][-1][0] <= edge_tol:
                        clusters[-1].append(e)
                    else:
                        clusters.append([e])
                for cl in clusters:
                    cut = min(e[0] for e in cl) - 1.0
                    if any(abs(cut - c) <= x_tol for c in cuts):
                        continue
                    support = {e[1] for e in cl if e[2] <= cut}
                    if len(support) < min_rows:
                        continue
                    # every supporting row's gap must be UNSLICED: a
                    # cut inside a gap that already carries an existing
                    # cut is a duplicate boundary (phantom empty column)
                    sliced = {e[1] for e in cl if any(e[2] - 1 <= c <= e[0] + 1 for c in cuts)}
                    if sliced & support:
                        continue
                    cuts.append(round(cut, 1))
                cuts = sorted(cuts)
            grid = []
            # Interior rows with fewer wide gaps (row '(0)' stacks its
            # sub-lines below the wide threshold) still belong: the y-band
            # between run bounds is contiguous by construction.
            # Slice the row BETWEEN consecutive words when a cut falls
            # inside their gap. The absorbed header (first row, when
            # absorbed) uses ITS OWN wide-gap centers — they drift from
            # the data cuts by up to a char width (measured: shared
            # cuts glued 'Mold 0 electrodes' into one cell).
            for ri_, r in enumerate(enriched):
                row_cuts = (
                    sorted(set(round(cx, 1) for _g, cx in r["wide"]))
                    if (absorbed and ri_ == 0)
                    else cuts
                )
                # Slice each SUB-LINE separately at its own word gaps,
                # then bucket the resulting segments into columns by
                # x-band (cut interval containing the segment). A global
                # x-sort of stacked cell lines interleaves them (0b100ba
                # pg4: 'new does protein not require synthesis');
                # vertical bucketing keeps each sub-line's text intact
                # inside its column: 'does not require' + 'new protein
                # synthesis' join in reading order.
                col_segs = [[] for _ in range(len(row_cuts) + 1)]
                for sub_words in r["subs"]:
                    segs, cur, seg_x0, prev_x1 = [], [], None, None
                    for x0, x1, txt in sub_words:
                        if cur and prev_x1 is not None:
                            if any(prev_x1 - 1 <= c <= x0 + 1 for c in row_cuts):
                                segs.append((seg_x0, prev_x1, " ".join(cur)))
                                cur, seg_x0 = [], None
                        if not cur:
                            seg_x0 = x0
                        cur.append(txt)
                        prev_x1 = x1
                    if cur:
                        segs.append((seg_x0, prev_x1, " ".join(cur)))
                    for sx0, sx1, stxt in segs:
                        col_segs[bisect.bisect_left(row_cuts, (sx0 + sx1) / 2.0)].append(stxt)
                cells = [" ".join(parts).strip() for parts in col_segs]
                if any(cells):
                    grid.append([_repair_broken_cmap_cell(c) for c in cells])
            if len(grid) >= min_rows and max(len(g) for g in grid) >= min_cols:
                # PROSE-CELL veto (iter-89, 20_pg46): multi-column prose
                # sidebars reconstruct as fake tables whose cells are
                # FULL SENTENCE LINES. Measured separator: sidebar grids
                # run median 11 words/cell; every real table measured
                # across the bench (numeric AND all-text like the
                # 0b100ba phenotype table) stays <=4. Median >7 =
                # prose displacement — emitting it duplicates the prose
                # copy, glued, inside table cells.
                _all_c = [c for g in grid for c in g if str(c).strip()]
                _wpc = sorted(len(str(c).split()) for c in _all_c)
                if _wpc and _wpc[len(_wpc) // 2] > 7:
                    grid = []
            if grid and len(grid) >= min_rows and max(len(g) for g in grid) >= min_cols:
                band_words = [
                    w
                    for w in words
                    if rows[run[0]]["yc"] - 20 <= (w[1] + w[3]) / 2 <= rows[run[-1]]["yc"] + 20
                ]
                x0 = min(w[0] for w in band_words)
                x1 = max(w[2] for w in band_words)
                y0 = rows[run[0]]["yc"] - h_med
                y1 = rows[run[-1]]["yc"] + h_med
                md = build_table_markdown(grid, "whitespace_align")
                if md:
                    tables.append(
                        {"bbox": (x0, y0, x1, y1), "rows": grid, "strategy": "whitespace_align"}
                    )
            i = run[-1] + 1
        else:
            i += 1
    return tables


# --- whitespace-guard v2 quality gates (run-11, 2026-09-11) --------------
# Mirrors the proven iter-58 recarve gate family from
# text_pipeline.detect_all_tables (kept local here: importing text_pipeline
# from detector would be circular). Run-10 lesson: page-level table-COUNT
# selection is not scorer-aligned — a guard-shredded grid keeps the count
# (same region, rows split) and wins the count tie (17 PASS->FAIL on
# stacked-subline pages). v2 judges each table's GRID instead.
_WS_NUM_TOK = re.compile(r"\b\d+\.\d+%?\b|\b\d+%?\b")
_WS_BARE_DECIMAL = re.compile(r"\d+\.\d+")
# BARE-PAIR glued cell: two or more whitespace-separated bare numerics and
# NOTHING else ('2 68', '6 3 26 27' — fa18a15c fused data rows). Cells like
# '0 μM 250 μM' (1651cbb4) carry unit tokens: the gold expects those MERGED
# (stacked doses of one visual cell), so they are NOT pathology. This cell-
# level distinction is the only scorer-aligned discriminator between the
# two identically-geometric pages (run-10 vs run-11 word-geometry census).
_WS_BARE_PAIR_CELL = re.compile(r"\s*\d+(?:[.,]\d+)*%?(?:\s+\d+(?:[.,]\d+)*%?)+\s*")


def _ws_bare_pair_cells(cells):
    """Count of bare-numeric-pair glued cells — the fused-data-row pathology
    the guard exists to fix (scorer-proven on fa18a15c pg2 only)."""
    n = 0
    for row in cells:
        for c in row:
            if _WS_BARE_PAIR_CELL.fullmatch(str(c).replace("\n", " ")):
                n += 1
    return n


def _ws_num_tokens(cells):
    return set(_WS_NUM_TOK.findall(" ".join(str(c) for r in cells for c in r if c)))


def _ws_label_value_pairs(cells):
    """Adjacent (label, single-numeric-value) pairs — the olmOCR table
    scorer's adjacency contract (immediate-left cell is the label)."""
    pairs = set()
    for row in cells:
        vals = [str(c).strip() for c in row]
        for i in range(1, len(vals)):
            v, l = vals[i], vals[i - 1]
            if v and l and _WS_NUM_TOK.fullmatch(v) and not _WS_BARE_DECIMAL.search(l):
                pairs.add((l, v))
    return pairs


def _ws_bbox_embeds(outer, inner):
    """True when `outer` covers >50% of `inner`'s area (region match)."""

    def area(b):
        return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])

    ix = max(0.0, min(outer[2], inner[2]) - max(outer[0], inner[0]))
    iy = max(0.0, min(outer[3], inner[3]) - max(outer[1], inner[1]))
    return ix * iy > 0.5 * max(1.0, area(inner))


def detect_tables_whitespace(pdf_path, page_num, min_rows=4, min_cols=3):
    """Public entry: pathology-gated guard with PER-TABLE conservation gates.

    v2 (run-11): both merge policies run; for each overlapping region pair
    the guarded carve replaces the legacy carve ONLY when (a) the legacy
    grid shows the glued-numeric/ragged pathology the guard exists for and
    (b) the guarded grid passes the iter-58 conservation family: glue
    strictly reduced, numerics preserved, raggedness not introduced,
    label->value adjacency preserved. No pathology -> the legacy grid IS
    the proven floor behavior and passes through untouched (run-10's 17
    regressions were clean stacked-subline pages the guard shredded).
    Tables found by only one policy pass through unchanged. The count-tie
    preference (run-10) is gone — detection count is not a quality signal.
    """
    guarded = _ws_impl(pdf_path, page_num, min_rows, min_cols, guard=True)
    legacy = _ws_impl(pdf_path, page_num, min_rows, min_cols, guard=False)
    if not legacy:
        # Guard-only finds (legacy=0) are SUPPRESSED: they enter the
        # text_pipeline arbitration as new tables the floor never emitted
        # and the row-count coverage/dedup contest displaces proven
        # competitors (83f40aac pg26: guard table displaced the text_only
        # table carrying Cancer/500k/Muscle/220k -> 4 PASS->FAIL; a45ce384
        # pg7: Cuneus positional suppressed). Measured tally: emitting them
        # nets -2 (31d7e5e5 +2, a45ce384 +1/-1, 83f40aac -4). Recovering
        # the 31d7e5e5 family needs scorer-aligned arbitration upstream
        # (run-12 lane), not emission here.
        return []
    if not guarded:
        return legacy
    out = []
    claimed = set()
    for lt in legacy:
        l_rows = lt.get("rows") or []
        bare_l = _ws_bare_pair_cells(l_rows)
        best = None
        for gi, gt in enumerate(guarded):
            if gi in claimed:
                continue
            if _ws_bbox_embeds(gt.get("bbox", (0, 0, 0, 0)), lt.get("bbox", (0, 0, 1, 1))):
                best = gt, gi
                break
        if best is None or bare_l == 0:
            # No overlapping guarded carve, or no scorer-proven pathology
            # (bare-pair fused data rows) on the legacy grid: the legacy
            # carving IS the floor behavior — keep it untouched.
            out.append(lt)
            continue
        gt, gi = best
        g_rows = gt.get("rows") or []
        adopt = (
            _ws_bare_pair_cells(g_rows) < bare_l
            and _ws_num_tokens(l_rows) <= _ws_num_tokens(g_rows)
            and not _ws_label_value_pairs(l_rows) - _ws_label_value_pairs(g_rows)
        )
        if adopt:
            claimed.add(gi)
            out.append(gt)
        else:
            out.append(lt)
    # Guarded-only EXTRA tables are dropped on legacy-bearing pages: the
    # floor behavior never emitted them (0b100ba0 pg4 guarded-only sub-table
    # = run-10 regression noise); a page where legacy finds nothing at all
    # was handled by the early return above.
    return out


def _valley_resplit(run, h_med):
    """Full-height whitespace-valley column re-split (iter-65, 0953927d).

    The fixed cell-split threshold max(8, 2*h_med) under-splits tables
    whose column separator whitespace is narrower than 2x word height
    (measured: 26.3pt separator vs 26.68 threshold — every row's value
    glued to the next column's label, '16.0 Karl/Karl 92', 6 gold tests
    cell-missing). A whitespace channel that NO word in the whole run
    crosses is column evidence at ANY width above the run's word-space
    scale. That scale is the 2-means split of the run's within-cell word
    gaps (space mode vs separator mode), floored at 6pt.
    TYPE-CHANGE GATE: cuts are kept only where the separated pieces
    change type (numeric|text) in most straddling rows — full-height
    valleys inside aligned numeric arrays are layout, not boundaries.


    Monotone-additive: only splits coarse cells that straddle a full-height
    valley; never merges or moves anything else. Skips (returns run
    unchanged) when the gap distribution is unimodal or evidence is thin —
    then behavior is byte-identical to the fixed-threshold path.
    """
    intra = [b[0] - a[2] for rc in run for c in rc for a, b in zip(c["words"], c["words"][1:])]
    if len(intra) < 8 or len(run) < 3:
        return run
    lo, hi = min(intra), max(intra)
    if hi - lo < 4.0:  # unimodal — no separator scale to find
        return run
    # Deterministic 1-D 2-means: spaces cluster vs separator cluster.
    c1, c2 = lo, hi
    g1 = g2 = None
    for _ in range(20):
        g1 = [g for g in intra if abs(g - c1) <= abs(g - c2)]
        g2 = [g for g in intra if abs(g - c1) > abs(g - c2)]
        if not g1 or not g2:
            return run
        n1, n2 = sum(g1) / len(g1), sum(g2) / len(g2)
        if abs(n1 - c1) < 1e-9 and abs(n2 - c2) < 1e-9:
            break
        c1, c2 = n1, n2
    if len(g2) < 2 or len(g1) < 5 or c2 - c1 < 4.0:
        return run
    vmin = max(6.0, (c1 + c2) / 2.0)
    # Coverage union of every word in the run; uncovered intervals that
    # span all rows are the valleys (column separators).
    iv = sorted((w[0], w[2]) for rc in run for c in rc for w in c["words"])
    merged = []
    for a, b in iv:
        if merged and a <= merged[-1][1] + 0.5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    cuts = []
    for (a1, b1), (a2, b2) in zip(merged, merged[1:]):
        if a2 - b1 < vmin:
            continue
        vm = (b1 + a2) / 2.0
        # TYPE-CHANGE GATE (iter-65, 53d3f304 regression): a full-height
        # valley inside aligned NUMERIC arrays (math matrices, mean+sd
        # pairs) is layout, not a cell boundary — splitting there shifts
        # the anchor pool and degrades already-correct carves. Keep a cut
        # only when the pieces it separates CHANGE TYPE across the
        # boundary in most rows (value→label adjacency, 0953927d).
        same = mixed = 0
        for rc in run:
            for c in rc:
                if not (c["x0"] + 1.0 < vm < c["x1"] - 1.0):
                    continue
                lw = [w for w in c["words"] if (w[0] + w[2]) / 2.0 <= vm]
                rw = [w for w in c["words"] if (w[0] + w[2]) / 2.0 > vm]
                if not lw or not rw:
                    continue
                lt = _valley_piece_type(" ".join(w[4] for w in lw))
                rt = _valley_piece_type(" ".join(w[4] for w in rw))
                if lt == "empty" or rt == "empty":
                    continue
                if lt == rt:
                    same += 1
                else:
                    mixed += 1
        if mixed >= 2 and 3 * mixed >= same:
            cuts.append(vm)
    if not cuts:
        return run
    out = []
    for rc in run:
        new_rc = []
        for c in rc:
            local = [vm for vm in cuts if c["x0"] + 1.0 < vm < c["x1"] - 1.0]
            if not local:
                new_rc.append(c)
                continue
            pieces, curw = [], []
            for w in sorted(c["words"], key=lambda w: w[0]):
                while local and (w[0] + w[2]) / 2.0 > local[0]:
                    pieces.append(curw)
                    curw = []
                    local = local[1:]
                curw.append(w)
            pieces.append(curw)
            for p in [p for p in pieces if p]:
                new_rc.append(
                    {
                        "x0": p[0][0],
                        "x1": p[-1][2],
                        "y0": p[0][1],
                        "y1": p[-1][3],
                        "text": " ".join(w[4] for w in p),
                        "words": p,
                    }
                )
        out.append(new_rc)
    return out


def _valley_piece_type(txt: str) -> str:
    """Coarse cell-piece type for the valley type-change gate."""
    if any(len(t) >= 2 for t in re.findall(r"[A-Za-z]+", txt)):
        return "text"
    if any(ch.isdigit() for ch in txt):
        return "num"
    return "empty"


def _row_gap_threshold(rw, h_loc):
    """Local cell-separator threshold (iter-71, 0091c5b2).

    The page-scale split max(8, 2*h_med) over-glues smaller-font rows:
    stacked headers 20.8-21.3pt apart glued into one cell at the body
    scale (26pt) threshold, so 'Local council' never became its own
    column cell. Word spaces and column separators are two gap
    populations separated by deterministic 1-D 2-means; the local
    threshold is their midpoint when the bimodality is table-scale:
      word-space cluster  c1 <= 0.35*h_loc  (spaces are sub-glyph)
      separator cluster   c2 >= 0.90*h_loc  (columns are glyph-scale+)
    Two gaps suffice (L3 header rows carry one separator) — the scale
    gates, not the gap count, exclude justified-prose false splits
    (prose gaps 8-10pt < 0.9*h_loc). Fallback = page behavior.
    """
    ws = sorted(rw, key=lambda w: w[0])
    gaps = [b[0] - a[2] for a, b in zip(ws, ws[1:])]
    default = max(8.0, 2.0 * h_loc)
    if len(gaps) < 2:
        return default
    # Exact 1-D 2-means over the n-1 candidate splits (min/max-seeded
    # Lloyd iteration mis-converges when a separator gap sits closer to
    # the word-space cluster than to the widest gap: 53d3f304 header
    # gaps [2.7x4, 21.2x2, 26.0, 41.6] pulled the 21.2s into c1 and
    # rejected the split). n is tiny; exhaustive is exact.
    sg = sorted(gaps)
    best = None
    for k in range(1, len(sg)):
        a, b = sg[:k], sg[k:]
        m1, m2 = sum(a) / len(a), sum(b) / len(b)
        var = sum((g - m1) ** 2 for g in a) + sum((g - m2) ** 2 for g in b)
        if best is None or var < best[0]:
            best = (var, m1, m2)
    _, c1, c2 = best
    if c2 < 1.4 * c1:
        return default
    if c1 > 0.35 * h_loc or c2 < 0.90 * h_loc:
        return default
    thr = (c1 + c2) / 2.0
    return min(max(thr, max(6.0, 0.5 * h_loc)), default)


def _x_intervals(words, thr):
    """Word x-coverage intervals of one line, split at gaps > thr."""
    ws = sorted(words, key=lambda w: w[0])
    if not ws:
        return []
    ivs = [[ws[0][0], ws[0][2]]]
    for w in ws[1:]:
        if w[0] - ivs[-1][1] > thr:
            ivs.append([w[0], w[2]])
        else:
            ivs[-1][1] = max(ivs[-1][1], w[2])
    return ivs


def _subline_split(row_words):
    """Split an over-merged VISUAL row into lattice sub-lines (iter-71).

    The page-scale row threshold (0.7*h_med) fuses sub-pitch lattices —
    BMJ tables run 6.4pt data pitch and 6-7pt stacked headers against a
    9.1pt threshold, so 'Sydney'/'Elsewhere' rows merged into
    'Elsewhere Sydney | 160 44 | ...' and the header stack collapsed
    into a <2-cell mega-row that died at the run gate. Local
    single-linkage (0.5*h_loc) separates the lattice lines; single-word
    stacks re-combine downstream via _cell_continuation (text order
    preserved — x-sorted cell glue reads 'verified immunisations').
    """
    if len(row_words) < 2:
        return [row_words]
    hs = sorted(w[3] - w[1] for w in row_words)
    h_loc = hs[len(hs) // 2] or 10.0
    # 0.5 (not 0.55): ascender/descender-bearing words inflate the row
    # median ~15% — 'verified' (h 13.4) over 'immunisations' (h 11.2)
    # missed its 7.0pt lattice gap at 0.55*12.9 (0091c5b2 L3/L4).
    t = 0.5 * h_loc
    ws = sorted(row_words, key=lambda w: (w[1] + w[3]) / 2.0)
    subs, cur = [], [ws[0]]
    for w in ws[1:]:
        yc = (w[1] + w[3]) / 2.0
        prev = sum((x[1] + x[3]) / 2.0 for x in cur) / len(cur)
        if yc - prev <= t:
            cur.append(w)
        else:
            subs.append(cur)
            cur = [w]
    subs.append(cur)
    return subs


def detect_tables_rowspan_labels(pdf_path, page_num, min_values=4, min_labels=2):
    """Rowspan-label value-band tables (det42, 9c389720 Table S3 class).

    Two-column supplemental tables typeset with sparse section labels
    ('Education', 'Gender', 'Age', ...) each governing a run of dense
    value rows CENTER-aligned in a right-hand band: the label column is
    empty row-by-row (rowspan), so per-row multi-cell run collectors
    die; and center-aligned value cells leave x0 edges drifting by tens
    of pt, so x0-anchor clustering dies too. Every strategy missed the
    page (measured: 0/6 strategies on 9c389720 pg17).

    Emission contract: labels are FILLED DOWN onto every value row they
    govern — the bench scorer resolves left-heading from column 0 of the
    SAME row, and the first label row whose value-side cell is short and
    digit-free doubles as the header row.

    Fire conditions (rowspan signature, all required): >=2 label rows
    sharing one tight x0 cluster, >=4 value rows in a right-hand band,
    n_values >= 1.2 * n_labels (equal counts is a definition table other
    strategies own), >=2 digit-bearing values, median value <= 8 words.
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        if page_num - 1 >= len(doc):
            return []
        page = doc[page_num - 1]
        words = [w for w in page.get_text("words") if w[4].strip()]
        if len(words) < min_values + min_labels:
            return []
        heights = sorted(w[3] - w[1] for w in words)
        h_med = heights[len(heights) // 2] or 10.0
        page_w = page.rect.width

        # visual rows (same y-center pitch as the positional detector)
        ws = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
        vrows = []
        cur = [ws[0]]
        for w in ws[1:]:
            yc = (w[1] + w[3]) / 2
            prev = sum((x[1] + x[3]) / 2 for x in cur) / len(cur)
            if yc - prev <= 0.7 * h_med:
                cur.append(w)
            else:
                vrows.append(cur)
                cur = [w]
        vrows.append(cur)

        # per-row cells: split word runs at cell-scale gaps only
        def _cells(rw):
            rw = sorted(rw, key=lambda w: w[0])
            thr = max(1.8 * h_med, 12.0)
            cells = [[rw[0]]]
            for w in rw[1:]:
                if w[0] - cells[-1][-1][2] > thr:
                    cells.append([w])
                else:
                    cells[-1].append(w)
            return cells

        row_cells = [(r, _cells(r)) for r in vrows]

        # label column: tight x0 cluster (tol 3pt) LEFT of a dense band
        firsts = [(rc[0][0][0], len(rc)) for _, rc in row_cells if rc]
        if not firsts:
            return []
        x0_counts = {}
        for x0, _ in firsts:
            x0_counts[round(x0)] = x0_counts.get(round(x0), 0) + 1
        # candidate label x = leftmost x0 with >= min_labels supporters
        # (within 3pt), whose rows are mostly short first cells
        label_x = None
        for x0 in sorted(x0_counts):
            supporters = [(x, n) for x, n in firsts if abs(x - x0) <= 3.0]
            if len(supporters) < min_labels:
                continue

            # label cells are SHORT (1-5 words) and start uppercase —
            # justified prose lines glued into one long "cell" are not
            # labels (b5d9db35 measured: 12-word sentence "labels")
            def _label_ok(cell_words):
                t = " ".join(w[4] for w in cell_words).strip()
                wc = len(t.split())
                return 1 <= wc <= 5 and not t.endswith((",", ";", ","))

            short = sum(
                1
                for rw2, rc2 in row_cells
                if rc2 and abs(rc2[0][0][0] - x0) <= 3.0 and _label_ok(rc2[0])
            )
            if short >= 0.7 * len(supporters) and short >= min_labels:
                right_rows = [x for x, _ in firsts if x > x0 + 3.0]
                if len(right_rows) >= min_values:
                    label_x = x0
                    break
        if label_x is None:
            return []

        # collect label rows (y-center + text) and value cells (y, text).
        # Rowspan labels are typeset CENTERED on their group, so the first
        # value row often sits ABOVE its label — assignment is by NEAREST
        # label y, not "last label above".
        header = None
        labels = []
        values = []
        for rw, rc in row_cells:
            rc = sorted(rc, key=lambda c: c[0][0])
            label_cell = next((c for c in rc if abs(c[0][0] - label_x) <= 3.0), None)
            value_cells = [c for c in rc if c is not label_cell and c[0][0] > label_x + 3.0]
            if label_cell is not None:
                y_lab = sum((w[1] + w[3]) / 2 for w in label_cell) / len(label_cell)
                text = " ".join(w[4] for w in label_cell).strip()
                if value_cells:
                    vt = " ".join(w[4] for w in value_cells[0]).strip()
                    if (
                        header is None
                        and not labels
                        and len(vt.split()) <= 2
                        and not any(ch.isdigit() for ch in vt)
                    ):
                        header = [text, vt]
                        continue
                labels.append((y_lab, text))
            for c in value_cells:
                yc = sum((w[1] + w[3]) / 2 for w in c) / len(c)
                values.append((yc, " ".join(w[4] for w in c).strip(), c))

        if not labels:
            return []
        # segment values by inter-row GAPS (groups sit ~1.5x the intra-
        # group pitch apart), then assign each segment to the label whose
        # y falls INSIDE it. Nearest-label fails on centered rowspan
        # labels (ideology label y276.8, group starts y243.2) and drags
        # running headers into the first group.
        values.sort(key=lambda v: v[0])
        pitches = sorted(b[0] - a[0] for a, b in zip(values, values[1:]))
        pitch = pitches[len(pitches) // 2] if pitches else h_med
        seg_break = 1.35 * pitch
        segments = []
        cur_seg = [values[0]] if values else []
        for prev, nxt in zip(values, values[1:]):
            if nxt[0] - prev[0] > seg_break:
                segments.append(cur_seg)
                cur_seg = [nxt]
            else:
                cur_seg.append(nxt)
        if cur_seg:
            segments.append(cur_seg)
        grid = []
        used = []
        for seg in segments:
            y_lo, y_hi = seg[0][0], seg[-1][0]
            owner = next(
                (lt for lt in labels if y_lo - 2.0 <= lt[0] <= y_hi + 2.0),
                None,
            )
            if owner is None:
                continue
            for _, vtext, cell in seg:
                grid.append([owner[1], vtext])
                used.append(cell)

        n_labels = len({g[0] for g in grid}) + (1 if header else 0)
        if header is None or len(grid) < min_values or n_labels < min_labels:
            return []
        # rowspan signature: values outnumber labels (equal counts is a
        # definition table other strategies own)
        if len(grid) < 1.2 * (n_labels - 1):
            return []
        # every real label owns >=1 value (a label with zero values is a
        # misdetected cluster — e.g. a left-margin page-number column)
        if len({g[0] for g in grid}) < n_labels - (1 if header else 0):
            return []
        vals = [g[1] for g in grid]
        if sum(1 for v in vals if any(ch.isdigit() for ch in v)) < 2:
            return []
        med_words = sorted(len(v.split()) for v in vals)[len(vals) // 2]
        if med_words > 8:
            return []
        # math / mojibake guards (same contract as the positional path).
        # Extended glyph set, occurrence-counted: display-equation blocks
        # align like rowspan groups (2503.04024 pg4: 'B(u(x); g(x)) =
        # η(x)' labels with '∀x ∈Γη,' values) — distinct-glyph counting
        # missed them (∀, Γ absent from the classic set).
        mglyphs = "√∑∫≤≥→∈∞±×÷∂∇∝⊕⊗∼≈≠∓∴∵∀∃ΩΓΛΘΦΨ"
        all_txt = " ".join(v for v in vals)
        if sum(all_txt.count(g) for g in mglyphs) >= 3:
            return []
        if sum(1 for g in grid if "=" in g[0] and "(" in g[0] and ")" in g[0]) > 0:
            return []
        if "~" in all_txt and any(ch.isdigit() for ch in all_txt.split("~")[0][-3:]):
            moji = sum(1 for v in vals if "~" in v and any(ch.isdigit() for ch in v))
            if moji > 0.3 * len(vals):
                return []
        # value band must sit right of the label column and inside the page
        x0 = min(w[0] for c in used for w in c)
        y0 = min(w[1] for c in used for w in c)
        x1 = max(w[2] for c in used for w in c)
        y1 = max(w[3] for c in used for w in c)
        if x1 - x0 > 0.9 * page_w:
            return []
        return [
            {
                "bbox": (x0, y0, x1, y1),
                "rows": [header] + grid,
                "strategy": "rowspan_labels",
            }
        ]
    finally:
        doc.close()


def _raster_region_table(engine, page, rect):
    """OCR one image region and rebuild its table grid (det43 helper).

    Returns a rows list (list of cell lists) or [] when the region is
    not table-like. Geometry contract, measured on the two bench
    exemplars (3a91fad8 leave-balance, 587a1f5a author list):

    - column bands merge line x-spans only on SUBSTANTIAL overlap
      (>= 25% of the shorter span): plain interval-union fuses
      'Andreas Heinrich' [176-251pt] into the affiliation column
      (starts x=240) — one giant column, no table; while center-aligned
      value bands still fuse with their headers, which x0-clustering
      would split (3a91: header x0 142.5 vs values 175-187).
    - rows split at y-center gaps > 0.75 * median line height.
    - 2-band name|detail layouts pair multi-line right blocks to the
      first left row BELOW them (affiliations typeset above authors).
    """
    import fitz
    import numpy as np

    zoom = 6.0
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=rect, colorspace=fitz.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    out = engine(img)
    if out is None or getattr(out, "txts", None) is None:
        return []
    raw = [(t, s, bx) for t, s, bx in zip(out.txts, out.scores, out.boxes) if t and t.strip()]
    if len(raw) < 4:
        return []
    scores = sorted(s for _, s, _ in raw)
    if scores[len(scores) // 2] < 0.75:
        return []
    lines = []
    for t, _, bx in raw:
        xs = [p[0] for p in bx]
        ys = [p[1] for p in bx]
        lines.append([min(xs), min(ys), max(xs), max(ys), t.strip()])
    if len(lines) < 4:
        return []

    all_text = " ".join(l[4] for l in lines)
    # prose guard: paragraphs are not tables
    if sum(1 for l in lines if len(l[4].split()) > 10) > 0.5 * len(lines):
        return []
    # math guard (det42 glyph set): equation screenshots align like grids
    mglyphs = "√∑∫≤≥→∈∞±×÷∂∇∝⊕⊗∼≈≠∓∴∵∀∃ΩΓΛΘΦΨ"
    if sum(all_text.count(g) for g in mglyphs) >= 3:
        return []
    # mojibake guard (broken-CMap '~2' digit smear)
    if "~" in all_text and sum(
        1 for l in lines if "~" in l[4] and any(ch.isdigit() for ch in l[4])
    ) > 0.3 * len(lines):
        return []
    # chart-annotation guard: R^2 fit labels are scatter-plot signature
    if "R²" in all_text or "R2 =" in all_text:
        return []
    # CJK prose guard: figure captions/headings from CJK papers render as
    # aligned text blocks; targets so far are Latin-script only
    cjk = sum(1 for ch in all_text if "\u4e00" <= ch <= "\u9fff" or "\u3040" <= ch <= "\u30ff")
    if len(all_text.replace(" ", "")) and cjk > 0.4 * len(all_text.replace(" ", "")):
        return []

    heights = sorted(l[3] - l[1] for l in lines)
    h_med = heights[len(heights) // 2] or 10.0

    # column bands via substantial-overlap merge (left-to-right greedy)
    spans = sorted((l[0], l[2], i) for i, l in enumerate(lines))
    bands = []  # [x0, x1, {line idx}]
    for x0, x1, i in spans:
        for b in bands:
            ov = min(x1, b[1]) - max(x0, b[0])
            if ov > 0.25 * min(x1 - x0, b[1] - b[0]):
                b[0] = min(b[0], x0)
                b[1] = max(b[1], x1)
                b[2].add(i)
                break
        else:
            bands.append([x0, x1, {i}])
    bands = [b for b in bands if len(b[2]) >= 2]
    if len(bands) < 2:
        return []
    bands.sort(key=lambda b: b[0])

    # drop lines in sparse (single-line) bands — stray captions/noise
    kept = set().union(*(b[2] for b in bands))
    lines = [l for i, l in enumerate(lines) if i in kept]

    # y-bands: rows split at gaps > 0.75 * median height
    order = sorted(range(len(lines)), key=lambda i: (lines[i][1] + lines[i][3]) / 2)
    row_ids = []
    cur = [order[0]]
    for i in order[1:]:
        yc = (lines[i][1] + lines[i][3]) / 2
        prev = sum((lines[j][1] + lines[j][3]) / 2 for j in cur) / len(cur)
        if yc - prev <= 0.75 * h_med:
            cur.append(i)
        else:
            row_ids.append(cur)
            cur = [i]
    row_ids.append(cur)

    def _band_of(i):
        for bi, b in enumerate(bands):
            if i in b[2]:
                return bi
        return None

    # PAIRING MODE (name | detail, 2 columns): left rows are single
    # short lines, right column carries >= 1.5x as many lines —
    # multi-line right blocks attach to the first left row BELOW them
    # (587a1f5a: affiliation block typeset ~30pt above its author).
    if len(bands) == 2:
        left_rows = [r for r in row_ids if all(_band_of(i) == 0 for i in r)]
        left_lines = [i for r in left_rows for i in r]
        right_lines = [i for r in row_ids for i in r if _band_of(i) == 1]
        if (
            len(left_rows) >= 3
            and all(len(lines[i][4].split()) <= 5 for i in left_lines)
            and len(right_lines) >= 1.5 * len(left_lines)
        ):
            rows = []
            left_ys = [
                (sum((lines[i][1] + lines[i][3]) / 2 for i in r) / len(r), r) for r in left_rows
            ]
            left_ys.sort()
            for k, (ly, r) in enumerate(left_ys):
                # detail blocks sit ABOVE their owner row (587a1f5a:
                # affiliation ~30pt above its author) — attach right
                # lines in (previous left row, this left row]
                prev = left_ys[k - 1][0] if k > 0 else float("-inf")
                block = [
                    lines[i] for i in right_lines if prev < (lines[i][1] + lines[i][3]) / 2 <= ly
                ]
                if not block:
                    continue
                block.sort(key=lambda l: l[1])
                rows.append(
                    [
                        " ".join(lines[i][4] for i in sorted(r, key=lambda j: lines[j][0])),
                        " ".join(l[4] for l in block),
                    ]
                )
            if len(rows) >= 3:
                # prose right-column guard: address blocks pair like
                # name|affiliation but their cells are word-heavy
                rwords = sorted(len(r[1].split()) for r in rows)
                if rwords[len(rwords) // 2] <= 10:
                    return rows

    # GRID MODE: every row needs >=2 populated bands; >=3 rows total
    grid = []
    for r in row_ids:
        cells = [""] * len(bands)
        for i in r:
            bi = _band_of(i)
            if bi is None:
                continue
            cells[bi] = (cells[bi] + " " + lines[i][4]).strip()
        if sum(1 for c in cells if c) >= 2:
            grid.append(cells)
    if len(grid) < 3:
        return []
    # density guard: chart labels/legends scatter into a sparse lattice;
    # real raster tables fill ~100% of their grid
    populated = sum(1 for r in grid for c in r if c)
    if populated < 0.55 * len(grid) * len(bands):
        return []
    # tiny-grid guard: legends pair a tick stack with a caption ('50 |
    # Keterangan :') — real tables carry >= 8 populated cells
    if populated < 8:
        return []
    # glyph-soup guard: broken figure OCR emits symbol crumbs; targets
    # have no <=2-char cells at all
    cells_all = [c for r in grid for c in r if c]
    if sum(1 for c in cells_all if len(c) <= 2) > 0.4 * len(cells_all):
        return []
    # two-band wordy stacks are parallel prose (address columns), not
    # tables; name|detail pages go through pairing mode instead
    if len(bands) == 2:
        wwords = sorted(len(c.split()) for c in cells_all)
        if wwords[len(wwords) // 2] > 6:
            return []
    # chart-axis guard: a tick column is a monotone run of short pure
    # ints ('0','10','20','30','40','50'); decimals ('0.0') and years
    # (4-digit) are excluded on purpose
    for bi in range(len(bands)):
        col = [r[bi].replace(",", "") for r in grid if r[bi]]
        col = [c for c in col if c.lstrip("-").isdigit() and len(c) <= 3]
        if len(col) < 4:
            continue
        inc = sum(1 for a, b in zip(col, col[1:]) if int(b) > int(a))
        dec = sum(1 for a, b in zip(col, col[1:]) if int(b) < int(a))
        if inc >= len(col) - 1 or dec >= len(col) - 1:
            return []
    # grid tables must be numeric-bearing (pairing mode owns the
    # digit-free name|affiliation class)
    if sum(1 for l in lines if any(ch.isdigit() for ch in l[4])) < 2:
        return []
    return grid


def detect_tables_raster(pdf_path, page_num, max_regions=2):
    """Raster-embedded tables on born-digital pages (det43).

    Some born-digital pages typeset whole tables as raster images with
    at most a partial text layer (3a91fad8 leave-balance: labels in the
    text layer but '+ Adjustments'/'1,154.5' raster-only; 587a1f5a
    VASSCAA-7 author table: nothing in the text layer) — every
    text-layer strategy misses, measured 0 tables on both pages.

    Renders the qualifying image regions (>=80x40pt, aspect 0.8-12,
    <=2 largest per page) at zoom 6 grayscale, rapidocr's them
    (deterministic PP-OCRv6 small, CPU, shared engine singleton), and
    rebuilds the grid from OCR box geometry — see
    :func:`_raster_region_table` for the geometry contract.

    Fires only from Strategy 4e when no other strategy found a table
    (chatty-list contract — never displaces an existing detection).
    """
    import fitz

    try:
        from scientific_parser.ingest.ocr_engine import _get_rapidocr
    except Exception:
        return []
    engine = _get_rapidocr()
    if engine is None:
        return []

    doc = fitz.open(pdf_path)
    try:
        if page_num - 1 >= len(doc):
            return []
        page = doc[page_num - 1]
        rects = []
        for img in page.get_images(full=True):
            try:
                for r in page.get_image_rects(img[0]):
                    if (
                        r.width >= 80
                        and r.height >= 40
                        and 0.8 <= r.width / max(r.height, 1.0) <= 12
                    ):
                        rects.append((r.width * r.height, r))
            except Exception:
                continue
        rects.sort(key=lambda x: -x[0])
        tables = []
        for _, rect in rects[:max_regions]:
            try:
                rows = _raster_region_table(engine, page, rect)
            except Exception as e:
                log.warning("raster region OCR failed: %s", e)
                continue
            if rows:
                tables.append(
                    {
                        "bbox": (rect.x0, rect.y0, rect.x1, rect.y1),
                        "rows": rows,
                        "strategy": "raster_ocr",
                    }
                )
        return tables
    finally:
        doc.close()


def detect_tables_positional(pdf_path, page_num, min_rows=3, min_cols=2):
    """Reconstruct borderless tables from WORD COORDINATES.

    Closes the "cell-per-line layer" gap (bench tables 2026-08-15: 57
    digital pages where every strategy failed): text extractors emit each
    CELL as its own line — no gaps, no rules — so lines/pdfplumber/camelot
    all miss. The geometry survives in word positions:

      1. cluster words into VISUAL ROWS by y-center (side-by-side cells
         of one table row share y)
      2. persistent-gutter STREAM SPLIT (iter-71): a word-free x-channel
         shared by >=4 consecutive rows with content on both sides is a
         column-stream boundary (side-by-side prose+table layers, 12pt
         gutters the page-scale cell threshold glues over). Split only
         when exactly one side still yields a table — two table-like
         halves are one wide table (conservative, keeps status quo).
      3. per-row SUB-LINE SPLIT (iter-71): local 0.55*h_loc lattice
         lines with multi-line-cell re-merge; LOCAL bimodal cell-gap
         threshold instead of the page-scale 2*h_med.
      4. cluster cell x-starts into COLUMN anchors
      5. emit grid runs of >=min_rows rows x >=min_cols columns;
         leading all-text rows above the first numeric row merge
         vertically into one space-joined header cell per column
         (stacked headers 'Local council'+'only' -> 'Local council
         only', exact-match gold cells).

    Deterministic, pure geometry — the classic positional reconstruction.
    """
    import fitz

    doc = fitz.open(pdf_path)
    try:
        if page_num - 1 >= len(doc):
            return []
        page = doc[page_num - 1]
        words = [w for w in page.get_text("words") if w[4].strip()]
        if len(words) < min_rows * min_cols:
            return []

        heights = sorted(w[3] - w[1] for w in words)
        h_med = heights[len(heights) // 2] or 10.0

        # 1. visual rows: cluster by y-center while consecutive gap < 0.7*h_med
        ws = sorted(words, key=lambda w: ((w[1] + w[3]) / 2, w[0]))
        vrows = []
        cur = [ws[0]]
        for w in ws[1:]:
            yc = (w[1] + w[3]) / 2
            prev_yc = sum((x[1] + x[3]) / 2 for x in cur) / len(cur)
            if yc - prev_yc <= 0.7 * h_med:
                cur.append(w)
            else:
                vrows.append(cur)
                cur = [w]
        vrows.append(cur)

        _mglyphs = "√∑∫≤≥→∈∞±×÷∂∇∝⊕⊗∼≈≠∓∴∵"

        def _row_cells(rw):
            # cells per row: split word run by the LOCAL gap threshold
            # (iter-71: page-scale 2*h_med glued 21pt column starts in
            # small-font rows; _row_gap_threshold separates word-space
            # and separator populations when the split is table-scale).
            hs = sorted(w[3] - w[1] for w in rw)
            h_loc = hs[len(hs) // 2] if hs else 10.0
            thr = _row_gap_threshold(rw, h_loc)
            rw = sorted(rw, key=lambda w: w[0])
            cells = [[rw[0]]]
            for w in rw[1:]:
                if w[0] - cells[-1][-1][2] > thr:
                    cells.append([w])
                else:
                    cells[-1].append(w)
            # DOT-LEADER DROP (iter-66b): fill runs '.......' between
            # label and value are table furniture, never content.
            return [
                {
                    "x0": c[0][0],
                    "x1": c[-1][2],
                    "y0": c[0][1],
                    "y1": c[-1][3],
                    "text": " ".join(
                        w[4] for w in c if not (len(w[4]) >= 4 and set(w[4]) <= {"."})
                    ),
                    "words": c,
                }
                for c in cells
            ]

        def _cell_continuation(run, rc, nxt):
            """Single-cell row fusing into the cell ABOVE (iter-71/66b).

            Priority 1 — GEOMETRIC STACK (iter-71): a single WORD, no
            digits/math, whose bbox OVERLAPS the x-aligned cell above
            (stacked sub-pitch line) fuses regardless of case — 'No'
            under 'Total', 'verified' under 'immunisations' (0091c5b2).
            The 1-word + exact x0 alignment + overlap combination does
            not occur in prose (prose lines are multi-word).
            Priority 2: text continuation (≤3 words, lowercase start or
            trailing period) into an x-aligned cell in ANY column
            (ragged stacked headers) or the label column (iter-66b,
            """
            if len(rc) != 1:
                return None
            c = rc[0]
            words = c["text"].split()
            # MATH GUARD (2503.09086_pg5): text-like content only.
            if re.search(r"[\\^_=$\d]|[" + _mglyphs + "]", c["text"]):
                return None
            if c["y0"] - max(cc["y1"] for cc in run[-1]) > h_med:
                return None
            if len(words) == 1:
                # GEOMETRIC STACK (iter-71): no case gate, no following-
                # full-row gate — the stack itself ('No' under 'Total')
                # is the evidence; prose lines are never single words
                # exactly x-aligned under a cell they overlap.
                if not run:
                    return None
                for cc in run[-1]:
                    ox = min(c["x1"], cc["x1"]) - max(c["x0"], cc["x0"])
                    if ox <= 0:
                        continue
                    aligned = abs(c["x0"] - cc["x0"]) <= 2.0 or ox >= 0.9 * (c["x1"] - c["x0"])
                    if aligned and c["y0"] - cc["y1"] < 0.25 * h_med:
                        return cc
                return None
            if nxt is None or len(nxt) < 2 or len(run) < 2:
                return None
            if len(words) > 3:
                return None
            if not (c["text"][:1].islower() or c["text"].rstrip().endswith(".")):
                return None
            for cc in run[-1]:
                ox = min(c["x1"], cc["x1"]) - max(c["x0"], cc["x0"])
                if ox <= 0:
                    continue
                if (
                    abs(c["x0"] - cc["x0"]) <= 2.0 or ox >= 0.9 * (c["x1"] - c["x0"])
                ) and ox >= 0.5 * (c["x1"] - c["x0"]):
                    return cc
            first_cells = [r[0] for r in run if r]
            label_x0 = min(cc["x0"] for cc in first_cells)
            label_x1 = max(cc["x1"] for cc in first_cells)
            val_x0 = min((cc["x0"] for r in run for cc in r[1:]), default=float("inf"))
            overlap = min(c["x1"], label_x1) - max(c["x0"], label_x0)
            if overlap < 0.5 * (c["x1"] - c["x0"]):
                return None
            if val_x0 < float("inf") and c["x1"] > val_x0 - 2.0:
                return None
            return run[-1][0]

        def _flush(run):
            # 2-row runs allowed only via the >=3-column path below.
            if len(run) < min_rows and len(run) < 2:
                return []
            # iter-65: split glued cells at full-height whitespace valleys
            # BEFORE anchor clustering — the glue starves anchor formation.
            run = _valley_resplit(run, h_med)
            # column anchors: cluster x0 across run
            x0s = sorted(c["x0"] for rc in run for c in rc)
            anchors = []
            # 2-row runs (action-item tables, acd61dce): cell x0s jitter
            # more than the 1.5*h_med anchor threshold ('Who' 376.9 vs
            # 'Dr.' 347.5 — 29pt), splitting one logical column into two
            # anchors and failing the shared-column gate. Widen the
            # anchor tolerance for SHORT runs: with only 2 rows of
            # evidence, each column needs ONE anchor, and genuine
            # adjacent columns sit a full cell-width apart.
            _anchor_tol = max(10.0, (3.0 if len(run) <= 2 else 1.5) * h_med)
            for x in x0s:
                if not anchors or x - anchors[-1][-1] > _anchor_tol:
                    anchors.append([x])
                else:
                    anchors[-1].append(x)
            anchors = [sum(a) / len(a) for a in anchors]
            # MAJORITY-BACKED ANCHORS (iter-105, 0953927d): an anchor
            # backed by only ONE row is x-jitter of that row's label
            # (district-number headers: 'DISTRICT 70'@403.5 while every
            # other row's third-group label sits @368.6) — it carves a
            # phantom column, empty in all body rows yet un-droppable
            # because its creator row fills it (broke left-adjacency
            # golds: '| Jagger | 69.8 | Jagger |  | 49.7 |'). Drop
            # single-row anchors in runs >2; their lone cells re-bucket
            # to the nearest surviving anchor in the grid mapping below.
            if len(run) > 2:
                _support: list[set] = [set() for _ in anchors]
                # cells assigned to each anchor, per row — a dropped
                # anchor's cells must re-bucket WITHOUT splitting a
                # multi-word cell: its lone cell must not x-overlap a
                # cell that the nearest surviving anchor already owns
                # in the SAME row (8ec7ac2f 'Freight Transport
                # Association': the affiliation column's anchor varies
                # row-to-row because multi-word cells span past it —
                # dropping it shredded every affiliation).
                _cells_at: list[dict[int, list]] = [{} for _ in anchors]
                for _ri, _rc in enumerate(run):
                    for _c in _rc:
                        _ci = min(
                            range(len(anchors)),
                            key=lambda i: abs(anchors[i] - _c["x0"]),
                        )
                        _support[_ci].add(_ri)
                        _cells_at[_ci].setdefault(_ri, []).append(_c)
                _drop_idx = []
                for i in range(len(anchors)):
                    if len(_support[i]) >= 2:
                        continue
                    # candidate jitter anchor: check each of its lone
                    # cells against the row's cells at the nearest
                    # KEPT anchor — overlap means the cell belongs to
                    # that anchor's column already (a spanning cell),
                    # so the anchor split a multi-word cell: KEEP it.
                    _safe = True
                    for _ri, _cs in _cells_at[i].items():
                        _row_others = [
                            c
                            for j in range(len(anchors))
                            if j != i and len(_support[j]) >= 2
                            for c in _cells_at[j].get(_ri, [])
                        ]
                        if not _row_others:
                            continue
                        _near = min(
                            _row_others,
                            key=lambda c: abs(c["x0"] - anchors[i]),
                        )
                        for _c in _cs:
                            if min(_c["x1"], _near["x1"]) - max(_c["x0"], _near["x0"]) > 0:
                                _safe = False
                                break
                        if not _safe:
                            break
                    if _safe:
                        _drop_idx.append(i)
                if 2 <= len(anchors) - len(_drop_idx) < len(anchors):
                    anchors = [a for i, a in enumerate(anchors) if i not in _drop_idx]
            if len(anchors) < min_cols:
                return []
            # 2-row candidate: require >=3 columns AND >=2 shared columns
            # between the two rows (prose lines align only by chance).
            if len(run) == 2:
                if len(anchors) < 3:
                    return []
                cols_r = [
                    min(range(len(anchors)), key=lambda i: abs(anchors[i] - c["x0"]))
                    for c in run[0]
                ]
                cols_r2 = [
                    min(range(len(anchors)), key=lambda i: abs(anchors[i] - c["x0"]))
                    for c in run[1]
                ]
                if len(set(cols_r) & set(cols_r2)) < 2:
                    return []
            grid = []
            for rc in run:
                row = [""] * len(anchors)
                for c in rc:
                    ci = min(range(len(anchors)), key=lambda i: abs(anchors[i] - c["x0"]))
                    row[ci] = (row[ci] + " " + c["text"]).strip() if row[ci] else c["text"]
                grid.append(row)
            # drop empty columns (anchors with no content)
            keep = [i for i in range(len(anchors)) if any(r[i] for r in grid)]
            if len(keep) < min_cols:
                return []
            grid = [[r[i] for i in keep] for r in grid]
            # filled-ratio guard: a real grid has most rows mostly filled
            filled = sum(1 for r in grid for c in r if c) / (len(grid) * len(grid[0]))
            if filled < 0.5:
                return []
            # MATH-SIGNATURE guard: display equations align like rows (√n, ≤,
            # sup …) but are NOT tables.
            all_cells = [c for r in grid for c in r if c]
            if not all_cells:
                return []
            mathy = sum(
                1
                for c in all_cells
                if any(g in c for g in _mglyphs) or re.search(r"\\[a-zA-Z]{2,}", c)
            )
            if mathy / len(all_cells) > 0.4:
                return []
            # PROSE-SPAN guard: justified text with wide word gaps forms fake
            # "cells" of 8+ words. Real table cells are short.
            med_words = sorted(len(c.split()) for c in all_cells)[len(all_cells) // 2]
            if med_words > 6:
                return []
            # MOJIBAKE GUARD (iter-71, 53d3f304): fitz words from cmap-
            # broken regions carry '~0,0.01!'-style damage — emitting
            # that carve displaces cleaner strategies over the same bbox
            # (whitespace reads the repaired layer; its cells matched 2/5
            # gold vs this carve's 1/5). A grid whose cells are >30%
            # tilde-numeric is not this detector's problem to solve.
            _moji = sum(1 for c in all_cells if "~" in c and any(ch.isdigit() for ch in c))
            if _moji > 0.3 * len(all_cells):
                return []
            # PROSE-CELL veto (iter-89, 20_pg46): multi-column prose
            # sidebars reconstruct as fake tables whose cells are full
            # sentence lines — median words/cell > 7 = displacement
            # (measured sidebar med=11, real tables <=4; column-type
            # rule removed: all-text real tables carry no numeric col).
            _all_c2 = [c for r in grid for c in r if str(c).strip()]
            _wpc2 = sorted(len(str(c).split()) for c in _all_c2)
            if _wpc2 and _wpc2[len(_wpc2) // 2] > 7:
                return []
            # Flow-chain (iter-89): prose columns flow cell-to-cell —
            # upper cell ends mid-sentence AND lower starts lowercase,
            # in chains. Measured: sidebar chains 8; real tables (even
            # stacked-phenotype 0b100ba, uncertainty-stack 53d3f304,
            # label-stack f45a188e) <=3. Chain >=6 = column of running
            # text = prose displacement.
            _nch = max(len(g) for g in grid)
            _max_chain = 0
            for _ci in range(_nch):
                _col = [str(r[_ci]).strip() if _ci < len(r) else "" for r in grid]
                _run = 0
                for _a, _b in zip(_col, _col[1:]):
                    if _a and _b and not re.search(r"[.!?;:]$", _a) and _b[:1].islower():
                        _run += 1
                        _max_chain = max(_max_chain, _run)
                    else:
                        _run = 0
            if _max_chain >= 6:
                return []
            # HEADER-ZONE VERTICAL MERGE (iter-71, 0091c5b2): leading
            # all-text rows above the first numeric row are a stacked
            # header; markdown/HTML need ONE space-joined cell
            # per column for exact gold-cell matching.
            first_data = None
            for ri, r in enumerate(grid):
                if any(c and (_is_numeric_cell(c) or c.strip()[:1].isdigit()) for c in r):
                    first_data = ri
                    break
            # STACK MERGE (iter-71, 53d3f304): a row of short numeric
            # cells at tight pitch (<=1.1*h_med) whose FIRST content
            # column starts RIGHT of the row above is the second line of
            # that row's cells (mean±sd stacks '(-0.04,0.04) 0.06' —
            # uncertainty lines are column-numeric with no label, while
            # label rows restart at the label column: 0091c5b2
            # Sydney/Elsewhere both start at column 0).
            _stk = re.compile(r"^[\d~()\[\],.+\-!±\s]+$")

            def _first_content(row):
                for k, c in enumerate(row):
                    if str(c).strip():
                        return k
                return len(row)

            i = (first_data or 0) + 1
            while i < len(grid) and i < len(run):
                prev, cur = grid[i - 1], grid[i]
                if (
                    _first_content(cur) > _first_content(prev)
                    and all((not str(c).strip()) or _stk.match(str(c).strip()) for c in cur)
                    and min(c["y0"] for c in run[i]) - min(c["y0"] for c in run[i - 1])
                    <= 1.1 * h_med
                ):
                    cur_pad = (list(cur) + [""] * len(prev))[: len(prev)]
                    grid[i - 1] = [f"{a} {b}".strip() for a, b in zip(prev, cur_pad)]
                    del grid[i]
                    del run[i]
                    continue
                i += 1
            if first_data is not None and 2 <= first_data <= 4:
                n_c = len(grid[0])
                hdr = []
                for ci in range(n_c):
                    parts = []
                    for r in grid[:first_data]:
                        if ci < len(r) and r[ci] and r[ci] not in parts:
                            parts.append(r[ci])
                    hdr.append(" ".join(parts))
                grid = [hdr] + grid[first_data:]
            x0 = min(c["x0"] for rc in run for c in rc)
            y0 = min(c["y0"] for rc in run for c in rc)
            x1 = max(c["x1"] for rc in run for c in rc)
            y1 = max(c["y1"] for rc in run for c in rc)
            return [{"bbox": (x0, y0, x1, y1), "rows": grid, "strategy": "positional"}]

        def _collect_runs(word_rows):
            # 3. per-row SUB-LINE SPLIT, then cells, then contiguous runs
            # of multi-cell rows = candidate tables.
            sub_rows = [sub for r in word_rows for sub in _subline_split(r)]
            row_cells = [_row_cells(r) for r in sub_rows]
            row_cells = [rc for rc in row_cells if rc]
            # STAGGERED-LABEL REPAIR (iter-105, 0c9f34ecf): tables
            # typeset with each row's label baseline a few pt BELOW the
            # PREVIOUS row's numbers y-cluster into LADDER vrows — each
            # vrow holds [next-row label, this-row numbers], shifting
            # every label one row up. Two passes:
            #   (a) LADDER DETACH (top-down, holding): a leading short
            #       non-numeric label cell sitting BELOW its own row's
            #       numbers band (y0 > others' min y0 + 0.25*h) is the
            #       next row's label — detach and prepend to the next
            #       entry (bottom-up variants hand labels to already-
            #       modified entries; order matters).
            #   (b) DOWN-FUSE: a lone label vrow close above and
            #       strictly left of a >=4-cell row joins it (else the
            #       continuation logic glues it onto the PREVIOUS row).
            _held = None
            for _pi in range(len(row_cells)):
                _rc = row_cells[_pi]
                _new_held = None
                if len(_rc) >= 4:
                    _lab = _rc[0]
                    _rest_y0 = min(c["y0"] for c in _rc[1:])
                    if (
                        len(_lab["text"].split()) <= 2
                        and not re.search(r"\d", _lab["text"])
                        and _lab["y0"] > _rest_y0 + 0.25 * h_med
                    ):
                        _rc = _rc[1:]
                        _new_held = _lab
                if _held is not None:
                    _rc = [_held] + _rc
                row_cells[_pi] = _rc
                _held = _new_held
            if _held is not None:
                row_cells.append([_held])
            _fused: list = []
            _i = 0
            while _i < len(row_cells):
                _rc = row_cells[_i]
                _nx = row_cells[_i + 1] if _i + 1 < len(row_cells) else None
                if (
                    _nx is not None
                    and len(_rc) == 1
                    and len(_nx) >= 4
                    and len(_rc[0]["text"].split()) <= 2
                    and not re.search(r"\d", _rc[0]["text"])
                    and 0 < _nx[0]["y0"] - _rc[0]["y0"] <= 1.5 * h_med
                    and 0 < _nx[0]["x0"] - _rc[0]["x1"] <= 2.5 * h_med
                ):
                    _fused.append([_rc[0]] + _nx)
                    _i += 2
                    continue
                _fused.append(_rc)
                _i += 1
            row_cells = _fused
            tables = []
            run = []

            for i, rc in enumerate(row_cells):
                nxt = row_cells[i + 1] if i + 1 < len(row_cells) else None
                if len(rc) >= min_cols:
                    run.append(rc)
                else:
                    tgt = _cell_continuation(run, rc, nxt) if run else None
                    if tgt is not None:
                        tgt["text"] = (tgt["text"] + " " + rc[0]["text"]).strip()
                        tgt["x1"] = max(tgt["x1"], rc[0]["x1"])
                        tgt["y1"] = max(tgt["y1"], rc[0]["y1"])
                        tgt["words"] = tgt.get("words", []) + rc[0].get("words", [])
                    else:
                        tables.extend(_flush(run))
                        run = []
            tables.extend(_flush(run))
            return tables

        # 2. persistent-gutter STREAM SPLIT (iter-71, 0091c5b2 table 2):
        # a word-free x-channel (width >= max(9, 0.8*h_med)) supported
        # by >=4 rows with words on both sides, crossed by NO word in
        # the span (1.5pt slack for ragged stream edges), is a stream
        # boundary. Splitting at the WORD level first also keeps
        # side-by-side streams from chaining the sub-line single-linkage.
        # SPLIT only when exactly one side yields a table AND the losing
        # side is PROSE-like (median >=4 words/row): the impostor
        # channels are a table's own internal column gaps — both halves
        # are table fragments, never prose, and stay together.
        chan_thr = max(9.0, 0.8 * h_med)

        def _free_ivs(rw):
            s = sorted(rw, key=lambda w: w[0])
            return [(a[2], b[0]) for a, b in zip(s, s[1:]) if b[0] - a[2] >= chan_thr]

        def _grow(i0, lo0, hi0):
            """Grow a channel span downward across CONTIGUOUS rows.

            Returns (end_row_exclusive, lo, hi, n_support). Neutral rows
            (one side only, nothing crossing) stay IN the span — dropping
            them would erase their words from every segment (0091c5b2:
            'Total | 155 | 49' row lost because its 11.64pt gutter sat
            0.15pt under the channel threshold). Only a row with a word
            penetrating the channel > 1.5pt ends it.
            """
            lo, hi = lo0, hi0
            n_support = 1
            j = i0 + 1
            while j < len(vrows):
                r = vrows[j]
                pen = any(w[2] - lo > 1.5 and hi - w[0] > 1.5 for w in r)
                if pen:
                    break
                inter = next(
                    (
                        (max(lo, a), min(hi, b))
                        for a, b in _free_ivs(r)
                        if min(hi, b) - max(lo, a) >= chan_thr
                    ),
                    None,
                )
                if inter is not None:
                    n_support += 1
                    lo, hi = inter
                j += 1
            return j, lo, hi, n_support

        def _prose_like(word_rows):
            # Dot-leader fillers are table furniture, not words — counted,
            # they pushed a 3-word label column's median to 4 and the
            # impostor channel (label col <-> first value col, cf7ccdf5)
            # fired and dropped the label column.
            txts = [
                " ".join(w[4] for w in r if not (len(w[4]) >= 4 and set(w[4]) <= {"."}))
                for r in word_rows
                if r
            ]
            if not txts:
                return False
            med = sorted(len(t.split()) for t in txts)[len(txts) // 2]
            return med >= 4

        def _try_span(i0):
            """Best qualifying channel starting at row i0, or None."""
            for lo, hi in sorted(_free_ivs(vrows[i0]), key=lambda t: t[1] - t[0], reverse=True):
                end, clo, chi, n_sup = _grow(i0, lo, hi)
                if n_sup < 4:
                    continue
                # PAGE-LEVEL GUTTER GATE (cf7ccdf5 impostor): a stream
                # boundary is a layout feature of the PAGE — the channel
                # must also be free in >=1 row ABOVE the span start. A
                # table's own internal column gap exists only across the
                # table's rows; the rows above it (full-width prose)
                # cross the channel.
                up = 0
                ulo, uhi = clo, chi
                for j in range(i0 - 1, -1, -1):
                    if any(w[2] - ulo > 1.5 and uhi - w[0] > 1.5 for w in vrows[j]):
                        break
                    inter = next(
                        (
                            (max(ulo, a), min(uhi, b))
                            for a, b in _free_ivs(vrows[j])
                            if min(uhi, b) - max(ulo, a) >= chan_thr
                        ),
                        None,
                    )
                    if inter is not None:
                        up += 1
                        ulo, uhi = inter
                if up < 1:
                    continue
                c = (clo + chi) / 2.0
                band = vrows[i0:end]
                left = [[w for w in r if w[2] <= c] for r in band]
                right = [[w for w in r if w[0] >= c] for r in band]
                t_l = _collect_runs([r for r in left if r])
                t_r = _collect_runs([r for r in right if r])
                if t_l and t_r:
                    continue  # one wide table
                if t_l and _prose_like(right):
                    return end, t_l
                if t_r and _prose_like(left):
                    return end, t_r
            return None

        tables = []
        pos = 0
        i = 0
        while i < len(vrows):
            if i < pos:
                i += 1
                continue
            hit = _try_span(i)
            if hit is None:
                i += 1
                continue
            end, t_band = hit
            tables.extend(_collect_runs(vrows[pos:i]))
            tables.extend(t_band)
            pos = end
            i = pos
        tables.extend(_collect_runs(vrows[pos:]))
        return tables
    finally:
        doc.close()


def detect_text_only_tables(pdf_path, page_num, min_rows=4, min_cols=2, prior_bboxes=None):
    """Pattern-based detector for tables WITHOUT decimal numbers.

    Catches: mineral-abbreviation tables, description tables, classification
    tables, stratigraphic tables — all of which have aligned short tokens
    but no numeric data.

    Algorithm (non-ML):
      1. Extract words with bbox via pdfplumber.
      2. Bucket into Y-lines (3pt snap).
      3. A line is a "candidate row" if:
         - Has ≥2 words
         - Average word length ≤15 chars (not prose)
         - Total line length ≤80 chars (not prose)
         - Has internal whitespace gaps ≥3x normal word spacing
      4. Cluster consecutive candidate rows by column structure.
      5. A cluster of ≥min_rows matching rows = a table.
      6. Build markdown by snapping words to nearest column center.
    """
    import pdfplumber
    from collections import defaultdict

    tables_found = []
    try:
        with pdfplumber.open(pdf_path) as doc:
            if page_num < 1 or page_num > len(doc.pages):
                return []
            page = doc.pages[page_num - 1]
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                return []
            lines = defaultdict(list)
            for w in words:
                y_key = round(w["top"] / 3) * 3
                lines[y_key].append(w)
            sorted_ys = sorted(lines.keys())
            candidates = []
            for y in sorted_ys:
                ws = sorted(lines[y], key=lambda x: x["x0"])
                if len(ws) < min_cols:
                    continue
                total_chars = sum(len(w["text"]) for w in ws)
                if total_chars > 80:
                    continue
                avg_len = total_chars / len(ws)
                if avg_len > 15:
                    continue
                gaps = []
                for i in range(1, len(ws)):
                    gap = ws[i]["x0"] - ws[i - 1]["x1"]
                    gaps.append(gap)
                if not gaps:
                    continue
                med_gap = sorted(gaps)[len(gaps) // 2]
                large_gaps = sum(1 for g in gaps if g >= max(med_gap * 3, 8))
                # EVEN-PITCH TABULAR ROW (iter-115, 83f40 pg26): the ×3
                # relative rule assumes multi-word columns (word gaps ≪
                # column gaps). Single-token columns make EVERY gap a
                # column gap (12-23pt, uniform) so no gap is 'large' and
                # the whole grid is rejected ('Data set n m Ratio /
                # Cancer 3.9k 217 1.14'). ≥2 gaps ≥8pt = tabular pitch;
                # prose lines run 3.5-5pt word gaps and never qualify.
                wide_gaps = sum(1 for g in gaps if g >= 8)
                # CONTENT GATE (iter-115 fix, 470591 pg3): figure axis
                # tick rows ('0 1 2 3 4 5 6 7' at 25pt pitch) and
                # subscript bands ('10 10 10 10') share even-pitch wide
                # gaps but carry NO cell ≥3 chars; a real table row
                # almost always has one (a label or value like '3.9k').
                if large_gaps < 1 and (wide_gaps < 2 or max(len(w["text"]) for w in ws) < 3):
                    continue
                # CELL GROUPING (iter-115): a ≥8pt gap IS the column
                # separator — words joined across <8pt gaps form ONE cell
                # ('Data set'), so candidate columns are CELLS, not words.
                # Per-word centers made canonical columns from word-count
                # maxima and shredded multi-word headers.
                cells = [[ws[0]]]
                for prev, w in zip(ws, ws[1:]):
                    if w["x0"] - prev["x1"] >= 8:
                        cells.append([w])
                    else:
                        cells[-1].append(w)
                col_centers = [
                    sum(w["x0"] + w["x1"] for w in cell) / (2 * len(cell)) for cell in cells
                ]
                candidates.append(
                    {
                        "y": y,
                        "words": ws,
                        "cells": cells,
                        "col_centers": col_centers,
                        "n_cols": len(cells),
                        "even_pitch": large_gaps < 1,
                    }
                )
            clusters = []
            current = []
            last_y = None
            for c in candidates:
                if last_y is not None and c["y"] - last_y > 20:
                    if len(current) >= min_rows:
                        clusters.append(current)
                    current = []
                if current:
                    prev = current[-1]
                    if abs(c["n_cols"] - prev["n_cols"]) <= 1:
                        match_count = 0
                        for cc in c["col_centers"][:3]:
                            for pc in prev["col_centers"][:3]:
                                if abs(cc - pc) <= 20:
                                    match_count += 1
                                    break
                        if match_count >= 2:
                            current.append(c)
                        else:
                            if len(current) >= min_rows:
                                clusters.append(current)
                            current = [c]
                    else:
                        if len(current) >= min_rows:
                            clusters.append(current)
                        current = [c]
                else:
                    current.append(c)
                last_y = c["y"]
            if len(current) >= min_rows:
                clusters.append(current)

            # LAST-RESORT SEMANTICS (iter-115 fix, 470591 pg3): an
            # even-pitch cluster admits NUMERIC single-token grids —
            # pdfplumber's contract. When an earlier strategy already
            # found a table here, our merged-header variant steals the
            # dedup win (uniform widths beat shredded-but-correct
            # headers) and the page regresses. Even-pitch clusters only
            # claim regions no prior strategy covered (same pattern as
            # whitespace_align in the cascade).
            def _overlaps_prior(cluster):
                if not prior_bboxes:
                    return False
                bx0 = min(c["words"][0]["x0"] for c in cluster)
                by0 = cluster[0]["y"]
                bx1 = max(c["words"][-1]["x1"] for c in cluster)
                by1 = cluster[-1]["y"] + 10
                area = max((bx1 - bx0) * (by1 - by0), 1)
                for pb in prior_bboxes:
                    ix0, iy0 = max(bx0, pb[0]), max(by0, pb[1])
                    ix1, iy1 = min(bx1, pb[2]), min(by1, pb[3])
                    if ix0 < ix1 and iy0 < iy1:
                        p_area = max((pb[2] - pb[0]) * (pb[3] - pb[1]), 1)
                        if (ix1 - ix0) * (iy1 - iy0) / min(area, p_area) > 0.4:
                            return True
                return False

            clusters = [
                cl
                for cl in clusters
                if not (any(c["even_pitch"] for c in cl) and _overlaps_prior(cl))
            ]
            for cluster in clusters:
                canonical_centers = sorted(max((c["col_centers"] for c in cluster), key=len))
                n_cols = len(canonical_centers)
                if n_cols < min_cols:
                    continue
                rows = []
                for c in cluster:
                    row = [""] * n_cols
                    # CELL PLACEMENT (iter-115): cells were grouped at
                    # ≥8pt gaps in the candidate stage; place each WHOLE
                    # cell at its nearest canonical column center. Per-word
                    # placement shredded multi-word cells ('Data set').
                    for cell_words in c["cells"]:
                        text = " ".join(w["text"] for w in cell_words)
                        wc = sum(w["x0"] + w["x1"] for w in cell_words) / (2 * len(cell_words))
                        best_idx = min(
                            range(n_cols),
                            key=lambda i: abs(canonical_centers[i] - wc),
                        )
                        if abs(canonical_centers[best_idx] - wc) <= 30:
                            if row[best_idx]:
                                row[best_idx] += " " + text
                            else:
                                row[best_idx] = text
                    rows.append(row)
                if len(rows) < min_rows:
                    continue
                # BODY-TEXT FALSE-POSITIVE GUARD (2026-07-25):
                # text_only_pattern can cluster 2-column body text rows as
                # "tables" because each row has aligned tokens. Reject if any
                # row has a cell with >15 words AND <30% decimals (same gate
                # as camelot_stream + ocr_text_parse).
                is_body_text = False
                for r in rows:
                    for c in r:
                        words_in_cell = c.split()
                        if len(words_in_cell) > 15:
                            decimals = sum(1 for w in words_in_cell if _is_numeric_cell(w))
                            if decimals / max(len(words_in_cell), 1) < 0.30:
                                is_body_text = True
                                break
                    if is_body_text:
                        break
                if is_body_text:
                    continue
                rows = [[_flat_cell(c) for c in r] for r in rows]  # det44: single-line cells
                md_lines = ["| " + " | ".join(rows[0]) + " |"]
                for r in rows[1:]:
                    r_padded = (r + [""] * n_cols)[:n_cols]
                    md_lines.append("| " + " | ".join(r_padded) + " |")
                md_lines.append("")
                tables_found.append(
                    {
                        "bbox": (
                            min(c["words"][0]["x0"] for c in cluster),
                            cluster[0]["y"],
                            max(c["words"][-1]["x1"] for c in cluster),
                            cluster[-1]["y"] + 10,
                        ),
                        "markdown": "\n".join(md_lines),
                        "rows": rows,
                        "strategy": "text_only_pattern",
                        # TENTATIVE (iter-115): admitted via the even-pitch
                        # branch — numeric single-token grids are outside
                        # this detector's contract, so downstream
                        # last-resort stages (whitespace_align) must NOT
                        # treat these as coverage.
                        "even_pitch": any(c["even_pitch"] for c in cluster),
                    }
                )
    except Exception as e:
        log.warning("detect_text_only_tables failed p%s: %s", page_num, e)
        return []
    return tables_found


def detect_tables_lines_strict(pdf_path, page_num):
    """Detector using PyMuPDF's lines_strict strategy for ruled tables.

    Catches tables with explicit borders (horizontal/vertical rules).
    Common in older typeset papers (pre-2000) and some modern journals
    (Am Min, CMP) using ruled table grids.
    """
    import fitz

    tables_found = []
    try:
        for hit in native_find_tables(pdf_path, page_num, strategy="lines_strict"):
            rows = hit["rows"]
            if len(rows) < 3:
                continue
            col_counts = [len(r) for r in rows]
            if max(col_counts) < 2:
                continue
            n_cols = max(col_counts)
            header = rows[0]
            md_lines = ["| " + " | ".join(_flat_cell(c) for c in header) + " |"]
            for r in rows[1:]:
                r_padded = (r + [""] * n_cols)[:n_cols]
                md_lines.append("| " + " | ".join(_flat_cell(c) for c in r_padded) + " |")
            md_lines.append("")
            tables_found.append(
                {
                    "bbox": tuple(hit["bbox"]),
                    "markdown": "\n".join(md_lines),
                    "rows": rows,
                    "strategy": "lines_strict",
                }
            )
    except Exception as e:
        log.warning("detect_tables_lines_strict failed p%s: %s", page_num, e)
        return []
    return tables_found


def detect_tables_camelot(pdf_path, page_num):
    """Camelot-based detector for ruled AND whitespace tables.

    Two flavors (both run, results merged + deduped):
      - **lattice**: OpenCV-based line/intersection detection. Best for
        ruled tables (older typeset papers, Am Min / CMP journal style).
        Returns 0 for whitespace-only tables.
      - **stream**: column detection via kmeans on text positions. Tight
        tolerances (column_tol=15, row_tol=3) prevent over-firing that
        camelot's defaults suffer from (441-column false positive on
        ballhaus2004 p2 in earlier test).

    Returns: list of tables (same format as detect_tables_pdfplumber).
    """
    try:
        import camelot
    except ImportError:
        return []
    tables_found = []
    try:
        # Lattice flavor — ruled tables
        try:
            lattice = camelot.read_pdf(
                pdf_path,
                pages=str(page_num),
                flavor="lattice",
                line_scale=40,
            )
            for t in lattice:
                if t.parsing_report.get("accuracy", 0) < 80:
                    continue
                df = t.df
                if len(df) < 3 or len(df.columns) < 2:
                    continue
                rows = df.values.tolist()
                rows = [[str(c) if c is not None else "" for c in r] for r in rows]
                # bbox from camelot's _bbox attribute (page coordinates)
                bbox = (0, 0, 0, 0)
                try:
                    bbox = tuple(t._bbox) if hasattr(t, "_bbox") else (0, 0, 0, 0)
                except Exception as e:
                    log.debug("detect_tables_camelot: Exception degraded: %s", e)
                tables_found.append(
                    {
                        "bbox": bbox,
                        "rows": rows,
                        "strategy": "camelot_lattice",
                    }
                )
        except Exception as e:
            log.warning("camelot lattice failed p%s: %s", page_num, e)
        # Stream flavor — whitespace tables (tight tolerances)
        try:
            stream = camelot.read_pdf(
                pdf_path,
                pages=str(page_num),
                flavor="stream",
                columns=None,  # auto-detect
                row_tol=3,
                column_tol=15,
            )
            for t in stream:
                if t.parsing_report.get("accuracy", 0) < 80:
                    continue
                # Reject if whitespace% >85 (almost-empty = false positive)
                if t.parsing_report.get("whitespace", 0) > 85:
                    continue
                df = t.df
                if len(df) < 3 or len(df.columns) < 2:
                    continue
                # Reject if column count > 30 (over-firing signature)
                if len(df.columns) > 30:
                    continue
                rows = df.values.tolist()
                rows = [[str(c) if c is not None else "" for c in r] for r in rows]
                # BODY-TEXT FALSE-POSITIVE GUARD (2026-07-25):
                # camelot wraps 2-column body text as fake tables. Real table
                # cells have short values (numbers, labels, single tokens).
                # Reject if ANY of these body-text signatures appear:
                #   1. Cell contains newline (real cells are single-line)
                #   2. Cell has >8 words AND <30% decimals (prose signature)
                #   3. Row has avg word count >5 (multi-word prose cells)
                is_body_text = False
                for r in rows:
                    row_word_count = 0
                    row_cell_count = 0
                    for c in r:
                        if not c.strip():
                            continue
                        row_cell_count += 1
                        # Signature 1: newline = body text wrap
                        if "\n" in c.strip():
                            is_body_text = True
                            break
                        words = c.split()
                        row_word_count += len(words)
                        # Signature 2: >8 words with low decimal ratio
                        if len(words) > 8:
                            decimals = sum(1 for w in words if _is_numeric_cell(w))
                            if decimals / max(len(words), 1) < 0.30:
                                is_body_text = True
                                break
                    if is_body_text:
                        break
                    # Signature 3: avg words per cell >5
                    if row_cell_count > 0 and row_word_count / row_cell_count > 5:
                        is_body_text = True
                        break
                if is_body_text:
                    continue
                bbox = (0, 0, 0, 0)
                try:
                    bbox = tuple(t._bbox) if hasattr(t, "_bbox") else (0, 0, 0, 0)
                except Exception as e:
                    log.debug("detect_tables_camelot: Exception degraded: %s", e)
                tables_found.append(
                    {
                        "bbox": bbox,
                        "rows": rows,
                        "strategy": "camelot_stream",
                    }
                )
        except Exception as e:
            log.warning("camelot stream failed p%s: %s", page_num, e)
    except Exception as e:
        log.warning("detect_tables_camelot failed p%s: %s", page_num, e)
        return []
    return tables_found


def detect_tables_in_figure(pdf_path, page_num, figure_bbox):
    """OCR a figure region with table-aware prompt to find image-embedded tables.

    GATED by module-level `_FIGURE_TABLE_OCR` flag (default False). OCR is
    5-15s per figure region — only enable via --figure-table-ocr when the
    input is known to have raster-table figures (scanned PDFs).

    Some tables are rendered as raster images (scanned, or vector graphics
    that don't expose text). Page-geometry detectors miss these. This function:
      1. Crops the figure region at high DPI (300).
      2. Sends to glm-ocr with table-aware prompt.
      3. Parses the OCR output for markdown tables via detect_tables_in_text.

    Returns: list of detected tables (same format as detect_all_tables).
    """
    # Centralized gate (was previously scattered across callers).
    # Move check to the function itself so all callers respect the flag.
    if not _FIGURE_TABLE_OCR:
        return []
    import fitz

    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        clip = fitz.Rect(figure_bbox)
        pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72), clip=clip)
        import tempfile, os, base64, json as _json, urllib.request

        img_path = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
        pix.save(img_path)
        doc.close()
        with open(img_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        os.unlink(img_path)
        # Table-aware OCR prompt
        payload = {
            "model": "glm-ocr",
            "prompt": "Extract any tables from this image as markdown pipe-tables. If no table, respond 'NO TABLE'.",
            "images": [img_b64],
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 32768,
                "num_ctx": 32768,
                "repeat_penalty": 1.1,
            },
        }
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/generate",
            data=_json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            data = _json.loads(r.read())
        ocr_text = data.get("response", "") or ""
        if "NO TABLE" in ocr_text.upper() or len(ocr_text) < 50:
            return []
        tables = detect_tables_in_text(ocr_text)
        # Set bbox to figure region for dedup
        for t in tables:
            t["bbox"] = figure_bbox
            t["strategy"] = "figure_ocr"
        return tables
    except Exception as e:
        log.warning("detect_tables_in_figure failed p%s: %s", page_num, e)
        return []


def detect_tables_pdfplumber(
    pdf_path,
    page_num,
    min_rows=3,
    min_decimals_per_row=2,
    max_prose_words=25,
    # Empirical (188-PDF bench sweep, 2026-08-16): consecutive data-row
    # y-gaps — median 15, p75 27, p90 63. 15 split real tables whose rows
    # carry 2-line row labels (f45a pg8: P/R/F1 rows at gaps 27/24 → three
    # singleton clusters, all dropped, 5 bench tests lost). 30 keeps us
    # under the table-to-table separation floor (caption + spacing ≈ 40+)
    # while covering 2-line labels.
    y_cluster_gap=30,
):
    """High-precision table detector using pdfplumber char-level geometry.

    Replaces ``reconstruct_tables_fitz`` for born-digital pages. Finds clusters
    of "data rows" (lines with ≥N decimal numbers), filters each row to the
    actual table X-extent (dropping co-linear body text on the same Y-line),
    and returns a list of validated markdown tables.

    Detection logic (all non-ML, pure geometry + regex):
      1. Extract all words on the page with bbox.
      2. Bucket into Y-lines (3pt tolerance).
      3. A line is a "data row candidate" if it has ≥``min_decimals_per_row``
         decimal numbers AND ≤``max_prose_words`` total words (filters out
         prose paragraphs that happen to contain numbers).
      4. Cluster consecutive candidates (Y-gap ≤ ``y_cluster_gap``) into tables.
      5. For each cluster: compute table X-extent (leftmost label to rightmost
         decimal), filter each row to that X-extent (drops co-linear body
         text from the right column in side-by-side layouts).
      6. Validate: ≥``min_rows`` rows × ≥3 cols. If invalid, reject.
      7. Emit markdown table (header row inferred from first row + separator).

    Returns: list of {"bbox": (x0,y0,x1,y1), "markdown": str, "rows": list}.
    Empty list if no real tables found (body text never falsely promoted).
    """
    try:
        import pdfplumber
    except ImportError as e:
        raise RuntimeError(
            f"pdfplumber is a HARD dependency (per projectwide no-optional policy): {e}"
        ) from e
    from collections import defaultdict

    tables_found = []
    try:
        with pdfplumber.open(pdf_path) as doc:
            if page_num < 1 or page_num > len(doc.pages):
                return []
            page = doc.pages[page_num - 1]
            words = page.extract_words(use_text_flow=True, keep_blank_chars=False)
            if not words:
                return []
            # Bucket words into Y-lines (3pt snap)
            lines = defaultdict(list)
            for w in words:
                y_key = round(w["top"] / 3) * 3
                lines[y_key].append(w)
            sorted_ys = sorted(lines.keys())

            # Find data-row candidates
            candidates = []
            for y in sorted_ys:
                ws = sorted(lines[y], key=lambda w: w["x0"])
                decimals = [w for w in ws if re.match(r"^\d+\.\d+$", w["text"])]
                if len(decimals) < min_decimals_per_row:
                    continue
                if len(ws) > max_prose_words:
                    continue
                dec_x_min = min(w["x0"] for w in decimals)
                dec_x_max = max(w["x1"] for w in decimals)
                candidates.append(
                    {
                        "y": y,
                        "words": ws,
                        "decimals": decimals,
                        "dec_x_min": dec_x_min,
                        "dec_x_max": dec_x_max,
                    }
                )

            # Cluster consecutive candidates by Y-gap
            clusters = []
            current = []
            last_y = None
            for c in candidates:
                if last_y is not None and c["y"] - last_y > y_cluster_gap:
                    if len(current) >= min_rows:
                        clusters.append(current)
                    current = []
                current.append(c)
                last_y = c["y"]
            if len(current) >= min_rows:
                clusters.append(current)

            # For each cluster, compute table X-extent + extract rows
            for cluster in clusters:
                all_words = [w for c in cluster for w in c["words"]]
                all_decimals = [w for c in cluster for w in c["decimals"]]
                dec_left = min(w["x0"] for w in all_decimals)
                dec_right = max(w["x1"] for w in all_decimals)
                # Find leftmost label (non-decimal word immediately left of dec_left)
                labels = [
                    w
                    for w in all_words
                    if w["x1"] <= dec_left + 5 and not re.match(r"^\d+\.\d+$", w["text"])
                ]
                tbl_x_left = (
                    min((w["x0"] for w in labels), default=dec_left) if labels else dec_left
                )
                tbl_x_right = dec_right

                rows = []
                for c in cluster:
                    row_words = sorted(
                        [
                            w
                            for w in c["words"]
                            if w["x0"] >= tbl_x_left - 2 and w["x1"] <= tbl_x_right + 2
                        ],
                        key=lambda w: w["x0"],
                    )
                    if row_words:
                        rows.append([w["text"] for w in row_words])

                # HEADER ABSORPTION (iter-11, 2026-08-16): rows directly above
                # the first data row that overlap the table X-extent and are
                # short/non-decimal are column headers (P/R/F1-style column
                # labels sit INSIDE the decimal X-extent, so the label-pass
                # above never captured them — gold table tests check cell↔
                # header relations like 'R' above '0.85' and fail without
                # them). Walk upward from the first data row, absorbing up to
                # 2 header rows; SKIP rows that don't overlap the X-extent
                # (those are row-labels like 'discrete decoding', which sit
                # left of the table) without stopping the walk.
                first_y = cluster[0]["y"]
                absorbed_ys = []
                absorbed = 0
                for hy in [y for y in sorted_ys if first_y - y_cluster_gap <= y < first_y][::-1]:
                    if absorbed >= 2:
                        break
                    hw = sorted(lines[hy], key=lambda w: w["x0"])
                    inside = [
                        w for w in hw if w["x0"] >= tbl_x_left - 2 and w["x1"] <= tbl_x_right + 2
                    ]
                    if len(inside) < 0.6 * len(hw) or len(inside) < 2:
                        continue  # row-label or unrelated line — keep walking up
                    n_dec = sum(1 for w in inside if re.match(r"^\d+\.\d+$", w["text"]))
                    if n_dec > 1:
                        continue  # another data row belongs to a different cluster
                    rows.insert(0, [w["text"] for w in inside])
                    absorbed_ys.insert(0, hy)
                    absorbed += 1

                # SUBSCRIPT ATTACHMENT (iter-11): after absorbing header rows,
                # a following short row (≤3-char words, sparse) within ~8pt
                # below is the header's subscript/unit line ('F'+'1' → 'F1').
                # Attach each word to the x-nearest header cell ABOVE it
                # (x-overlap required) — never as a separate row: a '1' row
                if absorbed and rows:
                    sub_y = absorbed_ys[0]  # y of the TOPMOST absorbed header row
                    for sy in [y for y in sorted_ys if sub_y < y <= sub_y + 10]:
                        sw = sorted(lines[sy], key=lambda w: w["x0"])
                        cand = [
                            w
                            for w in sw
                            if w["x0"] >= tbl_x_left - 2
                            and w["x1"] <= tbl_x_right + 2
                            and len(w["text"].strip()) <= 3
                        ]
                        if not cand or len(cand) < 2:
                            continue
                        # Need header word x-centers to attach to
                        hw0 = sorted(lines[sub_y], key=lambda w: w["x0"])
                        hwords = [
                            w
                            for w in hw0
                            if w["x0"] >= tbl_x_left - 2 and w["x1"] <= tbl_x_right + 2
                        ]
                        for sw_word in cand:
                            for hw_word in hwords:
                                # +4pt right tolerance: subscripts render
                                # ADJACENT-right of the base glyph (F1), not
                                # strictly overlapping it.
                                if (
                                    sw_word["x0"] < hw_word["x1"] + 4
                                    and sw_word["x1"] > hw_word["x0"]
                                ):
                                    hw_word["text"] = hw_word["text"] + sw_word["text"]
                                    break
                        # Rebuild header row with attached subscripts and DROP
                        # the standalone short-token row (it is now part of
                        # row 0 — leaving it would break cell↔header adjacency).
                        rows[0] = [w["text"] for w in hwords]
                        if (
                            len(rows) > 1
                            and rows[1]
                            and all(len(str(c).strip()) <= 3 for c in rows[1] if c)
                        ):
                            rows.pop(1)
                        break

                if len(rows) < min_rows:
                    continue
                # Validate: ≥3 cols in at least 50% of rows
                col_counts = [len(r) for r in rows]
                if max(col_counts) < 3:
                    continue
                adequate = sum(1 for c in col_counts if c >= 3)
                if adequate < len(rows) * 0.5:
                    continue

                # Build markdown table
                header = [_flat_cell(c) for c in rows[0]]
                body = rows[1:] if len(rows) > 1 else []
                md_lines = ["| " + " | ".join(header) + " |"]
                for r in body:
                    # Pad/truncate row to match header column count
                    r_padded = (r + [""] * len(header))[: len(header)]
                    md_lines.append("| " + " | ".join(_flat_cell(c) for c in r_padded) + " |")
                md_lines.append("")  # blank line after table

                tables_found.append(
                    {
                        "bbox": (
                            tbl_x_left,
                            cluster[0]["y"],
                            tbl_x_right,
                            cluster[-1]["y"] + 10,
                        ),
                        "markdown": "\n".join(md_lines),
                        "rows": rows,
                        "strategy": "pdfplumber_text",
                    }
                )
    except Exception as e:
        log.warning("detect_tables_pdfplumber failed p%s: %s", page_num, e)
        return []
    return tables_found


def reconstruct_tables_fitz(pdf_path, page_num, existing_text):
    """Comprehensive table reconstruction with Cell objects and confidence.

    Two detection strategies: fitz find_tables + numeric density scan.
    Column detection: alignment graph (primary) + DBSCAN (fallback).
    Grid optimization: EM (boundary refinement).
    Cell objects carry confidence, bbox, font metadata.
    Decimal repair reduces cell confidence to 0.9 (repair was applied).
    """
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return detect_tables(existing_text)

    try:
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
    except Exception as e:
        log.warning("table recon: fitz open failed: %s", e)
        return detect_tables(existing_text)

    try:
        # Phase 1: TABLE DETECTION
        table_regions = []

        try:
            for hit in native_find_tables(pdf_path, page_num, strategy="text"):
                table_regions.append(("fitz", hit["bbox"]))
        except Exception as e:
            log.debug("reconstruct_tables_fitz: Exception degraded: %s", e)

        numeric_regions = _detect_table_regions_numeric(pd)
        for region in numeric_regions:
            is_dup = False
            for _, existing_bbox in table_regions:
                overlap_x = max(
                    0,
                    min(region[2], existing_bbox[2]) - max(region[0], existing_bbox[0]),
                )
                overlap_y = max(
                    0,
                    min(region[3], existing_bbox[3]) - max(region[1], existing_bbox[1]),
                )
                if overlap_x * overlap_y > 0.5 * (region[2] - region[0]) * (region[3] - region[1]):
                    is_dup = True
                    break
            if not is_dup:
                table_regions.append(("numeric", region))

        if not table_regions:
            doc.close()
            return detect_tables(existing_text)

        # Phase 2: Reconstruct each table region
        table_markdowns = []

        for source, tbl_bbox in table_regions:
            result = _build_table_grid(pd, tbl_bbox)
            if result is None:
                continue

            cells, n_rows, n_cols = result

            accepted, cells = _quality_gate(cells)
            if not accepted:
                continue

            # Decimal repair: reduces confidence to 0.9 on repaired cells
            for row in cells:
                for cell in row:
                    ct = cell.text if hasattr(cell, "text") else str(cell)
                    if ct and re.search(r"\d", ct):
                        repaired = re.sub(r"(\d)[\u00b0\ufffd\u00b7\u00d7](\d)", r"\1.\2", ct)
                        if repaired != ct and hasattr(cell, "text"):
                            cell.text = repaired
                            cell.confidence = 0.9  # repair applied — slightly less certain

            md = _cells_to_markdown(cells)
            table_markdowns.append(md)

        doc.close()

        if not table_markdowns:
            return detect_tables(existing_text)

        result = existing_text.rstrip()
        for md in table_markdowns:
            result += "\n\n" + md

        return result
    except Exception as e:
        log.warning("table reconstruction failed: %s", e)
        try:
            doc.close()
        except Exception as e:
            log.debug("reconstruct_tables_fitz: Exception degraded: %s", e)
        return detect_tables(existing_text)


# =============================================================================


def detect_tables(text):
    """Detect aligned-column tables and convert to markdown pipe tables.

    Conservative: requires >=2 consecutive lines with >=2 column-like gaps
    (≥3 spaces) or pipe separators. Code blocks (indented ≥4 spaces) are
    skipped to avoid mangling them.
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]
        # Skip indented code blocks
        if line.startswith("    ") or line.startswith("\t"):
            out.append(line)
            i += 1
            continue
        # Check if this line looks like a table row
        gap_count = len(TABLE_GAP_RE.findall(line))
        has_pipe = "|" in line and line.strip().startswith("|")
        looks_tabular = (gap_count >= 2) or has_pipe
        if not looks_tabular:
            out.append(line)
            i += 1
            continue
        # Collect consecutive tabular lines
        block = [line]
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if nxt.startswith("    ") or nxt.startswith("\t") or not nxt.strip():
                break
            nxt_gaps = len(TABLE_GAP_RE.findall(nxt))
            nxt_pipe = "|" in nxt and nxt.strip().startswith("|")
            if (nxt_gaps >= 2) or nxt_pipe:
                block.append(nxt)
                j += 1
            else:
                break
        if len(block) < 2:
            out.append(line)
            i += 1
            continue
        # Convert block to markdown table
        table_rows = [_line_to_table_cells(b) for b in block]
        # All rows must have same column count to form a valid table
        col_counts = {len(r) for r in table_rows}
        if len(col_counts) != 1 or col_counts.pop() < 2:
            out.extend(block)
            i = j
            continue
        # Emit markdown table: header row + body (det44: sepless)
        header = table_rows[0]
        body = table_rows[1:]
        out.append("| " + " | ".join(_flat_cell(c) for c in header) + " |")
        for row in body:
            out.append("| " + " | ".join(_flat_cell(c) for c in row) + " |")
        out.append("")  # blank line after table
        i = j
    return "\n".join(out)


def find_table_bbox(pdf_path, page_num):
    """Detect table regions via pdftotext -tsv bbox analysis.

    Returns list of (x, y, w, h) bboxes in PDF points (top-left origin), or
    empty list if no tables detected.

    Heuristic: scan word bboxes (level-5 rows from `pdftotext -tsv`), cluster
    by y-coordinate into lines, then find runs of lines that share ≥3
    "column x-positions" (alignment). Returns the bbox enclosing the longest
    such run.

    Use case: when SOTA routing flags a page as `fragmented_table`, OCR just
    the detected table region instead of the whole page. Faster + preserves
    surrounding good text.
    """
    try:
        proc = subprocess.run(
            [
                "pdftotext",
                "-tsv",
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
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ):
        return []

    # Parse word-level rows (level == 5)
    words = []  # (left, top, width, height, text)
    for line in proc.stdout.split("\n")[1:]:  # skip header
        parts = line.split("\t")
        if len(parts) < 12:
            continue
        try:
            level = int(parts[0])
            if level != 5:
                continue
            left = float(parts[5])
            top = float(parts[6])
            width = float(parts[7])
            height = float(parts[8])
            text = parts[11]
            if text.startswith("###"):
                continue
            words.append((left, top, width, height, text))
        except (ValueError, IndexError):
            continue

    if len(words) < 6:
        return []  # not enough words to form a table

    # Cluster words into lines by top-y (within 5pt tolerance)
    words.sort(key=lambda w: (w[1], w[0]))
    lines = []  # list of [(left, top, w, h, text), ...]
    current_line = [words[0]]
    for w in words[1:]:
        if abs(w[1] - current_line[0][1]) <= 5:
            current_line.append(w)
        else:
            lines.append(current_line)
            current_line = [w]
    lines.append(current_line)

    # For each line, compute column x-positions (left edges rounded to 5pt)
    def col_positions(line):
        return sorted(set(int(w[0] // 5) * 5 for w in line))

    # Find runs of consecutive lines that share ≥3 column positions
    # (table signal: multiple rows aligned to same columns)
    table_runs = []
    current_run = []
    for i, line in enumerate(lines):
        cols = col_positions(line)
        if len(cols) >= 3 and len(line) >= 3:
            current_run.append(line)
        else:
            if len(current_run) >= 3:
                table_runs.append(current_run)
            current_run = []
    if len(current_run) >= 3:
        table_runs.append(current_run)

    if not table_runs:
        return []

    # Take the longest run, compute enclosing bbox
    longest = max(table_runs, key=len)
    all_words = [w for line in longest for w in line]
    min_x = min(w[0] for w in all_words)
    min_y = min(w[1] for w in all_words)
    max_x = max(w[0] + w[2] for w in all_words)
    max_y = max(w[1] + w[3] for w in all_words)
    # Add small padding (10pt each side)
    pad = 10
    return [(min_x - pad, min_y - pad, (max_x - min_x) + 2 * pad, (max_y - min_y) + 2 * pad)]


__all__ = [
    "Cell",
    "_detect_table_regions_numeric",
    "_optimize_grid_em",
    "_build_table_grid",
    "detect_multirow_header",
    "infer_column_types",
    "build_table_markdown",
    "detect_tables_in_text",
    "detect_text_only_tables",
    "detect_tables_lines_strict",
    "detect_tables_camelot",
    "detect_tables_in_figure",
    "detect_tables_pdfplumber",
    "reconstruct_tables_fitz",
    "detect_tables",
    "find_table_bbox",
]
