"""Table structure — Table dataclass + cell classification + spanning inference.

Extracted from pdf_ocr.py during R9 final slim-down (2026-08-12).
"""

from __future__ import annotations
import re, logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state

# EXP-1.5: control-char glyph noise from unmapped symbol fonts (bullets,
# diagram labels) leaks into cells when tables are re-extracted from the
# PDF — independent of the page-text scrub. Lowest tables-lane module, so
# the scrub lives here and detector/agents inherit it. \t\n\r kept.
_CTRL_CELL_SCRUB_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


def scrub_cell_text(value):
    """Strip control glyphs from a single string (None-safe, passthrough)."""
    if isinstance(value, str) and _CTRL_CELL_SCRUB_RE.search(value):
        return _CTRL_CELL_SCRUB_RE.sub("", value)
    return value


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
    for n in ("Cell",):
        if hasattr(p, n):
            g[n] = getattr(p, n)
    _CONSTS_LOADED = True


@dataclass
class Table:
    """Structured table with preserved header hierarchy.

    Internally uses List[List[Cell]] — not plain strings.
    Each renderer (to_markdown, to_html, to_json) serializes differently.

    NOTE (2026-08-16 iter-9): the @dataclass decorator was dropped when this
    class was extracted from pdf_ocr.py (R9 slim-down) — Table() raised
    TypeError, silently killing header-aware table rendering (measured:
    tables emitted without header rows, 117 REL-broken bench tests).
    """

    header_rows: list  # List[List[Cell]]
    body_rows: list  # List[List[Cell]]
    n_cols: int
    bbox: tuple = (0, 0, 0, 0)  # table bounding box

    @staticmethod
    def _ct(cell):
        """Extract text from Cell or string (backward compatible)."""
        return scrub_cell_text(cell.text if hasattr(cell, "text") else str(cell))

    @staticmethod
    def _flat_md_cell(c):
        """Single-line, pipe-safe cell text (det44 sepless emission)."""
        return " ".join(str(c).replace("|", "/").split())

    def to_markdown(self):
        """Render as markdown pipe table. Flattens multi-row headers."""
        if not self.header_rows:
            return ""
        if len(self.header_rows) > 1:
            header = []
            for col in range(self.n_cols):
                parts = [self._ct(r[col]).strip() if col < len(r) else "" for r in self.header_rows]
                parts = [p for p in parts if p]
                header.append(" ".join(parts))
        else:
            header = [self._ct(c) for c in self.header_rows[0]]

        body = [[self._ct(c) for c in r] for r in self.body_rows]
        if not body:
            body = [header]

        max_cols = max(len(header), max(len(r) for r in body))
        while len(header) < max_cols:
            header.append("")
        for r in body:
            while len(r) < max_cols:
                r.append("")

        md = "|" + "|".join(self._flat_md_cell(h) for h in header) + "|"
        for row in body:
            md += "\n|" + "|".join(self._flat_md_cell(c) for c in row) + "|"
        return md + "\n"  # trailing blank line (det44: sepless emission)

    def to_html(self):
        """Render as HTML table. Preserves multi-row headers via <thead>."""
        html = "<table>\n"
        if self.header_rows:
            html += "<thead>\n"
            for row in self.header_rows:
                html += "<tr>" + "".join(f"<th>{self._ct(c)}</th>" for c in row) + "</tr>\n"
            html += "</thead>\n"
        html += "<tbody>\n"
        for row in self.body_rows:
            html += "<tr>" + "".join(f"<td>{self._ct(c)}</td>" for c in row) + "</tr>\n"
        html += "</tbody>\n</table>"
        return html


def _detect_table_structure(cells):
    """Detect header/body boundary and return structured Table.

    Header rows: rows with no (or few) numeric values (titles, column labels).
    Body rows: rows with numeric data (measurements, values).

    Returns Table object with header_rows and body_rows preserved separately.
    """

    def _txt(cell):
        return cell.text if hasattr(cell, "text") else str(cell)

    # Find first data row: first row with ≥2 cells containing numeric values
    header_end = 0
    for i, row in enumerate(cells):
        num_count = sum(1 for c in row if _txt(c) and re.search(r"\d+\.\d+|\d{2,}", _txt(c)))
        if num_count >= 2:
            header_end = i
            break

    if header_end > 0:
        # Off-by-one fix (iter-9): header_end is the FIRST DATA row index —
        # the header is everything BEFORE it. The old `+1` fused the first
        # data row into the header ('|Trial TDF2|Study Population 1,219…').
        header_rows = cells[:header_end]
        body_rows = cells[header_end:]
    else:
        header_rows = [cells[0]]
        body_rows = cells[1:]

    if not body_rows:
        body_rows = [list(header_rows[0])]

    n_cols = max(len(r) for r in header_rows + body_rows)

    return Table(header_rows=header_rows, body_rows=body_rows, n_cols=n_cols)


def classify_cell(content):
    """Classify a table cell's content type for structure validation.

    Returns one of: 'number', 'percent', 'date', 'scientific', 'empty',
    'identifier', 'text'.

    Used by:
      - validate_table_structure: detect column type mismatches
      - infer_spanning_cells: group label detection
      - rebuild headers with semantic type info
    """
    if not content or not content.strip():
        return "empty"
    s = content.strip()
    # Empty-like placeholders
    if s in ("-", "n.d.", "nd", "b.d.l.", "—", "–", "n/a", "N/A", "n/c", "—"):
        return "empty"
    # Percent
    if s.endswith("%") and not s.startswith("%"):
        try:
            float(s[:-1].strip())
            return "percent"
        except ValueError as e:
            log.debug("classify_cell: ValueError degraded: %s", e)
    # Scientific notation: 1.23e-4, 1.23E+5
    if re.match(r"^[+-]?\d+\.?\d*[eE][+-]?\d+$", s):
        return "scientific"
    # Date patterns: YYYY-MM-DD, DD/MM/YYYY, Mon YYYY, etc.
    if re.match(r"^\d{4}[-/]\d{1,2}([-/]\d{1,2})?$", s):
        return "date"
    if re.match(r"^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}$", s):
        return "date"
    if re.match(r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}$", s, re.I):
        return "date"
    # Pure number (decimal, integer, with ± or ±uncertainty)
    # Strip leading ±, trailing ±uncertainty (e.g., "5.2 ± 0.1")
    cleaned = re.sub(r"^[±~≈]\s*", "", s)
    cleaned = re.sub(r"\s*[±~≈]\s*\d+\.?\d*$", "", cleaned)
    # Allow thousands separators, leading ±, parentheses for negative
    cleaned = cleaned.replace(",", "").replace("(", "-").replace(")", "")
    try:
        float(cleaned)
        return "number"
    except ValueError as e:
        log.debug("classify_cell: ValueError degraded: %s", e)
    # Identifier: short alphanumeric code (sample IDs like "MPY30a", "26hss", "S1297")
    if re.match(r"^[A-Za-z]{0,3}\d+[A-Za-z]{0,3}$", s) and len(s) <= 12:
        return "identifier"
    # Default: text
    return "text"


def infer_spanning_cells(rows):
    """Detect rowspan/colspan patterns in detected table.

    Rowspan signature: same value in same column across N consecutive rows
    (group label spanning multiple data rows). Returns list of (row_idx,
    col_idx, row_span) tuples for cells that should be rowspan-merged.

    Colspan signature: a cell value followed by empty cells in the same row
    (group label spanning multiple columns). Returns (row_idx, col_idx,
    col_span) tuples.
    """
    if not rows:
        return [], []
    rowspans = []  # (row, col, span)
    colspans = []  # (row, col, span)
    # Detect rowspans: same value vertically across ≥2 rows
    n_cols = max(len(r) for r in rows)
    for ci in range(n_cols):
        run_start = None
        run_value = None
        for ri, r in enumerate(rows):
            v = r[ci].strip() if ci < len(r) and r[ci] else ""
            if v and v == run_value:
                # Continues run
                continue
            else:
                # End previous run
                if run_start is not None and ri - run_start >= 2 and run_value:
                    # Only count as rowspan if value is non-numeric (text label)
                    if classify_cell(run_value) in ("text", "identifier"):
                        rowspans.append((run_start, ci, ri - run_start))
                run_start = ri if v else None
                run_value = v if v else None
        # Handle run extending to last row
        if run_start is not None and len(rows) - run_start >= 2 and run_value:
            if classify_cell(run_value) in ("text", "identifier"):
                rowspans.append((run_start, ci, len(rows) - run_start))
    # Detect colspans: value followed by ≥2 empty cells in same row
    for ri, r in enumerate(rows):
        ci = 0
        while ci < len(r):
            v = r[ci].strip() if r[ci] else ""
            if not v:
                ci += 1
                continue
            # Check how many empty cells follow
            span = 1
            while ci + span < len(r) and (not r[ci + span] or not r[ci + span].strip()):
                span += 1
            if span >= 3 and classify_cell(v) == "text":
                colspans.append((ri, ci, span))
            ci += span
    return rowspans, colspans


def validate_table_structure(rows, strategy="unknown"):
    """Validate detected table structure. Returns (is_valid, issues).

    Checks:
      1. Column count consistency (≥80% rows have same col count)
      2. Numeric columns have ≥50% numeric cells (catches fake tables)
      3. No body-text-as-table (cells with >15 words + low decimal ratio)
      4. At least 1 column should be "informative" (not all empty/text)

    Returns (True, []) if valid, (False, [issue_descriptions]) if not.
    """
    if not rows or len(rows) < 2:
        return False, ["too_few_rows"]
    issues = []
    # Check 1: column count consistency
    col_counts = [len(r) for r in rows]
    max_cols = max(col_counts)
    consistent = sum(1 for c in col_counts if c == max_cols)
    if consistent / len(rows) < 0.60:
        issues.append("inconsistent_column_count")
    # Check 2: at least 1 numeric column (real data has numbers)
    # infer_column_types lives in sibling detector module (circular at
    # module-load time). Import lazily here.
    from .detector import infer_column_types

    col_types = infer_column_types(rows)
    has_numeric = any(t in ("numeric", "percent", "scientific") for t in col_types)
    has_text_header = any(t in ("text", "identifier") for t in col_types)
    if not has_numeric and not has_text_header:
        issues.append("no_informative_columns")
    # Check 3: body-text-as-table guard (RATE-based, iter-11 2026-08-16).
    # Old single-cell veto (any cell >15 low-numeric words → reject) killed
    # legitimate annotation-heavy tables: a674...pg120 is a real ruled 6×4
    # grid where the "Note" column carries 17-35-word prose per cell — one
    # long note rejected a structurally perfect table (bench: whole PDF
    # scored 0, 5 table tests). Prose PROMOTION is when long cells dominate
    # the grid; a minority of prose cells is a real annotation column.
    long_cells = 0
    total_cells = 0
    for r in rows:
        for c in r:
            if not c:
                continue
            total_cells += 1
            words = c.split()
            if len(words) > 15:
                decimals = sum(1 for w in words if classify_cell(w) == "number")
                if decimals / max(len(words), 1) < 0.30:
                    long_cells += 1
    if total_cells and long_cells / total_cells > 0.50:
        issues.append("body_text_as_table")
    # Check 4: empty-cell ratio (garbage tables have ≥80% empty cells)
    total_cells = sum(len(r) for r in rows)
    empty_cells = sum(1 for r in rows for c in r if not c or not c.strip())
    if total_cells > 0 and empty_cells / total_cells > 0.80:
        issues.append("mostly_empty_cells")
    return (len(issues) == 0), issues


def _width_raggedness(rows):
    """Row-width raggedness: 0 = every row has identical column count.

    A structure-quality score for dedup tiebreaks — uniform-width rows
    (lines_strict) survive validate_table_structure while ragged ones
    (pdfplumber header split 'Std.|Dev' vs 6-wide body) get rejected
    downstream as inconsistent_column_count, killing the whole table.
    """
    widths = [len(r) for r in rows if r is not None]
    if not widths:
        return 0
    return max(widths) - min(widths)


def _mojibake_cells(t):
    """Count cells containing broken-CMap artifacts ('(cid:NN)')."""
    n = 0
    for r in t.get("rows", []):
        for c in r:
            if "(cid:" in str(c):
                n += 1
    return n


def _grid_words(rows):
    """Total whitespace-separated words across all cells (dedup tiebreak)."""
    return sum(len(str(c).split()) for r in rows or [] for c in (r or []) if str(c).strip())


def _n_cols(rows):
    """Modal row width — the grid's effective column count (dedup tiebreak)."""
    from collections import Counter

    lens = [len(r) for r in rows or [] if r]
    return Counter(lens).most_common(1)[0][0] if lens else 0


def _dedup_tables(tables_list, return_suppressed=False):
    """Remove duplicate tables across strategies (bbox overlap >50%).

    When multiple strategies detect the same region, keep the one with
    more rows (higher recall). With return_suppressed=True, also returns
    every suppressed candidate (iter-67: when the dedup winner later dies
    in build_table_markdown, the caller can reinstate the best suppressed
    alternative instead of emitting nothing for the region).
    """
    if not tables_list:
        return ([], []) if return_suppressed else []
    deduped = []
    suppressed = []
    for t in tables_list:
        bbox = t["bbox"]
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        is_dup = False
        for i, kept in enumerate(deduped):
            k_bbox = kept["bbox"]
            ix0 = max(bbox[0], k_bbox[0])
            iy0 = max(bbox[1], k_bbox[1])
            ix1 = min(bbox[2], k_bbox[2])
            iy1 = min(bbox[3], k_bbox[3])
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            intersection = (ix1 - ix0) * (iy1 - iy0)
            kept_area = (k_bbox[2] - k_bbox[0]) * (k_bbox[3] - k_bbox[1])
            # Containment-aware ratio (2026-08-15): intersection/min-area.
            # max-area ratio missed the wheat-page case — positional 18-row
            # table CONTAINING three 5-row pdfplumber sub-tables deduped at
            # ~0.07 → 4 overlapping tables in output (content inflation,
            # downstream guards rejecting correct region-routed tables).
            overlap_ratio = intersection / max(min(area, kept_area), 1)
            if overlap_ratio > 0.6:
                if len(t["rows"]) > len(kept["rows"]):
                    deduped[i] = t
                    suppressed.append(kept)
                elif len(t["rows"]) == len(kept["rows"]) and _width_raggedness(
                    t["rows"]
                ) < _width_raggedness(kept["rows"]):
                    # STRUCTURE tiebreak (2026-08-19, iter-57): equal row
                    # counts, challenger strictly less ragged (uniform row
                    # widths). pdfplumber split 'Std. Dev' across header
                    # cells (7-wide header vs 6-wide body → downstream
                    # inconsistent_column_count rejection, empty markdown)
                    # while lines_strict read the same region uniformly —
                    # first-seen kept the broken one and the table VANISHED
                    # ('No tables found', 240cc9b2 4 gold tests).
                    deduped[i] = t
                    suppressed.append(kept)
                elif (
                    len(t["rows"]) == len(kept["rows"])
                    and _grid_words(t["rows"]) >= _grid_words(kept["rows"])
                    and _n_cols(t["rows"]) < _n_cols(kept["rows"])
                ):
                    # CELL-INTEGRITY tiebreak (iter-105, 8ec7ac2f): equal
                    # rows, challenger carries ALL the words in FEWER
                    # columns — text_only's word-per-cell shred ('Freight |
                    # Transport | Association' = 3 columns of single words)
                    # loses to positional's whole-cell grid. A shred has
                    # MORE filled cells, so fills alone invert; words-
                    # preserved vs column-count is the true signal.
                    deduped[i] = t
                    suppressed.append(kept)
                elif (
                    len(t["rows"]) >= 0.8 * len(kept["rows"])
                    and _mojibake_cells(t) < _mojibake_cells(kept)
                    and _mojibake_cells(t) == 0
                ):
                    # Text-quality tiebreak (2026-08-18): near-equal rows,
                    # strictly cleaner cell text. pdfplumber reads table
                    # renders through broken CMaps ('(cid:126)0,0(cid:33)'
                    # cells, measured 53d3f304 pg3) while the word-layer
                    # strategies carry the true text.
                    deduped[i] = t
                    suppressed.append(kept)
                is_dup = True
                if not any(d is t for d in deduped):
                    suppressed.append(t)
                break
        if not is_dup:
            deduped.append(t)
    return (deduped, suppressed) if return_suppressed else deduped


def _line_to_table_cells(line):
    """Split a tabular line into cells by multi-space gaps or pipes."""
    stripped = line.strip()
    if stripped.startswith("|"):
        # Pipe table — split on | and drop empty edges
        parts = [c.strip() for c in stripped.split("|")]
        if parts and not parts[0]:
            parts = parts[1:]
        if parts and not parts[-1]:
            parts = parts[:-1]
        return parts
    # Multi-space gap table
    return [c.strip() for c in TABLE_GAP_RE.split(stripped) if c.strip()]


# =============================================================================
# SOTA per-page routing — synthesized from Marker + unstructured.io research,
# validated on 90 measurements across 40 reference PDFs.
# =============================================================================

# Tuned thresholds (production values)

__all__ = [
    "Table",
    "_detect_table_structure",
    "classify_cell",
    "infer_spanning_cells",
    "validate_table_structure",
    "_dedup_tables",
    "_line_to_table_cells",
]
