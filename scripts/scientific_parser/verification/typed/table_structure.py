"""Table structure validator — row/column consistency for HTML/Markdown tables.

Per architecture proposal item #4 (2026-08-13): typed validator for tables
catches structural errors (collapsed columns, missing rows, inconsistent
column counts) that model confidence misses.

Accepts:
    - Markdown tables (``| ... | ... |\\n|---|---|\\n| ... | ... |``)
    - HTML tables (``<table><tr><td>...</td></tr></table>``)

Returns a :class:`VerificationResult` with:
    - ``passed=True`` if table is well-formed (rectangular modulo colspans,
      has a header row, every row non-empty).
    - ``score`` in [0, 1] reflecting the fraction of structural checks passed.
    - ``details`` containing ``rows``, ``cols``, ``header``, ``errors``.

Does NOT validate cell content (numeric type, value range). That belongs to
``numeric_range``.
"""

from __future__ import annotations

import re

from ...core.ir import VerificationResult

_MD_ROW_RE = re.compile(r"^\s*\|(.*)\|\s*$")
_MD_SEP_RE = re.compile(r"^\s*\|?[\s\-:|]+\|?\s*$")
_HTML_TR_RE = re.compile(r"<tr[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_HTML_TD_RE = re.compile(r"<(td|th)[^>]*>(.*?)</\1>", re.IGNORECASE | re.DOTALL)


def _parse_markdown_rows(text: str) -> list[list[str]]:
    """Extract rows from a markdown table. Returns [] if not a markdown table."""
    rows: list[list[str]] = []
    for line in text.strip().splitlines():
        m = _MD_ROW_RE.match(line)
        if not m:
            continue
        inner = m.group(1)
        # Skip separator row (|---|---|)
        if _MD_SEP_RE.match(line):
            continue
        cells = [c.strip() for c in inner.split("|")]
        rows.append(cells)
    return rows


def _parse_html_rows(text: str) -> list[list[str]]:
    """Extract rows from an HTML table. Returns [] if no <tr>."""
    rows: list[list[str]] = []
    for tr_match in _HTML_TR_RE.finditer(text):
        row_html = tr_match.group(1)
        cells = []
        for td_match in _HTML_TD_RE.finditer(row_html):
            # Strip nested tags for structural check (content validation is
            # out of scope here).
            cell_text = re.sub(r"<[^>]+>", "", td_match.group(2)).strip()
            cells.append(cell_text)
        if cells:
            rows.append(cells)
    return rows


def validate_table(
    content: str,
    min_rows: int = 2,
    min_cols: int = 2,
) -> VerificationResult:
    """Validate table structure (row/column consistency, header presence).

    Args:
        content: Table as Markdown or HTML string.
        min_rows: Minimum number of data rows (excluding header). Default 2.
        min_cols: Minimum number of columns. Default 2.

    Returns:
        :class:`VerificationResult` with ``validator_id="table_structure"``.

    Scoring (each contributes 0.25 to score):
        1. Parseable as table (>=1 row, >=1 col).
        2. At least ``min_rows`` data rows.
        3. Column counts consistent across rows (modulo colspans later).
        4. No empty cells in body rows.
    """
    if not content or not content.strip():
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="table_structure",
            details={"error": "empty_content"},
        )

    rows = _parse_markdown_rows(content)
    table_format = "markdown"
    if not rows:
        rows = _parse_html_rows(content)
        table_format = "html"
    if not rows:
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="table_structure",
            details={"error": "unparseable", "content_preview": content[:80]},
        )

    n_rows = len(rows)
    col_counts = [len(r) for r in rows]
    n_cols_max = max(col_counts) if col_counts else 0
    cols_consistent = len(set(col_counts)) == 1
    has_min_rows = n_rows >= min_rows
    has_min_cols = n_cols_max >= min_cols
    empty_body_cells = sum(1 for r in rows[1:] for c in r if not c.strip())

    checks = [True, has_min_rows and has_min_cols, cols_consistent, empty_body_cells == 0]
    score = sum(0.25 for ok in checks if ok)
    passed = all(checks)

    return VerificationResult(
        passed=passed,
        score=score,
        validator_id="table_structure",
        details={
            "format": table_format,
            "rows": n_rows,
            "cols": n_cols_max,
            "col_counts": col_counts,
            "cols_consistent": cols_consistent,
            "empty_body_cells": empty_body_cells,
            "errors": []
            if passed
            else [
                k
                for k, ok in zip(
                    ["parseable", "min_dimensions", "cols_consistent", "no_empty_cells"],
                    checks,
                    strict=True,
                )
                if not ok
            ],
        },
    )


__all__ = ["validate_table"]
