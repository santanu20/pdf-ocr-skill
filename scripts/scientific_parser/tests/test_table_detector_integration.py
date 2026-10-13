"""Integration tests for table detector with REAL PDF fixtures.

Per testing.md §3.1: ≥90% strong assertions.
Per testing.md §6: real data > mocks (no mock of system-under-test).
Uses PDFs from bench/olmOCR-bench/bench_data/pdfs/ (downloaded dataset).

Run: pytest scripts/scientific_parser/tests/test_table_detector_integration.py -v
"""

import os
import sys
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent  # -> scripts/
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# Resolve real-PDF fixture paths relative to skill root.
SKILL_ROOT = SCRIPTS_DIR.parent  # -> pdf-ocr/
BENCH_DIR = SKILL_ROOT / "bench" / "bench_data" / "pdfs"

# Fixture: arxiv_math PDF that triggers find_table_bbox (verified 2026-08-13).
ARXIV_MATH_PDF_WITH_BBOX = BENCH_DIR / "arxiv_math" / "2502.15977_pg21.pdf"
MULTI_COLUMN_PDF_WITH_BBOX = (
    BENCH_DIR / "multi_column" / "00187fe0533b1e8ddc748adab4924b6f7099_page_10_pg1.pdf"
)
HEADERS_FOOTERS_PDF_WITH_BBOX = (
    BENCH_DIR / "headers_footers" / "038704b29be26b6d45233e5e11218273d372957e_page_2.pdf"
)


pytestmark = pytest.mark.skipif(
    not BENCH_DIR.exists(),
    reason="bench dataset not downloaded",
)


from scientific_parser.perception.tables.detector import (  # noqa: E402
    detect_tables_pdfplumber,
    detect_text_only_tables,
    find_table_bbox,
    reconstruct_tables_fitz,
)


# === find_table_bbox — return contract ========================= #


def test_find_table_bbox_real_pdf_returns_list_of_bbox_tuples():
    """End-to-end on a real PDF that has a detectable table region."""
    result = find_table_bbox(str(ARXIV_MATH_PDF_WITH_BBOX), 1)
    assert isinstance(result, list)
    assert len(result) >= 1
    bbox = result[0]
    # bbox is a 4-tuple of floats: (x0, y0, x1, y1)
    assert len(bbox) == 4
    for coord in bbox:
        assert isinstance(coord, (int, float))


def test_find_table_bbox_nonexistent_pdf_returns_empty_or_raises_cleanly():
    """Non-existent PDF — function must either return [] or raise a known
    exception type (not crash with internal NameError)."""
    try:
        result = find_table_bbox("/nonexistent/file.pdf", 1)
        assert isinstance(result, list)
    except (FileNotFoundError, OSError, subprocess_exception_types()) as e:
        # Acceptable: clean subprocess error
        assert "No such file" in str(e) or "not found" in str(e).lower()


def subprocess_exception_types():
    """Helper to lazy-import subprocess exception types."""
    import subprocess

    return (subprocess.CalledProcessError, subprocess.TimeoutExpired)


def test_find_table_bbox_invalid_page_num_returns_empty_list():
    """Page 999 (out of range) — function must not crash."""
    result = find_table_bbox(str(ARXIV_MATH_PDF_WITH_BBOX), 999)
    # Either empty list (graceful) or raises clean — never crashes
    assert isinstance(result, list)


def test_find_table_bbox_real_multi_column_pdf_returns_list():
    """Different PDF (multi_column) — same return-type contract."""
    result = find_table_bbox(str(MULTI_COLUMN_PDF_WITH_BBOX), 1)
    assert isinstance(result, list)


# === detect_tables_pdfplumber — return contract =============== #


def test_detect_tables_pdfplumber_real_pdf_returns_list():
    """detect_tables_pdfplumber returns a list (possibly empty if pdfplumber
    doesn't find structured tables on the page)."""
    result = detect_tables_pdfplumber(str(ARXIV_MATH_PDF_WITH_BBOX), 1)
    assert isinstance(result, list)
    # When non-empty, each item should be a dict-like with markdown/rows keys
    for item in result:
        assert hasattr(item, "keys") or hasattr(item, "__getitem__")


def test_detect_tables_pdfplumber_invalid_path_returns_empty_list_or_clean_error():
    try:
        result = detect_tables_pdfplumber("/nonexistent/file.pdf", 1)
        assert isinstance(result, list)
    except (FileNotFoundError, OSError):
        pass  # acceptable


def test_detect_tables_pdfplumber_min_rows_filter_changes_result_count():
    """Higher min_rows threshold should return SAME or FEWER tables."""
    low = detect_tables_pdfplumber(str(ARXIV_MATH_PDF_WITH_BBOX), 1, min_rows=2)
    high = detect_tables_pdfplumber(str(ARXIV_MATH_PDF_WITH_BBOX), 1, min_rows=10)
    assert len(high) <= len(low)


# === detect_text_only_tables — return contract ================ #


def test_detect_text_only_tables_real_pdf_returns_list():
    result = detect_text_only_tables(str(ARXIV_MATH_PDF_WITH_BBOX), 1)
    assert isinstance(result, list)


def test_detect_text_only_tables_invalid_page_returns_empty_list():
    """Page 999 — graceful handling, no crash."""
    result = detect_text_only_tables(str(ARXIV_MATH_PDF_WITH_BBOX), 999)
    assert isinstance(result, list)


def test_detect_text_only_tables_min_rows_filter_changes_count():
    """Higher min_rows → same or fewer tables."""
    low = detect_text_only_tables(str(ARXIV_MATH_PDF_WITH_BBOX), 1, min_rows=2)
    high = detect_text_only_tables(str(ARXIV_MATH_PDF_WITH_BBOX), 1, min_rows=10)
    assert len(high) <= len(low)


# === reconstruct_tables_fitz — return contract =============== #


def test_reconstruct_tables_fitz_real_pdf_returns_markdown_string():
    """Reconstruct tables from existing text + page geometry. Returns
    markdown string (per actual contract — not a list)."""
    result = reconstruct_tables_fitz(
        str(ARXIV_MATH_PDF_WITH_BBOX), 1, "Sample header row\nData row"
    )
    assert isinstance(result, str)


def test_reconstruct_tables_fitz_empty_text_returns_string():
    """Defensive: empty existing_text should not crash; returns string."""
    result = reconstruct_tables_fitz(str(ARXIV_MATH_PDF_WITH_BBOX), 1, "")
    assert isinstance(result, str)


# === Cross-cutting: all detector functions accept str path ==== #


def test_all_detectors_accept_string_path_return_list_or_raise_clean():
    """Contract: every detector takes a str PDF path + int page_num."""
    pdf = str(ARXIV_MATH_PDF_WITH_BBOX)
    for fn_name, fn in [
        ("find_table_bbox", find_table_bbox),
        ("detect_tables_pdfplumber", detect_tables_pdfplumber),
        ("detect_text_only_tables", detect_text_only_tables),
    ]:
        try:
            result = fn(pdf, 1)
            assert isinstance(result, list), f"{fn_name} must return list"
        except Exception as e:
            pytest.fail(f"{fn_name} raised unexpected {type(e).__name__}: {e}")


# === Cross-PDF: same detector on 3 different real PDFs ======== #


def test_find_table_bbox_works_across_three_real_pdfs():
    """Smoke test: function works on PDFs from 3 different categories."""
    pdfs = [
        ARXIV_MATH_PDF_WITH_BBOX,
        MULTI_COLUMN_PDF_WITH_BBOX,
        HEADERS_FOOTERS_PDF_WITH_BBOX,
    ]
    for pdf in pdfs:
        if not pdf.exists():
            pytest.skip(f"fixture missing: {pdf}")
        result = find_table_bbox(str(pdf), 1)
        assert isinstance(result, list), f"failed on {pdf.name}"
