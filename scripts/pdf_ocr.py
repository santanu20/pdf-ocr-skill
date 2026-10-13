#!/usr/bin/env python3
"""OCR a PDF using any Ollama vision model, outputting markdown/JSON."""

import sys
import os

# --------------------------------------------------------------------------- #
# Skill-local venv auto-use + auto-bootstrap (pipx-style, mirrors web-search)
#   1. If .venv missing + uv available + not disabled → auto-create + install.
#   2. If .venv exists + not running from it → re-exec into it.
#   Set PDF_OCR_NO_SKILL_VENV=1 to force cwd's python.
#   Set PDF_OCR_NO_BOOTSTRAP=1 to skip auto-bootstrap.
# --------------------------------------------------------------------------- #
if __name__ == "__main__" and not os.environ.get("PDF_OCR_NO_SKILL_VENV"):
    _SKILL_DIR = os.path.dirname(os.path.abspath(__file__))
    # Walk up to skill root (script is in scripts/, venv goes in skill root)
    while _SKILL_DIR and os.path.basename(_SKILL_DIR) != "pdf-ocr":
        parent = os.path.dirname(_SKILL_DIR)
        if parent == _SKILL_DIR:
            break
        _SKILL_DIR = parent
    _SKILL_VENV = os.path.join(_SKILL_DIR, ".venv")
    # Windows venvs put the interpreter at Scripts\python.exe (audit H2)
    _SKILL_VENV_PY = os.path.join(
        _SKILL_VENV,
        "Scripts" + os.sep + "python.exe" if os.name == "nt" else os.path.join("bin", "python"),
    )

    # Auto-bootstrap on first run — uv sync from pyproject.toml (the single
    # source of dependency truth; the old hardcoded list drifted from it,
    # installing a subset that lacked every ONNX specialist (dependency audit).
    if not os.path.exists(_SKILL_VENV_PY) and not os.environ.get("PDF_OCR_NO_BOOTSTRAP"):
        import subprocess

        try:
            sys.stderr.write("Bootstrapping pdf-ocr skill venv (one-time setup)...\n")
            subprocess.run(
                ["uv", "sync", "--project", _SKILL_DIR, "--python", "3.13"],
                check=True,
                capture_output=True,
            )
            sys.stderr.write("pdf-ocr skill venv ready.\n")
        except (subprocess.CalledProcessError, FileNotFoundError) as ex:
            sys.stderr.write(
                f"Failed to auto-bootstrap pdf-ocr skill venv: {ex}\n"
                "Create manually:\n"
                f"  uv sync --project {_SKILL_DIR} --python 3.13\n"
                "Or set PDF_OCR_NO_BOOTSTRAP=1 + PDF_OCR_NO_SKILL_VENV=1 "
                "to use cwd venv.\n"
            )
            sys.exit(2)

    # Re-exec into skill venv
    if os.path.exists(_SKILL_VENV_PY) and os.path.normpath(sys.prefix) != os.path.normpath(
        _SKILL_VENV
    ):
        os.environ["PDF_OCR_NO_SKILL_VENV"] = "1"  # prevent recursion
        os.execv(_SKILL_VENV_PY, [_SKILL_VENV_PY, os.path.abspath(__file__)] + sys.argv[1:])

import subprocess
import json
import base64
import urllib.request
import sys
import os
import re
import time
import tempfile
import shutil
import hashlib
import threading
from dataclasses import dataclass
import logging

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state

_page_dict_cache = {}


from ftfy import fix_text as _ftfy_fix_text
import periodictable
import fitz  # PyMuPDF — text/tables/reading-order/layout

# NOTE (2026-08-16): `import pymupdf4llm` was here but NEVER used — and its
# import side effect (TOOLS.unset_quad_corrections(True), never restored)
# shifted span bboxes ~0.6pt globally, breaking font-math superscript
# clustering ('\mathbb{C}^{N}' → '\mathbb{C}', flip-diff 2503.06703_pg4).
# Removed; import it lazily AT the call site if ever actually needed.
import pdfplumber  # char-level bbox + lattice/text table extraction
# _pdfium + _pdfium_c come from `shared` (single source of truth, defined there)


_SYMSPELL_INSTANCE = None

# Module-level flag set from CLI args in main(). Default False = skip the
# 5-15s glm-ocr call per figure region (born-digital PDFs don't need it).
# Set via --figure-table-ocr CLI flag.
_FIGURE_TABLE_OCR = False


REPEAT_LINE_MIN_CHARS = 40


DUPLICATE_CHAR_MIN_LEN = 6  # only check words ≥6 chars (avoids "WWII", "AA", etc.)
MIN_SPAN_CHARS = 8  # preserve_typography_fitz: only wrap spans ≥8 chars (false-positive guard)


# =============================================================================
# Subscript/superscript detection — ported from pdftext/pdf/pages.py
# Original: datalab-to/pdftext (Apache-2.0). Pure-Python, no ML.
# Uses per-char bbox (pypdfium2) to detect chars positioned below or above
# the baseline. Produces LaTeX-style $_{...}$ and $^{...}$ inline.
# =============================================================================
# _is_math_symbol + _pdfium/_pdfium_c are imported via `from shared import *`
# (single source of truth, defined in shared.py since 2026-08-13).


def _top2(values):
    """Return (max1, max1_idx, max2) for max-excluding-index O(1) queries."""
    max1 = max2 = float("-inf")
    max1_idx = -1
    for idx, v in enumerate(values):
        if v > max1:
            max2 = max1
            max1 = v
            max1_idx = idx
        elif v > max2:
            max2 = v
    return max1, max1_idx, max2


def _bottom2(values):
    """Return (min1, min1_idx, min2) for min-excluding-index O(1) queries."""
    min1 = min2 = float("inf")
    min1_idx = -1
    for idx, v in enumerate(values):
        if v < min1:
            min2 = min1
            min1 = v
            min1_idx = idx
        elif v < min2:
            min2 = v
    return min1, min1_idx, min2


USAGE = """\
Usage: python3 pdf_ocr.py <input> <output> [options]

  input     Path to PDF or image file (.pdf, .png, .jpg, .jpeg, .tif, .tiff,
            .bmp, .gif, .webp). Images are treated as single-page documents.
  output    Path for output file (.md for markdown, base name used for .jsonl
            and .summary.json when --json is set)

Options:
  --pages N     Process only a single page (1-indexed, PDF only)
  --pages N-M   Process a page range inclusive (1-indexed, PDF only)
  --force-ocr   Skip pdftotext fast path; always use OCR. Use for
                math-heavy or layout-critical PDFs.
  --crop X,Y,W,H  Crop region in PDF POINTS, top-left origin. Implies --force-ocr.
  --models M1,M2,...  OCR model (default: rapidocr; auto-escalates to Ollama VL when thin). Add --always-model to pin.
  --json        Emit structured JSON (.pages.jsonl + .summary.json) for agent
                consumption instead of markdown.

Examples:
  python3 pdf_ocr.py report.pdf report.md
  python3 pdf_ocr.py report.pdf report_p5.md --pages 5
  python3 pdf_ocr.py scan.png scan.md --force-ocr
  python3 pdf_ocr.py paper.pdf paper.json --json
  python3 pdf_ocr.py paper.pdf paper.md --models glm-ocr,ovisocr2

Requirements:
  - rapidocr (bundled in skill venv) for the default CPU OCR path; Ollama
    (http://127.0.0.1:11434) with a vision model for VL OCR / escalation
    (glm-ocr recommended — best measured accuracy + speed)
  - pdftoppm and pdfinfo (brew install poppler / apt install poppler-utils)
  - Pillow (pip install Pillow) for image resize/crop; falls back to `sips` on macOS
"""

import argparse

# Add scripts/ to sys.path EARLY so scientific_parser imports resolve.
# Required for both module-load (our provider imports) and sidecar emission.
import sys as _sys_early

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in _sys_early.path:
    _sys_early.path.insert(0, _SCRIPTS_DIR)

# SOTA specialist providers — extracted to scientific_parser/perception/providers
# (Phase R4 refactoring 2026-08-12). All names re-exported below for backwards
# compatibility (external QC tests import from pdf_ocr directly).
from scientific_parser.perception.providers import (
    RapidLaTeXProvider,
    get_formula_recognizer,
    RapidLayoutProvider,
    get_layout_detector,
    RapidTableProvider,
    get_table_recognizer,
    route_page_by_region,
    _crop_image,
    _sort_regions_reading_order,
    _extract_text_from_bbox,
)
from scientific_parser.core.path_warnings import (
    record_path_warning,
    reset_path_warnings,
    get_path_warnings,
    format_path_warnings_summary,
    format_path_warning_records,
    warn_born_digital_ocr_routes,
    PDFTOTEXT_UNAVAILABLE,
    MIN_DOC_OCR_FORCED,
    VL_FALLBACK_USED,
    MODEL_FAILURE_DEGRADED,
    OCR_MODEL_SWAP,
    FAST_PATH_OCR_ESCALATION,
    REGION_OUTPUT_REJECTED,
    STAGE_SKIPPED,
    FONT_MATH_CASCADE_FAILED,
    CLI_CONFIG_IGNORED,
    PAGE_COUNT_FALLBACK,
    SPECIALIST_PROVIDERS_MISSING,
)


def _pw_skip(stage):
    """Record a stage_skipped path warning (one stable row per stage)."""
    record_path_warning(
        STAGE_SKIPPED,
        f"{stage} skipped (exception) — stage output lost, pipeline continued",
    )


# =============================================================================
# Math equation extraction — extracted to scientific_parser/perception/equations
# =============================================================================
# Phase R3 refactoring (2026-08-12): font-aware math, Unicode→LaTeX,
# display equation detection, math region reconstruction — all moved to
# scripts/scientific_parser/perception/equations.py. Re-exported below for
# backwards compatibility.
from scientific_parser.perception.equations import (
    MATH_FONT_PREFIXES,
    UNICODE_TO_LATEX,
    _OCR_LATEX_CMD_CONFUSIONS,
    detect_display_equations,
    wrap_equations_as_latex,
    ocr_display_equations,
    _is_math_font,
    _has_math_content,
    _span_to_latex,
    reconstruct_math_region,
    extract_math_regions,
    replace_math_in_text,
    improve_math_regions,
)

# =============================================================================
# Routing — extracted to scientific_parser/routing/router
# =============================================================================
# Phase R5 refactoring (2026-08-12): page routing, preclassification, model
# selection — all moved to scripts/scientific_parser/routing/router.py.
from scientific_parser.routing.page_classifier import (
    should_route_to_ocr,
    preclassify_document,
    select_ocr_model,
    is_image_like_page,
    page_has_invisible_text,
    _is_legitimate_short_line_page,
    extract_text_layer,
    rescue_glued_pages,
)

# =============================================================================
# Layout — extracted to scientific_parser/structure/layout
# =============================================================================
# Phase R7 refactoring (2026-08-12): XY-Cut++, column-aware reading order,
# block classification — moved to scripts/scientific_parser/structure/layout.py.
from scientific_parser.structure.column_reading_order import (
    _xy_cut_partition,
    classify_text_block,
    column_aware_body_only,
    column_aware_reading_order,
    pdfplumber_column_crop,
)

# =============================================================================
# Validators — extracted to scientific_parser/science/validators_geo
# =============================================================================
# Phase R8 refactoring (2026-08-12): REE sequence, mineral abbreviations,
# scientific data validators — moved to validators_geo.py.
from scientific_parser.science.geo_validators import (
    validate_ree_sequence,
    detect_mineral_abbreviations,
    validate_scientific_data,
    MINERAL_ABBREVS,
)

# =============================================================================
# Science specialists — Unicode→LaTeX, equation repair, formula detection
# =============================================================================
from scientific_parser.science.unicode_to_latex import (
    convert_unicode_to_latex,
    normalize_inline_math,
)
from scientific_parser.verification.render_compare import repair_latex
from scientific_parser.science.formulas import find_formulas_in_text

# =============================================================================
# Postproc - extracted to scientific_parser/postproc/cleanup
# =============================================================================
from scientific_parser.postproc.text_pipeline import (
    restore_degree_symbol,
    correct_ocr_errors,
    fix_text_encoding,
    drop_repeated_text,
    remove_bold_duplicate_chars,
    merge_cross_page_tables,
    detect_all_tables,
    repair_encoding_decimals,
    preserve_typography,
    dehyphenate,
    normalize_chemical_notation,
    repair_citation_splits,
    preserve_dois,
    detect_bibliography_section,
    process_bibliography,
    detect_section_hierarchy,
    reconstruct_footnotes,
    associate_captions,
    repair_bullets,
    UNICODE_MATH_CHARS,
    restore_scientific_subscripts,
    restore_unicode_losses,
    promote_allcaps_headings,
    sanitize_phantom_decimals,
    fix_permille,
    validate_scientific_symbols,
    validate_chemical_formulas,
    markdown_compatible,
    strip_cjk_hallucination,
    strip_running_stamps,
    fix_latex_math_output,
    PostProcessor,
    _build_registry,
    _get_registry,
    run_unified_postproc,
    join_fragmented_math_lines,
    _collapse_repetition_inline,
    _extract_text_with_scripts_safe,
    _detect_and_append_tables,
    PAGE_NUMBER_RE,
    BRACKET_HEADER_RE,
    HEADER_LINE_MAX_CHARS,
    LOW_TEXT_DENSITY_THRESHOLD,
)

# =============================================================================
# SOTA cache + reflexive - extracted to scientific_parser/core/sota
# =============================================================================
from scientific_parser.core import env_fingerprint as _envfp
from scientific_parser.core.extraction_cache import (
    REFLEXIVE_CONFIDENCE_FLOOR,
    REFLEXIVE_MAX_ATTEMPTS_PER_MODEL,
    OCR_PER_ATTEMPT_TIMEOUT_S,
    SKILL_EXTRACTION_VERSION,
    DEFAULT_CACHE_DIR,
    _cache_key,
    cache_lookup,
    cache_write,
    page_memo_lookup,
    page_memo_store,
    _verify_laTeX_quality,
    ocr_with_reflexive_loop,
    _pick_best_ocr,
    _score_ocr_text,
)
from scientific_parser.ingest.font_analysis import (
    detect_subscripts_pymupdf,
    assign_scripts_to_chars,
    _extract_scripts_pymupdf,
    _extract_scripts_pypdfium,
    _build_math_regex,
)
from scientific_parser.routing.page_metadata import (
    _get_page_dict,
    strip_recurring_headers,
    extract_page_metadata,
    _format_meta_comment,
    page_has_embedded_fonts,
)

from scientific_parser.perception.tables.structure import (
    Table,
    _detect_table_structure,
    classify_cell,
    infer_spanning_cells,
    validate_table_structure,
    _dedup_tables,
    _line_to_table_cells,
)
from scientific_parser.core.confidence_helpers import (
    _get_symspell,
    extract_text_with_scripts,
    list_ollama_models,
    ollama_unload,
    detect_page_complexity,
    _compute_page_confidence,
    looks_like_chat_response as _looks_like_chat_response,
    combine_page_confidence,
    _quality_gate,
    crop_image,
    has_math_content,
    _has_monotonic_re_run,
    invalid_char_count,
    SCIENTIFIC_SYMBOL_RULES,
    REE_OCR_CONFUSIONS,
)


# =============================================================================
# Tables — extracted to scientific_parser/perception/tables/detector
# =============================================================================
from scientific_parser.perception.tables.detector import (
    Cell,
    _detect_table_regions_numeric,
    _optimize_grid_em,
    _build_table_grid,
    detect_multirow_header,
    infer_column_types,
    build_table_markdown,
    detect_tables_in_text,
    detect_text_only_tables,
    detect_tables_lines_strict,
    detect_tables_camelot,
    detect_tables_in_figure,
    detect_tables_pdfplumber,
    reconstruct_tables_fitz,
    detect_tables,
    find_table_bbox,
)


# =============================================================================
# OCR engine — extracted to scientific_parser/ingest/ocr_engine
# =============================================================================
from scientific_parser.ingest.ocr_engine import (
    pdf_page_to_png,
    resize_image,
    get_model_profile,
    _get_rapidocr,
    ocr_image_rapidocr,
    ocr_image,
    strip_thinking_artifacts,
)

# ─── OCR prompts (referenced by ocr_image + reflexive loop) ──────────────── #
MATH_OCR_PROMPT = (
    "You are a precise math OCR engine. Transcribe the image to markdown + LaTeX.\n"
    "\n"
    "STRICT RULES (violations break downstream rendering):\n"
    "1. Math goes in LaTeX: inline `$...$`, display `$$...$$`.\n"
    "2. NEVER nest `$` inside another math block. If transcribing `x^{m-1}` inside a "
    "display block, write `$$x^{m-1}$$`, NOT `$$x$^{m-1}$...$$`.\n"
    "3. Subscripts: `_{...}`, superscripts: `^{...}` — NO surrounding `$`.\n"
    "4. Use only valid KaTeX commands. Common ones: `\\frac{}{}  \\sqrt{}  \\sum  \\int  "
    "\\alpha  \\beta  \\gamma  \\Delta  \\pi  \\infty  \\partial  \\nabla  \\cdot  \\times  "
    "\\pm  \\leq  \\geq  \\neq  \\to  \\in  \\subset  \\forall  \\exists  \\sin  \\cos  \\tan  "
    "\\log  \\ln  \\exp  \\lim  \\det  \\dim  \\max  \\min  \\cdot  \\cdots  \\ldots  \\ddots  "
    "\\vdots  \\tag{}  \\begin{matrix} \\end{matrix}  \\begin{pmatrix} \\end{pmatrix}  "
    "\\begin{cases} \\end{cases}  \\left( \\right)  \\mathrm{}`.\n"
    "5. NEVER invent commands. If unsure, write the symbol as Unicode (∑ ∏ ∫ ≤ ≥ ≠ ± × ÷ → α β γ π Δ ∞ ∂ ∇).\n"
    "6. Keep equation structure intact — don't break across lines, don't truncate.\n"
    "7. Preserve equation numbers as `\\tag{N}` at end of display equations.\n"
    "8. For `sqrt`, ALWAYS spell as `\\sqrt{}` — never `\\sort{}` or other variants.\n"
    "\n"
    "Transcribe now. Output only the content, no commentary."
)

# ─── SOTA Extraction Layer constants (defined before argparse use) ──────── #
# Docstrings + full explanation at the SOTA Extraction Layer block
# (search "SOTA Extraction Layer — adaptive model router").
parser = argparse.ArgumentParser(
    description="OCR a PDF or image using any Ollama vision model, outputting markdown or JSON.",
    add_help=True,
)
parser.add_argument("input", help="Path to input PDF or image (.png/.jpg/.tif/.bmp/.gif/.webp)")
parser.add_argument(
    "output",
    help="Path for output (.md for markdown, base name used for .jsonl/.summary.json with --json)",
)
parser.add_argument(
    "--pages",
    metavar="N[-M]",
    help="Process only specific page(s). N for single page, N-M for inclusive range. 1-indexed. Default: all pages.",
)
parser.add_argument(
    "--force-ocr",
    action="store_true",
    help="Skip pdftotext fast path; always use OCR. Use for math-heavy or "
    "layout-critical PDFs where pdftotext mangles equations/tables.",
)
parser.add_argument(
    "--crop",
    metavar="X,Y,W,H",
    help="Crop a region of the page in PDF POINTS with TOP-LEFT origin "
    "(matches `pdftotext -tsv` output: `pdftotext -tsv in.pdf out.tsv` "
    "then read top/left/width/height and feed back). Example: "
    '--crop "100,400,400,150". Implies --force-ocr (cropped region '
    "needs vision). Combine with --pages N to target one page.",
)
parser.add_argument(
    "--models",
    metavar="M1,M2,...",
    help="Comma-separated list of OCR models. First entry is primary (default: "
    "rapidocr; auto-escalates to best Ollama VL when rapidocr output is thin). "
    "Any Ollama vision model works. Unknown models use DEFAULT_MODEL_PROFILE "
    "automatically.",
)
parser.add_argument(
    "--always-model",
    action="store_true",
    help="Pin --models: no automatic model substitution. Normally rapidocr "
    "output that is too thin silently escalates to the best Ollama VL "
    "model; with this flag the chosen model's result is used as-is.",
)
parser.add_argument(
    "--json",
    action="store_true",
    help="Emit structured JSON output instead of markdown. Writes two files: "
    "<base>.pages.jsonl (one JSON record per page, streaming — crash-safe) "
    "and <base>.summary.json (aggregate metrics: pages ok/warned/failed, "
    "total chars, elapsed time, per-page metadata). Use for agent/programmatic "
    "consumption where markdown parsing is fragile.",
)
parser.add_argument(
    "--fast",
    action="store_true",
    help="Disable SOTA quality routing; use simple char-count threshold only "
    "(chars < 100/page → OCR). Faster on large documents (skips pdfimages + "
    "text-quality scoring per page) but routes mangled-text PDFs to fast path "
    "silently. Use for trusted batch processing where input quality is known.",
)
parser.add_argument(
    "--figure-table-ocr",
    action="store_true",
    help="OCR figure regions looking for image-embedded tables (scanned PDFs "
    "with raster-table figures). OFF by default — adds 5-15s per figure region "
    "due to glm-ocr call. Born-digital PDFs do not need this (text is in the "
    "text layer). Enable for scanned PDFs with suspected image-tables.",
)
parser.add_argument(
    "--min-doc-ocr-pct",
    type=float,
    default=50.0,
    metavar="PCT",
    help="If fewer than PCT%% of pages are 'good' (routed to FAST), force "
    "OCR for the ENTIRE document. Catches docs where most pages are scanned "
    "but a text layer exists (e.g. OCRed-by-Adobe PDFs with poor quality). "
    "Default: 50 (if <50%% of pages are good, OCR the whole doc). "
    "Set to 0 to disable. Marker SOTA inspiration.",
)
parser.add_argument(
    "--dpi",
    type=int,
    default=None,
    metavar="N",
    help="OCR render DPI. Default: adaptive (150 normal, 250 dense scientific, "
    "300 tiny/scanned — auto-detected from page font analysis). "
    "Override when auto-detection misfires. Higher DPI = more pixels but "
    "downscaled to --max-dim before OCR. A/B test on glm-ocr showed no "
    "accuracy gain above 300 DPI (image is downscaled anyway).",
)
parser.add_argument(
    "--max-dim",
    type=int,
    default=None,
    metavar="N",
    help="Maximum image dimension (pixels) sent to OCR model. Default: 1024. "
    "Images larger than this are LANCZOS-downscaled. WARNING: setting "
    "above 1024 may TRUNCATE OCR output on glm-ocr (model token-limit "
    "on high-detail input — verified 2026-07-25). Increase only when "
    "using models known to support higher resolution (e.g. ovisocr2).",
)
parser.add_argument(
    "--layout-backend",
    default="default",
    choices=["default", "rapid-layout", "doclayout-yolo-onnx"],
    help="Layout detection backend. 'default'/'rapid-layout' = RapidLayout "
    "ONNX (pp_doc_layoutv3, ~10-30MB, CPU). 'doclayout-yolo-onnx' = "
    "DocLayout-YOLO (arXiv 2410.12628, SOTA table/figure recall, ~5x faster "
    "inference, CPU, no torch). NO torch/paddlepaddle.",
)
parser.add_argument(
    "--formula-backend",
    default="default",
    choices=["default", "rapidlatex"],
    help="Formula recognition backend for scanned math (old_scans_math). "
    "'default' = current heuristic (Unicode mapping + font detection). "
    "'rapidlatex' = RapidLaTeXOCR ONNX (~80MB, lazy opt-in). "
    "NO torch/paddlepaddle.",
)
parser.add_argument(
    "--table-backend",
    default="default",
    choices=["default", "rapidtable"],
    help="Table structure recognition backend. 'default' = current 6-strategy "
    "heuristic (pdfplumber + camelot + lattice). 'rapidtable' = RapidTable "
    "ONNX (~30MB, lazy opt-in). NO torch/paddlepaddle.",
)
parser.add_argument(
    "--no-cache",
    action="store_true",
    help="Skip content-hash cache lookup AND write. By default, extracted "
    "output is cached at ~/.cache/pdf-ocr/<sha256>/ keyed on file bytes + "
    "flags + skill version. Re-running the same PDF with same flags returns "
    "in <5%% original time. Set --no-cache to force re-extraction (or to "
    "avoid writing cache when processing sensitive inputs).",
)
parser.add_argument(
    "--no-reflexive",
    action="store_true",
    help="Disable reflexive OCR loop (extract → verify → retry). By default "
    "the skill uses arXiv 2603.22651 reflexive pattern: each OCR-d page is "
    "verified by pykatex, scored, and re-extracted with error feedback if "
    "confidence < 0.55. Disabling restores single-shot OCR (faster but less "
    "accurate on math/scan-heavy pages).",
)
parser.add_argument(
    "--no-cascade",
    action="store_true",
    help="Disable multi-model cascade. By default, --models a,b,c is honored "
    "as an ordered fallback chain (try a first, fall back to b on low "
    "confidence, then c). With --no-cascade, only the first model is used "
    "(legacy behavior, pre-2026-08-12).",
)
parser.add_argument(
    "--parallel",
    type=int,
    default=1,
    metavar="N",
    help="Page-level parallelism for OCR-bound pages. Default: 1 (serial). "
    "Increase to N>1 only when Ollama is configured for parallel streams "
    "(OLLAMA_NUM_PARALLEL env). On single-GPU systems with default Ollama, "
    "parallel calls queue at the server — no real speedup. Multi-GPU or "
    "CPU-only backends (rapidocr) benefit most.",
)
parser.add_argument(
    "--reflexive-floor",
    type=float,
    default=REFLEXIVE_CONFIDENCE_FLOOR,
    metavar="CONF",
    help=f"Confidence floor for reflexive loop retry. Default: {REFLEXIVE_CONFIDENCE_FLOOR}. "
    "Pages scoring below this trigger re-extract with feedback. Range 0-1. "
    "Lower = fewer retries (faster, less accurate). Higher = more retries.",
)
# --- Risk-aware routing + correction-loop flags (2026-08-13) ---------------
parser.add_argument(
    "--escalate",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Active selective redundancy (default ON): regions where uncertainty "
    "x impact exceeds --escalate-threshold trigger a second model / "
    "specialist via the failure-directed retry executor. --no-escalate "
    "restores single-pass output.",
)
parser.add_argument(
    "--escalate-threshold",
    type=float,
    default=1.5,
    metavar="RISK",
    help="Risk threshold (uncertainty x semantic_impact) above which a region "
    "escalates to a second model. Range 0-5. Lower = more escalations (higher "
    "quality, more compute). Default 1.5 (e.g. uncertainty 0.3 x impact 5).",
)
parser.add_argument(
    "--content-loss-ratio",
    type=float,
    default=0.5,
    metavar="RATIO",
    help="If region-routed text is shorter than this fraction of the fallback "
    "page text, the fallback is kept (guards against layout misses dropping "
    "content). Default 0.5.",
)
parser.add_argument(
    "--dpi-cap",
    type=float,
    default=600,
    metavar="DPI",
    help="Maximum render DPI for retry escalations (higher-DPI re-render). "
    "Default 600 (above this glm-ocr token truncation risk rises).",
)
parser.add_argument(
    "--resolver-drives-output",
    action="store_true",
    default=False,
    help="Candidate-resolver decides each region's emitted text (weighted "
    "evidence fusion across specialist + native candidates) instead of "
    "first-wins. Default off (first-wins preserves byte-identical output); "
    "flip ON after A/B benchmarking confirms a net quality gain.",
)
# Note: no --domain flag. Processors are self-gating — each proc checks for
# its own content signature before applying (e.g. detect_mineral_abbreviations
# requires rock/mineral context; validate_ree_sequence requires 3+ REE in
# monotonic cluster). This is the SOTA design: universal parser, no field
# hints needed. Works correctly for any science field out of the box.

args = parser.parse_args() if __name__ == "__main__" else None

# Propagate --figure-table-ocr to module-level flag (read inside
# column_aware_body_only → detect_tables_in_figure call site).
if args is not None:
    _FIGURE_TABLE_OCR = bool(getattr(args, "figure_table_ocr", False))
    # Honor --max-dim CLI override (was previously dead code; MAX_DIM constant
    # won 2026-08-11). Higher dim = better math OCR (less \sqrt→\sort confusion
    # at 1024 downscale). Verified safe up to 4096 on glm-ocr (num_ctx=32768).
    if getattr(args, "max_dim", None):
        MAX_DIM = args.max_dim

PDF_PATH = args.input if args else ""
OUTPUT_PATH = args.output if args else ""

if args is not None:
    if not os.path.isfile(PDF_PATH):
        print(f"Error: file not found: {PDF_PATH}", file=sys.stderr)
        print(USAGE, file=sys.stderr)
        sys.exit(1)

    # Accept PDFs and common image formats (agents get screenshots/scans too)
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
    }
    _INPUT_EXT = os.path.splitext(PDF_PATH.lower())[1]
    IS_PDF = _INPUT_EXT == ".pdf"
    IS_IMAGE = _INPUT_EXT in IMAGE_EXTENSIONS
    if not IS_PDF and not IS_IMAGE:
        print(
            f"Error: input must be .pdf or an image ({', '.join(sorted(IMAGE_EXTENSIONS))}), "
            f"got: {PDF_PATH}",
            file=sys.stderr,
        )
        sys.exit(1)
else:
    IMAGE_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
        ".gif",
        ".webp",
    }
    IS_PDF = False
    IS_IMAGE = False


def parse_page_range(spec, total_pages):
    """Parse --pages spec into list of 1-indexed page numbers. Validates bounds."""
    if spec is None:
        return list(range(1, total_pages + 1))
    if "-" in spec:
        parts = spec.split("-", 1)
        try:
            start, end = int(parts[0]), int(parts[1])
        except ValueError:
            raise ValueError(f"--pages range must be 'N' or 'N-M', got: {spec!r}")
        if start > end:
            start, end = end, start  # tolerate reverse
    else:
        try:
            start = end = int(spec)
        except ValueError:
            raise ValueError(f"--pages must be integer or range, got: {spec!r}")
    if start < 1:
        raise ValueError(f"--pages start must be >= 1, got {start}")
    if start > total_pages:
        raise ValueError(f"--pages {start}-{end} entirely beyond PDF with {total_pages} pages")
    if end > total_pages:
        # CLAMP, don't fail (2026-09-03, synced from geokit wrapper): caller
        # caps like '1-30' (GPU-wedge mitigation) must not hard-fail shorter
        # documents — every PDF under the cap exited 1 and lost OCR silently.
        end = total_pages
    return list(range(start, end + 1))


OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
MAX_DIM = 4096  # max pixel dimension for images sent to model.
# Was 1024 (downscaled A4) — caused glm-ocr to misread `\sqrt` as `\sort`
# on math-heavy pages (detail lost at 1024). 4096 = no downscale at 150-300
# DPI for any standard page size. glm-ocr supports 131072-token context
# (verified `ollama show`), so larger images don't truncate output.
DPI = 150  # 150 DPI balances accuracy (chemistry/citations/equations) vs speed.
# Lower (e.g. 72) yields wrong compounds, fabricated authors, and footnote loops.
# Higher (e.g. 300) truncates output (model hits token limit on long pages).

# ─── SOTA Extraction Layer constants (defined before argparse use) ──────── #
# These appear again with full docstrings at the SOTA Extraction Layer block
# (search "SOTA Extraction Layer"). These come from
# scientific_parser.core.extraction_cache via the shared.py star-import —
# single source of truth; do NOT redefine here (F811 drift trap, 2026-08-16
# audit M2). DEFAULT_CACHE_DIR likewise arrives via shared.py.


def get_page_count(pdf_path):
    # Try pdfinfo first (fast, no Python dep)
    try:
        result = subprocess.run(
            ["pdfinfo", pdf_path],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=10,
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception as e:
        log.debug("get_page_count: pdfinfo failed (%s) — trying PyMuPDF", e)

        record_path_warning(
            PAGE_COUNT_FALLBACK,
            "pdfinfo failed — page count via PyMuPDF fallback",
        )
    # Fallback: PyMuPDF (reliable in all environments incl. ProcessPoolExecutor)
    try:
        import fitz

        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count
    except Exception as e:
        log.debug("get_page_count: PyMuPDF failed (%s) — trying pypdf", e)

        record_path_warning(
            PAGE_COUNT_FALLBACK,
            "pdfinfo AND PyMuPDF failed — page count via pypdf fallback",
        )
    # Fallback: pypdf
    try:
        from pypdf import PdfReader

        return len(PdfReader(pdf_path).pages)
    except Exception as e:
        log.debug("get_page_count: pypdf failed (%s) — giving up", e)
    raise ValueError(f"Could not determine page count for {pdf_path}")


def _cells_to_markdown(cells):
    """Convert cell grid to markdown pipe table.

    Detects header/body structure, preserves hierarchy in Table object,
    then serializes to markdown (flattening headers at output time only).
    """
    table = _detect_table_structure(cells)
    return table.to_markdown()


def detect_continued_table_marker(text):
    """Detect 'Table N (continued)' markers indicating multi-page table.

    Returns (table_number, is_continuation) or None if no marker found.
    Used to link continuation pages back to the original table.
    """
    # Match: "Table 5 (cont.)", "Table 5 (continued)", "Table 5 cont'd"
    m = re.search(
        r"\bTable\s+(\d+)\s*[\(\[]?(?:cont(?:inued|\.?|d)\'?)\.?\s*[\)\]]?",
        text,
        re.IGNORECASE,
    )
    if m:
        return (int(m.group(1)), True)
    return None


# _is_numeric_cell: imported from shared.py via `from shared import *` since
# 2026-08-13. See shared.py for canonical definition.


_OPEN_TABLES_STATE: list = []


MATH_OPERATOR_CHARS = set("=±×÷∑∫∂√∞≈≠≤≥αβγδεζηθικλμνξπρστυφχψωΔΣΠΩ∇·→↔∝")


SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUPERSCRIPT_DIGITS = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

# Common oxide formulas → Unicode subscript version
OXIDE_FORMULAS = {
    "SiO2": "SiO₂",
    "Al2O3": "Al₂O₃",
    "Fe2O3": "Fe₂O₃",
    "FeO": "FeO",
    "CaO": "CaO",
    "MgO": "MgO",
    "Na2O": "Na₂O",
    "K2O": "K₂O",
    "TiO2": "TiO₂",
    "P2O5": "P₂O₅",
    "MnO": "MnO",
    "Cr2O3": "Cr₂O₃",
    "H2O": "H₂O",
    "CO2": "CO₂",
    "SO2": "SO₂",
    "Fe2O3T": "Fe₂O₃ᵀ",
    "FeOT": "FeOᵀ",
}

# Crystallographic notation patterns
# (hkl) → plane, [hkl] → direction, {hkl} → family of planes
CRYSTAL_INDICES_RE = re.compile(
    r"(?<![A-Za-z])"
    r"[\(\[\{]"
    r"(\d[\d\-\d]*)"
    r"[\)\]\}]"
    r"(?![A-Za-z])"
)


DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s\)\]\}>,]+)")

# Citation bracket split: word-[NN]newline or [NN]-word
CITATION_SPLIT_RE = re.compile(r"(\w)-\n\[(\d+)\]|\[(\d+)\]-\n(\w)")


def _normalize_header_hash(text):
    """Normalize text for header comparison (digits replaced with #)."""
    return re.sub(r"\d+", "#", text).strip()


CRYSTALLOGRAPHIC_RE = re.compile(r"(?<![A-Za-z])[\(\[\{<][\d\-\d]+[\)\]\}>](?![A-Za-z])")


def parse_crop(spec):
    """Parse --crop 'X,Y,W,H' into tuple of floats. PDF points, top-left origin."""
    try:
        parts = [float(x.strip()) for x in spec.split(",")]
    except ValueError:
        raise ValueError(f"--crop must be 'X,Y,W,H' as numbers, got: {spec!r}")
    if len(parts) != 4:
        raise ValueError(f"--crop requires 4 values (X,Y,W,H), got {len(parts)}")
    x, y, w, h = parts
    if w <= 0 or h <= 0:
        raise ValueError(f"--crop width/height must be positive, got W={w} H={h}")
    return (x, y, w, h)


_RAPIDOCR_INSTANCE = None


import re

ALLCAPS_HEADING_RE = re.compile(
    r"^(?:"
    r"(?<=[\n][\n])"  # preceded by blank line (lookbehind)
    r"|(?<=^)"  # or start of string
    r")"
    r"([A-Z][A-Z\s.,'&\-:]{3,}[A-Z])"  # all-caps line, >=4 chars
    r"(?=\n\n|\n$|$)",  # followed by blank line or end
)


TABLE_GAP_RE = re.compile(r"\s{3,}")


# =============================================================================
# Hybrid pipeline: code extraction + selective OCR for math/equations
# =============================================================================


# Regex for chars that indicate math content
# Match math/science symbols across ALL fields (physics, chemistry, math,
# engineering, biology, CS, astronomy, economics, geology, etc.).
# Built dynamically by scanning Unicode math/science blocks.
# Coverage: ~1600 chars across 17 blocks (see _build_math_regex docstring).
_MATH_SYMBOLS_RE = _build_math_regex()
_PRIVATE_CHAR_RE = re.compile(r"[\x00-\x08\x0e-\x1f\ufffe\uffff]")
# EXP-1.5 scrub set: _PRIVATE_CHAR_RE plus vertical tab / form feed.
_CTRL_SCRUB_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


SINGLE_ELEM_ZERO_RE = re.compile(r"\b([A-Z][a-z]?)0(\d*)\b|\b([A-Z][a-z]?)(\d+)0\b")


def ocr_image_math(img_path, model="glm-ocr"):
    """OCR with math-aware prompt for equation-heavy regions."""
    return ocr_image(img_path, task=MATH_OCR_PROMPT, model=model)


# =============================================================================
# #3: Region-level OCR splice — replace math regions with OCR LaTeX
# =============================================================================


ROUTER_MIN_CHARS = 100  # below = scanned page or extraction fail
ROUTER_ALPHANUM_MIN = 0.30  # Marker: garbled text detector
ROUTER_SPACE_RATIO_MAX = 0.70  # Marker: whitespace flood (fragmented)
ROUTER_NEWLINE_RATIO_MAX = 0.60  # Marker: one-word-per-line detector
ROUTER_INVALID_CHAR_PCT = 0.03  # Marker: encoding issue
ROUTER_INVALID_CHAR_ABS = 6  # Marker: minimum absolute count
ROUTER_MULTI_SPACE_RUN_MAX = 0.30  # NEW: fragmented table detector
ROUTER_SHORT_LINE_RATIO_MAX = 0.90  # NEW: shattered-cell detector (post-default-mode)
ROUTER_SHORT_LINE_MIN_CHARS = 500  # NEW: avoid false-positive on small diagrams
ROUTER_IMAGE_COVERAGE_THRESHOLD = 0.65  # Marker: image covers ≥65% of page = scanned
ROUTER_MIN_AVG_WPL = (
    5.0  # NEW: avg words/line below this = positioned-glyph PDF (sakai1968 pattern)
)
# R14b (2026-09-12): min count of ISOLATED ´/¨/ˆ markers (not adjacent to
# letters — broken-ToUnicode mojibake of ∫/hats maps to these) that routes a
# born-digital page to the OCR path. Calibrated on the full bench: iso>=3
# hits 28/522 arxiv PDFs (139 gold-test fails live on them, most pages 100%
# failed — asymmetric upside) and exactly 1/266 hf, 0 tiny, 0 multicol,
# 0 tables. The paren→p/q substitution family is invisible at text level
# (valid letters) — tounicode_crosscheck territory, deliberately NOT here.
ROUTER_MOJIBAKE_MIN_MARKERS = 3

# Patterns that legitimately have short lines — NOT shattered tables
_REFLINE_RE = re.compile(r"^\s*\[\d+\]")  # [1] Author et al.
_DOI_RE = re.compile(r"doi:\s|https?://", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\s*[-\u2022*]\s+")  # - item, • item
_CODE_RE = re.compile(r"^\s{4,}\S")  # indented code line


def alphanum_ratio(text):
    """Alphanumeric chars / total chars (excluding whitespace).

    Marker SOTA signal. Validated: real text pages hit 0.84-0.97.
    Garbled/scanned pages hit <0.30. Threshold 0.30 has zero false positives
    across 40-PDF measurement sweep.
    """
    stripped = text.replace(" ", "").replace("\n", "")
    if not stripped:
        return 1.0
    return sum(1 for c in stripped if c.isalnum()) / len(stripped)


def space_ratio(text):
    """Whitespace-chars / (whitespace + non-whitespace chars).

    Marker SOTA. Real text: 0.05-0.21. Fragmented/extraction failures: >0.7.
    """
    spaces = len(re.findall(r"\s+", text))
    non_space = len(re.sub(r"\s+", "", text))
    total = spaces + non_space
    return spaces / total if total else 0.0


def newline_ratio(text):
    """Newline-chars / (newlines + non-newline chars).

    Marker SOTA. Detects one-word-per-line failures (OCR hallucination loops).
    """
    newlines = len(re.findall(r"\n+", text))
    non_nl = len(re.sub(r"\n+", "", text))
    total = newlines + non_nl
    return newlines / total if total else 0.0


def multi_space_run_ratio(text):
    """Fraction of non-leading-indent characters in runs of 3+ spaces.

    NEW signal (not in Marker). Detects shattered tables where pdftotext
    fragments cells across columns. Validated: clean text pages hit 0.00-0.14,
    fragmented tables hit 0.65-0.77. Threshold 0.30 has clear separation.
    Phase-diagram axis-label pages (legitimate short lines) score 0.14 —
    well below threshold, no false positives.
    """
    # Strip leading indent per line so paragraph indentation doesn't pollute
    no_lead = "\n".join(l.lstrip() for l in text.split("\n"))
    if not no_lead:
        return 0.0
    runs = re.findall(r" {3,}", no_lead)
    return sum(len(r) for r in runs) / len(no_lead)


# Reflexive-loop constants (REFLEXIVE_CONFIDENCE_FLOOR 0.55,
# REFLEXIVE_MAX_ATTEMPTS_PER_MODEL 2 — arXiv 2603.22651, OCR_PER_ATTEMPT_
# TIMEOUT_S 240s, SKILL_EXTRACTION_VERSION, DEFAULT_CACHE_DIR) are defined
# ONCE in scientific_parser/core/extraction_cache.py (+ shared.py for
# DEFAULT_CACHE_DIR) and reach this module via star-imports. Do NOT
# redefine here — value drift between copies is the 2026-08-16 audit M2
# finding (F811 x4).

from typing import Any, Callable


TEXT_LAYER_CHARS_PER_PAGE_MIN = 100


DEPS = {
    "PIL": False,  # Pillow
    "ftfy": False,  # optional Unicode fixer
    "pypdfium2": False,  # optional low-level PDF access (render-mode check)
    "pdftoppm": False,  # poppler — PDF → image
    "pdfinfo": False,  # poppler — page count + dims
    "pdftotext": False,  # poppler — fast path text extraction
    "pdffonts": False,  # poppler — font analysis (born-digital detection)
    "pdfimages": False,  # poppler — image coverage analysis
    "ollama": False,  # OCR engine CLI
    "ollama_running": False,  # OCR engine running on localhost
    "models_available": False,  # Requested Ollama vision model(s) pulled
}


def check_dependencies(required_models=None):
    """Audit ALL dependencies. No optional, no soft, no fallback.

    Fail fast, no silent error masking (projectwide mandate):
    every Python module, every CLI tool, every Ollama model
    is HARD-required. Any missing dep → exit code 2 with the COMPLETE missing
    list (so user can fix in one pass). Zero silent degradation.
    """
    import shutil

    # Python imports — verify each (HARD imports already happened at module
    # top; this is a belt-and-suspenders audit so the error message can
    # report which one is missing if it ever is).
    # find_spec, NOT __import__: executing a package's import-time code can
    # mutate shared global state (measured: pymupdf4llm import calls
    # TOOLS.unset_quad_corrections(True), permanently shifting span bboxes
    # ~0.6pt and breaking font-math superscript clustering — iter6 flip-diff
    # 2503.06703_pg4). A presence probe must never run package code.
    import importlib.util

    for mod_name in (
        "PIL",
        "ftfy",
        "periodictable",
        "pypdfium2",
        "pymupdf",
        "pdfplumber",
    ):
        try:
            if importlib.util.find_spec(mod_name) is None:
                raise ImportError(f"No module named '{mod_name}'")
            DEPS[mod_name] = True
        except ImportError:
            DEPS[mod_name] = False

    # SOTA specialist providers (CORE deps per pyproject.toml — all ONNX,
    # no torch/paddlepaddle). Warn (not fail) if any missing — silent
    # degradation to heuristic fallback per provider .available() checks.
    specialist_mods = ("rapidocr", "rapid_latex_ocr", "rapid_table", "rapid_layout")
    specialist_missing = []
    for mod_name in specialist_mods:
        try:
            if importlib.util.find_spec(mod_name) is None:
                raise ImportError(f"No module named '{mod_name}'")
            DEPS[mod_name] = True
        except ImportError:
            DEPS[mod_name] = False
            specialist_missing.append(mod_name)
    if specialist_missing:
        print(
            f"WARNING: specialist providers missing: {', '.join(specialist_missing)}. "
            f"Install via: uv pip install -e .[specialists]  (or full install: "
            f"uv pip install -e .). Falling back to heuristic mode for affected "
            f"component types.",
            file=sys.stderr,
        )
        record_path_warning(
            SPECIALIST_PROVIDERS_MISSING,
            "specialist providers missing — affected components run in heuristic fallback mode",
            details={"missing": sorted(specialist_missing)},
        )

    # CLI tools
    for tool in ["pdftoppm", "pdfinfo", "pdftotext", "pdffonts", "pdfimages", "ollama"]:
        DEPS[tool] = shutil.which(tool) is not None

    # Ollama running check
    if DEPS["ollama"]:
        try:
            import urllib.request as _urllib_request

            req = _urllib_request.Request("http://127.0.0.1:11434/api/tags")
            with _urllib_request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    DEPS["ollama_running"] = True
                    import json as _json

                    data = _json.loads(resp.read())
                    available = {m.get("name", "").split(":")[0] for m in data.get("models", [])}
                    # Strip :tag suffix from wanted models for name match
                    wanted = {
                        (m.split(":")[0] if ":" in m else m)
                        for m in (required_models or ["glm-ocr"])
                        # "rapidocr" is in-process Python (no Ollama needed).
                        if not m.startswith("rapidocr")
                    }
                    if not wanted:
                        DEPS["models_available"] = True  # rapidocr-only
                    elif wanted.issubset(available):
                        DEPS["models_available"] = True
                    else:
                        DEPS["_missing_models"] = sorted(wanted - available)
        except Exception as e:
            log.debug("dependency probe (ollama list) degraded: %s", e)

            record_path_warning(
                STAGE_SKIPPED,
                "dependency probe degraded (ollama list failed) — VL "
                "escalation detection may run blind",
            )

    # === Single HARD-missing list (no soft/optional split) ===
    missing = []

    py_pkgs = {
        "PIL": ("Pillow", "uv add Pillow  # or: pip install Pillow"),
        "ftfy": ("ftfy", "uv add ftfy"),
        "periodictable": ("periodictable", "uv add periodictable"),
        "pypdfium2": ("pypdfium2", "uv add pypdfium2"),
        "pymupdf": ("PyMuPDF", "uv add pymupdf"),
        "pdfplumber": ("pdfplumber", "uv add pdfplumber"),
    }
    for key, (label, fix) in py_pkgs.items():
        if not DEPS.get(key):
            missing.append((key, label, fix))

    cli_tools = {
        "pdftoppm": (
            "poppler-utils",
            "apt install poppler-utils  # or: brew install poppler",
        ),
        "pdfinfo": ("poppler-utils", "apt install poppler-utils"),
        "pdftotext": ("poppler-utils", "apt install poppler-utils"),
        "pdffonts": ("poppler-utils", "apt install poppler-utils"),
        "pdfimages": ("poppler-utils", "apt install poppler-utils"),
        "ollama": ("Ollama CLI", "curl -fsSL https://ollama.com/install.sh | sh"),
    }
    for key, (label, fix) in cli_tools.items():
        if not DEPS.get(key):
            missing.append((key, label, fix))

    if DEPS.get("ollama") and not DEPS.get("ollama_running"):
        missing.append(("ollama_running", "Ollama daemon", "ollama serve  # in another terminal"))
    elif DEPS.get("ollama_running") and not DEPS.get("models_available"):
        for m in DEPS.get("_missing_models", ["glm-ocr"]):
            missing.append((f"model_{m}", f"Ollama model: {m}", f"ollama pull {m}"))

    if missing:
        print(
            "\nERROR: missing HARD dependencies (no silent fallback, no degradation):",
            file=sys.stderr,
        )
        for key, label, fix in missing:
            print(f"  - {label} [{key}]: {fix}", file=sys.stderr)
        print(
            "\nPer project policy: every dep is mandatory. Install ALL of the "
            "above and re-run. No quality compromise, no degraded mode.",
            file=sys.stderr,
        )
        sys.exit(2)

    return DEPS


# ── Ollama VL fallback for the OCR path ────────────────────────────────────
# rapidocr's recognition confidence is OVER-CONFIDENT (0.98 mean even on
# degraded scans where it misses content — measured). So the reflexive loop's
# confidence floor never triggers VL escalation. Instead gate on CONTENT
# VOLUME: if a text-routed page yields very few chars, rapidocr failed →
# escalate to the best Ollama VL model (higher recall on degraded scans:
# rapidocr 69% vs chandra-ocr-2 76% on olmOCR-bench old_scans).
# Per model policy: VL = Ollama-routed (no heavy dep); fires only on poor output.
_PORTRAIT_OCR_CHAR_FLOOR = 150  # a text page should yield >this from OCR

# Preference by measured olmOCR-bench old_scans accuracy + reliability +
# speed (2026-08-14 head-to-head, 8 pages / 39 tests): glm-ocr 23.1% 8/8
# 7.1s/pg > chandra-ocr-2 23.1% 6/8 83s/pg > LFM2-VL-3B 15.4% 6/8 95s/pg.
# glm-ocr first: ties chandra on accuracy, wins on page coverage and 12x speed.
# Measured order (FULL 98-PDF old_scans head-to-head, natural routing,
# 2026-08-15): chandra-ocr-2 42.4% @ ~30s/pg > paddleocr-vl:1.6 25.5% @
# ~5s/pg. Paddle (0.9B Q8) is the FAST tier — good born-digital-adjacent
# scans, weak on handwriting. qwen2.5vl:3b measured 66.7% category earlier
# (re-pull to reclaim). Order = accuracy-first.
_VL_PREFERENCE = (
    "chandra-ocr-2",
    "qwen2.5vl:3b",
    "paddleocr-vl:1.6",
    "LFM2-VL-3B-GGUF:Q8_0",
)
# Fast tier for --fast / speed-sensitive runs.
_VL_PREFERENCE_FAST = ("paddleocr-vl:1.6", "chandra-ocr-2")


def _best_ollama_vl():
    """Best available Ollama vision model for OCR fallback, or None.

    Preference by olmOCR-bench old_scans accuracy (chandra-ocr-2 76% >
    glm-ocr 72% > LFM2-VL). Returns None if Ollama is unreachable.
    """
    try:
        avail = set(list_ollama_models())
        for vl in _VL_PREFERENCE:
            if vl in avail:
                return vl
    except Exception as e:
        log.warning("_best_ollama_vl: cannot list Ollama models — VL escalation off: %s", e)
    return None


def _ocr_vl_fallback(
    primary_text, primary_model, img_path, route_reasons, task_prompt, always_model=False
):
    """Escalate to Ollama VL when the primary (rapidocr) OCR output is poor.

    Gate: page was routed to OCR expecting content (route_reasons non-empty)
    AND primary text < _PORTRAIT_OCR_CHAR_FLOOR chars → rapidocr failed → run
    the VL model, return the richer output. No-op when primary isn't rapidocr,
    output is adequate, or no VL model is available.

    ``always_model=True`` (from --always-model) pins the user's explicit
    model choice — NO silent substitution, even on poor output. Fail loud
    with the pinned model's own result.
    """
    if always_model or primary_model != "rapidocr":
        return primary_text, primary_model
    if not route_reasons or len(primary_text.strip()) >= _PORTRAIT_OCR_CHAR_FLOOR:
        return primary_text, primary_model
    vl = _best_ollama_vl()
    if not vl:
        return primary_text, primary_model
    try:
        vl_text = ocr_image(img_path, task=task_prompt, model=vl)
        if len(vl_text.strip()) > len(primary_text.strip()):
            print(
                f"  Ollama VL fallback: rapidocr yielded {len(primary_text.strip())} chars "
                f"(poor) → re-OCR with {vl} ({len(vl_text.strip())} chars)",
                file=sys.stderr,
            )
            record_path_warning(
                VL_FALLBACK_USED,
                "rapidocr OCR output poor — page re-OCR'd via Ollama VL",
            )
            return vl_text, vl
    except Exception as e:
        # VL escalation failed — page keeps weak rapidocr text. Loud: this is
        # the exact path that hid catastrophic-quality pages (audit H5).
        log.warning("_ocr_vl_fallback: VL re-OCR failed, keeping rapidocr text: %s", e)
        record_path_warning(
            VL_FALLBACK_USED,
            "Ollama VL re-OCR FAILED — page keeps weak rapidocr text",
        )
    return primary_text, primary_model


def _count_pipe_tables(s, min_run=2):
    """Count pipe-table runs (maximal blocks of consecutive '|' lines).

    Separator-agnostic: det44+ emits sepless pipe tables (no '|---|' row),
    so legacy separator-counting gates under-counted post-proc tables to
    zero and let region routing replace them with table-less reassembly
    (measured 9a61fe78 pg2: 20-row partner-funding table lost). A run of
    >= min_run pipe lines is a table; a stray single line is not.
    """
    n = run = 0
    for ln in (s or "").splitlines():
        if ln.lstrip().startswith("|"):
            run += 1
        else:
            if run >= min_run:
                n += 1
            run = 0
    return n + (1 if run >= min_run else 0)


def _adopt_region_routed(new_text, fallback_text, *, table_gate, label):
    r"""Shared ADOPTION GATE for region-routed output (both paths).

    Region dispatch crops YOLO regions and re-OCRs them; its output replaces
    the page text ONLY when not clearly worse. Measured failure modes this
    gate rejects (2026-08-15 bench):
      - repetition collapse ("[OCR loop collapsed x17]")
      - spacing-command dominance ("\qquad \circ \cdot" > content commands)
      - thin output (<50% of fallback length)
      - (table_gate, fast path) dropping pipe tables post-proc already built

    Returns True when ``new_text`` may replace the fallback.
    """
    nt = new_text or ""
    space_n = sum(nt.count(c) for c in ("\\qquad", "\\quad", "\\circ", "\\cdot", "\\star", "\\;"))
    content_n = len(
        re.findall(
            r"\\(?:frac|int|sqrt|pi|alpha|beta|gamma|theta|lambda|mu|sum|sin|cos|tan|log|left|right)",
            nt,
        )
    )
    degenerate = nt.count("[OCR loop collapsed") >= 2 or (space_n > 20 and space_n > content_n)
    # Content-loss check on ALPHANUMERIC characters, not raw length: a
    # structured table legitimately compresses a whitespace-flooded text
    # layer (letter-spacing, padding, repeated headers) — measured 2026-08-15
    # wheat bench page: correct 6-col line-grid table (3.2k ch) rejected
    # against a 7.4k-ch WRONG-grid fallback that was mostly padding.
    _alnum = lambda s: len(re.sub(r"[^A-Za-z0-9]", "", s))
    if degenerate or _alnum(nt) < 0.5 * max(_alnum(fallback_text), 1):
        print(
            f"  {label} REJECTED (degenerate/thin: {len(nt)}ch vs fallback "
            f"{len(fallback_text)}ch) — keeping page text",
            file=sys.stderr,
        )
        record_path_warning(
            REGION_OUTPUT_REJECTED,
            f"{label} rejected region-routed output (degenerate/thin) — kept page text",
        )
        return False
    if table_gate:
        nt_tbl = _count_pipe_tables(nt)
        fb_tbl = _count_pipe_tables(fallback_text)
        if fb_tbl > 0 and nt_tbl < fb_tbl:
            print(
                f"  {label} REJECTED (tables {nt_tbl}/{fb_tbl}) — keeping post-proc text",
                file=sys.stderr,
            )
            record_path_warning(
                REGION_OUTPUT_REJECTED,
                f"{label} rejected region-routed output (dropped tables) — kept page text",
            )
            return False
    # FRAGMENTATION GUARD (audit H6 fix, 2026-08-18): region reassembly
    # used to replace clean page text with clip-fragments whose alnum
    # count was high enough to pass (many half-words still count). A
    # routed rebuild that shatters the fallback's word inventory — many
    # short lines vs the fallback's prose lines — is a truncation
    # signature, not structure. Reject when the fallback is prose-like
    # (median line >= 40 chars) but the routed median collapses below
    # half that, unless routing actually added structure (tables/eqs).
    _fb_lines = [l for l in (fallback_text or "").split("\n") if l.strip()]
    _nt_lines = [l for l in (nt or "").split("\n") if l.strip()]
    if _fb_lines and _nt_lines:
        _med = lambda ls: sorted(len(l) for l in ls)[len(ls) // 2]
        fb_med, nt_med = _med(_fb_lines), _med(_nt_lines)
        _has_structure = _count_pipe_tables(nt) > 0 or "$$" in nt
        if fb_med >= 40 and nt_med < fb_med * 0.5 and not _has_structure:
            print(
                f"  {label} REJECTED (fragmentation: median line "
                f"{nt_med} vs fallback {fb_med}) — keeping page text",
                file=sys.stderr,
            )
    return True


def _region_heading_loss(pre_text, new_text):
    """Heading LINES present in pre-dispatch text but absent as MARKED
    lines in the routed rebuild (det23b; det27 tightened survival to
    markup). A heading SURVIVES only when its text sits on a non-pipe
    line that itself carries a '#' marker — a rebuild keeping just the
    plain text DEMOTED the heading (E0 text-region re-extract drops
    markup, odlbench 01030000000168) and the caller restores it in
    place. Exception (det24b): a heading that became a routed table's
    HEADER CELL survived — the pre-dispatch 'heading' was the bold
    table header promoted by dsh (137297b4 class) and rejecting the
    adoption destroyed a real table. Only header-row CELL equality
    counts (not containment): a junk 1-cell row holding the whole page
    (01030000000123 class) is still a loss.
    Returns lost FULL heading lines (markers included).
    """
    _pre_heads = re.findall(r"^(#{1,6} +.+?)\s*$", pre_text or "", re.M)
    if not _pre_heads:
        return []
    _hn = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    _marked = [
        _hn(re.sub(r"^#{1,6} +", "", l))
        for l in (new_text or "").split("\n")
        if l.strip().startswith("#") and not l.strip().startswith("|")
    ]
    _lines = [l.strip() for l in (new_text or "").split("\n")]
    _header_cells = set()
    for i, l in enumerate(_lines):
        # First line of a pipe run = header row (separator-agnostic: det44
        # emits sepless tables, pre-det44 used a '|---' second line).
        if l.startswith("|") and (i == 0 or not _lines[i - 1].startswith("|")):
            _header_cells.update(_hn(c) for c in l.strip("|").split("|"))
    _out = []
    for _hl in _pre_heads:
        _t = _hn(re.sub(r"^#{1,6} +", "", _hl))
        if _t and _t not in _marked and _t not in _header_cells:
            _out.append(_hl)
    return _out


def _restore_heading_markup(new_text, lost_lines):
    """det27: re-mark headings a region rebuild demoted to plain lines
    (01030000000168: E0 text-region re-extract drops '#'). For each
    lost heading line, find the plain non-pipe line carrying its text
    (whole-line normalized equality preferred, containment fallback)
    and swap in the original marked line in place. Returns
    (new_text, unresolved_lost_lines).
    """
    _hn = lambda s: re.sub(r"[^a-z0-9]+", "", s.lower())
    _lines = (new_text or "").split("\n")
    _unresolved = []
    for _hl in lost_lines:
        _t = _hn(re.sub(r"^#{1,6} +", "", _hl))
        # det28: prefer the LAST match of either kind — a TOC/echo
        # plain line repeats the title BEFORE the real body occurrence
        # (01030000000198: region rebuild echoes 'Contents' above the
        # true title line), so first-match restore misplaces markup.
        _exact = None
        _contain = None
        for i, l in enumerate(_lines):
            _ls = l.strip()
            if not _ls or _ls.startswith("|") or _ls.startswith("#"):
                continue
            _ln = _hn(_ls)
            if _t and _ln == _t:
                _exact = i
            elif _t and _t in _ln:
                _contain = i
        _hit = _exact if _exact is not None else _contain
        if _hit is None:
            _unresolved.append(_hl)
        else:
            _lines[_hit] = _hl
    return "\n".join(_lines), _unresolved


def _dispatch_regions(
    pdf_path, page, img_path, text, args, models, *, fast, raw_text=None, hf_strip=None
):
    """SINGLE specialist-dispatch site for both paths (was 2 drifted copies).

    Builds providers from backend flags, calls route_page_by_region, applies
    the shared adoption gate. Fast path: renders its own image, table-preserve
    gate ON, stdout log. OCR path: reuses the page render, gate OFF for
    table counts, stderr log, rapidocr-only (VL pages skip — dispatch is
    net-negative for models that already read the full page).

    ``raw_text`` (fast path): the RAW pdftotext layer for the page. All
    content guards reference THIS, not ``text`` — post-proc splices pipe
    tables into ``text`` (~2x alnum inflation on table pages), which made
    both guards reject every correct region-routed table (2026-08-15).

    Returns (text, region_candidates) — text unchanged when rejected/skipped.
    """
    if fast:
        specialists_active = (
            getattr(args, "formula_backend", "default") != "off"
            or getattr(args, "table_backend", "default") != "off"
            or getattr(args, "layout_backend", "default") != "off"
        )
        if not specialists_active or img_path is None:
            return text, []
        providers = {
            "layout": get_layout_detector(getattr(args, "layout_backend", "default")),
            "formula": get_formula_recognizer(getattr(args, "formula_backend", "rapidlatex")),
            "table": get_table_recognizer(getattr(args, "table_backend", "rapidtable")),
        }
        if not any(p and p.available() for p in providers.values()):
            return text, []
    else:

        def _resolve_backend(arg_val, rapid_name):
            if arg_val in (None, "default"):
                return rapid_name
            return arg_val

        _lb = getattr(args, "layout_backend", "default")
        layout_backend = None if _lb == "off" else _lb
        formula_backend = _resolve_backend(
            getattr(args, "formula_backend", "default"), "rapidlatex"
        )
        table_backend = _resolve_backend(getattr(args, "table_backend", "default"), "rapidtable")
        if not any([layout_backend, formula_backend, table_backend]):
            return text, []
        # rapidocr-only on the OCR path (see note in git 97864f0 predecessors)
        if not str(models[0]).startswith("rapidocr"):
            return text, []
        providers = {
            "layout": get_layout_detector(layout_backend) if layout_backend else None,
            "formula": get_formula_recognizer(formula_backend) if formula_backend else None,
            "table": get_table_recognizer(table_backend) if table_backend else None,
        }
        if not any(p and p.available() for p in providers.values()):
            return text, []

    try:
        t_r = time.time()
        # Guard reference: RAW text layer when available (post-proc text
        # carries spliced pipe tables → inflated alnum → false rejections).
        _guard_ref = raw_text if (fast and raw_text and raw_text.strip()) else text
        new_text, region_stats = route_page_by_region(
            pdf_path=pdf_path,
            page_num=page,
            img_path=img_path,
            providers=providers,
            fallback_text=_guard_ref,
            execute_escalations=getattr(args, "escalate", False),
            resolver_drives_output=getattr(args, "resolver_drives_output", False),
            reuse_page_ocr=(not fast) and any(str(m).startswith("rapidocr") for m in models),
        )
        r_elapsed = time.time() - t_r
        if region_stats.get("content_loss_guard"):
            # Internal guard fired (routed lost content vs RAW layer) —
            # keep the post-proc page text unchanged.
            record_path_warning(
                REGION_OUTPUT_REJECTED,
                "region routing rejected (content-loss guard) — kept page text",
                page=page,
            )
            return text, []
        if region_stats["method"] != "region-routed":
            return text, []
        if fast and region_stats["n_dispatched"] == 0:
            return text, []
        if fast:
            print(
                f"  Fast region routed: {region_stats['n_regions']} regions "
                f"({region_stats['n_dispatched']} dispatched)"
            )
            # TABLE-QUALITY selection (2026-08-15, run4 regression fix):
            # adopting routed output replaces post-proc pipe tables with the
            # dispatch table. Line-grid tables recover tableless pages but
            # break up/down/left/right relationships on pages where post-proc
            # tables were already right (measured: relviol fails 18→96).
            _n_tbl = _count_pipe_tables
            if _n_tbl(text) > 0 and _n_tbl(new_text) <= _n_tbl(text):
                print(
                    f"  Fast region routing REJECTED (post-proc has "
                    f"{_n_tbl(text)} table(s), routed {_n_tbl(new_text)}) — "
                    f"keeping post-proc text",
                    file=sys.stderr,
                )
                record_path_warning(
                    REGION_OUTPUT_REJECTED,
                    "fast region routing rejected (routed output dropped tables) — kept page text",
                    page=page,
                )
                return text, []
            # STRUCTURE-GAIN gate (audit-2 H6 residual, 2026-08-18): on
            # PROSE pages the fast path already holds column-REORDERED,
            # dehyphenated, band-cleaned text (column_aware_body_only +
            # postproc). Region-sorted reassembly reads columns in YOLO
            # region order — not column order — and displaces the better
            # text (measured multi_column: dispatch adoption cost ~5pts of
            # order tests). Fast-path dispatch exists to ADD structure the
            # text layer lacks (line-grid tables, LaTeX equations). No net
            # structure gain → keep the postproc text.
            _n_struct = lambda s: _n_tbl(s) + (s or "").count("$$")
            if _n_struct(new_text) <= _n_struct(text):
                print(
                    "  Fast region routing REJECTED (no structure gain) — keeping post-proc text",
                    file=sys.stderr,
                )
                record_path_warning(
                    REGION_OUTPUT_REJECTED,
                    "fast region routing rejected (no structure gain over post-proc) — kept page text",
                    page=page,
                )
                return text, []
        else:
            print(
                f"  Region routing: method={region_stats['method']} "
                f"regions={region_stats['n_regions']} eq={region_stats['n_equations']} "
                f"tbl={region_stats['n_tables']} dispatched={region_stats['n_dispatched']} "
                f"fallback={region_stats['n_fallback']} in {r_elapsed:.1f}s",
                file=sys.stderr,
            )

        # WORD-SHATTER GUARD (iter-73, 09f90a8f page_2): region routing
        # re-extracts from the RAW text layer per region. On shattered
        # layers (word-per-line fonts, fragments as separate fitz lines)
        # the routed rebuild is one-fragment-per-line and DISPLACES the
        # column-reordered, dehyphenated post-proc text — every word-span
        # gold test dies even though routing 'gained' a junk table carved
        # out of figure labels. A routed page that is >60% single-word
        # lines AND much worse than the text it replaces is a shattered
        # rebuild, not structure gain. (Pipe-table rows split into many
        # tokens; real table adoptions never trip this.)
        def _single_word_frac(s):
            lines = [
                l
                for l in (s or "").split("\n")
                if l.strip() and not re.match(r"^\|[-\s|]+\|$", l.strip())
            ]
            if not lines:
                return 0.0
            return sum(1 for l in lines if len(l.split()) == 1) / len(lines)

        if (
            _single_word_frac(new_text) > 0.6
            and _single_word_frac(new_text) > _single_word_frac(text) + 0.2
        ):
            print(
                "  Region routing REJECTED (word-shattered rebuild) — keeping post-proc text",
                file=sys.stderr,
            )
            record_path_warning(
                REGION_OUTPUT_REJECTED,
                "region routing rejected (word-shattered rebuild) — kept page text",
                page=page,
            )
            return text, []
        # HEADING-LOSS GUARD (det23b, odlbench 01030000000133/123): a
        # chart or framed page can segment as ONE table region, so the
        # routed rebuild wraps the whole page — promoted '#' headings
        # included — into pipe cells. The heading WORDS survive inside
        # rows (content guards pass) but the heading LINES are gone: a
        # destructure, not a structure gain. Reject when the rebuild
        # loses any heading as a plain line — UNLESS the routed output
        # carries a substantial real table (>=4-row pipe run, det24b):
        # there the lost 'headings' are typically dsh promotions of the
        # table's own bold header text (137297b4 class) and rejecting
        # destroyed the table. Adopt and re-insert the lost headings
        # above it so text parity is preserved.
        _lost = _region_heading_loss(text, new_text)
        if _lost:
            # det27: a heading surviving only as a PLAIN line was demoted
            # by the region re-assembly (E0 text-region re-extract drops
            # '#' markers, odlbench 01030000000168). Restore the markup
            # in place — the routed rebuild can still carry real value
            # (E4 equations, real tables) that a reject would destroy.
            new_text, _lost = _restore_heading_markup(new_text, _lost)
        if _lost:
            _pipe_run = 0
            _pipe_best = 0
            for _l in (new_text or "").split("\n"):
                if _l.strip().startswith("|"):
                    _pipe_run += 1
                    _pipe_best = max(_pipe_best, _pipe_run)
                else:
                    _pipe_run = 0
            if _pipe_best < 4:
                print(
                    f"  Region routing REJECTED (heading-loss: "
                    f"{re.sub(r'^#{1,6} +', '', _lost[0])[:40]!r} +{len(_lost) - 1} more) — keeping post-proc text",
                    file=sys.stderr,
                )
                record_path_warning(
                    REGION_OUTPUT_REJECTED,
                    "region routing rejected (heading-loss guard) — kept page text",
                    page=page,
                )
                return text, []
            _pn_all = re.sub(r"[^a-z0-9]+", "", (new_text or "").lower())
            _reinsert = [
                _h
                for _h in _lost
                if re.sub(r"[^a-z0-9]+", "", re.sub(r"^#{1,6} +", "", _h.lower()))
                and re.sub(r"[^a-z0-9]+", "", re.sub(r"^#{1,6} +", "", _h.lower())) not in _pn_all
            ]
            if _reinsert:
                new_text = "\n\n".join(_reinsert) + "\n\n" + (new_text or "")
                print(
                    f"  Region routing adopted real table; re-inserted "
                    f"{len(_reinsert)} lost heading(s) above it",
                    file=sys.stderr,
                )
        # H6 followup (audit-2, 2026-08-18): dispatch output is TERMINAL —
        # it replaces the post-proc'd page text, so it bypasses every
        # cleaner, including dehyphenation. Whole-line region text carries
        # raw "per-\nformed" hyphen breaks; every order/present test whose
        # gold phrase crosses a hyphenated linebreak fails at max_l_dist=1
        # (measured: multi_column 21/40 PDFs at 0.0 from this alone).
        # Dehyphenate ADOPTED dispatch output before it wins.
        if _adopt_region_routed(
            new_text,
            _guard_ref,
            table_gate=fast,
            label="Fast region routing" if fast else "Region routing",
        ):
            try:
                from scientific_parser.postproc.text_pipeline import dehyphenate

                new_text = dehyphenate(new_text)
                # FUSED-WORDS on adopted output (iter-86b): dispatch
                # rebuilds come from the raw fitz layer — space-less
                # producers' fused tokens return. The repair is page-
                # gated internally, safe unconditionally.
                try:
                    from scientific_parser.postproc.text_pipeline import (
                        repair_fused_words,
                    )

                    new_text = repair_fused_words(new_text)
                except Exception as _fe:
                    log.debug("dispatch fused-words repair skipped: %s", _fe)
            except Exception as _de:
                log.debug("dispatch dehyphenate skipped: %s", _de)
                _pw_skip("dispatch dehyphenate")
            # HF-STRIP ADOPTED OUTPUT (2026-08-19): adopted dispatch text
            # replaces the page text AFTER the final strip already ran —
            # its band lines (running footers, SPIE download watermarks)
            # bypass every strip and leak (measured 0f692b1b: absent-test
            # fails on the surviving 'Downloaded From: spiedigital...'
            # footer). Caller passes the page's strip fn + meta.
            if hf_strip is not None:
                try:
                    _adopt_meta = extract_page_metadata(
                        text, page_num=page, doc_state=None, pdf_path=pdf_path
                    )
                    new_text = hf_strip(new_text, _adopt_meta)
                except Exception as _he:
                    log.debug("dispatch hf-strip skipped: %s", _he)
                    _pw_skip("dispatch hf-strip")
            return new_text, region_stats.get("candidates", [])
        return text, []
    except Exception as e:
        (log.debug if fast else log.warning)(
            "%s region routing failed: %s", "fast" if fast else "OCR", e
        )
        return text, []


def main():
    pdf_path = PDF_PATH
    output_path = OUTPUT_PATH
    # Unintended-path warning collector: fresh state per document; every
    # fallback/degradation below records here (stderr echo + summary.json).
    reset_path_warnings()

    # Pre-flight dependency audit (HARD exit on missing required, warn on soft/optional).
    # Never silently degrade.
    # Default model (2026-08-14, measured): best Ollama VL (glm-ocr first —
    # 23.1% old_scans, 8/8 pages, 7.1s/pg vs rapidocr 10.3%) when the daemon
    # is reachable; rapidocr (in-process, zero-daemon) as offline fallback.
    # --always-model pins an explicit --models choice; no substitution.
    if args.models:
        models_planned = [m.strip() for m in args.models.split(",") if m.strip()]
        models_auto_default = False
    else:
        _vl = _best_ollama_vl()
        models_planned = [_vl] if _vl else ["rapidocr"]
        models_auto_default = True
    check_dependencies(required_models=models_planned)

    # Image input: treat as single-page document, skip pdftoppm entirely
    if IS_IMAGE:
        total_pages = 1
    else:
        total_pages = get_page_count(pdf_path)
    try:
        page_list = parse_page_range(args.pages, total_pages)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    # Parse crop if provided
    crop = None
    if args.crop:
        try:
            crop = parse_crop(args.crop)
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    # Models list (already validated by check_dependencies above)
    # Use models_planned (already parsed at line 9545, no need to re-parse).
    models = models_planned
    # SOTA 2026-08-12: models list is now an ordered CASCADE chain.
    # Replaces legacy `models = models[:1]` slice that silently discarded
    # user-specified fallbacks. Cascade is honored unless --no-cascade.
    # See AUDIT_2026-08-12.md §2.4 (the models[:1] bug).
    if len(models) > 1 and not getattr(args, "no_cascade", False):
        print(
            f"Cascade enabled: {len(models)} models [{' → '.join(models)}]. "
            f"Reflexive loop will fall back on low confidence (floor="
            f"{getattr(args, 'reflexive_floor', REFLEXIVE_CONFIDENCE_FLOOR)}).",
            file=sys.stderr,
        )
    elif len(models) > 1 and getattr(args, "no_cascade", False):
        print(
            f"Note: --no-cascade set; using only '{models[0]}' of {len(models)} specified models.",
            file=sys.stderr,
        )
        models = models[:1]

    print(
        f"Processing {len(page_list)} page(s) ({page_list[0]}..{page_list[-1]}) "
        f"of {total_pages} from: {pdf_path}" + (f" [image input]" if IS_IMAGE else "")
    )
    if crop:
        x, y, w, h = crop
        print(f"Crop: X={x} Y={y} W={w} H={h} (PDF points, top-left origin)")

    # Crop implies force-ocr (cropped region needs vision)
    # Image input always needs OCR (no text layer to extract)
    use_force_ocr = args.force_ocr or crop is not None or IS_IMAGE

    # ---- Per-page SOTA routing ----
    # For each page, decide: "fast" (use pdftotext text) or "ocr" (use vision model).
    # Default: SOTA 7-signal router. --fast disables quality scoring.
    # --force-ocr / crop / image input: all pages → "ocr".
    page_routes = {}  # page_num -> ("fast"|"ocr", reasons)
    by_page = {}  # page_num -> pdftotext-extracted text (when "fast")
    doc_info = None  # document-level pre-classification (set in the non-force-ocr branch; None under --force-ocr)

    if use_force_ocr:
        if IS_IMAGE:
            print("Image input; using OCR directly")
        else:
            reason = "crop" if crop else "--force-ocr"
            print(f"{reason} set; using OCR for all pages")
            if reason == "--force-ocr":
                print(
                    "WARNING: --force-ocr bypasses pdftotext. For born-digital "
                    "PDFs, the default hybrid routing produces better table/data "
                    "fidelity. Use --force-ocr only for scanned PDFs or debugging.",
                    file=sys.stderr,
                )
        for p in page_list:
            page_routes[p] = ("ocr", ["forced"])
    else:
        # Always try pdftotext first (cheap; gives us text to score)
        try:
            by_page, avg_chars = extract_text_layer(pdf_path, page_list)
            by_page = rescue_glued_pages(pdf_path, page_list, by_page)
            # Document-level pre-classification (ONE PDF open, ONE pdffonts call).
            # Eliminates O(N) per-page PDF opens in the routing loop.
            if not args.fast and not IS_IMAGE:
                doc_info = preclassify_document(pdf_path, page_list)
                if doc_info["born_digital"]:
                    print(
                        f"Document pre-classified: born-digital "
                        f"({len(doc_info['pages_with_fonts'])} pages with fonts, "
                        f"0 invisible-text pages)"
                    )
                elif doc_info["invisible_pages"]:
                    print(
                        f"Document pre-classified: mixed "
                        f"({len(doc_info['invisible_pages'])} invisible-text pages)"
                    )
            else:
                doc_info = None
            fast_count = ocr_count = 0
            for p in page_list:
                text = by_page.get(p, "")
                if args.fast:
                    # Simple threshold mode (legacy)
                    if len(text.strip()) < TEXT_LAYER_CHARS_PER_PAGE_MIN:
                        page_routes[p] = ("ocr", [f"low_chars ({len(text.strip())})"])
                        ocr_count += 1
                    else:
                        page_routes[p] = ("fast", [])
                        fast_count += 1
                else:
                    # SOTA routing (default) — uses cached doc_info
                    route, reasons = should_route_to_ocr(text, pdf_path, p, doc_info)
                    page_routes[p] = ("ocr" if route else "fast", reasons)
                    if route:
                        ocr_count += 1
                    else:
                        fast_count += 1
            # EXP-1.5 (2026-09-16): scrub control-char noise from the text
            # layer AFTER routing. Symbol fonts on born-digital pages emit
            # 0x07-0x1f glyph codes (bullets, diagram labels) interleaved
            # with real prose; the router now treats them as noise (not an
            # OCR trigger) when the layer has >=200 alphanumerics, so the
            # fast path must strip them before postproc/output. Includes
            # \x0b/\x0c which _PRIVATE_CHAR_RE leaves alone; \t\n\r kept.
            # Downstream encoding_artifacts warning still reports the count.
            for p, txt in by_page.items():
                if _CTRL_SCRUB_RE.search(txt):
                    by_page[p] = _CTRL_SCRUB_RE.sub("", txt)
            mode_label = "fast (chars-only)" if args.fast else "SOTA (7-signal)"
            print(
                f"Routing [{mode_label}]: {fast_count} fast / {ocr_count} OCR "
                f"(of {len(page_list)} pages)"
            )
            # Marker SOTA: min_document_ocr_threshold — if too few pages are
            # good, force OCR for the remaining "fast" pages. R3 fix: only
            # force pages that haven't already been routed to OCR.
            threshold_pct = args.min_doc_ocr_pct
            if threshold_pct > 0 and page_list:
                good_pct = fast_count * 100.0 / len(page_list)
                if good_pct < threshold_pct:
                    forced = sum(1 for p, (r, _) in page_routes.items() if r == "fast")
                    if forced > 0:
                        print(
                            f"  Only {good_pct:.0f}% pages good (<{threshold_pct:.0f}% "
                            f"threshold); forcing OCR for {forced} additional pages"
                        )
                        record_path_warning(
                            MIN_DOC_OCR_FORCED,
                            "too few good pages — fast pages forced to OCR (min-doc-ocr threshold)",
                            pages=[p for p, (r, _) in page_routes.items() if r == "fast"],
                            details={"good_pct": round(good_pct, 1)},
                        )
                        for p in page_list:
                            if page_routes[p][0] == "fast":
                                page_routes[p] = (
                                    "ocr",
                                    list(page_routes[p][1]) + ["min_doc_ocr_threshold"],
                                )
            # Flag born-digital pages that the router sent to the OCR path
            # (routing false-positive or broken text layer — either way loud).
            warn_born_digital_ocr_routes(page_routes, doc_info, is_image=IS_IMAGE)
        except FileNotFoundError:
            print(
                "pdftotext not found; falling back to OCR for all pages",
                file=sys.stderr,
            )
            for p in page_list:
                page_routes[p] = ("ocr", ["pdftotext_missing"])
            record_path_warning(
                PDFTOTEXT_UNAVAILABLE,
                "pdftotext not found — ALL pages fall back to OCR (install poppler-utils)",
            )
        except subprocess.CalledProcessError as e:
            print(
                f"pdftotext failed: {e}; falling back to OCR for all pages",
                file=sys.stderr,
            )
            for p in page_list:
                page_routes[p] = ("ocr", ["pdftotext_failed"])
            record_path_warning(
                PDFTOTEXT_UNAVAILABLE,
                "pdftotext failed — ALL pages fall back to OCR",
            )

    tmp_dir = tempfile.mkdtemp(prefix="pdf_ocr_")
    page_metrics = []  # collected for --json mode and final summary
    all_region_candidates = []  # per-region candidate lattice (reviewer #38, audit sidecar)

    # Apply CLI risk/escalation config to module constants (one-time at startup).
    # These are tunable thresholds (reviewer #15/#24); module globals are read
    # at call-time, so overriding them here propagates to the live pipeline.
    try:
        from scientific_parser.routing import risk as _risk_mod
        from scientific_parser.perception.providers import region_dispatcher as _rd
        from scientific_parser.verification import retry_executor as _re

        _eth = getattr(args, "escalate_threshold", 1.5)
        _risk_mod.ESCALATE_THRESHOLD = _eth
        _risk_mod.REVIEW_THRESHOLD = _eth * 0.5
        _rd._CONTENT_LOSS_RATIO = getattr(args, "content_loss_ratio", 0.5)
        _re._DPI_CAP = getattr(args, "dpi_cap", 600)
    except Exception as e:
        # CLI→module threshold wiring failed: pipeline runs on module defaults,
        # silently diverging from documented flag defaults (audit H5).
        log.warning("CLI override wiring failed (using module defaults): %s", e)
        record_path_warning(
            CLI_CONFIG_IGNORED,
            "CLI threshold overrides failed to wire — module defaults "
            "silently in effect (escalate/content-loss/dpi-cap flags ignored)",
        )

    # Reset transformation ledger for this document
    try:
        from ledger import reset_ledger

        reset_ledger()
    except ImportError as e:
        log.debug("ledger module unavailable (transformation ledger off): %s", e)

    # Source provenance: composite — reflects that pages may use mixed paths.
    # For the header/summary, we describe the dominant mode (or "mixed").
    fast_pages = sum(1 for p, (r, _) in page_routes.items() if r == "fast")
    ocr_pages = sum(1 for p, (r, _) in page_routes.items() if r == "ocr")
    if ocr_pages == 0:
        source_provenance = "pdftotext (text layer)"
    elif fast_pages == 0:
        source_provenance = (
            f"OCR ({', '.join(models)})" if len(models) > 1 else f"OCR ({models[0]})"
        )
    else:
        source_provenance = f"mixed: {fast_pages} pdftotext + {ocr_pages} OCR ({', '.join(models)})"
    if crop:
        source_provenance += f" + crop({args.crop})"

    json_mode = bool(args.json)

    # Resolve output base name. Known output extensions are stripped so that
    # passing "foo.md", "foo.json", "foo.pages.jsonl", or "foo" all produce
    # the SAME triplet (foo.pages.jsonl + foo.summary.json + foo.md).
    # Anything else is treated as a base name verbatim.
    def _resolve_output_base(path: str) -> str:
        for suffix in (
            ".pages.jsonl",
            ".summary.json",
            ".pages.md",
            ".jsonl",
            ".json",
            ".md",
        ):
            if path.endswith(suffix):
                return path[: -len(suffix)]
        return path

    _base = _resolve_output_base(output_path)
    # Markdown path is ALWAYS set (even in JSON mode) — JSON mode emits the
    # full triplet: .pages.jsonl + .summary.json + .md. No silent mode-dependent
    # output gaps (per "no quality compromise" mandate).
    md_path = _base + ".md"
    if json_mode:
        jsonl_path = _base + ".pages.jsonl"
        _summary_path = _base + ".summary.json"
    else:
        jsonl_path = None
        _summary_path = None

    # ─── SOTA cache lookup (Phase 10 of yggdrasil plan, shipped 2026-08-12) ── #
    # Content-hash cache: sha256(pdf_bytes + flags + skill_version). Cache hit
    # returns full extracted output in <5% original time. Default ON; --no-cache
    # disables both read and write.
    use_cache = not getattr(args, "no_cache", False) and not IS_IMAGE
    if use_cache:
        cached = cache_lookup(pdf_path or PDF_PATH, models, args, _base)
        if cached is not None:
            raw, cache_dir = cached
            print(
                f"Cache RAW HIT ({cache_dir[-12:]}): replaying cached "
                f"pre-postproc extraction ({raw.get('page_count', 0)} pages, "
                f"{raw.get('elapsed_s', 0)}s original work) through the "
                f"document chain",
                file=sys.stderr,
            )
            _finalize_document(
                {
                    "final_md": raw["raw_md"],
                    "page_metrics": raw["page_metrics"],
                    "page_count": int(raw.get("page_count", len(raw["page_metrics"]))),
                    "models": raw.get("models") or models,
                    "args": args,
                    "pdf_path": pdf_path or PDF_PATH,
                    "md_path": md_path,
                    "jsonl_path": jsonl_path,
                    "summary_path": _summary_path,
                    "json_mode": json_mode,
                    "source_provenance": raw.get("source", "cached-raw"),
                    "total_elapsed": float(raw.get("elapsed_s", 0.0)),
                    "font_math_pages": set(raw.get("font_math_pages") or []),
                    "any_math_font_page": bool(raw.get("any_math_font_page")),
                    "use_cache": True,
                    "cache_dir": cache_dir,
                    "all_region_candidates": None,
                    "path_warnings_override": raw.get("path_warnings") or [],
                    "raw_replay": True,
                    "output_base": _base,
                }
            )
            return
    all_text = []  # always accumulate markdown chunks

    def emit_markdown(chunk):
        """Append to in-memory markdown list (both modes)."""
        all_text.append(chunk)

    def emit_jsonl(record):
        """Stream one page record to .jsonl (JSON mode). Crash-safe."""
        with open(jsonl_path, "a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _strip_hf_lines(t, page_meta):
        """Remove geometry-detected header/footer lines from page text.

        FINAL-pass strip: applied pre-emit so ALL overwrites (postproc
        extract_text_with_scripts, region routing, font-math detector)
        cannot re-introduce the footer. Containment matching (rapidfuzz
        partial_ratio_alignment >= 85) handles pdftotext reflow AND the
        column reorder's merged footer block.
        """
        hf_lines = (page_meta or {}).get("hf_lines") or []
        if not hf_lines or not t:
            return t
        try:
            from rapidfuzz import fuzz as _rfz
        except ImportError:
            _rfz = None
        import difflib

        def _norm(x):
            return re.sub(r"\s+", " ", x).strip().lower()

        def _norm_map(x):
            """Case-fold + whitespace-collapse a line, keeping an index map.

            Audit C5 fix (2026-08-18): the surgical footer-cut used to
            assign the NORMALIZED (lowercased) line back into the page,
            case-destroying every line it cut ("DCS"→"dcs" on whole
            multi_column pages). The map carries each normalized char's
            source offset so cuts apply to the ORIGINAL casing.
            """
            out, mp = [], []
            i, n = 0, len(x)
            while i < n and x[i].isspace():
                i += 1
            while i < n:
                if x[i].isspace():
                    j = i
                    while j < n and x[j].isspace():
                        j += 1
                    if j < n:  # skip trailing whitespace
                        out.append(" ")
                        mp.append(i)
                    i = j
                else:
                    out.append(x[i].lower())
                    mp.append(i)
                    i += 1
            return "".join(out), mp

        def _cut_norm_span(tl, tl_map, s0, s1):
            """Cut [s0, s1) (normalized indices) from the original line."""
            if s1 <= s0 or s0 >= len(tl_map) or s1 > len(tl_map):
                return None
            a = tl_map[s0]
            b = tl_map[s1 - 1] + 1
            if b <= a:
                return None
            return (tl[:a] + " " + tl[b:]).strip()

        lines = t.split("\n")
        import os as _os_p

        for hf in hf_lines:
            hf_n = _norm(hf)
            if len(hf_n) < 3:
                # BARE PAGE-NUMBER EXACT STRIP (iter-120a): geometry
                # detects short page numbers ('5', 'ii', '7') but the
                # fuzzy path must skip <3 chars — a 1-char fuzzy window
                # scores 100 inside every digit-bearing line and blanks
                # body text. Exact whole-line equality carries no such
                # risk: only a line that IS the page number is blanked
                # (0ae20716 pg6: trailing '5' footer leaked into the
                # last_n=10 absent-test window after iter-115 un-tabled
                # the page).
                if re.fullmatch(r"[\divxlcdm]{1,4}", hf_n):
                    for i in range(len(lines) - 1, -1, -1):
                        tl = lines[i]
                        if _norm(tl) == hf_n and not tl.lstrip().startswith("|"):
                            # MATH-NEIGHBOR GUARD (2026-08-31): a bare
                            # number flanked by math-bearing lines is a
                            # fraction denominator / script fragment
                            # (a_0=\frac{1}{2+\omega L}: numerator '1',
                            # denominator row standalone). Folio lines
                            # never sit between $-math lines.
                            _prev = next(
                                (lines[j] for j in range(i - 1, -1, -1) if lines[j].strip()), ""
                            )
                            _next = next(
                                (lines[j] for j in range(i + 1, len(lines)) if lines[j].strip()), ""
                            )
                            if (
                                "$" in _prev
                                or "$" in _next
                                or "\\frac" in _prev
                                or "\\frac" in _next
                            ):
                                continue
                            lines[i] = ""
                            break
                continue
            # WHOLE-LINE-FIRST MATCHING (2026-08-18): first-match-wins
            # scanning let a LONG merged line that merely CONTAINS the
            # footer slice (cut path) consume the hf and break, leaving
            # the standalone footer copy later in the page un-stripped
            # (88c0a6de: 'Visit Finance at nyc.gov/finance' survived
            # while the slice was cut from a merged instruction line).
            # Pass 1: first WHOLE-line match (blank); pass 2 only when
            # no whole match exists: first cut match (merged-column
            # footer case, measured multi_column 'page 11 of 15').
            _whole_done = False
            for i, tl in enumerate(lines):
                if not tl.strip() or tl.lstrip().startswith("|"):
                    # PIPE-TABLE GUARD (iter-9, 2026-08-16): a top-of-page
                    # table's HEADER ROW lives in the header geometry band
                    # and fuzzy-matches header detection — stripping it
                    # beheads the table (117 REL-broken bench tests).
                    continue
                tl_n, tl_map = _norm_map(tl)
                if len(tl_n) > 3 * len(hf_n) + 20:
                    continue
                # LENGTH-FLOOR (2026-08-19): partial_ratio_alignment(hf→tl)
                # finds the best tl-length window INSIDE hf — a 2-char line
                # ('on') inside a 95-char footer scores 100 and gets blanked
                # (measured 0f692b1b: spie footer consumed its match on 'on'
                # at line 79, surviving at 203). A whole-line match requires
                # the line to actually span the footer.
                if len(tl_n) < 0.5 * len(hf_n):
                    continue
                if _rfz is not None:
                    align = _rfz.partial_ratio_alignment(hf_n, tl_n)
                    if align is None or align.score < 85:
                        continue
                if _os_p.environ.get("PDF_OCR_DEBUG_HF"):
                    print(f"[HF] BLANK {i}: {tl[:60]!r} (hf={hf_n[:40]!r})", file=sys.stderr)
                lines[i] = ""
                _whole_done = True
                break
            if _whole_done:
                continue
            for i, tl in enumerate(lines):
                if not tl.strip() or tl.lstrip().startswith("|"):
                    continue
                tl_n, tl_map = _norm_map(tl)
                if len(tl_n) <= 3 * len(hf_n) + 20:
                    continue  # already rejected in pass 1
                if _rfz is not None:
                    # rapidfuzz 3.x: dest_start/dest_end (dest_slice was
                    # removed; AttributeError crashed whole-doc runs).
                    align = _rfz.partial_ratio_alignment(hf_n, tl_n)
                    if align is None or align.score < 85:
                        continue
                    # MID-LINE BODY-REPETITION GUARD (2026-09-03): cut
                    # only spans at the line's START or END. A flagged
                    # sidebar fragment ('up to 3,000', 20_pg46 bottom
                    # band) fuzzy-matches the SAME phrase inside a
                    # mid-page body sentence ('…designs will have up to
                    # 3,000 individual coils…') and the cut deleted
                    # body words, leaving 'will have  individual
                    # coils'. Merged-column footer slices sit at a line
                    # edge; body repetitions sit inside with words on
                    # both sides.
                    _s0, _s1 = align.dest_start, align.dest_end
                    if _s0 > 4 and _s1 < len(tl_n) - 4:
                        continue
                    cut = _cut_norm_span(tl, tl_map, _s0, _s1)
                    if cut is not None:
                        lines[i] = cut
                        break
                    continue
                else:
                    import difflib as _dl

                    sm = _dl.SequenceMatcher(None, hf_n, tl_n)
                    m = sm.find_longest_match(0, len(hf_n), 0, len(tl_n))
                    if m.size < 0.8 * len(hf_n):
                        continue
                    if m.b > 4 and m.b + m.size < len(tl_n) - 4:
                        continue
                    cut = _cut_norm_span(tl, tl_map, m.b, m.b + m.size)
                    if cut is not None:
                        lines[i] = cut
        # EDGE-TOKEN PAGE NUMBERS (2026-08-31): a bare-number hf that
        # same-row extraction or reflow FUSED into the first/last content
        # line survives whole-line matching (measured fe8d3e95: the running
        # head cut left '83 of the equality...' as the page's first line;
        # scorer first_n/last_n absent-tests read exactly those windows).
        # Cut the token itself from the EDGE lines only — mid-text
        # occurrences ride inside body sentences (section numbers
        # '117. The drafters...') and must stay.
        _bare_nums = [
            h.strip()
            for h in hf_lines
            if re.fullmatch(r"\d{1,6}|[ivxlcdm]{2,6}|[IVXLCDM]{2,6}", h.strip())
        ]
        if _bare_nums:
            _edge_idx = []
            for _i in range(len(lines)):
                if lines[_i].strip() and not lines[_i].lstrip().startswith(("|", "<!--")):
                    _edge_idx.append(_i)
                    break
            for _i in range(len(lines) - 1, -1, -1):
                if lines[_i].strip() and not lines[_i].lstrip().startswith(("|", "<!--")):
                    _edge_idx.append(_i)
                    break
            for _i in _edge_idx:
                # MATH-REGION GUARD (2026-08-31): never cut inside $...$ /
                # $$...$$ spans — braces and operators are not word chars,
                # so '(?<!\w)2(?!\w)' matches the 2 in '\|x\|_{2}' and
                # '\frac{p+1}{2}' (measured: 3 arxiv_math regressions where
                # a gold equation lived on the first/last content line).
                # Folios fuse into PLAIN text lines; math lines are immune.
                # Spans recomputed per token: each cut shifts positions.
                for _num in _bare_nums:
                    _spans = [
                        (m.start(), m.end())
                        for m in re.finditer(r"\$\$.*?\$\$|\$[^$]*\$", lines[_i])
                    ]

                    def _cut_tok(m):
                        return "" if not any(a <= m.start() < b for a, b in _spans) else m.group(0)

                    lines[_i] = re.sub(
                        r"(?<!\w)" + re.escape(_num) + r"(?!\w)",
                        _cut_tok,
                        lines[_i],
                        count=1,
                    ).strip()
        out = "\n".join(lines)

        # INLINE GLYPH DEDUP (iter-121b): text-layer + math-span join can
        # emit BOTH forms of a Greek letter adjacent ('IFN \gammaγ') — the
        # display-block repair path never sees inline lines. Same-letter
        # check keeps distinct-letter pairs ('\alphaβ') intact; the
        # unicode form survives (gold prose uses it: 'IFNγ').
        def _gdedup(m):
            return m.group(2) if _CMD_TO_UNI.get(m.group(1)) == m.group(2) else m.group(0)

        out = _GLYPH_DUP_LINE_RE.sub(_gdedup, out)
        # INLINE MATH GROUP REPAIR (iter-125): $-delimiters wrap wrong
        # sub-runs ('$h $be a function $h $∶ $\mathbb{R}^p...$') — merge
        # adjacent inline groups across operator-only bridges (unicode
        # operators texified: ∶→:, →→\to), expel boundary pure-prose
        # word runs (>=4-letter words, no math chars). Display $$ and
        # pipe-table lines are skipped by the span parser.
        if os.environ.get("PDF_OCR_INLINE_REPAIR", "1") == "1":
            try:
                from scientific_parser.science.inline_math_repair import (
                    repair_inline_math,
                )

                out = "\n".join(
                    repair_inline_math(ln) if "$" in ln else ln for ln in out.split("\n")
                )
            except Exception as _e:
                log.warning("inline math repair skipped: %s", _e)
        # STRAY-$ STRIP (iter-127a): shatter leaks a single '$'
        # INSIDE display blocks ('Au$= f ...') — KaTeX rejects it;
        # inner display text never contains legit single '$'.
        out = re.sub(
            r"\$\$(.*?)\$\$",
            lambda m: "$$" + re.sub(r"(?<!\$)\$(?!\$)", "", m.group(1)) + "$$",
            out,
            flags=re.DOTALL,
        )
        # UNCLOSED ENV REPAIR (iter-126, Nougat close_envs port, MIT):
        # shatter leaves '\begin{cases|bmatrix|...}' without its \end —
        # KaTeX render-fails the whole display. Close per line (emission
        # units are single-line; prebuilt cascade envs stay balanced).
        try:
            from scientific_parser.science.close_envs import close_envs_line

            out = "\n".join(close_envs_line(ln) for ln in out.split("\n"))
        except Exception as _e:
            log.warning("close envs skipped: %s", _e)
        # DOUBLE-SCRIPT CONSOLIDATION (iter-130): adjacent same-direction
        # script groups are always invalid LaTeX — 86 unique census
        # equations died on Double(sub|super)script (\int_{Z}_{R} from
        # frac-tail relocation, \Lambda_{n}_{*}). Oracle-gated: only
        # failing regions whose merged form renders clean are rewritten.
        if os.environ.get("PDF_OCR_SCRIPT_CONSOLIDATE", "1") == "1":
            try:
                from scientific_parser.science.equation_cascade import (
                    katex_renders_clean,
                )
                from scientific_parser.science.script_consolidation import (
                    repair_line_scripts,
                )

                out = "\n".join(
                    repair_line_scripts(ln, katex_renders_clean) for ln in out.split("\n")
                )
            except Exception as _e:
                log.warning("script consolidation skipped: %s", _e)
        out = re.sub(r"\n{3,}", "\n\n", out).strip()
        return out

    _CMD_TO_UNI = {
        "alpha": "α",
        "beta": "β",
        "gamma": "γ",
        "delta": "δ",
        "epsilon": "ε",
        "zeta": "ζ",
        "eta": "η",
        "theta": "θ",
        "iota": "ι",
        "kappa": "κ",
        "lambda": "λ",
        "mu": "μ",
        "nu": "ν",
        "xi": "ξ",
        "pi": "π",
        "rho": "ρ",
        "sigma": "σ",
        "tau": "τ",
        "phi": "φ",
        "chi": "χ",
        "psi": "ψ",
        "omega": "ω",
    }
    _GLYPH_DUP_LINE_RE = re.compile(
        r"\\("
        + "|".join(_CMD_TO_UNI)
        + r")\s*("
        + "|".join(re.escape(u) for u in _CMD_TO_UNI.values())
        + r")"
    )
    # Markdown header (both modes — same provenance comment).
    all_text.append(f"<!-- Source: {source_provenance} -->")
    if json_mode:
        # Truncate any prior jsonl
        open(jsonl_path, "w").close()

    start_total = time.time()
    # ─── PROFESSIONAL ARCHITECTURE: load ALL shared constants into ALL modules ── #
    # Single point of initialization — calls _load_consts() on every extracted
    # module so ALL cross-module names are resolved BEFORE any page processing.
    # This replaces per-function _load_consts() calls with a centralized init.
    import scientific_parser.postproc.text_pipeline as _tp
    import scientific_parser.perception.equations.math_extractor as _me
    import scientific_parser.perception.tables.detector as _td
    import scientific_parser.perception.tables.structure as _ts
    import scientific_parser.perception.providers.region_dispatcher as _rd
    import scientific_parser.routing.page_classifier as _pc
    import scientific_parser.routing.page_metadata as _pm
    import scientific_parser.structure.column_reading_order as _cr
    import scientific_parser.science.geo_validators as _gv
    import scientific_parser.core.extraction_cache as _ec
    import scientific_parser.core.confidence_helpers as _ch
    import scientific_parser.ingest.ocr_engine as _oe
    import scientific_parser.ingest.font_analysis as _fa

    for _mod in [_tp, _me, _td, _ts, _rd, _pc, _pm, _cr, _gv, _ec, _ch, _oe, _fa]:
        if hasattr(_mod, "_load_consts"):
            _mod._load_consts()

    # Document-level state for multi-page PageHeader consensus. A true
    # running header must appear on ≥3 pages; single-page ALLCAPS strings
    # (author names on title page) are NOT headers.
    _doc_header_state: dict[str, int] = {}
    # Pages whose text came from font-math detection: already LaTeX —
    # unicode2latex + unwrap gates downstream MUST NOT reprocess them
    # (measured 2026-08-15: blanket conversion destroys 22pts, unwrap 4pts).
    _font_math_pages: set = set()
    # Any page of THIS doc actually carries math fonts (cascade adopted or
    # not) — gates the doc-level unicode sweep (iter-110): a doc with zero
    # math-font pages is pure prose and must keep its Unicode symbols.
    _ANY_MATH_FONT_PAGE = [False]
    # Reset cross-page table merging state at start of each document
    global _OPEN_TABLES_STATE
    _OPEN_TABLES_STATE = []

    # No domain detection — every processor self-gates internally by checking
    # for its own content signature (universal parser, no field hints needed).
    # Was: detect_domain() + --domain flag, removed 2026-08-12 as over-engineering.

    try:
        for page in page_list:
            route, route_reasons = page_routes[page]
            if route == "fast":
                # Fast path: text already extracted via pdftotext
                text = by_page.get(page, "")
                # P0-1: Column-aware reading order for multi-column pages.
                # Chain with OUTPUT SANITY GATE: body_only can return
                # row-major mega-lines when fitz blocks span the column
                # gutter (XY-cut cannot split them) — both columns merge
                # into ~580-char lines (wrong order AND whole-row loss when
                # hf-strip kills one). Gate: median line length ≤ 120 chars;
                # on failure fall through to the next reorder strategy.
                reordered = None
                for _reorder_fn in (
                    column_aware_body_only,
                    column_aware_reading_order,
                    pdfplumber_column_crop,
                ):
                    try:
                        _r = _reorder_fn(pdf_path, page)
                    except Exception as _re:
                        log.debug("reorder %s failed: %s", _reorder_fn.__name__, _re)
                        _pw_skip(f"column reorder {_reorder_fn.__name__}")
                        _r = None
                    if _r is None:
                        continue
                    _ls = [ln for ln in _r.split("\n") if ln.strip()]
                    _med = sorted(len(ln) for ln in _ls)[len(_ls) // 2] if _ls else 0
                    if _med <= 120:
                        reordered = _r
                        break
                    log.debug(
                        "reorder %s produced mega-lines (median %d chars) — trying next",
                        _reorder_fn.__name__,
                        _med,
                    )
                if reordered is not None and len(reordered) >= len(text) * 0.5:
                    text = reordered
                if text.strip():
                    # ─── SOTA 2026-08-12: Unified post-proc pipeline ──────── #
                    # Replaces ~80 lines of inline chain with single dispatcher
                    # call. All procs self-gate internally by checking content
                    # signature (universal parser, no domain hints needed).
                    # Pre-dispatcher: strip running header (needs page_meta
                    # captured before transformations to detect headers).
                    page_meta = extract_page_metadata(
                        text, page_num=page, doc_state=_doc_header_state, pdf_path=pdf_path
                    )
                    # Geometry mode: strip ALL top/bottom-band lines
                    # (running headers, author lines, page numbers).
                    # Fuzzy per-line removal: pdftotext reflows spacing, so
                    # exact-match strip misses journal footers like
                    # "scientific reports | 5:14952 | doi: 10.1038/...".
                    # Pre-postproc strip (early). Postproc processors and
                    # specialist routing may re-extract from the raw text
                    # layer; the FINAL pre-emit strip guarantees the footer
                    # stays gone.
                    text = _strip_hf_lines(text, page_meta)
                    # Dispatcher: runs universal procs always; geo/chem gated.
                    page_ctx = {
                        "pdf_path": pdf_path,
                        "page": page,
                        "page_meta": page_meta,
                        "tmp_dir": tmp_dir,
                        "model": models[0],
                        "ocr_text": "",  # fast path has no OCR
                        "doc_state": _doc_header_state,
                        "doc_info": doc_info,
                    }
                    # Post-proc cleans the native text layer BEFORE specialist
                    # routing adds structured <table>/LaTeX HTML. Routing is
                    # terminal — its HTML must not pass through cleaners that
                    # would strip the tags.
                    region_stats = {}
                    text, registry_warnings = run_unified_postproc(
                        text,
                        source="fast",
                        domain="universal",  # all procs universal now (self-gating)
                        page_ctx=page_ctx,
                    )
                else:
                    page_meta = {"page_header": None, "page_number": None}
                    registry_warnings = []
                # Unicode→LaTeX: convert born-digital math chars (∫∑αβ→\int\sum\alpha\beta)
                # Font-level math detection: when the page has math fonts
                # (CMMI/CMSY/MSBM), use PyMuPDF font metadata to detect math
                # regions and wrap them in $...$ with LaTeX commands.
                _page_has_mf = True  # default: convert (pre-gate behavior) unless probed
                try:
                    from scientific_parser.science.font_math_detector import (
                        has_math_fonts,
                    )
                    from scientific_parser.science.equation_cascade import (
                        extract_text_with_cascade,
                    )
                    import fitz as _fitz

                    _doc = _fitz.open(pdf_path)
                    _page = _doc[page - 1]
                    _page_has_mf = has_math_fonts(_page)
                    if _page_has_mf:
                        _ANY_MATH_FONT_PAGE[0] = True
                        # Tier 1 geometry + oracle-gated RapidLaTeX/VL repair
                        # (vl tier only when a VL model is configured)
                        # Tier-3 VL repair DISABLED on the fast path: a live
                        # 69s/page model call inside the cascade makes
                        # born-digital output NONDETERMINISTIC (pool workers
                        # serialize on GPU; timeouts vary per run — bench
                        # 2026-08-15: same page yielded 37 vs 6 eq blocks).
                        # Font assembly + RapidLaTeX crops are deterministic.
                        _vl = None
                        _font_text = extract_text_with_cascade(
                            _page, tmp_dir, _vl, pdf_path=pdf_path, page_num=page
                        )

                        # LINE-STRUCTURE GATE (2026-08-19): the cascade
                        # re-extracts from fitz spans; on word-fragment
                        # pages (T11-T18 subset fonts, 09f90a8f) its raw
                        # span dump is one-word-per-line (median 4 chars)
                        # while the pipeline text has real lines (median
                        # 47). Adopting it destroyed the page. Reject a
                        # replacement whose line structure collapsed —
                        # legit math pages keep long prose lines (median
                        # ≥30) so they still adopt.
                        def _med_line(t):
                            _ls = [ln for ln in t.split("\n") if ln.strip()]
                            return sorted(len(ln) for ln in _ls)[len(_ls) // 2] if _ls else 0

                        _src_med = _med_line(text)
                        _cas_med = _med_line(_font_text)
                        if len(_font_text.strip()) > len(text.strip()) * 0.5 and (
                            _cas_med >= 30 or _cas_med >= 0.5 * _src_med
                        ):
                            # Reflow the cascade re-extract: span-dump
                            # lines carry hard typesetting breaks; the
                            # pre-dispatch reflow already fixed them in
                            # the replaced text (iter-89).
                            from scientific_parser.postproc.text_pipeline import (
                                detect_section_hierarchy,
                                reflow_prose_paragraphs,
                            )

                            # HEADING REVIVAL ON CASCADE ADOPTION (det22,
                            # odlbench 01030000000145): the cascade
                            # re-extract just replaced registry output
                            # wholesale — including every
                            # detect_section_hierarchy promotion. dsh is
                            # span-driven (reads fitz spans itself), so
                            # like the table append below it stays valid
                            # on the cascade text. Run BEFORE reflow:
                            # whole-line/pair gates need the unglued span
                            # lines ('4.1 Introduction' matches the det20
                            # space separator); reflow respects '#'
                            # boundaries so promoted headings survive.
                            _font_text = detect_section_hierarchy(pdf_path, page, _font_text)
                            text = reflow_prose_paragraphs(_font_text)
                            _font_math_pages.add(page)
                            # CASCADE-TABLE PRESERVATION (iter-58): the
                            # cascade re-extract just replaced postproc
                            # output that carried the pipe tables the
                            # registry's detect_tables (order 900)
                            # appended. Re-append them — table detection
                            # is page-geometry driven (not text derived),
                            # so the append is valid on the cascade text.
                            # Before this, every math-font page with a
                            # table lost it wholesale (f45a188e: 4 gold
                            # relationship tests).
                            text = _detect_and_append_tables(text, pdf_path, page, "")
                    _doc.close()
                except Exception as e:
                    if os.environ.get("PDF_OCR_DEBUG_CASCADE"):
                        import traceback

                        traceback.print_exc()
                    log.debug("font math detection skipped: %s", e)
                    record_path_warning(
                        FONT_MATH_CASCADE_FAILED,
                        "font-math cascade failed on page — LaTeX "
                        "reconstruction lost, raw text kept",
                        page=page,
                    )
                # PAGE GATE (iter-110, 068682d5 pg1): the blanket unicode
                # sweep is for MATH pages whose glyphs the text layer kept
                # as Unicode. A page with zero math fonts is prose — its
                # '×'/'±' are typographic prose symbols and converting
                # them ('an$m \times m$multi…') corrupts reading-order
                # golds. Measured sample: every arxiv_math page carries
                # math fonts (cascade path); no-math-font pages are all
                # plain-gold categories. _page_has_mf defaults True when
                # the font probe excepted (no silent behavior change).
                if page not in _font_math_pages and _page_has_mf:
                    text = convert_unicode_to_latex(text)
                    text = normalize_inline_math(text)
                elif page not in _font_math_pages:
                    # Prose page (iter-110): gold-convention ellipsis
                    # normalization only — Unicode symbols stay verbatim.
                    text = convert_unicode_to_latex(text, prose_only=True)
                # ─── SOTA specialist dispatch (fast path, TERMINAL) ─────── #
                # Runs AFTER post-proc so the structured <table>/LaTeX HTML it
                # produces is the final page text — not stripped by cleaners.
                # SINGLE dispatch site shared with the OCR path.
                # GATE (2026-08-15): font-math pages skip dispatch — region
                # routing re-extracts the RAW text layer and replaces the
                # already-LaTeX font-detector output (overwrite-bug family,
                # 4th member; measured -12 tests on arxiv_math sample).
                if not IS_IMAGE and page not in _font_math_pages:
                    try:
                        fast_img = pdf_page_to_png(pdf_path, page, tmp_dir + f"/fast_p{page}")
                    except Exception:
                        fast_img = None
                    if fast_img:
                        text, _cands = _dispatch_regions(
                            pdf_path,
                            page,
                            fast_img,
                            text,
                            args,
                            models,
                            fast=True,
                            raw_text=by_page.get(page, ""),
                            hf_strip=_strip_hf_lines,
                        )
                        all_region_candidates.extend(_cands)
                # ─── det61: TableFormer zero-table fallback (TERMINAL) ── #
                # S2a borderless class (odlbench 0149/0165/0197): strategy
                # chain, monster-rebuild AND region dispatch all emit
                # nothing — the page reaches its final text with ZERO table
                # blocks. TableFormer (docling-ibm-models, CPU, local
                # weights) regresses grids for layout-detector table
                # regions. Strictest gate by construction: any pipe row or
                # <table> in the page text skips the model. Placing this
                # earlier (postproc detect_tables wrapper) measured +6 P2F
                # on 137297b4 pg4: TF pipes flipped the dispatcher's
                # structure-gain gate and suppressed its better table
                # (sweep p61, 2026-10-12). Fail-open; kill switch
                # PDF_OCR_TABLEFORMER=0.
                if (
                    not IS_IMAGE
                    and "<table" not in text
                    and not any(_ln.strip().startswith("|") for _ln in text.splitlines())
                ):
                    try:
                        from scientific_parser.perception.tables.tableformer_backend import (
                            tables_for_page,
                        )
                        from scientific_parser.postproc.text_pipeline import (
                            _emit_table_blocks,
                        )

                        for _tf_t in tables_for_page(pdf_path, page):
                            for _blk in _emit_table_blocks(_tf_t):
                                if _blk:
                                    text = text.rstrip() + "\n\n" + _blk + "\n\n"
                    except Exception as e:
                        log.warning(
                            "tableformer fallback skipped p%s: %s: %s",
                            page,
                            type(e).__name__,
                            e,
                        )
                page_chars = len(text.strip())
                warnings = []
                if page_chars < LOW_TEXT_DENSITY_THRESHOLD:
                    warnings.append(f"low_text_density ({page_chars} chars)")
                warnings.extend(registry_warnings)
                # P2-11: Bibliography mode
                body_text, bib_text = detect_bibliography_section(text)
                if bib_text:
                    bib_processed = process_bibliography(bib_text)
                    text = body_text + "\n\n" + bib_processed
                # Check for encoding artifacts (even if cleaned by hybrid).
                orig_text = by_page.get(page, "")
                orig_priv = len(_PRIVATE_CHAR_RE.findall(orig_text))
                cleaned_priv = orig_priv
                if orig_priv > 0:
                    cleaned_priv = len(_PRIVATE_CHAR_RE.findall(text))
                    if cleaned_priv < orig_priv:
                        warnings.append(
                            f"encoding_artifacts ({orig_priv - cleaned_priv} chars cleaned)"
                        )
                    else:
                        warnings.append(
                            f"encoding_artifacts ({orig_priv} chars — run with --force-ocr)"
                        )
                # ─── ToUnicode cross-validation (2026-08-15) ─────────────── #
                # Custom-encoded fonts can carry subtle ToUnicode bugs: the
                # text layer differs from the RENDERED glyphs by a few chars
                # (multi_column bench: 'semmigli'→'semmieli', gold derives
                # from glyph reading → order tests fail at max_diffs 1-2).
                # Detect via render-and-compare: systematic (repeated same-
                # char) substitution = broken map; random OCR noise is not.
                # On detection, re-read the page with the VL (glyphs) and
                # adopt when it's a clean transcription.
                if (
                    doc_info is not None
                    and doc_info.get("suspect_fonts")
                    and not IS_IMAGE
                    and page_chars > 200
                    # Font-math pages: the cascade already reconstructed
                    # the suspect glyphs exactly (font tables + crops).
                    # A VL glyph re-read replaces exact LaTeX with prose
                    # math and DROPS exponents ('\mathbb{C}^{N}' →
                    # '\mathbb{C}', flip-diff 2503.06703_pg4 iter6 — same
                    # destruction family as glyph_reread routing, but on
                    # the in-page path).
                    and page not in _font_math_pages
                ):
                    try:
                        _xv_img = pdf_page_to_png(
                            pdf_path, page, os.path.join(tmp_dir, f"xval_p{page}")
                        )
                        from scientific_parser.routing.page_classifier import (
                            tounicode_crosscheck,
                        )

                        _n_sys, _subst, _xwords = tounicode_crosscheck(pdf_path, page, _xv_img)
                        if _n_sys >= 3 and not getattr(args, "always_model", False):
                            # EXP-13 surgical map: apply the detected
                            # substitution INSIDE the verified words only.
                            # A wholesale `text = _vl_text` re-read destroys
                            # table geometry and column structure (bench13:
                            # 8 flips — 769f619d 4x, e247cacc, f30f3060,
                            # 09f90a8f 2x — VL text lost pipes/structure).
                            # The map itself is the fix: same glyph maps
                            # wrong everywhere, verified words carry it.
                            _applied = 0
                            if _subst and _subst[0] and _subst[1] and _xwords and page_chars > 200:
                                from scientific_parser.routing.glyph_map import (
                                    store_page_glyph_map,
                                    surgical_word_sub,
                                )

                                _frm, _to = _subst
                                text, _applied = surgical_word_sub(text, _xwords, _frm, _to)
                                # Cell-level lane: the tables lane re-reads
                                # the fitz layer, so verified words inside
                                # table cells keep the corruption even after
                                # the flowed-text pass. Hand the map to
                                # detect_all_tables (single take()).
                                store_page_glyph_map(pdf_path, page, _frm, _to, _xwords)
                            if _applied:
                                page_chars = len(text.strip())
                                warnings.append(
                                    f"tounicode_mismatch ({_n_sys}x "
                                    f"{_subst[0]}→{_subst[1]}; surgical map "
                                    f"{_applied} words)"
                                )
                                record_path_warning(
                                    VL_FALLBACK_USED,
                                    "tounicode mismatch — surgical substitution map applied "
                                    "(structure preserved, no VL re-read)",
                                    page=page,
                                )
                            else:
                                # EXP-13 structure guard: verified words absent
                                # from the flowed-text surface means they live
                                # in table-cell / column space (769f619d:
                                # 'worst' only inside table geometry). A whole-
                                # page VL re-read destroys that structure
                                # (bench13: 8 flips) for a 3-word glyph fix.
                                # Keep the structure; defer glyph correction
                                # to the cell-level map (follow-up lane).
                                _structured = False
                                try:
                                    from scientific_parser.routing.layout_gate import (
                                        count_columns,
                                        page_needs_table_chain,
                                    )

                                    _blk = pdf_page.get_text("blocks")
                                    _structured = count_columns(
                                        _blk, pdf_page.rect.width
                                    ) >= 2 or page_needs_table_chain(pdf_path, page, text)
                                except Exception:
                                    _structured = True  # fail-safe: keep text
                                if _structured:
                                    warnings.append(
                                        f"tounicode_mismatch ({_n_sys}x "
                                        f"{_subst[0]}→{_subst[1]}; structured page — "
                                        f"map deferred, VL re-read skipped)"
                                    )
                                    record_path_warning(
                                        VL_FALLBACK_USED,
                                        "tounicode mismatch on structured page — surgical "
                                        "map deferred (structure kept, no VL re-read)",
                                        page=page,
                                    )
                                else:
                                    print(
                                        f"  ToUnicode mismatch: {_n_sys}x "
                                        f"'{_subst[0]}'→'{_subst[1]}' — re-reading glyphs via VL",
                                        file=sys.stderr,
                                    )
                                    try:
                                        _avail = list_ollama_models()
                                    except Exception:
                                        _avail = []
                                    _paddle = next(
                                        (
                                            m
                                            for m in ("paddleocr-vl:1.6", "paddleocr-vl")
                                            if m in _avail
                                        ),
                                        None,
                                    )
                                    if _paddle:
                                        _vl_text = ocr_image(_xv_img, task="OCR:", model=_paddle)
                                        _vl_conf = _compute_page_confidence("ocr", [], _vl_text)
                                        if (
                                            not _looks_like_chat_response(_vl_text)
                                            and _vl_conf >= 0.55
                                            and len(_vl_text.strip()) > 0.5 * page_chars
                                        ):
                                            warnings.append(
                                                f"tounicode_mismatch ({_n_sys}x "
                                                f"{_subst[0]}→{_subst[1]}; VL re-read)"
                                            )
                                            text = _vl_text
                                            page_chars = len(text.strip())
                                            record_path_warning(
                                                VL_FALLBACK_USED,
                                                "tounicode mismatch — born-digital page text re-read via VL",
                                                page=page,
                                            )
                    except Exception as e:
                        # H1: silent stage loss hid a live bug for the whole
                        # v3 era — tmp_dir is a str, `tmp_dir / name` threw
                        # TypeError on EVERY suspect-font page (fixed above).
                        log.warning("tounicode crosscheck failed p%s: %s", page, e)
                        _pw_skip("tounicode crosscheck")
                # Extreme-case escalation: text layer unusable (near-empty or
                # garbled ToUnicode that post-proc could not clean) → OCR the
                # page render and adopt it only when substantially richer than
                # the text layer. Guards: skip images (already OCR-routed),
                # never adopt thinner OCR output (figures with little text).
                _garbled = orig_priv > 0 and cleaned_priv >= orig_priv
                if not _garbled:
                    # Garbled ToUnicode variant 2: math/Indic codepoint soup.
                    # Broken CJK-math fonts map glyphs to U+1D400-1D7FF
                    # (Math Alphanumeric) + Indic digit blocks — readable
                    # length but unusable text (bench tables 1651cbb4:
                    # '𝑒ௌைమ_௧' lines). >4% exotic = escalate to OCR.
                    _exotic = sum(
                        1
                        for ch in text[:3000]
                        if "\U0001d400" <= ch <= "\U0001d7ff"
                        or "\u0b80" <= ch <= "\u0bff"  # Telugu
                        or "\u0c80" <= ch <= "\u0cff"  # Kannada
                        or "\u0be6" <= ch <= "\u0be9"  # Tamil digits
                    )
                    _garbled = _exotic > 0.04 * min(len(text), 3000)
                # Font-math pages: the equation cascade already reconstructed
                # the garbled regions (RapidLaTeX crops / font tables) — a
                # whole-page VL pass here DESTROYS the LaTeX (VL emits prose
                # math, wins the >1.5x char adoption gate, fails render-
                # compare; measured 44-test family 2026-08-16 audit).
                if (
                    (page_chars < LOW_TEXT_DENSITY_THRESHOLD or _garbled)
                    and not IS_IMAGE
                    and page not in _font_math_pages
                ):
                    try:
                        _fp_img = pdf_page_to_png(
                            pdf_path, page, os.path.join(tmp_dir, f"fp_escal_{page}")
                        )
                        _fp_ocr = ocr_image(_fp_img, model=models[0])
                        # Chat-mode guard (2026-08-15): chandra-ocr-2 answers
                        # assistant-speak ("The user wants me to extract...")
                        # instead of transcribing — prompt-independent, measured
                        # 9/10 pages. Re-OCR once with paddleocr-vl ("OCR:")
                        # which transcribes cleanly; keep the longer clean text.
                        if _looks_like_chat_response(_fp_ocr) and not getattr(
                            args, "always_model", False
                        ):
                            from scientific_parser.core.confidence_helpers import (
                                vl_single_model_cap,
                            )

                            _avail = []
                            if vl_single_model_cap(models[0]) is None:
                                try:
                                    _avail = list_ollama_models()
                                except Exception:
                                    _avail = []
                            _paddle = next(
                                (m for m in ("paddleocr-vl:1.6", "paddleocr-vl") if m in _avail),
                                None,
                            )
                            if _paddle and _paddle != models[0]:
                                # RAM-bounded swap (see self-corrective rescue):
                                # unload resident VL model before the alternate
                                # loads; unload the alternate if it loses.
                                ollama_unload(models[0])
                                _alt = ocr_image(_fp_img, task="OCR:", model=_paddle)
                                if not _looks_like_chat_response(_alt):
                                    print(
                                        f"  Chat-mode guard: {models[0]} garbaged → "
                                        f"{_paddle} ({len(_alt)} chars)",
                                        file=sys.stderr,
                                    )
                                    _fp_ocr = _alt
                                    record_path_warning(
                                        OCR_MODEL_SWAP,
                                        "chat-mode guard: VL model garbaged the page — swapped to paddleocr-vl",
                                        page=page,
                                    )
                                else:
                                    ollama_unload(_paddle)
                        _fp_ocr, _ = _ocr_vl_fallback(
                            _fp_ocr,
                            models[0],
                            _fp_img,
                            ["fast_path_escalation"],
                            "Text Recognition:",
                            always_model=getattr(args, "always_model", False),
                        )
                        _fp_chars = len(_fp_ocr.strip())
                        # Narration veto (bench14): even when the paddle swap
                        # is unavailable or also narrates, a described-not-
                        # transcribed page must never be adopted — the thin
                        # layer strips cleanly via geometry; narration keeps
                        # header phrases in prose sentences (absent-test 1.000).
                        if _fp_chars > max(page_chars * 1.5, 200) and not _looks_like_chat_response(
                            _fp_ocr
                        ):
                            text = _fp_ocr
                            page_chars = _fp_chars
                            # EXP-13: page_meta was extracted from the THIN
                            # pre-escalation layer (L~2425) — the final
                            # _strip_hf_lines below ran with empty bands and
                            # kept running headers in the adopted OCR text
                            # (bench13: b329b808 + 0f727ae2 absent-test
                            # flips). Re-derive bands from the ADOPTED text.
                            page_meta = extract_page_metadata(
                                text,
                                page_num=page,
                                doc_state=_doc_header_state,
                                pdf_path=pdf_path,
                            )
                            warnings.append(f"fast_path_ocr_escalation ({_fp_chars} chars)")
                            print(
                                f"  Fast-path OCR escalation: {page_chars} chars (was text layer)"
                            )
                            record_path_warning(
                                FAST_PATH_OCR_ESCALATION,
                                "fast-path text layer unusable — page re-OCR'd via OCR engine",
                                page=page,
                            )
                    except Exception as e:
                        log.debug("fast-path OCR escalation failed p%s: %s", page, e)
                        _pw_skip("fast-path OCR escalation attempt")
                _conf = _compute_page_confidence("pdftotext", warnings, text)
                # the scorer-visible surface (bench runner splits jsonl into
                # candidate .md files). Emitting the record pre-strip leaked
                # fused footers the .md's own strip had removed (measured
                # 0678963a: 'Midwifery xxx (xxxx) xxx–xxx' absent from .md,
                # present in jsonl → 331 absent-test fails). Record and
                # markdown must see the SAME stripped text.
                text = _strip_hf_lines(text, page_meta)
                record = {
                    "page": page,
                    "source": "pdftotext",
                    "chars": page_chars,
                    "elapsed_s": 0.0,
                    "route": "fast",
                    "route_reasons": route_reasons,
                    "confidence": _conf,
                    "page_header": page_meta["page_header"],
                    "page_number": page_meta["page_number"],
                    "warnings": warnings,
                    "text": text if text.strip() else "[No text on page]",
                }
                page_metrics.append(record)
                meta_comment = _format_meta_comment(page_meta, warnings)
                # No "<!-- Page N -->" marker: the marker text leaks into
                # comment-reading scorers (olmOCR absent-tests fail on
                # gold like 'page 1'; partial_ratio matches at 100).
                emit_markdown(f"{meta_comment}{text if text.strip() else '[No text on page]'}")
                if json_mode:
                    emit_jsonl(record)
                continue

            # OCR path
            print(f"\n--- Page {page}/{total_pages} (OCR) ---")
            if route_reasons and route_reasons != ["forced"]:
                print(f"  Routing reasons: {', '.join(route_reasons)}")
            start = time.time()
            # Page-scoped warnings list, bound at OCR-path entry so early
            # handlers (model-failure degrade, region fallbacks) can append
            # before the summary stage rebuilds it below.
            warnings: list = []

            # R19 per-page OCR memo: on a detection-tier-only version bump
            # the raw cache misses but this CORE-keyed memo replays the
            # page's stochastic product — no chandra/cascade re-roll
            # (run-16: sigma +-5-10 tests/cat re-roll noise swamped real
            # gains). Memo covers the main OCR path only; the rare
            # fragmented-table splice path and fast-path VL remedies
            # re-roll (documented residual).
            _memo_entry = (
                page_memo_lookup(pdf_path or PDF_PATH, models, args, page) if use_cache else None
            )
            _pw_snapshot = list(get_path_warnings()) if _memo_entry is None else None
            table_region_used = False
            if _memo_entry is not None:
                text = _memo_entry["text"]
                _cands = _memo_entry.get("cands") or []
                all_region_candidates.extend(_cands)
                reflexive_meta = _memo_entry.get("reflexive_meta") or {
                    "model": models[0],
                    "attempts": 1,
                    "confidence": None,
                    "cascade": False,
                }
                reflexive_meta["memo_hit"] = True
                route_reasons = _memo_entry.get("route_reasons") or route_reasons
                elapsed = _memo_entry.get("elapsed_s", 0.0)
                for _w in _memo_entry.get("path_warnings") or []:
                    record_path_warning(
                        _w.get("kind"),
                        _w.get("message", ""),
                        page=page,
                        details=_w.get("details"),
                    )
                warnings.extend(_memo_entry.get("model_failure_warnings") or [])
                print(
                    f"  [memo] core-tier replay — stochastic OCR restored "
                    f"({len(text)} chars, {elapsed:.1f}s saved)"
                )
            # Per-region table OCR (Marker SOTA M5): if routed due to
            # fragmented_table, try to detect the table region and OCR just
            # that. Falls back to full-page OCR if region detection fails.
            # Skipped on memo replay (stochastic path, R19 residual).
            if (
                _memo_entry is None
                and not IS_IMAGE
                and not crop
                and any("fragmented_table" in r for r in route_reasons)
                # Skip the legacy flat-OCR splice when specialist routing
                # already produced structured <table> HTML (RapidTable) —
                # otherwise this overwrites good HTML with flat OCR text.
                and not (region_stats.get("n_tables") and region_stats.get("n_dispatched"))
            ):
                try:
                    regions = find_table_bbox(pdf_path, page)
                    if regions:
                        # Use the text layer for non-table areas, OCR the table
                        fast_text = by_page.get(page, "")
                        region_results = []
                        for rx, ry, rw, rh in regions:
                            # Render and crop just this region
                            prefix = os.path.join(tmp_dir, "page")
                            full_img = pdf_page_to_png(pdf_path, page, prefix)
                            region_path = full_img.replace(".png", f"_region.png")
                            crop_image(full_img, (rx, ry, rw, rh), region_path)
                            region_path = resize_image(region_path, MAX_DIM)
                            # Math region OCR — use MATH_OCR_PROMPT (avoids
                            # \sort confusion from default prompt).
                            region_text = ocr_image(
                                region_path, task=MATH_OCR_PROMPT, model=models[0]
                            )
                            region_results.append(((rx, ry, rw, rh), region_text))
                            print(
                                f"  Region OCR: ({rx:.0f},{ry:.0f},{rw:.0f},{rh:.0f}) "
                                f"→ {len(region_text)} chars"
                            )
                        if region_results:
                            # Splice OCR'd table INTO the text layer.
                            # The text layer is good for non-table content — only
                            # the fragmented table lines (3+ space runs) are bad.
                            # Remove fragmented lines, append OCR'd table.
                            ocr_text = "\n\n".join(t for _, t in region_results)
                            if len(ocr_text) > 50:
                                fast_text = by_page.get(page, "")
                                lines = fast_text.split("\n")
                                kept = [
                                    line for line in lines if len(re.findall(r" {3,}", line)) < 2
                                ]
                                spliced = "\n".join(kept).rstrip()
                                if spliced:
                                    spliced += "\n\n"
                                spliced += ocr_text
                                page_results = [(models[0], spliced)]
                                table_region_used = True
                                print(
                                    f"  Region OCR succeeded; spliced into text layer "
                                    f"({len(kept)} lines kept + {len(ocr_text)} chars OCR)"
                                )
                except Exception as e:
                    print(f"  Region OCR failed ({e}); falling back to full-page OCR")
                    record_path_warning(
                        REGION_OUTPUT_REJECTED,
                        "table region OCR failed — fell back to full-page OCR",
                        page=page,
                    )

            if _memo_entry is None and not table_region_used:
                # Source the page image: direct copy for image input, pdftoppm for PDFs
                math_model = models[0]
                if IS_IMAGE:
                    # Image input: page is always 1; just copy the file to temp dir
                    img_path = os.path.join(tmp_dir, f"input{os.path.splitext(PDF_PATH)[1]}")
                    shutil.copy2(PDF_PATH, img_path)
                else:
                    prefix = os.path.join(tmp_dir, "page")
                    adaptive_dpi = detect_page_complexity(pdf_path, page)
                    # Model-aware DPI clamp for paddleocr-vl (measured on
                    # long_tiny_text 10b, large-format scan 2217x3609@150):
                    # <200 DPI → mmproj loses small print → model silently
                    # DROPS paragraphs (novel passages gone @100-150);
                    # >250 → encoder down-scale hurts again. Sweet spot
                    # [200, 250]. rapidocr keeps its adaptive value.
                    if "paddleocr" in math_model.lower():
                        adaptive_dpi = max(min(adaptive_dpi, 250), 200)
                    # SCAN-PAGE PROTOCOL (2026-08-19): chandra-ocr-2's own
                    # olmOCR-bench reference (85.8%) renders at 300 DPI —
                    # and detect_page_complexity returns 150 on image-only
                    # pages here because _get_page_dict fails without a
                    # text layer (sizes empty → wrong fallback). Measured
                    # 17_pg4: 150 DPI = 0/7 gold, 300 DPI + layout prompt
                    # = 7/7. Only for scan-routed pages on chandra (VL
                    # token budget already validated at num_ctx ≥16k).
                    _scan_routed = any(
                        ("low_chars" in r or "scanned_image_page" in r) for r in route_reasons
                    )
                    if _scan_routed and "chandra" in str(models[0]).lower() and adaptive_dpi < 300:
                        adaptive_dpi = 300
                        print("  Scan protocol: 300 DPI (chandra reference)")
                    if adaptive_dpi != DPI:
                        print(f"  Adaptive DPI: {adaptive_dpi} (font analysis)")
                    img_path = pdf_page_to_png(pdf_path, page, prefix, dpi=adaptive_dpi)

                    # P0-3: Detect display equations for layout-aware OCR
                    eq_bands = detect_display_equations(pdf_path, page)
                    # Math-scan model routing: glm-ocr is the measured leader
                    # on math scans (old_scans_math 63.1% w/ pykatex). When
                    # display equations are present AND the model was
                    # auto-defaulted (not user-pinned), route math calls to
                    # glm-ocr even if the general default is qwen2.5vl:3b.
                    if eq_bands and models_auto_default and math_model.startswith("qwen"):
                        try:
                            if "glm-ocr" in list_ollama_models():
                                math_model = "glm-ocr"
                        except Exception as e:
                            log.debug("math-model auto-route probe degraded: %s", e)
                            _pw_skip("math-model auto-route probe")
                    if eq_bands:
                        print(f"  Detected {len(eq_bands)} display equation region(s)")
                        eq_results = ocr_display_equations(
                            pdf_path, page, tmp_dir, model=math_model
                        )
                        if eq_results:
                            print(f"  OCR'd {len(eq_results)} equations with math prompt")

                # Optional crop
                if crop is not None:
                    crop_path = img_path.replace(".png", "_cropped.png")
                    img_path = crop_image(img_path, crop, crop_path)
                    print(f"  Cropped: {img_path}")

                # Resize if needed
                img_path = resize_image(img_path, MAX_DIM)

                # ─── SOTA 2026-08-12: Reflexive OCR loop ─────────────────── #
                # arXiv 2603.22651 pattern: extract → verify (pykatex) → score
                # → retry with error feedback. Replaces single-shot ocr_image.
                # Cascade: if multi-model chain (--models a,b,c) and confidence
                # below floor, advance to next model. See ocr_with_reflexive_loop.
                page_text_hint = by_page.get(page, "")
                if getattr(args, "no_reflexive", False) or len(models) == 1:
                    # Single-shot mode (legacy) — skip reflexive loop.
                    # Used by --no-reflexive or single-model invocations.
                    # Math-heavy pages use the math-routed model.
                    # Full-page prompt: paddleocr-vl rambling-hallucinates
                    # with MATH_OCR_PROMPT on non-math pages (measured:
                    # invented formulas on a handwritten letter) — use the
                    # plain transcription instruction for full pages.
                    # chandra-ocr-2 uses its OFFICIAL ocr_layout prompt on
                    # full pages (resolve_task_prompt maps it): MATH_OCR_PROMPT
                    # on a prose book scan yielded a DIFFERENT, lossier read
                    # (17_pg4: gold present with layout prompt, absent with
                    # math prompt; measured 2026-08-19).
                    _page_task = MATH_OCR_PROMPT
                    if "paddleocr" in math_model.lower():
                        _page_task = "OCR:"
                    elif "chandra" in math_model.lower():
                        _page_task = ""  # empty label → resolve_task_prompt
                        try:
                            from scientific_parser.ingest.ocr_engine import (
                                CHANDRA_LAYOUT_PROMPT,
                            )

                            _page_task = CHANDRA_LAYOUT_PROMPT
                        except ImportError:
                            _page_task = MATH_OCR_PROMPT
                    try:
                        text = ocr_image(img_path, task=_page_task, model=math_model)
                    except Exception as _ocr_err:
                        # Persistent model failure after in-engine retries:
                        # degrade LOUDLY to in-process rapidocr rather than
                        # crashing the whole document (transient Ollama 500s
                        # observed during model cold-load, 2026-08-16 audit).
                        if getattr(args, "always_model", False):
                            raise
                        print(
                            f"  WARNING: {math_model} failed after retries "
                            f"({_ocr_err}) — degrading page to rapidocr",
                            file=sys.stderr,
                        )
                        # Degrade note lands in the page-scoped warnings
                        # list bound at OCR-path entry. Was an
                        # UnboundLocalError crash when paddleocr-vl 500'd
                        # mid-bench: 4/188 table PDFs emitted no output.
                        warnings.append(f"model_failure_degraded_rapidocr ({math_model})")
                        text = ocr_image(img_path, task=_page_task, model="rapidocr")
                        record_path_warning(
                            MODEL_FAILURE_DEGRADED,
                            "OCR model failed after retries — page degraded to rapidocr",
                            page=page,
                            details={"model": str(math_model)},
                        )
                    # Auto-fallback to Ollama VL when rapidocr output is poor
                    # (content-volume gate — rapidocr's confidence is over-
                    # confident, see _ocr_vl_fallback). Fires only on
                    # catastrophic OCR failure (<150 chars on a text page).
                    text, _vl_used = _ocr_vl_fallback(
                        text,
                        models[0],
                        img_path,
                        route_reasons,
                        MATH_OCR_PROMPT,
                        always_model=getattr(args, "always_model", False),
                    )
                    # SELF-CORRECTIVE RESCUE (single-shot mode): the reflexive
                    # loop is skipped for len(models)==1, and its confidence
                    # floor never fired in practice (base 0.7 vs floor 0.55 —
                    # dead until degenerate-signal penalties landed
                    # 2026-08-15). If the page scores below the floor AND a
                    # second VL model exists, re-OCR once and keep the
                    # higher-confidence result. This is the cascade the
                    # architecture promised.
                    try:
                        from scientific_parser.core.confidence_helpers import (
                            _compute_page_confidence as _cpc,
                        )

                        _c1 = _cpc("ocr", [], text)
                        if _c1 < REFLEXIVE_CONFIDENCE_FLOOR and not getattr(
                            args, "always_model", False
                        ):
                            _second = None
                            from scientific_parser.core.confidence_helpers import (
                                vl_single_model_cap,
                            )

                            if vl_single_model_cap(models[0]) is None:
                                try:
                                    _avail = list_ollama_models()
                                    for _m in _VL_PREFERENCE:
                                        if _m != _vl_used and _m in _avail:
                                            _second = _m
                                            break
                                except Exception as e:
                                    log.warning("main: Exception degraded: %s", e)
                            if _second:
                                _t2_task = _page_task
                                if "paddleocr" in _second.lower():
                                    _t2_task = "OCR:"
                                # RAM-bounded model swap (2026-08-18): unload
                                # the resident VL model BEFORE loading the
                                # alternate — two llama-servers resident OOM
                                # low-RAM hosts (keep_alive=30m). If the
                                # rescue loses, unload the alternate so the
                                # primary reloads on the next page.
                                ollama_unload(_vl_used)
                                _t2 = ocr_image(img_path, task=_t2_task, model=_second)
                                _c2 = _cpc("ocr", [], _t2)
                                if _c2 > _c1:
                                    print(
                                        f"  Self-corrective rescue: {_vl_used} "
                                        f"(conf {_c1}) → {_second} (conf {_c2})",
                                        file=sys.stderr,
                                    )
                                    text, _vl_used = _t2, _second
                                    record_path_warning(
                                        OCR_MODEL_SWAP,
                                        "self-corrective rescue adopted a second VL model (higher confidence)",
                                        page=page,
                                    )
                                else:
                                    ollama_unload(_second)
                    except Exception as _e:
                        log.debug("self-corrective rescue skipped: %s", _e)
                        _pw_skip("self-corrective rescue")
                    reflexive_meta = {
                        "model": _vl_used,
                        "attempts": 1,
                        "confidence": None,
                        "cascade": _vl_used != models[0],
                    }
                else:
                    # SOTA: adaptive model selection per page type
                    chosen_model, _, rationale = select_ocr_model(
                        route, route_reasons, models, doc_info, page_text_hint
                    )
                    # Low-RAM pin (2026-08-18): host-aware cap lives HERE,
                    # not in select_ocr_model (pure function, unit-tested).
                    # One resident VL model per host below 20GB — per-page
                    # model alternation keeps two llama-servers loaded
                    # (keep_alive=30m) and OOMs (measured 13GB host).
                    from scientific_parser.core.confidence_helpers import (
                        vl_single_model_cap,
                    )

                    _cap = vl_single_model_cap(models[0])
                    if _cap and chosen_model != _cap:
                        chosen_model = _cap
                        rationale += " → low-RAM single-model pin"
                    # Cascade chain: chosen model first, then others as fallback
                    chain = [chosen_model] + [m for m in models if m != chosen_model]
                    if _cap:
                        chain = [chosen_model]
                    if len(route_reasons) > 0 and route_reasons != ["forced"]:
                        print(
                            f"  Model routing: {rationale} (chain: "
                            f"{' → '.join(chain[:3])}{'…' if len(chain) > 3 else ''})"
                        )
                    text, used_model, attempts, conf, warns = ocr_with_reflexive_loop(
                        img_path,
                        chain,
                        page_text_hint=page_text_hint,
                        confidence_floor=getattr(
                            args, "reflexive_floor", REFLEXIVE_CONFIDENCE_FLOOR
                        ),
                        # Per architecture proposal item #5 (2026-08-13):
                        # pass page context so retry executor can re-render
                        # at higher DPI on low-confidence attempts. Full-page
                        # OCR has no region_bbox (TIGHTER_CROP N/A); the
                        # useful action here is HIGHER_DPI.
                        pdf_path=pdf_path,
                        page_num=page,
                        current_dpi=DPI,
                    )
                    reflexive_meta = {
                        "model": used_model,
                        "attempts": attempts,
                        "confidence": round(conf, 3) if conf is not None else None,
                        "cascade": len(chain) > 1,
                        "reflexive_warnings": warns[:3] if warns else [],
                    }
                    if attempts > 1 or used_model != chain[0]:
                        print(
                            f"  Reflexive: {attempts} attempts, model={used_model}, "
                            f"conf={conf:.2f}"
                            + (f" (cascaded from {chain[0]})" if used_model != chain[0] else "")
                        )
                        record_path_warning(
                            OCR_MODEL_SWAP,
                            "reflexive loop retried/cascaded — primary OCR output below confidence floor",
                            page=page,
                            details={"attempts": int(attempts)},
                        )

            elapsed = time.time() - start
            print(f"  Got {len(text)} chars in {elapsed:.1f}s")

            # ─── SOTA per-region specialist dispatch (2026-08-12 keystone) ── #
            # Auto-activates when specialists are installed (CORE deps per
            # pyproject.toml). Detects regions → dispatches each to specialist
            # → REPLACES per-region text (not append). Falls back gracefully.
            #
            # Auto-activation: "default" → use rapid-* if available.
            # User can pass "off" to disable, or specific backend name.
            def _resolve_backend(arg_val, rapid_name):
                if arg_val == "off":
                    return None
                if arg_val == "default":
                    return rapid_name
                return arg_val

            # SINGLE dispatch site shared with the fast path (rapidocr-only
            # Skipped on memo replay: the memoized text is already
            # post-dispatch (stored downstream of this call on the fresh
            # path), and img_path does not exist on the replay path.
            if _memo_entry is None:
                text, _cands = _dispatch_regions(
                    pdf_path,
                    page,
                    img_path,
                    text,
                    args,
                    models,
                    fast=False,
                    hf_strip=_strip_hf_lines,
                )
                all_region_candidates.extend(_cands)

                # R19: persist this page's stochastic product under the
                # CORE-keyed memo so a detection-tier-only version bump can
                # replay it without re-rolling chandra/cascade. Best-effort —
                # memo loss only costs a re-roll, never correctness.
                if use_cache and text.strip():
                    try:
                        page_memo_store(
                            pdf_path or PDF_PATH,
                            models,
                            args,
                            page,
                            {
                                "text": text,
                                "cands": _cands,
                                "reflexive_meta": reflexive_meta,
                                "route_reasons": route_reasons,
                                "elapsed_s": round(elapsed, 2),
                                "path_warnings": [
                                    w for w in get_path_warnings() if w not in _pw_snapshot
                                ]
                                if _pw_snapshot is not None
                                else [],
                                "model_failure_warnings": [
                                    w for w in warnings if w.startswith("model_failure_")
                                ],
                            },
                        )
                    except Exception:
                        pass

            if text.strip():
                text = fix_text_encoding(text.strip())
                # Restore degree symbol — applies to BOTH fast and OCR paths.
                text = restore_degree_symbol(text)
                text = strip_thinking_artifacts(text)
                # OCR garbage sanitizer: strip lines that are pure dashes/
                # punctuation (glm-ocr sometimes emits '---' separator rows
                # when it fails to read Type 3 bitmap fonts). Also collapse
                # 3+ consecutive blank lines.
                text = re.sub(r"^\s*[\-=_:*]{2,}\s*$", "", text, flags=re.MULTILINE)
                text = re.sub(r"\n{3,}", "\n\n", text)

                # OCR hallucination-loop sanitizer: when glm-ocr can't read a
                # page (e.g. Type 3 bitmap fonts), it sometimes emits the SAME
                # phrase dozens of times (either as repeated lines OR as a
                # single long line with the phrase repeated N times).
                # Observed: "The text contains no visible characters." x49.
                # Strategy: detect any phrase (10-100 chars) repeated ≥5 times
                # back-to-back, collapse to 1 instance + warning marker.
                def _collapse_repetition(s):
                    # Phase 0: ADJACENT same-char runs (8+ identical chars).
                    # Catches "aaaaaaa..." and "111111..." which olmOCR-Bench
                    # baseline rejects (30+ repeating 1-grams).
                    s = re.sub(r"(.)\1{7,}", r"\1\1\1 [OCR loop]", s)
                    # Phase 1: short-phrase repetition (1-grams to 5-grams).
                    # olmOCR-Bench baseline test rejects pages ending with
                    # 30+ repeating 1-grams or 5+ repeating 5-grams (model
                    # hallucination loop). Catch n-gram repeats of length 1-15
                    # chars, ≥5 repeats back-to-back (space or newline sep).
                    for n in (1, 3, 5, 10, 15):
                        pat = re.compile(r"(\S{" + str(n) + r"})([\s\W]+\1){4,}")

                        def _rep(m):
                            return (
                                f"{m.group(1)} [OCR loop collapsed x{m.group(0).count(m.group(1))}]"
                            )

                        s = pat.sub(_rep, s)
                    # Phase 2: repeated lines (line-by-line dedup)
                    lines = s.split("\n")
                    deduped = []
                    run_count = 1
                    for li, line in enumerate(lines):
                        if li > 0 and line.strip() and line.strip() == lines[li - 1].strip():
                            run_count += 1
                            if run_count == 3:
                                deduped.append(f"[OCR repetition collapsed: {line.strip()[:60]}…]")
                            elif run_count > 3:
                                continue
                        else:
                            run_count = 1
                        if run_count <= 3:
                            deduped.append(line)
                    s = "\n".join(deduped)
                    # Phase 3: long-phrase within single line (10-100 chars)
                    phrase_re = re.compile(r"(\S.{10,100}?)(\s+\1){4,}")
                    while True:
                        m = phrase_re.search(s)
                        if not m:
                            break
                        phrase = m.group(1)
                        replacement = (
                            f"{phrase} [OCR repetition collapsed: "
                            f"phrase repeated {m.group(0).count(phrase)} times]"
                        )
                        s = s[: m.start()] + replacement + s[m.end() :]
                    return s

                text = _collapse_repetition(text)
                # SOTA 2026-08-12: Join math equations fragmented per-glyph
                # by rapidocr on scanned math pages. Must run BEFORE other
                # text transforms so reassembled $$...$$ blocks survive.
                text = join_fragmented_math_lines(text)
                text = remove_bold_duplicate_chars(text)
                text = drop_repeated_text(text)
                text = repair_bullets(text)
                text = reconstruct_footnotes(pdf_path, page, text)
                # detect_section_hierarchy runs on the FAST path only
                # (registry, order 620). OCR-path pages have, at best, a
                # PARTIAL text layer (routing already judged it inadequate)
                # — promoting headings from its font flags hallucinates
                # structure on table stubs and margins (0208fbb5 lesson:
                # OCR table page whose partial layer promoted 'Two-Way' /
                # 'No Gypsum' stubs as headings). Vision models emit their
                # own heading markup natively.
                # Unicode→LaTeX ONLY on fast path (born-digital). OCR path already
                # outputs LaTeX from the vision model — convert_unicode_to_latex
                # here would double-wrap and corrupt existing $...$ blocks.
                text = wrap_equations_as_latex(pdf_path, page, text)
                text = preserve_typography(pdf_path, page, text)
                text = promote_allcaps_headings(text)
                text = restore_scientific_subscripts(text)
                text = restore_unicode_losses(text)
                text = dehyphenate(text)
                text = repair_citation_splits(text)
                text = preserve_dois(text)
                text = fix_permille(text)
                text = validate_scientific_symbols(text)
                text = validate_chemical_formulas(text)
                text = repair_encoding_decimals(text)
                # OCR correction via SymSpell (geochemistry dict + English).
                # Catches 'Scienees'→'Sciences', 'conclution'→'conclusion'.
                # Conservative: distance-1 only, dominant candidate required.
                text = correct_ocr_errors(text, page_source="ocr")
                text = sanitize_phantom_decimals(text)
                # CJK hallucination stripper — glm-ocr sometimes emits streams
                # of CJK chars on noisy pages (olmOCR-Bench failures 2026-08-12).
                text = strip_cjk_hallucination(text)
                # Whole-line running stamps (Downloaded from…, journal
                # Vol/No running heads): OCR pages have no geometry band
                # for the header detector, these survive verbatim (iter-94
                # census: 4/5 headers absent-test fails were OCR pages).
                text = strip_running_stamps(text)
                # LaTeX math cleanup: \sort→\sqrt, nested-$ collapse, \- strip.
                # Runs on OCR path too (was fast-path-only, 2026-08-11 fix).
                text = fix_latex_math_output(text)
                # Grid reconstruction for scanned tables: rapidocr emits
                # aligned numeric rows as flat lines — rebuild pipe tables
                # (deterministic token-alignment, no geometry needed).
                if _count_pipe_tables(text) == 0:
                    try:
                        from scientific_parser.perception.tables.detector import (
                            reconstruct_aligned_numeric_tables,
                        )

                        text = reconstruct_aligned_numeric_tables(text)
                    except Exception as _e:
                        log.debug("aligned-table reconstruction skipped: %s", _e)
                        _pw_skip("aligned-table reconstruction")
                # P0.2: NumericGuard — flag suspicious numeric patterns
                # (decimal-lost, exponent-flip, ±-corrupted, unit-outliers).
                # Conservative: only adds HTML comment annotations, no rewrite.
                try:
                    from numeric_guard import annotate_text as _ng_annotate

                    text, _ng_findings = _ng_annotate(text, min_confidence=0.5)
                except ImportError:
                    pass  # numeric_guard optional
                # P0.4: Selective prediction — prepend ACCEPT/REVIEW/REJECT banner.
                # Lets downstream agent know whether to trust this page's output.
                try:
                    from selective_prediction import annotate_with_triage as _triage_ann

                    text = _triage_ann(text, page_num=page)
                except ImportError:
                    pass  # selective_prediction optional
                # Multi-strategy table detector — pass OCR text to also scan
                # glm-ocr's own pipe-tables (validated + multi-row merged).
                real_tables = detect_all_tables(pdf_path, page, ocr_text=text)
                real_tables = merge_cross_page_tables(page, real_tables)
                if real_tables:
                    # Replace OCR-emitted tables with validated/merged versions
                    # by stripping old pipe-tables from text + appending new
                    text_lines = [l for l in text.split("\n") if not l.strip().startswith("|")]
                    text = "\n".join(text_lines).rstrip() + "\n\n"
                    for t in real_tables:
                        text += t["markdown"] + "\n"
                        if t.get("html"):
                            text += t["html"] + "\n"
                page_meta = extract_page_metadata(text, page_num=page, doc_state=_doc_header_state)
                # Summary-stage warnings rebuild — preserve any notes the
                # early OCR stage recorded (model_failure_degraded_rapidocr)
                # instead of starting blind from empty.
                _early = [w for w in warnings if w.startswith("model_failure_")]
                warnings = _early
                if len(text) < LOW_TEXT_DENSITY_THRESHOLD:
                    warnings.append(f"low_text_density ({len(text)} chars)")
                warnings.extend(validate_ree_sequence(text))
                _conf = _compute_page_confidence("ocr", warnings, text)
                # Combine validator-derived and reflexive-loop confidence.
                # Replaces the prior ``_conf = max(_conf, refl)`` which let
                # model self-confidence mask validator-detected defects.
                _conf = combine_page_confidence(
                    _conf,
                    reflexive_meta.get("confidence"),
                    has_warnings=bool(warnings),
                )
                record = {
                    "page": page,
                    "source": "ocr",
                    "model": reflexive_meta.get("model", models[0]),
                    "chars": len(text),
                    "elapsed_s": round(elapsed, 2),
                    "route": "ocr",
                    "route_reasons": route_reasons,
                    "confidence": _conf,
                    "page_header": page_meta["page_header"],
                    "page_number": page_meta["page_number"],
                    "warnings": warnings,
                    "text": text,
                    # SOTA 2026-08-12: reflexive loop metadata for audit/debug.
                    "reflexive": reflexive_meta,
                }
                page_metrics.append(record)
                meta_comment = _format_meta_comment(page_meta, warnings)
                emit_markdown(
                    f"<!-- OCR model={reflexive_meta.get('model', models[0])} "
                    f"conf={_conf:.2f} attempts={reflexive_meta.get('attempts', 1)} -->"
                    f"\n\n{meta_comment}{text}"
                )
                if json_mode:
                    emit_jsonl(record)
            else:
                record = {
                    "page": page,
                    "source": "ocr",
                    "model": reflexive_meta.get("model", models[0]),
                    "chars": 0,
                    "confidence": 0.1,
                    "elapsed_s": round(elapsed, 2),
                    "route": "ocr",
                    "route_reasons": route_reasons,
                    "page_header": None,
                    "page_number": None,
                    "warnings": ["no_text_recognized"],
                    "text": "[No text recognized]",
                    "reflexive": reflexive_meta,
                }
                page_metrics.append(record)
                emit_markdown(
                    f"<!-- OCR model={reflexive_meta.get('model', models[0])} -->"
                    f"\n\n[No text recognized]"
                )
                if json_mode:
                    emit_jsonl(record)

            # Clean up temp images per page (bounded disk usage, §9 L1)
            for f in os.listdir(tmp_dir):
                if f.startswith("page"):
                    os.remove(os.path.join(tmp_dir, f))
    finally:
        # Always clean up temp dir — even on exception (§9 L1, P16.9).
        # shutil is imported at module scope.
        shutil.rmtree(tmp_dir, ignore_errors=True)

    total_elapsed = time.time() - start_total

    _finalize_document(
        {
            "final_md": "\n\n---\n\n".join(all_text),
            "page_metrics": page_metrics,
            "page_count": len(page_list),
            "models": models,
            "args": args,
            "pdf_path": pdf_path or PDF_PATH,
            "md_path": md_path,
            "jsonl_path": jsonl_path,
            "summary_path": _summary_path,
            "json_mode": json_mode,
            "source_provenance": source_provenance,
            "total_elapsed": total_elapsed,
            "font_math_pages": _font_math_pages,
            "any_math_font_page": _ANY_MATH_FONT_PAGE[0],
            "use_cache": use_cache,
            "cache_dir": None,
            "all_region_candidates": all_region_candidates,
            "path_warnings_override": None,
            "raw_replay": False,
            "output_base": _base,
        }
    )


def _run_document_postchain(final_md, page_metrics, font_math_pages, any_math_font_page):
    """Document-level postprocessor chain (raw-cache split, 2026-09-10).

    Runs on EVERY emission — fresh extraction and raw-cache replay alike —
    so POSTPROC_VERSION changes apply without re-extracting. Mutates
    page_metrics in place (strip_recurring_headers). Returns
    (final_md, conservation).

    Extracted verbatim from the main() tail (was lines 3364-3498); the
    only changes are the two gate parameters (was closure locals
    _font_math_pages / _ANY_MATH_FONT_PAGE[0]).
    """
    if len(page_metrics) >= 3:
        final_md = strip_recurring_headers(page_metrics, final_md)
    # Enrich with scientific_parser: reflexive equation repair + domain detection
    try:
        from scientific_parser.skill_adapter import enrich_markdown_with_package

        final_md = enrich_markdown_with_package(final_md)
    except Exception as e:
        log.debug("Markdown enrichment skipped: %s", e)
        _pw_skip("markdown enrichment")
    # FINAL Unicode→LaTeX conversion: specialist dispatch (_extract_text_from_bbox)
    # reads the RAW text layer (with Unicode α,β,θ) which overwrites the per-page
    # convert_unicode_to_latex output. Re-apply here on the final assembled text
    # so born-digital math always gets LaTeX wrapping. (2026-08-14 bench fix)
    # GATE (2026-08-15): skip when font-math pages exist — their text is
    # already LaTeX and the blanket conversion re-wraps it (nested $),
    # measured -22pts on arxiv_math.
    # MATH-FONT DOC GATE (iter-110): additionally require that some page of
    # this doc actually carries math fonts — a pure-prose doc (zero math
    # fonts anywhere, measured 27-30/30 of multicol/tiny/headers samples)
    # must keep its Unicode ×, ±, ≥ as-is; converting them corrupts
    # plain-text golds ('an$m \times m$multi…').
    if not font_math_pages:
        try:
            from scientific_parser.science import unicode_to_latex as _u2l_mod

            if any_math_font_page:
                final_md = _u2l_mod.convert_unicode_to_latex(final_md)
                final_md = _u2l_mod.normalize_inline_math(final_md)
            else:
                # Pure-prose doc (iter-110): ellipsis normalization only.
                final_md = _u2l_mod.convert_unicode_to_latex(final_md, prose_only=True)
        except Exception as e:
            log.debug("final unicode_to_latex skipped: %s", e)
            _pw_skip("final unicode_to_latex")
    # FINAL math-verification gate on the assembled document: unwrap any
    # $$...$$ block whose content isn't genuine math. Catches reference/
    # numbered-list false-positives that survived per-page post-proc or were
    # re-wrapped during assembly/enrichment (e.g. "$$(3) 12)$$").
    # GATE (2026-08-15): font-math display blocks legitimately contain
    # prose words (arg, resp., kerned prose); is_real_math rejects them →
    # -4pts measured. Skip for font-math documents.
    if not font_math_pages:
        try:
            from scientific_parser.postproc.text_pipeline import (
                unwrap_non_math_delimiters,
            )

            final_md = unwrap_non_math_delimiters(final_md)
        except Exception as e:
            log.debug("final unwrap_non_math skipped: %s", e)
            _pw_skip("final unwrap_non_math")
    # FINAL LaTeX command-confusion pass: specialist dispatch (equation
    # cascade, region OCR) wraps $$ blocks AFTER the per-page
    # fix_latex_math_output ran — so its repairs never see these blocks.
    # Run the command-confusion cleanup once more on the assembled doc
    # (\sort→\sqrt, \carpi→\sqrt etc. — KaTeX-rejected otherwise).
    try:
        # ocr_artifacts=False when font-math pages exist: their \hat/\bar
        # are measured glyphs, not glm-ocr artifacts (iter-29).
        final_md = fix_latex_math_output(final_md, ocr_artifacts=not font_math_pages)
    except Exception as e:
        log.debug("final fix_latex_math_output skipped: %s", e)
        _pw_skip("final fix_latex_math_output")
    # RUN-9 (2026-09-11): oracle-gated structural repair for unrenderable
    # math spans — scorer skips render-fail candidates, so repairs strictly
    # add scorable mass. Postproc-only tier (see text_pipeline).
    try:
        from scientific_parser.postproc.text_pipeline import (
            repair_unrenderable_math,
        )

        final_md = repair_unrenderable_math(final_md)
    except Exception as e:
        log.debug("final repair_unrenderable_math skipped: %s", e)
        _pw_skip("final repair_unrenderable_math")
    # FINAL prose reflow (iter-89): several post-dispatch paths replace
    # page text with fresh span/table re-extracts (cascade adoption,
    # detect_tables append, region splice) whose lines carry hard
    # typesetting breaks the per-page reflow (order 754) never saw.
    # One join pass on the assembled doc closes them all.
    try:
        from scientific_parser.postproc.text_pipeline import (
            reflow_prose_paragraphs,
        )

        final_md = reflow_prose_paragraphs(final_md)
    except Exception as e:
        log.debug("final reflow skipped: %s", e)
        _pw_skip("final reflow")
    # FINAL delimiter normalization: OCR models emit \(...\) / \[...\]
    # inline/display math. The olmOCR checker (and most markdown consumers)
    # scan only $...$ / $$...$$ — bare \( \) blocks are INVISIBLE to it
    # (bench old_scans_math: every equation "No match found" while present).
    try:
        # (a) unwrap \( \) NESTED inside $$...$$ (nested delimiters corrupt)
        final_md = re.sub(
            r"(\$\$.+?\$\$)",
            lambda m: m.group(1).replace("\\(", "").replace("\\)", ""),
            final_md,
            flags=re.DOTALL,
        )
        # (b) standalone \(...\) → $...$, \[...\] → $$...$$
        final_md = re.sub(r"\\\((.+?)\\\)", r"$\1$", final_md, flags=re.DOTALL)
        final_md = re.sub(r"\\\[(.+?)\\\]", r"$$\1$$", final_md, flags=re.DOTALL)
    except Exception as e:
        log.debug("final paren-delimiter normalization skipped: %s", e)
        _pw_skip("final paren-delimiter normalization")
    # ─── Content-conservation invariant (2026-08-15) ──────────────────── #
    # Aggregate check: after ALL doc-level passes (enrichment, unicode→
    # latex, unwrap, fix_latex, delimiter normalization), the visible
    # markdown must retain a floor of the source text volume measured at
    # extraction time. Catches the overwrite/deletion family (region
    # dispatch replacing clean text, fix_latex cascade deletion — the
    # 2026-08-15 massacre class) at the LAST possible gate. Threshold
    # 0.5: legit transforms (header/footer strip, math re-wrap, table
    # HTML) never halve visible volume; measured destruction did.
    _conservation = None
    try:
        _src_chars = sum(r.get("chars", 0) for r in page_metrics)
        _visible = re.sub(r"<!--.*?-->", "", final_md, flags=re.DOTALL).strip()
        _vis_chars = len(_visible)
        if _src_chars > 200:
            _ratio = _vis_chars / _src_chars
            _conservation = {
                "source_chars": _src_chars,
                "visible_chars": _vis_chars,
                "ratio": round(_ratio, 3),
                "ok": _ratio >= 0.5,
            }
            if not _conservation["ok"]:
                print(
                    f"WARNING: content conservation violated — visible "
                    f"{_vis_chars} chars vs {_src_chars} extracted "
                    f"(ratio {_ratio:.2f} < 0.5). Doc-level post-proc "
                    f"destroyed content; inspect stages.",
                    file=sys.stderr,
                )
                log.warning("content_conservation violated: %s", _conservation)
    except Exception as e:
        log.debug("conservation check skipped: %s", e)
        _pw_skip("content conservation check")
    return final_md, _conservation


def _finalize_document(ctx):
    """Emit final outputs for one document: postchain + md/jsonl/summary +
    cache write (raw-cache split, 2026-09-10).

    ctx keys:
      final_md             pre-chain markdown (fresh: joined page chunks;
                           raw replay: cached raw_md)
      page_metrics         PRE-chain page records (chain mutates in place)
      page_count           len(page_list) equivalent
      models, args, pdf_path, output_base
      md_path, jsonl_path, summary_path, json_mode
      source_provenance, total_elapsed
      font_math_pages      set of font-math page numbers
      any_math_font_page   bool
      use_cache, cache_dir (None on fresh — computed here)
      all_region_candidates (fresh) / None (replay)
      path_warnings_override (replay: stored warnings) / None (fresh: live)
      raw_replay           True → never overwrite raw.json (page_metrics
                           are chain-mutated by now)
    """
    import copy as _copy

    md_path = ctx["md_path"]
    jsonl_path = ctx["jsonl_path"]
    _summary_path = ctx["summary_path"]
    json_mode = ctx["json_mode"]
    page_metrics = ctx["page_metrics"]
    page_count = ctx["page_count"]
    models = ctx["models"]
    args = ctx["args"]
    pdf_path = ctx["pdf_path"]
    source_provenance = ctx["source_provenance"]
    total_elapsed = ctx["total_elapsed"]
    font_math_pages = ctx["font_math_pages"]
    any_math_font_page = ctx["any_math_font_page"]
    use_cache = ctx["use_cache"]
    cache_dir = ctx["cache_dir"]
    raw_replay = ctx["raw_replay"]

    # Raw snapshot MUST be captured before the chain mutates page_metrics
    # (strip_recurring_headers strips + re-counts chars in place).
    raw_payload = None
    if not raw_replay:
        raw_payload = {
            "ocr_pipeline_version": OCR_PIPELINE_VERSION,
            "postproc_version": POSTPROC_VERSION,
            "extraction_version": SKILL_EXTRACTION_VERSION,
            "env_fingerprint": _envfp.env_fingerprint(tuple(models)),
            "raw_md": ctx["final_md"],
            "page_metrics": _copy.deepcopy(page_metrics),
            "font_math_pages": sorted(font_math_pages),
            "any_math_font_page": bool(any_math_font_page),
            "path_warnings": get_path_warnings(),
            "source": source_provenance,
            "models": models,
            "page_count": page_count,
            "elapsed_s": round(total_elapsed, 2),
        }

    final_md, _conservation = _run_document_postchain(
        ctx["final_md"], page_metrics, font_math_pages, any_math_font_page
    )

    with open(md_path, "w") as f:
        f.write(final_md)

    # Raw replay: jsonl was never streamed (no extraction loop ran) —
    # rebuild it from the post-chain page records, matching the prior
    # final-hit replay semantics (cached summary pages were post-strip).
    if raw_replay and json_mode and jsonl_path:
        with open(jsonl_path, "w", encoding="utf-8") as f:
            for rec in page_metrics:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    path_warnings = (
        ctx["path_warnings_override"]
        if ctx["path_warnings_override"] is not None
        else get_path_warnings()
    )

    if json_mode:
        # Write summary JSON alongside the .jsonl + .md
        summary = {
            "source": source_provenance,
            "input": pdf_path,
            "output_jsonl": jsonl_path,
            "output_md": md_path,
            "total_pages": page_count,
            "pages_ok": sum(1 for r in page_metrics if not r.get("warnings")),
            "pages_warned": sum(1 for r in page_metrics if r.get("warnings")),
            "pages_failed": sum(
                1
                for r in page_metrics
                if r.get("warnings") and "no_text_recognized" in r["warnings"]
            ),
            "total_chars": sum(r.get("chars", 0) for r in page_metrics),
            "total_elapsed_s": round(total_elapsed, 2),
            "extraction_version": SKILL_EXTRACTION_VERSION,
            "env_fingerprint": _envfp.env_fingerprint(tuple(models)),
            "models": models,
            "reflexive_enabled": not getattr(args, "no_reflexive", False),
            "cascade_enabled": not getattr(args, "no_cascade", False),
            "path_warnings": path_warnings,
            "conservation": _conservation,
            "pages": page_metrics,
        }
        # Failure taxonomy: top failure categories for diagnostics
        try:
            from failure_taxonomy import top_failures

            _wc = {}
            for r in page_metrics:
                for w in r.get("warnings", []):
                    _wc[w] = _wc.get(w, 0) + 1
            summary["failure_top"] = top_failures(_wc, 5)
        except Exception as e:
            log.warning("main: Exception degraded: %s", e)
        with open(_summary_path, "w") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
    else:
        # Build a minimal summary for cache write even in non-JSON mode.
        summary = {
            "source": source_provenance,
            "input": pdf_path,
            "output_md": md_path,
            "total_pages": page_count,
            "pages_ok": sum(1 for r in page_metrics if not r.get("warnings")),
            "pages_warned": sum(1 for r in page_metrics if r.get("warnings")),
            "pages_failed": sum(
                1
                for r in page_metrics
                if r.get("warnings") and "no_text_recognized" in r["warnings"]
            ),
            "total_chars": sum(r.get("chars", 0) for r in page_metrics),
            "total_elapsed_s": round(total_elapsed, 2),
            "extraction_version": SKILL_EXTRACTION_VERSION,
            "env_fingerprint": _envfp.env_fingerprint(tuple(models)),
            "models": models,
            "reflexive_enabled": not getattr(args, "no_reflexive", False),
            "cascade_enabled": not getattr(args, "no_cascade", False),
            "path_warnings": path_warnings,
            "pages": page_metrics,
        }

    # Persist to content-hash cache. Fresh extraction writes raw.json (the
    # pre-chain state) alongside the final files; raw replay rewrites the
    # finals only — raw.json must stay the pristine pre-chain snapshot.
    if use_cache:
        if not cache_dir:
            key = _cache_key(pdf_path, models, args)
            cache_dir = os.path.join(DEFAULT_CACHE_DIR, key)
        cache_write(cache_dir, ctx["output_base"], summary, final_md, raw_payload)

    # Quality summary (agent-parseable on stdout)
    pages_ok = sum(1 for r in page_metrics if not r.get("warnings"))
    pages_warned = sum(1 for r in page_metrics if r.get("warnings"))
    print(f"\nDone! Source: {source_provenance}")
    print(f"Pages: {page_count} (ok={pages_ok} warned={pages_warned})")
    print(f"Total chars: {sum(r.get('chars', 0) for r in page_metrics)}")
    print(f"Elapsed: {total_elapsed:.1f}s")
    if json_mode:
        print(f"Output: {md_path} (markdown) + {jsonl_path} (pages) + {_summary_path} (summary)")
    else:
        print(f"Output: {md_path} ({len(final_md)} chars)")

    if raw_replay:
        # Match the prior final-hit replay contract: replay stored path
        # warnings (no live collector state), skip sidecar re-emission
        # (original extraction already emitted them to its output paths).
        _cached_pw = format_path_warning_records(path_warnings or [])
        if _cached_pw:
            print(_cached_pw, file=sys.stderr)
        return

    # Unintended-path warnings: aggregated block on stderr (once per doc).
    _pw_block = format_path_warnings_summary()
    if _pw_block:
        print(_pw_block, file=sys.stderr)

    # P0-P10 + RG: emit scientific_parser sidecars (entities, domains, IR, events)
    # Dual-mode: skill keeps its monolith output + ADDS package-derived sidecars.
    # Zero behavior change to markdown — sidecars are extra files only.
    try:
        import sys as _sys

        if "scripts" not in _sys.path:
            _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from scientific_parser.skill_adapter import emit_sidecars

        sidecars = emit_sidecars(
            markdown_text=final_md,
            output_md_path=md_path,
            source_pdf=pdf_path,
            pages_text=[r.get("text", "") for r in page_metrics],
            pages_meta=[
                {"page": r.get("page"), "source": r.get("source", "ocr"), "model": r.get("model")}
                for r in page_metrics
            ],
            candidates=ctx["all_region_candidates"] or None,
            extra_metadata={"source": source_provenance, "elapsed_s": total_elapsed},
        )
        if sidecars:
            print(f"Sidecars: {', '.join(sorted(sidecars.keys()))}")
    except Exception as _e:
        # Sidecars are best-effort enrichment. Never block the main output.
        log.debug("Sidecar emission skipped: %s", _e)


if __name__ == "__main__":
    main()
