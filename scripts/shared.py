"""Shared constants, patterns, and state for scientific_parser.

Single source of truth. Imported by pdf_ocr.py AND ALL extracted modules.
No circular dependency — shared.py imports nothing from pdf_ocr.
"""

import re, os, sys, json, time, hashlib, logging, subprocess, tempfile, shutil
from pathlib import Path
from dataclasses import dataclass

# From: pdf_ocr.py
ALLCAPS_HEADING_RE = re.compile(
    r"^(?:"
    r"(?<=[\n][\n])"  # preceded by blank line (lookbehind)
    r"|(?<=^)"  # or start of string
    r")"
    r"([A-Z][A-Z\s.,'&\-:]{3,}[A-Z])"  # all-caps line, >=4 chars
    r"(?=\n\n|\n$|$)",  # followed by blank line or end
)

# From: scientific_parser/postproc/text_pipeline.py
BIBLIOGRAPHY_MARKERS = [
    "References",
    "REFERENCES",
    "Literature Cited",
    "Bibliography",
    "BIBLIOGRAPHY",
    "参考文献",
    "引用文献",
]

# From: scientific_parser/postproc/text_pipeline.py
BRACKET_HEADER_RE = re.compile(r"^\s*\[.*\]\s*$")

# From: pdf_ocr.py
CITATION_SPLIT_RE = re.compile(r"(\w)-\n\[(\d+)\]|\[(\d+)\]-\n(\w)")

# From: pdf_ocr.py
CRYSTALLOGRAPHIC_RE = re.compile(r"(?<![A-Za-z])[\(\[\{<][\d\-\d]+[\)\]\}>](?![A-Za-z])")

# From: pdf_ocr.py
CRYSTAL_INDICES_RE = re.compile(
    r"(?<![A-Za-z])"
    r"[\(\[\{]"
    r"(\d[\d\-\d]*)"
    r"[\)\]\}]"
    r"(?![A-Za-z])"
)

# From: pdf_ocr.py
DEFAULT_CACHE_DIR = os.path.expanduser(os.environ.get("PDF_OCR_CACHE_DIR", "~/.cache/pdf-ocr"))

# From: scientific_parser/ingest/ocr_engine.py
DEFAULT_MODEL_PROFILE = {
    "temperature": 0.0,
    "num_predict": 131072,
    "num_ctx": 131072,
    "repeat_penalty": 1.1,
}

# From: pdf_ocr.py
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

# From: pdf_ocr.py
DOI_RE = re.compile(r"\b(10\.\d{4,}/[^\s\)\]\}>,]+)")

# From: pdf_ocr.py
DPI = 150  # 150 DPI balances accuracy (chemistry/citations/equations) vs speed.

# From: pdf_ocr.py
DUPLICATE_CHAR_MIN_LEN = 6  # only check words ≥6 chars (avoids "WWII", "AA", etc.)

# From: scientific_parser/postproc/text_pipeline.py
FORMULA_RE = re.compile(
    r"\b(?:[A-Z][a-z]?\d*){2,}\b"
    r"|\b[A-Z][a-z]?\d+0\d+\b"
)

# From: scientific_parser/postproc/text_pipeline.py
HEADER_LINE_MAX_CHARS = 80

# From: scientific_parser/postproc/text_pipeline.py
LOW_TEXT_DENSITY_THRESHOLD = 200

# From: scientific_parser/perception/equations/math_extractor.py
MATH_FONT_PREFIXES = (
    "CMMI",  # Computer Modern Math Italic (variables)
    "CMSY",  # Computer Modern Math Symbols (−, ·, ⋯)
    "CMEX",  # Computer Modern Extension (∑, ∏, ∫, brackets)
    "MSBM",  # AMS Blackboard Bold (ℕ, ℝ, ℂ)
    "MSAM",  # AMS Symbols A
    "MSBM",  # AMS Blackboard Bold
    "EU1",
    "EU2",  # Euler math fonts
    "Frak",  # Fraktur
    "Bbb",  # Blackboard bold variants
    "Math",  # Generic math fonts
    "CMMIB",  # Computer Modern Math Italic Bold
    "CMSYB",  # Computer Modern Symbol Bold
)

# From: pdf_ocr.py
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

# From: pdf_ocr.py
MATH_OPERATOR_CHARS = set("=±×÷∑∫∂√∞≈≠≤≥αβγδεζηθικλμνξπρστυφχψωΔΣΠΩ∇·→↔∝")

# From: pdf_ocr.py
MAX_DIM = 4096  # max pixel dimension for images sent to model.

# From: scientific_parser/science/geo_validators.py
MINERAL_ABBREVS = {
    "Ab": "albite",
    "Act": "actinolite",
    "Bt": "biotite",
    "Cal": "calcite",
    "Chl": "chlorite",
    "Cpx": "clinopyroxene",
    "Grt": "garnet",
    "Hbl": "hornblende",
    "Ilm": "ilmenite",
    "Kfs": "K-feldspar",
    "Ky": "kyanite",
    "Mag": "magnetite",
    "Ms": "muscovite",
    "Ol": "olivine",
    "Opx": "orthopyroxene",
    "Pl": "plagioclase",
    "Qtz": "quartz",
    "Rt": "rutile",
    "Sil": "sillimanite",
    "Spl": "spinel",
    "St": "staurolite",
    "Ttn": "titanite",
    "Tur": "tourmaline",
    "Zrn": "zircon",
    "Amp": "amphibole",
    "Px": "pyroxene",
    "Ep": "epidote",
    "Ser": "serpentine",
    "Tlc": "talc",
    "Tr": "tremolite",
    "Ap": "apatite",
    "Sp": "spinel",
}

# From: pdf_ocr.py
MIN_SPAN_CHARS = 8  # preserve_typography_fitz: only wrap spans ≥8 chars (false-positive guard)

# From: scientific_parser/ingest/ocr_engine.py
MODEL_PROFILES = {
    "glm-ocr": {
        "temperature": 0.0,
        # num_ctx=16384: image (~2-4K tokens) + prompt (~200) + output (~3K) = ~6K used.
        # Reserving more VRAM (e.g. 131072) slows attention with no quality gain.
        # num_predict=8192: OCR output is rarely >3K tokens; 8K is comfortable headroom.
        "num_predict": 8192,
        "num_ctx": 16384,
        "repeat_penalty": 1.1,
    },
    # qwen2.5vl — 2026 OCR winner (nullmirror benchmark). Needs /api/chat
    # endpoint (auto-detected in ocr_image). num_ctx=8192 fits 8GB VRAM with
    # image tokens.
    "qwen2.5vl": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen2.5vl:3b": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen2.5vl:7b": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen3-vl": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "qwen3-vl:4b": {
        "temperature": 0.0,
        "num_predict": 8192,
        "num_ctx": 8192,
        "repeat_penalty": 1.1,
    },
    "ovisocr2": {
        "temperature": 0.0,
        "num_predict": 131072,
        "num_ctx": 131072,
        "repeat_penalty": 1.1,
        # Suppress thinking-mode artifacts (Qwen3.5 chat template always emits <think>...</think>)
        "system": "You are an OCR engine. Transcribe text from the image exactly as shown. Do not add commentary, explanations, or reasoning.",
    },
    # dots.ocr (rednote-hilab) — HIGHEST ACCURACY on scientific PDFs:
    # correctly formats LaTeX subscripts ($f_{O_2}$), captures most decimals.
    # Trade-off: ~7x slower than glm-ocr (6.5 min vs 56s per page). Use for
    # small critical docs where accuracy > speed.
    # Pull: ollama pull hf.co/ggml-org/dots.ocr-GGUF:Q8_0
    # Alias created via: ollama create dots.ocr -f Modelfile (FROM hf.co/...)
    "hf.co/ggml-org/dots.ocr-GGUF:Q8_0": {
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    "dots.ocr": {  # short alias
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    # dots.mocr (SandLogic IQ4_XS quantization) — FASTER but less accurate
    # than Q8_0. Introduces "liquidous"/"komalites"-style errors due to 4-bit
    # quantization. Use only when speed > accuracy.
    "hf.co/SandLogicTechnologies/dots.mocr-GGUF:IQ4_XS": {
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
    "dots.mocr": {  # short alias
        "temperature": 0.0,
        "num_predict": 32768,
        "num_ctx": 32768,
        "repeat_penalty": 1.1,
    },
}

# From: pdf_ocr.py
OCR_PER_ATTEMPT_TIMEOUT_S = 240  # per-OCR timeout; on timeout, cascade to next

# From: pdf_ocr.py
OLLAMA_URL = "http://127.0.0.1:11434/api/generate"

# From: pdf_ocr.py

# From: pdf_ocr.py
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

# From: scientific_parser/routing/page_metadata.py
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

# From: scientific_parser/postproc/text_pipeline.py
PAGE_NUMBER_RE = re.compile(r"^\s*\d{1,4}\s*$")

# From: pdf_ocr.py

# From: scientific_parser/postproc/text_pipeline.py
PRESERVE_HYPHENS = {
    # Element-element compounds
    "Fe-bearing",
    "Fe-rich",
    "Fe-poor",
    "Fe-free",
    "U-Pb",
    "Sm-Nd",
    "K-Ar",
    "Ar-Ar",
    "Rb-Sr",
    "Lu-Hf",
    "Re-Os",
    "La-Ce",
    "Pb-Pb",
    "Mg-Fe",
    "Ca-Mg",
    "Si-Al",
    "Na-K",
    "C-O-H",
    "Fe-Ti",
    "Cr-Ni",
    # Mineral descriptors
    "low-K",
    "high-K",
    "medium-K",
    "high-Mg",
    "low-Mg",
    "high-Al",
    "low-Al",
    "Mg-rich",
    "K-rich",
    "Si-rich",
    # Temperature/pressure
    "high-T",
    "low-T",
    "high-P",
    "low-P",
    "T-P",
    "P-T",
    # Size fractions
    "coarse-grained",
    "fine-grained",
    "medium-grained",
    # Common geochemistry terms
    "H2O-saturated",
    "fluid-saturated",
    "non-conservative",
    "non-ideal",
    # ICP-MS type instruments
    "LA-ICP-MS",
    "HR-ICP-MS",
    "MC-ICP-MS",
    "SIMS",
    "TIMS",
    "EPMA",
}

# From: scientific_parser/core/confidence_helpers.py
REE_OCR_CONFUSIONS = {"Cd": "Gd", "Ib": "Tb", "Or": "Er", "Tni": "Tm", "Cin": "Sm"}

# From: scientific_parser/postproc/text_pipeline.py
REE_SEQUENCE = [
    "La",
    "Ce",
    "Pr",
    "Nd",
    "Pm",
    "Sm",
    "Eu",
    "Gd",
    "Tb",
    "Dy",
    "Ho",
    "Er",
    "Tm",
    "Yb",
    "Lu",
]

# From: pdf_ocr.py
REFLEXIVE_CONFIDENCE_FLOOR = 0.55  # pages scoring below trigger retry with feedback

# From: pdf_ocr.py
REFLEXIVE_MAX_ATTEMPTS_PER_MODEL = 2  # hard cap per model (arXiv 2603.22651)

# From: pdf_ocr.py
REPEAT_LINE_MIN_CHARS = 40

# From: pdf_ocr.py
ROUTER_ALPHANUM_MIN = 0.30  # Marker: garbled text detector

# From: pdf_ocr.py
ROUTER_IMAGE_COVERAGE_THRESHOLD = 0.65  # Marker: image covers ≥65% of page = scanned

# From: pdf_ocr.py
ROUTER_INVALID_CHAR_ABS = 6  # Marker: minimum absolute count

# From: pdf_ocr.py
ROUTER_INVALID_CHAR_PCT = 0.03  # Marker: encoding issue

# From: pdf_ocr.py
ROUTER_MIN_AVG_WPL = (
    5.0  # NEW: avg words/line below this = positioned-glyph PDF (sakai1968 pattern)
)

# From: pdf_ocr.py
ROUTER_MOJIBAKE_MIN_MARKERS = (
    3  # R14b: isolated ´/¨/ˆ count that routes to OCR (ToUnicode mojibake)
)

# From: pdf_ocr.py
ROUTER_MIN_CHARS = 100  # below = scanned page or extraction fail

# From: pdf_ocr.py
ROUTER_MULTI_SPACE_RUN_MAX = 0.30  # NEW: fragmented table detector

# From: pdf_ocr.py
ROUTER_NEWLINE_RATIO_MAX = 0.60  # Marker: one-word-per-line detector

# From: pdf_ocr.py
ROUTER_SHORT_LINE_MIN_CHARS = 500  # NEW: avoid false-positive on small diagrams

# From: pdf_ocr.py
ROUTER_SHORT_LINE_RATIO_MAX = 0.90  # NEW: shattered-cell detector (post-default-mode)

# From: pdf_ocr.py
ROUTER_SPACE_RATIO_MAX = 0.70  # Marker: whitespace flood (fragmented)

# From: scientific_parser/core/confidence_helpers.py
SCIENTIFIC_SYMBOL_RULES = [
    (r"(\d)\s*u(?=[mglL]\b)", r"\1μ"),
    (r"(\d)\s*W(?=\s|$)", r"\1 Ω"),
    (r"(\d)\s*[xX]\s*(\d)", r"\1 × \2"),
    (
        r"(\d)\s*-\s*(\d)",
        r"\1-\2",
    ),  # normalize range spacing, KEEP ASCII hyphen (U+2212 breaks exact-match recall)
    (r"(\d)\s*[oO]\s*([CFK])\b", r"\1°\2"),
    (r"(\d)\s*A(?=\s|$)", r"\1 Å"),
]

# From: pdf_ocr.py
SINGLE_ELEM_ZERO_RE = re.compile(r"\b([A-Z][a-z]?)0(\d*)\b|\b([A-Z][a-z]?)(\d+)0\b")

# From: pdf_ocr.py
# Version scheme (2026-09-10 raw-cache split; 2026-09-13 stage-granular split
# R19): the extraction cache stores PRE-postprocessor ("raw") state keyed by
# OCR_PIPELINE_VERSION, and the document postprocessor chain re-runs on every
# cache replay. R19 splits OCR_PIPELINE_VERSION into two STAGES so
# deterministic work never re-rolls stochastic work:
#   OCR_CORE_VERSION  — stochastic tier: routing signals (incl. signal-11),
#                       OCR models/prompts, image preprocessing, math
#                       cascade, VL fallbacks. Bumping re-rolls routed pages.
#   DETECTION_VERSION — deterministic tier: table detection (incl.
#                       img2table last-resort), layout, reading order, page
#                       assembly. Bumping replays WARM per-page OCR memos
#                       (pages.ocr.json keyed by core only) — zero re-roll.
# OCR_PIPELINE_VERSION = f"{OCR_CORE_VERSION}+{DETECTION_VERSION}" feeds the
# level-2 raw-cache key as before. Bump POSTPROC_VERSION when only the
# document chain (pdf_ocr.py _run_document_postchain) changes — that must NOT
# invalidate cached extractions (run-7 RCA: monolithic key re-rolled
# stochastic OCR on scanned pages, tiny 0.8235->0.8009). Motivation (run-16):
# monolithic bumps re-rolled ALL chandra pages bench-wide, sigma +-5-10
# tests/cat roll noise swamping small real gains (-15 roll vs +12 real).
OCR_CORE_VERSION = "2026.09.13.core1"
# det3 (2026-09-15, EXP-1): table-evidence gate (page_needs_table_chain,
# raw+stage dual-surface), native_find_tables memo consolidation, and the
# merge_cross_page_tables _OPEN_TABLES_STATE init fix (order-dependent
# silent table loss). Raw cache must NOT replay pre-det3 page outputs.
# det4 (2026-09-16, EXP-1.5): router corroboration (born-digital alpha>=200
# suppresses low_alphanum/invalid_chars), by_page control-glyph scrub,
# fix_text_encoding registry scrub, and table-cell scrub across ALL
# renderers (structure.Table to_markdown/to_html, normalize_grid_rows,
# detector emitters). Raw cache must not replay un-scrubbed page outputs.
# det5 (2026-09-16, EXP-1.5b): corroboration now requires the FITZ surface
# (consumed by the fast path) to be healthy too — 05a0e5bc-class pages read
# healthy via pdftotext but carry symbol-glyph garbage in fitz; EXP-1.6 CNN
# gate REVERTED (47a4c46d) after its hidden 04af23cb regression surfaced
# once det4 forced fresh extraction (bench5 had been a vacuous cache replay).
# det6 (2026-09-16, EXP-3): cell x-clip post-pass in detect_all_tables —
# geometry-splits jammed cells ('9 M-PSI/06' one-cell jams, 033c1efd class)
# via sibling-row column consensus, empty-neighbor-only moves. Page-level
# baked lane: raw cache must not replay un-split page outputs.
# det7 (2026-09-16, EXP-3b): split iterates to FIXPOINT (max 3 passes) —
# single-pass split of multi-column jams ('No Element Weight %',
# b8d3ad0f pg4) left partial jams that BLOCKED the emission-time header
# splitter (iter-119a). Bumped post-bench8: raw cache must not replay
# single-pass outputs.
# det8 (2026-09-16, EXP-4): CNN-escalation evidence gate re-landed
# (2aac643a) + label-value prose carve guard — both region-lane and
# page-baked; raw cache must not replay pre-guard carves.
# det10 (2026-09-16, EXP-4b): evidence-gate space fix — candidate bbox
# is render-px, fitz probes need PDF-pt (dispatcher now passes
# bbox_pdf_pt); raster-overlap >= 0.5 rung added (9a61fe78 raster
# tables carry zero text/vector evidence); retyped_from_figure bypass.
# Region lane + page-baked; det8 cache entries for affected raster
# pages are poisoned (bench10 c8c85869/9a61fe78).
# det11 (2026-09-16, EXP-7, DISCARDED): text_recon regrid wired then
# reverted — bench12 tables 0.6939->0.6383 (59 P->F vs 8 F->P);
# whole-table regrids destroy passing relational geometry.
# det12 (2026-09-16): EXP-7 revert — raw cache must not replay the
# poisoned det11 regridded outputs; behavior returns to det10-equivalent.
# det13 (2026-09-17): tounicode crosscheck + font-page escalation path
# fix — tmp_dir is a str, `tmp_dir / name` threw TypeError on EVERY
# suspect-font fast page since v3.0 (227 bench pages silently skipped
# the crosscheck stage; VL glyph re-read never ran via the engine).
# Page-level lane: raw cache must not replay crosscheck-skipped outputs.
# det14 (2026-09-17): xval adoption made surgical — verified-word frm→to
# substitution map instead of wholesale VL page replacement; structured
# pages (>=2 columns / table chain) defer the map and skip VL re-read
# (bench13: wholesale adoption flipped 8 relational tests); font-page
# escalation now recomputes page_meta from post-escalation text so the
# final HF strip sees real bands (3 hf flips). Page-level lanes.
# det15 (2026-09-17): VL narration guard — looks_like_chat_response now
# also rejects image-narration preambles ("The image is a page from...")
# and the fast-path escalation adoption gate vetoes narrated output even
# when the paddleocr-vl swap is unavailable (bench14 hf: narration embed-
# ded header phrases in prose; absent-tests found them at ratio 1.000).
# det16 (2026-09-18): cell-level glyph map — the xval crosscheck's verified
# (frm, to, layer-words) map is now handed to the tables lane and applied
# to table cells (verified words often live only in cell space, 769f619d
# 'worst' class). Also fixes the surgical dual-case bug: the inline loop
# OVERWROTE the lowercase replacement with the uppercase-variant replace,
# silently skipping lowercase words for lowercase frm (det14/15 flowed-text
# surgical pass was partial). Page-level lane: raw cache must not replay.
# det17 (2026-09-18): heading emission revived — explicit _get_page_dict
# import (silent NameError killed detect_section_hierarchy +
# reconstruct_footnotes since refactor 8cbb51f4); bold pass default ON
# (2026-08-28 -3-test census was measured on the dead function = vacuous);
# two-tier numbered-pair merge ('8' + title adjacent bold lines -> one
# heading, docling numbering levels). Page-level lane, output changes
# broadly (MHS 0.102 on opendataloader-bench).
# det18 (2026-09-20): EXP-6c structured bold — PyMuPDF bold-flag
# (bit 16) carried per-line into bold_heading_lines (NimbusRomNo9L-Medi
# = LaTeX \textbf face parses as weight 500 by NAME — invisible before);
# numbered-pair/standalone promote switched to finditer (first PASSING
# occurrence; gate-failed first match no longer masks later clean ones);
# allcaps-prev accepted as heading predecessor; preserve_typography
# masks heading lines from **-wrap injection. Page-level lane.
# det19 (EXP-6c follow-up): bold headings WRAPPED across two adjacent
# PDF lines merge into ONE heading (_promote_wrapped; ducts-flyer
# 'Leaking Ducts...to' + 'Your Health' was promoted as two ##, breaking
# reading-order probes); detect_section_hierarchy removed from the OCR
# path — partial text layers there hallucinate headings on table stubs.
# det20 (EXP-8): sparse-title size fallback — when the heading cluster
# is <10% of lines the p90 statistic reads body size and goes blind;
# walk distinct sizes top-down admitting levels >=1.15x median with
# cumulative share <=15% (docling size clustering). Numbered-pair
# separator now also matches tab-joined single lines (fitz/extraction
# line-split disagreement, Brill-002 class).
# det21 (EXP-8 run-37): prev-gate + single-word vetoes — numbered-item
# predecessors ('2.3 Problem framing') and abbreviation continuations
# ('No. of Seats Aquino') no longer pass the heading-shaped exception
# (odlbench 146/51); single-word bold titles require page uniqueness
# ('Contents' survives, echoing table stubs 'Ramos' do not).
# det22 (EXP-10 run-38 bench): size-lane adjacent-line merge — display
# titles wrapped across consecutive span lines merge into ONE heading
# via _promote_multiline (9530001d 'Die Kunst der/richtigen Dichtung'
# and 075cc4e9 3-line medical section heads shattered into fragments);
# font-math cascade adoption now re-runs detect_section_hierarchy on
# the cascade text pre-reflow (145-class: adoption had replaced every
# heading promotion wholesale).
# det27 (EXP-11): region-dispatch heading survival tightened to MARKUP (E0 text-region re-extract keeps heading as plain line = demoted, odlbench 01030000000168/109); _restore_heading_markup re-marks in place so adoption keeps E4/table value
# det28 (EXP-11): TOC/list-block veto (>=3 adjacent numbered bold titles, 01030000000198); promote_allcaps Strategy-2 stands down when dsh emitted a heading (legacy ungated 'N. Title' heuristic double-promoted TOC entries); heading restore prefers LAST plain-line match (TOC echo precedes body)
# det29 (EXP-11 run-45): masthead-strip requires chrome corroboration (pattern/cluster/allcaps-journal) — bare Title-Case top-band display lines over a gapped body are page TITLES (01030000000156 'Fact-Checking'); caption-continuation prev gate (caption wraps 2 lines: pattern sits on prev2, odlbench 111 '7. THEORY')
# det30 (EXP-11 run-46): size-lane stacked-prev allowance — a size-lane title whose predecessor line is ITSELF a size-lane title passes the terminal gate (kicker-above-title display block, 01030000000200 'Introduction...'@11 kicker above 'Key Functions...'@17 title); bold lane keeps the strict gate
# det26 (EXP-11): standalone promoter full-title match (prefix-match doubled the line tail); chrome-cluster adjacency size guard (section titles > body+0.5 stay rescuable) on top of the det25
# geometry (URL/date-stamp/attribution-byline/trailing-page-number/
# 'et al.' heads, vol:issue), chrome-cluster adjacency, right-float
# letterhead + citation-block vetoes on the det23a rescue; word-gap
# aware span join (035 'BasisFields' TeX-kern glue); quote-normalized
# promoter matching (133 curly-vs-straight apostrophes).
# det31 (EXP-12 det30 bench P2F): masthead label-cluster horizontal arm (allcaps label BESIDE same-size block text, chained one level — 9530001d 'TITELTHEMA' topic block); citation-block veto on the running-head-over-references class (e71f5c59 p6); lowercase decks are never titles, every path (0b045ab8 'focus on youth')
# det42 (2026-10-02 run-51, kept): rowspan-label supplemental tables — centered-group tables (labels centered on their value groups, 9c389720 pg17 'Category|Groups') recovered via label-column clustering + value-segment assignment; chatty-list gate (fires only when no other table found); math-glyph guard.
# det43 (2026-10-02 run-52, kept): raster-embedded tables via rapidocr (3a91fad8 leave-balance, 587a1f5a affiliations) — zoom6 grayscale region render, interval-union column bands, y-band rows, 2-col pairing mode (detail-above-name); chart-axis/CJK/glyph-soup/glyph guards. NOTE: inert in bench production until router gate learns raster geometry (det45).
# det44 (2026-10-02): sepless emission — pipe tables drop the '| --- |' separator row, cells whitespace-flattened (kills phantom continuation rows), single blank line between adjacent tables/twin, html twin retained after pipes. odlbench converter requires separator to convert pipes→HTML, so pipes pass as text: TEDS forest = twin only (no duplicate-table deletions); measured on real evaluator: TEDS +0.135 / NID +0.012.
# det45 (2026-10-02): merge of the parked det32-det41 tables campaign (commit 434e7ad3, measured on odlbench det41-vs-det43: TEDS +0.107 all-19-movers-up, NID +0.017 38up/2down) on top of det44: cell ASCII fold, furniture scrub (mega-cell row drop, page-digit/Vol-No/pubnote edge arms, all-furniture veto), content-duplicate + materially-differing dedup, gated dual emission (pipes+raw twin when grids differ row-for-row), monster-row trigger, pubnote regex widened.
# det46 (2026-10-03): separator-agnostic table counting — pdf_ocr.py region-routing gates (table-protection, structure-gain, OCR pipe check) previously counted tables via the '|---|' separator row that det44 removed, so postproc table count read 0 and table-less region-routed reassembly could REPLACE text carrying the real table (run-53: 9a61fe78 20-row partner-funding table lost). _count_pipe_tables(s, min_run=2) counts maximal runs of consecutive pipe lines at all 4 gate sites.
# det47 (2026-10-04): router + FP vetoes + twin reachability — (a) layout_gate page_needs_table_chain ORs-in table-ish raster geometry (det43 filter >=80x40pt aspect 0.8-12), making det42/det43 raster strategies reachable on prose-heavy raster-table pages (3a91/587a/odlbench-122); (b) leader-page-list veto (TOC/index ['title'|int] 2-col grids, zero measured false-veto exposure); (c) img2table_lastresort prose guard (no numeric + no code-like + >=50% mid-sentence rows → skip; keeps 7e850300/c6673ff6 wins); (d) _emit_table_blocks twin fallback parses rows from pipe markdown when the table dict carries markdown only (odlbench doc-0110 TEDS 0 class).
# det48 (2026-10-04): raster-only scoping + prose-guard numeric broadening — (a) layout_gate raster_only_table_evidence(pdf,page,text): True when a table-ish raster is present AND every other chain-evidence probe is False; _detect_and_append_tables passes it as detect_all_tables(raster_only=...) which skips all text/geometry strategies (pdfplumber/lines_strict/text_only/positional/whitespace/rowspan/camelot) and runs the raster strategy as primary — closes det47's FP class where the raster OR-in opened the FULL chain on raster-only pages (odlbench 0056/0066/0098/0152 gained 3-11 junk pipes via positional/whitespace); (b) looks_like_prose_shatter treats digit-leading cells with short unit suffixes ('24 ml') as numeric — un-vetoes odlbench 0117's true measurement table.
# det50 (2026-10-06): cascade column-aware emission — extract_text_with_cascade
# final assembly sorted ALL page clusters by y0 alone, interleaving two-column
# math pages row-major (0853d8a0 pg2: 'BRM consists of 2 linear feedback guess
# the initial state...' left+right fused per visual row; hyphenated citations
# split around equation blocks). New _column_aware_emitted_order: full-width
# clusters cut bands, left column reads before right within each band; pages
# without >=3 clusters on both sides keep the plain y-sort (byte-identical).
# det51 (2026-10-06): cascade conservation audits split parts — conservation_pass
# received the ORIGINAL y-fused clusters while the det50 builder emits per split
# PART; the audit blamed a part's line for the whole fused cluster's spans,
# saw the other column's spans as missing, and re-injected them into the
# line's math (2503.04147 pg3: '\text{and \bm{z}} … N \times \times' instead of
# a clean \mathcal{N}=\{1,\cdots,N\}). Auditing the parts (check_clusters)
# restores per-column coverage accounting; det50 column wins unaffected.
# det49 (2026-10-04): equation-rows veto — detect_all_tables final filter drops tables whose cells carry operator-glyphs BETWEEN identifiers (sum/in/int/forall/Theta/Lambda/Omega or letter-op-letter at >=35% of non-empty cells). Table emission was stripping aligned equations from page text before the inline Unicode->LaTeX lane could run, losing GT LaTeX probes (arxiv census: 267 tests / 84 PDFs exposed; 03759 'H(Z)=...' entropy formula as pipes). Measured: 19/348 arxiv equation-tables killed, 1/390 tables-cat (a probe-less file); stats '0.83±0.13' and spec '1830±50' digit-attached forms survive by construction.
DETECTION_VERSION = "2026.10.12.det63"
OCR_PIPELINE_VERSION = f"{OCR_CORE_VERSION}+{DETECTION_VERSION}"
POSTPROC_VERSION = "2026.09.11.pp2"
SKILL_EXTRACTION_VERSION = f"{OCR_PIPELINE_VERSION}+{POSTPROC_VERSION}"

# From: pdf_ocr.py
SUBSCRIPT_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

# From: pdf_ocr.py
SUPERSCRIPT_DIGITS = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")

# From: pdf_ocr.py
TABLE_GAP_RE = re.compile(r"\s{3,}")

# From: pdf_ocr.py
TEXT_LAYER_CHARS_PER_PAGE_MIN = 100

# From: scientific_parser/postproc/text_pipeline.py
UNICODE_MATH_CHARS = frozenset(
    chr(c)
    for c in list(range(0x2070, 0x209F + 1))  # super/subscripts: ²³ⁿ⁻⁺₀₁₂...
    + list(range(0x2200, 0x22FF + 1))  # math operators: ∀∂∃∇∑∏∫√∝∞∠∧∨∩∪
    + list(range(0x2190, 0x21FF + 1))  # arrows: →←↔⇒⇐⇔
    + list(range(0x2100, 0x214F + 1))  # letterlike: ℕℤℚℝℂℵℏ
    + [
        0x221E,
        0x2202,
        0x2206,
        0x2207,
        0x2208,
        0x2209,
        0x220B,
        0x2211,
        0x221A,
        0x221E,
        0x222B,
        0x222C,
        0x2229,
        0x222A,
        0x2227,
        0x2228,
        0x2264,
        0x2265,
        0x2260,
        0x2248,
        0x00B1,
        0x00D7,
        0x00F7,
        0x223C,
        0x2243,
        0x2282,
        0x2283,
        0x2286,
        0x2287,
        0x2295,
        0x2297,
        0x22A2,
        0x22A3,
        0x22A4,
        0x22A5,
        # Greek letters commonly used in math (lowercase)
        0x03B1,
        0x03B2,
        0x03B3,
        0x03B4,
        0x03B5,
        0x03B6,
        0x03B7,
        0x03B8,
        0x03B9,
        0x03BA,
        0x03BB,
        0x03BC,
        0x03BD,
        0x03BE,
        0x03BF,
        0x03C0,
        0x03C1,
        0x03C3,
        0x03C4,
        0x03C5,
        0x03C6,
        0x03C7,
        0x03C8,
        0x03C9,
        # Greek uppercase
        0x0393,
        0x0394,
        0x0398,
        0x039B,
        0x039E,
        0x03A0,
        0x03A3,
        0x03A6,
        0x03A8,
        0x03A9,
    ]
)

# From: scientific_parser/perception/equations/math_extractor.py
UNICODE_TO_LATEX = {
    # Big operators (CMEX)
    "∑": r"\sum",
    "∑︂": r"\sum",
    "∏": r"\prod",
    "∏︂": r"\prod",
    "∫": r"\int",
    "∬": r"\iint",
    "∭": r"\iiint",
    "∮": r"\oint",
    "⋃": r"\bigcup",
    "⋂": r"\bigcap",
    "⨁": r"\bigoplus",
    "⨂": r"\bigotimes",
    "∐": r"\coprod",  # coproduct (n-ary operator)
    "—": "---",  # em-dash (LaTeX convention)
    "–": "--",  # en-dash
    # Binary operators
    "−": "-",
    "·": r"\cdot",
    "×": r"\times",
    "÷": r"\div",
    "±": r"\pm",
    "∓": r"\mp",
    "∧": r"\wedge",
    "∨": r"\vee",
    "⊕": r"\oplus",
    "⊖": r"\ominus",
    "⊗": r"\otimes",
    "⊙": r"\odot",
    # Relations
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≈": r"\approx",
    "≡": r"\equiv",
    "∼": r"\sim",
    "≅": r"\cong",
    "∝": r"\propto",
    "⊂": r"\subset",
    "⊃": r"\supset",
    "⊆": r"\subseteq",
    "⊇": r"\supseteq",
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    "∖": r"\setminus",
    # Arrows
    "→": r"\to",
    "←": r"\leftarrow",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto",
    "↑": r"\uparrow",
    "↓": r"\downarrow",
    # Dots/ellipsis
    "…": r"\dots",
    "⋯": r"\cdots",
    "⋮": r"\vdots",
    "⋱": r"\ddots",
    # Greek (common in math)
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",
    "σ": r"\sigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    # Accents/decorations
    "∇": r"\nabla",
    "∂": r"\partial",
    "∞": r"\infty",
    "√": r"\sqrt",
    "∀": r"\forall",
    "∃": r"\exists",
    "ℏ": r"\hbar",
    "ℓ": r"\ell",
    "ℜ": r"\Re",
    "ℑ": r"\Im",
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℙ": r"\mathbb{P}",
    # Brackets (stretchy)
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "〈": r"\langle",
    "〉": r"\rangle",  # alt angle brackets (CJK)
    "⟦": r"\llbracket",
    "⟧": r"\rrbracket",  # double brackets
    "‖": r"\|",
    "​": "",
}

# From: pdf_ocr.py
_BULLET_RE = re.compile(r"^\s*[-\u2022*]\s+")  # - item, • item

# From: pdf_ocr.py
_CODE_RE = re.compile(r"^\s{4,}\S")  # indented code line

# From: pdf_ocr.py
_DOI_RE = re.compile(r"doi:\s|https?://", re.IGNORECASE)

# From: pdf_ocr.py
_FIGURE_TABLE_OCR = False

# From: scientific_parser/postproc/text_pipeline.py
_GEOLOGY_DICT_TERMS = {
    # Mineral names (high-frequency IMA list)
    "quartz",
    "feldspar",
    "plagioclase",
    "alkali",
    "albite",
    "anorthite",
    "orthoclase",
    "microcline",
    "sanidine",
    "olivine",
    "forsterite",
    "fayalite",
    "pyroxene",
    "clinopyroxene",
    "orthopyroxene",
    "enstatite",
    "ferrosilite",
    "diopside",
    "hedenbergite",
    "augite",
    "pigeonite",
    "spodumene",
    "jadeite",
    "amphibole",
    "hornblende",
    "tremolite",
    "actinolite",
    "glaucophane",
    "riebeckite",
    "crocidolite",
    "anthophyllite",
    "cumingtonite",
    "gedrite",
    "mica",
    "muscovite",
    "biotite",
    "phlogopite",
    "lepidolite",
    "paragonite",
    "chlorite",
    "clinochlore",
    "chamosite",
    "serpentine",
    "antigorite",
    "lizardite",
    "chrysotile",
    "talc",
    "pyrophyllite",
    "kaolinite",
    "illite",
    "smectite",
    "montmorillonite",
    "vermiculite",
    "garnet",
    "almandine",
    "pyrope",
    "spessartine",
    "grossular",
    "andradite",
    "uvarovite",
    "epidote",
    "zoisite",
    "clinozoisite",
    "piedmontite",
    "allanite",
    "sillimanite",
    "kyanite",
    "andalusite",
    "topaz",
    "staurolite",
    "chloritoid",
    "cordierite",
    "spinel",
    "magnetite",
    "hercynite",
    "chromite",
    "titanite",
    "sphene",
    "perovskite",
    "zircon",
    "monazite",
    "xenotime",
    "apatite",
    "fluorapatite",
    "chlorapatite",
    "hydroxylapatite",
    "calcite",
    "dolomite",
    "magnesite",
    "siderite",
    "rhodochrosite",
    "smithsonite",
    "aragonite",
    "strontianite",
    "witherite",
    "barite",
    "celestine",
    "gypsum",
    "anhydrite",
    "halite",
    "sylvite",
    "fluorite",
    "pyrite",
    "pyrrhotite",
    "chalcopyrite",
    "bornite",
    "chalcocite",
    "covellite",
    "galena",
    "sphalerite",
    "wurtzite",
    "cinnabar",
    "realgar",
    "orpiment",
    "stibnite",
    "bismuthinite",
    "molybdenite",
    "pentlandite",
    "violarite",
    "millerite",
    "niccolite",
    "skutterudite",
    "smaltite",
    "cobaltite",
    "arsenopyrite",
    "marcasite",
    "loellingite",
    "safflorite",
    "rutile",
    "anatase",
    "brookite",
    "ilmenite",
    "hematite",
    "goethite",
    "limonite",
    "lepidocrocite",
    "maghemite",
    "corundum",
    "cassiterite",
    "cuprite",
    "tenorite",
    "zincite",
    "brucite",
    "gibbsite",
    "boehmite",
    "diaspore",
    "manganite",
    "psilomelane",
    "pyrolusite",
    "cryptomelane",
    "hollandite",
    "romanechite",
    "todorokite",
    "birnessite",
    # Rock names
    "granite",
    "granodiorite",
    "tonalite",
    "diorite",
    "gabbro",
    "norite",
    "anorthosite",
    "peridotite",
    "dunite",
    "harzburgite",
    "lherzolite",
    "wehrlite",
    "pyroxenite",
    "komatiite",
    "basalt",
    "andesite",
    "dacite",
    "rhyolite",
    "trachyte",
    "phonolite",
    "pumice",
    "obsidian",
    "tuff",
    "ignimbrite",
    "pegmatite",
    "aplite",
    "porphyry",
    "sandstone",
    "shale",
    "limestone",
    "dolostone",
    "mudstone",
    "siltstone",
    "conglomerate",
    "breccia",
    "chert",
    "flint",
    "evaporite",
    "schist",
    "gneiss",
    "slate",
    "phyllite",
    "quartzite",
    "marble",
    "skarn",
    "hornfels",
    "mylonite",
    "migmatite",
    "eclogite",
    "blueschist",
    "greenschist",
    "amphibolite",
    "granulite",
    "serpentinite",
    "soapstone",
    # Geochemistry terms
    "geochemistry",
    "geothermometry",
    "geobarometry",
    "thermobarometry",
    "partitioning",
    "coefficient",
    "distribution",
    "fractionation",
    "fractional",
    "crystallization",
    "differentiation",
    "assimilation",
    "contamination",
    "weathering",
    "erosion",
    "deposition",
    "diagenesis",
    "lithification",
    "metamorphism",
    "metasomatism",
    "alteration",
    "hydrothermal",
    "metasomatic",
    "magmatic",
    "volcanic",
    "plutonic",
    "igneous",
    "aqueous",
    "fluid",
    "melt",
    "solution",
    "solubility",
    "saturation",
    "undersaturated",
    "saturated",
    "oversaturated",
    "volatile",
    "volatiles",
    "incompatible",
    "compatible",
    "refractory",
    "lithophile",
    "siderophile",
    "chalcophile",
    "atmophile",
    "rare",
    "earth",
    "elements",
    "trace",
    "isotope",
    "isotopes",
    "isotopic",
    "radiogenic",
    "decay",
    "radioactive",
    "geochronology",
    "dating",
    "cosmogenic",
    "nuclide",
    "concentration",
    "abundance",
    "enrichment",
    "depletion",
    "anomaly",
    "anomalies",
    "anomalous",
    "normalized",
    "chondrite",
    "primitive",
    "mantle",
    "depleted",
    "enriched",
    "lithosphere",
    "asthenosphere",
    "mesosphere",
    "crust",
    "core",
    "conduction",
    "convection",
    "advection",
    "diffusion",
    "percolation",
    "infiltration",
    "reaction",
    "transport",
    "kinetics",
    "thermodynamics",
    "equilibrium",
    "disequilibrium",
    "stoichiometry",
    "stoichiometric",
    "valence",
    "oxidation",
    "reduction",
    "redox",
    "buffer",
    "fugacity",
    "oxygen",
    "sulfur",
    "activity",
    "chemical",
    "potential",
    "composition",
    "compositional",
    "oxide",
    "mole",
    # Process / geology terms
    "petrology",
    "petrogenesis",
    "petrogenetic",
    "tectonic",
    "tectonics",
    "subduction",
    "collision",
    "rifting",
    "extension",
    "compression",
    "uplift",
    "subsidence",
    "deformation",
    "strain",
    "stress",
    "fault",
    "fracture",
    "joint",
    "fold",
    "foliation",
    "lineation",
    "fabric",
    "shear",
    "kinematic",
    "dynamic",
    "rheology",
    "viscosity",
    "plasticity",
    "elasticity",
    "brittle",
    "ductile",
    "friction",
    "pressure",
    "temperature",
    "depth",
    "lithology",
    "stratigraphy",
    "stratigraphic",
    "sedimentology",
    "depositional",
    "facies",
    "sequence",
    "formation",
    "unconformity",
    "correlation",
    # Common institutions / publishing
    "University",
    "Institute",
    "Research",
    "Department",
    "Laboratory",
    "School",
    "Sciences",
    "Science",
    "Technology",
    "Engineering",
    "Physics",
    "Chemistry",
    "Biology",
    "Geology",
    "Geophysics",
    "Geochemistry",
    "Mineralogy",
    "Petrology",
    "Earth",
    "Planetary",
    "Environmental",
    "Climate",
    "Atmospheric",
    "Oceanic",
    "Received",
    "Accepted",
    "Revised",
    "Abstract",
    "Introduction",
    "Methods",
    "Methodology",
    "Results",
    "Discussion",
    "Conclusion",
    "Conclusions",
    "Acknowledgements",
    "References",
    "Bibliography",
    "Appendix",
    "Figure",
    "Table",
    "Equation",
    # Common German / umlaut terms (frequent OCR error sources)
    "Zürich",
    "für",
    "Tatsumi",
    "Mössbauer",
    "Mößbauer",
    "Düsseldorf",
    "München",
    "Häfner",
    "Lüders",
    "Müller",
    "Schmidt",
    "Weber",
    "Meyer",
    "Wagner",
    "Becker",
    "Schneider",
    "Hoffmann",
    "Fischer",
    "Walter",
}

# From: scientific_parser/routing/page_classifier.py
_LATE = frozenset(
    {
        "IS_IMAGE",
        "LOW_TEXT_DENSITY_THRESHOLD",
        "ROUTER_MIN_CHARS",
        "ROUTER_ALPHANUM_MIN",
        "ROUTER_SPACE_RATIO_MAX",
        "ROUTER_NEWLINE_RATIO_MAX",
        "ROUTER_INVALID_CHAR_THRESHOLD",
        "ROUTER_MULTI_SPACE_RUN_MAX",
        "ROUTER_SHORT_LINE_THRESHOLD",
        "ROUTER_MIN_AVG_WPL",
        "TEXT_LAYER_CHARS_PER_PAGE_MIN",
        "_PRIVATE_CHAR_RE",
        "MATH_FONT_PREFIXES",
    }
)

# From: pdf_ocr.py

# From: scientific_parser/perception/equations/math_extractor.py
_OCR_LATEX_CMD_CONFUSIONS = [
    (re.compile(r"\\sort\b"), r"\\sqrt"),
    (re.compile(r"\\sqr\b"), r"\\sqrt"),
    # \carpi — glm-ocr garble of variable "z" (substitution z^{3/2} context,
    # old_scans_math bench 2026-08-14: "\carpi^3 = x^2" beside "x = z^{3/2}")
    (re.compile(r"\\carpi\b"), r"z"),
    (re.compile(r"\\fracc\b"), r"\\frac"),
    (re.compile(r"\\intt\b"), r"\\int"),
    (re.compile(r"\\textb\b"), r"\\textbf"),
    (re.compile(r"\\alp\b"), r"\\alpha"),
    (re.compile(r"\\bet\b"), r"\\beta"),
    (re.compile(r"\\lam\b"), r"\\lambda"),
    # Multi-letter garbles from arxiv_math OCR
    (re.compile(r"\\nifty\b"), r"\\infty"),
    (re.compile(r"\\inftyw"), r"\\infty w"),
    (re.compile(r"\\inftyd"), r"\\infty d"),
    (re.compile(r"\\pirT\b"), r"\\pi r T"),
    (re.compile(r"\\betai\b"), r"\\beta_i"),
    (re.compile(r"\\leqg\b"), r"\\leq g"),
    (re.compile(r"\\Sort\b"), r"\\sqrt"),
    (re.compile(r"\\Fracc\b"), r"\\frac"),
    # Space-split commands from old_scans_math OCR (glm-ocr artifacts)
    (re.compile(r"\\in\s+t(?=[{\\])"), r"\\int"),
    (re.compile(r"\\in\s+T(?=[{\\])"), r"\\int"),
    (re.compile(r"\\lo\sg(?=[({=\\])"), r"\\log"),
    (re.compile(r"\\co\ss(?=[({=\\])"), r"\\cos"),
    (re.compile(r"\\si\s+n(?=[({=\\])"), r"\\sin"),
    (re.compile(r"\\ta\s+n(?=[({=\\])"), r"\\tan"),
    (re.compile(r"\\operatorname\{\\sin\^\{-1\}"), r"\\sin^{-1"),
    (re.compile(r"\\operatorname\{\\cos\^\{-1\}"), r"\\cos^{-1"),
    (re.compile(r"\\operatorname\{\\tan\^\{-1\}"), r"\\tan^{-1"),
]

# From: pdf_ocr.py
_PRIVATE_CHAR_RE = re.compile(r"[\x00-\x08\x0e-\x1f\ufffe\uffff]")

# From: pdf_ocr.py
_RAPIDOCR_INSTANCE = None

# From: scientific_parser/ingest/ocr_engine.py
_RAPIDOCR_MODEL_TYPE = None

# From: pdf_ocr.py
_REFLINE_RE = re.compile(r"^\s*\[\d+\]")  # [1] Author et al.

# From: pdf_ocr.py
_SYMSPELL_INSTANCE = None

# From: pdf_ocr.py
_page_dict_cache = {}

__all__ = [
    "ALLCAPS_HEADING_RE",
    "BIBLIOGRAPHY_MARKERS",
    "BRACKET_HEADER_RE",
    "CITATION_SPLIT_RE",
    "CRYSTALLOGRAPHIC_RE",
    "CRYSTAL_INDICES_RE",
    "DEFAULT_CACHE_DIR",
    "DEFAULT_MODEL_PROFILE",
    "DEPS",
    "DOI_RE",
    "DPI",
    "DUPLICATE_CHAR_MIN_LEN",
    "FORMULA_RE",
    "HEADER_LINE_MAX_CHARS",
    "LOW_TEXT_DENSITY_THRESHOLD",
    "MATH_FONT_PREFIXES",
    "MATH_OCR_PROMPT",
    "MATH_OPERATOR_CHARS",
    "MAX_DIM",
    "MINERAL_ABBREVS",
    "MIN_SPAN_CHARS",
    "MODEL_PROFILES",
    "OCR_PER_ATTEMPT_TIMEOUT_S",
    "OLLAMA_URL",
    "OXIDE_FORMULAS",
    "OXIDE_MW",
    "PAGE_NUMBER_RE",
    "PRESERVE_HYPHENS",
    "REE_OCR_CONFUSIONS",
    "REE_SEQUENCE",
    "REFLEXIVE_CONFIDENCE_FLOOR",
    "REFLEXIVE_MAX_ATTEMPTS_PER_MODEL",
    "REPEAT_LINE_MIN_CHARS",
    "ROUTER_ALPHANUM_MIN",
    "ROUTER_IMAGE_COVERAGE_THRESHOLD",
    "ROUTER_INVALID_CHAR_ABS",
    "ROUTER_INVALID_CHAR_PCT",
    "ROUTER_MIN_AVG_WPL",
    "ROUTER_MIN_CHARS",
    "ROUTER_MULTI_SPACE_RUN_MAX",
    "ROUTER_NEWLINE_RATIO_MAX",
    "ROUTER_SHORT_LINE_MIN_CHARS",
    "ROUTER_SHORT_LINE_RATIO_MAX",
    "ROUTER_SPACE_RATIO_MAX",
    "SCIENTIFIC_SYMBOL_RULES",
    "SINGLE_ELEM_ZERO_RE",
    "OCR_PIPELINE_VERSION",
    "OCR_CORE_VERSION",
    "DETECTION_VERSION",
    "POSTPROC_VERSION",
    "SKILL_EXTRACTION_VERSION",
    "SUBSCRIPT_DIGITS",
    "SUPERSCRIPT_DIGITS",
    "TABLE_GAP_RE",
    "TEXT_LAYER_CHARS_PER_PAGE_MIN",
    "UNICODE_MATH_CHARS",
    "UNICODE_TO_LATEX",
    "_BULLET_RE",
    "_CODE_RE",
    "_DOI_RE",
    "_FIGURE_TABLE_OCR",
    "_GEOLOGY_DICT_TERMS",
    "_LATE",
    "_OCR_LATEX_CMD_CONFUSIONS",
    "_PRIVATE_CHAR_RE",
    "_RAPIDOCR_INSTANCE",
    "_RAPIDOCR_MODEL_TYPE",
    "_REFLINE_RE",
    "_SYMSPELL_INSTANCE",
    "__all__",
    "_page_dict_cache",
    "mask_math_regions",
    "unmask_math_regions",
    "MATH_MASK_TOKEN",
    "_is_math_symbol",
    "_is_numeric_cell",
]


# === PDF backend + math-symbol helper (single source of truth) ======== #
# These were originally defined in pdf_ocr.py monolith and accessed by
# scientific_parser modules via lazy binding — which silently broke when
# the monolith's globals were late-binding (see 2026-08-13 multi_column
# NameError fix). Moving them here so every consumer imports the SAME
# instance at module load time. See shared.py docstring: "single source
# of truth. Imported by pdf_ocr.py AND ALL extracted modules."

import unicodedata as _unicodedata_shared  # noqa: E402

# PERF-1 (2026-09-16): pypdfium2 costs ~85ms of import time and is only
# needed on extraction runs (invisible-text probe, script fallback) — not
# on --help / cache-hit replays. Deferred via module __getattr__ (PEP 562);
# NOT in __all__ so wildcard imports cannot eagerly trigger the load.
# page_classifier late-binds via its _load_consts registry; font_analysis
# accesses shared._pdfium at call time. Missing package still fails loud
# at first use (declared dep).


def __getattr__(name):
    if name in ("_pdfium", "_pdfium_c"):
        import pypdfium2
        import pypdfium2.raw

        globals()["_pdfium"] = pypdfium2
        globals()["_pdfium_c"] = pypdfium2.raw
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_math_symbol(char: str) -> bool:
    """True iff ``char`` is a single Unicode math symbol (category ``Sm``).

    Single source of truth. Replaces the monolith's ``_is_math_symbol``
    (pdf_ocr.py pre-2026-08-13) and the lazy-binding facade that lived
    briefly in ``font_analysis.py``.

    Args:
        char: String of any length; only length-1 inputs can return True.

    Returns:
        True iff ``char`` is exactly one Unicode math symbol.

    Reference:
        Unicode Standard, category ``Sm`` (Symbol, math).
    """
    if not isinstance(char, str) or len(char) != 1:
        return False
    try:
        return _unicodedata_shared.category(char) == "Sm"
    except ValueError:
        return False


def _is_numeric_cell(s) -> bool:
    """True if cell is numeric (decimal, integer, or 'n.d.'/'-' placeholder).

    Single source of truth since 2026-08-13. Used by table detector and
    multirow-header detector to count numeric vs text cells.

    Recognizes:
      - Empty / whitespace
      - Common placeholder tokens: ``-``, ``n.d.``, ``nd``, ``b.d.l.``,
        ``—``, ``–``, ``n/a``, ``N/A``
      - Integers and decimals (after stripping ``,`` and parens)

    Args:
        s: Cell value (string or anything convertible to string).

    Returns:
        True iff the cell is numeric or a recognized placeholder.
    """
    if not s:
        return True
    s = str(s).strip()
    if s in ("-", "n.d.", "nd", "b.d.l.", "—", "–", "n/a", "N/A"):
        return True
    try:
        float(s.replace(",", "").replace("(", "").replace(")", ""))
        return True
    except (ValueError, AttributeError):
        return False


# ── Math-region masking (shared by postproc + math_extractor) ──────────────
# Guards span/region replacement sites from inserting markers INSIDE
# existing $...$ math (measured corruption: '_{$k,m $}' broke delimiter
# pairing and downstream cleaners dropped '\mathbb{C}^{N}' — flip-diff
# 2503.06703_pg4, 2026-08-16).
MATH_MASK_TOKEN = "\ue00d"


def mask_math_regions(text: str) -> tuple[str, list[str]]:
    """Replace $...$ and $$...$$ regions with single-char sentinels.

    Returns (masked_text, saved_regions) — restore with unmask_math_regions.
    """
    saved: list[str] = []
    masked = re.sub(
        r"\$\$.+?\$\$|\$[^$\n]+?\$",
        lambda m: (saved.append(m.group(0)), MATH_MASK_TOKEN)[1],
        text,
        flags=re.DOTALL,
    )
    return masked, saved


def unmask_math_regions(masked: str, saved: list[str]) -> str:
    it = iter(saved)
    return re.sub(re.escape(MATH_MASK_TOKEN), lambda m: next(it), masked)
