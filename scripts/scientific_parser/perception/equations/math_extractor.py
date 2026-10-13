"""Math equation extraction — font-aware + Unicode + LaTeX reconstruction.

Extracted from scripts/pdf_ocr.py during Phase R3 refactoring (2026-08-12).
Pure extraction — no behavioral changes. All names re-exported by pdf_ocr.py.

Functions:
  - extract_math_regions: detect math spans via CMMI/CMSY/CMEX fonts + Unicode chars
  - reconstruct_math_region: convert span list to LaTeX
  - replace_math_in_text: walk PDF text + replace math regions inline
  - improve_math_regions: OCR-improve private-char regions
  - detect_display_equations: PyMuPDF block-based display equation detection
  - ocr_display_equations: OCR detected equation bands with math prompt
  - wrap_equations_as_latex: wrap detected math with $ delimiters

Constants:
  - MATH_FONT_PREFIXES: tuple of font-family prefixes that signal math content
  - UNICODE_TO_LATEX: dict mapping Unicode chars to LaTeX equivalents
  - _OCR_LATEX_CMD_CONFUSIONS: list of (regex, replacement) for OCR error repair
"""

from __future__ import annotations

import os
import re
import logging

log = logging.getLogger(__name__)
from shared import *  # ALL shared constants, patterns, state


# Late-import shims for cross-module dependencies (pdf_ocr.py).
# These resolve at call-time, avoiding circular import.
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
    own = set(g.keys())
    for k, v in vars(p).items():
        if k not in own and not k.startswith("__"):
            g[k] = v
    _CONSTS_LOADED = True


def detect_display_equations(pdf_path, page_num):
    """Detect display equation regions on a page.

    Returns list of (y0, y1) tuples representing equation vertical bands.
    Used to route those regions through math-aware OCR prompt.
    """
    _load_consts()
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return []
    try:
        page_rect = pd["rect"]
        page_w = page_rect.width
        mid_x = page_w / 2

        equation_bands = []

        for b in pd["dict"]["blocks"]:
            if b.get("type", 0) != 0:
                continue
            bbox = b["bbox"]
            block_cx = (bbox[0] + bbox[2]) / 2
            block_w = bbox[2] - bbox[0]
            block_h = bbox[3] - bbox[1]

            if abs(block_cx - mid_x) > page_w * 0.15:
                continue
            if block_h > 60:
                continue

            block_text = ""
            for line in b.get("lines", []):
                for span in line.get("spans", []):
                    block_text += span.get("text", "")

            has_math = any(c in MATH_OPERATOR_CHARS for c in block_text)
            if not has_math and "=" not in block_text:
                continue
            if block_w > page_w * 0.7:
                continue

            equation_bands.append((bbox[1], bbox[3], block_text.strip()))

        return equation_bands
    except Exception as e:
        log.warning("equation detection failed: %s", e)
        return []


# Patterns that are NOT real math — reference lists, numbered items, pure
# numbers/citations that the geometry heuristic false-positives as equations.
_REF_NUM_RE = re.compile(
    r"^[\s(]*\d+[.)\]]"
    r"(\s*\d+[.)\]]?)*"
    r"[\s).,]*$"
)
_REF_PREFIX_RE = re.compile(r"^[\s(]*\d+[.)\]]\s+\d+[.)\]]")
_LATEX_CMD_RE = re.compile(r"\\[a-zA-Z]+\{?")
_SUBSUP_RE = re.compile(r"[a-zA-Z0-9)\}][\^_]\{?")

# Non-math content patterns (2026-08-14 bench debug):
# - English words inside math delimiters (\mathrm{English words})
# - Accession IDs (D_Q453581), dates (06/03), URLs, "Job ID"
# - The `/` in dates/paths was matching the division operator check
_ENGLISH_WORD_RE = re.compile(
    r"\b(the|and|for|are|with|case|this|have|from|that|not|but|was|were|"
    r"been|will|would|could|should|may|might|can|all|any|some|such|also|"
    r"than|then|when|where|which|while|both|each|into|over|under|after|"
    r"before|between|through|during|without|within|against|around|above|"
    r"below|upon|job|http|https|www|fig|figure|table|example|section|"
    r"chapter|page|note|footnote|reference|ref|eq|equation)\b",
    re.IGNORECASE,
)
# Date-like patterns: 06/03, 12/25/2024 (NOT division)
_DATE_RE = re.compile(r"\d{1,4}[/-]\d{1,4}([/-]\d{2,4})?")
# Accession-ID pattern: Letter_Q/dash + digits (D_Q453581, AB-12345)
_ACCESSION_RE = re.compile(r"[A-Z][_-][A-Z]?\d{4,}")
# Greek letters / math symbols (strong math signal)
_GREEK_RE = re.compile(r"[αβγδεζηθικλμνξπρστυφχψωΔΘΛΞΠΣΦΨΩ]")
# Unicode mathematical operators (∫∑∏∂∇√∞∝∮ etc.) — strong math signal.
# Excludes arrows (→↔⇒⇔) which appear in non-math text ("A → B", "°C → K").
_MATH_OP_RE = re.compile(
    r"[∫∮∑∏∂∇√∞∝∠∡∢∧∨∩∪⊕⊗⊙∀∃∈∋∉⊂⊃⊆⊇≪≫⊕⊖⊗⊘"
    r"±∓×÷⋅⋆∘∙∥⊥∴∵ℂℕℚℝℤℍℙℏℓℱℒ]"
)


def is_real_math(text: str) -> bool:
    """Return True only if the text is genuinely mathematical.

    Rejects: reference lists, numbered items, pure-number bands, English prose
    wrapped in \\mathrm{}, accession IDs (D_Q453581), dates (06/03/2024),
    URLs, and other non-math content that mimics math notation.

    Requires ≥1 strong math signal: LaTeX command (with real math content, not
    just English words in dressing), subscript/superscript (in math context),
    Greek letters, or math operators between variable operands.
    """
    t = (text or "").strip()
    if not t or len(t) < 3:
        return False
    # Reject pure reference-number bands.
    if _REF_NUM_RE.match(t):
        return False
    # Reject reference-LIST items: "(3) 19) ..."
    if _REF_PREFIX_RE.match(t):
        return False
    # Reject accession IDs: "D_Q453581 99 0 2"
    if _ACCESSION_RE.search(t):
        return False
    # Reject content with common English words (not math).
    # Catches "\mathrm{(dinate~Superalgebras~ol}" and "Job ID:382287".
    if _ENGLISH_WORD_RE.search(t):
        return False

    # ── Strong signals (any one suffices) ────────────────────────────── #

    # Signal 1: Greek letters — unambiguous math.
    if _GREEK_RE.search(t):
        return True

    # Signal 1b: Unicode math operators (∫∑∏∂∇√∞∝ etc.) — unambiguous math.
    if _MATH_OP_RE.search(t):
        return True

    # Signal 2: LaTeX command with real math content (not English words).
    if _LATEX_CMD_RE.search(t):
        # Strip ALL LaTeX commands + dressing to see what remains.
        stripped = re.sub(r"\\[a-zA-Z]+", "", t)
        stripped = re.sub(r"[\{\}\(\)\[\]\s\d.,;:~]", "", stripped)
        if not stripped:
            return False
        # Reject if stripped content is English words (5+ consecutive
        # letters) without any Greek or math symbols.
        long_words = re.findall(r"[a-zA-Z]{5,}", stripped)
        if long_words and not _GREEK_RE.search(stripped):
            return False
        return True

    # Signal 3: Subscript/superscript in math context.
    # Reject when the only subscript is followed by pure numbers (accession
    # IDs like "D_Q453581"). Require a variable or operator nearby.
    if _SUBSUP_RE.search(t):
        # Check there's a math variable (single letter) or operator nearby
        # the subscript, not just digits.
        has_var = bool(re.search(r"[a-zA-Z]\s*[=+\-±]", t))
        has_greek = bool(_GREEK_RE.search(t))
        has_latex = bool(_LATEX_CMD_RE.search(t))
        if has_var or has_greek or has_latex:
            return True
        # Subscript followed by only digits is suspicious — check if it
        # looks like a data field (number number number pattern).
        if re.match(r"^[\w\s._/\-]*\d+\s+\d+", t):
            return False
        return True

    # Signal 4: Math operators between variable operands (not dates/paths).
    # `=` between an operand (letter/digit/closing-bracket) and something.
    if re.search(r"[a-zA-Z0-9)\]]\s*[=≈≠≤≥]\s*[a-zA-Z0-9(]", t):
        return True
    # `+` or `-` between letter operands: "a+b", "x-y"
    if re.search(r"[a-zA-Z]\s*[+\-]\s*[a-zA-Z]", t):
        return True
    # `±`, `×`, `÷` are unambiguous math operators.
    if re.search(r"[±×÷]", t) and re.search(r"[a-zA-Z\d]", t):
        return True
    # Do NOT count `/` alone as math operator (dates, paths, URLs use /).
    return False


def wrap_equations_as_latex(pdf_path, page_num, text):
    """Wrap detected display equations in LaTeX delimiters ($$...$$).

    Uses geometry detection (centered blocks with math operators) to find
    display equations, then wraps their text in $$ delimiters for downstream
    LaTeX rendering (MathJax, KaTeX, MathML conversion). VERIFIES each band is
    real math before wrapping — rejects reference/numbered-list false-positives.
    """
    eq_data = detect_display_equations(pdf_path, page_num)
    if not eq_data:
        return text

    for y0, y1, eq_text in eq_data:
        if not eq_text or len(eq_text) < 3:
            continue
        # VERIFICATION GATE: only wrap genuine math. Rejects garbage like
        # "$$(3) 12)$$" (reference numbers mis-detected as equations).
        if not is_real_math(eq_text):
            continue
        # Find the equation text in the output
        escaped = re.escape(eq_text[:40])
        match = re.search(escaped, text)
        if match:
            pos = match.start()
            # Check not already wrapped
            before = text[max(0, pos - 3) : pos]
            if "$$" in before:
                continue
            # Wrap in $$ delimiters
            text = text[:pos] + f"$${eq_text}$$" + text[match.end() :]

    return text


# =============================================================================
# P1-5: Typography preservation (italic, bold, monospace, superscript)
# =============================================================================


def ocr_display_equations(pdf_path, page_num, tmp_dir, model="glm-ocr"):
    """Detect and OCR display equation regions with math-aware prompt.

    For each detected equation band:
    1. Crop the page image to the equation region
    2. OCR with MATH_OCR_PROMPT (LaTeX-aware)
    3. Return list of (y_position, latex_text) pairs

    Used in OCR path to get better equation rendering than generic OCR.
    """
    if not DEPS.get("pdftoppm", False):
        return []

    eq_bands = detect_display_equations(pdf_path, page_num)
    if not eq_bands:
        return []

    import fitz

    results = []
    formula_specialist = None  # RapidLaTeX specialist (resolved lazily on first eq)
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        page_rect = page.rect

        # Render full page at current DPI for cropping
        prefix = os.path.join(tmp_dir, "eq_page")
        full_img = pdf_page_to_png(pdf_path, page_num, prefix)
        full_img = resize_image(full_img, MAX_DIM)

        from PIL import Image

        base_img = Image.open(full_img)
        img_w, img_h = base_img.size
        scale_x = img_w / page_rect.width
        scale_y = img_h / page_rect.height

        for i, (y0, y1) in enumerate(eq_bands):
            # Add padding
            pad = 10
            crop_y0 = max(0, int(y0 * scale_y) - pad)
            crop_y1 = min(img_h, int(y1 * scale_y) + pad)
            crop_x0 = 0
            crop_x1 = img_w

            if crop_y1 - crop_y0 < 10:
                continue

            # Crop and save
            cropped = base_img.crop((crop_x0, crop_y0, crop_x1, crop_y1))
            eq_path = os.path.join(tmp_dir, f"eq_{page_num}_{i}.png")
            cropped.save(eq_path)

            # ROOT-CAUSE FIX (2026-08-13): use the equation SPECIALIST
            # (RapidLaTeX) as PRIMARY — not the generic text OCR. Previously
            # this called ocr_image(model=models[0]) which defaulted to
            # rapidocr (a TEXT engine) and mangled equations, even though the
            # RapidLaTeX specialist was installed. The generic OCR is now the
            # FALLBACK only (used if the specialist returns nothing).
            latex = ""
            try:
                if formula_specialist is None:
                    from scientific_parser.perception.providers import (
                        get_formula_recognizer,
                    )

                    formula_specialist = get_formula_recognizer("default")
                if formula_specialist is not None and formula_specialist.available():
                    latex, _conf = formula_specialist.recognize(eq_path)
                    latex = (latex or "").strip()
            except Exception as fe:
                print(f"  RapidLaTeX specialist failed, falling back: {fe}")
            if not latex:
                latex = ocr_image(eq_path, task=MATH_OCR_PROMPT, model=model)
            if latex.strip():
                results.append((y0, latex.strip()))
                print(f"  Equation OCR at y={y0:.0f}: {len(latex)} chars")

        doc.close()
        return results
    except Exception as e:
        print(f"  Equation OCR failed: {e}")
        return []


# =============================================================================
# P1-7: Caption association (figures ↔ captions)
# =============================================================================


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

# Unicode math symbol → LaTeX command mapping.
# Covers: big operators, binary ops, arrows, relations, set theory, calculus.
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


def _is_math_font(font_name):
    """True if font is from a math font family (CMMI, CMSY, CMEX, etc.)."""
    if not font_name:
        return False
    name = font_name.split("-")[0].split("/")[0]
    return any(name.startswith(prefix) for prefix in MATH_FONT_PREFIXES)


# Unicode math chars that signal "this span is math content" even when the
# font is regular (CMR, Arial, Calibri). Born-digital PDFs from non-LaTeX
# sources (Word, LibreOffice, InDesign) often embed math as Unicode chars
# in body fonts rather than using dedicated math fonts.
#
# Sources: Unicode mathematical operators block (U+2200-22FF), Superscripts
# and Subscripts block (U+2070-209F), Arrows (U+2190-21FF), Letterlike
# Symbols (U+2100-214F — includes ℕℤℚℝℂ∀∃∂∇∞).


def _has_math_content(text):
    """True if text contains Unicode math characters.

    Used to detect math embedded in regular-text spans (born-digital PDFs
    from Word/LibreOffice). When True, the span should be treated as math
    even if its font is not in MATH_FONT_FAMILIES.

    Conservative: requires ≥2 distinct math chars OR ≥1 "strong" math signal
    (∑, ∏, ∫, √, ∂, ∇, ∀, ∃, ℝ, ℕ, etc.) to avoid false-positives on
    currency symbols (×, ± in prices) or simple arrows.
    """
    if not text:
        return False
    # Strong signals: a single occurrence is enough
    STRONG_MATH = frozenset("∑∏∫√∂∇∀∃ℝℕℤℚℂ⊆⊇⊕⊗⊢⊨⊤⊥")
    if any(c in STRONG_MATH for c in text):
        return True
    # Weak signal: need ≥2 distinct Unicode math chars
    found = UNICODE_MATH_CHARS.intersection(text)
    return len(found) >= 2


def _span_to_latex(span, baseline_y=None):
    """Convert a single math span to LaTeX notation.

    Detects subscripts/superscripts by font size + Y-position relative to
    baseline. Returns the LaTeX-formatted text.

    A span is a SUBSCRIPT if:
      - Its size is smaller than surrounding text (< 0.85x baseline size)
      - Its Y-center is LOWER than baseline (larger y = lower in PDF coords)
    A span is a SUPERSCRIPT if:
      - Smaller size
      - Y-center is HIGHER than baseline (smaller y)
    """
    text = span.get("text", "")
    font = span.get("font", "")
    size = span.get("size", 10.0)
    bbox = span.get("bbox", [0, 0, 0, 0])
    y_center = (bbox[1] + bbox[3]) / 2
    # Strip variation selectors (U+FE00-U+FE0F) — PDFs use them for math glyph variants.
    # Use \U0000XXXX syntax for chars above U+FFFF (VS17-VS256 = U+E0100-U+E01EF).
    text = re.sub(r"[\uFE00-\uFE0F\U000E0100-\U000E01EF]", "", text)
    # Convert Unicode chars to LaTeX equivalents (multi-char aware)
    out = []
    i = 0
    while i < len(text):
        # Try longest match first (3-char, 2-char, 1-char)
        matched = False
        for length in (3, 2, 1):
            chunk = text[i : i + length]
            if chunk in UNICODE_TO_LATEX:
                out.append(UNICODE_TO_LATEX[chunk])
                i += length
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out), size, y_center


def reconstruct_math_region(spans):
    """Convert a group of math spans to a single LaTeX expression.

    spans: list of dicts with text/font/size/bbox.

    Returns LaTeX string with proper subscripts/superscripts detected via
    font size + Y-offset from median baseline.
    """
    if not spans:
        return ""
    # Compute median baseline size (most common span size)
    sizes = [s.get("size", 10.0) for s in spans]
    median_size = sorted(sizes)[len(sizes) // 2]
    # Compute median Y-center (baseline reference)
    y_centers = [
        (s.get("bbox", [0, 0, 0, 0])[1] + s.get("bbox", [0, 0, 0, 0])[3]) / 2 for s in spans
    ]
    median_y = sorted(y_centers)[len(y_centers) // 2]
    # Build LaTeX, tracking when we enter/exit sub/super
    out = []
    in_sub = False
    in_sup = False
    for s in spans:
        text, size, y = _span_to_latex(s)
        if not text:
            continue
        is_small = size < median_size * 0.85
        is_lower = y > median_y + (median_size * 0.15)  # below baseline
        is_higher = y < median_y - (median_size * 0.15)  # above baseline
        if is_small and is_lower and not in_sub:
            out.append("_{")
            in_sub = True
        elif is_small and is_higher and not in_sup:
            out.append("^{")
            in_sup = True
        elif not is_small or (not is_lower and not is_higher):
            if in_sub:
                out.append("}")
                in_sub = False
            if in_sup:
                out.append("}")
                in_sup = False
        out.append(text)
    if in_sub or in_sup:
        out.append("}")
    return "".join(out)


def extract_math_regions(pdf_path, page_num):
    """Detect and reconstruct math regions on a PDF page.

    Algorithm (non-ML, font-based):
      1. Get all spans via PyMuPDF dict.
      2. Identify math spans (font in MATH_FONT_FAMILIES).
      3. Group consecutive math spans within same line.
      4. For each math group: reconstruct LaTeX with sub/super detection.
      5. Determine if math is DISPLAY (centered, isolated) or INLINE.
      6. Return list of {original_text, latex, bbox, display} dicts.

    Use in main() to REPLACE Unicode math in body text with proper LaTeX.
    """
    import fitz

    math_regions = []
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        page_width = page.rect.width
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                # Find groups of math spans. A "math region" is a sequence of
                # spans where ≥1 is math-font, allowing short structural spans
                # (parens, =, digits in CMR) between math spans. This groups
                # "(n_i - l_i)^2" as one region instead of fragments.
                line_math_groups = []
                current_group = []
                group_has_math = False
                # Chars that can appear between math spans without breaking region
                structural_chars = set("()[]{}=<>+-0123456789.,:;'\"/\\|")
                for span in spans:
                    is_math = _is_math_font(span.get("font", ""))
                    text = span.get("text", "")
                    # Phase B1: also treat span as math if it contains Unicode
                    # math characters (born-digital PDFs from non-LaTeX sources).
                    # Catches Word/LibreOffice/InDesign math that uses body font
                    # with Unicode chars instead of dedicated math fonts.
                    if not is_math and _has_math_content(text):
                        is_math = True
                    # Non-math span is "structural" if short and all structural chars
                    is_structural = (
                        not is_math
                        and len(text.strip()) <= 3
                        and all(c in structural_chars or c.isspace() for c in text)
                    )
                    if is_math:
                        current_group.append(span)
                        group_has_math = True
                    elif is_structural and current_group:
                        # Look ahead: is next non-structural span also math?
                        # If yes, include this structural span in group.
                        # (Heuristic: include if group already has math)
                        current_group.append(span)
                    else:
                        # Non-math, non-structural — close current group
                        if group_has_math and current_group:
                            line_math_groups.append(current_group)
                        current_group = []
                        group_has_math = False
                if group_has_math and current_group:
                    line_math_groups.append(current_group)
                # Process each math group
                for group in line_math_groups:
                    original = "".join(s.get("text", "") for s in group).strip()
                    if len(original) < 1:
                        continue
                    latex = reconstruct_math_region(group)
                    if not latex or latex == original:
                        continue  # nothing to convert
                    # Determine display vs inline:
                    #   Display math is centered horizontally + wide OR uses
                    #   CMEX big-operator fonts (∑∏∫ rendered at display size)
                    bbox = (
                        min(s["bbox"][0] for s in group),
                        min(s["bbox"][1] for s in group),
                        max(s["bbox"][2] for s in group),
                        max(s["bbox"][3] for s in group),
                    )
                    group_width = bbox[2] - bbox[0]
                    is_centered = (bbox[0] + bbox[2]) / 2
                    page_center = page_width / 2
                    has_big_operator = any(s.get("font", "").startswith("CMEX") for s in group)
                    is_display = has_big_operator or (
                        abs(is_centered - page_center) < page_width * 0.20
                        and group_width > page_width * 0.15
                    )
                    math_regions.append(
                        {
                            "original": original,
                            "latex": latex,
                            "bbox": bbox,
                            "display": is_display,
                        }
                    )
        doc.close()
    except Exception as e:
        log.warning("extract_math_regions failed p%s: %s", page_num, e)
    return math_regions


def replace_math_in_text(text, pdf_path, page_num):
    """Replace Unicode math in extracted text with proper LaTeX.

    Runs extract_math_regions, then for each region found, replaces the
    original text (if it appears in `text`) with $latex$ or $$latex$$.

    Conservative: only replaces exact substring matches. If math text was
    fragmented by other pipeline stages (column ordering, etc.), may not
    find a clean match — skips silently.
    """
    regions = extract_math_regions(pdf_path, page_num)
    if not regions:
        return text
    # Sort longest-original-first so longer matches win
    regions.sort(key=lambda r: len(r["original"]), reverse=True)
    for region in regions:
        original = region["original"]
        latex = region["latex"]
        if original not in text or len(original) < 2:
            continue
        # Skip if original already looks LaTeX-y (has \commands)
        if "\\" in original and any(c in original for c in "{}^_"):
            continue
        # MATH MASK (2026-08-16): never substitute inside existing $...$
        # math — a math-font span like 'k,m' matching a region original
        # produced '_{$k,m $}', corrupting delimiter pairing downstream
        # (flip-diff 2503.06703_pg4: '\mathbb{C}^{N}' dropped entirely).
        masked, saved = mask_math_regions(text)
        if original not in masked:
            continue
        if region["display"]:
            replacement = f"$${latex}$$"
        else:
            replacement = f"${latex}$"
        masked = masked.replace(original, replacement, 1)
        text = unmask_math_regions(masked, saved)
    return text


def improve_math_regions(text, pdf_path, page_num, tmp_dir, model="glm-ocr"):
    """Detect private-char regions and OCR-improve them via targeted region rendering.

    Hybrid pipeline: pdftotext + PyMuPDF subscripts get most content right.
    For chars where extraction failed (private/control codepoints = encoding
    failure), locate the region on the page via context search, render just
    that strip, OCR with math-aware prompt, and replace ONLY the problem chars.

    Uses PyMuPDF's page.search_for() for reliable context-to-position mapping
    (pypdfium2 text ≠ pdftotext text, making substring offset mapping unreliable).

    Returns improved text. Falls back to original on any failure.
    """
    if not DEPS.get("pdftoppm", False):
        return text
    import pymupdf as fitz

    # Find private-char clusters in the processed text
    priv_matches = list(_PRIVATE_CHAR_RE.finditer(text))
    if not priv_matches:
        return text

    try:
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num - 1]
            page_rect = page.rect

            # Process in REVERSE order so earlier offsets don't shift
            improved = text
            regions_fixed = 0
            for m in reversed(priv_matches):
                start, end = m.start(), m.end()
                # Expand to cover adjacent private chars
                while end < len(improved) and _PRIVATE_CHAR_RE.match(improved[end]):
                    end += 1

                # Build context snippet (surrounding non-private chars) to
                # locate this region on the page via search_for.
                ctx_start = max(0, start - 25)
                ctx_end = min(len(improved), end + 25)
                context = improved[ctx_start:end] + improved[end:ctx_end]
                context_clean = _PRIVATE_CHAR_RE.sub("", context).strip()
                # Collapse whitespace for search robustness
                context_clean = re.sub(r"\s+", " ", context_clean)
                if len(context_clean) < 5:
                    continue

                # Search for context on page (PyMuPDF handles coordinates)
                rects = page.search_for(context_clean)
                if not rects:
                    # Try progressively shorter context
                    for trim_len in (20, 15, 10):
                        shorter = context_clean[:trim_len].strip()
                        if len(shorter) >= 5:
                            rects = page.search_for(shorter)
                            if rects:
                                break
                if not rects:
                    continue

                # Use the match bbox, expanded with padding to capture full chars
                r = rects[0]
                min_x = max(r.x0 - 15, page_rect.x0)
                min_y = max(r.y0 - 8, page_rect.y0)
                max_x = min(r.x1 + 15, page_rect.x1)
                max_y = min(r.y1 + 8, page_rect.y1)
                if (max_x - min_x) <= 0 or (max_y - min_y) <= 0:
                    continue

                # Render just the problem region via PyMuPDF (no subprocess)
                clip = fitz.Rect(min_x, min_y, max_x, max_y)
                pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72, DPI / 72), clip=clip)
                region_path = os.path.join(tmp_dir, f"math_{page_num}_{start}.png")
                pix.save(region_path)
                region_path = resize_image(region_path, MAX_DIM)

                # OCR with math-aware prompt
                ocr_text = ocr_image_math(region_path, model=model)
                if not ocr_text or len(ocr_text.strip()) < 2:
                    continue

                # Splice: replace ONLY the private chars with OCR result
                improved = improved[:start] + ocr_text.strip() + improved[end:]
                regions_fixed += 1

            if regions_fixed > 0:
                return improved
            return text
        finally:
            doc.close()
    except Exception as e:
        log.warning("post-processor failed: %s", e)
        return text


# =============================================================================
# #4: Equation block detection via bbox analysis
# =============================================================================


# =============================================================================
# #5: Nougat-style markdown_compatible port (LaTeX cleanup)
# =============================================================================


# Regex patterns ported from nougat/postprocessing.py (Apache-2.0)
# Source: facebookresearch/nougat — cleaned up and minimal subset


_OCR_LATEX_CMD_CONFUSIONS = [
    (re.compile(r"\\sort\b"), r"\\sqrt"),
    (re.compile(r"\\sqr\b"), r"\\sqrt"),
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


# Late-import shims for cross-module dependencies (pdf_ocr.py).
# These resolve at call-time, avoiding circular import.
def _pdf_ocr():
    import sys

    m = sys.modules.get("__main__")
    if m is not None and str(getattr(m, "__file__", "")).endswith("pdf_ocr.py"):
        return m
    import pdf_ocr

    return pdf_ocr


def _get_PRIVATE_CHAR_RE():
    return _pdf_ocr()._PRIVATE_CHAR_RE


def _get_UNICODE_MATH_CHARS():
    return _pdf_ocr().UNICODE_MATH_CHARS


def _get_MATH_OCR_PROMPT():
    return _pdf_ocr().MATH_OCR_PROMPT


__all__ = [
    "detect_display_equations",
    "wrap_equations_as_latex",
    "ocr_display_equations",
    "MATH_FONT_PREFIXES",
    "UNICODE_TO_LATEX",
    "_is_math_font",
    "_has_math_content",
    "_span_to_latex",
    "reconstruct_math_region",
    "extract_math_regions",
    "replace_math_in_text",
    "improve_math_regions",
    "_OCR_LATEX_CMD_CONFUSIONS",
]
