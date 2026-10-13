"""Confidence scoring, quality gates, OCR helpers, SymSpell setup.

Extracted from pdf_ocr.py during R9 final slim-down (2026-08-12).
"""

from __future__ import annotations

import logging
import re
import subprocess

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
    own = set(g.keys())
    for k, v in vars(p).items():
        if k not in own and not k.startswith("__"):
            g[k] = v
    _CONSTS_LOADED = True


def _get_symspell():
    """Lazily build SymSpell instance with geology dict + standard English.

    Combines: (a) bundled geology dict above, (b) symspellpy's standard
    English frequency dict (82k words). Loaded ONCE per process via module
    global. Returns None if SymSpell not installed (graceful degradation
    per projectwide no-silent-fallback — but logged loudly).
    """
    global _SYMSPELL_INSTANCE
    if _SYMSPELL_INSTANCE is not None:
        return _SYMSPELL_INSTANCE
    try:
        from symspellpy import SymSpell
    except ImportError:
        log.warning(
            "symspellpy not installed — OCR correction disabled. Install via: uv add symspellpy"
        )
        return None
    import importlib.resources as _res

    ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
    # Load standard English frequency dict (bundled with symspellpy)
    try:
        en_dict = _res.files("symspellpy") / "frequency_dictionary_en_82_765.txt"
        ss.load_dictionary(str(en_dict), term_index=0, count_index=1)
    except Exception as e:
        log.warning("Failed to load English dict: %s", e)
    # Add geology terms (frequency=100 to bias toward our domain)
    for term in _GEOLOGY_DICT_TERMS:
        cleaned = term.strip()
        if not cleaned:
            continue
        ss.create_dictionary_entry(cleaned.lower(), 100)
        ss.create_dictionary_entry(cleaned.capitalize(), 50)
    _SYMSPELL_INSTANCE = ss
    return ss


def extract_text_with_scripts(pdf_path, page_num):
    """Extract page text with subscript/superscript annotation.

    Returns markdown-formatted text where detected subscripts are wrapped as
    $_{...}$ and superscripts as $^{...}$ (LaTeX inline math).

    Tries PyMuPDF first (correct reading order on 2-column papers, per-char
    bbox + font info). Falls back to pypdfium2 if PyMuPDF unavailable.
    Returns None if neither available or extraction fails.

    PyMuPDF advantage: respects 2-column reading order correctly. pypdfium2
    returns chars in PDF storage order which garbles 2-column papers —
    the sanity check inside then rejects the output, causing 2-column papers
    to skip subscript detection entirely. PyMuPDF doesn't have this issue.
    """
    # Try PyMuPDF first (preferred — handles 2-column)
    result = _extract_scripts_pymupdf(pdf_path, page_num)
    if result is not None:
        return result
    # Fall back to pypdfium2 (single-column only — sanity check rejects 2-col)
    return _extract_scripts_pypdfium(pdf_path, page_num)


def list_ollama_models():
    """Return set of model names available in local Ollama."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            errors="replace",
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return set()
    models = set()
    for line in result.stdout.split("\n")[1:]:
        line = line.strip()
        if not line:
            continue
        # ollama list format: "NAME:TAG    SIZE    MODIFIED"
        name = line.split()[0]
        # Keep BOTH the full "name:tag" (exact pinning) and the base name
        # (callers that pass bare names still match).
        models.add(name)
        models.add(name.split(":")[0])
    return models


def ollama_unload(model):
    """Unload a model from the Ollama server (keep_alive=0 expiry).

    Low-RAM hosts run ONE llama-server per loaded model; model-swap
    cascades (self-corrective rescue, chat-mode guard) otherwise leave
    two multi-GB llama-servers resident (OLLAMA_KEEP_ALIVE=30m) and OOM
    the box — measured 2026-08-18 bench: oom-kill task=llama-server on
    13GB host during a chandra→paddle rescue. Unloading the loser
    bounds resident VL models to one at a time.
    """
    if not model:
        return
    try:
        import json as _json
        import urllib.request as _urlreq

        req = _urlreq.Request(
            "http://127.0.0.1:11434/api/generate",
            data=_json.dumps({"model": model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        _urlreq.urlopen(req, timeout=30).read()
    except Exception as e:
        # Non-fatal by design: unload is a memory optimization, never a
        # correctness gate.
        logging.getLogger(__name__).debug("ollama_unload(%s) degraded: %s", model, e)


def vl_single_model_cap(default_model=None):
    """On low-RAM hosts, pin ONE VL model for the whole run.

    Ollama runs one llama-server per loaded model and keeps it resident
    (OLLAMA_KEEP_ALIVE=30m here). Per-page model selection (qwen for
    scanned pages, chandra for math, paddle for rescues) alternates
    models → two multi-GB llama-servers resident → OOM on small hosts
    (measured 2026-08-18: oom-kill llama-server + worker python on a
    13GB host mid-bench). Below 20GB total RAM the caller must request
    ONLY `default_model`; above (or unknown platform) alternates stay
    allowed. Returns default_model when capped, None when uncapped.
    """
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return default_model if kb < 20 * 1024 * 1024 else None
    except (OSError, ValueError, IndexError):
        pass
    return None


def detect_page_complexity(pdf_path, page_num):
    """Determine optimal OCR DPI from page font analysis.

    150=normal(>=8pt), 250=dense scientific(6-8pt), 300=tiny/scanned(<6pt).
    """
    _load_consts()
    pd = _get_page_dict(pdf_path, page_num)
    if pd is None:
        return DPI
    try:
        sizes = set()
        for block in pd["dict"]["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        sizes.add(round(span["size"], 1))
        doc.close()
        if not sizes:
            return 300
        min_size = min(sizes)
        if min_size < 6.0:
            return 300
        elif min_size < 8.0:
            return 250
        return 150
    except Exception:
        return DPI


# =============================================================================
# P0-1: Column-aware reading order reconstruction
# =============================================================================


def looks_like_chat_response(text: str) -> bool:
    """True when a VL OCR response is assistant-speak, not a transcription.

    Measured 2026-08-15 (tables bench): chandra-ocr-2 answered
    "The user wants me to extract..." on 9/10 born-digital table pages —
    prompt-independent (plain/short/math prompts all garbaged). A real
    OCR transcription never starts with chat preamble.
    """
    preamble = (
        "the user wants",
        "i will format",
        "i'll format",
        "i will create",
        "i'll create",
        "i need to extract",
        "let me extract",
        "based on the image",
        "based on the provided",
        "sure, here",
        "here is the transcription",
        "here's the transcription",
        "i see a",
        "i can see",
        # VL narration class (2026-09-17, bench14 hf flips): models describe
        # the page in prose ("The image is a page from a publication titled
        # ... At the top center, there is a small header line: ...") instead
        # of transcribing. Narration embeds header phrases in sentences, so
        # line-based HF strip can never remove them — reject like chat-speak.
        "the image is",
        "the image shows",
        "this image",
        "the page is",
        "the page shows",
        "this appears to be",
        "the scan shows",
        "the photograph",
    )
    head = text[:120].lstrip().lower()
    return any(head.startswith(p) for p in preamble)


def _compute_page_confidence(source, warnings, text):
    """Compute page-level extraction confidence for downstream RAG systems.

    Confidence hierarchy: page → block → line → cell.
    Page confidence is the aggregate signal — RAG systems use it to decide
    whether to trust, flag, or discard extracted content.

    Signals:
    - pdftotext source: base 1.0 (high quality text layer)
    - OCR source: base 0.7 (vision model, prone to errors)
    - Each warning: -0.05 (ree_, oxide_, encoding_, etc.)
    - encoding_artifacts warning: -0.15 (significant quality impact)
    - U+FFFD density in text: -0.1 per 5% replacement chars
    - Clamp to [0.1, 1.0]
    """
    if "ocr" in source.lower():
        conf = 0.7
    else:
        conf = 1.0

    for w in warnings:
        if "encoding_artifacts" in w:
            conf -= 0.15
        elif any(k in w for k in ("ree_", "oxide_", "age_", "mineral_")):
            conf -= 0.08
        else:
            conf -= 0.05

    if text:
        fffd_ratio = text.count("\ufffd") / max(len(text), 1)
        conf -= fffd_ratio * 2.0  # 5% U+FFFD → -0.1
        # Degenerate-generation signatures (measured 2026-08-15 bench):
        # base 0.7 vs floor 0.55 meant the loop NEVER fired. These are the
        # real failure modes VL models produce on bad crops/tiny text.
        import re as _re

        if text.count("[OCR loop collapsed") >= 2:
            conf -= 0.45  # repetition collapse
        # Chat-mode hallucination (see looks_like_chat_response — measured
        # 2026-08-15: chandra-ocr-2 scored base 0.70 so the rescue loop
        # never fired). Penalty drops it below the floor.
        if looks_like_chat_response(text):
            conf -= 0.45  # chat mode, not transcription
        _space_n = sum(
            text.count(c) for c in ("\\qquad", "\\quad", "\\circ", "\\cdot", "\\star", "\\;")
        )
        _content_n = len(
            _re.findall(r"\\(?:frac|int|sqrt|pi|alpha|beta|gamma|theta|lambda|mu|sum)", text)
        )
        if _space_n > 20 and _space_n > _content_n:
            conf -= 0.45  # spacing-loop dominance (garbage equation crops)
        if "\ufffd" not in text and len(text) > 200:
            # rambling-hallucination heuristic: formula-dense text on a page
            # whose OCR carried no dictionary words at all
            words = _re.findall(r"[A-Za-z]{3,}", text)
            if len(words) < len(text) / 60:  # <1 real word per 60 chars
                conf -= 0.35

    return round(max(0.1, min(1.0, conf)), 2)


def combine_page_confidence(
    validator_conf: float,
    reflexive_conf: float | None,
    has_warnings: bool,
) -> float:
    """Combine validator-derived and model-reported page confidence.

    Fixes the 2026-08-13 confidence-override bug where
    ``_conf = max(_conf, reflexive_meta["confidence"])`` let a model's
    self-reported confidence MASK validator-detected defects (a page could
    be accepted as high-confidence despite warnings).

    Policy:
      - No warnings: model self-confidence may raise the score (it can add
        signal the warning-based scorer cannot see, e.g. a KaTeX render
        check). Take the max.
      - Warnings present: blend validator + model, but the combined score
        CANNOT exceed the validator quality score. Model optimism never
        masks detected defects; model doubt can still lower the score.

    Args:
      validator_conf: score from :func:`_compute_page_confidence` (defect-based).
      reflexive_conf: model/reflexive-loop self-reported confidence, or None.
      has_warnings: True if the page produced any quality warnings.

    Returns:
      Combined confidence in [0.1, 1.0], rounded to 2 decimals.
    """
    base = max(0.1, min(1.0, float(validator_conf)))
    if reflexive_conf is None:
        return round(base, 2)
    refl = max(0.0, min(1.0, float(reflexive_conf)))
    if not has_warnings:
        return round(max(base, refl), 2)
    blended = 0.5 * base + 0.5 * refl
    return round(max(0.1, min(1.0, min(base, blended))), 2)


def _quality_gate(cells):
    """Check if extracted grid is a real table (not prose fragments).

    Accepts if: ≥3 rows AND (≥5% decimals OR ≥15% digits OR ≥10% chemistry).
    Works with Cell objects or plain strings (backward compatible).
    """

    def _txt(cell):
        return cell.text if hasattr(cell, "text") else str(cell)

    cells = [r for r in cells if any(_txt(c).strip() for c in r)]
    if len(cells) < 3:
        return False, cells
    if len(cells[0]) < 2:
        return False, cells

    total = sum(len(r) for r in cells)
    decimals = sum(1 for r in cells for c in r if _txt(c) and re.search(r"\d+\.\d+", _txt(c)))
    digits = sum(1 for r in cells for c in r if _txt(c) and re.search(r"\d", _txt(c)))
    chemistry = sum(
        1 for r in cells for c in r if _txt(c) and re.match(r"^[A-Z][a-z]?O?\d?", _txt(c).strip())
    )

    has_decimals = decimals / max(total, 1) >= 0.05
    has_digits = digits / max(total, 1) >= 0.15
    has_chemistry = chemistry / max(total, 1) >= 0.10

    return (has_decimals or has_digits or has_chemistry), cells


def crop_image(img_path, crop, output_path):
    """Crop image by (x, y, w, h) in PDF points, top-left origin, save to output_path."""
    from PIL import Image

    x, y, w, h = crop
    scale = DPI / 72.0
    bbox_px = (
        int(x * scale),
        int(y * scale),
        int((x + w) * scale),
        int((y + h) * scale),
    )
    with Image.open(img_path) as im:
        im_cropped = im.crop(bbox_px)
        im_cropped.save(output_path)
    return output_path


def has_math_content(text):
    """Quick check if text has any math content worth OCR improvement."""
    if not text:
        return False
    if _MATH_SYMBOLS_RE.search(text):
        return True
    if _PRIVATE_CHAR_RE.search(text):
        return True
    if "$_{" in text or "$^{" in text:
        return True
    return False


# =============================================================================
# #2: Math-aware OCR prompt
# =============================================================================

# Default prompt is "Text Recognition:" — minimal, lets model interpret freely
# For math-heavy regions, use a prompt that explicitly requests LaTeX
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


REE_OCR_CONFUSIONS = {"Cd": "Gd", "Ib": "Tb", "Or": "Er", "Tni": "Tm", "Cin": "Sm"}


def _has_monotonic_re_run(positions_tokens, min_run: int = 4) -> bool:
    """True if positions_tokens contains a run of ``min_run`` or more REE
    symbols in strictly increasing ``REE_SEQUENCE`` index order.

    This is the signature of an actual REE listing (axis label, plot row,
    table header) — body prose mentions of "Eu/Eu*", "La/Yb", "Sm/Nd" are
    scattered in non-monotonic order and filtered out.
    """
    best = current = 0
    last_idx: int | None = None
    for _pos, token in positions_tokens:
        resolved = REE_OCR_CONFUSIONS.get(token, token)
        if resolved not in REE_SEQUENCE:
            continue
        idx = REE_SEQUENCE.index(resolved)
        if last_idx is not None and idx > last_idx:
            current += 1
            if current > best:
                best = current
        else:
            current = 1
            if best == 0:
                best = 1
        last_idx = idx
    return best >= min_run


SCIENTIFIC_SYMBOL_RULES = [
    (r"(\d)\s*u(?=[mglL]\b)", r"\1μ"),
    (r"(\d)\s*W(?=\s|$)", r"\1 Ω"),
    (r"(\d)\s*[xX]\s*(\d)", r"\1 × \2"),
    (r"(\d)\s*-\s*(\d)", r"\1 − \2"),
    (r"(\d)\s*[oO]\s*([CFK])\b", r"\1°\2"),
    (r"(\d)\s*A(?=\s|$)", r"\1 Å"),
]


def invalid_char_count(text):
    """Count of replacement chars (U+FFFD), private-use area, and control chars.

    Marker SOTA. Detects encoding issues, embedded font problems, and
    invisible/glyphless text layers.
    """
    bad = 0
    for c in text:
        cp = ord(c)
        if cp == 0xFFFD:
            bad += 1
        elif 0xE000 <= cp <= 0xF8FF:
            bad += 1
        elif cp < 32 and c not in "\t\n\r\f":
            bad += 1
    return bad


__all__ = [
    "_get_symspell",
    "extract_text_with_scripts",
    "list_ollama_models",
    "detect_page_complexity",
    "_compute_page_confidence",
    "_quality_gate",
    "crop_image",
    "has_math_content",
    "_has_monotonic_re_run",
    "invalid_char_count",
]
