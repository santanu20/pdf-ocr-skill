"""Self-corrective display-equation cascade (KaTeX oracle + local repair).

Three-tier architecture (2026-08-14, per SOTA research — Marker balanced
scores 83.9 on arXiv math via equation-crop VLM OCR; text-layer ceiling is
~26% as measured on our bench):

  Tier 1  font-geometry candidate (font_math_detector v5) — free, fast
  Tier 2  KaTeX oracle validates each $$display$$ block; failures are
          repaired by RapidLaTeX (local ONNX) on a bbox crop of the exact
          cluster; repair is itself KaTeX-validated before acceptance
  Tier 3  optional VL escalation (Ollama glm-ocr w/ math prompt) when
          RapidLaTeX also fails validation

The oracle (olmocr.bench.katex.render.render_equation) gives a boolean
"renders cleanly" signal — the same signal the bench scorer uses — so the
cascade optimizes precisely what is measured. No GPU, no network: all
components are local; playwright/KaTeX ships with olmocr, RapidLaTeX is
an in-process ONNX encoder-decoder.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from scientific_parser.core.path_warnings import record_path_warning
from scientific_parser.science.span_conservation import conservation_pass

log = logging.getLogger(__name__)

_MAX_REPAIRS_PER_PAGE = 24  # RapidLaTeX is ~0.3s/crop; Hangul-mapped pages carry 16+ display eqs


# ── KaTeX oracle ──────────────────────────────────────────────────── #

_katex_ok_cache: dict[str, bool] = {}

# Broken-ToUnicode signature: arXiv math fonts sometimes map to Hangul
# syllables (U+AC00–D7AF) or private-use codepoints instead of real math
# chars. The text layer reads "fine" but every math glyph is garbage —
# KaTeX renders Hangul as text without error, so the oracle alone cannot
# catch this class (bench 2026-08-15: '푡 \to \infty' everywhere).
# NOTE: math-alphanumeric codepoints (U+1D400+) are a LEGITIMATE font
# encoding handled downstream by unicode→LaTeX — including them here
# force-RL'd healthy pages and REGRESSED the bench sample 33.9→27.5.
_BROKEN_MAP_CHARS_RE = re.compile(
    "[\uac00-\ud7af"  # Hangul syllables (arXiv broken ToUnicode class)
    "\ue000-\uf8ff]"  # private use area
)

# GLYPH DUPLICATION (iter-121a, 2026-08-28): text-layer + math-span join
# math font and the literal unicode from the text span ('IFN \gammaγ',
# measured 0225d691 pg2). KaTeX renders the duplicated form cleanly, so
# the oracle is blind; the pattern itself is the signature.
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
_GLYPH_DUP_RE = re.compile(
    r"\\("
    + "|".join(_CMD_TO_UNI)
    + r")\s*("
    + "|".join(re.escape(u) for u in _CMD_TO_UNI.values())
    + r")"
)


def _glyph_dup(inner: str) -> bool:
    """True when a \\command sits adjacent to its own unicode letter."""
    m = _GLYPH_DUP_RE.search(inner)
    return bool(m) and _CMD_TO_UNI.get(m.group(1)) == m.group(2)


def _needs_repair(inner: str, page_map_broken: bool = False) -> bool:
    import os as _os

    if _os.environ.get("PDF_OCR_NO_CASCADE"):
        return False
    return _needs_repair_real(inner, page_map_broken)


def _needs_repair_real(inner: str, page_map_broken: bool = False) -> bool:
    """True when a display block must be re-recognized from the page image.

    Beyond the KaTeX oracle, two SEMANTIC failure modes it cannot see:
      1. TRUNCATED start (base symbol lost): block begins with } ^ _ —
         renders fine, matches nothing (837 instances / 508 bench pages).
      2. Broken font→unicode mapping: Hangul/PUA chars in math where the
         real glyph is a Latin/Greek letter.
      3. CMEX bigop junk (2026-08-16): CMEX10 ToUnicode often maps the
         big-operator slots to ASCII — display ∫(0x5A)→'Z', text ∫(0x52)→
         'R', ∑(0x50)→'P', ∮(0x48)→'H', radical pieces→'n'/'o'/'#'.
         The assembled block KaTeX-RENDERS (junk chars are valid tokens)
         while reading nothing like the gold. Signature: a body-size
         CMEX-font span contributed suspicious ASCII. Detect post-hoc:
         display containing ≥1 of the junk letters adjacent to math
         structure where a big operator is expected.
    On pages whose math fonts are mapped broken (page_map_broken), every
    display block is re-recognized via RapidLaTeX crop regardless.
    """
    if page_map_broken:
        return True
    if not inner.strip():
        return True
    if re.match(r"^[}\^_]", inner.strip()):
        return True
    if _BROKEN_MAP_CHARS_RE.search(inner):
        return True
    if _CMEX_JUNK_RE.search(inner):
        return True
    if _glyph_dup(inner):
        return True
    return not katex_renders_clean(inner)


# CMEX bigop junk: ASCII letters that are actually display operators.
# High-precision shape: letter followed/preceded by spacing/command tokens,
# or the letter standing alone as a token. Measured on 2503.04993_pg18:
# '" Z n' = \Bigg[ \int; 'n f_1' = \frac{1}{\epsilon} f_1; bare 'Z P Q R'
# at display size = ∫ ∑ ∏. False-positive risk: prose words inside math
# runs (e.g. 'R' as a set name) — require the junk letter to be a lone
# single-letter token or preceded by a quote-mark/big-delim junk char.
_CMEX_JUNK_RE = re.compile(
    r"(?:"
    r"(?<![\w\\]){[ZPRQHXWF]}\s"  # {Z} lone token (script group contents)
    r"|(?<![\w\\])[ZPRQHXWF](?![\w{])\s+(?:d\\|\\partial|\\mathbb|\\frac|f_|\\zeta|\\lim)"  # Z d… / P ∂… bigop+operand shape
    r'|"\s*[A-Za-z\\]'  # '"' junk char itself (big-delim slot)
    r"|\bn\s+d\\|\bn\s+\\partial"  # 'n d\' radical/frac piece shape
    r")"
)


def _is_junk_display(inner: str) -> bool:
    import os as _os

    if _os.environ.get("PDF_OCR_NO_CASCADE"):
        return False
    return _is_junk_display_real(inner)


def _is_junk_display_real(inner: str) -> bool:
    """A 'display equation' with zero math structure is a misclassified
    prose word/title ('A', 'ON', 'A NOTE ON THE ...') — unwrap to text."""
    s = inner.strip()
    if not s:
        return True
    # No math structure AND no parens/digits (f(x)g(x), P(n) are real math).
    if re.search(r"[=\\^_{}±×÷≤≥∞∑∫√<>+()\d]", s):
        return False
    # Pure letters/spaces with ≤3 words at any length = prose/title.
    return len(s.split()) <= 6


def katex_renders_clean(latex: str) -> bool:
    """True iff KaTeX parses+renders the LaTeX without error (cached)."""
    key = latex.strip()
    if not key:
        return False
    hit = _katex_ok_cache.get(key)
    if hit is not None:
        return hit
    try:
        from olmocr.bench.katex.render import render_equation

        res = render_equation(key)
        ok = res is not None and not res.error
    except Exception:
        # oracle unavailable → accept candidates unvalidated (fail-open:
        # geometry output is still the best we have)
        return True
    _katex_ok_cache[key] = ok
    return ok


# ── Local repair engines (lazy singletons) ────────────────────────── #

_rapidlatex = None


def _get_rapidlatex():
    global _rapidlatex
    if _rapidlatex is None:
        from scientific_parser.perception.providers.formula_ocr import (
            RapidLaTeXProvider,
        )

        _rapidlatex = RapidLaTeXProvider()
    return _rapidlatex


def _crop_cluster_png(page: Any, bbox: tuple, dpi: int = 250) -> Optional[str]:
    """Render the cluster bbox (padded) to a PNG; return path or None."""
    try:
        import fitz

        x0, y0, x1, y1 = bbox
        rect = page.rect
        clip = fitz.Rect(
            max(0, x0 - 8), max(0, y0 - 6), min(rect.width, x1 + 8), min(rect.height, y1 + 6)
        )
        if clip.is_empty or clip.width < 4 or clip.height < 4:
            return None
        import os
        import tempfile

        pix = page.get_pixmap(dpi=dpi, clip=clip)
        fd, path = tempfile.mkstemp(suffix=".png", dir=None)
        os.close(fd)
        pix.save(path)
        return path
    except Exception as e:
        log.debug("crop failed: %s", e)
        return None


def _vl_repair(png_path: str, model: str) -> Optional[str]:
    """Tier 3: Ollama VL OCR of the crop with the math-aware prompt."""
    try:
        from pdf_ocr import MATH_OCR_PROMPT, ocr_image

        out = ocr_image(png_path, task=MATH_OCR_PROMPT, model=model)
        return out.strip() or None
    except Exception as e:
        log.debug("VL repair failed: %s", e)
        return None


def _clean_latex_shell(s: str) -> str:
    """Strip $$/$ wrappers and markdown fences a repair model may add."""
    s = s.strip()
    s = re.sub(r"^```(?:latex|math)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
    s = re.sub(r"^\$\$(.+)\$\$$", r"\1", s, flags=re.DOTALL).strip()
    s = re.sub(r"^\\\[(.+)\\\]$", r"\1", s, flags=re.DOTALL).strip()
    s = re.sub(r"^\\\((.+)\\\)$", r"\1", s, flags=re.DOTALL).strip()
    # RapidLaTeX systematic outputs KaTeX rejects:
    #   \d t  → \mathrm{d} t   (differential d, most common reject)
    #   \stackrel{a}_{b} nested under-accent → drop the stackrel shell
    s = re.sub(r"\\d\s+(?=[txyzsuro])", r"\\mathrm{d} ", s)
    s = re.sub(r"\\d(?=[xyzsuro]\b)", r"\\mathrm{d}", s)
    return s


def _looks_like_prose_latex(latex: str) -> bool:
    """Reject 'equation' recognition output that is actually PROSE.

    DocLayout mislabels italic theorem text as equations; RapidLaTeX then
    emits glued-word LaTeX ('GwithpartsA , BandC ... \\betaand\\gamma') that
    KaTeX renders as text without error — the oracle cannot catch it.
    Signature: long alpha runs dominate and math commands are sparse.

    STRENGTHENED 2026-08-15 (bench 2503.09086 evidence): multi-line display
    crops make RapidLaTeX emit garbled-prose-as-array that ALSO renders —
    '\\begin{array}{l}{{\\mathrm{~4pproach,the~parameters~}...}}' with ~
    used as space. Additional signatures:
      - \\begin{array} wrappers (real display eqs are never arrays here)
      - ≥6 '~' (garbled spacing)
      - >8 consecutive identical commands (\\cdots spam)
    """
    s = latex.strip()
    if len(s) < 40:
        # short output: still reject array wrappers + tilde-soup
        if re.search(r"\\begin\{array\}", s) or s.count("~") >= 4:
            return True
        return False
    if re.search(r"\\begin\{array\}", s):
        return True
    if s.count("~") >= 6:
        return True
    toks = re.findall(r"\\[a-zA-Z]+", s)
    run = best = 1
    for i in range(1, len(toks)):
        run = run + 1 if toks[i] == toks[i - 1] else 1
        best = max(best, run)
    if best > 8:
        return True
    alpha_run_len = sum(len(m.group(0)) for m in re.finditer(r"[A-Za-z]{5,}", s))
    n_cmds = len(toks)
    return alpha_run_len > 0.4 * len(s) and n_cmds < len(s) / 40


def _semantic_retention(candidate: str, replacement: str) -> bool:
    """Repair must retain ≥50% of the font candidate's alnum content.

    Image-recognized replacements that share almost no letters/digits with
    the font-assembled candidate are hallucinations (different content
    read off the crop), not repairs.
    """

    def _sig(s):
        return re.sub(r"[^a-zA-Z0-9]", "", s).lower()

    a, b = _sig(candidate), _sig(replacement)
    if not a or not b:
        return True  # nothing to compare — allow
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    keep = sum((ca & cb).values())
    return keep >= 0.5 * len(a)


# ─── Symbol-agreement gate (2026-08-15, render-vs-source tier 1) ──────── #
# KaTeX is a SYNTAX oracle, not a semantic one — "renders cleanly" ≠ "is
# the correct equation". The font-assembled candidate carries GROUND-TRUTH
# codepoints from the PDF text layer (when the font map is intact), so a
# replacement that renders but reads a different equation can be caught by
# comparing math-symbol multisets. Alnum retention is blind to symbols
# (∑↔∏ swap passes); this gate closes that hole. True pixel-render
# comparison (KaTeX→HTML→screenshot) needs browser infra — symbol agreement
# is the lean research-grade equivalent.

_LATEX_CMD_TO_SYM = {
    # Greek (lowercase — epsilon/varepsilon are DISTINCT glyphs, keep distinct)
    r"alpha": "α",
    r"beta": "β",
    r"gamma": "γ",
    r"delta": "δ",
    r"epsilon": "ϵ",
    r"varepsilon": "ε",
    r"zeta": "ζ",
    r"eta": "η",
    r"theta": "θ",
    r"vartheta": "ϑ",
    r"iota": "ι",
    r"kappa": "κ",
    r"lambda": "λ",
    r"mu": "μ",
    r"nu": "ν",
    r"xi": "ξ",
    r"pi": "π",
    r"varpi": "ϖ",
    r"rho": "ρ",
    r"varrho": "ϱ",
    r"sigma": "σ",
    r"varsigma": "ς",
    r"tau": "τ",
    r"upsilon": "υ",
    r"phi": "ϕ",
    r"varphi": "φ",
    r"chi": "χ",
    r"psi": "ψ",
    r"omega": "ω",
    # Greek (uppercase)
    r"Gamma": "Γ",
    r"Delta": "Δ",
    r"Theta": "Θ",
    r"Lambda": "Λ",
    r"Xi": "Ξ",
    r"Pi": "Π",
    r"Sigma": "Σ",
    r"Phi": "Φ",
    r"Psi": "Ψ",
    r"Omega": "Ω",
    # Big operators / calculus
    r"sum": "∑",
    r"prod": "∏",
    r"int": "∫",
    r"oint": "∮",
    r"sqrt": "√",
    r"partial": "∂",
    r"nabla": "∇",
    r"infty": "∞",
    # Relations
    r"leq": "≤",
    r"geq": "≥",
    r"neq": "≠",
    r"approx": "≈",
    r"sim": "∼",
    r"equiv": "≡",
    r"propto": "∝",
    r"ll": "≪",
    r"gg": "≫",
    # Operators / arrows / sets
    r"pm": "±",
    r"mp": "∓",
    r"times": "×",
    r"div": "÷",
    r"cdot": "⋅",
    r"to": "→",
    r"rightarrow": "→",
    r"leftarrow": "←",
    r"gets": "←",
    r"leftrightarrow": "↔",
    r"in": "∈",
    r"notin": "∉",
    r"subset": "⊂",
    r"supset": "⊃",
    r"subseteq": "⊆",
    r"supseteq": "⊇",
    r"forall": "∀",
    r"exists": "∃",
    r"oplus": "⊕",
    r"otimes": "⊗",
    r"emptyset": "∅",
    r"cdots": "⋯",
    r"ldots": "…",
    r"vdots": "⋮",
    r"ddots": "⋱",
}
# Unicode symbols recognized directly (superset of the values above).
_UNI_SYM_CHARS = set(_LATEX_CMD_TO_SYM.values()) | {
    "∓",
    "⋅",
    "∼",
    "≪",
    "≫",
    "⊂",
    "⊃",
    "⊆",
    "⊇",
    "∉",
    "∅",
    "⊕",
    "⊗",
    "⋮",
    "⋱",
    "…",
    "∠",
    "∘",
    "⋆",
    "⩾",
    "⩽",
    "∆",
    "Ω",
    "ﬁ",
    "ﬂ",
}


def _symbol_multiset(s: str) -> dict:
    """Math-symbol kinds present in a LaTeX/Unicode mixed string.

    Picks up (a) Unicode math chars directly, (b) LaTeX commands mapped
    through _LATEX_CMD_TO_SYM. Prose letters/digits are ignored — only
    symbol KINDS matter (counts vary with recognition fidelity).
    """
    from collections import Counter

    out = Counter(ch for ch in s if ch in _UNI_SYM_CHARS)
    for cmd, sym in _LATEX_CMD_TO_SYM.items():
        if re.search(r"\\" + cmd + r"(?![a-zA-Z])", s):
            out[sym] += 1
    return out


def _symbols_agree(candidate: str, replacement: str) -> bool:
    """Gate: replacement must keep ≥50% of candidate's symbol KINDS and
    introduce at most 4 new kinds (a repair fixes symbols, it doesn't
    transcribe a different equation). Vacuous-pass when candidate has no
    symbols. Call only when the font map is intact (candidate codepoints
    are ground truth); with a broken map the candidate's symbols are junk.
    """
    ca = _symbol_multiset(candidate)
    if not ca:
        return True
    cb = _symbol_multiset(replacement)
    keep = len(set(ca) & set(cb))
    added = len(set(cb) - set(ca))
    return keep >= 0.5 * len(ca) and added <= 4


def _region_equations(pdf_path: str, page_num: int, page: Any) -> list[dict]:
    """DocLayout-YOLO equation regions → RapidLaTeX crops (validated).

    The text-layer ceiling on arXiv math is ~26% (measured); Marker breaks
    it (83.9) by recognizing EVERY equation region from the page image.
    This gives (bbox, latex, is_display) for each region the font pass can
    miss entirely (54% of bench math tests were absent, 2026-08-15).
    """
    try:
        from scientific_parser.perception.providers.layout_detector import (
            get_layout_detector,
        )

        layout = get_layout_detector(
            "default"
        )  # ONNX + RapidLayout fallback ("doclayout-yolo" = torch, unavailable)
        if not layout or not layout.available():
            return []
        regions = layout.detect_regions(pdf_path, page_num)
    except Exception as e:
        log.debug("region equation layout failed: %s", e)
        return []

    rl = None
    try:
        rl = _get_rapidlatex()
    except Exception as e:
        log.debug("rapidlatex unavailable: %s", e)
        return []

    out: list[dict] = []
    page_h = page.rect.height
    for r in regions:
        if r.get("type") != "equation":
            continue
        bbox = r["bbox"]
        h = bbox[3] - bbox[1]
        is_display = h > 18 or (bbox[2] - bbox[0]) > page.rect.width * 0.5
        png = _crop_cluster_png(page, bbox, dpi=300)
        if not png:
            continue
        try:
            latex, _conf = rl.recognize(png)
        except Exception:
            continue
        latex = _clean_latex_shell(latex or "")
        if (
            latex
            and len(latex) > 2
            and katex_renders_clean(latex)
            and not _looks_like_prose_latex(latex)
        ):
            out.append({"bbox": bbox, "latex": latex, "display": is_display})
        if len(out) >= 30:  # latency bound
            break
    return out


def _bbox_overlap(a, b) -> float:
    """IoU-lite: intersection area over smaller box area."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix0 >= ix1 or iy0 >= iy1:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    a_area = max((a[2] - a[0]) * (a[3] - a[1]), 1)
    b_area = max((b[2] - b[0]) * (b[3] - b[1]), 1)
    return inter / min(a_area, b_area)


# ── Main entry ────────────────────────────────────────────────────── #


def extract_text_with_cascade(
    page: Any,
    tmp_dir: str = "",
    vl_model: Optional[str] = None,
    pdf_path: str = "",
    page_num: int = 0,
) -> str:
    """v5 geometry assembly + oracle-gated repair of display equations.

    Returns the page markdown. Identical to extract_text_with_math_fonts
    when every display block renders cleanly (zero model calls); repair
    fires only on KaTeX failures, bounded at _MAX_REPAIRS_PER_PAGE.

    Region tier (pdf_path given): DocLayout equation regions recognized
    via RapidLaTeX REPLACE overlapping font-display clusters (image truth
    beats font assembly) and unmatched regions are INSERTED (recovers
    equations the font pass missed — the 54% absent class).
    """
    from scientific_parser.science.font_math_detector import (
        _build_line_with_math,
        _line_body_and_baseline,
        _ACTIVE_CMEX_GLYPHS,
        collect_math_clusters,
        load_cmex_glyph_map,
    )

    # Embedded-font CMEX glyph decoding must cover THIS assembly too —
    # _span_to_math_token consults the page-scoped map (iter-14).
    import scientific_parser.science.font_math_detector as _fmd

    _saved_glyphs = _fmd._ACTIVE_CMEX_GLYPHS
    _fmd._ACTIVE_CMEX_GLYPHS = load_cmex_glyph_map(page)
    _saved_phi = _fmd._PAGE_PHI_DUAL
    _fmd._PAGE_PHI_DUAL = _fmd._detect_phi_dual(page)
    try:
        clusters = collect_math_clusters(page)
        out_lines: list[str] = []
        repairs = 0

        # Page-level broken-mapping detection: any math-font span carrying
        # Hangul/PUA chars means the font→unicode map is junk → every display
        # equation on this page must come from image recognition, not fonts.
        page_map_broken = False
        for cl in clusters:
            for s_ in cl["spans"]:
                if s_.get("text") and _BROKEN_MAP_CHARS_RE.search(s_["text"]):
                    page_map_broken = True
                    break
            if page_map_broken:
                break

        # Region tier: image-based equation recognition (Marker-style).
        region_eqs: list[dict] = []
        if pdf_path and page_num:
            region_eqs = _region_equations(pdf_path, page_num, page)
        used_regions: set[int] = set()

        emitted: list[tuple[float, str]] = []  # (y0, line) for ordered assembly
        # Parallel x-extents per emission (det50): column-aware final
        # ordering needs geometry, not just y0.
        exts: list[tuple[float, float]] = []
        # det51: conservation must audit the span inventory the builder
        # actually consumed — the split PARTS, not the original fused
        # clusters. Auditing the fused cluster against a part's line sees
        # the other column's spans as "missing" and re-injects them,
        # gluing right-column fragments into left-column math
        # (2503.04147 pg3: '\text{and \bm{z}} … N \times \times').
        check_clusters: list[dict] = []
        _mid = page.rect.width / 2.0
        _page_two_col = (
            sum(1 for c in clusters if c["bbox"][2] <= _mid - 10) >= 3
            and sum(1 for c in clusters if c["bbox"][0] >= _mid + 10) >= 3
        )
        for cl in clusters:
            # det50: a baseline cluster on a two-column page can carry
            # spans from BOTH columns (baselines share y within the
            # cluster tolerance) — _build_line_with_math then FUSES
            # left+right text into one corrupted line ('BRM consists of
            # 2 linear feedback guess the initial state…', 0853d8a0
            # pg2). Split each cluster's span run at a gap that
            # STRADDLES the page midline so each column builds and emits
            # its own line. Gated on the PAGE being genuinely two-column
            # (>=3 clusters fully left AND >=3 fully right of mid):
            # single-column pages carry sparse display equations whose
            # wide inter-token gaps also straddle mid — splitting those
            # shredded 'P_{XY} = sd | s (1 - d )' (measured 2503.03754
            # pg10: fullR=0 → gate closed → byte-identical). Prebuilt
            # environment clusters stay whole (display math never spans
            # the gutter; the tall-delimiter merge is within-column).
            if cl.get("env_latex") is not None or not _page_two_col:
                _parts = [cl]
            else:
                _parts = _split_cluster_at_gutter(cl, page.rect.width)
            for cl in _parts:
                check_clusters.append(cl)
                # (x0, origin.y): deterministic sup-before-sub at equal x (iter-26)
                spans = sorted(cl["spans"], key=lambda s: (s["bbox"][0], s["origin"][1]))
                body_size, baseline_y = _line_body_and_baseline(spans)
                if cl.get("env_latex") is not None:
                    # Synthetic environment (cases/bmatrix from font_math_detector's
                    # tall-delimiter merge): prebuilt LaTeX, skip repair path.
                    emitted.append(
                        (cl["bbox"][1], f"$${cl.get('env_lhs', '')} {cl['env_latex']}$$")
                    )
                    exts.append((cl["bbox"][0], cl["bbox"][2]))
                    continue
                line_text = _build_line_with_math(spans, body_size, baseline_y)
                if not line_text.strip():
                    continue

                # Repair only display blocks that fail the oracle
                m = re.match(r"^\$\$(.+)\$\$$", line_text.strip(), flags=re.DOTALL)
                if m:
                    inner = m.group(1).strip()
                    if _is_junk_display(inner):
                        # prose word misclassified as display — emit as plain text
                        emitted.append((cl["bbox"][1], inner))
                        exts.append((cl["bbox"][0], cl["bbox"][2]))
                        continue
                    # Region tier FIRST: an image-recognized equation overlapping
                    # this display cluster replaces font assembly wholesale.
                    best_ri, best_ov = None, 0.0
                    for ri, re_ in enumerate(region_eqs):
                        if re_["display"] and ri not in used_regions:
                            ov = _bbox_overlap(cl["bbox"], re_["bbox"])
                            if ov > best_ov:
                                best_ri, best_ov = ri, ov
                    if best_ri is not None and best_ov > 0.3:
                        re_ = region_eqs[best_ri]
                        # semantic gate: replacement must retain the font
                        # candidate's content (hallucination guard, 2026-08-15)
                        # + symbol agreement when font map intact (render-vs-source
                        # tier 1: alnum retention is blind to ∑↔∏ swaps)
                        if _semantic_retention(inner, re_["latex"]) and (
                            page_map_broken or _symbols_agree(inner, re_["latex"])
                        ):
                            used_regions.add(best_ri)
                            emitted.append((cl["bbox"][1], "$$" + re_["latex"] + "$$"))
                            exts.append((cl["bbox"][0], cl["bbox"][2]))
                            continue
                    if repairs < _MAX_REPAIRS_PER_PAGE and _needs_repair(inner, page_map_broken):
                        repairs += 1
                        # DETERMINISTIC BEAM FIRST (iter-135): structural
                        # variants scored by the cached KaTeX oracle; content
                        # preserved (alnum multiset locked), render-clean
                        # adoption only. VL region/model tier gets second
                        # refusal. Kill-switch PDF_OCR_BEAM_REPAIR=0.
                        replacement = None
                        if os.environ.get("PDF_OCR_BEAM_REPAIR", "1") == "1":
                            try:
                                from scientific_parser.science.beam_repair import (
                                    beam_repair,
                                )

                                replacement = beam_repair(inner, katex_renders_clean)
                            except Exception as _be:
                                log.debug("beam repair skipped: %s", _be)
                        if replacement is None:
                            replacement = _repair_one(page, cl["bbox"], inner, vl_model)
                        if (
                            replacement
                            and _semantic_retention(inner, replacement)
                            and (page_map_broken or _symbols_agree(inner, replacement))
                        ):
                            line_text = "$$" + replacement + "$$"
                emitted.append((cl["bbox"][1], line_text))
                exts.append((cl["bbox"][0], cl["bbox"][2]))

        # SPAN CONSERVATION (2026-08-27): verify every cluster span made
        # it into the emitted line; re-inject the ones the builder
        # dropped (triple-space gap / unbalanced-frac census families).
        # det51: audit the split parts (see check_clusters above).
        emitted, cons_stats = conservation_pass(check_clusters, emitted)
        if cons_stats.get("repaired") or cons_stats.get("missing_total"):
            record_path_warning(
                "span_repair_applied",
                "cascade dropped spans — conservation pass re-injected "
                f"{cons_stats.get('missing_total', 0)} span(s) across "
                f"{cons_stats.get('repaired', 0)} line(s)",
                page=page_num or None,
                details=cons_stats,
            )

        # INSERT unmatched region equations (absent from font output entirely)
        for ri, re_ in enumerate(region_eqs):
            if ri in used_regions:
                continue
            # skip regions overlapping any emitted cluster (matched implicitly)
            wrap = "$$" if re_["display"] else "$"
            emitted.append((re_["bbox"][1], wrap + re_["latex"] + wrap))
            exts.append((re_["bbox"][0], re_["bbox"][2]))
            used_regions.add(ri)

        if len(exts) == len(emitted):
            order = _column_aware_emitted_order(emitted, exts, page.rect.width)
            return "\n".join(emitted[i][1] for i in order)
        # conservation_pass may have rebuilt `emitted` out of sync with
        # `exts` — fall back to the historical y-sort.
        emitted.sort(key=lambda e: e[0])
        return "\n".join(line for _, line in emitted)
    finally:
        _fmd._PAGE_PHI_DUAL = _saved_phi


def _split_cluster_at_gutter(cl: dict, page_width: float) -> list:
    """det50: split a baseline cluster whose spans straddle the column
    gutter into per-column cluster copies. Gutter rule: an x-gap >= 6pt
    that STRADDLES the page midline (left end below mid, right start
    above it) separates columns at any gap width. Single-column pages
    never carry such gaps mid-line (title word gaps <= 5.5pt), so the
    cluster comes back unchanged (byte-identical output).
    """
    spans = sorted(cl["spans"], key=lambda s: (s["bbox"][0], s["origin"][1]))
    mid = page_width / 2.0
    groups: list[list] = [[spans[0]]] if spans else []
    for prev, cur in zip(spans, spans[1:]):
        gap = cur["bbox"][0] - prev["bbox"][2]
        # A gap that STRADDLES the page midline separates columns at any
        # width (a left column ending early leaves a wide gap whose
        # midpoint sits far from mid — measured y152 cluster 'see Fig. 1.'
        # + 'decimation…' fused under the old midpoint rule).
        if gap >= 6 and prev["bbox"][2] < mid and cur["bbox"][0] > mid:
            groups.append([cur])
        else:
            groups[-1].append(cur)
    if len(groups) <= 1:
        return [cl]
    out = []
    for g in groups:
        c = dict(cl)
        c["spans"] = g
        c["bbox"] = (
            min(s["bbox"][0] for s in g),
            cl["bbox"][1],
            max(s["bbox"][2] for s in g),
            cl["bbox"][3],
        )
        out.append(c)
    return out


def _column_aware_emitted_order(emitted: list, exts: list, page_width: float) -> list:
    """det50: order cascade emissions column-major on two-column pages.

    The historical assembly sorted every emission by y0 alone, which
    interleaves the columns row-major on two-column math pages (measured
    0853d8a0 pg2: 'BRM consists of 2 linear feedback guess the initial
    state...' — left and right column lines merged per visual row, then
    reflow fused them into corrupted paragraphs and split hyphenated
    citations around equation blocks). Full-width clusters (titles,
    section headings) cut the page into horizontal bands; inside each
    band the left column reads fully before the right. Single-column
    pages keep the plain y-sort (byte-identical to the old behavior).

    Returns a list of indices into `emitted` in reading order.
    """
    mid = page_width / 2.0
    crosses = lambda x0, x1: x0 < mid - 6 and x1 > mid + 6
    idx_full = [i for i, (x0, x1) in enumerate(exts) if crosses(x0, x1)]
    idx_left = [i for i, (x0, x1) in enumerate(exts) if x1 <= mid + 6 and not crosses(x0, x1)]
    idx_right = [i for i, (x0, x1) in enumerate(exts) if x0 >= mid - 6 and not crosses(x0, x1)]
    if len(idx_left) < 3 or len(idx_right) < 3:
        return sorted(range(len(emitted)), key=lambda i: emitted[i][0])
    y = lambda i: emitted[i][0]
    order: list = []
    lo = float("-inf")
    for fi in sorted(idx_full, key=y):
        order.extend(sorted((i for i in idx_left if lo <= y(i) < y(fi)), key=y))
        order.extend(sorted((i for i in idx_right if lo <= y(i) < y(fi)), key=y))
        order.append(fi)
        lo = y(fi)
    order.extend(sorted((i for i in idx_left if y(i) >= lo), key=y))
    order.extend(sorted((i for i in idx_right if y(i) >= lo), key=y))
    if len(order) != len(emitted):
        # degenerate partition (straddling extents) — fall back to y-sort
        return sorted(range(len(emitted)), key=lambda i: emitted[i][0])
    return order


def _repair_one(page: Any, bbox: tuple, failed: str, vl_model: Optional[str]) -> Optional[str]:
    """Tier 2 (RapidLaTeX crop) → Tier 3 (VL crop) repair, oracle-gated."""
    png = _crop_cluster_png(page, bbox)
    if not png:
        return None
    try:
        # Tier 2: local ONNX
        try:
            latex, _conf = _get_rapidlatex().recognize(png)
            latex = _clean_latex_shell(latex or "")
            if latex and katex_renders_clean(latex):
                return latex
        except Exception as e:
            log.debug("rapidlatex repair failed: %s", e)

        # Tier 3: VL escalation
        if vl_model:
            vl = _vl_repair(png, vl_model)
            if vl:
                vl = _clean_latex_shell(vl)
                if vl and katex_renders_clean(vl):
                    return vl
        return None
    finally:
        try:
            import os

            os.unlink(png)
        except OSError as e:
            log.debug("_repair_one: OSError degraded: %s", e)


__all__ = ["extract_text_with_cascade", "katex_renders_clean"]
