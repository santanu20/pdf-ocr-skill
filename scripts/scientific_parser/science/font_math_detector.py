"""Font-level math detection for born-digital PDFs.

Root cause of arxiv_math 3% score: pdftotext strips font metadata. Math in
born-digital PDFs is encoded via FONT SELECTION (CMMI10, CMSY10, MSBM10,
MSAM10) + SIZE VARIATION (superscript/subscript), not Unicode. PyMuPDF's
get_text("dict") preserves this metadata.

This module reconstructs LaTeX from font metadata:
  1. Detect math-font spans (CMMI*, CMSY*, CMEX*, MSBM*, MSAM*, EU1*, etc.)
  2. Detect super/subscript via size ratio (< 0.85× line median)
  3. Group adjacent math spans into contiguous $...$ blocks
  4. Convert Unicode chars within math blocks to LaTeX commands
"""

from __future__ import annotations

import re
from typing import Any
import logging

log = logging.getLogger(__name__)

# Fonts that indicate math mode (case-insensitive substring match)
_MATH_FONT_RE = re.compile(
    r"cmmi|cmsy|cmex|msbm|msam|msbs|msbm|eufm|eufrak|eusm|eusb|"
    r"wncy|wncyb|lcirc|math|cmti|cmcsc|"
    r"texgyrepagella-math|texgyreschola-math|libertinusmath|"
    r"stix|noto.*math|latinmodern.*math|"
    r"asana.*math|garamond.*math|bonum.*math|"
    r"scholasmath|pmn.*math|pxmath|txmath|"
    # txfonts/newtx MATH faces (2503.03766: rtxr '=', rtxmi ',', txsy '≥';
    # corpus census: rtxr carries only Ψ/=/space, rtxmi only , < / . —
    # text small-caps rtxsc/rtcxr and text italic rtxi STAY OUT).
    r"rtxr\b|rtxmi|rtxb\b|rtxbmi|txsy|txex|txmia|txbsys|newtxmi"
    r"|mdugm|mdged|mdici|mdbch|fpl|jmn",
    re.IGNORECASE,
)

# Strict math fonts (always wrap in $...$, even single char)
_STRICT_MATH_FONT_RE = re.compile(
    r"cmmi|cmsy|cmex|msbm|msam|msbs|eufm|eufrak|eusm|eusb|"
    r"wncy|stix|mathitalic|mathsymbol|"
    r"rsfs|blackboard|mathextension|mathbb|"
    # Euler-package math fonts (dvips): EURM = math italic (the MAIN
    # variable font on euler pages), EUEX = extension/big-operators —
    # both carry cmex-layout letter slots (08800: EUEX 'Q'=producttext).
    # Without these, euler pages route math through the prose path
    # ('i-thP row: $Q^{\Delta n}...').
    r"eurm|euex"
    r"|rtxr\b|rtxmi|rtxb\b|rtxbmi|txsy|txex|txmia|txbsys|newtxmi",
    re.IGNORECASE,
)

# Font family → LaTeX style command for alphabetic glyphs.
# rsfs (Ralph Smith Formal Script) / Euler Script → \mathcal
# EUFM (Euler Fraktur) → \mathfrak
# MSBM / PazoMathBlackboardBold → \mathbb
_STYLE_FONT_RE = re.compile(r"rsfs|eusm|eusb", re.IGNORECASE)
_FRAK_FONT_RE = re.compile(r"eufm|eufrak", re.IGNORECASE)
# Bold MATH faces (cm-bx / cmmi-b / *Bold*Italic* / MathBold). Only spans
# already routed as math reach this check, so a broad 'bold' is safe —
# prose-bold spans never call _span_to_math_token as math tokens.
_BOLD_MATH_RE = re.compile(r"(?i)bold|cm[a-z]*bx|cmmib")
_BB_FONT_RE = re.compile(r"msbm|blackboard|mathbb", re.IGNORECASE)

# ── CMEX glyph-name decoding (iter-14, 2026-08-17) ──────────────────────
# Embedded CMEX subsets put glyphs at ARBITRARY code slots: measured,
# 'X'(0x58) = summationdisplay on SHMKRY/OSMJCA/FNWGOA subsets, while
# WZGPXL pairs unionsqdisplay (⨆) at 'G'(0x47) and ALSO maps X→\sum.
# The static letter-slot table below is the FALLBACK; when the embedded
# font program is parseable we decode via its own encoding → glyph name
# → LaTeX (ground truth, per document).
_CMEX_GLYPH_LATEX = {
    "summationtext": r"\sum ", "summationdisplay": r"\sum ",
    "producttext": r"\prod ", "productdisplay": r"\prod ",
    "coproducttext": r"\coprod ", "coproductdisplay": r"\coprod ",
    # Integral signs (iter-102, 2503.09086_pg5): the embedded CFF maps
    # slot 90 ('Z' via ToUnicode) to integraldisplay / slot 82 to
    # integraltext — without these the ∫ decodes as the LETTER Z and
    # the equation renders 'Z ( 1 |∇U…'.
    "integraltext": r"\int ", "integraldisplay": r"\int ",
    "integralbig": r"\int ", "integralBig": r"\int ",
    "integralbigg": r"\int ", "integralBigg": r"\int ",
    "tildewide": "",  # wide accent decoration — dropping beats garbage
    "contourintegraltext": r"\oint ", "contourintegraldisplay": r"\oint ",
    "unionsqtext": r"\bigsqcup ", "unionsqdisplay": r"\bigsqcup ",
    "uniontext": r"\bigcup ", "uniondisplay": r"\bigcup ",
    "tildewide": "˜",  # wide accent — ACCENT-GLYPH TEXT REWRITE routes it
    "circleplustext": r"\bigoplus ", "circleplusdisplay": r"\bigoplus ",
    "circletimestext": r"\bigotimes ", "circletimedisplay": r"\bigotimes ",
    "circlecdottext": r"\bigodot ", "circlecdotdisplay": r"\bigodot ",
    "logicalandtext": r"\bigwedge ", "logicalanddisplay": r"\bigwedge ",
    "logicalortext": r"\bigvee ", "logicalordisplay": r"\bigvee ",
    # Radical signs (iter-99, 2503.07226_pg7): J_0(√|β|r) — the CMEX10
    # 'p' slot (0x70) is radicalbig; ToUnicode emits the LETTER so the
    # radical decoded as italic 'p'. Size glyphs decode to \sqrt (the
    # flush-run radicand pass braces the operand); extensible PIECES
    # (bt/tp/vertex) drop like tildewide — a piece alone is not a sign.
    "radicalbig": r"\sqrt ", "radicalBig": r"\sqrt ",
    "radicalbigg": r"\sqrt ", "radicalBigg": r"\sqrt ",
    "radicalbt": "", "radicaltp": "", "radicalvertex": "",
}

# {span-font-name: {char: LaTeX}} for the document being extracted.
# Set/cleared by extract_text_with_math_fonts (single-threaded pipeline).
_ACTIVE_CMEX_GLYPHS: dict[str, dict[str, str]] = {}


# Phi-twin page dialect (iter-18): True when the page's math spans carry
# BOTH U+03C6 and U+03D5 — then the char-map distinction stands; the
# blanket \varphi rule is suspended.
_PAGE_PHI_DUAL: bool = False


def _detect_phi_dual(page: Any) -> bool:
    try:
        d = page.get_text("dict")
    except Exception as e:
        log.debug("phi dual detect failed: %s", e)
        return False
    seen = set()
    for b in d.get("blocks", []):
        if b.get("type", 0) != 0:
            continue
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                if re.search(r"(?i)cmmi|mathitalic|math.?it", s.get("font", "")):
                    t = s.get("text", "")
                    if "ϕ" in t:
                        seen.add(0x3D5)
                    if "φ" in t:
                        seen.add(0x3C6)
    return len(seen) == 2


def load_cmex_glyph_map(page: Any) -> dict[str, dict[str, str]]:
    """span-font-name → {char → LaTeX}, from embedded CMEX font programs.

    Reads each font's built-in /Encoding (PFA/PFB cleartext regex, or the
    CFF Encoding array via fontTools when installed). The PDF layer's
    /Differences is absent on these dvips subsets, so the font program is
    the only authoritative code→glyph-name source. Unreadable fonts are
    skipped — callers fall back to the static letter-slot table.
    """
    out: dict[str, dict[str, str]] = {}
    try:
        doc = page.parent
        fonts = page.get_fonts(full=True)
    except Exception as e:
        log.debug("cmex glyph map: page introspection failed: %s", e)
        return out
    for f in fonts:
        try:
            xref, basefont = f[0], f[3] or ""
            # CMEX, EUEX (Euler extension) and TXEX (txfonts extension)
            # share the big-operator/radical slot layout; all carry
            # arbitrary subset slots decoded from the embedded program
            # (iter-27: 'Q'=producttext on 08800 EUEX10; iter-99: 'p'=
            # radicalbig on 07226 CMEX10).
            if not re.search(r"CMEX|EUEX|TXEX", basefont.upper()):
                continue
            _, fext, _, buf = doc.extract_font(xref)
            code2glyph: dict[int, str] = {}
            if fext in ("pfa", "pfb"):
                if fext == "pfb":
                    raw = bytearray()
                    i = 0
                    while i < len(buf):
                        if buf[i] == 0x80:
                            n = int.from_bytes(buf[i + 2:i + 6], "little")
                            raw += buf[i + 6:i + 6 + n]
                            i += 6 + n
                        else:
                            break
                    text = bytes(raw).decode("latin-1", "replace")
                else:
                    text = buf.decode("latin-1", "replace")
                head = text.split("eexec", 1)[0]
                for code, gname in re.findall(r"dup\s+(\d+)\s+/([\w.]+)\s+put", head):
                    code2glyph[int(code)] = gname
            elif fext == "cff":
                from io import BytesIO

                from fontTools.cffLib import CFFFontSet

                top = CFFFontSet()
                top.decompile(BytesIO(buf), None)
                enc = top.topDictIndex[0].Encoding  # list: code → glyph name
                for code in range(min(len(enc), 0x80)):
                    gname = enc[code]
                    if gname and gname != ".notdef":
                        code2glyph[code] = gname
            cmap = {}
            for code, gname in code2glyph.items():
                latex = _CMEX_GLYPH_LATEX.get(gname)
                if latex is not None and 0x20 <= code < 0x80:
                    cmap[chr(code)] = latex
            if cmap:
                suffix = basefont.split("+", 1)[-1]
                out[suffix] = cmap
                out[basefont] = cmap
        except Exception as e:
            log.debug("cmex glyph map: skip font %s: %s", (f[3:4] or ["?"])[0], e)
    return out


# Operand-ish spans absorbable INSIDE an open math run. Digits, operators,
# brackets, single letters (possibly spaced: "D ( −", ") = H") absorb —
# multi-letter words ("of", "is", "where") are prose and CLOSE the run.
# Rule: no 2+ consecutive letters anywhere in the span.
def _is_operand_span(t: str) -> bool:
    if not t or re.search(r"[A-Za-z]{2,}", t):
        return False
    # '#' (cardinality \#S, primorial) is math furniture, never prose.
    return bool(re.match(r"^[\d\s+\-*/=<>|.,:;()\[\]'′#A-Za-z]+$", t))


# Function-name-like letter runs that stay INSIDE math (upright operators
# typeset in math fonts). Anything else ≥3 letters = prose → split out.
# Roman-font function/word spans that may sit INSIDE a math run. When a
# ROMAN font (CMR/LMRoman/…, i.e. not a math font) carries one of these
# while a run is open, it is \text{…}/\operatorname content (gold uses
# \text{Hom}, \text{iff}, …) — absorbing it as bare letters changes the
# render (upright vs italic) and breaks render-compare. Measured on
# arxiv_math bench 2026-08-16 (Hom/Fix/iff/when families).
_TEXT_IN_MATH_WORDS = {
    # algebra/topology/analysis operators
    "Hom",
    "End",
    "Ext",
    "Tor",
    "Fix",
    "Spec",
    "Supp",
    "supp",
    "span",
    "Span",
    "rank",
    "tr",
    "Trace",
    "det",
    "diag",
    "Res",
    "Ad",
    "Sym",
    "Alt",
    "Aut",
    "Gal",
    "Pic",
    "CH",
    "GL",
    "SL",
    "Sp",
    "Isom",
    "Dist",
    "dist",
    "sign",
    "sgn",
    "im",
    "Ind",
    "ord",
    "orb",
    "stab",
    "Fix",
    "coker",
    "Coker",
    "conv",
    "spt",
    "cl",
    "clint",
    "Arg",
    "Lag",
    "Tr",
    "Cor",
    "Var",
    # logic connectives set in roman inside math
    "iff",
    "when",
    "then",
    "where",
    "and",
    "or",
    "if",
    "for",
    "not",
    "so",
    "such",
    "that",
    "all",
    "some",
    "every",
    "no",
    "else",
    "since",
    "hence",
    "with",
    "without",
    "under",
    "over",
    "between",
    "against",
    "upon",
    "resp",
    "arg",
    "id",
    "Eff",
}

# Roman function words gold renders as LaTeX COMMANDS (measured:
# \deg \dim \max \min \ln in gold; 'arg'/'id' stay \text{...}).
_FUNC_CMD_WORDS = {
    "deg", "dim", "max", "min", "ln", "exp", "sup", "inf",
    "det", "gcd", "lcm", "ker", "Pr",
}

_MATH_WORDS = {
    "sin",
    "cos",
    "tan",
    "log",
    "ln",
    "exp",
    "max",
    "min",
    "lim",
    "sup",
    "inf",
    "det",
    "dim",
    "arg",
    "gcd",
    "lcm",
    "mod",
    "deg",
    "ker",
    "hom",
    "sgn",
    "erf",
    "Var",
    "Cov",
    "Re",
    "Im",
    "id",
    "ord",
    "rank",
    "spec",
}


def _split_prose_in_math(text: str) -> list[tuple[str, bool]]:
    """Split a math-font span into (fragment, is_prose) parts.

    CMMI spans in heavy-math PDFs carry glued prose ("βandγ", "ΓofIsom(").
    Letter runs ≥3 that are not math function names are prose. Runs of 1-2
    letters stay math (variables, "of"-style connectors render fine).
    """
    parts: list[tuple[str, bool]] = []
    for seg in re.split(r"([A-Za-z]{3,})", text):
        if not seg:
            continue
        if re.fullmatch(r"[A-Za-z]{3,}", seg) and seg not in _MATH_WORDS:
            parts.append((seg, True))
        else:
            parts.append((seg, False))
    return parts


# Combining long solidus overlay (U+0338): PDFs encode ∉ as ∈+̸ or ̸+∈.
_NEGATED_MAP = {
    # \not\in (composed) — NOT \notin: gold MathML containment matches
    # the composed form (11 golds vs 4); both render identically.
    r"\in": r"\not\in",
    r"=": r"\neq",
    r"<": r"\nless",
    r">": r"\ngtr",
    r"\leq": r"\nleq",
    r"\geq": r"\ngeq",
    r"\subset": r"\nsubset",
    r"\exists": r"\nexists",
}


def _bbox_overlap(a: tuple, b: tuple, eps: float = 0.5) -> bool:
    """True when two bboxes intersect horizontally+vertically (> eps)."""
    return a[0] < b[2] - eps and b[0] < a[2] - eps and a[1] < b[3] - eps and b[1] < a[3] - eps

def collect_math_clusters(page: Any) -> list[dict]:
    """Cluster page spans into visual lines with display-math stack merging.

    (v5, 2026-08-14) Three stages:
      1. Band clustering (v4): spans whose y-centers sit within 0.45×size
         share a visual line — reassembles fitz's fragmented equations.
      2. ATTACH-DON'T-SEPARATE: an all-small cluster (Σ/max/min limit line,
         7pt) vertically adjacent to a body-size cluster is MERGED into it —
         its spans become scripts of the x-nearest token (handles
         interleaved ``max_λ min_θ`` stacks; evidence: limit lines at
         y=438/587 on 2503.09086_pg5).
      3. FRAC PAIRING: a small span ABOVE the baseline x-aligned with a
         small span BELOW, with no body-size span between them at that x,
         is a stacked fraction → synthetic ``\\frac`` span (evidence:
         ``\\frac{1}{X(∂Ω)}`` gold rendering).

    Returns list of cluster dicts: {spans: [...], bbox: (x0,y0,x1,y1)}.
    """
    d = page.get_text("dict")
    all_spans: list[dict] = []
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            # ROTATED-TEXT veto (iter-91, 2503.06228_pg1): sidebar
            # stamps ('arXiv:2503.06228v1 …', dir (0,-1)) carry a
            # page-tall bbox that makes them a page-spanning "main"
            # cluster — display-equation limits glue into it
            # ('\frac{n}{n \prod =0}' garbage) and the operator's own
            # line loses its limits. Rotated text is never horizontal
            # math; exclude it from clustering entirely.
            _dir = line.get("dir") or (1.0, 0.0)
            if abs(_dir[0]) < 0.98:
                continue
            for s in line.get("spans", []):
                if s.get("text", "").strip():
                    all_spans.append(s)
    if not all_spans:
        return []
    # ACCENT-GLYPH TEXT REWRITE (iter-102, 09086): CMEX tildewide
    # decodes to the ˜ accent char but its ToUnicode text is a LETTER
    # ('e') — the accent pairing machinery matches on span TEXT.
    # Rewrite span text to ˜ so _pair_accents can wrap its x-overlapped
    # base into \tilde{…} (gold \widetilde{U} renders the same).
    for s in all_spans:
        if _ACTIVE_CMEX_GLYPHS.get(s.get("font", ""), {}).get(
            s.get("text", ""), ""
        ) == "˜":
            s["text"] = "˜"
    # VERTICAL-RULE COLLAPSE (iter-116, 2503.08031 pg36): symbol fonts
    # (MnSymbol…) build tall delimiters by STACKING identical glyph
    # pieces vertically — the text layer emits 'RRRRRRRRRRR' (eleven
    # pieces of one ToUnicode'd char; measured span 3.3pt wide ×
    # 64.9pt tall). A tall-thin span (height > 5× width) of ≥4
    # identical chars is a stacked vertical bar: rewrite to '|'.
    # Vertical bars are THE stacked delimiter family in these fonts —
    # parens/brackets use single bigger glyphs, not stacked pieces.
    for s in all_spans:
        t = s.get("text", "").strip()
        if len(t) >= 4 and len(set(t)) == 1:
            w = s["bbox"][2] - s["bbox"][0]
            h = s["bbox"][3] - s["bbox"][1]
            if w > 0.5 and h > 5 * w:
                s["text"] = "|"


    import os as _os
    # evaluation-bar glyph as '.' — glyph-box discrimination is
    # IMPOSSIBLE via rawdict (font-shaped boxes: real periods measure
    # w=3.03 h=10.91 identically, measured 04993 pg18). The true signal
    # is CONTEXT: \\Big| carries its subscript to the LOWER-LEFT, so a
    # script-size span abutting the '.' from the left at sub-band marks
    # the evaluation bar. Kill-switch: PDF_OCR_BIGBAR=0.
    if _os.environ.get("PDF_OCR_BIGBAR", "0") == "1":
        _body_sz = max((x.get("size") or 0) for x in all_spans)
        for s in list(all_spans):
            if set(s.get("text") or "") != {"."} or not _MATH_FONT_RE.search(
                s.get("font", "")
            ):
                continue
            sb = s["bbox"]
            sub = [
                q
                for q in all_spans
                if q is not s
                and q.get("text", "").strip()
                and q.get("size", 0) < 0.85 * _body_sz
                and 0 <= sb[0] - q["bbox"][2] <= 1.2 * _body_sz
                and q["bbox"][1] >= sb[1] + 0.25 * (sb[3] - sb[1])
            ]
            if sub:
                # Place the scripts just RIGHT of the bar: the line
                # builder's proven sub-band accumulation then binds
                # them to the bar token (text-folding fights the
                # brace-escaper: '\Big|_\{ =0...' measured).
                _x_shift = sb[2] - sub[0]["bbox"][0] + 0.5
                for q in sub:
                    q["bbox"] = (
                        q["bbox"][0] + _x_shift,
                        q["bbox"][1],
                        q["bbox"][2] + _x_shift,
                        q["bbox"][3],
                    )
                s["text"] = "│"
    # (v6, iter-68) The rawdict fetch below doubles as GLYPH EVIDENCE
    # source: TeX flattens script-of-script to the SAME size
    # (P_{\Delta_n}: '∆n' is one span, both glyphs 5.98pt — measured
    # 08800_pg9), so span-level size/origin cannot see the nesting.
    # Char origins carry the truth: n drops 0.96pt = 0.16×size below ∆,
    # x-tight, on EVERY instance. Attach per-glyph origins to small
    # spans; the line builder composes intra-span scripts from them.
    _raw = None
    # Origin correction (iter-26): MuPDF merges a kerning SPACE into the
    # following script glyph as one dict span and reports the SPACE's
    # baseline as the span origin — ' 2' raised ~4pt reads dy=+0.0
    # (2503.05506_pg11: sup '2' at 615.41 behind a space at 618.15).
    # Every downstream dy decision inherits the lie. Fix at the source:
    # for whitespace-padded spans, take the first/last non-space GLYPH
    # origin from rawdict char data.
    if any(s["text"] != s["text"].strip() for s in all_spans):
        try:
            _raw = page.get_text("rawdict")
            _raw_spans = [
                sp
                for bl in _raw.get("blocks", [])
                if bl.get("type", 0) == 0
                for ln in bl.get("lines", [])
                for sp in ln.get("spans", [])
            ]
            for s in all_spans:
                t = s["text"]
                if t == t.strip():
                    continue
                lead = len(t) - len(t.lstrip())
                chars = [
                    c
                    for sp in _raw_spans
                    if sp.get("font") == s.get("font")
                    and abs(sp.get("size", 0) - s.get("size", 0)) < 0.05
                    and _bbox_overlap(sp["bbox"], s["bbox"])
                    for c in sp.get("chars", [])
                ]
                chars.sort(key=lambda c: c["origin"][0])
                glyphs = [c for c in chars if c["c"].strip()]
                if not glyphs:
                    continue
                pick = glyphs[0] if lead else glyphs[-1]
                s["origin"] = tuple(pick["origin"])
        except Exception as e:  # infra failure must be loud, never silent
            log.warning("rawdict origin correction skipped: %s", e)

    # GLYPH EVIDENCE (iter-68): per-glyph origins for small spans whose
    # chars show an intra-span script drop/rise. Gates: span 2-8 chars,
    # no whitespace, size < 12 (script/inline context).
    try:
        _need = [
            s
            for s in all_spans
            if 2 <= len(s["text"].strip()) <= 8
            and not any(ch.isspace() for ch in s["text"])
            and s.get("size", 10.0) < 12.0
        ]
        if _need:
            if _raw is None:
                _raw = page.get_text("rawdict")
            _raw_spans2 = [
                sp
                for bl in _raw.get("blocks", [])
                if bl.get("type", 0) == 0
                for ln in bl.get("lines", [])
                for sp in ln.get("spans", [])
            ]
            for s in _need:
                chars = [
                    c
                    for sp in _raw_spans2
                    if sp.get("font") == s.get("font")
                    and abs(sp.get("size", 0) - s.get("size", 0)) < 0.05
                    and _bbox_overlap(sp["bbox"], s["bbox"])
                    for c in sp.get("chars", [])
                ]
                chars.sort(key=lambda c: c["origin"][0])
                glyphs = [c for c in chars if c["c"].strip()]
                if len(glyphs) < 2 or len(glyphs) != len(s["text"].strip()):
                    continue
                y0 = glyphs[0]["origin"][1]
                sz = s.get("size", 10.0)
                if any(
                    abs(c["origin"][1] - y0) > 0.11 * sz for c in glyphs[1:]
                ):
                    s["_glyphs"] = [
                        (c["c"], tuple(c["origin"]), tuple(c["bbox"])) for c in glyphs
                    ]
    except Exception as e:
        log.warning("glyph evidence skipped: %s", e)

    _BIGOPS = {
        "\\sum", "\\prod", "\\int", "\\oint", "\\coprod", "\\bigsqcup",
        "\\bigcup", "\\bigcap", "\\bigoplus", "\\bigotimes", "\\bigodot",
        "\\bigvee", "\\bigwedge", "\\biguplus",
    }

    # Stage 1: band clustering
    spans_sorted = sorted(all_spans, key=lambda s: (s["bbox"][1] + s["bbox"][3]) / 2)
    clusters: list[list[dict]] = []
    for s in spans_sorted:
        cy = (s["bbox"][1] + s["bbox"][3]) / 2
        placed = False
        _is_decoded_bigop = bool(
            _ACTIVE_CMEX_GLYPHS.get(s.get("font", ""), {}).get(
                s.get("text", ""), ""
            ).strip() in _BIGOPS
        )
        if not _is_decoded_bigop:
            for cl in clusters:
                _cl_max = max(c2["size"] for c2 in cl)
                body = [c for c in cl if c["size"] >= _cl_max * 0.8]
                ref = body[-1] if body else cl[-1]
                ref_cy = (ref["bbox"][1] + ref["bbox"][3]) / 2
                tol = max(ref["size"], s["size"]) * 0.45
                joins = abs(cy - ref_cy) <= tol
                # NOTE (2026-08-27 iter-116 investigation): a BASELINE-
                # BAND join for tall delimiters (MnSymbol/CMEX parens
                # sharing the body baseline, y-center ~half-glyph above)
                # was prototyped and measured: it recovers display-eq
                # parens (2503.08031 pg36 h(x,x')) but perturbs
                # stacked-frac pairing on passing pages (2503.09086 pg5
                # \frac{1}{2}|∇Ũ|² flattened to '( 2 1 |' — recompose-
                # winners, iter-59 class). Needs cluster-composition-
                # aware gating before landing; see MEMORY iter-116.
                if abs(cy - ref_cy) <= tol:
                    cl.append(s)
                    placed = True
                    break
        if _is_decoded_bigop:
            # BIG-OP LINE-THROUGH merge (iter-91, 2503.06228_pg1): a
            # decoded display big-op (∏ ∑ — CMEX) hangs ~2 body-heights
            # BELOW its line baseline, so band-tolerance splits it from
            # its own line. The owning line passes THROUGH the glyph
            # (baseline inside the bbox y-range); stacked candidates
            # (frac numerator above the main line) — pick by CAP ALIGNMENT:
            # the glyph TOP aligns with the owning line's cap height,
            # baseline ≈ glyph_top + one body size (measured ∑ top
            # 403.9 + 11.96 = 415.86 vs p-line baseline 415.78 — 0.08pt
            # apart; the (q;q) denominator line misses by 8.2pt). A
            # center-distance metric mis-picks the DENOMINATOR line
            # because the glyph hangs 2 body-heights below its own.
            _best = None
            _best_gap = None
            for cl in clusters:
                _cl_max = max(c2["size"] for c2 in cl)
                body = [c for c in cl if c["size"] >= _cl_max * 0.8]
                if not body:
                    continue
                _b_org = sorted(c["origin"][1] for c in body)
                _b_med = _b_org[len(_b_org) // 2]
                if s["bbox"][1] <= _b_med <= s["bbox"][3]:
                    _d = abs(_b_med - (s["bbox"][1] + s["size"]))
                    # X-GAP preference (iter-102, 09086 ∫): a through-line
                    # whose body span CROSSES the glyph's x-range is a
                    # different line (prose above/below); the OWNING line
                    # has a gap at the glyph (its spans end left, resume
                    # right). Prefer gap-lines; cap alignment breaks ties.
                    _crosses = any(
                        c["bbox"][0] < s["bbox"][2] - 1.0
                        and c["bbox"][2] > s["bbox"][0] + 1.0
                        for c in body
                        if c is not s
                    )
                    if _best is None or _d < _best[0]:
                        _best = (_d, cl)
                    if not _crosses and (
                        _best_gap is None or _d < _best_gap[0]
                    ):
                        _best_gap = (_d, cl)
            if _best_gap is not None:
                _best = _best_gap
            if _best is not None:
                _best[1].append(s)
                placed = True
        if not placed:
            clusters.append([s])

    # Stage 1.5: SCRIPT X-GAP REASSIGNMENT (det55/det58): a
    # superscript's y-center rides ABOVE its line's baseline band, so
    # y-center band clustering chains it to the PREVIOUS line. Geometry
    # that discriminates: the owning line has a TIGHT x-gap at the
    # script; the wrong line's nearest body span is a half-column away.
    # Move a script-size span when its current cluster's nearest body
    # x-gap is huge (> 3x body size) and another same-pitch cluster
    # offers a tight gap (<= 2x body size).
    # GATE (det58, 2503.06111 pg6): never move into/out of a
    # broken-ToUnicode cluster (Hangul jamo / PUA glyphs). Those pages
    # route through the cascade's broken-map repair lane where cluster
    # composition churn only re-rolls which garbage emission wins
    # (measured: 11/13 moves on 06111_pg6 blocked, 0/52 on the 11
    # recovery pages).
    _broken_map_re = re.compile(
        "[\u1100-\u11ff\u3130-\u318f\uac00-\ud7a3\ue000-\uf8ff]"
    )

    def _nearest_x_gap(cl, s):
        _cl_max = max(c2["size"] for c2 in cl)
        body = [c for c in cl if c is not s and c["size"] >= _cl_max * 0.8]
        if not body:
            return None
        g = min(
            min(abs(c["bbox"][0] - s["bbox"][2]), abs(s["bbox"][0] - c["bbox"][2])) for c in body
        )
        return g

    for cl in clusters:
        _cl_max = max(c2["size"] for c2 in cl)
        body = [c for c in cl if c["size"] >= _cl_max * 0.8]
        if not body:
            continue  # all-small cluster: Stage 2/3 own it
        for s in [c for c in cl if c["size"] < _cl_max * 0.8]:
            cur_gap = _nearest_x_gap(cl, s)
            if cur_gap is None or cur_gap <= 3.0 * _cl_max:
                continue
            _ref_cy = (body[-1]["bbox"][1] + body[-1]["bbox"][3]) / 2
            best = None
            for cl2 in clusters:
                if cl2 is cl:
                    continue
                _m2 = max(c2["size"] for c2 in cl2)
                body2 = [c for c in cl2 if c["size"] >= _m2 * 0.8]
                if not body2:
                    continue
                # same-or-adjacent line: y-band within ~1.4 line pitches
                _c2cy = (body2[-1]["bbox"][1] + body2[-1]["bbox"][3]) / 2
                if abs(_c2cy - _ref_cy) > 1.4 * _cl_max + _m2:
                    continue
                # BASELINE DY BAND (04147 regression): a script's origin
                # sits within ~half a body size of ITS OWN line's
                # baseline (TeX raises/lowers scripts by ~0.35x). A
                # candidate a full line-pitch away fails this even when
                # its x-gap is tight -- display-eq tokens (J_{N,1} with
                # \ddots spacing) were being stolen by the dense line
                # below. Veto: script origin must lie within +-0.55x
                # body of the candidate's median body baseline.
                _b2_org = sorted(c["origin"][1] for c in body2)
                _b2_med = _b2_org[len(_b2_org) // 2]
                if abs(s["origin"][1] - _b2_med) > 0.55 * _m2:
                    continue
                # BROKEN-MAP GATE (det58): clusters whose text carries
                # Hangul jamo or PUA glyphs are broken-ToUnicode
                # artifacts; reassigning their fragments is churn.
                if _broken_map_re.search("".join(c.get("text", "") for c in cl2)) or _broken_map_re.search(
                    "".join(c.get("text", "") for c in cl)
                ):
                    continue
                g2 = _nearest_x_gap(cl2, s)
                if g2 is None or g2 > 2.0 * _m2:
                    continue
                if s["size"] >= _m2 * 0.8:
                    continue  # must stay script-size in target too
                if best is None or g2 < best[0]:
                    best = (g2, cl2)
            if best is not None:
                cl.remove(s)
                best[1].append(s)

    # Stage 2: attach-don't-separate
    def _meta(cl):
        sizes = sorted(s["size"] for s in cl)
        body = sizes[len(sizes) // 2]
        ys = [s["bbox"][1] for s in cl] + [s["bbox"][3] for s in cl]
        xs = [s["bbox"][0] for s in cl] + [s["bbox"][2] for s in cl]
        return {
            "spans": cl,
            "body": body,
            "x0": min(xs),
            "x1": max(xs),
            "y0": min(ys),
            "y1": max(ys),
            "all_small": all(
                s["size"] < body * 1.01
                and s["size"] < max(sizes) * 0.8
                or s["size"] < max(sizes) * 0.8
                for s in cl
            ),
        }

    metas = sorted((_meta(c) for c in clusters), key=lambda m: m["y0"])
    # Page body reference (iter-26): sub-sub script glyphs cluster into
    # UNIFORM-size y-bands (the three '_i' at 6pt on 05506) — an
    # intra-cluster ratio test can never classify them small (nothing is
    # < 0.8×max of itself), so they classified as MAINS and their glyphs
    # silently never attached to any line. Recover them by distributing
    # EACH SPAN to the main whose baseline band contains it — never via
    # the cluster-level smalls path (that routes whole clusters through
    # tiny-stack merging and cross-line attach scoring, mixing two
    # lines' scripts into one cluster, measured 05506).
    _pw = sorted(
        s["size"] for s in all_spans for _ in range(max(len(s["text"].strip()), 1))
    )
    page_body_ref = _pw[len(_pw) // 2] if _pw else 10.0
    metas = [m for m in metas if m["spans"]]
    smalls = [m for m in metas if m["all_small"]]
    mains = [m for m in metas if not m["all_small"]]
    # Stage 2.55: BIG-OPERATOR SUBSCRIPT consumption (iter-15, 2026-08-17).
    # A script-size mathy cluster centered under / right-of a glyph-decoded
    # display big-op (∑ ∏ ∫ ⨆ …) is that operator's SUBSCRIPT — e.g.
    # \sum_{x∈X(Ω)}. Previously such clusters attached to the nearest main
    # by raw gap, which the FOLLOWING PROSE LINE often wins (measured
    # 09086: 'x∈X(Ω)' stolen by the text line below the equation), so
    # display sums lost their limits entirely.
    # NOTE: uniformly script-size limit clusters classify as MAINS (the
    # all_small test needs size < max*0.8 within the cluster) — so the
    # scan covers ALL metas, gated on every span being script-size vs
    # the OPERATOR's size. "Below" is judged against the OP's CLUSTER
    # body baseline (a display ∑'s own origin sits at the glyph bottom
    # and misjudges both limits; measured 06570 sup 'b(X)' leaking in).
    def _mathy(s) -> bool:
        # Strict math fonts plus roman operand spans (CMR7 '(∂Ω)',
        # CMBX7 'x' — bold vectors): no 3+ lowercase run = equation
        # furniture, not prose.
        if _STRICT_MATH_FONT_RE.search(s.get("font", "")):
            return True
        return not re.search(r"[a-z]{3,}", s.get("text", ""))

    if _ACTIVE_CMEX_GLYPHS:
        for mn in list(mains):
            if not mn["spans"]:
                continue
            sizes_c = sorted(s["size"] for s in mn["spans"])
            body_c = sizes_c[len(sizes_c) // 2]
            orgs_c = sorted(
                s["origin"][1] for s in mn["spans"] if s["size"] >= body_c * 0.9
            )
            base_c = orgs_c[len(orgs_c) // 2] if orgs_c else body_c
            # (DISPLAY GLYPH ORIGIN LIE — iter-91 — is corrected PER-OP
            ops = [
                s for s in mn["spans"]
                if _ACTIVE_CMEX_GLYPHS.get(s.get("font", ""), {}).get(
                    s.get("text", ""), ""
                ).strip() in _BIGOPS
            ]
            for op in ops:
                ox0, ox1 = op["bbox"][0], op["bbox"][2]
                ow = max(ox1 - ox0, 1.0)
                for meta in metas:
                    if meta is mn or not meta["spans"]:
                        continue
                    if any(s["size"] >= op["size"] * 0.85 for s in meta["spans"]):
                        continue  # body-size cluster: not a limit line
                    w_tot = sum(
                        s["bbox"][2] - s["bbox"][0] for s in meta["spans"]
                    )
                    w_math = sum(
                        s["bbox"][2] - s["bbox"][0]
                        for s in meta["spans"]
                        if _mathy(s)
                    )
                    if w_math / w_tot < 0.6:
                        continue  # footnote markers / prose stays out
                    sspans = sorted(meta["spans"], key=lambda s: s["bbox"][0])
                    groups: list[list] = []
                    for s in sspans:
                        if (
                            groups
                            and s["bbox"][0] - groups[-1][-1]["bbox"][2] <= 2.0 * ow
                        ):
                            groups[-1].append(s)
                        else:
                            groups.append([s])
                    for g in groups:
                        cx = (g[0]["bbox"][0] + g[-1]["bbox"][2]) / 2
                        if not (ox0 - 0.5 * ow <= cx <= ox1 + 2.5 * ow):
                            continue
                        gmin = min(s["origin"][1] for s in g)
                        gmax = max(s["origin"][1] for s in g)
                        if not (
                            base_c + 0.15 * body_c
                            <= gmin
                            < base_c + 2.0 * body_c
                        ):
                            # SUP limit (iter-91, 2503.06228_pg1): a
                            # display big-op's UPPER limit (∞ over ∏)
                            # sits in a band ABOVE the glyph-bottom
                            # baseline. Without consumption here, FRAC
                            # PAIRING steals it ('\frac{\infty (1}{n
                            # \prod =0}'). Tighter x-window than sub:
                            # the exponent of a LEFT neighbor must not
                            # be taken.
                            if (
                                ox0 - 0.5 * ow <= cx <= ox1 + 1.0 * ow
                                and base_c - 2.0 * body_c
                                < gmax
                                < base_c - 0.10 * body_c
                            ):
                                op["_opsup_spans"] = (
                                    op.get("_opsup_spans", []) + list(g)
                                )
                                gids = {id(s) for s in g}
                                meta["spans"] = [
                                    s
                                    for s in meta["spans"]
                                    if id(s) not in gids
                                ]
                            continue
                        op["_opsub_spans"] = op.get("_opsub_spans", []) + list(g)
                        gids = {id(s) for s in g}
                        meta["spans"] = [
                            s for s in meta["spans"] if id(s) not in gids
                        ]
                    if not meta["spans"]:
                        if meta in smalls:
                            smalls.remove(meta)
                        if meta in mains:
                            mains.remove(meta)
                    else:
                        xs = [s["bbox"][0] for s in meta["spans"]] + [
                            s["bbox"][2] for s in meta["spans"]
                        ]
                        ys = [s["bbox"][1] for s in meta["spans"]] + [
                            s["bbox"][3] for s in meta["spans"]
                        ]
                        meta["x0"], meta["x1"], meta["y0"], meta["y1"] = (
                            min(xs),
                            max(xs),
                            min(ys),
                            max(ys),
                        )
            # Roman FUNC_CMD ops (\max \min): uniform script-size limit
            # clusters classify as MAINS (all_small needs size < max*0.8
            # within-cluster — never true when all spans are 7pt), so the
            # \max_\lambda \min_\theta limit pair sat in its OWN cluster.
            # Attach each small span x-centered under its word directly.
            for op in [
                s for s in mn["spans"]
                if s.get("size", 0) >= body_c * 0.9
                and s.get("text", "").strip() in _FUNC_CMD_WORDS
            ]:
                ow = max(op["bbox"][2] - op["bbox"][0], 1.0)
                ox0, ox1 = op["bbox"][0], op["bbox"][2]
                for meta in metas:
                    if meta is mn or not meta["spans"]:
                        continue
                    if any(
                        s["size"] >= op["size"] * 0.85 for s in meta["spans"]
                    ):
                        continue
                    taken = [
                        s for s in meta["spans"]
                        if s.get("text", "").strip()
                        and _mathy(s)
                        and ox0 - 0.3 * ow
                        <= (s["bbox"][0] + s["bbox"][2]) / 2
                        <= ox1 + 0.3 * ow
                        and base_c + 0.10 * body_c
                        < s["origin"][1]
                        < base_c + 2.0 * body_c
                    ]
                    if not taken:
                        continue
                    op["_opsub_spans"] = op.get("_opsub_spans", []) + taken
                    tids = {id(s) for s in taken}
                    meta["spans"] = [
                        s for s in meta["spans"] if id(s) not in tids
                    ]
                    if not meta["spans"]:
                        if meta in smalls:
                            smalls.remove(meta)
                        if meta in mains:
                            mains.remove(meta)
    # Stage 2.7: TINY VERTICAL-STACK pre-merge (iter-15). Band clustering
    # isolates the parts of a display \frac{1}{2} as TWO single-span
    # clusters ('1' at y470-482, '2' at y486-498 beside the equation line
    # at 487 — measured 06570). No stage ever re-joins them, so the frac
    # never forms. Pair x-aligned tiny mathy clusters stacked within a
    # body; the lower becomes den_spans so the existing point-frac
    # assembly emits \frac{num}{den}. The pair then attaches to its
    # equation cluster via the normal stage-2 loop.
    def _tiny(m) -> bool:
        if not m["spans"] or len(m["spans"]) > 2:
            return False
        w = m["x1"] - m["x0"]
        sz = max(s["size"] for s in m["spans"])
        return w <= 3.5 * sz and all(
            _STRICT_MATH_FONT_RE.search(s.get("font", ""))
            or not re.search(r"[a-z]{3,}", s.get("text", ""))
            for s in m["spans"]
        )

    tinies = [m for m in metas if m["spans"] and _tiny(m)]
    merged_tiny_ids: set[int] = set()
    for ai in range(len(tinies)):
        A = tinies[ai]
        if id(A) in merged_tiny_ids or not A["spans"]:
            continue
        acx = (A["x0"] + A["x1"]) / 2
        for bj in range(ai + 1, len(tinies)):
            B = tinies[bj]
            if id(B) in merged_tiny_ids or not B["spans"]:
                continue
            bcx = (B["x0"] + B["x1"]) / 2
            if abs(acx - bcx) > 0.8 * max(A["x1"] - A["x0"], B["x1"] - B["x0"], 2.0):
                continue
            top, bot = (A, B) if A["y0"] <= B["y0"] else (B, A)
            sz = max(s["size"] for s in top["spans"] + bot["spans"])
            if bot["y0"] - top["y1"] > 0.6 * sz or bot["y0"] < top["y0"]:
                continue
            # merge: lower tiny becomes the denominator
            top["spans"] = top["spans"] + bot["spans"]
            top["den_spans"] = top.get("den_spans", []) + list(bot["spans"])
            xs = [s["bbox"][0] for s in top["spans"]] + [s["bbox"][2] for s in top["spans"]]
            ys = [s["bbox"][1] for s in top["spans"]] + [s["bbox"][3] for s in top["spans"]]
            top["x0"], top["x1"], top["y0"], top["y1"] = min(xs), max(xs), min(ys), max(ys)
            bot["spans"] = []
            merged_tiny_ids.add(id(bot))
    for m in metas:
        if not m["spans"]:
            if m in smalls:
                smalls.remove(m)
            if m in mains:
                mains.remove(m)
    _main_lines = [m for m in metas if not m["all_small"]]
    # Destinations include SMALL metas too: a sub-sub glyph (the '{\bf x}'
    # under a \sum limit line, 09086) belongs to a limit-line cluster that
    # is itself small — main-only candidates steal it for the equation's
    # body line ('\frac{...}_{\bm{x}}', measured).
    _dests = [m for m in metas if m["spans"]]
    _main_geo = {id(mn): _line_body_and_baseline(mn["spans"]) for mn in _dests}
    for m in metas:
        if m["all_small"] or not m["spans"] or max(s["size"] for s in m["spans"]) >= 0.75 * page_body_ref:
            continue
        leftovers = []
        for s in list(m["spans"]):
            cy = (s["bbox"][1] + s["bbox"][3]) / 2
            best_m, best_d = None, None
            for mn in _dests:
                if mn is m or not mn["spans"]:
                    continue
                _b, _base = _main_geo[id(mn)]
                d = abs(cy - _base)
                if d < 0.9 * _b and (best_d is None or d < best_d):
                    best_m, best_d = mn, d
            if best_m is not None:
                best_m["spans"].append(s)
            else:
                leftovers.append(s)
        m["spans"] = leftovers
    # Stolen-script recovery (iter-27): stage-1 cy-insertion lets a
    # script span that hovers ABOVE a lower line's base get claimed by
    # the line ABOVE (measured 08800: sup-limit P of \prod stolen by the
    # prose line → 'ThenP ... $_{P}$'). Signature: a STRICT-math-font
    # script-size span whose dy in its OWN cluster reads as a subscript
    # of PROSE, but which x-abuts a MATH body span of another cluster
    # and sits ABOVE that cluster's base by the script band → move it
    # there (it is that token's superscript).
    for m in metas:
        if not m["spans"]:
            continue
        _b, _base = _line_body_and_baseline(m["spans"])
        if _b <= 0:
            continue
        _moved = []
        for s in m["spans"]:
            if (
                s["size"] >= 0.8 * _b
                or not _STRICT_MATH_FONT_RE.search(s.get("font", ""))
                or not (_base + 0.15 * _b < s["origin"][1] < _base + 2.0 * _b)
            ):
                continue  # not a script reading as sub of this line
            # Home check: a math body span already ending at this span's
            # left edge = a plausible base IN ITS OWN line (v_{τ^1_i}:
            # 'v' ends exactly at 'τ', 05506) — the span is NOT stolen.
            home_abuts = any(
                t is not s
                and _STRICT_MATH_FONT_RE.search(t.get("font", ""))
                and t["size"] >= 0.8 * _b
                and t["bbox"][2] <= s["bbox"][0] + 1.5
                and t["bbox"][2] > s["bbox"][0] - 2.0 * _b
                for t in m["spans"]
            ) or any(
                # SCRIPT-RUN neighbor (iter-98, 05068): a MID-RUN
                # subscript glyph — ',' after '·' inside K_{k∆t,U(·,0),θ}
                # — has no body-size left neighbor near its edge, so the
                # recovery read it as stolen and moved it DOWN to the
                # next line (which its dy band happens to match), losing
                # the comma from every K-subscript. An immediately
                # preceding script-size span at the same dy (ANY font —
                # the '(' / '0)' run furniture is roman CMR) = a home
                # run continuation.
                t is not s
                and t["size"] < 0.8 * _b
                and t["bbox"][2] <= s["bbox"][0] + 1.5
                and t["bbox"][2] > s["bbox"][0] - 2.0
                and abs(t["origin"][1] - s["origin"][1]) <= 1.0
                for t in m["spans"]
            )
            if home_abuts:
                continue
            for mn in metas:
                if mn is m or not mn["spans"]:
                    continue
                nb, nbase = _line_body_and_baseline(mn["spans"])
                if not (
                    nbase - 2.0 * nb < s["origin"][1] < nbase - 0.25 * nb
                ):
                    continue  # not in the sup band of the other line
                abuts = any(
                    t is not s
                    and _STRICT_MATH_FONT_RE.search(t.get("font", ""))
                    and t["size"] >= 0.8 * nb
                    and t["bbox"][2] <= s["bbox"][0] + 1.5
                    and t["bbox"][2] > s["bbox"][0] - 2.0 * nb
                    for t in mn["spans"]
                )
                if abuts:
                    mn["spans"].append(s)
                    _moved.append(s)
                    break
        if _moved:
            _ids = {id(s) for s in _moved}
            m["spans"] = [s for s in m["spans"] if id(s) not in _ids]
    for sm in smalls:
        # nearest main cluster vertically, horizontally centered on it
        best = None
        for mn in mains:
            gap = max(sm["y0"] - mn["y1"], mn["y0"] - sm["y1"], 0)
            overlap = min(
                1.0,
                max(
                    0.0,
                    (min(sm["x1"], mn["x1"]) - max(sm["x0"], mn["x0"]))
                    / max(1.0, sm["x1"] - sm["x0"]),
                ),
            )
            if overlap < 0.3:
                continue
            if gap > 0.8 * mn["body"]:
                continue
            score = gap - overlap * mn["body"]
            if best is None or score < best[0]:
                best = (score, mn)
        if best:
            best[1]["spans"].extend(sm["spans"])

    # merged tiny stacks (stage 2.7) attach like smalls, but their
    # den_spans must ride along — the point-frac assembly needs them.
    for tm in [t for t in tinies if t.get("den_spans") and t["spans"]]:
        best = None
        for mn in mains:
            if mn is tm or not mn["spans"]:
                continue
            gap = max(tm["y0"] - mn["y1"], mn["y0"] - tm["y1"], 0)
            overlap = min(
                1.0,
                max(
                    0.0,
                    (min(tm["x1"], mn["x1"]) - max(tm["x0"], mn["x0"]))
                    / max(1.0, tm["x1"] - tm["x0"]),
                ),
            )
            if overlap < 0.3:
                continue
            score = gap - overlap * 6.0
            if best is None or score < best[0]:
                best = (score, mn)
        if best:
            best[1]["spans"].extend(tm["spans"])
            best[1]["den_spans"] = (
                best[1].get("den_spans") or []
            ) + list(tm["den_spans"])
            tm["spans"] = []
    mains = [m for m in mains if m["spans"]]

    # Stage 2.5: STACKED-FRACTION cluster merge (2026-08-16). TeX display
    # fractions put the numerator on one visual line and the denominator on
    # the NEXT line, both at body size — band clustering (stage 1) keeps
    # them as separate clusters and stage 3 only pairs SMALL spans, so
    # ``\frac{\partial H_1/\partial u_1 - ...}{\partial u_1}``-style stacks
    # were emitted as two unrelated display lines (measured: arxiv_math
    # 265 frac-family gold fails).
    # ponytail: EXPERIMENTAL — CMEX big-op glyphs (∑∏∫ extend below the
    # line) still trigger false merges; OFF by default until the merge
    # gates handle them. Enable: PDF_OCR_STACKED_FRAC=1.
    import os as _os

    if _os.environ.get("PDF_OCR_STACKED_FRAC", "0") == "1":
        merged_ids = set()

        def _line_origin(m):
            """Body-baseline (median origin.y) of a cluster's body-size spans."""
            body = m["body"]
            orgs = sorted(s["origin"][1] for s in m["spans"] if s["size"] >= body * 0.85)
            return orgs[len(orgs) // 2] if orgs else m["y1"]

        for ai in range(len(mains)):
            if ai in merged_ids:
                continue
            A = mains[ai]
            for bj in range(ai + 1, len(mains)):
                if bj in merged_ids:
                    continue
                B = mains[bj]
                gap = B["y0"] - A["y1"]
                if gap > 0.45 * max(A["body"], B["body"]) or gap < -0.6 * A["body"]:
                    continue
                # TRUE STACK: B must start clearly below A's body BASELINE —
                # bounding-box centers lie when A carries hanging CMEX glyphs
                # (integral signs extend below the line and fooled the
                # y-center gate; measured inverted merge on 2503.04993_pg18).
                a_base = _line_origin(A)
                if B["y0"] < a_base + 0.10 * A["body"]:
                    continue
                ox = min(A["x1"], B["x1"]) - max(A["x0"], B["x0"])
                wmin = min(A["x1"] - A["x0"], B["x1"] - B["x0"])
                if ox < 0.5 * max(wmin, 1.0):
                    continue
                # A denominator LINE is typeset at ~70-100% of the numerator
                # size, spans several glyphs, and is a few body-widths wide.
                # (Rejects subscript-dangler clusters: a lone 6pt 'Z' under
                # a body line measured as a false denominator.)
                b_sizes = sorted(s["size"] for s in B["spans"])
                if b_sizes[len(b_sizes) // 2] < 0.7 * A["body"]:
                    continue
                if (B["x1"] - B["x0"]) < 1.2 * A["body"]:
                    continue

                def _math_frac_ratio(m):
                    # Equation furniture: strict math fonts PLUS roman-font
                    # spans that are pure operators/parens/digits (CMR10 '('
                    # '+' '=' are equation parts, not prose).
                    tot = sum(s["bbox"][2] - s["bbox"][0] for s in m["spans"])
                    if tot <= 0:
                        return 0.0

                    def _is_mathy(s):
                        if _STRICT_MATH_FONT_RE.search(s.get("font", "")):
                            return True
                        return not re.search(r"[a-z]{3,}", s.get("text", ""))

                    mu = sum(s["bbox"][2] - s["bbox"][0] for s in m["spans"] if _is_mathy(s))
                    return mu / tot

                def _has_prose(m):
                    return any(
                        re.search(r"[a-z]{4,}", s["text"])
                        and not _STRICT_MATH_FONT_RE.search(s.get("font", ""))
                        for s in m["spans"]
                    )

                if _math_frac_ratio(A) < 0.55 or _math_frac_ratio(B) < 0.55:
                    continue
                if _has_prose(A) or _has_prose(B):
                    continue
                # Merge: B's spans join A's cluster; record denominator spans.
                A["spans"].extend(B["spans"])
                A["den_spans"] = A.get("den_spans", []) + list(B["spans"])
                xs = [s["bbox"][0] for s in A["spans"]] + [s["bbox"][2] for s in A["spans"]]
                ys = [s["bbox"][1] for s in A["spans"]] + [s["bbox"][3] for s in A["spans"]]
                A["x0"], A["x1"], A["y0"], A["y1"] = min(xs), max(xs), min(ys), max(ys)
                merged_ids.add(bj)
                break

        mains = [m for i, m in enumerate(mains) if i not in merged_ids]

    # DISPLAY-STACK host merge (iter-92, 2503.06228_pg1): a display
    # equation's stacked fraction renders its numerator and denominator
    # as SEPARATE body-size lines ('1' above, '(q;q)_∞' below the '='
    # line) — band clustering keeps them apart because their baselines
    # differ by ~1.4 body sizes. Merge such stacks into the HOST line
    # (the main whose baseline sits BETWEEN them and whose body spans
    # leave an x-GAP at the stack's shared center — the fraction slot);
    # the proven 3b full-size pairing then synthesizes \frac{num}{den}.
    # Gates: both parts fully mathy, centers x-aligned, baseline gaps
    # within (0.8, 2.8) body — consecutive prose lines are closer and
    # never leave a shared-x gap in an intervening line.
    if len(mains) >= 3:
        def _meta_base(m):
            _sp = sorted(m["spans"], key=lambda s: s["bbox"][0])
            _b, _bs = _line_body_and_baseline(_sp)
            return _bs

        _merged_ids = set()
        for ai in range(len(mains)):
            if ai in _merged_ids:
                continue
            A = mains[ai]
            if not A["spans"]:
                continue
            _ab = _meta_base(A)
            for bj in range(len(mains)):
                if bj in _merged_ids or bj == ai:
                    continue
                B = mains[bj]
                if not B["spans"]:
                    continue
                _bb = _meta_base(B)
                _dy = abs(_bb - _ab)
                _body_ref = max(
                    (s["size"] for s in A["spans"] + B["spans"]), default=10.0
                )
                if not (0.8 * _body_ref < _dy <= 2.8 * _body_ref):
                    continue
                # both fully mathy
                if not all(_mathy(s) for s in A["spans"] + B["spans"]):
                    continue
                # x-centers aligned
                _ac = (A["x0"] + A["x1"]) / 2
                _bc = (B["x0"] + B["x1"]) / 2
                if abs(_ac - _bc) > 0.6 * max(A["x1"] - A["x0"], B["x1"] - B["x0"], 4):
                    continue
                # host: baseline between, x-overlap, gap at the stack center
                _host = None
                for hi in range(len(mains)):
                    if hi in (ai, bj) or hi in _merged_ids:
                        continue
                    H = mains[hi]
                    if not H["spans"]:
                        continue
                    _hb = _meta_base(H)
                    if not (min(_ab, _bb) < _hb < max(_ab, _bb)):
                        continue
                    if H["x1"] < min(A["x0"], B["x0"]) or H["x0"] > max(A["x1"], B["x1"]):
                        continue
                    _covers = any(
                        s["size"] >= 0.8 * _body_ref
                        and s["bbox"][0] - 2 <= (_ac + _bc) / 2 <= s["bbox"][2] + 2
                        for s in H["spans"]
                    )
                    if not _covers:
                        _host = H
                        break
                if _host is not None:
                    _host["spans"].extend(A["spans"])
                    _host["spans"].extend(B["spans"])
                    _xs = [s["bbox"][0] for s in _host["spans"]] + [
                        s["bbox"][2] for s in _host["spans"]
                    ]
                    _ys = [s["bbox"][1] for s in _host["spans"]] + [
                        s["bbox"][3] for s in _host["spans"]
                    ]
                    _host["x0"], _host["x1"] = min(_xs), max(_xs)
                    _host["y0"], _host["y1"] = min(_ys), max(_ys)
                    _merged_ids.update({ai, bj})
                    break
        if _merged_ids:
            mains = [m for i, m in enumerate(mains) if i not in _merged_ids]
    # STRANDED-SCRIPT HOME RETURN (iter-100, 08800): band clustering
    # can park a subscript run on the NEXT line when its glyphs dip
    # into the inter-line gap ('∆n' of P_{∆n} rides the 'Proposition'
    # cluster ~18pt above ITS base). Signature: a strict-math script
    # span stranded far above its own cluster's base that x-abuts a
    # same-dy script of the cluster above — it continues that run.
    # NUMERATOR-STRAND MERGE (iter-100, 07897 eq A.4): an inline frac
    # at a line's end strands its lone full-size numerator in a tiny
    # cluster above the host line, whose HEAD span is the lowered
    # denominator ('γ' over head-'ρ' of µ∆(t)=…=γ/ρ ln(1+ρt)). Merge
    # the strand into the host so stage-3b pairing synthesizes the frac.
    if len(mains) >= 2:
        _srt = sorted(
            mains, key=lambda m: m["spans"][0]["bbox"][1] if m["spans"] else 0.0
        )
        for _bi in range(1, len(_srt)):
            B = _srt[_bi]
            if not B["spans"]:
                continue
            _bb, _bbase = _line_body_and_baseline(
                sorted(B["spans"], key=lambda s: s["bbox"][0])
            )
            if _bb <= 0:
                continue
            for s in list(B["spans"]):
                if (
                    s["size"] >= 0.8 * _bb
                    or not _STRICT_MATH_FONT_RE.search(s.get("font", ""))
                    or not s["origin"][1] < _bbase - 0.9 * _bb
                ):
                    continue  # not a script stranded far above its line
                for _ai in range(_bi):
                    A = _srt[_ai]
                    if not A["spans"]:
                        continue
                    _ab, _abase = _line_body_and_baseline(
                        sorted(A["spans"], key=lambda t: t["bbox"][0])
                    )
                    if _ab <= 0 or _abase >= s["origin"][1]:
                        continue
                    if any(
                        t is not s
                        and t["bbox"][2] <= s["bbox"][0] + 2.0
                        and t["bbox"][2] > s["bbox"][0] - 2.0 * _ab
                        and abs(t["origin"][1] - s["origin"][1]) <= 2.0
                        for t in A["spans"]
                    ):
                        B["spans"] = [t for t in B["spans"] if t is not s]
                        A["spans"].append(s)
                        break
        # SUP-BAND ORPHAN RELOCATION (iter-101, 08800): a display's
        # raised sup glyphs (∏'s limit P_{∆n}, c's exponent P_{∆n}) band
        # into the line ABOVE when they hover between lines — reading as
        # a DEEP subscript there (>0.8 body below their home base, far
        # past the genuine script band +0.15..0.7). Signature: mathy
        # script span, orphan-deep at home, sitting in the NEXT line's
        # SUP band, x-abutting one of its body spans. Move it down; its
        # lowered partner glyphs are already in the target cluster.
        for _bi in range(1, len(_srt)):
            B = _srt[_bi]
            if not B["spans"]:
                continue
            _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])
            _bb, _bbase = _line_body_and_baseline(_bsp)
            if _bb <= 0:
                continue
            for _ai in range(_bi):
                A = _srt[_ai]
                if not A["spans"]:
                    continue
                _ab, _abase = _line_body_and_baseline(
                    sorted(A["spans"], key=lambda s: s["bbox"][0])
                )
                if _ab <= 0:
                    continue
                for s in list(A["spans"]):
                    if (
                        s["size"] >= 0.8 * _bb
                        or not _STRICT_MATH_FONT_RE.search(s.get("font", ""))
                        or not (s["origin"][1] - _abase > 0.8 * _ab)
                        or not (
                            _bbase - 1.3 * _bb
                            < s["origin"][1]
                            < _bbase - 0.10 * _bb
                        )
                    ):
                        continue
                    # LOCAL ANCHOR test (06703 regression): a genuine
                    # script has a body-size x-adjacent span at matching
                    # dy in its own cluster (its base, or its column's
                    # line on interleaved pages). An orphan sup glyph
                    # between lines has NONE — nearest body origins sit
                    # a full script-drop away. (Origin arithmetic fails
                    # here: EUEX big-op origins sit ~0.8 body above the
                    # line base, faking a low local baseline for '1'.)
                    if any(
                        t is not s
                        and t["size"] >= 0.8 * _ab
                        and t["bbox"][2] <= s["bbox"][0] + 1.5
                        and t["bbox"][2] > s["bbox"][0] - 3.0 * _ab
                        and abs(t["origin"][1] - s["origin"][1]) <= 0.55 * _ab
                        for t in A["spans"]
                    ):
                        continue  # anchored left neighbor at same dy = base's script
                    if not any(
                        t["size"] >= 0.8 * _bb
                        and t["bbox"][2] <= s["bbox"][0] + 1.5
                        and t["bbox"][2] > s["bbox"][0] - 2.0 * _bb
                        for t in _bsp
                    ):
                        continue
                    A["spans"] = [t for t in A["spans"] if t is not s]
                    B["spans"].append(s)
                    _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])
        # NUM-BAND MERGE (iter-100): an inline/display frac whose
        # numerator rides a raised band gets its own cluster when band
        # clustering splits there ('∥t^k∥²' + the sum's '∞' limit as a
        # lone band above the host, 06476; lone 'γ' above host-'ρ',
        # 07897 eq A.4). Signature: A (above) is a SHORT all-mathy
        # cluster whose body baseline sits 0.3–0.85 body ABOVE B's; B
        # carries a LOWERED body-size mathy span x-aligned with A (the
        # denominator) with a free slot around it. Merge A into B so
        # stage 3b/3b-G pairing synthesizes \frac{num}{den}.
        for _bi in range(1, len(_srt)):
            B = _srt[_bi]
            if not B["spans"]:
                continue
            _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])
            _bb, _bbase = _line_body_and_baseline(_bsp)
            if _bb <= 0:
                continue
            for _ai in range(_bi):
                A = _srt[_ai]
                if not A["spans"] or len(A["spans"]) > 6:
                    continue
                if not all(
                    _STRICT_MATH_FONT_RE.search(s.get("font", ""))
                    or not re.search(r"[a-z]{3,}", s.get("text", ""))
                    for s in A["spans"]
                ):
                    continue  # numerators never carry prose words
                _ab, _abase = _line_body_and_baseline(A["spans"])
                if _ab <= 0 or not (
                    _bbase - 0.85 * _bb < _abase < _bbase - 0.30 * _bb
                ):
                    continue  # A must be the raised band, not a line above
                _ax0 = min(s["bbox"][0] for s in A["spans"])
                _ax1 = max(s["bbox"][2] for s in A["spans"])
                if A["spans"][0]["bbox"][3] < B["spans"][0]["bbox"][1] - 1.2 * _bb:
                    continue  # A floats far above B — different construct
                # denominator: a lowered body-size mathy span of B,
                # x-aligned with A, no other B body span in its slot
                _den = None
                for t in _bsp:
                    if (
                        t["size"] >= 0.85 * _bb
                        and (
                            _STRICT_MATH_FONT_RE.search(t.get("font", ""))
                            or not re.search(r"[a-z]{3,}", t.get("text", ""))
                        )
                        and _bbase + 0.20 * _bb < t["origin"][1] < _bbase + 1.0 * _bb
                        and min(_ax1, t["bbox"][2]) - max(_ax0, t["bbox"][0])
                        >= 0.3 * min(t["bbox"][2] - t["bbox"][0], _ax1 - _ax0)
                    ):
                        _den = t
                        break
                if _den is None:
                    continue
                _dm = (_den["bbox"][0] + _den["bbox"][2]) / 2
                if any(
                    t is not _den
                    and t["size"] >= 0.85 * _bb
                    and t["bbox"][0] - 2 <= _dm <= t["bbox"][2] + 2
                    for t in _bsp
                ):
                    continue  # slot busy — not a fraction column
                B["spans"] = A["spans"] + B["spans"]
                A["spans"] = []
                break

        # DEN-BAND PULL (iter-102, 09086): the mirror image of the
        # num-band merge — a lone body-size mathy span stranded in the
        # band BELOW a display (band clustering split \frac{1}{2}'s
        # denominator '2' away from its host, '1' raised in-band).
        # Pull the span when it sits 0.2–1.0 body BELOW the host base,
        # x-aligns with a RAISED body span of the host (the numerator),
        # and no host body span covers that x. Stage-3b pairing then
        # synthesizes \frac{num}{den}.
        for _mi, B in enumerate(_srt):
            if not B["spans"]:
                continue
            _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])
            _bb, _bbase = _line_body_and_baseline(_bsp)
            if _bb <= 0:
                continue
            for _ni in range(_mi + 1, len(_srt)):
                N = _srt[_ni]
                if not N["spans"] or len(N["spans"]) > 6:
                    continue
                for t in list(N["spans"]):
                    if (
                        t["size"] < 0.85 * _bb
                        or t.get("text", "").strip() in _ENV_ACCENT_CMDS
                        or not (
                            _STRICT_MATH_FONT_RE.search(t.get("font", ""))
                            or not re.search(r"[a-z]{3,}", t.get("text", ""))
                        )
                        or not (
                            _bbase + 0.20 * _bb
                            < t["origin"][1]
                            < _bbase + 1.0 * _bb
                        )
                    ):
                        continue
                    _tm = (t["bbox"][0] + t["bbox"][2]) / 2
                    if any(
                        s["size"] >= 0.85 * _bb
                        and s["bbox"][0] - 2 <= _tm <= s["bbox"][2] + 2
                        and abs(s["origin"][1] - _bbase) <= 0.30 * _bb
                        for s in _bsp
                    ):
                        continue  # host already covers this x at its band
                    if not any(
                        s["size"] >= 0.85 * _bb
                        and s["origin"][1] < _bbase - 0.30 * _bb
                        and min(s["bbox"][2], t["bbox"][2])
                        - max(s["bbox"][0], t["bbox"][0])
                        > 0
                        for s in _bsp
                    ):
                        continue  # no raised numerator over it
                    N["spans"] = [x for x in N["spans"] if x is not t]
                    B["spans"].append(t)
                    _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])

        # ACCENT PULL (iter-102, 09086): stranded tildewide accents
        # x-overlapping a body span of a NEARBY main (their base) move
        # there so _pair_accents can wrap it — band clustering parks
        # overstrike accents in the inter-line band cluster.
        for _mi, B in enumerate(_srt):
            if not B["spans"]:
                continue
            _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])
            _bb, _bbase = _line_body_and_baseline(_bsp)
            if _bb <= 0:
                continue
            for _ni in range(len(_srt)):
                if _ni == _mi:
                    continue
                N = _srt[_ni]
                if not N["spans"] or len(N["spans"]) > 6:
                    continue
                for t in list(N["spans"]):
                    if t.get("text", "").strip() not in _ENV_ACCENT_CMDS:
                        continue
                    # CMEX wide accents carry huge advance bboxes (37pt
                    # tall) — compare ORIGINS: the accent's origin sits
                    # at/above its base's baseline, x-overlapping it.
                    if not any(
                        s["size"] >= 0.85 * _bb
                        and t["origin"][1] <= s.get("origin", s["bbox"])[1] + 2.5
                        and min(t["bbox"][2], s["bbox"][2])
                        - max(t["bbox"][0], s["bbox"][0])
                        > 0
                        for s in _bsp
                    ):
                        continue
                    N["spans"] = [x for x in N["spans"] if x is not t]
                    B["spans"].append(t)
                    _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])

        # LIMIT PULL (iter-102, 09086): a script-size mathy span stranded
        # in a band cluster whose x-center sits inside a glyph-decoded
        # big-op's limit window of a nearby main is that op's lower
        # limit (∫_Ω) — move it so stage-3c rides it on the op tag.
        for _mi, B in enumerate(_srt):
            if not B["spans"]:
                continue
            _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])
            _bb, _bbase = _line_body_and_baseline(_bsp)
            if _bb <= 0:
                continue
            _ops = [
                s
                for s in _bsp
                if _ACTIVE_CMEX_GLYPHS.get(s.get("font", ""), {}).get(
                    s.get("text", ""), ""
                ).strip() in _BIGOPS
            ]
            if not _ops:
                continue
            for _ni in range(len(_srt)):
                if _ni == _mi:
                    continue
                N = _srt[_ni]
                if not N["spans"] or len(N["spans"]) > 6:
                    continue
                for t in list(N["spans"]):
                    if (
                        t["size"] >= 0.85 * _bb
                        or not (
                            _STRICT_MATH_FONT_RE.search(t.get("font", ""))
                            or not re.search(r"[a-z]{3,}", t.get("text", ""))
                        )
                        or not (
                            _bbase + 0.20 * _bb
                            < t["origin"][1]
                            < _bbase + 2.0 * _bb
                        )
                    ):
                        continue
                    _tc = (t["bbox"][0] + t["bbox"][2]) / 2
                    if not any(
                        op["bbox"][0] - 0.5 * max(op["bbox"][2] - op["bbox"][0], 1.0)
                        <= _tc
                        <= op["bbox"][2] + 1.6 * max(op["bbox"][2] - op["bbox"][0], 1.0)
                        for op in _ops
                    ):
                        continue
                    N["spans"] = [x for x in N["spans"] if x is not t]
                    B["spans"].append(t)
                    _bsp = sorted(B["spans"], key=lambda s: s["bbox"][0])

    # Stage 3: frac pairing inside each main cluster
    out = []
    for mn in mains:
        if not mn["spans"]:
            continue  # num-band merge consumed this main into its host
        den_spans = mn.get("den_spans")
        spans = sorted(mn["spans"], key=lambda s: s["bbox"][0])
        # Shared two-stage estimator (iter-34; the iter-30 postmortem's
        # "estimator-only" split): the span-median collapses on
        # display/script-dense clusters (09577 cluster 2: median 8pt →
        # the group-frac pass saw nothing raised/lowered, and every
        # stage-3 window misjudged). One estimator, all call sites.
        body, base = _line_body_and_baseline(spans)
        # Stage 3c: IN-CLUSTER big-op subscripts (iter-15). When band
        # clustering merged the operator's limit line into the equation
        # cluster (09086 'x∈X(Ω)', 06570 'j=1'), those small spans flow
        # through generic script accumulation and glue onto whatever
        # precedes them (measured '\sum^{b(X)}_{j^2=1}'). Pull small
        # spans under the op's x-window out; they ride the op tag.
        if _ACTIVE_CMEX_GLYPHS:
            _op_re = re.compile(r"[=∈≤≥<≤⩾≪≫≠+\-]")
            for op in [
                s for s in spans
                if _ACTIVE_CMEX_GLYPHS.get(s.get("font", ""), {}).get(
                    s.get("text", ""), ""
                ).strip() in _BIGOPS
            ]:
                ow = max(op["bbox"][2] - op["bbox"][0], 1.0)
                ox0, ox1 = op["bbox"][0], op["bbox"][2]
                cand = sorted(
                    (
                        s for s in spans
                        if s is not op
                        and s["size"] < body * 0.8
                        and s.get("text", "").strip()
                        and s["bbox"][0] < ox1 + 1.6 * ow
                        and s["bbox"][2] > ox0 - 0.5 * ow
                        and base + 0.25 * body
                        < s["origin"][1]
                        < base + 2.0 * body
                        and not s.get("frac")
                    ),
                    key=lambda s: s["bbox"][0],
                )
                # contiguous groups; keep only LIMIT LINES: ≥2 spans and
                # a relation/operator glyph ('j=1', 'x∈X(Ω)', '1≤i,j≤2').
                # Bare variable scripts (λ,θ of \max_\lambda\min_\theta)
                # belong to their own tokens — never steal them.
                cgroups: list[list] = []
                for s in cand:
                    if (
                        cgroups
                        and s["bbox"][0] - cgroups[-1][-1]["bbox"][2] <= 1.5 * ow
                    ):
                        cgroups[-1].append(s)
                    else:
                        cgroups.append([s])
                take = []
                for cg in cgroups:
                    if len(cg) < 2:
                        continue
                    if not any(_op_re.search(s["text"]) for s in cg):
                        continue
                    take.extend(cg)
                if take:
                    op["_opsub_spans"] = op.get("_opsub_spans", []) + take
                    tids = {id(s) for s in take}
                    spans = [s for s in spans if id(s) not in tids]
            # Roman FUNC_CMD words (\max \min …) carry SINGLE-glyph limits
            # directly under the word (λ/θ at 7pt under CMR10 'max'/'min',
            # x-centered — measured 09086 arg-equation). No operator
            # requirement; tight x-window [ox0, ox1] avoids stealing the
            # subscripts of neighbouring θ_P λ_P tokens.
            for op in [
                s for s in spans
                if s.get("size", 0) >= body * 0.9
                and s.get("text", "").strip() in _FUNC_CMD_WORDS
            ]:
                ox0, ox1 = op["bbox"][0], op["bbox"][2]
                subs = [
                    s for s in spans
                    if s is not op
                    and s["size"] < body * 0.8
                    and s.get("text", "").strip()
                    and ox0 <= (s["bbox"][0] + s["bbox"][2]) / 2 <= ox1
                    and base + 0.15 * body < s["origin"][1] < base + 1.5 * body
                    and not s.get("frac")
                    and not s.get("_opsub_spans")
                ]
                if subs:
                    op["_opsub_spans"] = op.get("_opsub_spans", []) + subs
                    sids = {id(s) for s in subs}
                    spans = [s for s in spans if id(s) not in sids]
        above = [s for s in spans if s["size"] < body * 0.8 and s["origin"][1] < base - 0.5]
        below = [s for s in spans if s["size"] < body * 0.8 and s["origin"][1] > base + 0.5]
        bodyx = [(s["bbox"][0], s["bbox"][2]) for s in spans if s["size"] >= body * 0.8]
        used_below = set()
        for a in above:
            ax = (a["bbox"][0] + a["bbox"][2]) / 2
            for j, b in enumerate(below):
                if j in used_below:
                    continue
                bx = (b["bbox"][0] + b["bbox"][2]) / 2
                if abs(ax - bx) > 0.6 * max(
                    a["bbox"][2] - a["bbox"][0], b["bbox"][2] - b["bbox"][0], 4
                ):
                    continue
                # Gate 1: no body-size span covering that x between them
                if any(x0 - 2 <= ax <= x1 + 2 for x0, x1 in bodyx):
                    break
                # Gate 2: denominator substantial (>=2 chars). H_0^1 style
                # script pairs have 1-char "denominators".
                if len(b["text"].strip()) < 2:
                    continue
                # Gate 3: body span immediately LEFT must be an operator —
                # a letter ending right at the pair means these are its
                # scripts (H_0^1), not a stacked fraction.
                px1 = min(a["bbox"][0], b["bbox"][0])
                left = [
                    sp
                    for sp in spans
                    if sp["size"] >= body * 0.8
                    and sp["bbox"][2] <= px1 + 1.5
                    and sp["bbox"][2] > px1 - 6 * body
                ]
                if left:
                    lt = max(left, key=lambda sp: sp["bbox"][2])["text"].strip()
                    if lt and re.search(r"[A-Za-z0-9)]$", lt):
                        continue
                # synthetic \frac span replaces the pair
                a["frac"] = (a["text"], b["text"])
                used_below.add(j)
        # Stage 3c-g (iter-34): SCRIPT-SIZE stacked fracs with GROUP
        # numerators — \frac{k^{n}(k-1)}{4} sets its numerator as a
        # contiguous raised run of script spans (carrying its own nested
        # scripts) over a lowered denominator. Single-span pairing (stage
        # 3) cannot match a multi-glyph numerator; full-size pairing (3b)
        # requires body-size parts. Gates: the left body token must be an
        # OPERATOR (unicode-aware raw test) and NOT a decoded big-op —
        # an operator's above/below bands are LIMITS, not fractions —
        # and no body span may cover the groups' x-overlap midpoint.
        def _x_groups(items):
            items = sorted(items, key=lambda s: s["bbox"][0])
            out_g: list[list] = []
            for s in items:
                if out_g and s["bbox"][0] - out_g[-1][-1]["bbox"][2] <= 0.6 * body:
                    out_g[-1].append(s)
                else:
                    out_g.append([s])
            return out_g

        raised_small = [
            s for s in spans
            if s["size"] < body * 0.9 and s["origin"][1] < base - 0.30 * body
            and not s.get("frac") and not s.get("_opsub_spans")
        ]
        lowered_small = [
            s for s in spans
            if s["size"] < body * 0.9 and s["origin"][1] > base + 0.22 * body
            and not s.get("frac") and not s.get("_opsub_spans")
        ]
        _consumed_g: set = set()
        for _ag in _x_groups(raised_small):
            if id(_ag[0]) in _consumed_g or _ag[0].get("frac"):
                continue
            _ax0, _ax1 = _ag[0]["bbox"][0], _ag[-1]["bbox"][2]
            for _bg in _x_groups(lowered_small):
                if any(id(s) in _consumed_g for s in _bg):
                    continue
                _bx0, _bx1 = _bg[0]["bbox"][0], _bg[-1]["bbox"][2]
                if min(_ax1, _bx1) - max(_ax0, _bx0) < -0.25 * body:
                    continue
                _mx = (min(_ax1, _bx1) + max(_ax0, _bx0)) / 2
                if any(x0 - 2 <= _mx <= x1 + 2 for x0, x1 in bodyx):
                    continue
                import os as _os
                if _os.environ.get("PDF_OCR_DEBUG_GFRAC"):
                    print(f"GFRAC-CAND num={_ag[0]['text'][:4]!r}@{_ag[0]['bbox'][0]:.0f}-{_ag[-1]['bbox'][2]:.0f} den={_bg[0]['text'][:4]!r}@{_bg[0]['bbox'][0]:.0f}-{_bg[-1]['bbox'][2]:.0f}")
                _px1 = min(_ax0, _bx0)
                _left = [
                    sp for sp in spans
                    if sp["size"] >= body * 0.8
                    and sp["bbox"][2] <= _px1 + 1.5
                    and sp["bbox"][2] > _px1 - 6 * body
                ]
                if _left:
                    _lsp = max(_left, key=lambda sp: sp["bbox"][2])
                    _lraw = _lsp["text"].rstrip()
                    _ldec = _ACTIVE_CMEX_GLYPHS.get(
                        _lsp.get("font", ""), {}
                    ).get(_lsp["text"].strip(), "")
                    if _ldec.strip() in _BIGOPS:
                        continue  # above/below bands of a big-op = LIMITS
                    if (
                        _lraw
                        and (_lraw[-1].isalnum() or _lraw[-1] == ")")
                        and _px1 - _lsp["bbox"][2] < 0.3 * body
                    ):
                        # operand's scripts (H_0^1, τ_i^2) HUG their base
                        # (<0.3 body); an inline frac after a prose word
                        # keeps a visible gap ('of 1/P_∆n', 08800)
                        continue
                # CMEX big-paren flanks (iter-97, 09577): a script-size
                # vertical stack enclosed by body-size CMEX delimiters —
                # open at the left gate, close just right of the bands —
                # is a BINOMIAL. \binom renders its own parens and the
                # line builder's 2-band stack arm synthesizes it; making
                # \frac here strands the parens ('( \frac{k}{2} )').
                if _left:
                    _px0 = max(_ax1, _bx1)
                    _rfl = [
                        sp for sp in spans
                        if sp["size"] >= body * 0.8
                        and _px0 - 1.5 <= sp["bbox"][0] < _px0 + 6 * body
                    ]
                    _rsp = min(_rfl, key=lambda sp: sp["bbox"][0]) if _rfl else None
                    if (
                        _rsp is not None
                        and re.search(r"(?i)cmex|euex", _lsp.get("font", ""))
                        and re.search(r"(?i)cmex|euex", _rsp.get("font", ""))
                        and (_lsp["text"] or "\x00")[0] in "\x00\x10\x12"
                        and (_rsp["text"] or "\x01")[0] in "\x01\x11\x13"
                    ):
                        continue  # binom — leave the stack to the line builder
                import os as _os
                if _os.environ.get("PDF_OCR_DEBUG_GFRAC"):
                    print(f"GFRAC num={_ag[0]['text'][:4]!r}@{_ag[0]['bbox'][0]:.0f} den={_bg[0]['text'][:4]!r}@{_bg[0]['bbox'][0]:.0f} left={_lraw if _left else None!r}")
                _num_txt = _stack_group_text(_ag, base)
                _den_txt = _stack_group_text(_bg, base)
                if _num_txt and _den_txt:
                    _ag[0]["frac"] = (_num_txt, _den_txt)
                    for s_ in _ag[1:]:
                        s_["_frac_consumed_g"] = True
                    for s_ in _bg:
                        s_["_frac_consumed_g"] = True
                    _consumed_g.update(id(s_) for s_ in _ag + _bg)
                break
        spans = [s for s in spans if not s.get("_frac_consumed_g")]
        # Stage 3b: FULL-SIZE vertical fracs (iter-15). TeX display style
        # sets \frac{1}{2} with BOTH parts at body size — one clearly
        # raised line over one clearly lowered band. Stage 3's small-only
        # pairing never sees them (measured: "1 2" emitted inline,
        # 09086 eq 2.7). Denominator = contiguous x-run of lowered body
        # spans so \frac{1}{X(\partial\Omega)} works too.
        raised_f = [
            s for s in spans
            if s["size"] >= body * 0.9 and s["origin"][1] < base - 0.30 * body
        ]
        lowered_f = [
            s for s in spans
            if s["size"] >= body * 0.9 and s["origin"][1] > base + 0.22 * body
        ]
        lowered_f.sort(key=lambda s: s["bbox"][0])
        drop_f: list = []
        # Stage 3b-G (iter-100): full-size fracs with GROUP numerators.
        # The single-span loop below pairs one raised span × one lowered
        # group by center; a multi-glyph numerator ('∥ t^k ∥²' over '2',
        # 06476) never center-aligns as a unit. Pair x-contiguous raised
        # runs with lowered runs under the same no-body-cover and
        # left-operator guards; _stack_group_text assembles the group so
        # nested scripts ride along (t^{k}, ∥^{2}).
        def _big_delim(s) -> bool:
            # CMEX/EUEX extensible-delimiter slots — parens/brackets of a
            # \left…\right pair flank the fraction; they are NEVER frac parts
            return bool(
                re.search(r"(?i)cmex|euex", s.get("font", ""))
                and (s.get("text") or " ")[0] in "\x00\x01\x02\x03\x10\x11\x12\x13"
            )

        _gmark: set = set()
        for _rg in _x_groups(
            [
                s
                for s in raised_f
                if not s.get("frac")
                and not s.get("_opsub_spans")
                and not _big_delim(s)
            ]
        ):
            if any(id(s) in _gmark for s in _rg):
                continue
            _gx0 = _rg[0]["bbox"][0]
            _gx1 = _rg[-1]["bbox"][2]
            # membership: ANY raised span (script-size included — the
            # numerator's own sub/sups) whose x-center falls in the run's
            # window; delimiters of the enclosing fence pair stay out
            _members = [
                s
                for s in spans
                if s["origin"][1] < base - 0.30 * body
                and not s.get("frac")
                and not s.get("_opsub_spans")
                and not s.get("_frac_consumed_g")
                and not _big_delim(s)
                and _gx0 - 0.6 * body
                <= (s["bbox"][0] + s["bbox"][2]) / 2
                <= _gx1 + 0.6 * body
            ]
            if not _members:
                continue
            _gx0 = min(s["bbox"][0] for s in _members)
            _gx1 = max(s["bbox"][2] for s in _members)
            for _dg in _x_groups(
                [s for s in lowered_f if s not in drop_f and id(s) not in _gmark]
            ):
                _dx0, _dx1 = _dg[0]["bbox"][0], _dg[-1]["bbox"][2]
                # pull the denominator's own lowered subscripts into the
                # group ('α' + 8pt 'k' → α_k) — x-contiguous script band
                for s in spans:
                    if (
                        s not in _dg
                        and s["size"] < body * 0.9
                        and s["origin"][1] > base + 0.22 * body
                        and not s.get("frac")
                        and not s.get("_opsub_spans")
                        and s["bbox"][0] - _dx1 <= 0.6 * body
                        and s["bbox"][2] > _dx0 - 0.6 * body
                    ):
                        _dg.append(s)
                _dg.sort(key=lambda s: s["bbox"][0])
                _dx0, _dx1 = _dg[0]["bbox"][0], _dg[-1]["bbox"][2]
                if min(_gx1, _dx1) - max(_gx0, _dx0) < 0.35 * min(
                    _gx1 - _gx0, _dx1 - _dx0
                ):
                    continue
                _mx = (min(_gx1, _dx1) + max(_gx0, _dx0)) / 2
                if any(
                    s["size"] >= body * 0.9
                    and s not in _members
                    and s not in _dg
                    and s["bbox"][0] - 2 <= _mx <= s["bbox"][2] + 2
                    for s in spans
                ):
                    continue
                _gpx1 = min(_gx0, _dx0)
                _gleft = [
                    sp for sp in spans
                    if sp["size"] >= body * 0.8
                    and sp["bbox"][2] <= _gpx1 + 1.5
                    and sp["bbox"][2] > _gpx1 - 6 * body
                ]
                if _gleft:
                    _glsp = max(_gleft, key=lambda sp: sp["bbox"][2])
                    _gltok = _span_to_math_token(
                        _glsp["text"], False, False, _glsp.get("font", "")
                    )
                    if (
                        _gltok
                        and re.search(r"[A-Za-z0-9)]$", _gltok.rstrip())
                        and _gpx1 - _glsp["bbox"][2] < 0.3 * body
                    ):
                        continue  # hugging scripts of an operand, not a frac
                _gnum = _stack_group_text(_members, base)
                _gden = _stack_group_text(_dg, base)
                if _gnum and _gden:
                    _members[0]["frac"] = (_gnum, _gden)
                    _gmark.update(id(s) for s in _members)
                    drop_f.extend(_dg)
                    break
        if _gmark:
            spans = [s for s in spans if id(s) not in _gmark or s.get("frac")]
        for a in raised_f:
            if a.get("frac"):
                continue
            ax = (a["bbox"][0] + a["bbox"][2]) / 2
            # contiguous lowered groups (gap > 0.6 body splits columns)
            groups: list[list] = []
            for s in lowered_f:
                if s in drop_f:
                    continue
                if groups and s["bbox"][0] - groups[-1][-1]["bbox"][2] <= 0.6 * body:
                    groups[-1].append(s)
                else:
                    groups.append([s])
            best_g, best_d = None, None
            for g in groups:
                gx = (g[0]["bbox"][0] + g[-1]["bbox"][2]) / 2
                if abs(ax - gx) > 0.8 * max(g[-1]["bbox"][2] - g[0]["bbox"][0], 4.0):
                    continue
                # Gate: no OTHER body span (outside the group) covering ax
                if any(
                    s is not a and s not in g and s["size"] >= body * 0.9
                    and s["bbox"][0] - 2 <= ax <= s["bbox"][2] + 2
                    for s in spans
                ):
                    continue
                # Gate: body span immediately LEFT must render as an
                # operator (glyph-decoded ∑ counts; raw CMEX letter-slots
                # would read as operands and veto the pair).
                px1 = min(a["bbox"][0], g[0]["bbox"][0])
                left = [
                    sp for sp in spans
                    if sp["size"] >= body * 0.8
                    and sp["bbox"][2] <= px1 + 1.5
                    and sp["bbox"][2] > px1 - 6 * body
                ]
                if left:
                    lsp = max(left, key=lambda sp: sp["bbox"][2])
                    ltok = _span_to_math_token(lsp["text"], False, False, lsp.get("font", ""))
                    if (
                        ltok
                        and re.search(r"[A-Za-z0-9)]$", ltok.rstrip())
                        and px1 - lsp["bbox"][2] < 0.3 * body
                    ):
                        # scripts hug their operand; a frac after an
                        # operand keeps a visible gap (γ/ρ after '=',
                        # 07897 eq A.4)
                        continue
                best_g, best_d = g, abs(ax - gx)
                break
            if best_g:
                den_txt = "".join(
                    _span_to_math_token(s["text"], False, False, s.get("font", "")) or s["text"]
                    for s in best_g
                ).strip()
                num_tok = _span_to_math_token(a["text"], False, False, a.get("font", "")) or a["text"]
                if den_txt and num_tok:
                    a["frac"] = (num_tok.strip(), den_txt)
                    drop_f.extend(best_g)
        spans = [s for s in spans if s not in drop_f]
        xs = [s["bbox"][0] for s in spans] + [s["bbox"][2] for s in spans]
        ys = [s["bbox"][1] for s in spans] + [s["bbox"][3] for s in spans]
        out.append(
            {
                "spans": spans,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "den_spans": den_spans,
            }
        )
    # small clusters that found no main neighbor stay as their own lines
    small_ids = {id(s["spans"]) for s in smalls}
    for m in metas:
        if id(m["spans"]) in small_ids and not any(
            m["spans"][0] is s for mn in out for s in mn["spans"]
        ):
            out.append({"spans": m["spans"], "bbox": (m["x0"], m["y0"], m["x1"], m["y1"])})
    out = _merge_environment_clusters(out)
    # iter-127b: bar-evidenced fraction stacks — \frac{num}{den} carries
    # NO tall delimiter, so the env merge never sees it; band clustering
    # splits the stacked rows across clusters and the display assembles
    # in the wrong structure. No try/except here: a bug in the merge
    # must crash loudly, not no-op through screens (the NameError
    # incident of commit 6d8d6b8 — silent skip made every screen
    # vacuously flat).
    out = _merge_fraction_clusters(out, page)
    out.sort(key=lambda c: c["bbox"][1])
    return out


# Tall-delimiter PUA pieces in MathExtension/CMEX fonts (cmex10 layout,
# evidence: 2503.04620_pg35 bmatrix + three cases constructs, 2026-08-16).
_PUA_LEFT_BRACE = frozenset("\uf8f1\uf8f2\uf8f3\uf8f4")
_PUA_LEFT_SQUARE = frozenset("\uf8ee\uf8ef\uf8f0")


def _dollarless(spans: list, body: float | None = None) -> str:
    """Assemble spans via the standard line builder, strip $...$ wrappers."""
    if not spans:
        return ""
    if body is None:
        sizes = sorted(s.get("size", 10.0) for s in spans)
        body = sizes[len(sizes) // 2]
    orgs = sorted(s["origin"][1] for s in spans if s.get("size", 10.0) >= body * 0.85)
    base = orgs[len(orgs) // 2] if orgs else spans[0]["origin"][1]
    t = _build_line_with_math(spans, body, base)
    return t.strip().strip("$").strip()

def _line_body_and_baseline(spans: list[dict]) -> tuple[float, float]:
    """(body_size, baseline_y) for one visual line.

    Two-stage estimator (iter-25):
    1. Dominant baseline by SPAN COUNT — a visual line's body baseline
       carries the most spans; scripts sit at ±dy offsets (fewer spans),
       and a foreign span merged into the cluster (page header at 20pt
       on a 10pt line, 2503.04077) sits ALONE at its own origin.
       Char-weighting alone would let one long header span outvote the
       whole math line.
    2. Body size = CHAR-weighted median WITHIN that baseline cohort —
       script-dense inline lines carry more script spans than body
       spans (27 vs 25, 2503.05506_pg11), so the span-count median
       collapses onto script size and scripts emit at face value
       ('τ 2 i'). Body tokens own the character mass.
    """
    from collections import Counter

    if not spans:
        return 10.0, 0.0
    tally: Counter = Counter()
    chars: Counter = Counter()
    for s in spans:
        y = round(s["origin"][1])
        tally[y] += 1
        chars[y] += max(len(s["text"].strip()), 1)
    # most spans; ties → more chars; still tied → the UPPER line
    dom_y = max(tally, key=lambda y: (tally[y], chars[y], -y))
    # Stage 0 (iter-33): DISPLAY-EQUATION guard. A display equation's
    # limit bands and script runs OUTNUMBER its body spans — the
    # dominant-by-count baseline lands on a SCRIPT band (09577 cluster 2:
    # base=113.51 @8pt while the body sits at 117.47 @10.9pt). Big glyphs
    # define the line: when a strictly LARGER size cohort exists near the
    # dominant baseline (proximity-bounded, so a far-away foreign span —
    # the 20pt page header 184pt off, 04077 — cannot win), prefer it.
    _dom_size = max(
        (s["size"] for s in spans if round(s["origin"][1]) == dom_y), default=0.0
    )
    _big = sorted(
        {round(s["size"], 2) for s in spans if s["size"] > _dom_size * 1.15}
    )
    if _big:
        _target = _big[-1]
        _cand = [
            s
            for s in spans
            if s["size"] >= _target * 0.9
            and abs(s["origin"][1] - dom_y) <= 2.5 * max(_dom_size, 1.0)
        ]
        if len(_cand) >= 2:
            _ct: Counter = Counter(round(s["origin"][1]) for s in _cand)
            _cchars: Counter = Counter()
            for s in _cand:
                _cchars[round(s["origin"][1])] += max(len(s["text"].strip()), 1)
            _cy = max(_ct, key=lambda y: (_ct[y], _cchars[y], -y))
            cohort = [s for s in _cand if round(s["origin"][1]) == _cy]
        else:
            cohort = [s for s in spans if round(s["origin"][1]) == dom_y]
    else:
        cohort = [s for s in spans if round(s["origin"][1]) == dom_y]
    _chars = sorted(
        s["size"] for s in cohort for _ in range(max(len(s["text"].strip()), 1))
    )
    body = _chars[len(_chars) // 2] if _chars else spans[0]["size"]
    orgs = sorted(s["origin"][1] for s in cohort)
    return body, orgs[len(orgs) // 2]


_ENV_FOR_WORDS = ("for", "when", "if", "unless", "otherwise")

# Spacing-diacritic spans (roman font) that TeX places as separate overstrike
# glyphs; pairing = accent command around the overlapped base token.
_ENV_ACCENT_CMDS = {
    "˜": r"\tilde",
    "¯": r"\bar",
    "˙": r"\dot",
    "¨": r"\ddot",
    "ˆ": r"\hat",
    "ˇ": r"\check",
    "˘": r"\breve",
}


def _pair_accents(spans: list) -> tuple[list, list]:
    """Drop overstrike accent spans; return (kept_spans, [(cmd, base_text)]).

    An accent span: text == one accent char, roman font, sitting ABOVE its
    base span's baseline with x-overlap (measured: '˜' at y-4 over CMMI 'a',
    2503.04620_pg35 gold \\tilde{a})."""
    accents = [s for s in spans if s.get("text", "").strip() in _ENV_ACCENT_CMDS]
    kept = [s for s in spans if s not in accents]
    # TeX sometimes merges digit+accent into one span ('2˜'): strip the
    # accent glyph, pair it with the next x-neighbor as base.
    for s in kept:
        t = s.get("text", "")
        if 1 < len(t) <= 3 and t[-1] in _ENV_ACCENT_CMDS:
            s["text"] = t[:-1]
            accents.append({**s, "text": t[-1]})
    if not accents:
        return spans, []
    kept = [s for s in kept if s.get("text", "").strip()]
    pairs = []
    for a in accents:
        base = None
        best_ov = 0.0
        for s in kept:
            if s["bbox"] == a["bbox"]:
                continue  # the digit the accent glyph was merged INTO
            ov = min(a["bbox"][2], s["bbox"][2]) - max(a["bbox"][0], s["bbox"][0])
            if ov <= 0:
                continue
            # accent sits above the base's origin (small tolerance: TeX
            # overstrikes may dip 1-2pt past it at small sizes)
            if a["bbox"][3] > s.get("origin", s["bbox"])[1] + 2.5:
                continue
            if ov > best_ov:
                best_ov = ov
                base = s
        if base is not None:
            pairs.append((_ENV_ACCENT_CMDS[a["text"].strip()], base["text"].strip()))
    return kept, pairs


def _apply_accent_pairs(row: str, pairs: list) -> str:
    for cmd, base in pairs:
        row = re.sub(
            rf"(?<![\\a-zA-Z{{]){re.escape(base)}(?![a-zA-Z])",
            lambda m, c=cmd, b=base: c + "{" + b + "}",
            row,
            count=1,
        )
    return row


def _env_row_spans(spans: list) -> tuple[list, list]:
    r"""Sort by x, drop equation-number spans '(\d+)' (gold drops them;
    aligns with iter-6 inline decision), pair accents."""
    spans = sorted((s for s in spans if s.get("text", "").strip()), key=lambda s: s["bbox"][0])
    spans = [s for s in spans if not re.fullmatch(r"\.?\s*\(\d+\)\s*\.?", s["text"].strip())]
    return _pair_accents(spans)


def _env_cases_row(spans: list, body: float | None = None) -> str:
    """One cases row: 'value , & \\text{for } condition' (gold pattern:
    \\delta_{k,j-1}, & \\text{for } k = 1, 2, \\ldots, 2r-1 ,\\\\)."""
    spans, acc = _env_row_spans(spans)
    k = None
    for i, s in enumerate(spans):
        if s["text"].strip().lower() in _ENV_FOR_WORDS and not _STRICT_MATH_FONT_RE.search(
            s.get("font", "")
        ):
            k = i
            break
    if k is None:
        return _apply_accent_pairs(_value_with_frac(spans, body), acc)
    word = spans[k]["text"].strip().lower()
    val = _value_with_frac(spans[:k], body)
    rest = _dollarless(spans[k + 1 :])
    cond = rf"\text{{{word}}} {rest}".strip()
    return _apply_accent_pairs(f"{val} & {cond}", acc)


def _value_with_frac(spans: list, body: float | None = None) -> str:
    r"""Value part of a cases row: if its spans form two y-bands with
    x-overlap (stacked dfrac numerator/denominator), emit
    \dfrac{top}{bottom} (leading '-' stays outside).

    ``body`` = the ROW's body size (threaded from the environment merge —
    a value slice like [10, 7, 7, 7] has median 7, which misclassifies
    every script as body size and loses subscripts, measured iter-10)."""
    if not spans:
        return ""
    ycs = sorted((s["bbox"][1] + s["bbox"][3]) / 2 for s in spans)
    spread = ycs[-1] - ycs[0]
    if body is None:
        sizes = sorted(s.get("size", 10.0) for s in spans)
        body = sizes[len(sizes) // 2]
    if spread <= 0.9 * body:
        return _dollarless(spans, body)
    # Split by ORIGIN (baseline), not bbox-center: spine glyphs (',',
    # 'for') share y-centers with numerators but their origins sit a full
    # body lower (measured iter-10: center-split put ',' in the numerator
    # and flipped h_{j-1} -> h^{j-1} via a polluted baseline).
    # Largest-GAP split on origins: TeX stacks = tight baseline clusters
    # (numerator ~315, spine ~322, denominator ~327); a fixed mid-formula
    # lands arbitrarily inside a cluster (measured: mid 322.37 vs spine
    # 322.07 put the leading '-' in the numerator). The biggest gap between
    # consecutive origins is the true band boundary.
    org_span = sorted((s.get("origin", s["bbox"])[1], i) for i, s in enumerate(spans))
    _best_gap, _cut = 0.0, None
    for (_a, _ia), (_b, _ib) in zip(org_span, org_span[1:]):
        if _b - _a > _best_gap:
            _best_gap, _cut = _b - _a, _b
    if _cut is None or _best_gap < 0.25 * body:
        return _dollarless(spans, body)
    top = [s for s in spans if s.get("origin", s["bbox"])[1] < _cut]
    bot = [s for s in spans if s.get("origin", s["bbox"])[1] >= _cut]
    if not top or not bot:
        return _dollarless(spans, body)
    # Denominator x-window: the frac column (numerator x-range +/- 1.5
    # body). Spine glyphs right of it (',') are row tail, not denominator.
    if top:
        _nx0 = min(s["bbox"][0] for s in top)
        _nx1 = max(s["bbox"][2] for s in top)
        _win_bot = [
            s for s in bot
            if s["bbox"][0] >= _nx0 - 1.5 * body and s["bbox"][2] <= _nx1 + 1.5 * body
        ]
        _spine = [s for s in bot if s not in _win_bot]
    else:
        _win_bot, _spine = bot, []
    # Leading sign (gold '-\dfrac{...}'): a bare '-'/'+' at the LEFT edge
    # of the value (x-left of every numerator span) — typographic truth,
    # robust to band-mid ambiguity (iter-10).
    lead = ""
    xsorted = sorted(spans, key=lambda s: s["bbox"][0])
    num_x0 = min(s["bbox"][0] for s in top)
    sign_cands = [
        s for s in xsorted[:2]
        if s["text"].strip() in ("-", "+") and s["bbox"][2] <= num_x0 + 1.5
    ]
    if sign_cands:
        m = sign_cands[0]
        lead = m["text"].strip() + " "
        top = [s for s in top if s is not m]
        bot = [s for s in bot if s is not m]
    t = _dollarless(top, body)
    b = _dollarless(_win_bot, body)
    if not t or not b:
        return _dollarless(spans, body)
    # Spine glyphs (',') after the frac (gold: ', & \text{for}')
    tail = _dollarless(_spine, body) if _spine else ""
    return f"{lead}\\dfrac{{{t}}}{{{b}}}{tail}"

def _env_matrix_row(spans: list, body: float) -> str:
    """One matrix row: cells split on x-gaps > 1.2*body, joined with ' &'."""
    if not spans:
        return ""
    cells: list[list] = [[spans[0]]]
    for s in spans[1:]:
        if s["bbox"][0] - cells[-1][-1]["bbox"][2] > 1.2 * body:
            cells.append([s])
        else:
            cells[-1].append(s)
    return _apply_accent_pairs(" & ".join(_dollarless(c) for c in cells), {})


def _merge_environment_clusters(clusters: list) -> list:
    """Detect tall LEFT delimiters (stacked ext-glyph pieces) spanning >= 2
    row bands; consume the row/LHS spans to their right; emit one synthetic
    environment cluster carrying prebuilt LaTeX.

    Span-level (2026-08-16): band clustering entangles delimiter pieces,
    rows and LHS inside shared clusters (measured 2503.04620_pg35: cluster
    = 'B =' + tall-brace pieces + row digits + right-brace pieces), so the
    environment is carved from RAW spans, not whole clusters.
    """
    sizes = [
        s["size"]
        for cl in clusters
        for s in cl["spans"]
        if s.get("text", "").strip() and not any("\uf8e0" <= c <= "\uf8ff" for c in s["text"])
    ]
    if not sizes:
        return clusters
    body = sorted(sizes)[len(sizes) // 2]

    def _is_pua(s) -> bool:
        t = (s.get("text") or "").strip()
        return bool(t) and all("\uf8e0" <= c <= "\uf8ff" for c in t)

    def _is_left_delim(s) -> str | None:
        t = (s.get("text") or "").strip()
        if not t:
            return None
        if set(t) <= _PUA_LEFT_BRACE:
            return "cases"
        if set(t) <= _PUA_LEFT_SQUARE:
            return "bmatrix"
        return None

    pieces = [
        (ci, s, kind)
        for ci, cl in enumerate(clusters)
        for s, kind in ((s, _is_left_delim(s)) for s in cl["spans"])
        if kind
    ]
    if not pieces:
        return clusters

    groups = []
    for ci, s, kind in sorted(pieces, key=lambda p: (p[2], p[1]["bbox"][0], p[1]["bbox"][1])):
        x0, y0, y1 = s["bbox"][0], s["bbox"][1], s["bbox"][3]
        for g in groups:
            if (
                g["kind"] == kind
                and abs(g["x0"] - x0) < 5
                and not (y1 < g["y0"] - 1.2 * body or y0 > g["y1"] + 1.2 * body)
            ):
                g["x0"] = min(g["x0"], x0)
                g["x1"] = max(g["x1"], s["bbox"][2])
                g["y0"] = min(g["y0"], y0)
                g["y1"] = max(g["y1"], y1)
                g["cis"].add(ci)
                g["spans"].append(s)
                break
        else:
            groups.append(
                {
                    "kind": kind,
                    "x0": x0,
                    "x1": s["bbox"][2],
                    "y0": y0,
                    "y1": y1,
                    "cis": {ci},
                    "spans": [s],
                }
            )

    all_spans_with_ci = [(ci, s) for ci, cl in enumerate(clusters) for s in cl["spans"]]
    consumed: set[int] = set()
    synthetics = []
    for g in groups:
        if g["y1"] - g["y0"] < 1.6 * body:
            continue
        lo, hi = g["y0"] + 0.3 * body, g["y1"]
        # right edge: tallest PUA stack to the right within y-range
        right_pua = [
            s
            for _, s in all_spans_with_ci
            if _is_pua(s)
            and s["bbox"][0] >= g["x1"] - 1.5
            and lo <= (s["bbox"][1] + s["bbox"][3]) / 2 <= hi + 2 * body
        ]
        right_x = max((s["bbox"][2] for s in right_pua), default=None)

        def _yc(s):
            return (s["bbox"][1] + s["bbox"][3]) / 2

        row_spans = sorted(
            (
                s
                for _, s in all_spans_with_ci
                if not _is_pua(s)
                and s.get("text", "").strip()
                and s["bbox"][0] >= g["x1"] - 1.5
                and (right_x is None or s["bbox"][2] <= right_x + 2)
                and lo < _yc(s) <= hi + 0.5 * body
            ),
            key=_yc,
        )
        if len(row_spans) < 2:
            continue
        # band into rows (y-center gap > 0.6*body starts a new row)
        rows: list[list] = [[row_spans[0]]]
        for s in row_spans[1:]:
            if _yc(s) - _yc(rows[-1][-1]) > 0.6 * body:
                rows.append([s])
            else:
                rows[-1].append(s)
        if len(rows) < 2:
            continue
        # LHS: spans left of the delimiter in y-range, assembled text ends '='
        lhs_spans = sorted(
            (
                s
                for _, s in all_spans_with_ci
                if not _is_pua(s)
                and s.get("text", "").strip()
                and s["bbox"][2] <= g["x0"] + 2
                and abs(_yc(s) - (g["y0"] + g["y1"]) / 2) < 1.5 * body
            ),
            key=lambda s: s["bbox"][0],
        )
        lhs = _dollarless(lhs_spans) if lhs_spans else ""
        if not lhs.endswith("="):
            lhs = ""
        # consume every cluster that contributed spans to the construct
        env_span_ids = {id(s) for s in g["spans"]} | {id(s) for r in rows for s in r}
        if lhs:
            env_span_ids |= {id(s) for s in lhs_spans}
        for ci, cl in enumerate(clusters):
            if ci in g["cis"]:
                consumed.add(ci)
                continue
            live = [s for s in cl["spans"] if s.get("text", "").strip()]
            if live and sum(1 for s in live if id(s) in env_span_ids) >= 0.6 * len(live):
                consumed.add(ci)
        if g["kind"] == "cases":
            row_texts = [_env_cases_row(r, body) for r in rows]
        else:
            row_texts = [_env_matrix_row(r, body) for r in rows]
        env = f"\\begin{{{g['kind']}}} " + " \\\\ ".join(row_texts) + f" \\end{{{g['kind']}}}"
        spans_all = list(g["spans"]) + [s for r in rows for s in r]
        xs = [s["bbox"][0] for s in spans_all] + [s["bbox"][2] for s in spans_all]
        ys = [s["bbox"][1] for s in spans_all] + [s["bbox"][3] for s in spans_all]
        synthetics.append(
            {
                "spans": spans_all,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "den_spans": None,
                "env_latex": env,
                "env_lhs": lhs,
            }
        )
    if not synthetics:
        return clusters
    out = [cl for ci, cl in enumerate(clusters) if ci not in consumed]
    out.extend(synthetics)
    out.sort(key=lambda c: c["bbox"][1])
    return out

def _merge_fraction_clusters(clusters: list, page: Any) -> list:
    """Bar-evidenced fraction stacks (iter-127b).

    A TeX \\frac bar renders as a thin — often ZERO-height — fill rule
    (measured 2503.04993_pg18: rule (253.3,217.2)-(263.4,217.2), the
    'd'/'J_1(...)'/'d\\epsilon' three-band split of gold
    \\frac{d}{d\\epsilon} J_1(...)). For every thin rule, PROSE-FREE
    spans directly above (numerator) and below (denominator) inside the
    rule's x-window rebuild as one synthetic ``span['frac']`` carrier —
    the stage-3 mechanism _build_line_with_math consumes — inserted
    into the row cluster on the rule's baseline (the spine; inline
    fractions inside prose lines included).

    Gates against the measured false families (plot axes, ruled boxes
    around captions, footnotes under rules — 7 passing pages carried
    naive pairs): rows must be math-font/operand-only spans (no 2+
    letter prose words), num and den must live in DIFFERENT clusters
    (an in-cluster split already has the stage-3 machinery), a
    same-x-extent stack of >= 4 rules at tight y-pitch is a ruled
    TABLE grid and disables the page, and the spine's baseline must
    sit within 0.9*body of the rule. Leftover spans of a consumed
    denominator cluster (the \\big| evaluation subscript glued into the
    den band) rejoin the spine as sub-band scripts.

    Kill-switch: PDF_OCR_FRAC_MERGE=0. Debug: PDF_OCR_DEBUG_FRAC=1.
    """
    import os as _os

    if _os.environ.get("PDF_OCR_FRAC_MERGE", "1") != "1":
        return clusters
    live = [s for cl in clusters for s in cl["spans"] if s.get("text", "").strip()]
    if not live:
        return clusters
    sizes = sorted(s["size"] for s in live)
    body = sizes[len(sizes) // 2]

    try:
        drawings = page.get_drawings()
    except Exception as e:  # page without drawing support
        log.debug("fraction merge: no drawings (%s)", e)
        return clusters
    bars = [
        (r.x0, r.y0 + r.height / 2, r.x1)
        for dr in drawings
        if (r := dr.get("rect")) is not None
        and r.height <= max(1.0, 0.10 * body)
        and r.width >= 0.8 * body
    ]
    if not bars:
        return clusters

    # Ruled TABLE grid: >= 4 rules sharing an x-extent at <= 2.5*body
    # y-pitch is a border grid, never a fraction stack family.
    def _x_overlap_kill(b1, b2) -> bool:
        return min(b1[2], b2[2]) - max(b1[0], b2[0]) > 0.6 * min(
            b1[2] - b1[0], b2[2] - b2[0]
        )

    for i, a in enumerate(bars):
        column = [b for b in bars if b is not a and _x_overlap_kill(a, b)]
        if len(column) < 3:
            continue
        ys = sorted([a[1]] + [b[1] for b in column])
        if all(ys[k + 1] - ys[k] <= 2.5 * body for k in range(len(ys) - 1)):
            log.debug("fraction merge: ruled grid detected (%d bars) — skipped", len(ys))
            return clusters

    env_owners = {
        id(cl) for cl in clusters if cl.get("env_latex") is not None
    }
    owner = {id(s): cl for cl in clusters for s in cl["spans"]}

    def _pure(s) -> bool:
        if _MATH_FONT_RE.search(s.get("font", "")):
            return True
        return _is_operand_span((s.get("text") or "").strip())

    def _cx(s):
        return (s["bbox"][0] + s["bbox"][2]) / 2

    debug = _os.environ.get("PDF_OCR_DEBUG_FRAC")
    used: set[int] = set()
    _WORD_RE = re.compile(r"[a-z]{3,}")
    owner_has_word = {
        id(cl): any(
            _WORD_RE.search(t := (s.get("text") or "").strip())
            and not _MATH_FONT_RE.search(s.get("font", ""))
            and t not in _TEXT_IN_MATH_WORDS
            and t not in _MATH_WORDS
            and t not in _FUNC_CMD_WORDS
            for s in cl["spans"]
        )
        for cl in clusters
    }
    inserts: dict[int, list] = {}
    for bx0, by, bx1 in sorted(bars, key=lambda b: (b[1], b[0])):
        lo, hi = bx0 - 0.6 * body, bx1 + 0.6 * body
        def _nearest_band(spans, above: bool):
            """Closest coherent y-band to the bar — the frac row itself.

            The x-window can reach a prose line one band further out
            (measured 2503.07421_pg26: '∈ L ( M,' from the text line
            above polluted the numerator). Greedy 0.45*body banding,
            nearest band wins.
            """
            if not spans:
                return []
            ys = sorted(
                ((s, (s["bbox"][1] + s["bbox"][3]) / 2) for s in spans),
                key=lambda t: t[1],
            )
            bands = [[ys[0]]]
            for s, y in ys[1:]:
                if y - bands[-1][-1][1] > 0.45 * body:
                    bands.append([(s, y)])
                else:
                    bands[-1].append((s, y))
            band = bands[0] if above else bands[-1]
            return [s for s, _ in band]

        num_spans = _nearest_band(
            sorted(
                (
                    s
                    for s in live
                    if id(s) not in used
                    and lo <= _cx(s) <= hi
                    and s["bbox"][3] <= by + 0.5
                    and by - s["bbox"][3] <= 2.2 * body
                    and _pure(s)
                    and id(owner[id(s)]) not in env_owners
                ),
                key=lambda s: s["bbox"][0],
            ),
            above=True,
        )
        den_spans = _nearest_band(
            sorted(
                (
                    s
                    for s in live
                    if id(s) not in used
                    and lo <= _cx(s) <= hi
                    and s["bbox"][1] >= by - 0.5
                    and s["bbox"][1] - by <= 2.2 * body
                    and _pure(s)
                    and id(owner[id(s)]) not in env_owners
                ),
                key=lambda s: s["bbox"][0],
            ),
            above=False,
        )
        if not num_spans or not den_spans:
            continue
        num_owners = {id(owner[id(s)]) for s in num_spans}
        den_owners = {id(owner[id(s)]) for s in den_spans}
        if len(num_owners) != 1 or len(den_owners) != 1 or num_owners & den_owners:
            continue  # the measured family: num and den are each ONE
            # whole distinct cluster (scattered fragments of an already
            # assembled display are not this family)
        # owner clusters must carry NO prose word — a pure span plucked
        # from a math-heavy PROSE line ('\in L ( M,' from 'Suppose
        # H ∈ L(M,…' measured 2503.07421) never becomes a frac side
        if owner_has_word[next(iter(num_owners))] or owner_has_word[next(iter(den_owners))]:
            continue
        # a body-size span straddling the rule's axis INSIDE the bar's
        # own x-range means the rule sits WITHIN a live text line
        # (underline / shattered fragment, not an isolated frac column)
        if any(
            s["size"] >= 0.8 * body
            and s["bbox"][2] - s["bbox"][0] > 0.3 * body
            and bx0 <= (s["bbox"][0] + s["bbox"][2]) / 2 <= bx1
            and s["bbox"][1] < by < s["bbox"][3]
            and id(s) not in used
            for s in live
        ):
            continue

        # _stack_group_text wraps each fragmented script span separately
        # ('^{1} ^{-} ^{p}' — measured 2503.04590_pg7); adjacent
        # same-direction groups are ALWAYS invalid LaTeX (double
        # superscript), consolidating them can only help the render.
        def _consolidate(s: str) -> str:
            prev = None
            while prev != s:
                prev = s
                s = re.sub(r"(\^)\{([^{}]*)\}\s*(\^)\{([^{}]*)\}", r"\1{\2 \4}", s)
                s = re.sub(r"(_)\{([^{}]*)\}\s*(_)\{([^{}]*)\}", r"\1{\2 \4}", s)
            return s

        num_txt = _consolidate(_stack_group_text(num_spans, _line_body_and_baseline(num_spans)[1]))
        den_txt = _consolidate(_stack_group_text(den_spans, _line_body_and_baseline(den_spans)[1]))
        if not num_txt or not den_txt or num_txt == den_txt:
            continue  # identical sides = band-selection artifact, never
            # a real fraction (measured 2503.04993 I1 display)

        # spine: nearest-baseline cluster at the rule's axis, excluding
        # the den owners (their baseline sits below) and prebuilt envs.
        cand = []
        for cl in clusters:
            if id(cl) in den_owners or id(cl) in env_owners or cl.get("den_spans"):
                continue
            if cl["bbox"][2] <= bx0 - 2.0:
                continue  # entirely left of the bar
            try:
                _, base = _line_body_and_baseline(cl["spans"])
            except Exception:
                continue
            if abs(base - by) <= 0.9 * body:
                cand.append((abs(base - by), cl["bbox"][0], cl))
        spine = min(cand, key=lambda t: (t[0], t[1]))[2] if cand else None
        if spine is None:
            # no row on the rule's axis: fragments only — emitting a
            # standalone $$\frac{}{}$$ from fragments measured as pure
            # garbage (2503.07421 y465 rules), never merge
            continue

        car = dict(num_spans[0])
        car["frac"] = (num_txt, den_txt)
        car["bbox"] = (bx0, by - 0.30 * body, bx1, by + 0.30 * body)
        car["size"] = body
        car["origin"] = (bx0, by)
        # every source span is consumed — the carrier copy carries the
        # content (leaving num_spans[0] alive emits a stray '$d$' line).
        used.update(id(s) for s in num_spans + den_spans)
        inserts.setdefault(id(spine), []).append(car)
        # small leftover fragments of consumed den clusters (the
        # evaluation subscript glued into the den band) rejoin the
        # spine as sub-band scripts; larger tails stay put.
        for doid in den_owners:
            docl = next(c for c in clusters if id(c) == doid)
            tail = [
                s
                for s in docl["spans"]
                if id(s) not in used and s.get("text", "").strip()
            ]
            if len(tail) <= 4:
                inserts[id(spine)].extend(tail)
                used.update(id(s) for s in tail)
        if debug:
            log.warning(
                "FRAC-MERGE bar=(%.1f,%.1f) num=%r den=%r",
                bx0, by, num_txt[:40], den_txt[:40],
            )

    # SQRT RADICAND PAIRING (iter-133): the \\sqrt overline is the SAME
    # zero-height fill-rule family as \\frac bars (measured 2503.05358:
    # rules at the radical glyph top, w≈radicand). 108 census equations
    # dangle a bare \\sqrt (KaTeX "Expected group argument") because the
    # radicand spans band elsewhere. Pair: rule starting at the glyph,
    # riding its top band -> pure spans right of the glyph under the
    # rule become span['sqrt'] consumed by _build_line_with_math.
    if _os.environ.get("PDF_OCR_SQRT_PAIR", "1") == "1":
        radicals = []
        for s in live:
            if id(s) in used or s.get("sqrt"):
                continue
            t = (s.get("text") or "").strip()
            tok = _span_to_math_token(t, False, False, s.get("font", "")) or ""
            if "√" in t or "sqrt" in tok:
                radicals.append(s)
        radical_ids = {id(s) for s in radicals}
        for rs in sorted(
            radicals, key=lambda s: s["bbox"][2] - s["bbox"][0]
        ):
            rx0, ry0, rx1, _ry1 = rs["bbox"]
            cand_rules = [
                b
                for b in bars
                if rx0 - 1 <= b[0] <= rx1 + 2
                and ry0 - 0.3 * body <= b[1] <= ry0 + 1.2 * body
                and b[2] - b[0] >= 0.4 * body
            ]
            radicand = []
            if cand_rules:
                bx0, by, bx1 = max(cand_rules, key=lambda b: b[2] - b[0])
                radicand = [
                    s
                    for s in live
                    if s is not rs
                    and id(s) not in used
                    and id(s) not in radical_ids
                    and _pure(s)
                    and rx1 - 1 <= _cx(s)
                    and s["bbox"][2] <= bx1 + 0.5
                    and by - 0.2 * body <= (s["bbox"][1] + s["bbox"][3]) / 2 <= by + 1.8 * body
                ]
                # boundary ambiguity: a span CROSSING the rule end (the
                # glued 'πrT' case) means the radicand edge is intra-span
                # — a wrong-but-renderable \sqrt{2} would BLOCK the
                # cascade's oracle repair (measured 05717 pg14). Skip.
                crossing = [
                    s
                    for s in live
                    if s is not rs
                    and id(s) not in radical_ids
                    and _pure(s)
                    and rx1 - 1 <= s["bbox"][0]
                    and s["bbox"][2] > bx1 + 0.3 * body
                    and by - 0.2 * body <= (s["bbox"][1] + s["bbox"][3]) / 2 <= by + 1.8 * body
                ]
                if crossing:
                    # PAREN-EVIDENCE SPLIT (2026-08-31, 2503.05218_pg5
                    # Conjecture 2.1): MuPDF can glue the radicand's
                    # closing ')' to the following tail ('+ 2γ − 2' →
                    # ')+ ') into ONE span crossing the rule end. When
                    # the under-rule radicand has unclosed '(' and the
                    # leftmost crossing span LEADS with ')', those
                    # closers belong under the rule: consume them as a
                    # synthetic radicand span and trim the carrier in
                    # place (bbox narrowed proportionally). No paren
                    # evidence (glued 'πrT', iter-133 / 05717 pg14)
                    # keeps the full veto.
                    rad_probe = "".join(s.get("text") or "" for s in radicand)
                    unclosed = rad_probe.count("(") - rad_probe.count(")")
                    split_done = False
                    if unclosed > 0 and len(crossing) >= 1:
                        cr = min(crossing, key=lambda s: s["bbox"][0])
                        ct = cr.get("text") or ""
                        k = 0
                        while k < unclosed and k < len(ct) and ct[k] == ")":
                            k += 1
                        if k:
                            cx0, cy0, cx1, cy1 = cr["bbox"]
                            cw = (cx1 - cx0) * k / max(len(ct), 1)
                            radicand.append(
                                {
                                    "text": ct[:k],
                                    "bbox": (cx0, cy0, cx0 + cw, cy1),
                                    "font": cr.get("font", ""),
                                    "size": cr.get("size", body),
                                    "origin": cr.get("origin", (cx0, cy1 - 2.0)),
                                }
                            )
                            cr["text"] = ct[k:]
                            cr["bbox"] = (cx0 + cw, cy0, cx1, cy1)
                            if not cr["text"].strip():
                                used.add(id(cr))
                            split_done = True
                    if not split_done:
                        radicand = []
            else:
                # NO-RULE FALLBACK: x-contiguous pure spans in the glyph's
                # own y-band (rule-less radicals — index glyphs, lost
                # rules). VETO when the row continues at ~zero gap right
                # of the candidate radicand (measured 05717 pg14: MuPDF
                # glues 'πrT' into one span; the pairing ate T and the
                # renderable-but-wrong \sqrt{2πrT} BLOCKED the cascade's
                # oracle repair that had fixed this line before).
                ymid = (ry0 + _ry1) / 2
                radicand = [
                    s
                    for s in live
                    if s is not rs
                    and id(s) not in used
                    and id(s) not in radical_ids
                    and _pure(s)
                    and 0 <= s["bbox"][0] - rx1 <= 0.35 * body
                    and abs((s["bbox"][1] + s["bbox"][3]) / 2 - ymid) <= 0.6 * body
                ][:6]
                if radicand:
                    x_end = max(s["bbox"][2] for s in radicand)
                    y_lo = min(s["bbox"][1] for s in radicand) - 0.3 * body
                    y_hi = max(s["bbox"][3] for s in radicand) + 0.3 * body
                    cont = [
                        s
                        for s in live
                        if id(s) not in used
                        and id(s) not in radical_ids
                        and s is not rs
                        and 0 <= s["bbox"][0] - x_end <= 0.15 * body
                        and y_lo <= (s["bbox"][1] + s["bbox"][3]) / 2 <= y_hi
                        and s.get("size", body) >= 0.7 * body
                    ]
                    if cont:
                        radicand = []
            if not radicand:
                continue
            radicand.sort(key=lambda s: s["bbox"][0])
            rtxt = _stack_group_text(
                radicand, _line_body_and_baseline(radicand)[1]
            )
            # trailing spine operators/commas ride OUTSIDE the radical
            rtxt = re.sub(r"[\s\+\-=,;\.\−–—]+$", "", rtxt).strip()
            if not rtxt:
                continue
            rs["sqrt"] = rtxt
            used.update(id(s) for s in radicand)
            if debug:
                log.warning("SQRT-PAIR at (%.0f,%.0f) radicand=%r", rx0, ry0, rtxt[:40])

    if not used and not inserts:
        return clusters
    out = []
    for cl in clusters:
        spans = [s for s in cl["spans"] if id(s) not in used]
        spans += inserts.get(id(cl), [])
        if not spans:
            continue
        xs = [s["bbox"][0] for s in spans] + [s["bbox"][2] for s in spans]
        ys = [s["bbox"][1] for s in spans] + [s["bbox"][3] for s in spans]
        out.append(
            {
                "spans": spans,
                "bbox": (min(xs), min(ys), max(xs), max(ys)),
                "den_spans": cl.get("den_spans"),
            }
        )
    return out


def extract_text_with_math_fonts(page: Any) -> str:
    """Extract text from a PyMuPDF page with font-aware math detection.

    Returns markdown text where math regions are wrapped in $...$ with
    LaTeX commands. Non-math text is preserved as-is.

    Args:
        page: PyMuPDF page object.

    Returns:
        Markdown text for the whole page.
    """
    global _ACTIVE_CMEX_GLYPHS, _PAGE_PHI_DUAL
    _saved = _ACTIVE_CMEX_GLYPHS
    _ACTIVE_CMEX_GLYPHS = load_cmex_glyph_map(page) or _saved
    _saved_phi = _PAGE_PHI_DUAL
    _PAGE_PHI_DUAL = _detect_phi_dual(page)
    try:
        clusters = collect_math_clusters(page)
        output_lines: list[str] = []
        for cl in clusters:
            # (x0, origin.y): deterministic left-to-right, and at equal
            # x the UPPER glyph first — a base's sup and sub share the
            # x window (τ^2_i); sup-first lets the nested attach build
            # the composite before the sub arrives (iter-26).
            spans = sorted(cl["spans"], key=lambda s: (s["bbox"][0], s["origin"][1]))
            body_size, baseline_y = _line_body_and_baseline(spans)

            if cl.get("env_latex") is not None:
                # Synthetic multi-row environment (cases/bmatrix): prebuilt.
                line_text = f"$${cl.get('env_lhs', '')} {cl['env_latex']}$$"
            else:
                line_text = _build_line_with_math(
                    spans, body_size, baseline_y, den_spans=cl.get("den_spans")
                )
            if line_text.strip():
                output_lines.append(line_text)

        return "\n".join(output_lines)
    finally:
        _PAGE_PHI_DUAL = _saved_phi


def _stack_token(s: dict) -> str:
    """One span → LaTeX token for stack num/den assembly."""
    return _span_to_math_token(s["text"], False, False, s.get("font", "")) or s["text"].strip()


def _stack_group_text(spans: list[dict], baseline_y: float) -> str:
    """Assemble an x-sorted span group into LaTeX with script attachment.

    Small spans adjacent (in x) to a body span become its sub/superscript
    (dy vs baseline picks which) — enough for stack groups like ``∂H`` +
    ``1`` → ``\\partial H_{1}``.
    """
    xs = sorted((s for s in spans if s.get("text", "").strip()), key=lambda s: s["bbox"][0])
    if not xs:
        return ""
    # GROUP-local baseline: a merged cluster's global baseline sits on the
    # denominator line — using it would flip every numerator subscript to
    # a superscript (measured ∂H^{1} vs ∂H_{1} on 2503.04993_pg18).
    g_sizes = sorted(s.get("size", 10.0) for s in xs)
    g_body = g_sizes[len(g_sizes) // 2]
    g_orgs = sorted(s["origin"][1] for s in xs if s.get("size", 10.0) >= g_body * 0.85)
    g_base = g_orgs[len(g_orgs) // 2] if g_orgs else baseline_y
    parts: list[str] = [_stack_token(xs[0])]
    for s in xs[1:]:
        tok = _stack_token(s)
        if not tok:
            continue
        if s.get("size", 10.0) < g_body * 0.8 and parts:
            oy = s.get("origin", (0, g_base))[1]
            script = "^" if oy < g_base - 0.5 else "_"
            parts[-1] = f"{parts[-1]}{script}{{{tok}}}"
        else:
            parts.append(tok)
    joined = " ".join(parts)
    joined = re.sub(r"\s+(_|\^)\{", r"\1{", joined)
    return joined.replace("{ ", "{").replace(" }", "}")


def _build_stacked_frac(
    spans: list[dict], den_spans: list[dict], body_size: float, baseline_y: float
) -> str:
    """Emit \\frac{num}{den} for a merged numerator-line/denominator-line stack.

    ``baseline_y`` splits RAISED spans (true fraction numerators like the
    ``d`` in d/dϵ) from the baseline spine (``J_1(u_1 + ϵv_1, u_2) = ...``
    continues right of the frac). Two regimes:

    - POINT-FRAC (single narrow denominator group): numerator = raised
      spans in the group's x-window; baseline spans stay spine.
      ``d`` over ``dϵ`` beside ``J_1(...)`` → ``\\frac{d}{d\\epsilon} J_{1}(...)``
    - LINE-FRAC (≥2 groups or a wide den line): the whole top line is the
      numerator; per-group x-windows carve it into per-fraction segments.
    """
    den_ids = {id(s) for s in den_spans}
    dens = sorted(
        (
            s
            for s in den_spans
            if s.get("text", "").strip()
            and (
                _STRICT_MATH_FONT_RE.search(s.get("font", ""))
                # Roman operands are legal denominators too (CMR12 '2',
                # 'X(∂Ω)' — measured 06570 \frac{1}{2} lost to this filter)
                or not re.search(r"[a-z]{3,}", s.get("text", ""))
            )
        ),
        key=lambda s: s["bbox"][0],
    )
    live = [s for s in spans if s.get("text", "").strip()]
    if not dens or not live:
        return ""

    # x-contiguous denominator groups (gap > 2×body separates columns)
    groups: list[dict] = []
    for s in dens:
        if groups and s["bbox"][0] - groups[-1]["x1"] <= 2.0 * body_size:
            groups[-1]["spans"].append(s)
            groups[-1]["x1"] = max(groups[-1]["x1"], s["bbox"][2])
        else:
            groups.append({"spans": [s], "x0": s["bbox"][0], "x1": s["bbox"][2]})
    if not groups:
        return ""

    def _cx(s):
        return (s["bbox"][0] + s["bbox"][2]) / 2

    cluster_x0 = min(s["bbox"][0] for s in live)
    cluster_x1 = max(s["bbox"][2] for s in live)
    den_line_w = max(g["x1"] - g["x0"] for g in groups)
    line_frac = len(groups) >= 2 or den_line_w > 0.40 * max(cluster_x1 - cluster_x0, 1.0)

    if not line_frac:
        # POINT-FRAC: raised spans inside the window are the numerator;
        # baseline spans (even inside the window) continue as spine.
        g = groups[0]
        lo = g["x0"] - 0.6 * body_size
        hi = g["x1"] + 0.6 * body_size
        raised = [
            s
            for s in live
            if id(s) not in den_ids
            and lo <= _cx(s) <= hi
            and (s["bbox"][1] + s["bbox"][3]) / 2 < baseline_y - 0.20 * body_size
        ]
        if not raised:
            return ""
        num_txt = _stack_group_text(raised, baseline_y)
        den_txt = _stack_group_text(g["spans"], baseline_y)
        if not num_txt or not den_txt:
            return ""
        frac = f"\\frac{{{num_txt}}}{{{den_txt}}}"
        frac_x = (_cx(dens[0]) + _cx(dens[-1])) / 2
        spine_ids = {id(s) for s in raised} | den_ids
        tokens = [
            (_cx(s), _stack_token(s)) for s in live if id(s) not in spine_ids and _stack_token(s)
        ]
        tokens.append((frac_x, frac))
        tokens.sort(key=lambda t: t[0])
        return " ".join(t for _, t in tokens)

    # LINE-FRAC: per-group windows carve the top line into segments.
    taken: set[int] = set()
    fracs: list[tuple[float, str]] = []
    for g in groups:
        lo = g["x0"] - 0.8 * body_size
        hi = g["x1"] + 0.8 * body_size
        gnums = [
            s
            for i, s in enumerate(live)
            if i not in taken and id(s) not in den_ids and lo <= _cx(s) <= hi
        ]
        for s in gnums:
            taken.add(id(s))
        den_txt = _stack_group_text(g["spans"], baseline_y)
        num_txt = _stack_group_text(gnums, baseline_y)
        if not num_txt or not den_txt:
            continue
        fracs.append(((g["x0"] + g["x1"]) / 2, f"\\frac{{{num_txt}}}{{{den_txt}}}"))
    if not fracs:
        return ""
    spine_ids = den_ids | taken
    tokens = [(_cx(s), _stack_token(s)) for s in live if id(s) not in spine_ids and _stack_token(s)]
    tokens.extend(fracs)
    tokens.sort(key=lambda t: t[0])
    return " ".join(t for _, t in tokens)


def _math_continues(items: list, i: int, n: int) -> bool:
    """True when math content (math-font span or pure operand) follows item i
    within the next 2 spans — distinguishes 'iff' inside an equation from a
    prose connective between equations."""
    for j in (i + 1, i + 2):
        if j >= n:
            return False
        it = items[j]
        t = it["text"].strip()
        if not t:
            continue
        if _STRICT_MATH_FONT_RE.search(it["font"]) or it.get("unicode_math"):
            return True
        if _is_operand_span(t):
            return True
        return False
    return False


def _strict_math_ahead(items: list, i: int, n: int) -> bool:
    """True when a MATH-FONT span (not a bare operand) follows within 2 spans.

    Gates opening a math run from a leading fence: '(v, w)' (v is MathItalic
    → open) vs prose '(1)' (roman digit → stay text). Fences must land INSIDE
    $...$ or the scorer's delimiter regex drops them (measured:
    gold (v,x_1) vs extracted 'v , x_1' — render-compare missing <mo> fence,
    2503.07368_pg8)."""
    _ref = items[i].get("size", 10.0)
    for j in (i + 1, i + 2):
        if j >= n:
            return False
        it = items[j]
        if not it["text"].strip():
            continue
        if (
            it.get("size", 10.0) < 0.8 * _ref
            and not (
                _STRICT_MATH_FONT_RE.search(it["font"]) or it.get("unicode_math")
            )
        ):
            # SCRIPT-SIZE roman furniture (iter-98, 05068): 'max' is
            # followed by its LIMIT line '1≤i≤k' whose first glyph is a
            # roman '1' — a script-size digit is equation furniture,
            # never a prose signal. Body-size roman digits still veto.
            continue
        return bool(_STRICT_MATH_FONT_RE.search(it["font"]) or it.get("unicode_math"))
    return False

def _wordrun_continues(items: list, i: int, n: int) -> bool:
    """WORD-RUN gate (iter-93): item i starts a run of lowercase roman
    words sitting INSIDE an equation ('..., for any integer k ≥ 1').
    True when the run is <=4 words and MATH resumes immediately after —
    prose lines never have math right after a 2-4-word lowercase run."""
    _j = i
    _count = 0
    while _j < n and _count < 4:
        t = items[_j]["text"].strip()
        if not t:
            _j += 1
            continue
        if re.fullmatch(r"[a-z]{2,10}", t) and not _STRICT_MATH_FONT_RE.search(
            items[_j]["font"]
        ):
            _count += 1
            _j += 1
        else:
            break
    if _count == 0 or _j >= n:
        return False
    # skip trailing whitespace-only items
    while _j < n and not items[_j]["text"].strip():
        _j += 1
    if _j >= n:
        return False
    _nxt = items[_j]["text"].strip()
    return bool(
        _nxt
        and (
            _STRICT_MATH_FONT_RE.search(items[_j]["font"])
            or items[_j].get("unicode_math")
            or _is_operand_span(_nxt)
        )
    )


_LIGATURES_RE = re.compile("ﬁ|ﬂ|ﬀ|ﬃ|ﬄ")
_LIGATURE_MAP = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}


def _prev_run_ends_conditional(run: list) -> bool:
    """WORDRUN flank gate (iter-93): the open run's last meaningful token
    ends a CONDITIONAL clause — punctuation, a digit, or a closing fence.
    '…, \\quad\\text{for any integer}\\quad k ≥ 1' (after ',' / '1' / ')')
    is a math-side condition; 'Let k such that …' (after a bare letter
    variable) is prose."""
    for tok in reversed(run):
        t = str(tok).strip()
        if not t:
            continue
        return bool(re.search(r"[,;:)\]\}\d\\}]$", t))
    return False

def _ligature_expand(m: "re.Match[str]") -> str:
    return _LIGATURE_MAP[m.group(0)]


_RELATION_HEAD_RE = re.compile(
    r"^\s*(<=|>=|\\leq\b|\\geq\b|\\le\b|\\ge\b|\\ll\b|\\gg\b|<|>|=|≤|≥|≠|≪|≫)"
)


def _peel_leading_operand(
    result: list, items: list, i: int, body_size: float, first_tok: str
) -> list:
    """Pull tight-kerned left operands into a run opening at items[i].

    TeX kerns an inline equation's head to the preceding prose with no
    word space, and MuPDF keeps the pieces as separate roman spans:
    "…time 0" + CMMI " < t_j…" and "E" "[" "T" "]" + rtxr " =" (2503.07226
    math_002, 2503.03766 math_004). The run then opens one token late and
    the gold equations lose their head ('0 < t_j+…', 'E[T] = …').

    Two peels, both strictly reversible on any mismatch:
    1. DIGIT TAIL — the run's first token is a binary relation and the
       emitted prose ends in a lone number: move the number into the run
       (relations need a left operand; prose numbers before prose are
       never followed by a relation-led equation).
    2. SPAN CLUSTER — whole operand spans (brackets, single letters,
       digits) whose x-gap to the next item is under a word space and
       that sit on the body baseline, popped back out of the emitted
       text. Any word / space span / wide gap stops the cluster.
    """
    peeled: list[str] = []
    if (
        result
        and isinstance(result[-1], str)
        and _RELATION_HEAD_RE.match(first_tok or "")
        and (m := re.match(r"^(.*\S)(\s+)(\d+(?:\.\d+)?)$", result[-1]))
    ):
        result[-1] = m.group(1)
        peeled.append(m.group(3))
    cluster: list[str] = []
    cut = i
    j = i - 1
    while j >= 0:
        t = items[j]["text"]
        if not t.strip():
            break
        try:
            gap = items[j + 1]["bbox"][0] - items[j]["bbox"][2]
        except (KeyError, TypeError, IndexError):
            break
        if not (-2.0 <= gap <= 0.15 * body_size):
            break
        if abs(items[j].get("dy", 0.0)) > 0.3 * body_size:
            break
        if t.strip().endswith(".") or not _is_operand_span(t):
            break
        tok = _span_to_math_token(t.strip(), False, False, items[j].get("font", ""))
        cluster.insert(0, tok or t.strip())
        cut = j
        j -= 1
    if cluster:
        k = i - 1
        while k >= cut:
            if not result or result[-1] != items[k]["text"]:
                cluster = []  # emitted text diverged — abort, keep prose
                break
            result.pop()
            k -= 1
    return cluster + peeled


def _build_line_with_math(
    spans: list[dict], body_size: float, baseline_y: float, den_spans=None
) -> str:
    """Assemble one line into text + math runs. (v3 — script-run accumulation)

    Root causes fixed vs v2 (2026-08-14 KaTeX failure census):
    1. DOUBLE SUB/SUPERSCRIPTS: v2 wrapped EACH small span as its own
       `_{...}` → `x_{k}_{+}_{m}` (KaTeX rejects). v3 accumulates ALL
       consecutive small spans into ONE script group: `x_{k + m}`.
    2. NESTED SCRIPTS: a deeper small span (smaller size + larger |dy|)
       attaches to the previous element — `V ρ1` → `V_{\rho _{1}}`
       (MathML-equal to gold `V_{\rho_1}`), not `V_{\rho}_{1}`.
    3. PROSE LEAKS: gap absorption admitted 2-letter English words
       ("of","is") → "\betaof", "wherez_0is". Now operands only:
       digits/punct/single letters.
    4. SCRIPT-FONT FIDELITY: rsfs/Euler-script → \\mathcal, EUFM →
       \\mathfrak, MSBM/Blackboard → \\mathbb.
    5. DISPLAY LINES: a line ≥85% math-font width (and no prose words)
       emits as a standalone `$$...$$` equation — the shape olmOCR math
       gold uses.
    """
    spans = [s for s in spans if s.get("text", "").strip()]
    if not spans:
        return ""

    # Stacked-fraction cluster (stage-2.5 merge): dedicated assembly —
    # per-x \frac{num}{den} pairs. Falls through to the normal path when
    # assembly yields nothing.
    if den_spans:
        stacked = _build_stacked_frac(spans, den_spans, body_size, baseline_y)
        if stacked:
            return "$$" + stacked + "$$"

    # Display-line test (fix 5)
    math_w = sum(
        s["bbox"][2] - s["bbox"][0] for s in spans if _STRICT_MATH_FONT_RE.search(s.get("font", ""))
    )
    total_w = sum(s["bbox"][2] - s["bbox"][0] for s in spans)
    has_prose = any(
        re.search(r"[a-z]{4,}", s["text"]) and not _STRICT_MATH_FONT_RE.search(s.get("font", ""))
        for s in spans
    )
    # Prose set in a MATH font (CMMI italic phrases) also disqualifies a
    # display line — "Sof primes, the algebraic integer..." was emitted as
    # $$..$$ and KaTeX-rejected by the checker.
    if not has_prose:
        for s in spans:
            if any(p for p, is_prose in _split_prose_in_math(s["text"]) if is_prose):
                if len("".join(p for p, ip in _split_prose_in_math(s["text"]) if ip)) > 20:
                    has_prose = True
                    break
    is_display = total_w > 60 and math_w / total_w >= 0.85 and not has_prose

    result: list[str] = []
    run: list[str] = []

    def _flush_run():
        if run:
            # RADICAND GROUPING (iter-99, 2503.07226_pg7): the radical
            # sign is its own span ('\sqrt ' token — CMEX/txex glyph
            # decode or a '√' unicode span) and nothing braces the
            # radicand, so '\sqrt | \beta | r' puts only '|' under the
            # sign. Brace the operand after a bare \sqrt: a balanced
            # |…| / (…) / […] run, else the single next token. A
            # line-broken/unpaired radical stays bare (render-check
            # flags it) — inventing a group there is worse.
            _j = 0
            while _j < len(run):
                if not re.fullmatch(r"\\sqrt\s*", run[_j] or "") or _j + 1 >= len(run):
                    _j += 1
                    continue
                _nxt = run[_j + 1].strip()
                if _nxt[:1] in ("{", "["):
                    _j += 2
                    continue
                if _nxt[:1] in ("|", "("):
                    _op, _cl = (_nxt[:1], {"|": "|", "(": ")"}[_nxt[:1]])
                elif _nxt.startswith(r"\{"):
                    _op, _cl = r"\{", r"\}"
                else:
                    _op = _cl = None
                if _op is None:
                    run[_j : _j + 2] = [r"\sqrt{" + _nxt + "}"]
                    _j += 1
                    continue
                _end = None
                if _op == "|":
                    _bars = 0
                    for _k in range(_j + 1, len(run)):
                        _bars += run[_k].count("|")
                        if _bars >= 2:
                            _end = _k
                            break
                else:
                    _depth = 0
                    for _k in range(_j + 1, len(run)):
                        _depth += run[_k].count(_op) - run[_k].count(_cl)
                        if _depth <= 0:
                            _end = _k
                            break
                if _end is not None:
                    _grp = " ".join(t.strip() for t in run[_j + 1 : _end + 1])
                    run[_j : _end + 1] = [r"\sqrt{" + _grp + "}"]
                _j += 1
            inner = " ".join(run)
            inner = re.sub(r"\s+(_|\^\{)", r"\1", inner)
            # TeX typesets ⋯ as ONE glyph run; MuPDF yields three '·'
            # spans → three \cdot tokens. Gold collapses them to
            # \cdots (renders differ; 06597 gold census). Exactly-3
            # runs collapse; 6+ collapses pairwise (\cdots\cdots).
            inner = re.sub(
                r"(\\cdot(?:\s+\\cdot){2})(?!\s*\\cdot)", r"\\cdots", inner
            )
            # PRIME normalization (iter-38): a lone text-quote in a
            # script group renders as the quote glyph, not \prime —
            # gold writes \alpha' for a prime-marked sup span. Strip the
            # group shell: ^\{'+\} → '+ (KaTeX maps to \prime).
            inner = re.sub(r"\^\{('+)\}", r"\1", inner)
            if is_display:
                # Equation numbers "(2.8)" / "(12)" are typography, not
                # math — strip from display blocks (KaTeX-safe, matches
                # olmOCR gold which drops them).
                inner = re.sub(r"\s*\(\s*\d+(?:\.\d+)?\s*\)\s*$", "", inner)
            result.append(("$$" if is_display else "$") + inner + ("$$" if is_display else "$"))
            run.clear()

    items = [
        {
            # Ligature-normalize AT INTAKE: TeX ToUnicode emits 'iﬀ' (ﬀ
            # U+FB00) which defeats [A-Za-z]+ word matching in every branch
            # below (\text{iff} never fired, 2503.05360_pg2). Map early.
            "text": _LIGATURES_RE.sub(_ligature_expand, s["text"]),
            "font": s.get("font", ""),
            "size": s.get("size", body_size),
            "dy": s.get("origin", (0, baseline_y))[1] - baseline_y,
            "frac": s.get("frac"),
            "sqrt": s.get("sqrt"),
            "_opsub": s.get("_opsub_spans"),
            "_opsup": s.get("_opsup_spans"),
            "_glyphs": s.get("_glyphs"),
            # Ligature-origin flag (iter-99): txfonts sets text ligatures
            # (ﬀ ﬁ ﬂ …) in the MATH-classified roman face (rtxr, 2503.06817
            # 'diﬀerent'). Math fonts have NO ligatures — the tag routes
            # the expanded letters back to prose.
            "lig": bool(_LIGATURES_RE.search(s.get("text", ""))),
            "bbox": s.get("bbox", (0, 0, 0, 0)),
        }
        for s in spans
    ]

    i = 0
    n = len(items)
    # >0 while a \binom replaced its visual parens: consume that many
    # leading ')' from subsequent spans (binom renders its own delimiters).
    pending_close = 0
    # Precompute: span carries Unicode math symbols (∆ Ω ⟨ ⟩ ∥ ⩾ …) with no
    # prose words. Some PDFs encode these as plain-font Unicode, not CMMI
    # glyphs — font test alone misses them (root cause of ∆ landing outside
    # math runs on 2503.04917).
    for it in items:
        it["unicode_math"] = bool(
            it["text"].strip()
            and any(u in it["text"] for u in _FONT_MATH_CHARS)
            and not re.search(r"[A-Za-z]{2,}", it["text"])
        )
    # PAGE GATE (iter-110, 068682d5 pg1): a page whose EVERY span is a
    # prose font (TimesNewRoman, Georgia … — zero strict-math-font hits)
    # is prose: its lone '×'/'≥'-class glyphs are typographic prose
    # symbols ('an m × m multivariable design problem'), not math runs.
    # Keeping the flags opened $m \times m$ in the middle of a sentence
    # and broke reading-order probes. Pages that genuinely encode math
    # in plain fonts (the 2503.04917 ∆ case) always carry strict math
    # fonts elsewhere on the page (measured: CMMI12/CMSY10/MSAM10), so
    # the gate leaves them untouched.
    if not any(_STRICT_MATH_FONT_RE.search(it["font"]) for it in items):
        for it in items:
            it["unicode_math"] = False

    # WORD-RUN dominance gate (iter-93): 'for any integer' inside an
    # equation (math-dominant line) vs 'of the map' between math islands
    # in a prose sentence (word-dominant line). Only math-dominant lines
    # wrap roman word-runs into \quad\text{...}\quad; prose keeps its
    # words outside the $...$ runs.
    _math_w = 0.0
    _word_w = 0.0
    for it in items:
        _t = it["text"].strip()
        _w = it["bbox"][2] - it["bbox"][0]
        if _t and _w > 0:
            if (
                _STRICT_MATH_FONT_RE.search(it["font"])
                or it["unicode_math"]
                or (not re.fullmatch(r"[A-Za-z]{2,}", _t) and _is_operand_span(_t))
            ):
                _math_w += _w
            elif re.fullmatch(r"[a-z]{2,10}", _t):
                _word_w += _w
    _math_dominant = _math_w > _word_w
    while i < n:
        it = items[i]
        # GAP-AWARE SEPARATION: PDF spans often omit trailing spaces; joining
        # them raw glues words ('tripartitegraph'). When the x-gap to the
        # previous item exceeds a word-space, emit a separator (space in
        # math runs is collapsed by KaTeX; the pre-script space is stripped
        # by the \s+(_|^\{) rule below, so subscripts stay attached).
        if i > 0:
            _prev = items[i - 1]
            try:
                _gap = it["bbox"][0] - _prev["bbox"][2]
                if _gap > 0.15 * body_size:
                    if run:
                        run.append(" ")
                    else:
                        result.append(" ")
            except (KeyError, TypeError) as e:
                log.debug("gap-aware space separator skipped item (no bbox): %s", e)
        # Consume the visual ')' that a \binom replaced (renders its own).
        if pending_close and it["text"].lstrip().startswith(")"):
            rest = it["text"].lstrip()[1:].lstrip()
            pending_close -= 1
            if not rest:
                i += 1
                continue
            it = {**it, "text": rest}
        if it.get("sqrt"):
            # iter-133: radical + overline-rule-paired radicand —
            # prebuilt group (already LaTeX from _stack_group_text).
            run.append("\\sqrt{" + it["sqrt"] + "}")
            i += 1
            continue
        if it.get("frac"):
            top, bottom = it["frac"]

            def _frac_side(s: str, fnt: str) -> str:
                # Composed sides (group numerators assembled by
                # _stack_group_text, iter-34) are already LaTeX —
                # re-tokenizing would brace-escape their structure.
                if re.search(r"[\\{}]", s):
                    return s
                return _span_to_math_token(s, False, False, fnt) or s

            run.append(
                "\\frac{" + _frac_side(top, it["font"]) + "}{" + _frac_side(bottom, it["font"]) + "}"
            )
            i += 1
            continue
        # Overstrike accent span (iter-25): TeX prints \hat{\tau} as the
        # base glyph followed by an accent glyph x-overlapping it. The
        # bare accent char ('ˆ') is usually a roman-font glyph — without
        # this it flushes the run and splits the math region
        # ('τ$ˆ$2 i', 2503.05506_pg11). Wrap the previous token instead.
        # Script-size accents are handled inside the accumulation below.
        # Accent spans may carry leading punctuation ('; ˆ', '(ˆ', ' ˆ' —
        # MuPDF merges the kerning before the accent glyph into the span).
        _acc_m = re.fullmatch(
            r"([^ˆ˜¨¯˙˘ˇ]*?)([ˆ˜¨¯˙˘ˇ])", it["text"].strip()
        )
        _acc_cmd = _ENV_ACCENT_CMDS.get(_acc_m.group(2)) if _acc_m else None
        _acc_pre = _acc_m.group(1) if _acc_m else ""
        if _acc_cmd and it["size"] >= body_size * 0.8:
            if _acc_pre:
                # merged '; ' / '(' before the accent glyph: keep it in
                # the run — it is content the accent span swallowed.
                run.append(_acc_pre)
            if run and re.match(r"\\?[A-Za-z0-9]", run[-1].strip()):
                # Overstrike accent AFTER its base (iter-25): wrap the
                # previous token — \hat{\tau} from 'τ ˆ' spans.
                run[-1] = _acc_cmd + "{" + run[-1].strip() + "}"
                i += 1
                continue
            # Accent BEFORE its base (iter-29): some producers order the
            # accent glyph first — 'ˆ U(·,0)' (05068_pg47). Wrap the
            # NEXT body item when it is a single math-ish token.
            _nxt = items[i + 1] if i + 1 < n else None
            _nxt_tok = (
                _span_to_math_token(_nxt["text"], False, False, _nxt["font"])
                if _nxt is not None
                else ""
            )
            if _nxt_tok and re.fullmatch(
                r"\\[A-Za-z]+|[A-Za-z0-9]", _nxt_tok.strip()
            ):
                run.append(_acc_cmd + "{" + _nxt_tok.strip() + "}")
                i += 2
                continue
            # accent with no wrappable neighbor: drop into prose as-is
        small = it["size"] < body_size * 0.8
        if small:
            # ── Script-run accumulation (fixes 1+2) ──────────────────
            # Guard: only when math context exists — run already open,
            # span itself in a math font, or an adjacent math-font span.
            # (Pure-prose small spans — footnote markers — stay text.)
            near_math = (
                bool(run)
                or _STRICT_MATH_FONT_RE.search(it["font"])
                or it.get("unicode_math")
                or (
                    (i > 0 and _STRICT_MATH_FONT_RE.search(items[i - 1]["font"]))
                    or (i + 1 < n and _STRICT_MATH_FONT_RE.search(items[i + 1]["font"]))
                )
            )
            if near_math:
                parts_dirs: list[tuple[str, float, tuple]] = []
                prev = None
                _last_nested = False
                # SCRIPT-RUN tracking (iter-97): index/size/dy/x1 of the
                # last NESTED script glyph — see the continuation check
                # below (binom argument runs x-interleave the other band).
                _run_tgt = None
                _run_sz = _run_dy = 0.0
                _run_x1 = 0.0
                while i < n and items[i]["size"] < body_size * 0.8:
                    cur = items[i]
                    _gl = cur.get("_glyphs")
                    tok = (
                        _glyph_composed_token(cur["font"], _gl, cur["size"])
                        if _gl
                        else ""
                    ) or _span_to_math_token(cur["text"], False, False, cur["font"])
                    # Roman-font word in a SCRIPT position renders upright
                    # (gold \rm LoS): italic p_{LoS} fails render-compare.
                    if (
                        tok
                        and re.fullmatch(r"[A-Za-z]{2,}", cur["text"].strip())
                        and not _STRICT_MATH_FONT_RE.search(cur["font"])
                    ):
                        tok = r"\text{" + cur["text"].strip() + "}"
                    # Overstrike accent in script position (iter-25):
                    # v_{\hat{\tau}...} — the accent span follows its
                    # script base at the same dy; wrap the last token.
                    if (
                        len(cur["text"].strip()) == 1
                        and cur["text"].strip() in _ENV_ACCENT_CMDS
                        and parts_dirs
                        and re.match(r"\\?[A-Za-z0-9]", parts_dirs[-1][0].strip())
                    ):
                        _cmd = _ENV_ACCENT_CMDS[cur["text"].strip()]
                        parts_dirs[-1] = (
                            _cmd + "{" + parts_dirs[-1][0].strip() + "}",
                            parts_dirs[-1][1],
                            parts_dirs[-1][2],
                        )
                        prev = cur

                        i += 1
                        continue
                    # SCRIPT-RUN CONTINUATION (iter-97, 09577 binom args):
                    # a stacked construct x-interleaves its bands —
                    # \binom{k^{n-i}}{2} sorts as k, n, 2, −, i because
                    # the bottom arg sits under the top arg's script run.
                    # The size/dy nested-attach below then glues the top
                    # band's '−' onto the bottom '2' ('2^{-}') and strands
                    # 'i'. A span at the SAME size and dy as the last
                    # nested script, x-adjacent to it, continues that
                    # script's text — splice inside the trailing group.
                    if (
                        _run_tgt is not None
                        and abs(cur["size"] - _run_sz) <= 0.4
                        and abs(cur["dy"] - _run_dy) <= 0.4
                        and -0.75 <= cur["bbox"][0] - _run_x1 <= 1.5
                        and parts_dirs[_run_tgt][0].endswith("}")
                    ):
                        _old = parts_dirs[_run_tgt][0]
                        parts_dirs[_run_tgt] = (
                            _old[:-1] + " " + tok + "}",
                            parts_dirs[_run_tgt][1],
                            parts_dirs[_run_tgt][2],
                        )
                        _run_x1 = cur["bbox"][2]
                        prev = cur
                        i += 1
                        continue
                    # Nested attach (deeper script), extended (iter-26):
                    # a glyph x-overlapping the PREVIOUS script glyph on
                    # the opposite side attaches ONLY when the previous
                    # MEASURED CROSS-SPAN SUB (iter-68, 08800 display case):
                    # a short alnum script span sitting a FLATTENED
                    # sub-of-script drop (0.11–0.28×size) below an
                    # x-adjacent part is that part's inner subscript —
                    # char-level typography, not a size heuristic. A
                    # FIRST-ORDER sub (drop 0.33–0.5×size: τ's i at
                    # 2.65pt = 0.44×size, 05506) belongs to the x-overlap
                    # clique machinery that already renders τ_i^2 —
                    # measured drop windows exclude it. Siblings share a
                    # baseline (drop 0: K_{k∆t} factors stay parts).
                    _xsub = (
                        len(cur["text"].strip()) <= 3
                        and cur["text"].strip().isalnum()
                        and any(
                            0.11 * cur["size"] < cur["dy"] - p_[1] <= 0.28 * cur["size"]
                            and cur["bbox"][0] - p_[2][2] < 1.5
                            and cur["bbox"][0] >= p_[2][0]
                            for p_ in parts_dirs
                        )
                    )
                    if (
                        prev is not None
                        and parts_dirs
                        and (
                            (
                                cur["size"] < prev["size"] - 0.4
                                and abs(cur["dy"] - prev["dy"]) > 0.8
                            )
                            or (
                                _last_nested
                                and cur["size"] <= prev["size"] + 0.4
                                and abs(cur["dy"] - prev["dy"]) > 0.8
                                and _bbox_overlap(cur["bbox"], prev["bbox"])
                            )
                            or _xsub
                        )
                    ):
                        # Nested deeper script (iter-27): attach to the
                        # x-adjacent part with the SMALLEST |Δdy| — TeX
                        # sets a sub-sub glyph beside its own base on the
                        # SAME side (b^{P_{∆n}}: '∆n' is 1.4pt from P but
                        # 8pt from the sibling 'i,∆' — blindly appending
                        # to parts_dirs[-1] nested it into the wrong
                        # base). Direction relative to that part.
                        _tgt = min(
                            (
                                k
                                for k, p_ in enumerate(parts_dirs)
                                if p_[2][0] < cur["bbox"][2] + 0.5
                                and cur["bbox"][0] < p_[2][2] + 0.5
                            ),
                            key=lambda k: abs(parts_dirs[k][1] - cur["dy"]),
                            default=None,
                        )
                        if _tgt is None:
                            _tgt = len(parts_dirs) - 1
                        nd = "^" if cur["dy"] < parts_dirs[_tgt][1] else "_"
                        _old = parts_dirs[_tgt][0]
                        if _xsub and _old.endswith("}"):
                            # SPLICE (iter-68): the measured sub belongs to
                            # the part's inner base — append INSIDE the
                            # trailing group ('P_{\Delta}' + n →
                            # 'P_{\Delta_{n}}'), never a double subscript.
                            _new = _old[:-1] + f"{nd}{{{tok}}}" + "}"
                        else:
                            _new = _old + f"{nd}{{{tok}}}"
                        parts_dirs[_tgt] = (
                            _new,
                            parts_dirs[_tgt][1],
                            parts_dirs[_tgt][2],
                        )
                        _last_nested = True
                        _run_tgt, _run_sz = _tgt, cur["size"]
                        _run_dy, _run_x1 = cur["dy"], cur["bbox"][2]
                    else:
                        parts_dirs.append((tok, cur["dy"], cur["bbox"]))
                        _last_nested = False
                    prev = cur
                    i += 1
                if parts_dirs:
                    # Direction per part. DEFAULT: absolute dy vs the line
                    # baseline. OVERRIDE (iter-25): x-OVERLAPPING script
                    # parts take direction RELATIVE to each other. TeX sets
                    # one base's sub and sup into the same x window
                    # (τ_i^2: 'i' then '2' shifted +1.2pt), and MuPDF
                    # contaminates script-span origins when a leading space
                    # at the BODY baseline merges into the span — ' 2'
                    # reports dy=+0.0 for a glyph raised ~4pt (05506).
                    # Inside an x-overlap clique the highest part (min dy)
                    # is the superscript; the rest are subscripts.
                    dirs = ["^" if p[1] < -0.5 else "_" for p in parts_dirs]
                    _parent = list(range(len(parts_dirs)))

                    def _find(x: int) -> int:
                        while _parent[x] != x:
                            _parent[x] = _parent[_parent[x]]
                            x = _parent[x]
                        return x

                    for _a in range(len(parts_dirs)):
                        for _b in range(_a + 1, len(parts_dirs)):
                            _ba, _bb = parts_dirs[_a][2], parts_dirs[_b][2]
                            if _ba[0] < _bb[2] - 0.5 and _bb[0] < _ba[2] - 0.5:
                                _ra, _rb = _find(_a), _find(_b)
                                if _ra != _rb:
                                    _parent[max(_ra, _rb)] = min(_ra, _rb)
                    _comp: dict[int, list[int]] = {}
                    for _k in range(len(parts_dirs)):
                        _comp.setdefault(_find(_k), []).append(_k)
                    for _members in _comp.values():
                        if len(_members) < 2:
                            continue
                        # Two-band split at the clique's dy midpoint (a
                        # plain min() demotes tied members: '( k )' all
                        # share min dy over sub '∞', 2503.05698_pg14 —
                        # k landed in the subscript group). Upper band →
                        # '^', lower band → '_'. Flat clique (all dy
                        # within 0.5pt) keeps the absolute rule.
                        _ys = [parts_dirs[_k][1] for _k in _members]
                        if max(_ys) - min(_ys) < 0.5:
                            continue
                        _mid = (max(_ys) + min(_ys)) / 2.0
                        for _k in _members:
                            dirs[_k] = "^" if parts_dirs[_k][1] < _mid else "_"
                    # per-span direction: group same-direction spans; a
                    # direction switch closes the group.
                    # "H 0(sub) 1(sup)" → H_{0}^{1}, not H_{0 1}.
                    groups: list[tuple[str, list[str]]] = []
                    for _k, _part in enumerate(parts_dirs):
                        d = dirs[_k]
                        if groups and groups[-1][0] == d:
                            groups[-1][1].append(_part[0])
                        else:
                            groups.append((d, [_part[0]]))
                    if len(groups) >= 3:
                        # 3+ alternating groups on ONE base = a STACK, not
                        # scripts (KaTeX rejects `^{[n}_{r}^{]}` double-sup).
                        # (a) base is an opening delimiter + bracket-wrapped
                        #     top → binomial: (^{[ n}_{r}^{]} ) → \binom{[ n ]}{r}
                        # (b) otherwise merge same-direction groups:
                        #     λ^{(}_{i}^{N −k )} → λ^{( N −k )}_{i}
                        base_tok = run[-1] if run else ""
                        tops = [t for d, toks in groups if d == "^" for t in toks]
                        bots = [t for d, toks in groups if d == "_" for t in toks]
                        top_j = " ".join(tops)
                        bot_j = " ".join(bots)
                        if base_tok in ("(", "[", "") and (
                            top_j.lstrip().startswith(("[", "(")) or base_tok == "("
                        ):
                            # \binom renders its own delimiters — drop the
                            # visual '(' and consume the matching ')' later.
                            if run and run[-1] in ("(", "["):
                                run.pop()
                            run.append(f"\\binom{{{top_j}}}{{{bot_j}}}")
                            pending_close += 1
                        else:
                            run.append(f"^{{{top_j}}}")
                            if bot_j:
                                run.append(f"_{{{bot_j}}}")
                    elif (
                        # 2-BAND STACK over a bare open delimiter (iter-97,
                        # 09577): \binom{k}{2} and \binom{k^{n-i}}{2} put
                        # the top band at script dy above the bottom band,
                        # both x-centered on the CMEX '(' — exactly one
                        # '^' group then one '_' group with the delimiter
                        # as the run tail. Scripts never attach to a bare
                        # '(' as an operand; gold renders \binom.
                        len(groups) == 2
                        and run
                        and run[-1] in ("(", "[")
                        and groups[0][0] == "^"
                        and groups[1][0] == "_"
                    ):
                        _top_b = " ".join(groups[0][1])
                        _bot_b = " ".join(groups[1][1])
                        run.pop()
                        run.append(f"\\binom{{{_top_b}}}{{{_bot_b}}}")
                        pending_close += 1
                    else:
                        for d, toks in groups:
                            run.append(d + "{" + " ".join(toks) + "}")
                continue
            # small prose span, no math context → plain text
            _flush_run()
            result.append(it["text"])
            i += 1
            continue
        if _STRICT_MATH_FONT_RE.search(it["font"]) or it.get("unicode_math"):
            tok = _span_to_math_token(it["text"], False, False, it["font"])
            # LIGATURE fragment in a math-classified roman face (iter-99,
            # 2503.06817 rtxr 'ﬀ' in 'diﬀerent'): TeX ligatures exist only
            # in TEXT mode — the span is body prose regardless of font.
            if it.get("lig") and re.fullmatch(r"[a-z]{1,8}", it["text"].strip()):
                _flush_run()
                result.append(it["text"])
                i += 1
                continue
            # BIOMEDICAL GREEK NAME (iter-89b, 0225d691): 'IFNγ' /
            # 'TNFα' — MinionMathItalic sets the lone Greek glyph, but
            # the flanking spans are PROSE (≥2-capital cytokine prefix
            # at body size, lowercase word after). Emit the glyph as
            # plain text; fencing it ('$\\gamma$') splits the name and
            # breaks gold order windows ('IFN$\\gamma  $expression').
            if (
                re.fullmatch(r"[α-ωΑ-Ω]", it["text"].strip())
                and not run
                and result
                and isinstance(result[-1], str)
                and re.search(r"[A-Z]{2,}\s*$", result[-1])
                and i + 1 < n
                and not _STRICT_MATH_FONT_RE.search(items[i + 1]["font"])
                and re.match(r"\s*[a-z]", items[i + 1]["text"])
            ):
                _flush_run()
                result.append(it["text"])
                i += 1
                continue
            if not tok:
                i += 1
                continue
            # \binom consumed its visual parens: a converted CMEX ')' (raw
            # text is '\x01', so the loop-head raw check can't see it) or a
            # plain ')' leading token must be dropped here.
            if pending_close and tok.startswith(")"):
                tok = tok[1:].lstrip()
                pending_close -= 1
                if not tok:
                    i += 1
                    continue
            # Prose-word guard: heavy-math PDFs set entire phrases in CMMI
            # italic ("βandγ", "ΓofIsom("). Letter runs ≥3 that are not
            # math function names are prose — split them OUT of the run
            # (root cause of "\betaand", "\GammaofIsom" gluing).
            if any(p for p, is_prose in _split_prose_in_math(it["text"]) if is_prose):
                _flush_run()
                for seg, is_prose in _split_prose_in_math(it["text"]):
                    if not seg.strip():
                        continue
                    if is_prose:
                        if seg in _TEXT_IN_MATH_WORDS and _math_continues(items, i, n):
                            # roman-style connective set in a MATH font
                            # ('$... iff M$' with iff in CMMI, 2503.05360):
                            # gold wraps it \text{...}, not bare italic.
                            run.append(r"\text{" + seg + "}")
                            _flush_run()
                        else:
                            result.append(seg)
                    else:
                        stok = _span_to_math_token(seg, False, False, it["font"])
                        if stok:
                            run.append(stok)
                            _flush_run()
            else:
                if not run:
                    run[:0] = _peel_leading_operand(
                        result, items, i, body_size, tok
                    )
                if it.get("_opsub"):
                    sub = _stack_group_text(it["_opsub"], baseline_y)
                    if sub:
                        tok = tok.rstrip() + "_{" + sub + "}"
                if it.get("_opsup"):
                    sup = _stack_group_text(it["_opsup"], baseline_y)
                    if sup:
                        tok = tok + "^{" + sup + "}"
                run.append(tok)
            i += 1
        elif (
            (run or _strict_math_ahead(items, i, n))
            and not _STRICT_MATH_FONT_RE.search(it["font"])
            and (_m := re.match(r"^\s*([(\[]?)([A-Za-z]+)(\W.*)?$", it["text"]))
            and (
                _m.group(2) in _TEXT_IN_MATH_WORDS
                or _m.group(2) in _FUNC_CMD_WORDS
                or (
                    # WORD-RUN continuation (iter-93, 2503.06228_pg1):
                    # a short run of common lowercase English words sits
                    # INSIDE a display equation ('f_k := …, for any
                    # integer k \geq 1'). Gold wraps the phrase
                    # '\quad \text{for any integer} \quad'. Gates: open
                    # run, MATH-DOMINANT line (math width > word width —
                    # 'of the map' in a prose sentence stays outside
                    # the $...$, 2503.06597_pg18), <=4-word lowercase
                    # run, math resuming right after.
                    run
                    and _math_dominant
                    and _m.group(2).islower()
                    and 2 <= len(_m.group(2)) <= 10
                    and not _m.group(1)
                    and not _m.group(3)
                    and _wordrun_continues(items, i, n)
                )
                or (
                    # ROMAN OPERATOR entry (iter-95): 'cr(' — upright
                    # letters + '(' + strict-math operand directly after
                    # (LMRoman/CMR span). Detailed gates in the body.
                    _m.group(2).islower()
                    and 2 <= len(_m.group(2)) <= 5
                    and (_m.group(3) or "").lstrip().startswith("(")
                    and _strict_math_ahead(items, i, n)
                )
            )
            and (
                _m.group(2) in _FUNC_CMD_WORDS
                or _math_continues(items, i, n)
                or (
                    _m.group(2).islower()
                    and not _m.group(3)
                    and _wordrun_continues(items, i, n)
                )
            )
        ):
            # WORDRUN (iter-93): takes precedence over the single-word
            # \text path when >=2 words continue. Gates mirror the
            # condition plus an OPEN run, math dominance, and the run's
            # last token ending a CONDITIONAL clause (punctuation, digit,
            # or closing fence) — 'for any integer' follows ',' / '1' /
            # ')', while 'such that' after a bare VARIABLE 'k' is prose
            # (2503.06597_pg18).
            _is_wordrun = (
                run
                and _math_dominant
                and not (_m.group(3) or "").strip()
                and not _m.group(1)
                and _wordrun_continues(items, i, n)
                and _prev_run_ends_conditional(run)
                and any(
                    True
                    for _k2 in range(i + 1, min(n, i + 6))
                    if re.fullmatch(r"[a-z]{2,10}", items[_k2]["text"].strip())
                    and not _STRICT_MATH_FONT_RE.search(items[_k2]["font"])
                )
            )
            _word = _m.group(2)
            _fence = _m.group(1) or ""
            if _fence:
                run.append(_fence)
            if _word in _FUNC_CMD_WORDS:
                tok_cmd = "\\" + _word
                if it.get("_opsub"):
                    sub = _stack_group_text(it["_opsub"], baseline_y)
                    if sub:
                        tok_cmd = tok_cmd + "_{" + sub + "}"
                if it.get("_opsup"):
                    sup = _stack_group_text(it["_opsup"], baseline_y)
                    if sup:
                        tok_cmd = tok_cmd + "^{" + sup + "}"
                run.append(tok_cmd)
            elif (
                # ROMAN OPERATOR (iter-95, 2503.08365_pg2): 'cr(' arrives
                # as ONE upright (LMRoman/CMR) span glued to a strict-math
                # operand — LaTeX \mathrm{cr}(G). Without this the run
                # opens only at G and 'cr(' stays prose, breaking every
                # gold \mathrm{cr}(G)\le chain. Gates: 2-5 lowercase
                # letters, rest starts with '(', strict math directly
                # inside the parens. The '(' itself is appended by the
                # shared _rest tail below (operand span).
                _word.islower()
                and 2 <= len(_word) <= 5
                and (_m.group(3) or "").lstrip().startswith("(")
                and _strict_math_ahead(items, i, n)
            ):
                run.append(r"\mathrm{" + _word + "}")
            elif _is_wordrun:
                # word-run continuation (takes precedence over the single-
                # word \text path: 'for' IS whitelisted but gold wraps the
                # whole phrase '\text{for any integer}').
                _words = [_word]
                _j = i + 1
                while _j < n:
                    _t = items[_j]["text"].strip()
                    if not _t:
                        _j += 1
                        continue
                    if re.fullmatch(r"[a-z]{2,10}", _t) and not _STRICT_MATH_FONT_RE.search(
                        items[_j]["font"]
                    ):
                        _words.append(_t)
                        _j += 1
                    else:
                        break
                run.append(r"\quad \text{" + " ".join(_words) + r"} \quad")
                i = _j
                continue
            elif _word in _TEXT_IN_MATH_WORDS:
                run.append(r"\text{" + _word + "}")
            else:
                run.append(r"\text{" + _word + "}")
            _rest = (_m.group(3) or "").strip()
            if _rest and _is_operand_span(_rest):
                run.append(_rest)
            elif _rest:
                _flush_run()
                result.append(_rest)
            i += 1
        elif (
            (
                _mix := re.match(
                    r"^([\s()\[\]\d.,+\-*/=<>*·×^_|]{1,14}?)\s+([A-Za-z][a-z]+.*)$",
                    it["text"].strip(),
                )
            )
            and _is_operand_span(_mix.group(1))
            and (run or _strict_math_ahead(items, i, n))
        ):
            # MIXED span (iter-37): MuPDF glues an operand tail and the
            # following prose word into ONE span ('1) and', 05218 — the
            # whole-span operand test fails and the run flushes
            # mid-expression: 'n ( n -$1) and'). Absorb the operand
            # prefix, emit the prose remainder as text.
            run.append(_mix.group(1))
            _flush_run()
            result.append(_mix.group(2))
            i += 1
        elif (
            _is_operand_span(it["text"].strip())
            and not (
                # '(\d+)' is an inline equation NUMBER only when math does
                # NOT continue past it (iter-6b, flip-diff 2503.07897_pg25:
                # 'µ_r(0) = 0' — '(0)' followed by '=' is CONTENT; refusing
                # it split the run and broke the gold match).
                re.fullmatch(r"\((\d+)\)", it["text"].strip()) and not _math_continues(items, i, n)
            )
            and (run or _strict_math_ahead(items, i, n))
        ):
            _op = it["text"].strip()
            # Sentence-final period belongs to prose, not math — absorbing
            # it changes the render ('(x)$ .' vs '(x) .'). Two shapes:
            # line-final '.', or '.' followed by a Capital-letter prose span.
            _nxt = items[i + 1] if i + 1 < n else None
            _prose_after = bool(
                _nxt
                and not _is_operand_span(_nxt["text"].strip())
                and re.match(r"[A-Z]", _nxt["text"].lstrip())
            )
            if (
                not is_display
                and _op.endswith(".")
                and len(_op) <= 3
                and (i == n - 1 or _prose_after)
            ):
                if len(_op) > 1:
                    # Fence+period span (').' / '].'): keep the FENCE in the
                    # run, push only the sentence period out — pushing the
                    # whole span uncloses the math ('$G ( m, n$).' —
                    # layer-interaction bug, pinned iter5-fence-close-period).
                    run.append(_op[:-1])
                    _flush_run()
                    result.append(_op[-1])
                else:
                    _flush_run()
                    result.append(_op)
                i += 1
                continue
            if not run:
                run[:0] = _peel_leading_operand(result, items, i, body_size, _op)
            run.append(_op.replace("#", "\\#"))
            i += 1
        else:
            _flush_run()
            result.append(it["text"])
            i += 1

    _flush_run()
    text = "".join(result)
    # merge trivially-adjacent fragments: "$a$$b$" "$a$,$b$"
    if not is_display:
        text = re.sub(r"\$\$", " ", text)
        text = re.sub(r"\$([,.;:])\$", r"\1", text)
    # combining solidus negation (fix from census bucket 4). NOTE:
    # re.sub replacement templates interpret \n as a REAL newline —
    # every negation emitted newline-corrupted text ('\n' + 'eq')
    # until the replacement was backslash-escaped (measured 06570
    # \text{Eff}_{<newline>eq0}, 2026-08-17 iter-23).
    for base, neg in _NEGATED_MAP.items():
        text = re.sub(r"̸\s*" + re.escape(base), neg.replace("\\", "\\\\"), text)
        text = re.sub(re.escape(base) + r"\s*̸", neg.replace("\\", "\\\\"), text)
    # Ellipsis: spaced-dot runs are the PDF's \ldots glyphs seen as separate
    # spans (gold uses \ldots; '. . .' renders as three periods with wide
    # spacing in MathML — fails render-compare).
    text = re.sub(r"(?:\.\s*){3,}", r"\\ldots ", text)
    text = text.replace("…", "\\ldots ")
    # Typographic minus U+2212 → ASCII '-' (gold LaTeX uses ASCII; MathML
    # <mo>−</mo> vs <mo>-</mo> differ under normalization).
    text = text.replace("−", "-")
    # SCRIPT TEXT-GROUP JOIN (iter-106, 04690): a subscript word-run set
    # as separate spans ('fixed' 'subtraction' in CMR7) emits as
    # k_{\text{fixed} \text{subtraction}} — TWO boxes in MathML; gold is
    # one \text{fixed\_subtraction}. Join adjacent \text groups INSIDE
    # one script group with \_ (both forms render-match gold).
    def _join_script_texts(m):
        inner = m.group(2)
        if inner.count("\\text{") < 2:
            return m.group(0)
        joined = inner
        while True:
            new = re.sub(
                r"\\text\{([^{}]*)\}\s*\\text\{([^{}]*)\}",
                "\\\\text{\\1\\\\_\\2}",
                joined,
            )
            if new == joined:
                break
            joined = new
        return m.group(0).replace(inner, joined)


    text = re.sub(r"([_^])\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", _join_script_texts, text)
    return text


def _glyph_composed_token(font: str, glyphs: list, size: float) -> str:
    """Token with intra-span scripts composed from per-glyph origins.

    TeX flattens script-of-script to one size, so '∆n' for P_{\\Delta_n}
    arrives as ONE span whose glyphs sit at the same font/size — only the
    char origins separate them (n drops 0.96pt = 0.16×size below ∆,
    measured 08800_pg9, every instance). Glyphs within 0.11×size of the
    first glyph's origin compose the base; a glyph 0.11–0.85×size BELOW
    the running base line starts '_{...}'; ABOVE starts '^{...}'. Returns
    '' when every glyph shares the first origin (caller falls back to the
    plain span token — sibling letters must stay factors, K_{k∆t}).
    """
    base: list[str] = []
    script: tuple[str, list[str]] | None = None
    out = ""
    y0 = glyphs[0][1][1]
    for ch, org, _bb in glyphs:
        dy = org[1] - y0
        if abs(dy) <= 0.11 * size:
            if script is not None:
                out += script[0] + "{" + "".join(script[1]) + "}"
                script = None
            base.append(ch)
        else:
            kind = "_" if dy > 0 else "^"
            if script is None or script[0] != kind:
                if script is not None:
                    out += script[0] + "{" + "".join(script[1]) + "}"
                script = (kind, [])
            script[1].append(ch)
    if script is not None:
        out += script[0] + "{" + "".join(script[1]) + "}"
    if not base:
        return ""
    tok = _span_to_math_token("".join(base), False, False, font)
    if not tok:
        return ""
    return tok + out


# OMS-font ToUnicode slot leaks (iter-114): slot char + adjacent-slot
# glyph pairs that mean ONE symbol. Keys are the raw leaked pairs; values
# the gold LaTeX. mapsto slot 0x37 measured 2503.06967 pg8 ('7→').
_OMS_SLOT_LEAK = {"7→": r"\mapsto"}
_OMS_SLOT_LEAK_RE = re.compile(
    "|".join(sorted((re.escape(k) for k in _OMS_SLOT_LEAK), key=len, reverse=True))
)

def _span_to_math_token(text: str, is_sup: bool, is_sub: bool, font: str = "") -> str:
    """Convert a math span's raw text to a LaTeX token.

    Unicode → LaTeX command conversion, then font-family styling:
    rsfs/Euler-script → \\mathcal, EUFM → \\mathfrak, MSBM/Blackboard →
    \\mathbb for bare A-Z (styled glyphs map to plain letters in
    PyMuPDF's Unicode mapping).
    """
    out = text
    if re.search(r"(?i)cmex|euex|txex", font or ""):
        # CMEX10 big delimiters: control chars encode L/R by parity
        # (0x00="(",0x01=")", 0x10/0x12 L, 0x11/0x13 R — measured on bench
        # PDFs); 0xF8xx PUA pieces are extension glyphs → drop.
        # cmex10 canonical glyph layout: big operators live at ASCII LETTER
        # slots (P=∑ Q=∏ R=∫ S=∮ …). PDFs with broken/naive ToUnicode emit
        # the LETTER instead of the Unicode operator (measured: 'P' in
        # CMEX10 span = \sum, 2503.04448_pg10). Table = plain.tex cmex10.
        # cmex10 canonical glyph layout (plain.tex): big operators at ASCII
        # letter slots 0x50-0x58 = P..X. Broken/naive ToUnicode emits the
        # LETTER. P=∑ verified 2503.04448_pg10; Q/R/S canonical layout.
        # T..X verified 2026-08-16 by glyph-pairing: 2503.08172_pg25 gold
        # contains \bigsqcup_{l\neq j} AND the page carries CMEX10 'X'
        # spans (render-truth pairing, H7 evidence bar met).
        _CMEX_LETTER_SLOTS = {
            "P": r"\sum ",
            "Q": r"\prod ",
            "R": r"\int ",
            "S": r"\oint ",
            "T": r"\bigodot ",
            "U": r"\bigoplus ",
            "V": r"\bigotimes ",
            "W": r"\biguplus ",
            "X": r"\bigsqcup ",
            # canonical cmex10 radical slot (0x70, plain.tex): the
            # radicalbig sign — static fallback when the font program
            # is unparseable (2503.07226_pg7 'p' = √|β|).
            "p": r"\sqrt ",
        }

        glyph_map = _ACTIVE_CMEX_GLYPHS.get(font or "")

        def _cmex(ch):
            o = ord(ch)
            if glyph_map and ch in glyph_map:
                # Embedded-font ground truth (iter-14): the document's own
                # code→glyph-name encoding. Supersedes the static table.
                return glyph_map[ch]
            if o < 0x20:
                return "(" if o % 2 == 0 else ")"
            if 0xE000 <= o <= 0xF8FF:
                return ""
            return _CMEX_LETTER_SLOTS.get(ch, ch)

        out = "".join(_cmex(c) for c in out)
    # BROKEN TOUNICODE OMS SLOT LEAKS (iter-114, 2503.06967 pg8): OMS
    # symbol fonts (cmsy/LMMathSymbols…) assign arrows to ASCII slots;
    # naive ToUnicode emits the SLOT CHAR next to the real Unicode glyph
    # from an adjacent slot ('7→' = mapsto, slot 0x37). Digits are never
    # typeset from these fonts, so slot-char+glyph pairs are unambiguous.
    # Table-driven: extend as new leaks are measured (same pattern as
    # _CMEX_LETTER_SLOTS above).
    if re.search(r"(?i)mathsymbol|cmsy|msam|msbm|lmmath", font or ""):
        out = _OMS_SLOT_LEAK_RE.sub(
            lambda m: _OMS_SLOT_LEAK[m.group(0)] + " ", out
        )
    if out.strip() == "\\" and re.search(
        r"(?i)mathsymbol|cmsy|cmex|msam|msbm|lmmath",
        font or "",
    ):
        # Math-symbol fonts whose setminus slot ToUnicode'd to a literal
        # backslash (LMMathSymbols10, measured 06597 S_β\P_β^Z).
        return r"\setminus "
    for u, latex in sorted(_FONT_MATH_CHARS.items(), key=lambda x: -len(x[0])):
        if u in out:
            out = out.replace(u, latex + " ")
    # Font-conditioned twin: in math-italic fonts (CMMI/LMMathItalic) BOTH
    # phi codepoints render as the CURLY glyph — ToUnicode may emit either
    # U+03C6 or U+03D5. Gold uses \varphi 3.5:1 (measured over full gold
    # set). Only roman/unicode-text contexts keep the straight \phi map.
    if (
        not _PAGE_PHI_DUAL
        and re.search(r"cmmi|mathitalic|math.?it", font or "", re.I)
    ):
        # Single-codepoint document: the author used one macro for both
        # forms — majority gold form is \varphi (3.5:1). Dual-codepoint
        # documents distinguish: U+03C6 curly → \varphi (char map),
        # U+03D5 straight → \phi (char map, untouched). (iter-18)
        out = re.sub(r"\\phi(?![a-zA-Z])", r"\\varphi", out)
    # Prime (U+2032) → ASCII apostrophe; KaTeX strict mode rejects the
    # Unicode char in math mode but renders ' as \prime.
    out = out.replace("′", "'").replace("″", "''")
    # KaTeX strict mode: bare # is a macro-parameter char — escape it
    # (seen in "#S" set-cardinality notation).
    out = out.replace("#", "\\#")
    # Set delimiters from CMSY braces → \{ \} (KaTeX rejects bare {} in
    # math mode as grouping, breaking unbalanced set notation).
    out = out.replace("{", "\\{ ").replace("}", "\\} ")
    # COMBINING VECTOR ARROW U+20D7 (iter-114, 2503.08227 pg2): CMMI
    # vector spans carry the arrow AFTER the base ('r⃗') or, from
    # text-order quirks, BEFORE it ('∂⃗n' — the arrow belongs to the NEXT
    # char). Raw combining marks render nowhere in KaTeX; both orders
    # become \vec{base}. Runs AFTER brace-escaping so the generated
    # braces are not re-escaped.
    out = re.sub(r"([A-Za-z0-9])⃗", r"\\vec{\1} ", out)
    out = re.sub(r"⃗\s*([A-Za-z0-9])", r"\\vec{\1} ", out)
    out = out.replace("⃗", "")
    if _BB_FONT_RE.search(font or ""):
        out = re.sub(r"(?<![\\a-zA-Z])([A-Z])(?![a-zA-Z])", r"\\mathbb{\1}", out)
    elif _BOLD_MATH_RE.search(font or "") and not out.startswith(
        ("\\bm", "\\boldsymbol", "\\text")
    ):
        # Bold math fonts (CMBX/CMMIB/bold-italic) = \bm{...} — gold uses
        # \bm for bold vectors/matrices (flip-diff 2503.06703_pg4 math_007:
        # '\bm{\varphi}_{k,m}' vs bare '\varphi_{k,m}').
        out = r"\bm{" + out.strip() + "}"
    elif _STYLE_FONT_RE.search(font or ""):
        out = re.sub(r"(?<![\\a-zA-Z])([A-Za-z])(?![a-zA-Z])", r"\\mathcal{\1}", out)
    elif re.search(r"mathsymbol|cmsy", font or "", re.I) and re.fullmatch(
        r"\s*[A-Z]\s*", out.replace("\\", "")
    ):
        # OMS/cmsy encoding: ASCII A-Z slots ARE the script/caligraphic
        # capitals (\mathcal alphabet lives in cmsy/LMMathSymbols). A bare
        # uppercase letter from this font renders as \mathcal{X} (measured:
        # ' D' in LMMathSymbols10 = \mathcal{D}, 2503.05614_pg12).
        out = re.sub(r"([A-Z])", r"\\mathcal{\1}", out.strip())
    elif _FRAK_FONT_RE.search(font or ""):
        out = re.sub(r"(?<![\\a-zA-Z])([A-Za-z])(?![a-zA-Z])", r"\\mathfrak{\1}", out)
    out = out.strip()
    if not out:
        return ""
    if (is_sup or is_sub) and len(out) <= 5 and not out.startswith(("\\", "^", "_")):
        return ("^" if is_sup else "_") + "{" + out + "}"
    return out


# Unicode → LaTeX for chars commonly found inside math-font spans
_FONT_MATH_CHARS = {
# Sentinel for glyph-evidenced tall vertical bars (iter-136): broken
# ToUnicode decodes the \\Big| evaluation-bar glyph to '.' (measured
# 04993 pg18: glyph w=3.03 h=10.91, h/w=3.6 — a real period is ~1:1).
# The intake rewrite below converts tall-thin '.' glyphs to this char.
"│": r"\Big|",
    # Greek lowercase (CMMI encoding often maps to Unicode Greek)
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\varepsilon",
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
    "φ": r"\varphi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    # Greek uppercase
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    # Operators
    "×": r"\times",
    "·": r"\cdot",
    "÷": r"\div",
    "±": r"\pm",
    "∓": r"\mp",
    "∗": r"\ast",
    "∘": r"\circ",
    "◦": r"\circ",  # U+25E6 white bullet — LMMathSymbols' \circ slot
    # Relations
    "≤": r"\leq",
    "≥": r"\geq",
    "≠": r"\neq",
    "≡": r"\equiv",
    "≈": r"\approx",
    "∼": r"\sim",
    "≃": r"\simeq",
    "≅": r"\cong",
    "≍": r"\asymp",
    "⋍": r"\backsimeq",
    "⊊": r"\subsetneq",
    "⊋": r"\supsetneq",
    "⊑": r"\sqsubseteq",
    "⊒": r"\sqsupseteq",
    "⊏": r"\sqsubset",
    "⊐": r"\sqsupset",
    "⊔": r"\sqcup",
    "⊓": r"\sqcap",
    "⊴": r"\trianglelefteq",
    "⊵": r"\trianglerighteq",
    "◁": r"\lhd",
    "▷": r"\rhd",
    "◂": r"\blacktriangleleft",
    "▸": r"\blacktriangleright",
    "⨁": r"\bigoplus",
    "⨂": r"\bigotimes",
    "⨄": r"\biguplus",
    "⨆": r"\bigsqcup",
    "⋎": r"\curlyvee",
    "⋏": r"\curlywedge",
    "≀": r"\wr",
    "⌣": r"\smile",
    "⌢": r"\frown",
    "⊆": r"\subseteq",
    "⊂": r"\subset",
    "⊇": r"\supseteq",
    "⊃": r"\supset",
    "∈": r"\in",
    "∉": r"\not\in",  # composed form — gold MathML majority
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
    # Blackboard bold (MSBM font)
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℚ": r"\mathbb{Q}",
    "ℍ": r"\mathbb{H}",
    "𝔽": r"\mathbb{F}",
    "𝔼": r"\mathbb{E}",
    # Misc
    "∞": r"\infty",
    "∂": r"\partial",
    "∇": r"\nabla",
    "√": r"\sqrt",
    "∫": r"\int",
    "∮": r"\oint",
    "∑": r"\sum",
    "∏": r"\prod",
    "∀": r"\forall",
    "∃": r"\exists",
    "→": r"\to",
    "↔": r"\leftrightarrow",
    "⇒": r"\Rightarrow",
    "⇔": r"\Leftrightarrow",
    "↦": r"\mapsto",
    "⊕": r"\oplus",
    "⊗": r"\otimes",
    "⊙": r"\odot",
    "⊥": r"\perp",
    # iter-114: ∥ from a math-font span is the NORM delimiter — the gold
    # corpus writes \| 85× and \parallel 0× (TeX \| renders as this same
    # glyph, and a relation never carries a subscript: these are norms).
    "∥": r"\|",
    "∝": r"\propto",
    "•": r"\bullet",
    "⋆": r"\star",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "‖": r"\|",
    "⋮": r"\vdots",
    "⋯": r"\cdots",
    "⋱": r"\ddots",
    # Letterlike symbols common in CMMI/MSBM spans
    "ℓ": r"\ell",
    "∠": r"\angle",
    "⪰": r"\succeq",
    "≻": r"\succ",
    "≺": r"\prec",
    "⪯": r"\preceq",
    "ȷ": r"\jmath",
    "ı": r"\imath",
    "ℵ": r"\aleph",
    "∗": r"\ast",
    # Codepoint twins TeX fonts map to (U+2206 increment vs U+0394 Greek,
    # U+2126 ohm vs U+03A9 omega, AMS slanted relations, ligatures)
    "∆": r"\Delta",
    "Ω": r"\Omega",
    "⩾": r"\geq",
    "⩽": r"\leq",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    # Strict-KaTeX-rejected glyphs seen on bench (vowel-variant Greek codepoints,
    # daggers, turnstiles)
    "ǫ": r"\epsilon",
    "µ": r"\mu",
    "ϕ": r"\phi",
    "ϑ": r"\vartheta",
    "ϰ": r"\kappa",
    "ϱ": r"\varrho",
    "ϵ": r"\epsilon",
    "†": r"\dagger",
    "‡": r"\ddagger",
    "⊢": r"\vdash",
    "⊣": r"\dashv",
    "⊨": r"\models",
    # iter-114: raw leaks inside math-font spans (CMSY '≈ℜ' stayed raw —
    # the char map only ran on prose text; double-arrow as TWO glyphs
    # '⇐⇒' in one CMSY span must join to the single AMS command gold uses)
    "ℜ": r"\Re",
    "ℑ": r"\Im",
    "⇐": r"\Leftarrow",
    "⇐⇒": r"\Longleftrightarrow",
}


def has_math_fonts(page: Any) -> bool:
    """Quick check: does this page contain any math-font spans?

    Used to decide whether to use font-aware extraction vs plain pdftotext.
    """
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type", 0) != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                if _STRICT_MATH_FONT_RE.search(font):
                    return True
    return False


__all__ = [
    "extract_text_with_math_fonts",
    "collect_math_clusters",
    "has_math_fonts",
]
