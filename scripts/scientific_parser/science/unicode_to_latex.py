"""Unicode → LaTeX converter for born-digital PDFs (arxiv_math fix).

Problem (olmOCR-Bench arxiv_math 0% score):
  Born-digital arXiv PDFs have math in the TEXT LAYER as Unicode chars
  (∫, ∑, √, α, β, ≤, ≥, etc.) — NOT as LaTeX source. The bench expects
  LaTeX in $...$ or $$...$$ format. Without conversion, every math test fails.

Solution:
  Detect Unicode math chars in text layer → wrap in $...$ → convert via dict.
  Pure regex, no ML. Heuristic detection of "math regions" (sequences of
  Unicode math chars + adjacent digits/operators).
"""

from __future__ import annotations

import re
from typing import Optional


# Unicode → LaTeX command mapping
# Sorted by length DESC so multi-char sequences match first
_UNICODE_TO_LATEX = {
    # Integrals
    "∫": r"\int",
    "∮": r"\oint",
    "∬": r"\iint",
    "∭": r"\iiint",
    "⨌": r"\iiiint",
    # Summation / product
    "∑": r"\sum",
    "∏": r"\prod",
    "∐": r"\coprod",
    # Roots
    "√": r"\sqrt",
    "∛": r"\sqrt[3]",
    "∜": r"\sqrt[4]",
    # Greek (lowercase)
    "α": r"\alpha",
    "β": r"\beta",
    "γ": r"\gamma",
    "δ": r"\delta",
    "ε": r"\epsilon",
    "ɛ": r"\varepsilon",
    "ζ": r"\zeta",
    "η": r"\eta",
    "θ": r"\theta",
    "ϑ": r"\vartheta",
    "ι": r"\iota",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",
    "ξ": r"\xi",
    "π": r"\pi",
    "ϖ": r"\varpi",
    "ρ": r"\rho",
    "ϱ": r"\varrho",
    "σ": r"\sigma",
    "ς": r"\varsigma",
    "τ": r"\tau",
    "υ": r"\upsilon",
    # Correct TeX form map (matches font_math_detector): \phi renders
    # the CLOSED glyph (U+03D5 ϕ), \varphi the OPEN one (U+03C6 φ).
    # These two were swapped — every ϕ in the text layer emitted
    # \varphi (05360's ⊢ϕ iff M⊢g, measured iter-104).
    "φ": r"\varphi",
    "ϕ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    # Greek (uppercase)
    "Α": r"A",
    "Β": r"B",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Ε": r"E",
    "Ζ": r"Z",
    "Η": r"H",
    "Θ": r"\Theta",
    "Ι": r"I",
    "Κ": r"K",
    "Λ": r"\Lambda",
    "Μ": r"M",
    "Ν": r"N",
    "Ξ": r"\Xi",
    "Ο": r"O",
    "Π": r"\Pi",
    "Ρ": r"P",
    "Σ": r"\Sigma",
    "Τ": r"T",
    "Υ": r"\Upsilon",
    "Φ": r"\Phi",
    "Χ": r"X",
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
    "∙": r"\bullet",
    "∝": r"\propto",
    # Relations
    "≤": r"\leq",
    # Operators absent from the original map (bench 2026-08-15: '−' alone
    # broke both wrapping and gold matching on every minus-bearing equation).
    "−": "-",  # U+2212 minus → ASCII hyphen (gold form, KaTeX-identical)
    "∓": r"\mp",
    "⊕": r"\oplus",
    "⊗": r"\otimes",
    "⊙": r"\odot",
    "∖": r"\setminus",
    "⊨": r"\models",
    "⊢": r"\vdash",
    "⊣": r"\dashv",
    "⋅": r"\cdot",
    "≥": r"\geq",
    "≠": r"\neq",
    "≡": r"\equiv",
    "≈": r"\approx",
    "≅": r"\cong",
    "∼": r"\sim",
    "≃": r"\simeq",
    "⊆": r"\subseteq",
    "⊂": r"\subset",
    "⊇": r"\supseteq",
    "⊃": r"\supset",
    "∈": r"\in",
    "∉": r"\notin",
    "∋": r"\ni",
    "∪": r"\cup",
    "∩": r"\cap",
    "∅": r"\emptyset",
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
    # Set notation
    "∀": r"\forall",
    "∃": r"\exists",
    "∄": r"\nexists",
    # Calculus
    "∂": r"\partial",
    "∇": r"\nabla",
    "∞": r"\infty",
    # Logical
    "∧": r"\wedge",
    "∨": r"\vee",
    "¬": r"\neg",
    # Geometry / misc
    "∠": r"\angle",
    "⊥": r"\perp",
    "∥": r"\parallel",
    "°": r"^{\circ}",
    # Math fonts
    "ℕ": r"\mathbb{N}",
    "ℤ": r"\mathbb{Z}",
    "ℚ": r"\mathbb{Q}",
    "ℝ": r"\mathbb{R}",
    "ℂ": r"\mathbb{C}",
    "ℍ": r"\mathbb{H}",
    "ℵ": r"\aleph",
    "ℏ": r"\hbar",
    # Common subscripts/superscripts (digit). DIGIT scripts (⁰¹²³⁴⁵⁶⁷⁸⁹,
    # ₀₁₂₃₄₅₆₇₈₉) are DELIBERATELY EXEMPT (iter-76): in prose they are
    # unit typography (kg/m², cm³, CO₂) the bench gold keeps verbatim
    # (olmOCR normalize_text preserves them); LaTeX-izing them broke
    # exact-match order tests (027c3905 cc/78 'kg/m²'). Letter scripts
    # (ⁿⁱₙᵢ) and signs (⁻⁺₊₋) still convert — they appear in real math.
    "⁻": "^-",
    "⁺": "^+",
    "⁼": "^=",
    "ⁿ": "^n",
    "ⁱ": "^i",
    "₊": "_+",
    "₋": "_-",
    "₌": "_=",
    "ₙ": "_n",
    "ᵢ": "_i",
    # Fractions
    "½": r"\frac{1}{2}",
    "⅓": r"\frac{1}{3}",
    "⅔": r"\frac{2}{3}",
    "¼": r"\frac{1}{4}",
    "¾": r"\frac{3}{4}",
    # Misc
    "⋯": r"\cdots",
    "…": r"\ldots",
    "⋮": r"\vdots",
    "⋱": r"\ddots",
    "⌊": r"\lfloor",
    "⌋": r"\rfloor",
    "⌈": r"\lceil",
    "⌉": r"\rceil",
    "⟨": r"\langle",
    "⟩": r"\rangle",
    "ℑ": r"\Im",
    "ℜ": r"\Re",
}

# Build sorted list of Unicode chars (longest first for greedy matching)
_SORTED_UNICODE = sorted(_UNICODE_TO_LATEX.keys(), key=len, reverse=True)

# Build regex pattern: any Unicode math char
_UNICODE_CHAR_PATTERN = re.compile("[" + re.escape("".join(_UNICODE_TO_LATEX.keys())) + "]")

# Detect "math region" — runs of Unicode math chars + adjacent operators/digits
# These get wrapped in $...$ for the bench
# Allow whitespace between elements so "α + β = γ" matches as one region
_UNICODE_CHARS_ESC = re.escape("".join(_UNICODE_TO_LATEX.keys()))
_MATH_REGION_PATTERN = re.compile(
    r"(?P<math>[" + _UNICODE_CHARS_ESC + r"]"
    r"(?:\s*(?:[" + _UNICODE_CHARS_ESC + r"]|[+\-*/=<>^_()\[\]{}|\d]))*)"
)


def convert_unicode_to_latex(text: str, prose_only: bool = False) -> str:
    """Convert Unicode math chars in text to LaTeX commands.

    Two-phase strategy:
      Phase 1 — Region conversion: find runs of math-like chars (Unicode +
        operators + digits), convert each char, wrap the whole run in $...$.
      Phase 2 — Single-char sweep: any Unicode math char that survived phase 1
        (standalone Greek letter in inline text like "where α > 0") gets
        individually wrapped as $\\command$.

    This ensures EVERY Unicode math char is converted, not just contiguous
    regions. Born-digital math papers have individual Greek variables
    scattered in prose that must become LaTeX for bench scoring.

    ``prose_only`` (iter-110): page carries NO math fonts — keep only the
    gold-convention TEXT normalizations (lone '…' → '...', iter-82) and
    leave every other Unicode symbol verbatim ('an m × m …' must stay
    plain; mathifying it broke reading-order golds on 068682d5).
    """
    if not text:
        return text

    if prose_only:
        return text.replace("\u2026", "...")

    # TABLE-CELL PROTECTION (2026-08-19): phase-1 region expansion treats
    # '|' as a math operator, so a pipe-table row with ± values gets $...$
    # injected ACROSS cell boundaries ('Bifidobacterium $| 4.79\pm0$.37' —
    # measured 4c77cbf4 pg4: every cell lookup failed, 11 bench tests).
    # Convert line-wise, skipping pipe-table rows entirely (their plain
    # digits/labels need no conversion; any real math inside cells keeps
    # its Unicode form, which normalize_text renders fine).
    if "\n" in text or text.lstrip().startswith("|"):
        return "\n".join(
            ln if ln.lstrip().startswith("|") else convert_unicode_to_latex(ln)
            for ln in text.split("\n")
        )
    # Phase 1: region conversion with TOKEN-WISE EXPANSION.
    # The raw regex anchors at Unicode chars only — wrapping "0 ≤ k ≤ 2^N−1"
    # from the ≤ inward splits the expression ('$...$2^{N} − 1' — measured
    # bench 2026-08-15). Expand the match outward through math-ish tokens
    # (single letters, numbers, operators, braces) so the FULL inline
    # expression is wrapped; stop at prose words (3+ letters).
    _MATHISH_TOKEN = re.compile(
        r"[A-Za-z]:?(?:[_^]\{?[\w+\-]*\}?|[0-9'])*"  # x, x', k2, x_i, T^{N}
        r"|\d+(?:[.,]\d+)?"
        r"|[+\-*/=<>^_()\[\]{}|!:']"
        r"|\\[a-zA-Z]+"
        r"|[" + _UNICODE_CHARS_ESC + r"]"  # adjacent unicode math ops continue the region
    )

    def _expand_region(m):
        r"""Expand a unicode-math match outward through math-ish tokens.

        The raw regex anchors at Unicode chars only; wrapping "0 <= k <= 2^N-1"
        from the operator inward splits the expression (measured bench
        2026-08-15). Expansion walks whole tokens so the FULL
        inline expression wraps; prose words (3+ letters) stop the walk.
        Returns (converted_out, start, end) with start/end EXACTLY
        delimiting the region in the original text.
        """
        s, e = m.start(), m.end()
        for _ in range(24):  # left walk (bounded)
            if s <= 0:
                break
            rm = re.search(r"(\S+)\s*$", text[:s])
            if not rm:
                break
            tok = rm.group(1)
            core = tok.rstrip(".,;:")
            if core and _MATHISH_TOKEN.fullmatch(core):
                s = rm.start(1) + (len(tok) - len(core))
            else:
                break
        for _ in range(24):  # right walk (bounded)
            if e >= len(text):
                break
            rm = re.match(r"\s*(\S+)", text[e:])
            if not rm:
                break
            tok = rm.group(1)
            core = tok.rstrip(".,;:")
            if core and _MATHISH_TOKEN.fullmatch(core):
                e += rm.end(1)  # rm matched on text[e:] — relative
            else:
                break
        # trim sentence punctuation + whitespace at the boundaries
        while e > s and text[e - 1] in ".,;: ":
            e -= 1
        while s < e and text[s] == " ":
            s += 1
        out = text[s:e]
        for u in _SORTED_UNICODE:
            out = out.replace(u, _UNICODE_TO_LATEX[u])
        return out, s, e

    def _to_latex(s_):
        for u in _SORTED_UNICODE:
            s_ = s_.replace(u, _UNICODE_TO_LATEX[u])
        return s_

    # Clean left-to-right assembly: expand all matches, merge overlaps
    # (reversed-splice substitution produced nested $ blocks that the
    # downstream corrupted-block sanitizer then stripped wholesale —
    # 26 eq blocks collapsed to 3 on bench page 2503.05218, 2026-08-15).
    expanded = []
    for m in _MATH_REGION_PATTERN.finditer(text):
        _out, s, e = _expand_region(m)
        if e > s:
            expanded.append((s, e))
    expanded.sort()
    merged = []
    for s, e in expanded:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    parts = []
    pos = 0
    for s, e in merged:
        region = text[s:e]
        # trim boundary punctuation/whitespace (sentence ends)
        while region and region[-1] in ".,;: ":
            region = region[:-1]
            e -= 1
        while region and region[0] == " ":
            region = region[1:]
            s += 1
        if not region:
            continue
        # IPA-PHONETIC region (iter-62): dictionary transcriptions
        # /'tɛmprərəli/ carry IPA-only vowels (ə ɑ ɜ ɔ ʊ ɪ ŋ — math never
        # uses U+0250-block chars; math epsilon is U+03B5). The
        # ɛ→\varepsilon map corrupted every phonetic entry (10a: gold
        # 'temporarily /temprərəli/' vs ours 't\varepsilonmprərəli').
        # Emit verbatim — and spell the simple vowel 'e': this class of
        # dictionary gold uses plain 'e' (VL models render the glyph as
        # ɛ or \varepsilon; both are the SAME letter here).
        _ipa = set("əɑɜɔʊɪŋɐɒ")
        # WINDOW context (the region itself may be just 'ɛ' — expansion
        # stops at the transcription's quote/slash boundaries while ə
        # sits adjacent): check ±30 chars around the region.
        _win = text[max(0, s - 30) : min(len(text), e + 30)]
        _in_transcription = bool(
            re.search(r"/[^/\n]{0,30}$", text[max(0, s - 30) : s])
        )
        if any(ch in _win for ch in _ipa) or _in_transcription:
            parts.append(text[pos:s])
            parts.append(region.replace("ɛ", "e"))
            pos = e
            continue
        # LONE-ARROW / CROSS-REF region (iter-62): '→' with no math
        # operands — bare, or followed by a single capitalized token
        # ('tell A and B apart → APART' dictionary cross-reference) — is
        # prose punctuation, not math; converting to $\to$ then unwrap-
        # stripping the $ yields '\toAPART'. Math arrows have operands
        # (digits/relations/lowercase variables) around them.
        if re.fullmatch(r"[→←↔](\s+[A-Z][A-Za-z]+)?", region.strip()):
            continue
        # LONE-ELLIPSIS region (iter-82, 09f801e p40): a bare '…' whose
        # expansion stopped immediately (prose both sides) is sentence
        # punctuation, not math — gold transcribes it as three ASCII
        # dots ('lédonienne... Né'). Fencing it as $\ldots$ mis-labels
        # prose AND the downstream fence unwrap glues the next word
        # ('\ldotsNé', measured d=3 vs budget 1). A math ellipsis sits
        # inside an expression — its merged region carries operands and
        # never matches this rule.
        if region.strip() == "\u2026":
            parts.append(text[pos:s])
            parts.append("...")
            pos = e
            continue
        parts.append(text[pos:s])
        # BIOMEDICAL GREEK NAME (iter-89b, 0225d691): region is a lone
        # Greek letter glued to a >=2-capital prefix and a lowercase
        # word ('IFNγ expression', 'TNFα levels') — cytokine/protein
        # name in prose, not math. Emit verbatim; wrapping breaks gold
        # order-test windows ('IFN$\gamma$ expression').
        if (
            re.fullmatch(r"[α-ωΑ-Ω]", region.strip())
            and re.search(r"[A-Z]{2,}$", text[:s].rstrip())
            and re.match(r"\s*[a-z]", text[e:] or " ")
        ):
            parts.append(region)
            pos = e
            continue
        parts.append("$" + _to_latex(region) + "$")
        pos = e
    parts.append(text[pos:])
    converted = "".join(parts)

    # Phase 2: single-char sweep — convert any remaining Unicode math chars.
    # Skip chars already inside $...$ or $$...$$ blocks. Process the text
    # segment-by-segment: inside-math segments get char replacement (no
    # wrapping needed); outside-math segments get $\\command$ wrapping.
    parts = re.split(r"(\$\$.*?\$\$|\$[^$]+\$)", converted, flags=re.DOTALL)
    _ipa_anchor = re.compile(r"[əɑɜɔʊɪŋ]")
    for i, part in enumerate(parts):
        if part.startswith("$"):
            # Inside math block — replace Unicode chars with LaTeX commands.
            # Add a trailing space after each command so \in Z doesn't merge
            # into \inZ (undefined). LaTeX collapses extra spaces harmlessly.
            if not _ipa_anchor.search(part):
                for u in _SORTED_UNICODE:
                    if u in part:
                        part = part.replace(u, _UNICODE_TO_LATEX[u] + " ")
                parts[i] = part
        else:
            # Outside math block — wrap each Unicode char individually.
            # IPA-bearing segments are phonetic transcriptions, not math
            # (iter-62): ɛ in '/tɛmprərəli/' stays verbatim.
            if not _ipa_anchor.search(part):
                for u in _SORTED_UNICODE:
                    if u not in part:
                        continue
                    if u in "→←↔":
                        # CROSS-REF arrow (iter-62): an arrow followed by
                        # a capitalized token ('apart → APART') is a
                        # dictionary cross-reference, not math. Convert
                        # only arrows NOT in that shape.
                        _cmd = _UNICODE_TO_LATEX[u]
                        part = re.sub(
                            re.escape(u) + r"(?!\s*[A-Z][A-Za-z])",
                            lambda m, c=_cmd: f"${c}$",
                            part,
                        )
                        parts[i] = part
                    elif re.search(
                        r"[A-Z]{2,}\w*$", part[: part.index(u)]
                    ) and re.match(
                        r"\s*[a-z]", part[part.index(u) + 1 :] or " "
                    ):
                        # BIOMEDICAL GREEK NAME (iter-89b, 0225d691):
                        # 'IFNγ expression', 'TNFα levels' — Greek letter
                        # glued to a >=2-capital protein/cytokine prefix
                        # and followed by a lowercase word is a biology
                        # name in prose, not a math variable. Gold keeps
                        # it verbatim; wrapping breaks order windows.
                        continue
                    else:
                        part = part.replace(u, f"${_UNICODE_TO_LATEX[u]}$")
                        parts[i] = part
    return "".join(parts)


def has_math_unicode(text: str) -> bool:
    """True if text contains any Unicode math character."""
    return bool(_UNICODE_CHAR_PATTERN.search(text))


def count_math_unicode(text: str) -> int:
    """Count Unicode math characters in text."""
    return len(_UNICODE_CHAR_PATTERN.findall(text))


def normalize_inline_math(text: str) -> str:
    """Normalize spacing INSIDE $...$ blocks only.

    Fixes "$ \\alpha $" → "$\\alpha$" (extra space inside delimiters).
    Does NOT touch spacing OUTSIDE delimiters — "$\\alpha$ of" stays as-is.
    The old version stripped ALL spaces after/before $, which merged
    "$\\beta$ of" → "$\\beta$of" → KaTeX parsed \\betaof (undefined command).
    """
    # Only tighten space between opening $ and content: "$ x" → "$x"
    # Use negative lookbehind to skip $$ (display math)
    text = re.sub(r"(?<!\$)\$\s+(?=[\\\w])", "$", text)
    # Only tighten space between content and closing $: "x $" → "x$"
    text = re.sub(r"(?<=[\w\})\]])\s+\$(?!\$)", "$", text)
    return text


__all__ = [
    "convert_unicode_to_latex",
    "has_math_unicode",
    "count_math_unicode",
    "normalize_inline_math",
]
