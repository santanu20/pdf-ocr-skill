"""Font-name weight/slant parsing for heading detection.

Ported from Docling (docling/utils/font_style.py, MIT License,
Copyright The Docling Contributors) — adapted to this codebase:
no docling_core dependency, frozen-dataclass output, plus a
page-level minority gate helper for heading promotion.

Why: detect_section_hierarchy promotes headings by SIZE separation
only (p90 >= 1.3x median). Books and reports commonly typeset
headings BOLD at BODY SIZE — invisible to the size signal. PDF font
names (/Helvetica-Bold, subset /NKDKGK+HelveticaNeueLTPro-Bd) are the
only styling metadata the text layer carries; this parser reads
weight and slant out of them conservatively (false "bold" silently
rewrites a heading level; a miss only falls back to prior behavior).

Docling's two conservative rules kept verbatim:
  1. Style words match as whole tokens (split on separators AND
     camel-case) — family "Bookman" is not "book" weight.
  2. Abbreviations honored only as a whole separator-delimited part —
     "LinLibertineTB"'s TB is a foundry tag, not bold.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

REGULAR_WEIGHT = 400

_SUBSET_PREFIX = re.compile(r"^[A-Z]{6}\+")
_SEPARATORS = re.compile(r"[-_,+ ]+")
_TOKENS = re.compile(r"[A-Z]+(?![a-z])|[A-Z][a-z]+|[a-z]+|\d+")

_WEIGHT_TOKENS = {
    "thin": 100,
    "hairline": 100,
    "extralight": 200,
    "ultralight": 200,
    "light": 300,
    "book": REGULAR_WEIGHT,
    "normal": REGULAR_WEIGHT,
    "plain": REGULAR_WEIGHT,
    "regular": REGULAR_WEIGHT,
    "roman": REGULAR_WEIGHT,  # upright, not italic, not a numeral
    "medium": 500,
    "demi": 600,
    "demibold": 600,
    "semi": 600,
    "semibold": 600,
    "bold": 700,
    "extrabold": 800,
    "ultrabold": 800,
    "black": 900,
    "fat": 900,
    "heavy": 900,
    "poster": 900,
    "ultra": 900,
}

_ITALIC_TOKENS = frozenset({"italic", "ital", "inclined", "kursiv", "oblique", "slanted"})

_WEIGHT_MODIFIERS = {
    "semi": {"bold": 600, "light": 350},
    "demi": {"bold": 600, "light": 350},
    "extra": {"bold": 800, "light": 200, "black": 900},
    "ultra": {"bold": 800, "light": 200, "black": 900},
    "x": {"bold": 800, "light": 200},
}

_PART_ABBREVIATIONS: dict[str, tuple[int | None, bool | None]] = {
    "b": (700, None),
    "bd": (700, None),
    "bi": (700, True),
    "bdit": (700, True),
    "blk": (900, None),
    "i": (None, True),
    "ita": (None, True),
    "it": (None, True),
    "lt": (300, None),
    "md": (500, None),
    "obl": (None, True),
    "reg": (REGULAR_WEIGHT, None),
    "rg": (REGULAR_WEIGHT, None),
    "rom": (REGULAR_WEIGHT, None),
    "sb": (600, None),
}


@dataclass(frozen=True)
class FontStyle:
    """Weight and slant read from a PDF font name."""

    weight: int = REGULAR_WEIGHT
    italic: bool = False
    known: bool = False  # False when the name carried no recognizable style


def weight_class(weight: int) -> int:
    """Bucket weight: 0 light/regular, 1 medium/semibold, 2 bold+.

    Coarse on purpose — heading promotion consumes the classes, and a
    finer scale would split near-identical styles.
    """
    if weight >= 700:
        return 2
    if weight >= 500:
        return 1
    return 0


@lru_cache(maxsize=1024)
def parse_font_style(font_name: str) -> FontStyle:
    """Read weight and slant from a PDF font name.

    Returns regular/upright with known=False when the name carries no
    recognizable styling (unstyled family, foundry-tagged name, or a
    bare resource key such as /F1).
    """
    name = _SUBSET_PREFIX.sub("", (font_name or "").lstrip("/"), count=1)
    if not name:
        return FontStyle()

    weight: int | None = None
    italic: bool | None = None

    for part in _SEPARATORS.split(name):
        if not part:
            continue
        abbreviation = _PART_ABBREVIATIONS.get(part.lower())
        if abbreviation is not None:
            part_weight, part_italic = abbreviation
            weight = part_weight if part_weight is not None else weight
            italic = part_italic if part_italic is not None else italic
            continue

        tokens = [token.lower() for token in _TOKENS.findall(part)]
        index = 0
        while index < len(tokens):
            token = tokens[index]
            modifier = _WEIGHT_MODIFIERS.get(token)
            if modifier is not None and index + 1 < len(tokens):
                combined = modifier.get(tokens[index + 1])
                if combined is not None:
                    weight = combined
                    index += 2
                    continue
            if token in _WEIGHT_TOKENS:
                weight = _WEIGHT_TOKENS[token]
            elif token in _ITALIC_TOKENS:
                italic = True
            index += 1

    if weight is None and italic is None:
        return FontStyle()
    return FontStyle(
        weight=weight if weight is not None else REGULAR_WEIGHT,
        italic=bool(italic),
        known=True,
    )


def bold_heading_lines(lines: list[dict], max_share: float = 0.3) -> list[str]:
    """Pick bold-same-size heading candidates from per-line records.

    Each record: {"text", "size", "font", "bold_share"?} (font = dominant
    font name by char count for that line; bold_share = fraction of the
    line's chars in PyMuPDF bold-flagged spans). A line qualifies when:

      * its dominant font parses to weight class 2 (>=700), OR the
        line is uniformly bold by the STRUCTURED span flag
        (bold_share >= 0.9) — the flag catches family-bold-is-medium
        naming (NimbusRomNo9L-Medi, the LaTeX \textbf face, parses as
        weight 500 by name; PyMuPDF still sets its bold bit). This is
        LiteParse is_bold_item's ordering: descriptor flags first,
        name parsing second;
      * its size is within 15% of the page median (same-size headings —
        the size-based pass handles the separated case);
      * bold lines are a MINORITY (<= max_share of non-empty lines) —
        a page set entirely in bold carries no heading signal, and
        bold-run-in leads must never promote wholesale.

    Returns the qualifying line texts (title-shape gates and the
    standalone-line gate are applied by the caller, which owns the
    surrounding text context).
    """
    nonempty = [ln for ln in lines if ln.get("text", "").strip()]
    if len(nonempty) < 5:
        return []
    sizes = sorted(ln["size"] for ln in nonempty)
    median_size = sizes[len(sizes) // 2] or 10.0

    def _is_bold(ln) -> bool:
        if ln.get("bold_share", 0.0) >= 0.9:
            return True
        return weight_class(parse_font_style(ln.get("font", "")).weight) == 2

    bold = [
        ln
        for ln in nonempty
        if _is_bold(ln) and abs(ln["size"] - median_size) <= 0.15 * median_size
    ]
    if not bold or len(bold) > max_share * len(nonempty):
        return []
    return [ln["text"].strip() for ln in bold]
