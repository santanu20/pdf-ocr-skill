"""iter-125 prototype: inline math group repair (merge + expel).

Measured family (attr_arxiv_full, 55% of residual fails = plain-inline):
our inline emission wraps WRONG sub-runs —
    '$h  $be a function $h  $∶ $\\mathbb{R}^p \\times ... \\mathbb{R}  $satisfying $'
Prose inside math ('satisfying'), operators outside ('∶'), adjacent
groups unmerged. Gold needs one contiguous inline span.

Repair (gold-free, mechanical):
  1. MERGE across operator bridges: '$A$ <op-only> $B$' -> '$A <op> B$'
     where <op-only> = colon/arrow/relation/punct + whitespace.
  2. EXPEL boundary prose: a LEADING or TRAILING run of pure-alpha
     words (>=4 letters, no digits/operators/backslashes) inside a math
     group moves outside the delimiters.
"""
from __future__ import annotations

import re

_UNI_TO_TEX = {
    "∶": ":", "→": r"\to ", "←": r"\leftarrow ", "↔": r"\leftrightarrow ",
    "×": r"\times ", "·": r"\cdot ", "−": "-", "–": "-", "≤": r"\leq ",
    "≥": r"\geq ", "≠": r"\neq ", "∼": r"\sim ", "≈": r"\approx ",
}


def _texify(seg: str) -> str:
    for u, t in _UNI_TO_TEX.items():
        seg = seg.replace(u, t)
    return seg


_OP_BRIDGE = re.compile(r"^[\s∶:;\|\.\,\-\–\=\<\>→←↔×·\*\+\~\(\)\[\]]+$")

_PROSE_WORD = re.compile(r"^[A-Za-z]{4,}$")

_CURRENCY = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?:\D|$)|\d+\.\d{2}\b")


def _math_spans(line: str):
    """[(start, end)] of $...$ spans, non-greedy, skipping $$.

    CURRENCY MASK (iter-132): a '$' opening a currency amount
    ('$164,700', '$8,000', '$12.50' — thousands groups or a 2-dp
    decimal) is literal prose, never a delimiter. Measured regression:
    tax-document sentences paired across '$164,700 … total $20,000'
    and scrambled by merge/expel (0083fb21 pg5, -2 order tests).
    """
    out = []
    i = 0
    n = len(line)
    while i < n:
        if line.startswith("$$", i):
            j = line.find("$$", i + 2)
            i = (j + 2) if j != -1 else n
            continue
        if line[i] == "$":
            if _CURRENCY.match(line[i + 1 :]):
                i += 1
                continue
            j = line.find("$", i + 1)
            if j == -1:
                break
            out.append((i, j + 1))
            i = j + 1
        else:
            i += 1
    return out


def _expel_edges(inner: str):
    """Move leading/trailing pure-prose word runs out of a math body."""
    toks = inner.split()
    if not toks:
        return inner, "", ""
    lead = []
    k = 0
    while k < len(toks) and _PROSE_WORD.match(toks[k].strip(",;:.")):
        lead.append(toks[k])
        k += 1
    trail = []
    m = len(toks) - 1
    while m >= k and _PROSE_WORD.match(toks[m].strip(",;:.")):
        trail.insert(0, toks[m])
        m -= 1
    if not lead and not trail:
        return inner, "", ""
    core = " ".join(toks[k : m + 1])
    return core, " ".join(lead), " ".join(trail)


def repair_inline_math(line: str) -> str:
    """Apply merge + expel to one emitted line."""
    changed = True
    while changed:
        changed = False
        spans = _math_spans(line)
        # MERGE pass
        for a in range(len(spans) - 1):
            s0, e0 = spans[a]
            s1, e1 = spans[a + 1]
            between = line[e0:s1]
            if _OP_BRIDGE.match(between) and between.strip() != "":
                merged = (
                    _texify(line[s0 + 1 : e0 - 1].strip())
                    + _texify(between)
                    + _texify(line[s1 + 1 : e1 - 1].strip())
                )
                line = line[:s0] + "$" + merged + "$" + line[e1:]
                changed = True
                break
            if between.strip() == "":
                merged = (
                    _texify(line[s0 + 1 : e0 - 1].strip())
                    + " "
                    + _texify(line[s1 + 1 : e1 - 1].strip())
                )
                line = line[:s0] + "$" + merged + "$" + line[e1:]
                changed = True
                break
        for s0, e0 in _math_spans(line):
            inner = line[s0 + 1 : e0 - 1]
            core, lead, trail = _expel_edges(inner)
            if core != inner:
                parts = [p for p in (lead, "$" + core + "$", trail) if p]
                line = line[:s0] + " ".join(parts) + line[e0:]
                changed = True
                break
    return line


if __name__ == "__main__":
    sample = (
        "$h  $be a function $h  $∶ $\\mathbb{R}^p   \\times   \\mathbb{R}^p   \\to   \\mathbb{R}  $satisfying $"
    )
    out = repair_inline_math(sample)
    print("IN :", sample)
    print("OUT:", out)
    gold = "h: \\mathbb{R}^p \\times \\mathbb{R}^p \\to \\mathbb{R}"
    flat = out.replace("∶", ":")
    print("gold-in-out:", gold.replace(" ", "") in flat.replace(" ", ""))
