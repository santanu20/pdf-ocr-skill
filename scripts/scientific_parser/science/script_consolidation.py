"""Double-script consolidation (iter-130).

Measured family (census KaTeX errors, fresh 60.03% outputs): 86 unique
equations fail KaTeX with Double(sub|super)script — adjacent
same-direction script groups on one base
(``\\int_{Z}_{R}``, ``\\Lambda_{\\bf n}_{*}``, ``x^{1}^{-}^{p}``).
Adjacent same-direction groups are ALWAYS invalid LaTeX, so the merge
can never damage a valid equation; still adoption is oracle-gated:
rewrite only math regions that currently fail KaTeX and whose
consolidated form renders clean (``\\text{}`` literal content is
never touched — the oracle rejects such rewrites unless they render).
"""
from __future__ import annotations

import re

_ADJ_SUP = re.compile(r"(\^)\{([^{}]*)\}\s*(\^)\{([^{}]*)\}")
_ADJ_SUB = re.compile(r"(_)\{([^{}]*)\}\s*(_)\{([^{}]*)\}")
_SUSPECT = re.compile(r"[\^_]\{[^{}]*\}\s*[\^_]\{")
_MATH_SPANS = re.compile(r"(\$\$.+?\$\$|\$[^\n$]+\$)", re.DOTALL)


def consolidate_scripts(s: str) -> str:
    """Merge adjacent same-direction script groups (idempotent)."""
    prev = None
    while prev != s:
        prev = s
        s = _ADJ_SUP.sub(r"\1{\2 \4}", s)
        s = _ADJ_SUB.sub(r"\1{\2 \4}", s)
    return s


def repair_line_scripts(line: str, katex_ok) -> str:
    """Oracle-gated consolidation of one output line's math regions."""
    if "$" not in line or not _SUSPECT.search(line):
        return line

    def _fix(m: re.Match[str]) -> str:
        region = m.group(0)
        display = region.startswith("$$")
        inner = region[2:-2] if display else region[1:-1]
        if katex_ok(inner):
            return region  # healthy — untouched
        cand = consolidate_scripts(inner)
        if cand != inner and katex_ok(cand):
            d = "$$" if display else "$"
            return d + cand + d
        return region

    return _MATH_SPANS.sub(_fix, line)
