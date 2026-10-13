"""iter-126 prototype: Nougat close_envs port (MIT, behavioral).

Nougat postprocessing.close_envs: count \\begin{env} vs \\end{env};
append the missing closers. Our census KaTeX flood shows unclosed
cases/bmatrix/array displays ('Expected EOF, got \\end' / truncated
cases) — render-fail = no MathML = automatic test fail even when the
content is otherwise right.

Port: broader env list (nougat ships bmatrix/pmatrix/matrix/tabular/
table; we add cases/aligned/gathered/split/array/alignedat), wired at
final assembly. KaTeX-gated: only when the unclosed count is >0.
"""
from __future__ import annotations

import re

_ENVS = (
    "bmatrix", "pmatrix", "matrix", "vmatrix", "Bmatrix", "smallmatrix",
    "tabular", "table", "cases", "dcases", "aligned", "alignedat",
    "gathered", "split", "array", "equation", "equation*", "align",
    "align*", "alignat", "multline",
)


def close_envs(s: str) -> str:
    for env in _ENVS:
        begins = len(re.findall(r"\\begin\{%s\}" % re.escape(env), s))
        ends = len(re.findall(r"\\end\{%s\}" % re.escape(env), s))
        if begins > ends:
            s += (r"\end{%s}" % env) * (begins - ends)
    return s


if __name__ == "__main__":
    a = r"\begin{cases} u > 0 \\ ( - \Delta )^s u = |u|^{2^\ast} s u"
    print(close_envs(a))
    assert close_envs(a).endswith(r"\end{cases}")
    b = r"\begin{bmatrix} 0 \& 1 \end{bmatrix}"
    assert close_envs(b) == b  # balanced untouched
    print("close_envs OK")


def close_envs_line(line: str) -> str:
    """Per-line variant: close unbalanced envs at line end (iter-126).

    Emission lines are the pipeline's unit (cascade envs are prebuilt
    single-line); table pipe rows and non-math lines pass through.
    """
    if "$" not in line or line.lstrip().startswith("|"):
        return line
    return close_envs(line)
