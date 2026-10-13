"""iter-135b: beam V2-V7 generators + offline yield measurement.

Variant set (all content-preserving, oracle-gated in the beam):
  V1 consolidate adjacent same-direction scripts (proven module)
  V2 edge-punctuation strip (trailing ., ; spines / leading operators)
  V3 dangling \\sqrt -> \\sqrt{}
  V4 empty-group cleanup: '{ }' removal, '{}' drop, '}{ ' merge
  V5 \\left/\\right balance: drop the unmatched side
  V7 in-group space merge for script groups ('_{a b}' stays; 'x^{1 }^{2}'
     already V1)

Measure: census-log failing equations -> how many become render-clean
under beam_repair (depth 4, width 24).
"""
from __future__ import annotations

import re
import sys

sys.path.insert(0, "scripts")

_EDGE_JUNK = re.compile(r"^[\s=+\-,.:;]+|[\s=+\-,.:;]+$")


def _v2_strip_edges(s: str) -> list[str]:
    t = _EDGE_JUNK.sub("", s)
    return [t] if t and t != s else []


def _v4_empty_groups(s: str) -> list[str]:
    outs = set()
    t = s
    for pat, rep in [
        (r"\{\s*\}", ""),
        (r"\}\s*\{", ""),
        (r"\{\s+", "{"),
        (r"\s+\}", "}"),
    ]:
        t2 = re.sub(pat, rep, s)
        if t2 != s:
            outs.add(t2)
    return list(outs)


def _v5_delim_balance(s: str) -> list[str]:
    outs = set()
    n_l = len(re.findall(r"\\left\b", s))
    n_r = len(re.findall(r"\\right\b", s))
    if n_l > n_r:
        outs.add(re.sub(r"\\left\s*(?=[^A-Za-z])", "", s))
    elif n_r > n_l:
        outs.add(re.sub(r"\\right\s*(?=[^A-Za-z])", "", s))
    return [o for o in outs if o]


def variants(inner: str) -> list[str]:
    from scientific_parser.science.script_consolidation import (
        consolidate_scripts,
    )

    vs: list[str] = []
    c = consolidate_scripts(inner)
    if c != inner:
        vs.append(c)
    if inner.rstrip().endswith("\\sqrt"):
        vs.append(inner.rstrip() + "{}")
    vs += _v2_strip_edges(inner)
    vs += _v4_empty_groups(inner)
    vs += _v5_delim_balance(inner)
    return vs


def _charsig(s: str) -> str:
    return "".join(sorted(c for c in s if c.isalnum()))


def beam_repair(inner: str, katex_ok, max_depth: int = 4, width: int = 24) -> str | None:
    sig = _charsig(inner)
    seen = {inner}
    frontier = [inner]
    for _ in range(max_depth):
        nxt = []
        for cur in frontier:
            for v in variants(cur):
                if v in seen or _charsig(v) != sig:
                    continue
                seen.add(v)
                if katex_ok(v):
                    return v
                nxt.append(v)
        frontier = nxt[:width]
        if not frontier:
            break
    return None


if __name__ == "__main__":
    import io
    import logging

    logging.disable(logging.CRITICAL)
    _real = sys.stdout
    sys.stdout = io.StringIO()
    from scientific_parser.science.equation_cascade import katex_renders_clean

    sys.stdout = _real

    if len(sys.argv) < 2:
        raise SystemExit("usage: beam_repair.py <katex-failures.log>")
    LOG = sys.argv[1]
    txt = open(LOG, errors="replace").read()
    eqs = []
    for m in re.finditer(
        r"Error rendering equation: '(.*?)'\nKaTeX parse error:", txt, re.DOTALL
    ):
        if m.group(1) not in eqs:
            eqs.append(m.group(1))
    fixed = []
    for e in eqs:
        r = beam_repair(e, katex_renders_clean)
        if r is not None:
            fixed.append((e, r))
    print(f"unique failing: {len(eqs)}  beam-repairable: {len(fixed)}")
    for e, r in fixed[:8]:
        print(f"  IN : {e[:90]}")
        print(f"  OUT: {r[:90]}")
