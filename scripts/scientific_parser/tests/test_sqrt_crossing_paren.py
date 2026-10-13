"""Pin (2026-08-31): sqrt radicand closing-paren recovery on crossing spans.

Fixture geometry replicates arxiv_math/2503.05218_pg5 Conjecture 2.1
(measured via rawdict + get_drawings on the real PDF):

    radical CMEX10 'p' [508.9 462.5 519.8 473.4]
    vinculum rule    (519.8, 463.0) -> (567.1, 463.0)   w=47.3
    radicand spans   'αβ' [519.8-533.0] '(1' [533.6-543.3]
                      ' −' [543.3-554.2] 'γ' [556.6-562.2]
    crossing span    ')+ 2γ − 2'-prefix glued as ')+ ' [562.9-575.6]
                      — ')' closes the radicand's '(', '+' starts the tail.

Gold equation: T_min(α,β,γ) = 2√(αβ(1−γ)) + 2γ − 2  (olmOCR-bench
math test 2503.05218_pg5_math_008; smoke2 scored it FAIL with
'\\sqrt{\\alpha\\beta} (1 - \\gamma )' — radicand extent lost).

Contract:
  1. When the under-rule radicand has unclosed '(' and the crossing span
     starts with ')', the closers join the radicand and the span is trimmed
     in place — parens balanced inside s['sqrt'].
  2. KaTeX renders the resulting \\sqrt{...} cleanly (oracle invariant).
  3. Veto preserved: crossing span with NO paren evidence (glued 'πrT'
     family, iter-133 measured 05717 pg14) still blocks pairing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.science.font_math_detector import (  # noqa: E402
    _merge_fraction_clusters,
)


class _Rect:
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.width, self.height = x1 - x0, y1 - y0


class _FakePage:
    def __init__(self, rules):
        self._rules = rules

    def get_drawings(self):
        return [{"rect": r, "items": []} for r in self._rules]


def _span(x0, y0, x1, y1, text, font, size=10.9):
    return {
        "text": text,
        "bbox": (x0, y0, x1, y1),
        "font": font,
        "size": size,
        "origin": (x0, y1 - 2.0),
    }


def _conjecture_spans():
    """Math spans of the Conjecture 2.1 line (measured geometry)."""
    return [
        _span(425.5, 464.0, 434.7, 474.9, " T", "CMMI10"),
        _span(450.7, 464.0, 454.9, 474.9, "(", "CMR10"),
        _span(454.9, 464.0, 484.1, 474.9, "α, β, γ", "CMMI10"),
        _span(484.7, 464.0, 508.9, 474.9, ") = 2", "CMR10"),
        _span(508.9, 462.5, 519.8, 473.4, "p", "CMEX10"),
        _span(519.8, 464.0, 533.0, 474.9, "αβ", "CMMI10"),
        _span(533.6, 464.0, 543.3, 474.9, "(1", "CMR10"),
        _span(543.3, 463.9, 554.2, 474.8, " −", "CMSY10"),
        _span(556.6, 464.0, 562.2, 474.9, "γ", "CMMI10"),
        _span(562.9, 464.0, 575.6, 474.9, ")+ ", "CMR10"),
    ]


def _clusters(spans):
    xs = [s["bbox"][0] for s in spans] + [s["bbox"][2] for s in spans]
    ys = [s["bbox"][1] for s in spans] + [s["bbox"][3] for s in spans]
    return [{"spans": spans, "bbox": (min(xs), min(ys), max(xs), max(ys))}]


VINCULUM = _Rect(519.8, 463.0, 567.1, 463.0)


def _radical_sqrt(clusters):
    for cl in clusters:
        for s in cl["spans"]:
            if s.get("sqrt"):
                return s
    return None


def test__sqrt_crossing_paren__radicand_balanced_and_trimmed():
    spans = _conjecture_spans()
    cross = spans[-1]  # the ')+ ' carrier — tracked by identity (bbox mutates)
    out = _merge_fraction_clusters(_clusters(spans), _FakePage([VINCULUM]))
    rs = _radical_sqrt(out)
    assert rs is not None, "sqrt pairing must fire on the conjecture line"
    sq = rs["sqrt"]
    for tok in (r"\alpha", r"\beta", r"\gamma"):
        assert tok in sq, f"radicand lost {tok}: {sq!r}"
    assert sq.count("(") == sq.count(")"), f"unbalanced parens: {sq!r}"
    # the crossing span keeps ONLY its post-radicand tail — no ')'
    assert ")" not in cross["text"], f"crossing span not trimmed: {cross['text']!r}"


def test__sqrt_crossing_paren__katex_renders_clean():
    pytest.importorskip("olmocr", reason="KaTeX render oracle lives in the 'bench' extra")
    spans = _conjecture_spans()
    out = _merge_fraction_clusters(_clusters(spans), _FakePage([VINCULUM]))
    rs = _radical_sqrt(out)
    assert rs is not None
    latex = "\\sqrt{" + rs["sqrt"] + "}"
    from olmocr.bench.katex.render import render_equation

    res = render_equation(latex)
    assert res is not None and not res.error, f"KaTeX rejected {latex!r}"


def test__sqrt_crossing_no_evidence__veto_preserved():
    """Glued 'πrT' crossing span with no open parens must still veto."""
    spans = _conjecture_spans()
    # radicand: αβ only (no '('), crossing span extends past rule end
    for s in spans:
        if s["text"] == "(1":
            s["text"] = "π"
        elif s["text"] == " −":
            s["text"] = "r"
        elif s["text"] == "γ":
            s["text"] = "T"
        elif s["text"] == ")+ ":
            s["text"] = "T+ "
    out = _merge_fraction_clusters(_clusters(spans), _FakePage([VINCULUM]))
    rs = _radical_sqrt(out)
    assert rs is None, (
        f"veto must survive: no paren evidence on crossing span (got sqrt={rs['sqrt']!r})"
    )
