"""PIN tests for repair_unrenderable_math (run-9 postproc-only step).

Census basis (2026-09-10): 197 render-fail of 20,742
arxiv equations; 14 adopt-a-rendering-variant (12 brace EOF-truncation,
2 env-mismatch). Strict additivity: already-rendering spans byte-untouched.

Fake oracle (no browser): deterministic renders_clean stubs.
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from scientific_parser.postproc.text_pipeline import repair_unrenderable_math


def _oracle_renders(bodies):
    """Oracle stub: exactly the given body strings render."""
    allowed = set(bodies)
    return lambda body: body in allowed


def test_truncated_brace_repaired():
    # '\lambda^{' + '}' renders (fake oracle), original does not
    md = "prefix $\\lambda^{$ suffix"
    oracle = _oracle_renders(["\\lambda^{}"])
    out = repair_unrenderable_math(md, renders_clean=oracle)
    assert "$\\lambda^{}$" in out


def test_already_rendering_span_untouched():
    md = "a $x^{2}$ b $\\lambda^{$ c"
    oracle = _oracle_renders(["x^{2}", "\\lambda^{}"])
    out = repair_unrenderable_math(md, renders_clean=oracle)
    assert "$x^{2}$" in out
    assert "$\\lambda^{}$" in out


def test_no_variant_adopts_body_unchanged():
    md = "keep $\\sum_{ garbage"  # nothing renders
    oracle = _oracle_renders([])
    out = repair_unrenderable_math(md, renders_clean=oracle)
    assert out == md


def test_env_mismatch_appends_end():
    md = "$$\\begin{cases} a & b \\\\$$"
    oracle = _oracle_renders(["\\begin{cases} a & b \\\\\\end{cases}"])
    out = repair_unrenderable_math(md, renders_clean=oracle)
    assert "\\end{cases}" in out


def test_display_inline_no_overlap_corruption():
    # inline span inside display span must not corrupt outer indices
    md = "$$a $b$ c$$ tail $\\lambda^{$ end"
    oracle = _oracle_renders(["a $b$ c", "\\lambda^{}"])
    out = repair_unrenderable_math(md, renders_clean=oracle)
    assert "$$a $b$ c$$" in out  # outer already renders -> untouched
    assert "$\\lambda^{}$" in out  # inner later span repaired


def test_budget_bounds_oracle_calls():
    # 3 suspicious spans, budget 2 -> only 2 attempted (3rd stays)
    md = "1 $a^{$ 2 $b^{$ 3 $c^{$ 4"
    calls = []

    def oracle(body):
        calls.append(body)
        return False

    import scientific_parser.postproc.text_pipeline as tp

    orig = tp._MAX_REPAIR_SPANS_PER_DOC
    tp._MAX_REPAIR_SPANS_PER_DOC = 2
    try:
        out = repair_unrenderable_math(md, renders_clean=oracle)
    finally:
        tp._MAX_REPAIR_SPANS_PER_DOC = orig
    # oracle fires per body AND per variant; only span-attempts count here
    bodies = ("a^{", "b^{", "c^{")
    attempted = [c for c in calls if c in bodies]
    assert len(attempted) == 2 and "a^{" not in attempted
    assert "$c^{$" in out
