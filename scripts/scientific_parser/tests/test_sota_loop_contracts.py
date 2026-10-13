"""SOTA-loop cumulative contracts (2026-08-16).

Binding protocol (user directive, 2026-08-16): every fix gets a pin
(fixture, invariant) BEFORE the fix; all historical pins + new pin + this
suite must pass; PASS→FAIL in the per-test flip diff = automatic reject.
These unit contracts mirror the page-level pins in bench/pins.py and pin
the *mechanisms* so regressions are caught without PDF extraction.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scientific_parser.postproc.text_pipeline import _fix_glued_commands  # noqa: E402
from scientific_parser.science.font_math_detector import (  # noqa: E402
    _build_line_with_math,
    _math_continues,
    _span_to_math_token,
)


def _sp(text, font, size=10.0, x=50, y=100):
    """Synthetic span: x advances by text width heuristics."""
    w = max(len(text) * 5, 4)
    span = {
        "text": text,
        "font": font,
        "size": size,
        "origin": (x, y),
        "bbox": (x, y - 10, x + w, y + 2),
    }
    return span, x + w


def _line(spans, body=10.0, baseline=100.0):
    return _build_line_with_math(spans, body, baseline)


class TestGluedCommandSplitter:
    """iter-3: whole-word aware — never bisect a valid longer command."""

    def test_valid_commands_survive(self):
        for cmd in (
            r"\bigsqcup x",
            r"\supsetneq",
            r"\supset \mathfrak{g}",
            r"\int_0^1",
            r"\to X",
            r"\bigodot",
            r"\bigvee",
        ):
            assert _fix_glued_commands(cmd) == cmd, cmd

    def test_ocr_glue_splits(self):
        assert _fix_glued_commands(r"\toCH") == r"\to CH"
        assert _fix_glued_commands(r"\leqn") == r"\leq n"


class TestCMEXLetterSlots:
    """iter-1 (+2026-08-16 X-slot restoration): broken-ToUnicode cmex10
    letter slots decode to their big operators."""

    def test_sum_slot(self):
        assert _span_to_math_token("P", False, False, "CMEX10").strip() == r"\sum"

    def test_bigsqcup_slot(self):
        assert _span_to_math_token("X", False, False, "cmex10").strip() == r"\bigsqcup"

    def test_control_parity_delimiters(self):
        assert _span_to_math_token("\x00", False, False, "CMEX10") == "("
        assert _span_to_math_token("\x01", False, False, "CMEX10") == ")"


class TestPhiTwins:
    """iter-5: math-italic fonts render BOTH phi codepoints curly →
    \\varphi (gold 3.5:1); straight \\phi only outside math-italic."""

    def test_cmmi_both_codepoints_varphi(self):
        for ch in ("φ", "ϕ"):
            out = _span_to_math_token(ch, False, False, "CMMI10")
            assert out.startswith(r"\varphi"), (ch, out)

    def test_roman_phi_stays_phi(self):
        out = _span_to_math_token("ϕ", False, False, "LMRoman10")
        assert out.strip() == r"\phi", out


class TestLigatureIntake:
    """iter-5: TeX ligature glyphs normalize BEFORE word matching —
    'iﬀ' (ﬀ U+FB00) must reach the \\text{iff} path."""

    def test_iff_ligature_wraps(self):
        spans = [
            _sp("⊢", "CMSY10")[0],
            _sp("φ", "CMMI10")[0],
            _sp(" iﬀ ", "CMR10")[0],
            _sp("M", "CMMI10")[0],
        ]
        out = _line(spans)
        assert r"\text{iff}" in out, out


class TestMathContinuesGate:
    """iter-5: prose connectives absorb only when math resumes."""

    def test_math_follows(self):
        items = [
            {"text": " and ", "font": "CMR10"},
            {"text": "x", "font": "CMMI10"},
        ]
        for it in items:
            it.setdefault("unicode_math", False)
        assert _math_continues(items, 0, 2) is True

    def test_prose_follows(self):
        items = [
            {"text": " and ", "font": "CMR10"},
            {"text": "all results hold", "font": "CMR10"},
        ]
        for it in items:
            it.setdefault("unicode_math", False)
        assert _math_continues(items, 0, 2) is False


class TestFencePlacement:
    """iter-4/5: fences land INSIDE $...$ around math-italic vars;
    sentence periods stay outside WITHOUT unclosing the math."""

    def test_fences_inside(self):
        spans = [
            _sp("(", "LMRoman10-Regular")[0],
            _sp("v", "LMMathItalic10-Regular")[0],
            _sp(", x", "LMMathItalic10-Regular")[0],
            _sp("1", "LMRoman7-Regular", 7.0)[0],
            _sp(")", "LMRoman10-Regular")[0],
        ]
        out = _line(spans)
        assert out.startswith("$(") and out.endswith(")$"), out

    def test_prose_paren_digit_stays_out(self):
        spans = [
            _sp("equation ", "LMRoman10-Regular")[0],
            _sp("(1)", "LMRoman10-Regular")[0],
            _sp(" shows", "LMRoman10-Regular")[0],
        ]
        out = _line(spans)
        assert "$" not in out, out

    def test_closing_fence_period_keeps_fence(self):
        """').' span: fence INSIDE $...$, period OUTSIDE — the exact
        layer-interaction regression that unclosed '$G ( m, n$).'."""
        spans = [
            _sp("G", "LMMathItalic10-Regular")[0],
            _sp(" ( m, n", "LMMathItalic10-Regular")[0],
            _sp(").", "LMRoman10-Regular")[0],
        ]
        out = _line(spans)
        assert r"$G ( m, n )$." == out.replace("  ", " ").strip(), repr(out)


class TestEllipsisAndMinus:
    """iter-3: spaced-dot runs → \\ldots; U+2212 → ASCII '-' inside math."""

    def test_ldots(self):
        spans = [_sp("x", "CMMI10")[0], _sp(", . . . , y", "CMR10")[0]]
        out = _line(spans)
        assert r"\ldots" in out, out

    def test_minus(self):
        spans = [_sp("a − b", "CMMI10")[0]]
        out = _line(spans)
        assert "−" not in out and "- " in out, out


class TestOcrPathWarningsBound:
    """iter-11: model-failure degrade must never UnboundLocalError-crash.

    Root cause (2026-08-16): paddleocr-vl 500'd mid-bench → degrade handler
    appended to `warnings` before OCR-path entry bound it → 4/188 table PDFs
    emitted NO output. Invariant: warnings bound at OCR-path entry, and the
    summary-stage rebuild preserves model_failure_* notes.
    """

    def test_warnings_bound_at_ocr_entry(self):
        src = (Path(__file__).resolve().parents[2] / "pdf_ocr.py").read_text()
        # Binding happens at OCR-path entry (before the degrade handler).
        entry = src.index("# OCR path")
        handler = src.index("model_failure_degraded_rapidocr")
        assert src.index("warnings: list = []", entry) < handler, (
            "warnings must be bound before the degrade handler can append"
        )

    def test_summary_rebuild_preserves_degrade_note(self):
        src = (Path(__file__).resolve().parents[2] / "pdf_ocr.py").read_text()
        assert '_early = [w for w in warnings if w.startswith("model_failure_")]' in src, (
            "summary-stage rebuild must not drop model_failure_* warnings"
        )
