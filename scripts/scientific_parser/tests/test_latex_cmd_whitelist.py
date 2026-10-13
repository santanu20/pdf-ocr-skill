r"""Pins: postprocessor must not corrupt valid LaTeX (2026-09-09 flip-class).

Two measured bench regressions from run 4b's flip-diff (+41/-3 arxiv):
  - 05407: _fix_glued_commands bisected \leftrightarrow -> \left rightarrow
    (longest-prefix split finds `left` before the full valid command).
  - 05062/05140: blanket bare-& escaping wrote \& inside \begin{cases},
    destroying the alignment separator the gold expects.

Every whitelist token here was verified KaTeX-renderable via pykatex
(strict=error) before inclusion. The glued-variable repair itself
(\toCH -> \to CH) must keep working — that is the function's purpose.
"""

from scientific_parser.postproc.text_pipeline import (
    _fix_glued_commands,
    fix_latex_math_output,
)


class TestGluedCommandSplit:
    def test_arrow_family_never_split(self):
        for cmd in (
            r"\leftrightarrow",
            r"\rightarrow",
            r"\leftarrow",
            r"\rightleftharpoons",
            r"\rightarrowtail",
        ):
            assert _fix_glued_commands(cmd + " 2") == cmd + " 2"

    def test_sizing_delimiters_never_split(self):
        for cmd in (r"\bigl", r"\bigr", r"\biggl", r"\Bigl", r"\Biggr"):
            assert _fix_glued_commands(cmd + "(x") == cmd + "(x"

    def test_prefixed_valid_cmds_never_split(self):
        for cmd in (r"\top", r"\injlim", r"\intercal", r"\sinh", r"\cosh", r"\bigstar"):
            assert _fix_glued_commands(cmd + " x") == cmd + " x"

    def test_glued_variable_still_split(self):
        assert _fix_glued_commands(r"\toCH") == r"\to CH"
        assert _fix_glued_commands(r"\fracx") == r"\frac x"


class TestAmpersandEscapeGate:
    def test_cases_env_keeps_bare_alignment_ampersand(self):
        out = fix_latex_math_output(r"$$\begin{cases} a , & b \\ c , & d \end{cases}$$")
        assert "& b" in out and "& d" in out
        assert r"\& b" not in out

    def test_matrix_env_keeps_bare_ampersand(self):
        out = fix_latex_math_output(r"$$\begin{pmatrix} a & b \\ c & d \end{pmatrix}$$")
        assert "a & b" in out

    def test_bare_ampersand_outside_env_still_escaped(self):
        out = fix_latex_math_output(r"$a & b$")
        assert r"\&" in out and "a & b" not in out


class TestFixLatexMathOutputArrows:
    def test_leftrightarrow_chain_preserved(self):
        src = r"$1   \leftrightarrow   2   \leftrightarrow   3 ,$ and $R(Q,d)$."
        out = fix_latex_math_output(src)
        assert r"\leftrightarrow   2   \leftrightarrow   3" in out
        assert "left rightarrow" not in out
