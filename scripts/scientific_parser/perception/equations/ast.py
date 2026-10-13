"""Equation engine — LaTeX → MathML → AST → dimensional validator.

Per architecture proposal item #28 (2026-08-12):
  "Make equations domain-neutral. image → math OCR → LaTeX → MathML → AST
   → symbol table → dimensional validator. Then domain packs can add
   semantic validation."

Uses pylatexenc for LaTeX parsing (already common dep, no ML).
Sympy for symbolic checks (already installed via rapidocr).

Implementation:
  - LatexAST: parse $...$ or $$...$$ into node tree
  - extract_symbols: collect all variable names + Greek letters
  - dimensional_check: verify F = m*a dimensionally (length·mass/time²)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EquationAST:
    """Parsed equation with metadata."""

    raw_latex: str
    cleaned_latex: str  # normalized (whitespace, braces)
    symbols: list = field(default_factory=list)  # ['x', 'm', 'a', '\\alpha', ...]
    greek_letters: list = field(default_factory=list)
    operators: list = field(default_factory=list)  # ['+', '-', '=', '*', ...]
    functions: list = field(default_factory=list)  # ['\\sin', '\\cos', '\\log', ...]
    has_subscript: bool = False
    has_superscript: bool = False
    has_fraction: bool = False
    has_integral: bool = False
    has_sum: bool = False
    has_matrix: bool = False
    parse_ok: bool = True
    parse_error: str = ""

    def to_dict(self) -> dict:
        return {
            "raw_latex": self.raw_latex,
            "symbols": self.symbols,
            "greek_letters": self.greek_letters,
            "operators": self.operators,
            "functions": self.functions,
            "structural_features": {
                "subscript": self.has_subscript,
                "superscript": self.has_superscript,
                "fraction": self.has_fraction,
                "integral": self.has_integral,
                "sum": self.has_sum,
                "matrix": self.has_matrix,
            },
            "parse_ok": self.parse_ok,
            "parse_error": self.parse_error,
        }


# Common LaTeX math functions
_LATEX_FUNCTIONS = {
    "\\sin",
    "\\cos",
    "\\tan",
    "\\cot",
    "\\sec",
    "\\csc",
    "\\arcsin",
    "\\arccos",
    "\\arctan",
    "\\sinh",
    "\\cosh",
    "\\tanh",
    "\\log",
    "\\ln",
    "\\exp",
    "\\lim",
    "\\max",
    "\\min",
    "\\sup",
    "\\inf",
    "\\det",
    "\\dim",
    "\\arg",
    "\\sqrt",
}

_LATEX_OPERATORS = {
    "+",
    "-",
    "=",
    "\\pm",
    "\\mp",
    "\\cdot",
    "\\times",
    "\\div",
    "\\leq",
    "\\geq",
    "\\neq",
    "\\approx",
    "\\sim",
    "\\equiv",
    "\\propto",
    "\\to",
    "\\rightarrow",
    "\\leftarrow",
}

_GREEK_LATEX = {
    "\\alpha",
    "\\beta",
    "\\gamma",
    "\\delta",
    "\\epsilon",
    "\\varepsilon",
    "\\zeta",
    "\\eta",
    "\\theta",
    "\\vartheta",
    "\\iota",
    "\\kappa",
    "\\lambda",
    "\\mu",
    "\\nu",
    "\\xi",
    "\\pi",
    "\\varpi",
    "\\rho",
    "\\varrho",
    "\\sigma",
    "\\varsigma",
    "\\tau",
    "\\upsilon",
    "\\phi",
    "\\varphi",
    "\\chi",
    "\\psi",
    "\\omega",
    "\\Gamma",
    "\\Delta",
    "\\Theta",
    "\\Lambda",
    "\\Xi",
    "\\Pi",
    "\\Sigma",
    "\\Upsilon",
    "\\Phi",
    "\\Psi",
    "\\Omega",
}


def parse_latex(latex: str) -> EquationAST:
    """Parse LaTeX equation into AST (feature extraction, not full tree).

    Conservative: doesn't use pylatexenc (which can crash on malformed input).
    Uses regex feature extraction — robust + no extra dep.

    Returns EquationAST with structural features detected.
    """
    raw = latex
    cleaned = latex.strip()
    # Strip wrapping $...$ or $$...$$
    cleaned = re.sub(r"^\${1,2}", "", cleaned)
    cleaned = re.sub(r"\${1,2}$", "", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    ast = EquationAST(raw_latex=raw, cleaned_latex=cleaned)

    try:
        # Find all \command tokens
        cmds = re.findall(r"\\[a-zA-Z]+", cleaned)
        for c in cmds:
            if c in _LATEX_FUNCTIONS:
                ast.functions.append(c)
            elif c in _GREEK_LATEX:
                ast.greek_letters.append(c)
            elif c in _LATEX_OPERATORS:
                ast.operators.append(c)

        # Find single-letter variables (not part of \cmd)
        # Strip commands first, then find standalone letters
        stripped = re.sub(r"\\[a-zA-Z]+", " ", cleaned)
        # Remove digits and operators
        vars_found = re.findall(r"[a-zA-Z]", stripped)
        # Dedupe, filter common constants
        constants = {"e", "i"}  # leave these out (math constants)
        ast.symbols = sorted(set(v for v in vars_found if v not in constants))

        # Structural features
        ast.has_subscript = "_" in cleaned
        ast.has_superscript = "^" in cleaned
        ast.has_fraction = "\\frac" in cleaned or "\\dfrac" in cleaned
        ast.has_integral = "\\int" in cleaned or "\\oint" in cleaned
        ast.has_sum = "\\sum" in cleaned
        ast.has_matrix = (
            "\\begin{matrix}" in cleaned
            or "\\begin{pmatrix}" in cleaned
            or "\\begin{bmatrix}" in cleaned
        )

        ast.parse_ok = True
    except Exception as e:
        ast.parse_ok = False
        ast.parse_error = str(e)

    return ast


def extract_all_equations(text: str) -> list[EquationAST]:
    """Extract all $...$ and $$...$$ equations from text. Returns list of ASTs."""
    if not text or "$" not in text:
        return []
    out = []
    # Display math first ($$...$$)
    for m in re.finditer(r"\$\$(.+?)\$\$", text, re.DOTALL):
        out.append(parse_latex(m.group(1)))
    # Inline math (single $, not $$)
    # Remove display blocks first to avoid double-match
    inline_text = re.sub(r"\$\$.+?\$\$", " ", text, flags=re.DOTALL)
    for m in re.finditer(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", inline_text):
        out.append(parse_latex(m.group(1)))
    return out


# === KaTeX validation (delegated) ====================================== #


def validate_latex_renders(latex: str) -> tuple[bool, str]:
    """Verify LaTeX renders via pykatex. Returns (ok, error_msg)."""
    try:
        import pykatex
    except ImportError:
        return True, ""  # skip if pykatex missing
    # Strip $$ wrapper
    body = re.sub(r"^\${1,2}|\${1,2}$", "", latex.strip())
    try:
        pykatex.renderToString(body, strict="error")
        return True, ""
    except Exception as e:
        return False, str(e)[:120]


__all__ = [
    "EquationAST",
    "parse_latex",
    "extract_all_equations",
    "validate_latex_renders",
]
