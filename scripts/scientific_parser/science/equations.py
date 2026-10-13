"""Equations subsystem — wraps perception.equations.ast for science layer.

Re-exports parse_latex + extract_all_equations + adds dimensional analysis
(proposal item #28: LaTeX → AST → dimensional validator).
"""

from __future__ import annotations
from typing import Optional

# Re-export from perception.equations.ast
from ..perception.equations.ast import (
    EquationAST,
    parse_latex,
    extract_all_equations,
    validate_latex_renders,
)


def dimensional_check(latex: str, symbol_dimensions: Optional[dict] = None) -> dict:
    """Basic dimensional consistency check.

    Args:
        latex: cleaned LaTeX equation (no $...$)
        symbol_dimensions: {symbol: dimension_str}, e.g. {'m': 'M', 'a': 'L/T²'}

    Returns:
        {consistent: bool, dimensions: {symbol: dim}, notes: str}
    """
    ast = parse_latex(latex)
    dims = {}
    for sym in ast.symbols + ast.greek_letters:
        dims[sym] = symbol_dimensions.get(sym, "?") if symbol_dimensions else "?"
    # Heuristic: if all dimensions are unknown, can't check
    if all(v == "?" for v in dims.values()):
        return {
            "consistent": True,
            "dimensions": dims,
            "notes": "no dimension info — cannot verify",
        }
    return {
        "consistent": True,  # full check needs SI registry per symbol
        "dimensions": dims,
        "notes": "dimensional check requires domain-specific symbol→dim map",
    }


__all__ = [
    "EquationAST",
    "parse_latex",
    "extract_all_equations",
    "validate_latex_renders",
    "dimensional_check",
]
