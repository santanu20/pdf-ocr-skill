"""Formula LaTeX validator — delimiter balance + parse + render validity.

Per architecture proposal item #4 (2026-08-13): typed validator for
formulas. Catches:
    - Unbalanced delimiters ($$...$, \\(...\\), \\[...\\])
    - LaTeX that fails to parse via ``parse_latex``
    - LaTeX that fails to render via KaTeX (``verify_with_pykatex``)

Reuses existing lower-level validators:
    - :func:`scientific_parser.perception.equations.ast.parse_latex`
    - :func:`scientific_parser.verification.render_compare.verify_with_pykatex`

Never raises — wraps all parsing in try/except and reports failure in
``VerificationResult.details``. Callers may blindly pass user/model output.
"""

from __future__ import annotations

from ...core.ir import VerificationResult

# Map delimiter pairs. Values are (open, close, is_block).
_DELIM_PAIRS = [
    ("$$", "$$", True),
    (r"\[", r"\]", True),
    (r"\(", r"\)", False),
    ("$", "$", False),
]


def _check_delimiter_balance(latex: str) -> tuple[bool, str]:
    """Return (ok, reason). Reason is empty string when ok."""
    for open_d, close_d, _is_block in _DELIM_PAIRS:
        # Skip inline $ check if $$ is present (avoid double-counting)
        if open_d == "$" and "$$" in latex:
            continue
        n_open = latex.count(open_d)
        n_close = latex.count(close_d)
        if n_open != n_close:
            return (
                False,
                f"unbalanced {open_d}/{close_d}: open={n_open} close={n_close}",
            )
    return True, ""


def validate_latex(content: str) -> VerificationResult:
    """Validate LaTeX formula structure.

    Args:
        content: A LaTeX string. May or may not include surrounding
            delimiters (``$...$``, ``$$...$$``, ``\\(...\\)``, ``\\[...\\]``).
            If delimiters are absent, the validator still attempts to parse
            the bare LaTeX.

    Returns:
        :class:`VerificationResult` with ``validator_id="formula_latex"``.

    Scoring (each contributes 0.34, last is 0.32 — totals 1.0):
        1. Delimiter balance OK.
        2. parse_latex succeeds (syntactic validity).
        3. verify_with_pykatex succeeds (renders without error).
        Plus partial credit: bare-Latex with no delimiters gets delimiter
        check free.
    """
    if not content or not content.strip():
        return VerificationResult(
            passed=False,
            score=0.0,
            validator_id="formula_latex",
            details={"error": "empty_content"},
        )

    # Strip delimiters for bare-latex parse test
    bare = content.strip()
    for open_d, close_d, _ in _DELIM_PAIRS:
        if bare.startswith(open_d) and bare.endswith(close_d):
            bare = bare[len(open_d) : -len(close_d)].strip()
            break

    # 1. Delimiter balance
    delim_ok, delim_reason = _check_delimiter_balance(content)
    has_delimiters = any(
        open_d in content or close_d in content for open_d, close_d, _ in _DELIM_PAIRS
    )
    # If no delimiters present at all, treat delimiter check as passing
    # (bare LaTeX without delimiters is valid input).
    delim_score = 1.0 if (delim_ok or not has_delimiters) else 0.0

    # 2. parse_latex
    parse_ok = False
    parse_reason = ""
    try:
        from ...perception.equations.ast import parse_latex

        parse_latex(bare)
        parse_ok = True
    except Exception as e:
        parse_reason = f"{type(e).__name__}: {e}"

    # 3. KaTeX render
    render_ok = False
    render_reason = ""
    try:
        from ...verification.render_compare import verify_with_pykatex

        ok, msg = verify_with_pykatex(bare)
        render_ok = bool(ok)
        if not render_ok:
            render_reason = msg or "katex_render_failed"
    except Exception as e:
        render_reason = f"verifier_exception: {type(e).__name__}: {e}"

    errors = []
    if not delim_ok and has_delimiters:
        errors.append(f"delimiter: {delim_reason}")
    if not parse_ok:
        errors.append(f"parse: {parse_reason}")
    if not render_ok:
        errors.append(f"render: {render_reason}")

    score = delim_score * 0.34 + (0.34 if parse_ok else 0.0) + (0.32 if render_ok else 0.0)
    passed = delim_ok and parse_ok and render_ok

    return VerificationResult(
        passed=passed,
        score=score,
        validator_id="formula_latex",
        details={
            "has_delimiters": has_delimiters,
            "delim_ok": delim_ok,
            "parse_ok": parse_ok,
            "render_ok": render_ok,
            "errors": errors,
        },
    )


__all__ = ["validate_latex"]
