"""Minimal-token LaTeX repair (reviewer #21).

The prior aggressive equation repair rewrote whole formulas and caused
benchmark regressions (it dropped valid math blocks — see skill_adapter.py).
It was disabled. Reviewer #21 asks for the opposite policy:

    extract -> parse -> render -> compare -> repair ONLY the minimum
                                      incorrect token.

Never rewrite an entire formula just because it fails. This module applies
single-token substitutions from a curated table of known OCR confusions, and
ONLY when an external render check says the input fails. If no single-token
substitution makes the render pass, the original is returned unchanged — we do
not guess.

The confusion table is seeded with the documented glm-ocr error
(``\\sort`` -> ``\\sqrt``, see pdf_ocr.py DPI comment) and a few conservative
high-confidence LaTeX-token confusions. It is extensible: add entries as new
systematic OCR errors are observed in benchmarks.
"""

from __future__ import annotations

from collections.abc import Callable
import logging

log = logging.getLogger(__name__)

# A render predicate: returns True if the LaTeX renders/parses OK.
RenderCheck = Callable[[str], bool]


# Curated single-token OCR confusions (left = OCR output, right = intended).
# Each is applied independently as a MINIMAL single-token substitution.
_TOKEN_CONFUSIONS: dict[str, str] = {
    # documented: glm-ocr misread \sqrt as \sort at low DPI (pdf_ocr.py:707)
    r"\sort": r"\sqrt",
    # other common LaTeX-token OCR confusions (high-confidence, conservative)
    r"\frace": r"\frac",
    r"\frca": r"\frac",
    r"\summ": r"\sum",
    r"\intt": r"\int",
    r"\alpha2": r"\alpha",
    r"\labda": r"\lambda",
    r"\lamda": r"\lambda",
    r"\thta": r"\theta",
    r"\Sigme": r"\Sigma",
    r"\Omegs": r"\Omega",
    r"\subs": r"\subset",
    r"\sups": r"\supset",
    r"\rigt": r"\right",
    r"\lef": r"\left",
}


def minimal_repair(latex: str, render_ok: RenderCheck) -> str:
    """Repair a LaTeX string by trying single-token substitutions.

    Policy (reviewer #21):
      1. If ``render_ok(latex)`` is True, return ``latex`` unchanged (nothing
         to repair — never touch a passing formula).
      2. Otherwise, try each token confusion as a SINGLE substitution. As soon
         as one makes ``render_ok`` return True, return that candidate.
      3. If no single-token fix works, return the ORIGINAL unchanged. We do
         not attempt multi-token rewrites (that is what caused the regression).

    Args:
      latex: the LaTeX string to repair (without surrounding ``$``).
      render_ok: a predicate that returns True iff the LaTeX renders/parses.
        Callers typically wrap pykatex / the typed formula_latex validator.

    Returns:
      The repaired LaTeX (single-token change at most), or the original.
    """
    if not latex:
        return latex
    try:
        if render_ok(latex):
            return latex
    except Exception as e:
        # A render predicate that raises is treated as "does not render".
        log.debug("minimal_repair: Exception degraded: %s", e)

    for bad, good in _TOKEN_CONFUSIONS.items():
        if bad not in latex:
            continue
        candidate = latex.replace(bad, good, 1)  # exactly one token
        try:
            if render_ok(candidate):
                return candidate
        except Exception:
            continue
    return latex  # no single-token fix -> leave unchanged (no wholesale rewrite)


def available_confusions() -> dict[str, str]:
    """Return a copy of the token-confusion table (for introspection/tests)."""
    return dict(_TOKEN_CONFUSIONS)


def add_confusion(ocr_token: str, intended_token: str) -> None:
    """Register a new observed OCR confusion at runtime (extensibility)."""
    # Validate both look like LaTeX commands (backslash-prefixed) — guard
    # against accidental whole-formula registration.
    if not ocr_token.startswith("\\") or not intended_token.startswith("\\"):
        raise ValueError(
            f"confusion tokens must be LaTeX commands (backslash-prefixed): "
            f"got {ocr_token!r} -> {intended_token!r}"
        )
    _TOKEN_CONFUSIONS[ocr_token] = intended_token


__all__ = ["minimal_repair", "available_confusions", "add_confusion"]
