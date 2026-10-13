r"""Render→verify→repair reflexive loop (proposal items #5, #10, #15).

Per arXiv 2603.22651 (March 2026): "Reflexive self-correcting loop wins
multi-agent benchmarks." Pattern: extract → verify → retry with feedback.

Pipeline:
  1. extract → LaTeX equation
  2. verify via pykatex (renders or ParseError)
  3. if fails: extract error → patch candidate → re-verify
  4. up to N retries; if still fails → mark as REVIEW/REJECT

Patch candidates come from error message (e.g. "Undefined control sequence:
\sort" → suggest \sqrt).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


# Error-message → fix mapping (auto-repair dictionary)
# Grows as we see new failure modes
_ERROR_FIX_MAP = {
    # "Undefined control sequence: \sort" → replace with \sqrt
    r"Undefined control sequence: \\sort": (r"\\sort", r"\\sqrt"),
    r"Undefined control sequence: \\sqr": (r"\\sqr\b", r"\\sqrt"),
    r"Undefined control sequence: \\fracc": (r"\\fracc", r"\\frac"),
    r"Undefined control sequence: \\intt": (r"\\intt", r"\\int"),
    r"Undefined control sequence: \\alp": (r"\\alp\b", r"\\alpha"),
    r"Undefined control sequence: \\bet": (r"\\bet\b", r"\\beta"),
    r"Undefined control sequence: \\lam": (r"\\lam\b", r"\\lambda"),
    r"Undefined control sequence: \\textb": (r"\\textb", r"\\textbf"),
}


@dataclass
class RepairResult:
    """One repair cycle result."""

    success: bool
    attempts: int
    original: str
    repaired: str
    final_error: str = ""
    patches_applied: list = field(default_factory=list)


def verify_with_pykatex(latex: str) -> tuple[bool, str]:
    """Verify LaTeX renders. Returns (ok, error_msg)."""
    try:
        import pykatex
    except ImportError:
        return True, ""  # skip
    body = re.sub(r"^\${1,2}|\${1,2}$", "", latex.strip())
    try:
        pykatex.renderToString(body, strict="error")
        return True, ""
    except Exception as e:
        return False, str(e)


def _apply_error_fix(latex: str, error: str) -> tuple[str, list]:
    """Try matching error against fix dictionary. Returns (repaired, patches)."""
    patches = []
    out = latex
    for err_pat, (bad_pat, replacement) in _ERROR_FIX_MAP.items():
        if re.search(err_pat, error):
            new = re.sub(bad_pat, replacement, out)
            if new != out:
                patches.append(f"{bad_pat!r} → {replacement!r}")
                out = new
    return out, patches


def repair_latex(
    latex: str,
    max_attempts: int = 3,
    retry_ocr_fn: Optional[Callable[[str, str], str]] = None,
    image_path: Optional[str] = None,
) -> RepairResult:
    """Reflexive verifier: extract → verify → patch → re-verify.

    Args:
        latex: initial LaTeX output from OCR
        max_attempts: max retry cycles
        retry_ocr_fn: optional callback to re-OCR with error feedback.
            Signature: (image_path, error_msg) → new_latex
        image_path: passed to retry_ocr_fn if provided

    Returns:
        RepairResult with final state
    """
    result = RepairResult(
        success=False,
        attempts=0,
        original=latex,
        repaired=latex,
    )

    current = latex
    for attempt in range(1, max_attempts + 1):
        result.attempts = attempt
        ok, err = verify_with_pykatex(current)
        if ok:
            result.success = True
            result.repaired = current
            result.final_error = ""
            return result

        # Try error-message-driven patch
        patched, patches = _apply_error_fix(current, err)
        if patched != current:
            result.patches_applied.extend(patches)
            current = patched
            continue

        # Try OCR retry if callback provided
        if retry_ocr_fn is not None and image_path is not None:
            feedback = (
                f"Your previous LaTeX output failed to render with error: "
                f"{err}. Please re-transcribe and ensure valid KaTeX syntax."
            )
            new_latex = retry_ocr_fn(image_path, feedback)
            if new_latex and new_latex != current:
                current = new_latex
                continue

        # No more fixes available
        result.repaired = current
        result.final_error = err
        return result

    # Used all attempts
    result.repaired = current
    return result


def repair_all_equations(
    text: str, max_attempts: int = 3
) -> tuple[str, list[RepairResult]]:
    """Scan text for $$...$$ and $...$ blocks, verify + repair each.

    Returns (repaired_text, list_of_repair_results).
    """
    if not text or "$" not in text:
        return text, []

    results: list[RepairResult] = []

    def _repair_inline(m):
        body = m.group(1)
        r = repair_latex(body, max_attempts=max_attempts)
        results.append(r)
        return (
            f"${r.repaired}$"
            if r.success
            else f"<!-- unrepairable: {r.final_error[:60]} -->"
        )

    def _repair_display(m):
        body = m.group(1)
        r = repair_latex(body, max_attempts=max_attempts)
        results.append(r)
        return (
            f"$${r.repaired}$$"
            if r.success
            else f"<!-- unrepairable: {r.final_error[:60]} -->"
        )

    # Display first
    text = re.sub(r"\$\$(.+?)\$\$", _repair_display, text, flags=re.DOTALL)
    # Then inline
    text = re.sub(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", _repair_inline, text)
    return text, results


__all__ = [
    "RepairResult",
    "verify_with_pykatex",
    "repair_latex",
    "repair_all_equations",
]
