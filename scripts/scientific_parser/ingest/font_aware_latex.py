r"""Font-aware LaTeX extraction for born-digital math PDFs.

Problem: arxiv_math PDFs have CMMI/CMSY/MSBM Computer Modern fonts.
The text layer loses LaTeX structure — "R" could be a variable or \mathbb{R}.
Font name tells us the truth:
  MSBM → \mathbb{X} (blackboard bold)
  CMMI → math italic variable
  CMSY → math symbol
  CMR  → regular text

This module walks spans, groups math-font spans into $...$ blocks,
and emits proper LaTeX.
"""

from __future__ import annotations

import re
from typing import Optional


def _font_to_math_role(font_name: str) -> str:
    """Classify font into math role."""
    f = font_name.upper()
    if "MSBM" in f or "MSAM" in f:
        return "blackboard"  # \mathbb{}
    if "CMMI" in f:
        return "math_italic"  # variable (x, y, z, α, β, γ)
    if "CMSY" in f:
        return "math_symbol"  # operators, symbols
    if "CMEX" in f:
        return "math_extension"  # large operators (∫, ∑)
    if "EU1" in f or "EU2" in f:
        return "unicode_math"  # Unicode math font
    if "FRAK" in f or "EUF" in f:
        return "fraktur"  # \mathfrak{}
    if "B" in f and "CMTI" not in f and "CMBX" in f:
        return "bold"
    return "text"


def _char_to_latex(text: str, role: str) -> str:
    """Convert a span's text to LaTeX based on its font role."""
    if not text:
        return text
    t = text.strip()
    if not t:
        return text

    if role == "blackboard":
        # Each uppercase letter → \mathbb{X}
        result = []
        for c in t:
            if c.isalpha():
                result.append(f"\\mathbb{{{c}}}")
            else:
                result.append(c)
        return "".join(result)

    if role == "fraktur":
        result = []
        for c in t:
            if c.isalpha():
                result.append(f"\\mathfrak{{{c}}}")
            else:
                result.append(c)
        return "".join(result)

    # For math_italic / math_symbol / math_extension / unicode_math:
    # Convert Unicode chars to LaTeX
    _UNICODE_MAP = {
        "α": r"\alpha",
        "β": r"\beta",
        "γ": r"\gamma",
        "δ": r"\delta",
        "ε": r"\epsilon",
        "ζ": r"\zeta",
        "η": r"\eta",
        "θ": r"\theta",
        "ι": r"\iota",
        "κ": r"\kappa",
        "λ": r"\lambda",
        "μ": r"\mu",
        "ν": r"\nu",
        "ξ": r"\xi",
        "π": r"\pi",
        "ρ": r"\rho",
        "σ": r"\sigma",
        "τ": r"\tau",
        "υ": r"\upsilon",
        "φ": r"\phi",
        "χ": r"\chi",
        "ψ": r"\psi",
        "ω": r"\omega",
        "Γ": r"\Gamma",
        "Δ": r"\Delta",
        "Θ": r"\Theta",
        "Λ": r"\Lambda",
        "Ξ": r"\Xi",
        "Π": r"\Pi",
        "Σ": r"\Sigma",
        "Φ": r"\Phi",
        "Ψ": r"\Psi",
        "Ω": r"\Omega",
        "∫": r"\int",
        "∑": r"\sum",
        "∏": r"\prod",
        "√": r"\sqrt",
        "×": r"\times",
        "·": r"\cdot",
        "÷": r"\div",
        "≤": r"\leq",
        "≥": r"\geq",
        "≠": r"\neq",
        "≈": r"\approx",
        "∞": r"\infty",
        "∂": r"\partial",
        "∇": r"\nabla",
        "→": r"\to",
        "←": r"\leftarrow",
        "↔": r"\leftrightarrow",
        "⇒": r"\Rightarrow",
        "⇔": r"\Leftrightarrow",
        "∈": r"\in",
        "∉": r"\notin",
        "⊂": r"\subset",
        "⊆": r"\subseteq",
        "∪": r"\cup",
        "∩": r"\cap",
        "∅": r"\emptyset",
        "∀": r"\forall",
        "∃": r"\exists",
        "±": r"\pm",
        "∓": r"\mp",
        "∘": r"\circ",
        "⋯": r"\cdots",
        "…": r"\ldots",
        "⋮": r"\vdots",
        "⋱": r"\ddots",
        "°": r"^{\circ}",
    }
    out = t
    for u, latex in sorted(_UNICODE_MAP.items(), key=lambda x: -len(x[0])):
        out = out.replace(u, latex)
    return out


def extract_latex_from_page(page) -> str:
    """Extract text from a PyMuPDF page with font-aware LaTeX conversion.

    Groups adjacent math-font spans into $...$ blocks. Regular text stays
    outside math mode. Short text spans (1-3 chars: ':', '2', '(', etc.)
    between math spans are absorbed into the math region.
    """
    try:
        blocks = page.get_text("dict", flags=page.parent.UNICODE_)
    except Exception:
        blocks = page.get_text("dict")

    # Phase 1: collect all spans with their roles
    all_spans = []
    for block in blocks.get("blocks", []):
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font = span.get("font", "")
                text = span.get("text", "")
                role = _font_to_math_role(font)
                all_spans.append({"text": text, "role": role, "font": font})
            # Line break marker
            all_spans.append({"text": "\n", "role": "newline", "font": ""})

    # Phase 2: absorb short text spans between math spans into math
    for i, span in enumerate(all_spans):
        if span["role"] == "text" and len(span["text"].strip()) <= 3:
            # Check if neighbors are math
            prev_math = (
                i > 0
                and all_spans[i - 1]["role"] != "text"
                and all_spans[i - 1]["role"] != "newline"
            )
            next_math = (
                i < len(all_spans) - 1
                and all_spans[i + 1]["role"] != "text"
                and all_spans[i + 1]["role"] != "newline"
            )
            if prev_math and next_math:
                span["role"] = "absorbed_text"

    # Phase 3: build output
    output_parts = []
    math_buffer = []

    def _flush_math():
        nonlocal math_buffer
        if math_buffer:
            latex = " ".join(math_buffer).strip()
            # Clean up multiple spaces
            latex = re.sub(r"\s+", " ", latex).strip()
            if latex:
                output_parts.append(f"${latex}$")
            math_buffer = []

    for span in all_spans:
        role = span["role"]
        text = span["text"]

        if role == "newline":
            _flush_math()
            output_parts.append("\n")
        elif role == "text":
            _flush_math()
            output_parts.append(text)
        else:
            # Math span (including absorbed_text)
            latex = _char_to_latex(text, role if role != "absorbed_text" else "text")
            math_buffer.append(latex)

    _flush_math()
    return "".join(output_parts)


__all__ = ["extract_latex_from_page", "_font_to_math_role", "_char_to_latex"]
