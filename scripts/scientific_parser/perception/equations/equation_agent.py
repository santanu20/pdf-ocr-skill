"""EquationAgent — deterministic equation-extraction state machine (no LLM).

Mirrors the TableAgent pattern: born-digital-first, model-last.

For born-digital PDFs, equations live in the text layer as Unicode math chars
(∫∑√αβ). Converting those to LaTeX is deterministic + EXACT — no OCR error.
95% of arxiv_math pages have such Unicode math (measured). RapidLaTeX (the CNN)
is reserved for the genuine tail: scanned equations, image/outlines, or text
layers that don't carry the math symbols.

Stages:
  1. EXTRACT native — PyMuPDF text-layer clip of the equation region.
  2. CONVERT — ``convert_unicode_to_latex`` (∫→\\int, α→\\alpha, …).
  3. VALIDATE — ``is_real_math`` (not garbage) + ``validate_latex`` (parseable).
  4. FALLBACK — RapidLaTeX CNN on the rendered image crop when native fails.

Replaces the model-first equation branch in ``route_page_by_region``.
"""

import re
from dataclasses import dataclass, field
import logging
from scientific_parser.core.path_warnings import (
    EQUATION_MODEL_FALLBACK,
    record_path_warning,
)

log = logging.getLogger(__name__)


@dataclass
class EquationResult:
    """Structured equation result."""

    latex: str
    confidence: float
    method: str  # "unicode-native" | "rapidlatex" | "native-low"
    source_text: str = ""
    needs_review: bool = False

    def to_dict(self):
        return {
            "method": self.method,
            "confidence": self.confidence,
            "needs_review": self.needs_review,
        }


class EquationAgent:
    """Deterministic equation agent: native Unicode→LaTeX → validate → model.

    Construct once per page; ``process(bbox)`` runs all stages for one equation.
    """

    # Confidence floor for accepting native LaTeX. Native comes from the exact
    # text layer so a passing validator is trusted fully (no model blend).
    ACCEPT_NATIVE = 0.70
    REVIEW = 0.60

    def __init__(
        self,
        pdf_path,
        page_num,
        render_dpi=150.0,
        img_path=None,
        formula_provider=None,
    ):
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.render_dpi = render_dpi
        self.img_path = img_path
        self.formula = formula_provider

    def process(self, bbox):
        """Native Unicode→LaTeX → validate → RapidLaTeX fallback. Per equation."""
        # Attempt 1: native text-layer Unicode → LaTeX (deterministic, exact).
        result = self._try_native(bbox)
        if result is not None and result.confidence >= self.ACCEPT_NATIVE:
            return result


        record_path_warning(
            EQUATION_MODEL_FALLBACK,
            "native Unicode→LaTeX below acceptance — falling back to RapidLaTeX CNN",
        )
        # Attempt 2: RapidLaTeX CNN (image OCR — scanned / image / no-symbols).
        result_rt = self._try_model(bbox)
        if result_rt is not None and result_rt.confidence >= 0.40:
            return result_rt

        # Attempt 3: keep the better of native / model even if both weak.
        if result is not None and (result_rt is None or result.confidence >= result_rt.confidence):
            return result
        return result_rt

    def _try_native(self, bbox):
        """Extract text-layer Unicode math → convert → validate."""
        text = _extract_region_text(self.pdf_path, self.page_num, bbox, self.render_dpi)
        if not text or len(text.strip()) < 2:
            return None
        try:
            from scientific_parser.science.unicode_to_latex import convert_unicode_to_latex
            from scientific_parser.perception.equations.math_extractor import is_real_math
        except Exception:
            return None
        latex = convert_unicode_to_latex(text).strip()
        if not latex or not is_real_math(latex):
            return None
        # Validate parseability (delimiter balance + parse/render).
        conf = 0.90
        try:
            from scientific_parser.verification.typed import validate_latex

            vr = validate_latex(latex)
            conf = 0.40 + 0.50 * vr.score  # 0.40 (exists) .. 0.90 (fully valid)
        except Exception as e:
            log.debug("_try_native: Exception degraded: %s", e)
        return EquationResult(
            latex=_wrap_latex(latex),
            confidence=round(conf, 3),
            method="unicode-native",
            source_text=text[:80],
            needs_review=conf < self.REVIEW,
        )

    def _try_model(self, bbox):
        """RapidLaTeX CNN on the cropped equation image (fallback)."""
        if not (self.formula and self.formula.available() and self.img_path):
            return None
        crop = None
        try:
            crop = _crop_image(self.img_path, bbox)
            latex, conf = self.formula.recognize(crop)
            if not (latex and conf > 0.3):
                return None
            return EquationResult(
                latex=_wrap_latex(latex),
                confidence=round(float(conf), 3),
                method="rapidlatex",
                needs_review=conf < self.REVIEW,
            )
        except Exception:
            return None
        finally:
            # M13 (audit-2, 2026-08-18): _crop_image's NamedTemporaryFile
            # leaked EVERY crop png (56 measured in /tmp) — every acquire
            # gets its release (§9 L1). Guard crop != img_path: the
            # degenerate-bbox branch returns the SOURCE image.
            if crop and crop != self.img_path:
                try:
                    import os

                    os.unlink(crop)
                except OSError:
                    pass


# ── Helpers (self-contained — no import from region_dispatcher) ─────────── #


def _extract_region_text(pdf_path, page_num, bbox, render_dpi):
    """PyMuPDF text-layer clip for a bbox (render-px → PDF-pt clip)."""
    try:
        import fitz

        scale = 72.0 / render_dpi  # render-px → PDF-pt
        x0, y0, x1, y1 = bbox
        clip = fitz.Rect(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
        with fitz.open(pdf_path) as doc:
            return doc[page_num - 1].get_text("text", clip=clip) or ""
    except Exception:
        return ""


def _wrap_latex(latex):
    """Ensure display equations are $$-delimited for downstream rendering."""
    s = latex.strip()
    if s.startswith("$$") or s.startswith("$"):
        return s
    return f"$${s}$$"


def _crop_image(img_path, bbox):
    """Crop ``img_path`` to ``bbox`` (render px) → temp PNG path."""
    import os
    import tempfile

    from PIL import Image

    x0, y0, x1, y1 = (int(v) for v in bbox)
    img = Image.open(img_path)
    w, h = img.size
    crop = img.crop((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
    out = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    out.close()
    crop.save(out.name)
    return out.name


__all__ = ["EquationAgent", "EquationResult"]
