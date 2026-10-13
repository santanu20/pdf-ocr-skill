"""Selective prediction — ACCEPT / REVIEW / REJECT output states.

Per architecture proposal item #25 (2026-08-12):
  "The parser should be allowed to say 'I don't know.' Three states:
   ACCEPT / REVIEW / REJECT — instead of 'always output something'.
   For scientific literature, abstention is a feature.
   A wrong scientific number is worse than a missing number."

Design:
  - ConfidenceAggregator: per-page / per-region confidence metric
  - Triager: maps confidence → {ACCEPT, REVIEW, REJECT}
  - Markdown output gets a top-level triage banner per page
  - .triage.json sidecar emitted with structured triage info

Confidence signals (weighted blend):
  - text_density (chars/page — born-digital high, scanned variable)
  - validator_warning_count (NumericGuard + pykatex + others)
  - ocr_repetition_ratio (Phase 0 of _collapse_repetition)
  - cjk_hallucination_present (strip_cjk_hallucination marker)
  - math_block_failure_rate (pykatex unrepairable count)
  - repetition_marker_count (Phase 0-3 markers)

Triage thresholds (tunable, conservative defaults):
  ACCEPT:  confidence >= 0.85  AND  zero REJECT-level findings
  REVIEW:  0.50 <= confidence < 0.85  OR  any REVIEW-level finding
  REJECT:  confidence < 0.50  OR  any REJECT-level finding
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# Triage states
ACCEPT = "ACCEPT"
REVIEW = "REVIEW"
REJECT = "REJECT"


@dataclass
class TriageResult:
    """Per-page (or per-document) triage verdict."""

    state: str  # ACCEPT / REVIEW / REJECT
    confidence: float  # 0.0 - 1.0
    signals: dict = field(default_factory=dict)
    findings: list = field(default_factory=list)
    recommendation: str = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "confidence": round(self.confidence, 3),
            "signals": {
                k: round(v, 3) if isinstance(v, float) else v for k, v in self.signals.items()
            },
            "findings": self.findings[:20],  # cap for compactness
            "recommendation": self.recommendation,
        }


# Signal extractors — each returns a float [0,1] where 1 = good
def _signal_text_density(text: str) -> float:
    """Higher density of alphanumeric chars = better signal."""
    if not text:
        return 0.0
    alnum = sum(c.isalnum() for c in text)
    return min(1.0, alnum / max(len(text), 1) * 1.5)


def _signal_validator_warnings(text: str) -> float:
    """Fewer validator warning markers = better. Returns 0-1 (1 = clean)."""
    if not text:
        return 0.0
    # Count warning markers we emit
    markers = re.findall(
        r"<!--\s*(NumericGuard|KaTeX unrepairable|OCR CJK hallucination"
        r"|OCR repetition collapsed|OCR loop)",
        text,
    )
    # Normalize: 0 markers → 1.0, 5+ markers → 0.0
    return max(0.0, 1.0 - len(markers) / 5.0)


def _signal_ocr_repetition(text: str) -> float:
    """Detect heavy OCR repetition (Phase 0-3 markers + actual repetition)."""
    if not text:
        return 0.0
    markers = re.findall(r"\[OCR (?:repetition collapsed|loop)", text)
    # Also detect: 5+ identical lines, 5+ identical adjacent chars
    lines = text.split("\n")
    identical_line_runs = 0
    for i in range(1, len(lines)):
        if lines[i].strip() and lines[i].strip() == lines[i - 1].strip():
            identical_line_runs += 1
    # Long runs of same character (10+)
    long_runs = len(re.findall(r"(.)\1{9,}", text))
    # N-gram duplication (2026-08-16): VL models degenerate into repeated
    # sentence templates ("It has a fleet of 300 ships. It has a navy of
    # ...") that identical-line checks miss. Distinct-8gram ratio: normal
    # prose 94-100% distinct; degenerate output < 88% (measured on
    # long_tiny 17_pg2 chandra output: ~72% distinct).
    words = re.findall(r"\w{2,}", text.lower())
    ngram_pen = 0
    if len(words) >= 200:
        n = 8
        total = len(words) - n + 1
        distinct = len({" ".join(words[i : i + n]) for i in range(total)})
        dup_ratio = 1.0 - distinct / total
        if dup_ratio > 0.06:  # normal text ≈ 0-6% dup
            ngram_pen = int(round((dup_ratio - 0.06) * 100))
    bad_count = len(markers) + identical_line_runs + long_runs + ngram_pen
    return max(0.0, 1.0 - bad_count / 10.0)


def _signal_math_quality(text: str) -> float:
    """Detect LaTeX math blocks; check for unrepairable markers."""
    if not text:
        return 0.0
    display_blocks = re.findall(r"\$\$.+?\$\$", text, re.DOTALL)
    inline_blocks = re.findall(r"(?<!\$)\$([^$\n]{1,200})\$(?!\$)", text)
    total = len(display_blocks) + len(inline_blocks)
    if total == 0:
        return 1.0  # no math = no math risk
    unrepairable = len(re.findall(r"KaTeX unrepairable", text))
    return max(0.0, 1.0 - unrepairable / max(total, 1))


def _signal_cjk_hallucination(text: str) -> float:
    """Detect CJK hallucination stripping marker."""
    if not text:
        return 0.0
    if "CJK hallucination stripped" in text:
        return 0.2  # heavy penalty
    return 1.0


# Weights for blending (sum to 1.0)
_WEIGHTS = {
    "text_density": 0.20,
    "validator_warnings": 0.25,
    "ocr_repetition": 0.25,
    "math_quality": 0.15,
    "cjk_hallucination": 0.15,
}


def triage(text: str, page_num: Optional[int] = None) -> TriageResult:
    """Score page text and return triage verdict.

    Args:
        text: page markdown output
        page_num: optional page number for finding context

    Returns:
        TriageResult with state, confidence, signals, findings
    """
    signals = {
        "text_density": _signal_text_density(text),
        "validator_warnings": _signal_validator_warnings(text),
        "ocr_repetition": _signal_ocr_repetition(text),
        "math_quality": _signal_math_quality(text),
        "cjk_hallucination": _signal_cjk_hallucination(text),
    }
    confidence = sum(signals[k] * w for k, w in _WEIGHTS.items())

    # Collect findings (REJECT-level issues)
    findings = []
    if signals["cjk_hallucination"] < 0.5:
        findings.append({"level": "REJECT", "type": "CJK hallucination", "page": page_num})
    if signals["ocr_repetition"] < 0.3:
        findings.append({"level": "REJECT", "type": "OCR repetition loop", "page": page_num})
    if signals["validator_warnings"] < 0.4:
        findings.append(
            {"level": "REVIEW", "type": "Multiple validator warnings", "page": page_num}
        )
    if signals["math_quality"] < 0.5:
        findings.append({"level": "REVIEW", "type": "Math blocks unrepairable", "page": page_num})

    # Decide state
    has_reject = any(f["level"] == "REJECT" for f in findings)
    has_review = any(f["level"] == "REVIEW" for f in findings)
    if has_reject or confidence < 0.50:
        state = REJECT
    elif has_review or confidence < 0.85:
        state = REVIEW
    else:
        state = ACCEPT

    recommendation = {
        ACCEPT: "Output is high-confidence. Safe to use.",
        REVIEW: "Output has some quality issues. Manual review suggested.",
        REJECT: "Output has critical quality issues. Do not use without re-OCR.",
    }[state]

    return TriageResult(
        state=state,
        confidence=confidence,
        signals=signals,
        findings=findings,
        recommendation=recommendation,
    )


def annotate_with_triage(text: str, page_num: Optional[int] = None) -> str:
    """Prepend triage banner to page text. Returns annotated text."""
    result = triage(text, page_num=page_num)
    banner = (
        f"<!-- TRIAGE: state={result.state} confidence={result.confidence:.2f} "
        f"— {result.recommendation} "
        f"signals={{{', '.join(f'{k}={v:.2f}' for k, v in result.signals.items())}}} -->\n"
    )
    return banner + text


__all__ = [
    "ACCEPT",
    "REVIEW",
    "REJECT",
    "TriageResult",
    "triage",
    "annotate_with_triage",
]
