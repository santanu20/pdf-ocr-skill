"""Span conservation contract — verify and repair cascade assembly.

The font-math cascade assembles clusters by span classification with no
verification: tokens (parens, evaluation bars, frac denominators) vanish
silently, leaving triple-space gaps and unbalanced ``\\frac{X}{(}``
KaTeX-breakers. Full-set census 2026-08-27: 66% of arxiv_math fails are
span-gap shattering, 10% unbalanced delimiters. 115 classifier patches
moved +0.5 each; this module inverts the architecture — emit-and-VERIFY:

  inventory (cluster spans, glyph-decoded) → accounting (per-span
  presence via greedy x-order subsequence match) → repair (geometric
  re-injection at the nearest present left-neighbor; brace balance
  lint) → report (conservation rate per page, repair events).

Deterministic: pure text-layer, no model calls. Repair touches ONLY
lines that fail the audit (iter-59 lesson: never recompose winners).
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

# Repair is env-gated during bring-up (deterministic-contract bench runs
# flip it explicitly; default ON once full-set census proves net-positive).
SPAN_REPAIR_ENV = "PDF_OCR_SPAN_REPAIR"
_SPAN_REPAIR_DEFAULT = os.environ.get(SPAN_REPAIR_ENV, "1") not in ("0", "")


def span_repair_enabled() -> bool:
    return os.environ.get(SPAN_REPAIR_ENV, "1" if _SPAN_REPAIR_DEFAULT else "0") not in (
        "0",
        "",
    )


@dataclass
class ClusterAudit:
    """Audit outcome for one emitted line against its cluster spans."""

    line: str
    n_spans: int
    missing_spans: list[dict] = field(default_factory=list)
    char_deficit: int = 0
    braces_balanced: bool = True
    repaired: bool = False

    @property
    def conservation(self) -> float:
        if self.n_spans == 0:
            return 1.0
        return 1.0 - len(self.missing_spans) / self.n_spans


_ALNUM_RE = re.compile(r"[a-zA-Z0-9]")


def _alnum(text: str) -> str:
    return "".join(_ALNUM_RE.findall(text))


def _decoded_token(span: dict) -> str:
    """Decode a span through the same glyph-truth decoder the builder uses.

    Importing lazily keeps this module importable without the detector
    (unit tests) while guaranteeing alias-consistency in production: the
    accounting compares what the builder SHOULD have emitted, not raw
    span text.
    """
    from scientific_parser.science.font_math_detector import _span_to_math_token

    tok = _span_to_math_token(span.get("text", ""), False, False, span.get("font", ""))
    return tok or span.get("text", "")


def _is_math_font_span(span: dict) -> bool:
    """True when the span comes from a strict math font (CMMI/CMEX/CMSY...)."""
    from scientific_parser.science.font_math_detector import _STRICT_MATH_FONT_RE

    return bool(_STRICT_MATH_FONT_RE.search(span.get("font", "")))


def audit_line(spans: list[dict], line: str) -> ClusterAudit:
    """Account every cluster span against the emitted line.

    Greedy x-order subsequence match: spans are visited left-to-right; a
    span is PRESENT when its decoded alnum sequence appears (in order)
    in the not-yet-consumed portion of the emitted alnum string. X-order
    matches the builder's emission order for body-size tokens; script
    attachment keeps script chars adjacent to their base, which the
    greedy consume also tolerates.
    """
    audit = ClusterAudit(line=line, n_spans=len(spans))
    # NORMALIZE builder-side collapses into the haystack so spans the
    # builder legitimately merged (· · · → \cdots, iter-28) count as
    # present — otherwise the audit re-injects them as duplicates and
    # un-collapses every \cdots run (06597 pg18: 9 corrupted lines).
    hay = _alnum(
        re.sub(r"(\\cdot(?:\s+\\cdot){2,})", r"\\cdots", line)
    )
    pos = 0
    for span in sorted(spans, key=lambda s: s["bbox"][0]):
        # PROSE SPANS OUT OF SCOPE (05218 pg5 regression): roman/bold
        # text spans have no single ground-truth emission form — the
        # builder may legitimately emit the decoded token, the raw
        # text, or both (bold \bm{} + prose fallback). Re-injecting a
        # second representation corrupts prose lines (\bm{Lemma}
        # Lemma...). Conservation is a MATH-assembly invariant: audit
        # strict-math-font spans only (CMMI/CMEX/CMSY/MSBM...), whose
        # emission the glyph-truth decoder fully determines.
        if not _is_math_font_span(span):
            continue
        tok = _decoded_token(span)
        # Collapse-side needle normalization (06597): a '· · ·' span
        # decodes to 3 \cdot commands, but the builder collapses every
        # 3-run to ONE \cdots — normalize the needle the same way so
        # the collapsed emission accounts for the span.
        tok = re.sub(r"(\\cdot(?:\s+\\cdot){2,})", r"\\cdots", tok)
        needle = _alnum(tok)
        if not needle:
            # Pure-punctuation span (parens, bars, commas): its decoded
            # form carries no alnum. Account by raw-char presence with
            # multiplicity against the line.
            audit.char_deficit += max(
                0,
                sum(1 for c in tok if c not in " " and line.count(c) == 0),
            )
            continue
        idx = hay.find(needle, pos)
        if idx < 0:
            # Script-attached reorder tolerance: retry from the start
            # once (builder may have emitted this span's chars inside a
            # base's script group before pos).
            idx = hay.find(needle)
        if idx < 0:
            audit.missing_spans.append(span)
        else:
            pos = max(pos, idx + len(needle))
    audit.char_deficit += sum(len(_alnum(_decoded_token(s))) for s in audit.missing_spans)
    audit.braces_balanced = line.count("{") == line.count("}")
    return audit


def _insert_after_neighbor(line: str, missing: dict, spans: list[dict], body_size: float) -> str:
    """Re-inject one missing span's decoded token at its geometric slot.

    Insertion point = right after the last PRESENT left-neighbor's
    matched text (x-order), keeping the token inside the same math run
    (before a closing ``$`` when the neighbor sits at line end).
    Script-size spans attach as sub/sup relative to body size.
    """
    tok = _decoded_token(missing).strip()
    if not tok:
        return line
    size = missing.get("size", body_size) or body_size
    dy = missing.get("origin", (0, 0))[1] if missing.get("origin") else None
    # Script wrap: small span with known origin below/above its left
    # neighbor's origin (dy sign) becomes _{}/^{} — conservative: only
    # when clearly script-size.
    left = [s for s in spans if s["bbox"][0] < missing["bbox"][0] and s is not missing]
    ref = left[-1] if left else None
    if size < 0.75 * body_size and ref is not None and dy is not None:
        rdy = ref.get("origin", (0, 0))[1] if ref.get("origin") else None
        if rdy is not None and dy > rdy + 0.5:
            tok = "_{" + tok + "}"
        elif rdy is not None and dy < rdy - 0.5:
            tok = "^{" + tok + "}"
        else:
            tok = " " + tok
    else:
        tok = " " + tok
    if ref is None:
        return tok.lstrip() + " " + line if line else tok.strip()
    anchor = _alnum(_decoded_token(ref))
    if not anchor:
        return line + tok
    idx = line.rfind(anchor) if anchor else -1
    if idx < 0:
        return line + tok
    # End of the anchor occurrence (alnum run end).
    end = idx + len(anchor)
    while end < len(line) and _ALNUM_RE.match(line[end]):
        end += 1
    # Keep injection inside the math region: never past a closing $.
    close = line.find("$", end)
    ins = end if close < 0 else min(end, close)
    return line[:ins] + tok + line[ins:]


def _balance_braces(line: str) -> str:
    """Append missing ``}`` so KaTeX can render (extra content is free
    for MathML-containment scoring; unbalanced braces hard-fail render).

    Escaped literals (``\\{`` / ``\\}``) are brace CHARACTERS, not group
    delimiters — KaTeX needs no balance for them, and a math set opened
    on a wrapped line legitimately closes on the next line. Counting
    them (05506 pg11) appends a bogus ``}`` after the region-closing
    ``$``, splitting wrapped equations the reflow would have joined.
    """
    stripped = re.sub(r"\\[{}]", "", line)
    depth = 0
    for ch in stripped:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
    if depth > 0:
        return line + "}" * depth
    return line


def repair_line(audit: ClusterAudit, spans: list[dict], body_size: float) -> str:
    """Repair one audited line: re-inject missing spans, balance braces."""
    line = audit.line
    for missing in sorted(audit.missing_spans, key=lambda s: s["bbox"][0]):
        line = _insert_after_neighbor(line, missing, spans, body_size)
    line = _balance_braces(line)
    return line


def conservation_pass(
    clusters: list[dict], emitted: list[tuple[float, str]]
) -> tuple[list[tuple[float, str]], dict]:
    """Audit+repair a cascade emission list.

    clusters: collect_math_clusters() output (span inventory, scoped by
        construction — exactly what the builder consumed).
    emitted: list of (y0, line) pairs from extract_text_with_cascade.

    Returns (possibly repaired) emitted list and a stats dict. Repair
    fires only on lines failing the audit; clean lines are untouched.
    """
    stats = {
        "lines": len(emitted),
        "audited": 0,
        "repaired": 0,
        "missing_total": 0,
        "low_conservation_lines": 0,
    }
    if not span_repair_enabled():
        return emitted, stats
    # Map emitted lines back to clusters by y0 proximity (cascade sorts
    # by y0; region-inserted equations have no cluster → skip).
    from scientific_parser.science.font_math_detector import _line_body_and_baseline

    cluster_ys = [cl["bbox"][1] for cl in clusters]
    out: list[tuple[float, str]] = []
    used_clusters: set[int] = set()
    for y0, line in emitted:
        best_ci, best_d = None, None
        for ci, cy in enumerate(cluster_ys):
            if ci in used_clusters:
                continue
            d = abs(cy - y0)
            if best_d is None or d < best_d:
                best_ci, best_d = ci, d
        if best_ci is None or (best_d is not None and best_d > 5.0):
            out.append((y0, line))
            continue
        used_clusters.add(best_ci)
        cl = clusters[best_ci]
        if cl.get("env_latex") is not None:
            # Prebuilt environment: span accounting does not apply.
            out.append((y0, line))
            continue
        audit = audit_line(cl["spans"], line)
        stats["audited"] += 1
        stats["missing_total"] += len(audit.missing_spans)
        if audit.conservation < 0.999:
            stats["low_conservation_lines"] += 1
        if audit.missing_spans or not audit.braces_balanced:
            spans_sorted = sorted(cl["spans"], key=lambda s: s["bbox"][0])
            body_size, _ = _line_body_and_baseline(spans_sorted)
            new_line = repair_line(audit, spans_sorted, body_size)
            if new_line != line:
                stats["repaired"] += 1
                line = new_line
        out.append((y0, line))
    return out, stats


def page_conservation_rate(clusters: list[dict], lines: list[str]) -> float:
    """Fraction of cluster spans accounted for in the emitted lines.

    Telemetry-only metric (Wave-B escalation gate + missing-region
    census: unaccounted math-font spans OUTSIDE any cluster are not
    visible here — the caller reports those separately).
    """
    total = missing = 0
    for cl, line in zip(clusters, lines, strict=False):
        if cl.get("env_latex") is not None:
            continue
        a = audit_line(cl["spans"], line)
        total += a.n_spans
        missing += len(a.missing_spans)
    return 1.0 if total == 0 else 1.0 - missing / total


__all__ = [
    "audit_line",
    "conservation_pass",
    "page_conservation_rate",
    "repair_line",
    "span_repair_enabled",
    "ClusterAudit",
]
