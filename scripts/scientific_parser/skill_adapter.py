"""Skill adapter — bridges pdf-ocr monolith and scientific_parser package.

Per user requirement: "maintain lib and skill both. Should work both as
standalone library and coding agent skill."

Two integration levels:
  - LIGHT: monolith emits DocumentIR + .ledger.jsonl + .triage.json sidecars
           alongside existing markdown output (zero behavior change)
  - FULL:  monolith delegates entity extraction, domain detection, validators
           to the package (faster + richer, but changes some output structure)

Default = LIGHT (backward-compat). Enable FULL via --use-package flag (TODO).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import TYPE_CHECKING
import logging

log = logging.getLogger(__name__)

if TYPE_CHECKING:
    from scientific_parser import DocumentIR
    from scientific_parser.core.ir import VerificationResult

# Make scientific_parser importable from skill context
if "scripts" not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _evidence_for_source(source: str | None, model: str | None) -> tuple[str, str]:
    """Map a page's real source to (evidence_level, source_label).

    Fixes the 2026-08-13 provenance bug where any non-empty page text was
    labelled ``E0_native_pdf_text`` / ``pdftotext`` regardless of whether it
    actually came from OCR.

    Mapping:
      - pdftotext / native text layer   -> E0_native_pdf_text
      - rapidocr (PP-OCRv6 CNN)         -> E3_classical_ocr
      - glm-ocr / *vl* / *vision*       -> E5_vlm
      - any other OCR engine           -> E4_specialist_ocr
    """
    from scientific_parser import (
        E0_NATIVE_PDF_TEXT,
        E3_CLASSICAL_OCR,
        E4_SPECIALIST_OCR,
        E5_VLM,
    )

    src = (source or "ocr").lower()
    if src in ("pdftotext", "native", "text_layer", "text"):
        return E0_NATIVE_PDF_TEXT, "pdftotext"
    # Source is OCR but the specific engine is unknown — never guess a VLM;
    # report generic specialist OCR (E4) with an explicit "ocr-unknown" label.
    if model is None:
        return E4_SPECIALIST_OCR, "ocr-unknown"
    mdl = model.lower()
    if "rapidocr" in mdl:
        return E3_CLASSICAL_OCR, mdl
    if mdl == "glm-ocr" or "vl" in mdl or "vision" in mdl:
        return E5_VLM, mdl
    return E4_SPECIALIST_OCR, mdl


def _aggregate_page_verification(
    page_text: str, block_type_hint: str = "text"
) -> VerificationResult | None:
    """Run typed validators on a page's text and aggregate the result.

    Per architecture proposal item #1 (2026-08-13): EvidenceResolver needs
    per-candidate verification signal to pick verified specialists over
    unverified native text. This function runs the typed validators that
    match the page's content type and aggregates their scores into a single
    page-level :class:`VerificationResult`.

    Heuristic content-type detection:
      - `$$ ... $$` or `\\(...\\)` or `\\[...\\]` present → run formula validator
      - `` <table> `` or 2+ `|---|` markdown rows → run table validator
      - Numeric strings present (e.g. ``12.5 wt%``) → run numeric validator

    Args:
        page_text: The page's markdown text.
        block_type_hint: Optional type hint ("text"/"equation"/"table").

    Returns:
        A :class:`VerificationResult` with ``validator_id="page_aggregate"``
        and score = mean of the per-validator scores. None if no validator
        fired (e.g. empty page or pure prose).
    """
    if not page_text or not page_text.strip():
        return None

    from scientific_parser.core.ir import VerificationResult

    scores: list[tuple[str, float, bool]] = []

    # Formula detection: math delimiters OR Greek/math symbols OR LaTeX commands
    has_formula = (
        "$$" in page_text
        or r"\(" in page_text
        or r"\[" in page_text
        or any(c in page_text for c in "αβγδεζηθικλμνξοπρστυφχψω∑∏∫√")
        # LaTeX command signal: \alpha, \beta, \sum, \int, \frac, etc.
        or re.search(r"\\[a-zA-Z]{3,}\b", page_text) is not None
    )
    if has_formula or block_type_hint == "equation":
        try:
            from scientific_parser.verification.typed import validate_latex

            # Extract the first formula span (don't validate whole page as LaTeX)
            matches = re.findall(
                r"\$\$[^$]+\$\$|\$[^$\n]+\$|\\\([^)]+\\\)|\\\[[^\]]+\\\]",
                page_text,
            )
            if matches:
                # Validate the longest match (most informative)
                target = max(matches, key=len)
                vr = validate_latex(target)
                scores.append(("formula_latex", vr.score, vr.passed))
        except Exception as e:
            log.debug("_aggregate_page_verification: Exception degraded: %s", e)

    # Table detection: HTML <table> or markdown table structure
    has_table_html = "<table" in page_text.lower()
    has_table_md = page_text.count("|---|") >= 1 or page_text.count("|:--") >= 1
    if has_table_html or has_table_md or block_type_hint == "table":
        try:
            from scientific_parser.verification.typed import validate_table

            # Extract the table span (rough — between first <table>/| and last </table>/|
            if has_table_html:
                m = re.search(r"<table>.*?</table>", page_text, re.IGNORECASE | re.DOTALL)
                target = m.group(0) if m else page_text
            else:
                # Markdown: lines that look like table rows
                lines = page_text.splitlines()
                table_lines = [
                    ln for ln in lines if ln.strip().startswith("|") and ln.strip().endswith("|")
                ]
                target = "\n".join(table_lines[:20])  # cap at 20 rows
            if target.strip():
                vr = validate_table(target)
                scores.append(("table_structure", vr.score, vr.passed))
        except Exception as e:
            log.debug("_aggregate_page_verification: Exception degraded: %s", e)

    # Numeric detection: number + unit pattern
    if re.search(
        r"\d+(?:\.\d+)?\s*(?:wt%|ppm|ppb|Ma|ka|Ga|°C|degC|kbar|MPa|GPa|km|m|kg|mol)", page_text
    ):
        try:
            from scientific_parser.verification.typed import validate_numeric

            # Pick the first plausible numeric token for the validator
            m = re.search(
                r"(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(wt%|ppm|ppb|Ma|ka|Ga|°C|degC|kbar|MPa|GPa|km|m|kg|mol)",
                page_text,
            )
            if m:
                vr = validate_numeric(m.group(1), expected_unit=m.group(2))
                scores.append(("numeric_range", vr.score, vr.passed))
        except Exception as e:
            log.debug("_aggregate_page_verification: Exception degraded: %s", e)

    if not scores:
        return None

    # Aggregate: mean score, passed iff ALL passed
    avg_score = sum(s for _, s, _ in scores) / len(scores)
    all_passed = all(p for _, _, p in scores)
    return VerificationResult(
        passed=all_passed,
        score=avg_score,
        validator_id="page_aggregate",
        details={
            "n_validators": len(scores),
            "validators": [
                {"validator_id": vid, "score": sc, "passed": ps} for vid, sc, ps in scores
            ],
        },
    )


def build_document_ir(
    source_pdf: str, pages_text: list, pages_meta: list | None = None
) -> DocumentIR:
    """Build a DocumentIR from the monolith's per-page text output.

    Args:
        source_pdf: path to PDF
        pages_text: list of strings (one per page)
        pages_meta: optional list of dicts with page_num/blocks/bbox info
    """
    from scientific_parser import (
        DocumentIR,
        Evidence,
        PageIR,
        make_block,
    )

    doc = DocumentIR(source_pdf=source_pdf)

    # Per-page real source lookup from pages_meta (1-indexed "page" key).
    # Fixes the provenance bug where any non-empty text was labelled E0
    # native PDF text regardless of whether it came from OCR. When the
    # real source is unknown we default to E4 (specialist OCR) — never
    # claim native provenance we cannot prove.
    source_by_page: dict[int, tuple[str, str]] = {}
    if pages_meta:
        for m in pages_meta:
            try:
                pno = int(m.get("page", 0))
                if pno > 0:
                    source_by_page[pno] = _evidence_for_source(
                        m.get("source", "ocr"), m.get("model")
                    )
            except (TypeError, ValueError):
                continue

    for i, text in enumerate(pages_text, start=1):
        page = PageIR(page_num=i)
        if text:
            level, src_label = source_by_page.get(i, _evidence_for_source("ocr", None))
            # Run typed validators on the page text. When verification data
            # is available, attach it to the Evidence so EvidenceResolver
            # can pick verified candidates over unverified ones.
            verification = _aggregate_page_verification(text)
            block = make_block(
                page_num=i,
                block_idx=0,
                block_type="text",
                bbox=(0, 0, 0, 0),
                text=text,
                level=level,
                source=src_label,
            )
            if verification is not None:
                # Replace the default Evidence with one carrying verification.
                block.evidence[0] = Evidence(
                    level=level,
                    source=src_label,
                    value=text,
                    confidence=block.evidence[0].confidence,
                    verification=verification,
                )
            page.add_block(block)
        doc.add_page(page)
    return doc


def emit_sidecars(
    markdown_text: str,
    output_md_path: str,
    source_pdf: str | None = None,
    pages_text: list | None = None,
    pages_meta: list | None = None,
    extra_metadata: dict | None = None,
    candidates: list | None = None,
) -> dict:
    """Emit DocumentIR JSON + triage banner + ledger sidecars next to .md output.

    Called from monolith's main() AFTER text is finalized. Adds zero-cost
    sidecars — markdown output unchanged.

    Returns dict with paths to emitted files.
    """
    out_path = Path(output_md_path)
    stem = out_path.stem
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    sidecars = {}

    # 1. Triage banner (already done in monolith — re-evaluate to be sure)
    try:
        from scientific_parser import triage

        result = triage(markdown_text)
        triage_path = out_dir / f"{stem}.triage.json"
        triage_path.write_text(json.dumps(result.to_dict(), indent=2))
        sidecars["triage"] = str(triage_path)
    except Exception as e:
        log.warning("emit_sidecars: Exception degraded: %s", e)

    # 2. Entity extraction
    try:
        from scientific_parser import extract_all

        entities = extract_all(markdown_text)
        ent_path = out_dir / f"{stem}.entities.json"
        ent_path.write_text(
            json.dumps(
                [
                    {
                        "id": e.id,
                        "type": e.type,
                        "value": e.value,
                        "normalized": e.normalized,
                        "confidence": e.confidence,
                        "metadata": e.metadata,
                    }
                    for e in entities
                ],
                indent=2,
                ensure_ascii=False,
            )
        )
        sidecars["entities"] = str(ent_path)
    except Exception as e:
        log.warning("emit_sidecars: Exception degraded: %s", e)

    # 3. Domain detection
    try:
        from scientific_parser import detect_domains

        domains = detect_domains(markdown_text)
        dom_path = out_dir / f"{stem}.domains.json"
        dom_path.write_text(json.dumps(domains, indent=2))
        sidecars["domains"] = str(dom_path)
    except Exception as e:
        log.warning("emit_sidecars: Exception degraded: %s", e)

    # 4. Document IR (if pages_text provided)
    if pages_text:
        try:
            doc = build_document_ir(source_pdf or "", pages_text, pages_meta)
            doc.metadata.update(extra_metadata or {})
            ir_path = out_dir / f"{stem}.ir.json"
            ir_path.write_text(json.dumps(doc.to_dict(), indent=2, ensure_ascii=False, default=str))
            sidecars["document_ir"] = str(ir_path)
        except Exception as e:
            log.warning("emit_sidecars: Exception degraded: %s", e)

        # Cross-page entity stitching (reviewer #28): link numbered labels
        # (Table N continued, Eq. N back-refs, Figure captions) across pages.
        try:
            from scientific_parser.core.cross_page import (
                build_cross_page_registry,
            )

            registry = build_cross_page_registry(list(pages_text))
            xp_path = out_dir / f"{stem}.cross_page.json"
            xp_path.write_text(
                json.dumps(
                    [e.to_dict() for e in registry if e.is_cross_page],
                    indent=2,
                    ensure_ascii=False,
                    default=str,
                )
            )
            sidecars["cross_page"] = str(xp_path)
        except Exception as e:
            log.warning("emit_sidecars: Exception degraded: %s", e)

    # 5. Transformation ledger
    try:
        _sdir = str(Path(__file__).resolve().parent.parent)
        if _sdir not in sys.path:
            sys.path.insert(0, _sdir)
        from ledger import get_ledger

        _led = get_ledger()
        if _led.count() > 0:
            ledger_path = out_dir / f"{stem}.ledger.jsonl"
            _led.set_output_path(ledger_path)
            _led.flush()
            sidecars["ledger"] = str(ledger_path)
    except Exception as e:
        log.warning("emit_sidecars: Exception degraded: %s", e)

    # 6. Event log flush
    try:
        from scientific_parser import flush_events_jsonl

        events_path = out_dir / f"{stem}.events.jsonl"
        flush_events_jsonl(events_path)
        sidecars["events"] = str(events_path)
    except Exception as e:
        log.warning("emit_sidecars: Exception degraded: %s", e)

    # 7. Candidate lattice (reviewer #38 — auditability). Per-region candidate
    #    sets captured by route_page_by_region, carrying full provenance
    #    (source_engine, evidence level, confidence, verification) from the
    #    provenance fix. Emitted for audit / abstention / training-data capture.
    if candidates:
        try:
            cand_path = out_dir / f"{stem}.candidates.json"
            cand_path.write_text(json.dumps(candidates, indent=2, ensure_ascii=False, default=str))
            sidecars["candidates"] = str(cand_path)
        except Exception as e:
            log.warning("emit_sidecars: Exception degraded: %s", e)

    return sidecars


def enrich_markdown_with_package(markdown_text: str) -> str:
    """Run package enrichments on existing markdown. Returns enriched text.

    Adds:
      - NumericGuard annotations (already done in monolith — skip)
      - Equation repair via reflexive loop
      - Domain detection banner at top
    """
    # Equation repair DISABLED — repair_all_equations drops valid math blocks
    # causing benchmark regression. Rule-based fix_latex_math_output handles
    # the repairable cases without dropping content.

    # Domain detection banner at top
    try:
        from scientific_parser import detect_domains

        domains = detect_domains(markdown_text[:5000])
        if domains:
            dom_str = ", ".join(f"{n}={s:.2f}" for n, s in domains)
            banner = f"<!-- Detected domains: {dom_str} -->\n"
            markdown_text = banner + markdown_text
    except Exception as e:
        log.debug("enrich_markdown_with_package: Exception degraded: %s", e)

    return markdown_text


__all__ = [
    "build_document_ir",
    "emit_sidecars",
    "enrich_markdown_with_package",
]
