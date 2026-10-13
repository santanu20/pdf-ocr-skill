"""Unintended-path warnings — loud accounting for silent degradations.

Every place the pipeline takes an UNINTENDED path (born-digital page forced
to OCR, specialist model failing, region output rejected, backend falling
back) records a warning here. The main flow aggregates them into ONE stderr
block per document (format_path_warnings_summary) and serializes them into
the --json summary ("path_warnings") so agents can see every degradation
instead of it vanishing into debug logs.

Reconstruction note (2026-08-18): the original module was lost to a harness
worktree reset while untracked; this file was rebuilt from the call sites
in pdf_ocr.py, region_dispatcher.py, layout_detector.py, and
equation_agent.py (API: record/reset/get/format* + kind constants).
"""

from __future__ import annotations

# Warning kinds (stable ids — summary consumers key on these)
PDFTOTEXT_UNAVAILABLE = "pdftotext_unavailable"
MIN_DOC_OCR_FORCED = "min_doc_ocr_forced"
VL_FALLBACK_USED = "vl_fallback_used"
MODEL_FAILURE_DEGRADED = "model_failure_degraded"
OCR_MODEL_SWAP = "ocr_model_swap"
FAST_PATH_OCR_ESCALATION = "fast_path_ocr_escalation"
REGION_OUTPUT_REJECTED = "region_output_rejected"
LAYOUT_DETECT_FAILED = "layout_detect_failed"
REGION_ROUTING_NO_REGIONS = "region_routing_no_regions"
SPECIALIST_FALLBACK = "specialist_fallback"
LAYOUT_BACKEND_FALLBACK = "layout_backend_fallback"
EQUATION_MODEL_FALLBACK = "equation_model_fallback"
TABLE_MODEL_FALLBACK = "table_model_fallback"

# Gap-wiring codes (2026-08-18 audit — every remaining silent site):
STAGE_SKIPPED = "stage_skipped"
PAGE_COUNT_FALLBACK = "page_count_fallback"
PAGE_RENDER_FALLBACK = "page_render_fallback"
SCRIPT_EXTRACTION_FALLBACK = "script_extraction_fallback"
OCR_PROMPT_FALLBACK = "ocr_prompt_fallback"
SPECIALIST_PROVIDERS_MISSING = "specialist_providers_missing"
CLI_CONFIG_IGNORED = "cli_config_ignored"
FONT_MATH_CASCADE_FAILED = "font_math_cascade_failed"
OCR_ENGINE_LEGACY_API = "ocr_engine_legacy_api"
FIGURE_TABLE_REROUTE = "figure_table_reroute"
SPAN_REPAIR_APPLIED = "span_repair_applied"

# Module-level collector (reset per document by the main flow)
_records: list[dict] = []


def reset_path_warnings() -> None:
    """Clear the collector (start of document)."""
    _records.clear()


def record_path_warning(kind, message, page=None, pages=None, details=None) -> None:
    """Record one unintended-path event.

    kind: one of the *_* constants above (or any stable string).
    message: human-readable, actionable description.
    page: single page number (optional). pages: list of page numbers
    (optional). details: arbitrary JSON-serializable context.
    """
    rec = {"kind": kind, "message": message}
    if page is not None:
        rec["page"] = page
    if pages:
        rec["pages"] = list(pages)
    if details:
        rec["details"] = details
    _records.append(rec)


def get_path_warnings() -> list[dict]:
    """All records for this document (serialized into the json summary)."""
    return list(_records)


def format_path_warning_records(records) -> str:
    """Render previously-serialized records (cache-replay path)."""
    if not records:
        return ""
    lines = ["⚠ Unintended-path warnings:"]
    for r in records:
        loc = f" p{r['page']}" if r.get("page") is not None else ""
        if r.get("pages"):
            loc = f" pages {r['pages']}"
        lines.append(f"  [{r.get('kind', '?')}] {r.get('message', '')}{loc}")
    return "\n".join(lines)


def format_path_warnings_summary() -> str:
    """One aggregated stderr block for the whole document (empty if clean)."""
    return format_path_warning_records(_records)


def warn_born_digital_ocr_routes(page_routes, doc_info, is_image=False) -> None:
    """Flag born-digital pages the router sent to the OCR path.

    Either a routing false-positive or a genuinely broken text layer —
    both are loud, neither is silent.
    """
    if is_image or not doc_info or not doc_info.get("born_digital"):
        return
    ocr_pages = [p for p, (route, _) in page_routes.items() if route == "ocr"]
    if ocr_pages:
        record_path_warning(
            "born_digital_ocr_routes",
            "born-digital document has OCR-routed pages "
            "(routing signal or broken text layer)",
            pages=ocr_pages,
        )


__all__ = [
    "PDFTOTEXT_UNAVAILABLE",
    "MIN_DOC_OCR_FORCED",
    "VL_FALLBACK_USED",
    "MODEL_FAILURE_DEGRADED",
    "OCR_MODEL_SWAP",
    "FAST_PATH_OCR_ESCALATION",
    "REGION_OUTPUT_REJECTED",
    "LAYOUT_DETECT_FAILED",
    "REGION_ROUTING_NO_REGIONS",
    "SPECIALIST_FALLBACK",
    "LAYOUT_BACKEND_FALLBACK",
    "EQUATION_MODEL_FALLBACK",
    "TABLE_MODEL_FALLBACK",
    "STAGE_SKIPPED",
    "PAGE_COUNT_FALLBACK",
    "PAGE_RENDER_FALLBACK",
    "SCRIPT_EXTRACTION_FALLBACK",
    "OCR_PROMPT_FALLBACK",
    "SPECIALIST_PROVIDERS_MISSING",
    "CLI_CONFIG_IGNORED",
    "FONT_MATH_CASCADE_FAILED",
    "OCR_ENGINE_LEGACY_API",
    "FIGURE_TABLE_REROUTE",
    "SPAN_REPAIR_APPLIED",
    "reset_path_warnings",
    "get_path_warnings",
    "format_path_warning_records",
    "format_path_warnings_summary",
    "warn_born_digital_ocr_routes",
]
