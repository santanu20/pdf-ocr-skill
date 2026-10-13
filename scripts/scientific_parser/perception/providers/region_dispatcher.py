"""route_page_by_region — SOTA per-region specialist dispatch (keystone).

Extracted from scripts/pdf_ocr.py during Phase R4 refactoring (2026-08-12).

Implements the proper SOTA workflow:
  1. RapidLayout detects regions (text/title/table/figure/equation/...)
  2. For each region by type:
     - text/title/list → keep text-layer OR OCR per region
     - equation        → RapidLaTeX (REPLACE OCR garble with LaTeX)
     - table           → RapidTable (REPLACE flat OCR cells with HTML)
     - figure          → caption only (existing associate_captions)
  3. Sort regions in reading order (XY-Cut++ on region bboxes)
  4. Return concatenated markdown

Robustness: every provider call wrapped in try/except — fall back to
text-layer / OCR per region on any specialist failure.
"""

from __future__ import annotations

import logging
import re

from shared import *  # ALL shared constants, patterns, state

from scientific_parser.core.candidates import Candidate, CandidateSet
from scientific_parser.core.evidence import DEFAULT_RESOLVER
from scientific_parser.core.ir import (
    E0_NATIVE_PDF_TEXT,
    E3_CLASSICAL_OCR,
    E4_SPECIALIST_OCR,
    Evidence,
)
from scientific_parser.ingest.coords import CoordinateTransform
from scientific_parser.routing.risk import risk as assess_risk
from scientific_parser.verification.retry_actions import (
    action_for,
    classify_failure,
)
from scientific_parser.verification.retry_executor import execute_action
from scientific_parser.core.path_warnings import (
    record_path_warning,
    LAYOUT_DETECT_FAILED,
    REGION_ROUTING_NO_REGIONS,
    SPECIALIST_FALLBACK,
    FIGURE_TABLE_REROUTE,
)

# Fraction of fallback-text length below which routed output is treated as a
# likely content loss (reviewer #5: specialist routing must not silently drop
# content the layout detector missed — e.g. a footnote).
_CONTENT_LOSS_RATIO = 0.5
# Minimum native text-block chars for the page geometry to be trusted as a
# reliable structural signal. Below this, the page is treated as structurally
# unknown (scanned / image-dominant) and the layout gate runs the CNN.
_MIN_NATIVE_CHARS = 200

log = logging.getLogger(__name__)


def _crop_image(img_path: str, bbox: tuple) -> str:
    """Crop image to bbox (x0, y0, x1, y1) in pixel coords. Returns new path."""
    try:
        from PIL import Image

        img = Image.open(img_path)
        x0, y0, x1, y1 = bbox
        x0 = max(0, int(x0))
        y0 = max(0, int(y0))
        x1 = min(img.width, int(x1))
        y1 = min(img.height, int(y1))
        if x1 <= x0 or y1 <= y0:
            return img_path
        cropped = img.crop((x0, y0, x1, y1))
        out_path = img_path + f".crop_{x0}_{y0}_{x1}_{y1}.png"
        cropped.save(out_path)
        return out_path
    except Exception as e:
        log.debug("crop failed (non-fatal, using full image): %s", e)
        return img_path


def _raster_overlap_ratio(pdf_path: str, page_num: int, bbox: tuple, render_dpi: float) -> float:
    """Fraction of ``bbox`` (render px) covered by raster images on the page.

    get_image_info() returns PDF points; DocLayout bboxes are render pixels.
    """
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            scale = render_dpi / 72.0
            x0, y0, x1, y1 = bbox
            r_area = max(0.0, (x1 - x0) * (y1 - y0))
            if r_area <= 0:
                return 0.0
            cov = 0.0
            for info in page.get_image_info():
                ib = info.get("bbox")
                if not ib:
                    continue
                ix0, iy0, ix1, iy1 = (v * scale for v in ib)
                ow = min(x1, ix1) - max(x0, ix0)
                oh = min(y1, iy1) - max(y0, iy0)
                if ow > 0 and oh > 0:
                    cov += ow * oh
            return min(1.0, cov / r_area)
    except Exception as e:
        log.debug("_raster_overlap_ratio failed: %s", e)
        return 0.0


def _figure_is_raster_table(
    pdf_path: str, page_num: int, img_path: str, bbox: tuple, render_dpi: float
) -> bool:
    """Evidence-fusion probe: is this 'figure' region actually a raster table?

    DocLayout labels are single-source evidence; tables exported as images
    (measured 2026-08-19: 3 bench PDFs, 25 gold tests) get labelled 'figure'
    and their content is never extracted. Probe = cheap geometric signature
    (detect_borderless_table) on OCR word boxes of the crop — the same
    x-recurrence signature used for native borderless tables. RapidTable's
    validation rubric still gates the final emission (probe false-positives
    cost one model call, never emit garbage).
    """
    if _raster_overlap_ratio(pdf_path, page_num, bbox, render_dpi) < 0.5:
        return False  # vector figure / photo not covering the region
    try:
        from scientific_parser.ingest.ocr_engine import ocr_image_rapidocr_full

        crop = _crop_image(img_path, bbox)
        _, _boxes, txts, scores = ocr_image_rapidocr_full(crop)
        if _boxes is None or txts is None:
            return False
        words = []
        for box, txt, sc in zip(_boxes, txts, scores or []):
            if sc is not None and sc < 0.5:
                continue
            pts = [(float(p[0]), float(p[1])) for p in box]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            words.append((min(xs), min(ys), max(xs), max(ys), str(txt)))
        return _ocr_grid_signature(words, min_rows=5)
    except Exception as e:
        log.debug("figure raster-table probe failed: %s", e)
        return False


def _ocr_grid_signature(words: list, min_rows: int = 5) -> bool:
    """x-recurrence table signature for OCR word boxes.

    ``detect_borderless_table`` groups rows by EXACT rounded bottom-y —
    valid for native PDF words (exact baselines), but OCR polygon y-jitter
    fragments rows. Here rows are y-CENTER clusters (tolerance = 0.6x
    median word height), then the same >=3 recurring x0-column test.
    """
    if len(words) < min_rows * 2:
        return False
    heights = sorted(w[3] - w[1] for w in words)
    tol = max(3.0, 0.6 * heights[len(heights) // 2])
    rows = []
    for w in sorted(words, key=lambda w: (w[1] + w[3]) / 2):
        cy = (w[1] + w[3]) / 2
        if rows and abs(cy - rows[-1]["cy"]) <= tol:
            rows[-1]["words"].append(w)
            n = len(rows[-1]["words"])
            rows[-1]["cy"] = (rows[-1]["cy"] * (n - 1) + cy) / n
        else:
            rows.append({"cy": cy, "words": [w]})
    n_rows = len(rows)
    if n_rows < min_rows:
        return False
    short = sum(1 for r in rows if len(r["words"]) <= 6)
    if short / n_rows < 0.6:
        return False
    from collections import Counter

    xs = sorted(round(w[0]) for r in rows for w in r["words"])
    clusters = []
    for x in xs:
        if clusters and x - clusters[-1][-1] <= 5:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    recurring = sum(1 for cl in clusters if len(cl) >= max(4, n_rows * 0.2))
    return recurring >= 3


def _sort_regions_reading_order(regions: list) -> list:
    """Sort regions in reading order using XY-Cut++ (arXiv 2504.10258).

    Primary: the canonical ``perception.layout.xy_cut.reading_order``
    implementation (hierarchical-mask recursive bisection, direction-aware
    for multi-column scientific layouts). Falls back to the legacy monolith
    ``_xy_cut_partition`` port, then to a top-to-bottom bbox sort.
    """
    if not regions:
        return []
    if len(regions) == 1:
        return regions

    # Build the {x0,y0,x1,y1,type} records xy_cut.reading_order expects,
    # carrying a back-reference to the original region dict.
    records = []
    for r in regions:
        bbox = r.get("bbox", (0, 0, 0, 0))
        if len(bbox) != 4:
            bbox = (0, 0, 0, 0)
        x0, y0, x1, y1 = bbox
        records.append(
            {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "type": r.get("type", "text"), "_region": r}
        )

    # Primary: canonical XY-Cut++ (documented, arXiv 2504.10258 Algorithm 1).
    try:
        from scientific_parser.perception.layout.xy_cut import reading_order

        order = reading_order(records, prefer_columns=True)
        out = [records[i]["_region"] for i in order]
        if len(out) == len(regions):
            return out
    except Exception as e:
        log.debug("_sort_regions_reading_order: Exception degraded: %s", e)

    # Fallback 1: legacy monolith XY-cut port.
    try:
        from pdf_ocr import _xy_cut_partition

        blocks = [
            {
                "x0": r["x0"],
                "y0": r["y0"],
                "x1": r["x1"],
                "y1": r["y1"],
                "cx": (r["x0"] + r["x1"]) / 2,
                "cy": (r["y0"] + r["y1"]) / 2,
                "text": r["type"],
                "_region": r["_region"],
            }
            for r in records
        ]
        ordered_blocks = _xy_cut_partition(blocks)
        out_fb1 = [b["_region"] for b in ordered_blocks]
        if len(out_fb1) == len(regions):
            return out_fb1
    except Exception as e:
        log.debug("_sort_regions_reading_order: Exception degraded: %s", e)

    # Fallback 2: top-to-bottom sort.
    return sorted(regions, key=lambda r: r.get("bbox", (0, 0, 0, 0))[1])


def _extract_text_from_bbox(
    pdf_path: str,
    page_num: int,
    bbox: tuple,
    transform: CoordinateTransform | None = None,
    render_dpi: float = 150.0,
) -> str:
    """Extract text from PDF text layer within bbox (whole LINES only).

    Args:
        pdf_path: PDF file path.
        page_num: 1-indexed page number.
        bbox: ``(x0, y0, x1, y1)`` in render-pixel coordinates (matches the
            rendered image consumed by the specialist providers).
        transform: Optional pre-built :class:`CoordinateTransform`. If None,
            one is constructed from the page rect + ``render_dpi``.
        render_dpi: DPI at which ``bbox`` was computed. Ignored when
            ``transform`` is provided.

    Returns:
        Extracted text, stripped. Empty string on any failure (logged debug).

    Audit H6 fix (2026-08-18): the old ``get_text("text", clip=...)``
    sliced lines at the region bbox — mid-word cuts ("conditions w|ith")
    whenever a layout-region boundary crossed a text line, and the
    fast-path reassembly then replaced clean page text with those
    fragments. Lines are now selected whole: a line belongs to the
    region containing its bbox CENTER (search expanded 3pt so
    boundary-adjacent lines are considered), and is emitted in full,
    in page order.
    """
    try:
        import fitz

        doc = fitz.open(pdf_path)
        page = doc[page_num - 1]
        if transform is None:
            transform = CoordinateTransform(
                pdf_w_pt=float(page.rect.width),
                pdf_h_pt=float(page.rect.height),
                render_dpi=float(render_dpi),
            )
        # bbox is in render px -> convert to PDF pt, then clip to page bounds.
        bbox_pdf_pt = transform.render_to_pdf(bbox)
        clipped = transform.clip_rect_pdf(bbox_pdf_pt)
        clip = fitz.Rect(*clipped)
        # H6 root fix: pass 1 uses a WIDE clip merely to FIND candidate
        # lines cheaply, then selection is by line-bbox CENTER inside the
        # region. Spans are read from the UNCLIPPED page dict — MuPDF's
        # clip trims span text at the clip edge (region bboxes from
        # layout models are routinely narrower than the text line:
        # 0514f667 region ends 568px, line ends 591px → "conditions wi").
        probe = page.get_text(
            "dict",
            clip=fitz.Rect(clip.x0 - 3, clip.y0 - 3, clip.x1 + 3, clip.y1 + 3),
        )
        selected = []
        for block in probe.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                lb = line.get("bbox", (0, 0, 0, 0))
                cx, cy = (lb[0] + lb[2]) / 2, (lb[1] + lb[3]) / 2
                if clip.x0 <= cx <= clip.x1 and clip.y0 <= cy <= clip.y1:
                    selected.append(lb)
        if not selected:
            doc.close()
            return ""
        # pass 2: full-page dict (no clip) — line texts are complete.
        full = page.get_text("dict")
        parts = []
        for block in full.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                if line.get("bbox", (0, 0, 0, 0)) in selected:
                    txt = "".join(s.get("text", "") for s in line.get("spans", []))
                    if txt.strip():
                        parts.append(txt.rstrip())
        doc.close()
        return "\n".join(parts).strip()
    except Exception as e:
        log.debug("bbox text extract failed: %s", e)
        return ""


def _run_domain_verifiers(text: str, rtype: str, stats: dict) -> None:
    """Run self-gating typed domain validators on a region's text.

    Chemistry (formula validity) + geology (oxide totals, U-Pb age, REE
    sequence). Each is self-gating (returns pass when the content signature
    is absent), so this is safe on any region. Failures are recorded into
    stats["verifications"] for audit and feed the risk/escalation layer.
    """
    try:
        from scientific_parser.science.formulas import find_formulas_in_text
        from scientific_parser.verification.typed import (
            validate_formula,
            validate_oxide_total,
            validate_ree_sequence_typed,
            validate_upb_age,
        )
    except Exception:
        return
    verifs = stats.setdefault("verifications", [])
    # Chemistry: validate any chemical formula tokens found in the text.
    try:
        for f in find_formulas_in_text(text)[:3]:
            vr = validate_formula(f.raw)
            verifs.append(
                {
                    "type": "chemistry",
                    "validator": "chemistry_formula",
                    "formula": f.raw,
                    "score": vr.score,
                    "passed": vr.passed,
                }
            )
    except Exception as e:
        log.debug("_run_domain_verifiers: Exception degraded: %s", e)
    # Geology: oxide totals, U-Pb ages, REE sequence (self-gating).
    for label, fn in (
        ("geology_oxide_total", validate_oxide_total),
        ("geology_upb_age", validate_upb_age),
        ("geology_ree_sequence", validate_ree_sequence_typed),
    ):
        try:
            vr = fn(text)
            if not vr.passed:
                verifs.append(
                    {"type": "geology", "validator": label, "score": vr.score, "passed": vr.passed}
                )
        except Exception as e:
            log.debug("_run_domain_verifiers: Exception degraded: %s", e)


def _profile_page(pdf_path, page_num, fallback_text="", render_dpi=150.0):
    """Build a PageProfile from PDF text-block geometry + text markers.

    The layout skip-gate needs reliable structural signals (columns, tables).
    Flat text — especially OCR output — cannot carry them: OCR flattens the
    page to a linear string, destroying all geometry. So this profiler reads
    the native text-block bboxes via PyMuPDF (free, no CNN) and measures real
    column structure. Pages with no native text layer (scanned, image-only)
    are marked ``structure_known=False`` so the gate conservatively runs
    layout rather than trusting a blind "looks simple".

    Text regex markers (table caption, equations, figures) are kept as
    supplementary positive signals — they can only FORCE layout, never skip.
    """
    from scientific_parser.routing.layout_gate import (
        PageProfile,
        count_columns,
        detect_borderless_table,
    )

    t = fallback_text or ""
    has_tables = bool(re.search(r"\|[-:| ]+\||\bTable\s+\d", t, re.IGNORECASE))
    has_equations = bool(re.search(r"\$\$|\\(?:frac|sqrt|sum|int|alpha|beta)\b|Equation", t))
    has_figures = bool(re.search(r"\bFig(?:ure)?\.?\s*\d", t, re.IGNORECASE))

    structure_known = False
    n_columns = 1
    text_chars = len(t.strip())
    has_native_tables = False
    has_borderless = False
    try:
        import fitz

        with fitz.open(pdf_path) as doc:
            page = doc[page_num - 1]
            blocks = page.get_text("blocks")
            page_width = float(page.rect.width)
            native_chars = sum(len(b[4].strip()) for b in blocks if len(b) > 4)
            if native_chars >= _MIN_NATIVE_CHARS:
                structure_known = True
                text_chars = native_chars
                n_columns = count_columns(blocks, page_width)
                # Geometric table signals (2026-08-15): pdftotext text
                # markers miss table pages without pipes/captions. Ruled
                # tables found natively (exact text); borderless via
                # x-recurrence on words.
                try:
                    from scientific_parser.perception.tables.detector import (
                        native_find_tables,
                    )

                    has_native_tables = bool(native_find_tables(pdf_path, page_num))
                    if not has_native_tables:
                        has_borderless = detect_borderless_table(page.get_text("words"))
                except Exception as e:
                    log.debug("geometric table probe failed: %s", e)
    except Exception as e:
        log.debug("page geometry profile failed (conservative fallback): %s", e)

    return PageProfile(
        text_chars=text_chars,
        n_columns=n_columns,
        has_tables=has_tables,
        has_equations=has_equations,
        has_figures=has_figures,
        structure_known=structure_known,
        has_native_tables=has_native_tables,
        has_borderless_table=has_borderless,
    )


def _region_risk(cset: CandidateSet, resolver=DEFAULT_RESOLVER):
    """Risk-assess one region from its winning candidate (reviewer #6/#24).

    uncertainty = 1 - resolver_score(winner), boosted by cross-source
    disagreement (agreement < 1). risk = uncertainty x semantic_impact.
    Returns a RiskAssessment or None when the region has no candidate.
    """
    winner = cset.winner(resolver)
    if winner is None:
        return None
    score = resolver.score(winner.evidence)
    uncertainty = max(0.0, 1.0 - score)
    agr = cset.agreement()
    if agr is not None and agr < 1.0:
        uncertainty = min(1.0, uncertainty + (1.0 - agr))
    return assess_risk(uncertainty, cset.region_type)


def route_page_by_region(
    pdf_path: str,
    page_num: int,
    img_path: str,
    providers: dict,
    fallback_text: str = "",
    render_dpi: float = 150.0,
    execute_escalations: bool = False,
    resolver_drives_output: bool = False,
    reuse_page_ocr: bool = False,
) -> tuple[str, dict]:
    """SOTA per-region specialist dispatch.

    Args:
        pdf_path: PDF file path
        page_num: 1-indexed page number
        img_path: rendered PNG of the page (for specialist inference)
        providers: dict with optional keys 'layout', 'formula', 'table'
                   (each value is a provider instance or None)
        fallback_text: text-layer or OCR output to use if region routing fails
        render_dpi: DPI at which ``img_path`` was rendered and at which region
            bboxes are expressed. Used to build a ``CoordinateTransform`` for
            PDF-text-clip extraction. Default 150 (matches the prior
            hardcoded /1240, /1754 heuristic on A4).

    Returns:
        (markdown_text, stats_dict)
        stats_dict: {n_regions, n_equations, n_tables, n_dispatched, n_fallback, method}
    """
    stats = {
        "n_regions": 0,
        "n_equations": 0,
        "n_tables": 0,
        "n_dispatched": 0,
        "n_fallback": 0,
        "method": "region-routed",
        "escalations": [],
    }

    layout_provider = providers.get("layout")
    formula_provider = providers.get("formula")
    table_provider = providers.get("table")

    # Build the coordinate transform ONCE from the actual page dimensions.
    # Replaces the prior hardcoded ``scale_x = page_w / 1240`` heuristic.
    transform: CoordinateTransform | None = None
    try:
        import fitz  # type: ignore[import]

        _doc = fitz.open(pdf_path)
        _page = _doc[page_num - 1]
        transform = CoordinateTransform(
            pdf_w_pt=float(_page.rect.width),
            pdf_h_pt=float(_page.rect.height),
            render_dpi=float(render_dpi),
        )
        _doc.close()
    except Exception as e:
        log.debug("transform build deferred to per-call: %s", e)
        # _extract_text_from_bbox will build a fallback transform itself.

    # Step 1: detect regions
    # Layout-model skip gate (reviewer #15): on a simple single-column page
    # with substantial native text and no table/equation/figure content, the
    # layout CNN adds nothing — skip it and use the native text directly.
    # Conservative: any structural-content marker forces layout to run.
    if layout_provider and layout_provider.available():
        from scientific_parser.routing.layout_gate import (
            should_run_layout,
        )

        profile = _profile_page(pdf_path, page_num, fallback_text, render_dpi)
        if not should_run_layout(profile):
            stats["method"] = "skipped-layout-simple"
            stats["n_fallback"] = 1
            stats["layout_skipped"] = True
            return fallback_text, stats
        try:
            regions = layout_provider.detect_regions(pdf_path, page_num)
        except Exception as e:
            log.warning("layout detect failed (falling back): %s", e)
            record_path_warning(
                LAYOUT_DETECT_FAILED,
                "layout detection failed — page falls back to text-layer/OCR",
                page=page_num,
            )
            regions = []
    else:
        regions = []

    if not regions:
        stats["method"] = "fallback-no-regions"
        stats["n_fallback"] = 1
        record_path_warning(
            REGION_ROUTING_NO_REGIONS,
            "layout detector found no regions — page falls back to text-layer/OCR",
            page=page_num,
        )
        return fallback_text, stats

    stats["n_regions"] = len(regions)
    regions = _sort_regions_reading_order(regions)

    # ── Ensemble table locate (TableAgent, stage 1) ───────────────────── #
    # DocLayout-YOLO detects ~92% of tables but misses some that PyMuPDF's
    # native find_tables catches (ruled tables, exact text layer). The agent
    # locates the union; native-only candidates (DocLayout-YOLO missed them)
    # are injected here so the per-region loop processes the ensemble (97%
    # recall). Per-region extract+validate+emit runs in the table branch below.
    table_agent = None
    equation_agent = None
    if formula_provider and formula_provider.available():
        from scientific_parser.perception.equations.equation_agent import EquationAgent

        equation_agent = EquationAgent(
            pdf_path, page_num, render_dpi, img_path=img_path, formula_provider=formula_provider
        )
    if table_provider and table_provider.available():
        _page_area_pt = None
        try:
            import fitz as _fitz

            with _fitz.open(pdf_path) as _doc:
                _pg = _doc[page_num - 1]
                _page_area_pt = float(_pg.rect.width * _pg.rect.height)
        except Exception as e:
            log.debug("route_page_by_region: Exception degraded: %s", e)
        from scientific_parser.perception.tables.table_agent import TableAgent

        table_agent = TableAgent(
            pdf_path,
            page_num,
            render_dpi,
            img_path=img_path,
            rapidtable_provider=table_provider,
            page_area_pt=_page_area_pt,
        )
        _n_injected = 0
        for cand in table_agent.locate(layout_regions=regions):
            # source="pymupdf" = native table DocLayout-YOLO missed → inject.
            if cand.get("source") == "pymupdf":
                regions.append(
                    {"type": "table", "bbox": cand["bbox"], "_agent_native": cand["native"]}
                )
                _n_injected += 1
        if _n_injected:
            regions = _sort_regions_reading_order(regions)
            stats["n_regions"] = len(regions)
            stats["n_tables_injected"] = _n_injected

        # COLUMN-GRID MISREAD GATE (2026-08-18): BOTH detectors — the
        # layout CNN AND pymupdf find_tables — misread a 2-column page's
        # aligned text grid as a table (measured 0621a009 pg31: refs
        # page, 81x7 "table", RapidTable emitted word-fragment pipes,
        # destroying the text). Evidence fusion gate: on a column-sharp
        # page, a table region that (a) crosses the column gutter,
        # (b) spans >50% of page height, and (c) whose text-layer lines
        # are PROSE-length (median >= 60 chars — real table cells are
        # short) is the column grid, not a table → downgrade to text.
        try:
            import fitz as _fz2

            from scientific_parser.routing.layout_gate import (
                left_edge_column_sharpness,
            )

            _scale = render_dpi / 72.0
            with _fz2.open(pdf_path) as _d:
                _pg = _d[page_num - 1]
                _blocks = [b for b in _pg.get_text("blocks") if b[4].strip()]
                _page_h_pt = float(_pg.rect.height)
            _, _col2 = left_edge_column_sharpness(_blocks)
        except Exception as _e:
            _col2 = None
            log.debug("column-grid misread probe degraded: %s", _e)
        # PROSE-HALLUCINATION GATE (2026-08-19): RapidTable fabricates 1-2
        # col pipe "tables" from PROSE renders (measured 05d9fd5d pg6:
        # paragraphs + the page footer wrapped in a table, absent-test
        # gold fails on the surviving footer). Universal signature — a
        # REAL table's text-layer lines are SHORT; a prose region's lines
        # are paragraph-length. Downgrade when region lines are
        # prose-length (median >= 60 chars) AND the region is not a
        # column-sharp grid (covered above). Applies on ANY page shape.
        _n_prose_downgraded = 0
        for r in regions:
            if r.get("type") != "table":
                continue
            bx0, by0, bx1, by1 = (v / _scale for v in r.get("bbox", (0, 0, 0, 0)))
            lens = sorted(
                len(b[4].strip())
                for b in _blocks
                if b[1] >= by0 - 2 and b[3] <= by1 + 2 and len(b[4].strip()) >= 2
            )
            med = lens[len(lens) // 2] if lens else 0
            if med >= 60:
                r["type"] = "text"
                r["_downgraded_prose_halluc"] = True
                _n_prose_downgraded += 1
        if _n_prose_downgraded:
            stats["n_tables_downgraded_prose"] = _n_prose_downgraded
        if _col2 is not None:
            _n_downgraded = 0
            for r in regions:
                if r.get("type") != "table":
                    continue
                bx0, by0, bx1, by1 = (v / _scale for v in r.get("bbox", (0, 0, 0, 0)))
                crosses_gutter = bx0 < _col2 - 20 and bx1 > _col2 + 20
                tall = (by1 - by0) > 0.5 * _page_h_pt
                lens2 = sorted(
                    len(b[4].strip())
                    for b in _blocks
                    if b[1] >= by0 - 2 and b[3] <= by1 + 2 and len(b[4].strip()) >= 2
                )
                med2 = lens2[len(lens2) // 2] if lens2 else 0
                if crosses_gutter and tall and med2 >= 60:
                    r["type"] = "text"
                    r["_downgraded_from_table"] = True
                    _n_downgraded += 1
            if _n_downgraded:
                stats["n_tables_downgraded_column_grid"] = _n_downgraded

        # RASTER-TABLE RETYPE (2026-08-19): tables exported as raster images
        # (measured: 3 bench tables PDFs, 25 gold tests) are labelled
        # 'figure' by DocLayout — single-source label error. Evidence-fusion
        # probe: region substantially covered by a raster image AND the OCR
        # word-grid shows the x-recurrence column signature → retype to
        # 'table'. RapidTable's own validation rubric still gates emission,
        # so a probe false-positive costs one model call, never bad output.
        if table_agent is not None:
            _n_retyped = 0
            for r in regions:
                if r.get("type") != "figure":
                    continue
                if _figure_is_raster_table(
                    pdf_path, page_num, img_path, r.get("bbox", (0, 0, 0, 0)), render_dpi
                ):
                    r["_retyped_from_figure"] = True
                    r["type"] = "table"
                    _n_retyped += 1
            if _n_retyped:
                stats["n_figures_retyped_tables"] = _n_retyped
                record_path_warning(
                    FIGURE_TABLE_REROUTE,
                    f"{_n_retyped} figure region(s) probed as raster tables — "
                    f"rerouted to table specialist",
                    page=page_num,
                )

    # Step 2: dispatch each region by type
    output_parts = []
    all_candidates: list[CandidateSet] = []
    # Lazy cell for page-level OCR reuse (feeds RapidTable so it skips
    # per-crop re-OCR). Pre-captured ONCE here (not lazily in the table branch)
    # to eliminate the race condition when specialist regions run in parallel.
    page_ocr: list = [None]
    if reuse_page_ocr and any(r.get("type") == "table" for r in regions):
        try:
            from scientific_parser.ingest.ocr_engine import (
                ocr_image_rapidocr_full,
            )

            _, bxs, tts, scs = ocr_image_rapidocr_full(img_path)
            page_ocr[0] = (bxs, tts, scs) if bxs else ()
            if table_agent:
                table_agent.page_ocr = page_ocr[0] or None
        except Exception as oe:
            log.debug("page ocr reuse capture failed: %s", oe)
            page_ocr[0] = ()

    # ── Parallel specialist pre-dispatch ───────────────────────────────── #
    # Equation/table/chemical regions are independent (no inter-region deps)
    # and each takes 3-5s (model inference). Running them concurrently via a
    # thread pool cuts a page with 4 specialists from ~16s sequential → ~5s
    # parallel. Text regions stay inline (instant: _extract_text_from_bbox).
    _specialist_types = ("equation", "table", "chemical_structure", "molecule", "reaction")
    _specialist_results: dict = {}
    # ONNX models (RapidTable, DECIMER) may not be thread-safe — serialize
    # their inference while allowing equation dispatches (native fitz or
    # subprocess Ollama — both thread-safe) to run fully parallel.
    import threading as _threading

    _onnx_lock = _threading.Lock()

    def _dispatch_specialist(ri, region):
        """Process one specialist region → result dict (thread-safe, no shared mutable state)."""
        rtype = region.get("type", "text")
        bbox = region.get("bbox", (0, 0, 0, 0))
        text = ""
        src = "native-pdf-text"
        ev = E0_NATIVE_PDF_TEXT
        conf = 1.0
        delta: dict = {}
        extras: dict = {}

        if rtype == "equation" and equation_agent is not None:
            delta["n_equations"] = 1
            try:
                _eres = equation_agent.process(bbox)
                if _eres and _eres.latex:
                    text = _eres.latex
                    src = _eres.method
                    ev = E4_SPECIALIST_OCR
                    conf = _eres.confidence
                    delta["n_dispatched"] = 1
                    if _eres.method == "unicode-native":
                        delta["n_eq_native"] = 1
                    extras["equation_results"] = [_eres.to_dict()]
                else:
                    text = _extract_text_from_bbox(pdf_path, page_num, bbox, transform=transform)
                    delta["n_fallback"] = 1
            except Exception as e:
                log.warning("equation agent failed: %s", e)
                text = _extract_text_from_bbox(pdf_path, page_num, bbox, transform=transform)
                delta["n_fallback"] = 1
                record_path_warning(
                    SPECIALIST_FALLBACK,
                    "equation agent failed — region falls back to text-layer clip",
                    page=page_num,
                )

        elif rtype == "table" and table_agent is not None:
            delta["n_tables"] = 1
            _cand = {
                "bbox": bbox,
                "native": region.get("_agent_native"),
                # EXP-4b: router raster-table probe already confirmed grid
                # evidence on the rendered crop — bypass agent text-layer gate.
                "retyped_from_figure": bool(region.get("_retyped_from_figure")),
                # EXP-4b: evidence probes (fitz drawings/images) live in
                # PDF-pt; candidate["bbox"] is render-px (9a61fe78 lesson).
                "bbox_pdf_pt": tuple(transform.render_to_pdf(bbox)),
            }
            with _onnx_lock:
                _tres = table_agent.process(_cand)
            if _tres and _tres.markdown:
                text = _tres.markdown
                src = _tres.method
                ev = E4_SPECIALIST_OCR
                conf = _tres.confidence
                delta["n_dispatched"] = 1
                if _tres.method.startswith("pymupdf"):
                    delta["n_pymupdf"] = 1
                extras["table_results"] = [_tres.to_dict()]
            else:
                text = _extract_text_from_bbox(pdf_path, page_num, bbox, transform=transform)
                delta["n_fallback"] = 1
                record_path_warning(
                    SPECIALIST_FALLBACK,
                    "table agent produced no result — region falls back to text-layer clip",
                    page=page_num,
                )

        elif rtype in ("chemical_structure", "molecule", "reaction"):
            delta["n_chemical"] = 1
            chem = providers.get("chemical")
            if chem is None:
                try:
                    from scientific_parser.perception.providers.chemical_recognizer import (
                        get_chemical_recognizer,
                    )

                    chem = get_chemical_recognizer("default")
                except Exception:
                    chem = None
            if chem and chem.available():
                try:
                    region_img = _crop_image(img_path, bbox)
                    with _onnx_lock:
                        smiles, cconf = chem.recognize(region_img)
                    if smiles and cconf > 0.3:
                        text = f"```smiles\n{smiles}\n```"
                        src = (
                            "decimer" if getattr(chem, "PROVIDES", "") == "decimer" else "molscribe"
                        )
                        ev = E4_SPECIALIST_OCR
                        conf = cconf
                        delta["n_dispatched"] = 1
                    else:
                        text = _extract_text_from_bbox(
                            pdf_path, page_num, bbox, transform=transform
                        )
                        delta["n_fallback"] = 1
                except Exception as e:
                    log.warning("chemical OCSR dispatch failed: %s", e)
                    text = _extract_text_from_bbox(pdf_path, page_num, bbox, transform=transform)
                    delta["n_fallback"] = 1
                    record_path_warning(
                        SPECIALIST_FALLBACK,
                        "chemical OCSR dispatch failed — region falls back to text-layer clip",
                        page=page_num,
                    )
            else:
                text = _extract_text_from_bbox(pdf_path, page_num, bbox, transform=transform)
                delta["n_fallback"] = 1

        return {"text": text, "src": src, "ev": ev, "conf": conf, "delta": delta, "extras": extras}

    _specialist_jobs = [
        (ri, region) for ri, region in enumerate(regions) if region.get("type") in _specialist_types
    ]
    # NOTE: specialist dispatch is SEQUENTIAL (not ThreadPoolExecutor) because
    # PyMuPDF (fitz) is NOT thread-safe — concurrent fitz operations across
    # threads corrupt MuPDF global state → deadlock. Every specialist uses
    # fitz (text extraction, native tables). Process-based parallelism would
    # work but has higher overhead (spawn + serialization). Sequential is safe.
    for ri, reg in _specialist_jobs:
        _specialist_results[ri] = _dispatch_specialist(ri, reg)

    # ── Sequential assembly loop (reading order preserved) ────────────── #
    for ri, region in enumerate(regions):
        rtype = region.get("type", "text")
        bbox = region.get("bbox", (0, 0, 0, 0))
        region_text = ""
        _src_engine = "native-pdf-text"
        _evidence_lvl = E0_NATIVE_PDF_TEXT
        _conf = 1.0
        _verification = None

        if ri in _specialist_results:
            # Use pre-computed parallel result (specialist already dispatched).
            sr = _specialist_results[ri]
            region_text = sr["text"]
            _src_engine = sr["src"]
            _evidence_lvl = sr["ev"]
            _conf = sr["conf"]
            for k, v in sr["delta"].items():
                stats[k] = stats.get(k, 0) + v
            for k, v in sr["extras"].items():
                stats.setdefault(k, []).extend(v)

        else:
            # text/title/list/figure/header/footer — text layer first
            region_text = _extract_text_from_bbox(pdf_path, page_num, bbox, transform=transform)
            # SCANNED PDF: when no text layer, DON'T re-OCR individual text
            # regions. The page-level OCR (caller's fallback_text) already
            # captured all text. Per-region OCR here duplicates that work
            # and adds ~30s per page (measured 2026-08-14 bench debug).
            # The content-loss guard (below) falls back to fallback_text
            # when region routing produces insufficient output.

        if region_text and region_text.strip():
            region_text = region_text.strip()
            output_parts.append(region_text)
            # Typed domain verification (reviewer #19): run self-gating
            # geochemistry/chemistry validators on table/text regions. They
            # return pass when the content signature is absent, so running
            # them universally is safe and makes the typed layer UTILIZED.
            if rtype in ("table", "text", "paragraph"):
                _run_domain_verifiers(region_text, rtype, stats)
            # Capture the selected candidate with full provenance (reviewer
            # #38: candidates are first-class + auditable, never discarded).
            cset = CandidateSet(
                region_id=f"p{page_num}_r{ri}",
                region_type=rtype,
                bbox=tuple(bbox),
            )
            cset.add(
                Candidate(
                    source_engine=_src_engine,
                    evidence=Evidence(
                        level=_evidence_lvl,
                        source=_src_engine,
                        value=region_text,
                        confidence=_conf,
                        verification=_verification,
                    ),
                    text=region_text,
                    crop_bbox=tuple(bbox),
                    render_dpi=float(render_dpi),
                )
            )
            # Resolver-drives-output (reviewer #5): when a specialist won, also
            # capture the native text-layer as a 2nd candidate so the weighted
            # resolver CAN override a weak specialist. Default off (first-wins
            # preserves byte-identical output); flip --resolver-drives-output
            # to let the resolver pick and emit cset.resolved_text().
            if resolver_drives_output and _src_engine not in (
                "native-pdf-text",
                "rapidocr",
            ):
                try:
                    native_txt = _extract_text_from_bbox(
                        pdf_path, page_num, bbox, transform=transform
                    )
                    if native_txt and native_txt.strip() != region_text:
                        cset.add(
                            Candidate(
                                source_engine="native-pdf-text",
                                evidence=Evidence(
                                    level=E0_NATIVE_PDF_TEXT,
                                    source="pdftotext",
                                    value=native_txt,
                                    confidence=1.0,
                                ),
                                text=native_txt,
                                crop_bbox=tuple(bbox),
                            )
                        )
                        region_text = cset.resolved_text() or region_text
                        # Replace the last appended part with resolver's choice.
                        if output_parts:
                            output_parts[-1] = region_text
                except Exception as e:
                    log.debug("route_page_by_region: Exception degraded: %s", e)
            all_candidates.append(cset)

            # Risk-aware selective redundancy (reviewer #6/#24, keystone D→F→H):
            # escalate only where uncertainty x impact justifies it. The
            # DECISION (risk -> classify -> prescribe) is always recorded in
            # stats["escalations"]; execute_action runs when
            # ``execute_escalations`` is enabled (default ON since 2026-08-14
            # — one cheap corrective action per risky region: tighter crop,
            # higher DPI; --no-escalate restores single-pass output).
            ra = _region_risk(cset)
            if ra is not None and ra.escalate:
                winner = cset.winner()
                fc = classify_failure(winner.evidence.verification if winner else None)
                action = action_for(
                    fc,
                    context={
                        "region_bbox": tuple(bbox),
                        "current_dpi": float(render_dpi),
                        "page_num": page_num,
                    },
                )
                esc_record = {
                    "region_id": cset.region_id,
                    "region_type": cset.region_type,
                    "risk": round(ra.risk, 3),
                    "uncertainty": round(ra.uncertainty, 3),
                    "impact": ra.impact,
                    "failure_class": fc.value,
                    "action": action.action.value,
                    "params": action.params,
                    "rationale": action.rationale,
                }
                if execute_escalations:
                    try:
                        outcome = execute_action(
                            action,
                            {
                                "img_path": img_path,
                                "pdf_path": pdf_path,
                                "page_num": page_num,
                                "region_bbox": tuple(bbox),
                                "current_dpi": float(render_dpi),
                            },
                        )
                        esc_record["executed"] = outcome.success
                        esc_record["execute_message"] = outcome.message
                        if outcome.output_path:
                            esc_record["output_path"] = outcome.output_path
                    except Exception as ee:  # executor never blocks routing
                        esc_record["executed"] = False
                        esc_record["execute_message"] = f"executor error: {ee}"
                stats.setdefault("escalations", []).append(esc_record)

    if not output_parts:
        stats["method"] = "fallback-empty-regions"
        stats["candidates"] = [c.to_dict() for c in all_candidates]
        return fallback_text, stats

    routed = "\n\n".join(output_parts)
    # Content-loss guard (reviewer #5): region routing must not silently drop
    # content the layout detector missed. If routed text is materially shorter
    # than the fallback (page-level OCR / native), prefer the fallback so a
    # missed footnote or undetected region is not lost.
    # ALNUM-normalized (2026-08-15): structured tables legitimately compress
    # whitespace-flooded text layers (letter-spacing, padding) — raw-length
    # comparison rejected CORRECT line-grid tables against padded WRONG
    # fallbacks (measured wheat bench page: 3.2k correct vs 7.4k padded).
    _alnum = lambda s: len(re.sub(r"[^A-Za-z0-9]", "", s or ""))
    if fallback_text and _alnum(routed) < _CONTENT_LOSS_RATIO * _alnum(fallback_text):
        log.warning(
            "region routing lost content (%d -> %d chars); using fallback",
            len(fallback_text),
            len(routed),
        )
        stats["content_loss_guard"] = True
        stats["candidates"] = [c.to_dict() for c in all_candidates]
        return fallback_text, stats

    stats["candidates"] = [c.to_dict() for c in all_candidates]
    return routed, stats


__all__ = [
    "route_page_by_region",
    "_crop_image",
    "_sort_regions_reading_order",
    "_extract_text_from_bbox",
]
