"""TableAgent — deterministic table-extraction state machine (no LLM).

Born-digital-first: PyMuPDF native geometry (exact text layer, fast) before the
RapidTable CNN. Validation-scored: each candidate is scored on a 7-metric
rubric; low-score candidates escalate to the model or get flagged ``needs_review``.

Four stages:
  1. LOCATE  — ensemble: DocLayout-YOLO table-regions ∪ PyMuPDF ``find_tables``,
              deduped by IoU. Catches tables either source misses (97% recall
              on the olmOCR-bench tables category vs 92% solo).
  2. EXTRACT — PyMuPDF native first (free — already located) → RapidTable CNN
              fallback when native is absent or scores poorly.
  3. VALIDATE — 7-metric rubric (geometry, text-coverage, column/row consistency,
              header quality, numeric consistency, caption proximity).
  4. EMIT    — structured ``TableResult`` (bbox/rows/cells/confidence/method/
              needs_review) + markdown.

Replaces the inline table branch in ``region_dispatcher.route_page_by_region``.
"""

import re
from dataclasses import dataclass, field
import logging
from scientific_parser.core.path_warnings import (
    TABLE_MODEL_FALLBACK,
    record_path_warning,
)
from scientific_parser.routing.layout_gate import region_has_table_evidence

log = logging.getLogger(__name__)


@dataclass
class TableResult:
    """Structured table-extraction result (stage 4 emit)."""

    bbox: tuple
    rows: int
    columns: int
    cells: list  # 2D list of cell strings
    markdown: str
    confidence: float
    method: str  # "pymupdf" | "rapidtable" | "pymupdf-low"
    score_breakdown: dict = field(default_factory=dict)
    needs_review: bool = False

    def to_dict(self):
        return {
            "bbox": list(self.bbox),
            "rows": self.rows,
            "columns": self.columns,
            "confidence": self.confidence,
            "method": self.method,
            "needs_review": self.needs_review,
            "score": self.score_breakdown,
        }


class TableAgent:
    """Deterministic table agent (state machine, no LLM).

    Locate → Extract → Validate → Emit. Construct once per page; ``locate``
    seeds the ensemble, ``process`` runs stages 2-4 per candidate.
    """

    # 7-metric rubric weights (sum = 1.0). Calibrated against table_tests.jsonl
    # gold where geometry + text-coverage dominate table quality.
    WEIGHTS = {
        "geometry": 0.25,
        "text_coverage": 0.20,
        "column_consistency": 0.15,
        "row_consistency": 0.15,
        "header_quality": 0.10,
        "numeric_consistency": 0.10,
        "caption_proximity": 0.05,
    }
    # Confidence floors: below ACCEPT a candidate escalates; below REVIEW it is
    # flagged. PyMuPDF native is trusted more than the blended model score.
    ACCEPT_NATIVE = 0.50
    ACCEPT_MODEL = 0.40
    REVIEW = 0.60

    def __init__(
        self,
        pdf_path,
        page_num,
        render_dpi=150.0,
        img_path=None,
        rapidtable_provider=None,
        page_ocr=None,
        page_area_pt=None,
    ):
        self.pdf_path = pdf_path
        self.page_num = page_num
        self.render_dpi = render_dpi
        self.img_path = img_path
        self.rapidtable = rapidtable_provider
        self.page_area_pt = page_area_pt  # PDF-point page area (for mega-table guard)
        self.page_ocr = page_ocr  # (boxes, txts, scores) or None
        self._layer_text_norm = None  # lazy: page text, whitespace-collapsed
        self._rawdict = None  # cached char-level page dict (line-grid extract)
        self._grid_cache = {}  # bbox → line-grid pack (grid, k, modal_ratio)
        self._native = None  # cache for _native_tables()

    # ── Stage 1: LOCATE (ensemble) ──────────────────────────────────────── #

    def locate(self, layout_regions=None):
        """Ensemble locate: DocLayout-YOLO table-regions ∪ PyMuPDF tables.

        Returns candidate list ``[{bbox, native}]`` where ``native`` is the
        PyMuPDF cells/markdown when one overlaps, else None. Candidates from
        either source that the other missed are included (the recall win).
        """
        candidates = []
        seen = []
        # Source 1: DocLayout-YOLO table regions.
        if layout_regions:
            for r in layout_regions:
                if r.get("type") == "table":
                    bbox = tuple(r["bbox"])
                    candidates.append(
                        {
                            "bbox": bbox,
                            "native": None,
                            "source": "yolo",
                            # Raster-table retyped by the router probe
                            # (region_dispatcher _figure_is_raster_table):
                            # the probe already confirmed the table signature
                            # via OCR on the rendered crop — stronger evidence
                            # than the agent-level text-layer gate.
                            "retyped_from_figure": bool(r.get("_retyped_from_figure")),
                        }
                    )
                    seen.append(bbox)
        # Source 2: PyMuPDF native tables (well-formed only).
        for nt in self._native_tables():
            match = None
            for c in candidates:
                if _iou(c["bbox"], nt["bbox"]) >= 0.30:
                    match = c
                    break
            if match is not None:
                match["native"] = nt  # attach native content to the YOLO candidate
            else:
                # Native-only: DocLayout-YOLO missed it → inject (the ensemble win).
                candidates.append({"bbox": nt["bbox"], "native": nt, "source": "pymupdf"})
        # Source 3 (iter-123): camelot-stream TextEdges proposals — RECALL
        # SUPPLEMENT ONLY (fires when YOLO + native found nothing). The
        # text layer's own x-alignment proposes table regions that no
        # detector saw (borderless disclosure lists, bimodal label|value
        # layouts). PROPOSAL ONLY: every downstream gate (line-grid
        # rebuild, shred checks, 7-metric rubric) still disposes — a bad
        # proposal dies in process() and the region stays text.
        if not candidates:
            for bbox in self._textedges_candidates():
                candidates.append({"bbox": bbox, "native": None, "source": "textedges"})
        return candidates

    def _textedges_candidates(self):
        """Camelot-stream TextEdges port (MIT, core.py TextEdges behavioral).

        Collect left/right/middle x-edges of text lines, cluster within a
        tolerance, pick the alignment class with maximum line support;
        consecutive valid edge clusters are column boundaries; a y-band
        dense with column-crossing lines is a table region. Gates for
        precision (raw prototype measured 5/14 recall, mixed): >=3 edge
        clusters (2+ columns), >=6 spanning lines, >=5 rows tall.
        """
        import os as _os

        if _os.environ.get("PDF_OCR_TEXTEDGES", "1") != "1":
            return []
        try:
            pd = self._page_rawdict()
            if not pd:
                return []
            W = pd["rect"].width
            H = pd["rect"].height
            lines = []
            for block in pd.get("dict", {}).get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for ln in block.get("lines", []):
                    txt = "".join(s.get("text", "") for s in ln.get("spans", [])).strip()
                    bb = ln.get("bbox", (0, 0, 0, 0))
                    if txt and bb[3] <= H * 0.92 and bb[1] >= H * 0.06:
                        lines.append((bb, txt))
            if len(lines) < 8:
                return []

            def _cluster(vals, tol):
                out = []
                for v in sorted(vals):
                    if out and v - out[-1][-1] <= tol:
                        out[-1].append(v)
                    else:
                        out.append([v])
                return out

            edge_tol = max(6.0, W * 0.012)
            support = {}
            for cls in ("left", "right", "middle"):
                vals = {
                    "left": lambda bb: bb[0],
                    "right": lambda bb: bb[2],
                    "middle": lambda bb: (bb[0] + bb[2]) / 2,
                }[cls]
                clusters = [
                    c for c in _cluster([vals(bb) for bb, _t in lines], edge_tol) if len(c) >= 3
                ]
                support[cls] = (sum(len(c) for c in clusters), clusters)
            best = max(support, key=lambda k: support[k][0])
            clusters = support[best][1]
            if len(clusters) < 3:  # 2+ column boundaries
                return []
            centers = sorted(sum(c) / len(c) for c in clusters)
            xs = [centers[0] - edge_tol] + centers + [centers[-1] + edge_tol]

            def _cols_of(bb):
                n = 0
                for i in range(len(xs) - 1):
                    if bb[0] < xs[i + 1] and bb[2] > xs[i]:
                        n += 1
                return n

            spanning = [(bb, t) for bb, t in lines if _cols_of(bb) >= 2]
            if len(spanning) < 6:
                return []
            ys = [bb[1] for bb, _t in spanning] + [bb[3] for bb, _t in spanning]
            y0, y1 = min(ys), max(ys)
            avg_h = max(6.0, (y1 - y0) / len(spanning))
            if (y1 - y0) < 4 * avg_h:  # at least ~5 rows tall
                return []
            return [
                (
                    max(0.0, xs[0] - 2 * edge_tol),
                    max(0.0, y0 - avg_h),
                    min(W, xs[-1] + 2 * edge_tol),
                    min(H, y1 + avg_h),
                )
            ]
        except Exception as e:
            log.warning("textedges candidates failed: %s", e)
            return []

    def _native_tables(self):
        """Union of PyMuPDF ``find_tables`` + pdfplumber text-strategy.

        PyMuPDF (lines strategy) is strong on ruled tables; pdfplumber's text
        strategy catches borderless tables via word alignment. Both give exact
        text-layer cells (born-digital). Mega-tables (bbox > 70% of page — the
        full-page false positives text-strategy sometimes produces) are rejected
        by the page-area guard. Cached per agent instance.
        """
        if self._native is not None:
            return self._native
        results = []
        scale = self.render_dpi / 72.0

        def _add(ex, bbox_pt):
            nr = len(ex)
            nc = max((len(r) for r in ex), default=0) if ex else 0
            nonempty = sum(1 for r in ex for c in r if c and str(c).strip())
            if nr < 2 or nc < 2 or nonempty < 4:
                return
            x0, y0, x1, y1 = bbox_pt
            if self.page_area_pt and (x1 - x0) * (y1 - y0) > 0.70 * self.page_area_pt:
                # Full-page tables are LEGITIMATE (catalogs, schedules —
                # measured tables bench: pdfplumber candidate covering 84%
                # rejected here, leaving the page tableless). Reject only
                # the false-positive this guard exists for: prose grouped
                # into a mega-table has LONG cells; real tables are short.
                lens = sorted(len(str(c).strip()) for r in ex for c in r if c and str(c).strip())
                if not lens or lens[len(lens) // 2] > 30:
                    return  # prose-shaped mega-table — reject
            bbox = (x0 * scale, y0 * scale, x1 * scale, y1 * scale)
            # dedupe vs already-collected (IoU)
            if any(_iou(bbox, r["bbox"]) >= 0.4 for r in results):
                return
            results.append({"bbox": bbox, "cells": ex, "nrows": nr, "ncols": nc})

        # Source A: PyMuPDF find_tables (lines strategy).
        try:
            from scientific_parser.perception.tables.detector import native_find_tables

            for hit in native_find_tables(self.pdf_path, self.page_num):
                _add(hit["rows"], hit["bbox"])
        except Exception as e:
            log.debug("_native_tables: Exception degraded: %s", e)
        # Source B: pdfplumber text strategy (borderless tables via alignment).
        try:
            import pdfplumber

            with pdfplumber.open(self.pdf_path) as d:
                # BUG FIX 2026-08-15: was d.pages[0] — always page 1,
                # ignoring self.page_num on multi-page PDFs.
                for t in d.pages[self.page_num - 1].find_tables(
                    {"vertical_strategy": "text", "horizontal_strategy": "text"}
                ):
                    _add(t.extract(), t.bbox)
        except Exception as e:
            log.debug("_native_tables: Exception degraded: %s", e)
        self._native = results
        return results

    # ── Stages 2+3+4: EXTRACT → VALIDATE → EMIT (per candidate) ─────────── #

    def _page_rawdict(self):
        """Cached fitz rawdict (char-level) for the agent's page."""
        if self._rawdict is None:
            import fitz

            with fitz.open(self.pdf_path) as doc:
                self._rawdict = doc[self.page_num - 1].get_text("rawdict")
        return self._rawdict

    def _grid_candidate(self, bbox):
        """Cached line-grid result: (grid, k, modal_ratio) or None."""
        key = tuple(round(v, 1) for v in bbox)
        if key in self._grid_cache:
            return self._grid_cache[key]
        from collections import Counter

        pack = None
        grid = self._line_grid_extract(bbox)
        if grid:
            k = Counter(len(r) for r in grid).most_common(1)[0][0]
            modal_ratio = sum(1 for r in grid if len(r) == k) / len(grid)
            pack = (grid, k, round(modal_ratio, 3))
        self._grid_cache[key] = pack
        return pack

    def _line_grid_extract(self, bbox):
        """Marker-style grid reconstruction from char geometry (Apache-2.0 port).

        1. Tokens: split each line's chars at gaps > 0.5×char-height
           (recovers cells pdftext merged into one span).
        2. Rows: lines inside bbox with ≥2 tokens.
        3. Grid: modal-K anchor rows; column cuts = midpoints of per-column
           median x0; assign every token by x0 bucket into K cells.
        Returns 2D cells or None.
        """
        try:
            rd = self._page_rawdict()
        except Exception:
            return None
        bx0, by0, bx1, by1 = bbox
        # bbox is in render-px (dpi-scaled); rawdict is pt. Convert.
        inv = 72.0 / self.render_dpi
        bx0, by0, bx1, by1 = bx0 * inv, by0 * inv, bx1 * inv, by1 * inv

        rows = []
        for block in rd.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                toks, cur = [], None
                for span in line.get("spans", []):
                    h = span["bbox"][3] - span["bbox"][1]
                    gap = 0.5 * max(h, 1.0)
                    for ch in span.get("chars", []):
                        c = ch.get("c", "")
                        cx0, cy0, cx1, cy1 = ch["bbox"]
                        if not (
                            bx0 - 2 <= (cx0 + cx1) / 2 <= bx1 + 2
                            and by0 - 2 <= (cy0 + cy1) / 2 <= by1 + 2
                        ):
                            continue
                        # Split at whitespace chars (cell separators on
                        # letter-spaced pages where gap-clustering fails)
                        # or at char gaps > 0.5×height (merged spans).
                        if c.isspace():
                            if cur:
                                toks.append(cur)
                                cur = None
                            continue
                        if cur is None:
                            cur = [c, cx0, cx1]
                        elif cx0 - cur[2] > gap:
                            toks.append(cur)
                            cur = [c, cx0, cx1]
                        else:
                            cur[0] += c
                            cur[2] = cx1
                if cur:
                    toks.append(cur)
                toks = [t for t in toks if t[0].strip()]
                if toks:
                    lb = line["bbox"]
                    rows.append((lb[1], lb[3], toks))
        if len(rows) < 3:
            return None
        # Merge fitz lines into VISUAL rows: letter-spaced tables come as
        # per-column line fragments; y-band overlap joins them (the row the
        # grid algorithm needs).
        rows.sort(key=lambda r: r[0])
        merged = []
        for y0, y1, toks in rows:
            if merged and y0 <= merged[-1][1] - 0.3 * (y1 - y0):
                m = merged[-1]
                m[1] = max(m[1], y1)
                m[2].extend(toks)
            else:
                merged.append([y0, y1, list(toks)])
        vrows = [sorted(t[2], key=lambda t: t[1]) for t in merged]
        vrows = [r for r in vrows if len(r) >= 2]
        if len(vrows) < 3:
            return None
        from collections import Counter

        k = Counter(len(r) for r in vrows).most_common(1)[0][0]
        if k < 2:
            return None
        anchors = [r for r in vrows if len(r) == k]
        if len(anchors) < 2:
            return None

        # median x0 per column over anchors
        def _med(vals):
            s = sorted(vals)
            return s[len(s) // 2]

        col_x = [_med(a[j][1] for a in anchors) for j in range(k)]
        cuts = [(col_x[j] + col_x[j + 1]) / 2 for j in range(k - 1)]
        grid = []
        for r in vrows:
            cells = [""] * k
            for text, x0, _x1 in r:
                c = sum(1 for cx in cuts if x0 >= cx)
                cells[c] = f"{cells[c]} {text}".strip() if cells[c] else text
            grid.append(cells)
        grid = _merge_label_columns(grid)
        return grid

    def _label_value_grid(self, bbox):
        """Deterministic label|value grid for borderless two-column tables.

        Fallback for when the CNN shreds: course catalogs, spec sheets, fee
        schedules — every row is a long label with a distant right-aligned
        value. fitz often emits the label and the value as SEPARATE line
        dicts sharing a baseline, so first group all fragments by baseline
        (2pt), then split each visual line at its LARGEST x-gap when wide
        (>= 12pt — real gutters; word gaps run 2-5pt). Continuation lines
        (no wide gap) merge into the previous row's label. Returns a 2-col
        grid or None (< 3 rows).
        """
        try:
            rd = self._page_rawdict()
        except Exception:
            return None
        # bbox arrives in render PIXELS; rawdict geometry is PDF POINTS.
        pt_scale = 72.0 / self.render_dpi
        x0b, y0b, x1b, y1b = (v * pt_scale for v in bbox)
        frags = []
        for block in rd.get("blocks", []):
            if block.get("type", 0) != 0:
                continue
            for line in block.get("lines", []):
                txt = "".join(
                    c["c"] for s in line.get("spans", []) for c in s.get("chars", [])
                ).strip()
                if not txt:
                    continue
                lb = line["bbox"]
                ly = (lb[1] + lb[3]) / 2
                if lb[0] < x0b - 6 or lb[2] > x1b + 6 or not (y0b - 4 <= ly <= y1b + 4):
                    continue
                frags.append((ly, lb[0], lb[2], txt))
        frags.sort(key=lambda f: (f[0], f[1]))
        # Group fragments into visual lines by baseline.
        visual = []  # [ [ (x0, x1, text), ... ] ]
        for ly, fx0, fx1, txt in frags:
            if visual and abs(ly - visual[-1][0]) <= 2:
                visual[-1][1].append((fx0, fx1, txt))
            else:
                visual.append((ly, [(fx0, fx1, txt)]))
        rows = []
        for _ly, parts in visual:
            parts.sort()
            # Split at the largest inter-part gap when it is a real gutter.
            gaps = [(parts[i + 1][0] - parts[i][1], i) for i in range(len(parts) - 1)]
            if gaps:
                gap, idx = max(gaps)
                if gap >= 12:
                    # 2-COLUMN GUARD: a real label|value row has a SINGLE
                    # value cluster — no internal gap >= 12pt on either
                    # side. Year-columns/multi-col tables (904a1b4e: '62,8
                    # 60,2 62.2' at 20-30pt pitch) must NOT collapse into
                    # one glued value cell — bail out entirely.
                    value_gaps = [
                        parts[i + 1][0] - parts[i][1] for i in range(idx + 1, len(parts) - 1)
                    ]
                    label_gaps = [parts[i + 1][0] - parts[i][1] for i in range(0, idx)]
                    if any(g2 >= 12 for g2 in value_gaps) or any(g2 >= 12 for g2 in label_gaps):
                        return None
                    label = " ".join(p[2] for p in parts[: idx + 1]).strip()
                    value = " ".join(p[2] for p in parts[idx + 1 :]).strip()
                    if label and value:
                        rows.append([label, value])
                        continue
            # Continuation: fold into the previous row's label.
            whole = " ".join(p[2] for p in parts).strip()
            if whole and rows:
                rows[-1][0] = (rows[-1][0] + " " + whole).strip()
        if len(rows) < 3:
            return None
        # EXP-4 carve guard (2026-09-16): dense two-column PROSE reaches
        # this fallback when model escalation is unavailable — the column
        # gutter then pairs left/right SENTENCES as label|value rows
        # (04af23cb: 37 rows, median value 50 chars). Real label|value
        # targets carry SHORT value tokens (137297b4 course catalog:
        # median 2, max 12). Prose-class values mean the region is not a
        # table; emitting one eats the body text.
        _vl = sorted(len(r[1]) for r in rows)
        _med_val = _vl[len(_vl) // 2]
        if _med_val > 30:
            return None
        return rows

    def _is_layer_shredded(self, cells):
        """True when adjacent cells were cut INSIDE a word of the text layer.

        the boundary sliced a layer word in half. Legitimate cell boundaries
        never produce a layer word when concatenated ('ENGR'+'351' →
        'ENGR351' is not in the layer; 'CHEM 2'+'10' → 'CHEM 210' is).
        """
        if self._layer_text_norm is None:
            try:
                import fitz as _fz

                with _fz.open(self.pdf_path) as _doc:
                    raw = _doc[self.page_num - 1].get_text()
                self._layer_text_norm = " ".join(raw.split())
            except Exception:
                self._layer_text_norm = ""
        if not self._layer_text_norm:
            return False
        layer = self._layer_text_norm
        for row in cells:
            vals = [str(c).strip() for c in row if c and str(c).strip()]
            for a, b in zip(vals, vals[1:]):
                # Multi-char joins only (iter-58): single-letter joins
                # ('P'+'R'→'PR' in P/R/F1 header rows) are coincidence,
                # not shred evidence. Real cuts join >=3 chars (137297b4
                # 'Requirements fo'+'r Major in Me').
                if a[-1].isalnum() and b[0].isalnum() and len(a + b) >= 3:
                    if a + b in layer:
                        return True
        return False

    def process(self, candidate):
        """Extract (native-first → RapidTable fallback) → score → emit."""
        bbox = candidate["bbox"]
        native = candidate.get("native")
        # Native content may already be attached (locate matched it); else look up.
        if native is None:
            native = self._match_native(bbox)

        # Attempt 1: PyMuPDF native (deterministic, exact text layer). Reject
        # shattered grids (wrong column inference on borderless tables → cells
        # split into single chars) — escalate to the model instead.
        native_conf = -1.0
        native_result = None
        if native and not _is_shattered(native["cells"]):
            native_result = self._finalize(native["cells"], bbox, "pymupdf")
            native_conf = native_result.confidence
            # MONSTER-CELL guard: rows merged into one giant cell = wrong
            # native grid (see _score). Never early-accept a monster grid —
            # let the line-grid reconstruction compete even at high conf
            # (measured 904a1b4e pg141: monster grid accepted at 0.887).
            # NB: score_breakdown carries {"score", "breakdown"} (outer).
            if (
                native_conf >= self.ACCEPT_NATIVE
                and not native_result.score_breakdown.get("breakdown", {}).get("monster_cell")
                and not self._is_layer_shredded(native["cells"])
            ):
                # LAYER-SHRED GATE applies to native grids too (2026-08-19):
                # pymupdf's own column inference cut cells inside words on
                # borderless catalogs (137297b4: 'B.S. Mechanic|al Enginee'
                # accepted at conf 0.8).
                return native_result

        # Attempt 1.5: line-grid reconstruction (Marker table_recon port,
        # Apache-2.0). When the native grid is missing/shattered/weak,
        # rebuild the grid from raw char geometry: tokenize line spans at
        # char gaps > 0.5x height, then modal-K median-x0 column cuts.
        # Pure geometry, deterministic — beats wrong native grids and
        # RapidTable OCR on born-digital text tables.
        grid_pack = self._grid_candidate(bbox)
        if grid_pack is not None:
            grid, k, modal_ratio = grid_pack
            if grid is not None and not _is_shattered(grid):
                # Strong modal signal = char-geometry grid is trustworthy;
                # let it compete even against an accepted native grid
                # (native fitz grids misalign columns on letter-spaced and
                # borderless tables — measured conf 0.63 wrong vs 0.94 right).
                if modal_ratio >= 0.8 or native_conf < self.ACCEPT_NATIVE:
                    result_grid = self._finalize(grid, bbox, "line-grid")
                    if (
                        result_grid.confidence > native_conf
                        and result_grid.confidence >= self.ACCEPT_NATIVE
                        and not self._is_layer_shredded(grid)
                    ):
                        return result_grid

        # Attempt 2: RapidTable CNN (model fallback — scanned/borderless/hard).
        # EXP-1.6 evidence gate: pseudo-table regions dispatched onto prose
        # pay ~9s of CNN for output the accept gates then reject (bench3
        # census: 78/98 escalations rejected, e.g. 0005784d 2-col body).
        # Escalate only with region-local evidence: thin text (scan crop /
        # sparse borderless), pipe/caption/aligned-gap text, or a ruled grid
        # inside the bbox. NB a native hit is deliberately NOT evidence —
        # any native reaching Attempt 2 was already rejected as shattered
        # or monster (fitz fires weak full-page natives on shattered 2-col
        # prose), which is anti-evidence. No evidence -> Attempt 3.
        # RASTER-RETYPE BYPASS (EXP-4b, bench10 c8c85869/9a61fe78): regions
        # retyped from figure by the router probe carry OCR-confirmed grid
        # evidence (x-recurrence on the rendered crop) — the text-layer
        # probes here would WEAKLY re-veto that stronger decision (chart
        # axis-label text looks non-table-ish). RapidTable's rubric still
        # gates emission, so a bypass only costs a model call on a probe
        # the router already paid for.
        result_rt = None
        # bbox space contract: candidate["bbox"] is RENDER-px (DocLayout);
        # evidence probes (fitz drawings/images) need PDF-pt. Dispatcher
        # supplies "bbox_pdf_pt"; fall back to bbox for direct callers
        # (tests) whose bboxes are already PDF-pt (9a61fe78 lesson).
        _ev_bbox = candidate.get("bbox_pdf_pt") or bbox
        if candidate.get("retyped_from_figure") or region_has_table_evidence(
            self.pdf_path, self.page_num, _ev_bbox, self._region_text(_ev_bbox)
        ):
            record_path_warning(
                TABLE_MODEL_FALLBACK,
                "native table extraction absent/weak — escalating to RapidTable CNN",
            )
            result_rt = self._try_rapidtable(bbox)
        if result_rt is not None and result_rt.confidence >= self.ACCEPT_MODEL:
            # LAYER-SHRED GATE (2026-08-19): RapidTable's CNN sometimes cuts
            # cells INSIDE words on borderless text tables ('CHEM 2|10',
            # 'Required ME C|ourses' — 137297b4 pg4). A mid-token cut whose
            # rejoin exists in the text layer is proof of shredding; a
            # legit cell boundary never rejoins into a layer word
            # ('ENGR'+'351' → 'ENGR351' absent from the layer). Reject →
            # the region falls back to text, keeping the clean positional
            # table the postproc pipeline already built.
            if self._is_layer_shredded(result_rt.cells):
                record_path_warning(
                    TABLE_MODEL_FALLBACK,
                    "RapidTable grid shredded mid-token (layer rejoin "
                    "evidence) — rejected, region kept as text",
                )
                lv_grid = self._label_value_grid(bbox)
                if lv_grid is not None:
                    return self._finalize(lv_grid, bbox, "label-value")
            else:
                return result_rt
        # Attempt 3: keep the better of native / model even if both weak.
        # NEVER return shattered native — it's false-positive table detection
        # (figure/text page misdetected as table → prose fragmented into grid).
        if (
            native
            and not _is_shattered(native["cells"])
            and not self._is_layer_shredded(native["cells"])
            and (result_rt is None or result_rt.confidence <= self.ACCEPT_MODEL)
        ):
            return self._finalize(native["cells"], bbox, "pymupdf-low")

        if result_rt is not None:
            # Shred gate covers the weak-confidence return too — a shredded
            # grid is worse than text no matter its confidence.
            if self._is_layer_shredded(result_rt.cells):
                lv_grid = self._label_value_grid(bbox)
                if lv_grid is not None:
                    return self._finalize(lv_grid, bbox, "label-value")
                return None
            return result_rt
        # Last resort: shred-free deterministic label|value grid (covers the
        # no-provider serial path where native shredded and RapidTable is
        # unavailable).
        lv_grid = self._label_value_grid(bbox)
        if lv_grid is not None:
            return self._finalize(lv_grid, bbox, "label-value")
        return None

    def _match_native(self, bbox):
        """Closest PyMuPDF/pdfplumber table overlapping ``bbox`` (IoU ≥ 0.3)."""
        best, best_iou = None, 0.0
        for nt in self._native_tables():
            iou = _iou(nt["bbox"], bbox)
            if iou > best_iou:
                best, best_iou = nt, iou
        return best if best_iou >= 0.30 else None

    def _region_text(self, bbox):
        """Text-layer content clipped to the region bbox (PDF points).

        Feeds the model-escalation evidence gate. Empty/short here means
        scan crop or sparse table — the gate treats that as evidence FOR
        escalation, so a probe failure returning '' is fail-safe.
        """
        try:
            import fitz as _fz

            with _fz.open(self.pdf_path) as _doc:
                page = _doc[self.page_num - 1]
                return page.get_text(clip=_fz.Rect(*bbox))
        except Exception:
            return ""

    def _try_rapidtable(self, bbox):
        """RapidTable extraction on the cropped region (model fallback)."""
        if not (self.rapidtable and self.rapidtable.available() and self.img_path):
            return None
        try:
            crop = _crop_image(self.img_path, bbox)
            ocr_result = None
            if self.page_ocr:
                ocr_result = _ocr_boxes_for_crop(
                    self.page_ocr[0], self.page_ocr[1], self.page_ocr[2], bbox
                )
            html, conf = self.rapidtable.extract_table(crop, ocr_result=ocr_result)
            if not (html and conf > 0.3):
                return None
            cells = _cells_from_html(html)
            if not cells:
                return None
            return self._finalize(cells, bbox, "rapidtable", model_conf=conf)
        except Exception:
            return None

    def _split_fused_header(self, cells, bbox):
        """iter-119a: split fused HEADER cells at true column boundaries.

        A grid whose header row carries multi-column text in one cell
        ('Q1 Q2 Q3' fused by native/line-grid column inference) hides
        gold cells from cell-match scoring. The header LINE's own word
        boxes are the column oracle: header words are typeset one per
        column, so a header cell whose tokens exist as separate page
        words with inter-word gaps >= 1.5x word-space width splits into
        one cell per token; phrases with normal spacing stay whole.
        Body rows are padded with '' to the new width (validated on
        1a67d1114 pg1: fitz header = 7 cols, our emit fused to 4).
        Kill-switch PDF_OCR_HEADER_SPLIT=0.
        """
        import os as _os

        if _os.environ.get("PDF_OCR_HEADER_SPLIT", "1") != "1":
            return cells
        if not cells or len(cells) < 2:
            return cells
        try:
            words = self._header_words(bbox)
            if not words:
                return cells
            new_row = []
            changed = False
            for c in cells[0]:
                parts = _split_cell_on_word_gaps(c, words)
                if len(parts) > 1:
                    changed = True
                new_row.extend(parts)
            if not changed:
                return cells
            width = len(new_row)
            return [new_row] + [
                (r + [""] * width)[:width] if len(r) < width else r for r in cells[1:]
            ]
        except Exception as e:
            log.warning("header split skipped: %s", e)
            return cells

    def _header_words(self, bbox):
        """Page words on the table's header line (top 2.5 line-heights)."""
        try:
            pd = self._page_rawdict()
            if not pd:
                return []
            x0, y0, x1, _y1 = bbox
            out = []
            for block in pd.get("dict", {}).get("blocks", []):
                if block.get("type", 0) != 0:
                    continue
                for line in block.get("lines", []):
                    for sp in line.get("spans", []):
                        sx0, sy0, sx1, sy1 = sp["bbox"]
                        if sy0 >= y0 - 2 and sy1 <= y0 + 28 and sx0 >= x0 - 6 and sx1 <= x1 + 6:
                            out.append((sx0, sx1, sp["text"].strip()))
            return [w for w in out if w[2]]
        except Exception:
            return []

    def _finalize(self, cells, bbox, method, model_conf=None):
        """Stage 3 (validate) + Stage 4 (emit)."""
        cells = self._split_fused_header(cells, bbox)
        score_bd = _score(cells, bbox, self.page_area_pt, self.render_dpi, self.WEIGHTS)
        confidence = score_bd["score"]
        if model_conf is not None:
            # Blend structural score with model self-confidence — the model has
            # seen the image; the rubric scores the structure. Average them.
            confidence = 0.5 * confidence + 0.5 * model_conf
        nr = len(cells)
        nc = max((len(r) for r in cells), default=0)
        return TableResult(
            bbox=bbox,
            rows=nr,
            columns=nc,
            cells=cells,
            markdown=_cells_to_markdown(cells),
            confidence=round(confidence, 3),
            method=method,
            score_breakdown=score_bd,
            needs_review=confidence < self.REVIEW,
        )


def _split_cell_on_word_gaps(cell, words):
    """Split a header cell when its tokens are separate words with
    columnar gaps (>= 1.5x space width). Phrases stay whole."""
    toks = cell.split()
    if len(toks) < 2:
        return [cell]
    by_text = {}
    for wx0, wx1, wt in words:
        by_text.setdefault(wt, (wx0, wx1))
    exts = []
    for t in toks:
        if t not in by_text:
            return [cell]
        exts.append(by_text[t])
    gaps = [exts[i + 1][0] - exts[i][1] for i in range(len(exts) - 1)]
    space_w = max(2.0, 0.25 * (exts[0][1] - exts[0][0]))
    if any(g >= 1.5 * space_w for g in gaps):
        return toks
    return [cell]


# ── Stage 3: VALIDATE — the 7-metric rubric (module function, pure) ──────── #


def _score(cells, bbox, page_area_pt, render_dpi, weights):
    """7-metric validation rubric. Pure function → {score, breakdown}.

    Metrics: geometry (bbox sane), text_coverage (non-empty cells), column
    consistency (stable column count), row consistency (enough rows), header
    quality (first row dense), numeric consistency (coherent column types),
    caption proximity (handled by caller via page context — default neutral).
    """
    nr = len(cells)
    row_lens = [len(r) for r in cells]
    nc = max(row_lens) if row_lens else 0
    total = nr * nc if nr and nc else 1
    nonempty = sum(1 for r in cells for c in r if c and str(c).strip())

    text_coverage = nonempty / total

    # MONSTER-CELL PENALTY (2026-08-18): borderless tables where the row
    # detector fails merge an entire label COLUMN into one giant cell
    # (measured 904a1b4e pg141: 15 label lines in one cell, values in a
    # single following row). The rubric scored that wrong grid 0.887 —
    # above ACCEPT_NATIVE — so the line-grid reconstruction never got to
    # compete. A cell >8x the median non-empty cell length means rows
    # were merged: scale text_coverage down (floor 0.2 of original).
    _lens = [len(str(c).strip()) for r in cells for c in r if c and str(c).strip()]
    monster_cell = False
    if len(_lens) >= 4:
        _med = sorted(_lens)[len(_lens) // 2]
        _monster = max(_lens) / max(_med, 1)
        # >15x AND >120 chars: legit long labels run ~8x (NUTS region
        # names, bef02dc2); true row-merge monsters run 40-80x.
        if _monster > 15 and max(_lens) > 120:
            text_coverage *= max(0.2, 1.0 - (_monster - 8) * 0.02)
            monster_cell = True

    bw, bh = bbox[2] - bbox[0], bbox[3] - bbox[1]
    # geometry: sane bbox (not tiny); render-px → pt for the page-area check
    scale = render_dpi / 72.0
    area_pt = bw * bh / (scale * scale)
    too_big = page_area_pt and area_pt > 0.70 * page_area_pt
    too_small = bw < 50 or bh < 50
    shattered = _is_shattered(cells)
    if shattered:
        geometry = 0.1  # wrong grid → near-zero, force needs_review / escalate
    elif too_big:
        geometry = 0.2
    elif too_small:
        geometry = 0.5
    else:
        geometry = 1.0

    distinct_cols = len(set(row_lens))
    col_consistency = 1.0 if distinct_cols <= 1 else min(row_lens) / max(max(row_lens), 1)

    row_consistency = min(nr / 3.0, 1.0)

    header = cells[0] if cells else []
    header_quality = (
        sum(1 for c in header if c and str(c).strip()) / max(len(header), 1) if header else 0.0
    )

    data_cells = [str(c).strip() for r in cells[1:] for c in r if c and str(c).strip()]
    if data_cells:
        n_num = sum(1 for c in data_cells if _is_numeric(c))
        # a coherent column is mostly-numeric OR mostly-text (not 50/50 noise)
        frac_num = n_num / len(data_cells)
        numeric_consistency = max(frac_num, 1.0 - frac_num)
    else:
        numeric_consistency = 0.5

    caption_proximity = 0.5  # neutral default (caller can refine)

    breakdown = {
        "geometry": geometry,
        "text_coverage": text_coverage,
        "column_consistency": col_consistency,
        "row_consistency": row_consistency,
        "header_quality": header_quality,
        "numeric_consistency": numeric_consistency,
        "caption_proximity": caption_proximity,
        "monster_cell": monster_cell,
    }
    score = sum(weights[k] * breakdown[k] for k in weights)
    return {"score": round(score, 3), "breakdown": breakdown}


# ── Helpers (self-contained — no import from region_dispatcher) ─────────── #


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    union = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by1) - inter
    return inter / (union + 1e-9)


def _is_numeric(s):
    s = str(s).strip().rstrip(".,;:%")
    if not s:
        return False
    try:
        float(s)
        return True
    except ValueError:
        pass
    # EU comma-decimal ('62,8' — Portuguese/German/French stats tables):
    # '1.234,5' → drop dots, comma→dot; '62,8' → comma→dot. Bare text
    # ('1998a', '15-64') still fails both paths.
    try:
        t = s.replace(".", "").replace(",", ".") if ("." in s and "," in s) else s.replace(",", ".")
        float(t)
        return True
    except ValueError:
        return False


def _merge_label_columns(grid):
    """Merge left text-fragment columns into one label column.

    Justified/letter-spaced label cells split at word gaps into several
    "columns" (measured 904a1b4e pg141: 'Taxa'|'de emprego'|''|'(15-64
    anos) total' — the olmocr left-heading lookup needs the label as ONE
    cell). Statistical tables' skeleton is the NUMERIC matrix: everything
    left of the first numeric-dominant column is the label span. Only
    merges when no left-fragment cell is itself numeric (numeric row-id
    columns make first_num_col 0 → no-op by construction).
    """
    if not grid:
        return grid
    nc = max(len(r) for r in grid)
    first_num = None
    for j in range(nc):
        vals = [str(r[j]).strip() for r in grid if j < len(r) and str(r[j]).strip()]
        if len(vals) < 3:
            continue
        if sum(1 for v in vals if _is_numeric(v)) / len(vals) >= 0.5:
            first_num = j
            break
    if first_num is None or first_num < 2:
        return grid
    for r in grid:
        for c in r[:first_num]:
            if str(c).strip() and _is_numeric(str(c)):
                return grid
    return [
        [" ".join(x for x in (str(c).strip() for c in r[:first_num]) if x)] + list(r[first_num:])
        for r in grid
    ]


def _is_shattered(cells):
    """Detect a false-positive or wrong-grid table extraction.

    Two failure modes:
    1. **Micro-fragmentation** (original): words split into single chars
       (median cell length ≤ 3). PyMuPDF/pdfplumber wrong grid inference.
    2. **Prose fragmentation** (2026-08-14 bench debug): figure/text pages
       misdetected as tables. Cells are word fragments of flowing prose —
       median length 4-8 but cells concatenate to form sentences.

    Detection uses a composite score: median_len, empty-cell ratio, and
    word-boundary violations (cells starting/ending mid-word).
    """
    if _is_misaligned_header(cells):
        return True
    flat = [str(c).strip() for row in cells for c in row if c and str(c).strip()]
    if len(flat) < 10:
        return False

    # Signal 1: micro-fragmentation (original check).
    flat_sorted = sorted(flat, key=len)
    median_len = len(flat_sorted[len(flat_sorted) // 2])
    if median_len <= 3:
        return True

    # Signal 2: prose fragmentation — cells are word fragments.
    # Total cells including empties (for empty-ratio).
    total_cells = sum(len(row) for row in cells)
    empty_count = total_cells - len(flat)
    empty_ratio = empty_count / total_cells if total_cells else 0

    # Word-boundary violations: cells that start with lowercase or end with
    # a hyphen indicate mid-word breaks in flowing prose.
    boundary_violations = sum(
        1 for c in flat if (c and c[0].islower()) or c.endswith("-") or c.endswith("­")
    )
    violation_ratio = boundary_violations / len(flat) if flat else 0

    # Prose-fragment composite: high empty ratio + short median + boundary
    # violations = flowing prose shattered into a grid.
    if empty_ratio > 0.45 and median_len <= 8 and violation_ratio > 0.25:
        return True
    # Very high boundary violations alone = prose in a grid.
    if violation_ratio > 0.50 and median_len <= 8:
        return True
    return False


# A cell that is 3+ short tokens joined by " / " — the signature of a
# COLUMN-WISE MERGED grid: PyMuPDF merged the first N visual rows into one
# cell per column ("No / 1 / 2 / 3" = column 0's values for rows 1-3).
_MERGED_CELL_RE = re.compile(r"^.{1,20}(?: / .{1,20}){2,}$")


def _is_misaligned_header(cells):
    """Detect row-merged grids: first-row cells contain slash-joined tokens.

    On borderless tables PyMuPDF groups the first 2-4 visual rows into ONE
    block per column, producing header cells like ``No / 1 / 2 / 3`` while
    later rows extract per-row. The grid is structurally WRONG (headings
    unanchored) even though text coverage looks fine — same treatment as
    shattered: reject native, escalate to RapidTable.
    """
    if len(cells) < 3:
        return False
    # Count slash-merged cells in the first two rows.
    merged = sum(1 for row in cells[:2] for c in row if _MERGED_CELL_RE.match(str(c).strip()))
    return merged >= 2


def _crop_image(img_path, bbox):
    """Crop ``img_path`` to ``bbox`` (x0,y0,x1,y1 render px) → temp PNG path."""
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


def _ocr_boxes_for_crop(boxes, txts, scores, bbox):
    """Filter page-level OCR boxes to a crop + translate to crop coords.

    Mirrors the helper in region_dispatcher (kept self-contained here to avoid
    a circular import). Returns (dt_boxes, (texts), (scores)) or None.
    """
    try:
        import numpy as np
    except ImportError:
        return None
    x0, y0, x1, y1 = bbox
    out_boxes, out_txts, out_scores = [], [], []
    for box, txt, sc in zip(boxes, txts, scores):
        b = np.asarray(box, dtype=np.float32)
        if b.size < 2:
            continue
        bx_min, by_min = b[:, 0].min(), b[:, 1].min()
        bx_max, by_max = b[:, 0].max(), b[:, 1].max()
        if bx_max < x0 or bx_min > x1 or by_max < y0 or by_min > y1:
            continue
        out_boxes.append(b - np.array([x0, y0], dtype=np.float32))
        out_txts.append(txt)
        out_scores.append(float(sc))
    if not out_boxes:
        return None
    return np.asarray(out_boxes, dtype=np.float32), tuple(out_txts), tuple(out_scores)


def _cells_from_html(html):
    """Parse ``<table><tr><td>…`` HTML → 2D list of cell strings."""
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html, re.DOTALL)
    cells = []
    for row in rows:
        cell_list = re.findall(r"<t[dh][^>]*>(.*?)</t[dh]>", row, re.DOTALL)
        cells.append([re.sub(r"<[^>]+>", "", c).strip() for c in cell_list])
    return [r for r in cells if r]


def _cells_to_markdown(cells):
    """Render 2D cells as a sepless markdown table (det44 emission)."""
    if not cells:
        return ""
    lines = []
    from scientific_parser.perception.tables.detector import (
        _defragment_row,
        normalize_grid_rows,
    )

    # Shared normalization (iter-113): label-wrap split + column
    # compaction — the agent's char-geometry grid hits the same
    # 'anos 31,0' glue and phantom-column shapes as the strategies.
    cells = normalize_grid_rows(
        [[getattr(c, "text", str(c)) if c is not None else "" for c in row] for row in cells]
    )
    for i, row in enumerate(cells):
        vals = [
            str(c).replace("|", "\\|").replace("\n", " ").strip() if c else ""
            for c in _defragment_row(row)
        ]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"  # trailing blank line (det44: sepless emission)


__all__ = ["TableAgent", "TableResult"]
