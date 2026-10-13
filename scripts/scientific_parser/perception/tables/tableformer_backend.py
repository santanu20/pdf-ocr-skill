"""TableFormer fallback backend (det61).

Fires ONLY on pages where the 7-strategy table chain emitted ZERO
tables (odlbench 01030000000149/0165/0197 class: borderless or sparse
tables with no vector-line evidence). Region proposals come from the
DocLayout-YOLO layout detector, whose bboxes are already in SF=2.0
render pixels — the SAME space as the tokens/image built here, so the
detector bbox is passed through VERBATIM (scaling it again was the
integration bug: crops land 2x off and the model regresses garbage).
TableFormer (docling-ibm-models, CPU torch) regresses the cell grid;
cell text comes from the matcher's token->cell map inverted per cell.

Kill-switch: PDF_OCR_TABLEFORMER=0 disables the fallback (returns []).
All failures fail OPEN with a warning — the page keeps its pre-det61
no-table emission. Import-heavy deps load lazily so environments
without the `tableformer` extra stay functional.
"""

import logging
import os
import time

log = logging.getLogger(__name__)

_SF = 2.0  # render scale; must equal the layout detector zoom
_MAX_REGIONS = 4  # bound worst-case latency (43s beam outlier measured)

_PREDICTOR = None
_PREDICTOR_TRIED = False


def _enabled():
    return os.environ.get("PDF_OCR_TABLEFORMER", "1") != "0"


def _get_predictor():
    """Lazy singleton; None (one warning) when the extra is absent."""
    global _PREDICTOR, _PREDICTOR_TRIED
    if _PREDICTOR is not None:
        return _PREDICTOR
    if _PREDICTOR_TRIED:
        return None
    _PREDICTOR_TRIED = True
    try:
        import docling_ibm_models.tableformer.common as c
        from docling_ibm_models.tableformer.data_management.tf_predictor import (
            TFPredictor,
        )

        root = os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            )
        )  # .../pdf-ocr
        mdl = os.path.join(root, ".models", "tableformer")
        cfg = c.read_config(os.path.join(mdl, "tm_config.json"))
        cfg["model"]["save_dir"] = mdl
        cfg["model"]["load_checkpoint"] = os.path.join(mdl, "tableformer_accurate.safetensors")
        _PREDICTOR = TFPredictor(cfg, device="cpu", num_threads=4)
    except Exception as e:  # fail-open: page keeps no-table emission
        log.warning("tableformer backend unavailable: %s: %s", type(e).__name__, e)
        return None
    return _PREDICTOR


def _grid_from_response(
    response, tokens_by_id, region=None, n_rows=None, n_cols=None, min_iopdf=0.25
):
    """Pure assembly: response cells + matches -> (rows, cols, grid).

    Grid dims come from predict_details num_rows/num_cols when given
    (authoritative — tf_responses can carry offset cells beyond the
    model's stated grid). Tokens contribute only when (a) their bbox
    center lies inside the region (matcher maps EVERY page token to its
    best cell regardless of overlap — without this filter the caption
    prose above the table leaks into the grid) and (b) iopdf >=
    min_iopdf. Spanned cells place text at the span's top-left; covered
    cells stay empty. Returns None for degenerate/blank grids.
    """
    cells = response.get("tf_responses") or []
    det = response.get("predict_details") or {}
    matches = det.get("matches") or {}
    if n_rows is None:
        n_rows = det.get("num_rows") or 0
    if n_cols is None:
        n_cols = det.get("num_cols") or 0
    rx0, ry0, rx1, ry1 = region if region else (0.0, 0.0, 1e18, 1e18)
    cell_tokens = {}
    for tok_id, hits in matches.items():
        # matcher keys are STRINGS (json round-trip inside TFPredictor);
        # tokens_by_id is keyed by int — normalize before lookup
        try:
            tok_key = int(tok_id)
        except (TypeError, ValueError):
            continue
        tok = tokens_by_id.get(tok_key)
        if tok is None:
            continue
        x0, y0, x1, y1 = tok[3]
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if not (rx0 - 2 <= cx <= rx1 + 2 and ry0 - 2 <= cy <= ry1 + 2):
            continue  # token lives outside the table region
        for hit in hits or []:
            cid = hit.get("table_cell_id")
            if cid is not None and (hit.get("iopdf") or 0.0) >= min_iopdf:
                cell_tokens.setdefault(cid, []).append(tok)
    model_rows, model_cols = n_rows, n_cols  # frozen for clamping
    placed = {}
    for i, cell in enumerate(cells):
        r0 = cell.get("start_row_offset_idx")
        c0 = cell.get("start_col_offset_idx")
        r1 = cell.get("end_row_offset_idx")
        c1 = cell.get("end_col_offset_idx")
        if r0 is None or c0 is None or r1 is None or c1 is None:
            continue
        # end offsets are EXCLUSIVE (docling OTSL convention): end == num_rows
        # spans the full grid — only drop when strictly beyond it
        if model_rows and (r1 > model_rows or r0 < 0):
            continue  # beyond the model's stated grid
        if model_cols and (c1 > model_cols or c0 < 0):
            continue
        toks = cell_tokens.get(i, [])
        toks.sort(key=lambda t: (t[1], t[0]))  # (y, x) of bbox top-left
        text = " ".join(t[2] for t in toks).strip()
        placed[(r0, c0)] = text
        n_rows = max(n_rows, r1)
        n_cols = max(n_cols, c1)
    if n_rows < 2 or n_cols < 1 or not placed:
        return None
    grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
    for (r, c), text in placed.items():
        if 0 <= r < n_rows and 0 <= c < n_cols:
            grid[r][c] = text
    if not any(any(row) for row in grid):
        return None
    return n_rows, n_cols, grid


def _trim_grid(grid):
    """Drop trailing all-empty rows/columns (matcher cells can define a
    wider grid than the matched text occupies). Leading empties stay —
    they carry the model's structural row/col offsets."""
    g = [list(r) for r in grid]
    while g and not any(g[-1]):
        g.pop()
    if g:
        w = len(g[0])
        while w and not any(r[w - 1] for r in g):
            w -= 1
        g = [r[:w] for r in g]
    return g


def _grid_html(grid):
    """Raw <table> twin for a TF grid. Same output shape as
    detector.build_table_html (thead/tbody + th/td) but WITHOUT
    _scrub_grid_rows — that scrubber is tuned for strategy-chain grids
    and deletes TF grids whose sparse columns are structural, not
    artifacts (measured: 0165 6x2 -> []). det45: without an html twin
    the table is TEDS-invisible on odlbench."""
    from html import escape as _esc

    if not grid or len(grid) < 2:
        return ""
    n_cols = max(len(r) for r in grid)

    def _cell(c, tag):
        return f"<{tag}>{_esc(str(c if c is not None else '').strip())}</{tag}>"

    out = [
        "<table>",
        "<thead>",
        "<tr>"
        + "".join(_cell(c, "th") for c in (list(grid[0]) + [""] * n_cols)[:n_cols])
        + "</tr>",
        "</thead>",
        "<tbody>",
    ]
    for br in grid[1:]:
        out.append(
            "<tr>" + "".join(_cell(c, "td") for c in (list(br) + [""] * n_cols)[:n_cols]) + "</tr>"
        )
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def tables_for_page(pdf_path, page_num):
    """Zero-emission fallback: TableFormer grids for a page's table
    regions. Returns engine-shaped table dicts ([] on any failure)."""
    if not _enabled():
        return []
    tfp = _get_predictor()
    if tfp is None:
        return []
    import numpy as np
    import fitz

    try:
        from scientific_parser.perception.providers.layout_detector import (
            get_layout_detector,
        )

        regions = [
            r
            for r in get_layout_detector("default").detect_regions(pdf_path, page_num)
            if r["type"] == "table"
        ]
        if not regions:
            return []
        doc = fitz.open(pdf_path)
        try:
            page = doc[page_num - 1]
            W, H = page.rect.width, page.rect.height
            # Region space sanity: detector px must fit the SF render.
            regions = [
                r for r in regions if r["bbox"][2] <= W * _SF + 2 and r["bbox"][3] <= H * _SF + 2
            ]
            if not regions:
                log.warning(
                    "tableformer regions exceed render dims %s p%s (wrong space?) — skipped",
                    pdf_path,
                    page_num,
                )
                return []
            regions = sorted(
                regions,
                key=lambda r: -((r["bbox"][2] - r["bbox"][0]) * (r["bbox"][3] - r["bbox"][1])),
            )[:_MAX_REGIONS]
            regions.sort(key=lambda r: r["bbox"][1])  # reading order
            pix = page.get_pixmap(matrix=fitz.Matrix(_SF, _SF))
            img = (
                np.frombuffer(pix.samples, np.uint8)
                .reshape(pix.height, pix.width, pix.n)[:, :, :3]
                .copy()
            )
            words = [w for w in page.get_text("words") if w[4].strip()]
            toks = [
                {"id": i, "bbox": [w[0] * _SF, w[1] * _SF, w[2] * _SF, w[3] * _SF], "text": w[4]}
                for i, w in enumerate(words)
            ]
            # value shape: (x, y, text, bbox-in-SF-px) — bbox feeds the region filter
            tokens_by_id = {
                i: (w[0], w[1], w[4], (w[0] * _SF, w[1] * _SF, w[2] * _SF, w[3] * _SF))
                for i, w in enumerate(words)
            }
            page_input = {"tokens": toks, "width": W * _SF, "height": H * _SF, "image": img}
        finally:
            doc.close()
        t0 = time.time()
        out = tfp.multi_table_predict(
            page_input, [list(r["bbox"]) for r in regions], do_matching=True
        )
        log.info(
            "tableformer predict p%s: %.1fs, %d region(s)", page_num, time.time() - t0, len(regions)
        )
        tables = []
        for resp, region in zip(out, regions):
            asm = _grid_from_response(resp, tokens_by_id, region=region["bbox"])
            if asm is None:
                continue
            n_rows, n_cols, grid = asm
            grid = _trim_grid(grid)
            if len(grid) < 2:
                continue
            md = "\n".join("| " + " | ".join(str(c) for c in row) + " |" for row in grid)
            tables.append(
                {
                    "strategy": "tableformer",
                    "bbox": list(region["bbox"]),
                    "rows": grid,
                    "markdown": md,
                    "html": _grid_html(grid),
                    "tableformer": True,
                }
            )
        return tables
    except Exception as e:  # fail-open (H1: loud, never crash the page)
        log.warning(
            "tableformer fallback failed %s p%s: %s: %s", pdf_path, page_num, type(e).__name__, e
        )
        return []
